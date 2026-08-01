# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
ONNX inference runtime for models exported with the Schola ONNX conventions.

The graphs produced by ``schola.core.model.ScholaModel`` (and the SB3/RLlib
exporters built on top of it) follow a fixed input/output contract:

- One input tensor per observation space key (``"obs"`` for non-Dict spaces),
  shaped ``[batch, *space.shape]`` with a dynamic batch dimension.
- One output tensor per action space key (``"action"`` for non-Dict spaces).
- Box tensors use the space's declared dtype (``float32`` for Unreal
  environments), Discrete/MultiDiscrete tensors are ``int64``, and
  MultiBinary tensors are ``bool``.
- Discrete indices (argmax) are computed inside the graph, not at runtime.
- Stateful graphs (``state_in_*`` / ``state_out_*``) are not produced by the
  SB3 exporter and are rejected here.

This module is the inference-side mirror of the exporter: it consumes raw
environment observations verbatim (no preprocessing, transposing, or clipping)
and returns actions verbatim, exactly like the Unreal Engine NNE runtime.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)

_STATE_IN_PREFIX = "state_in"
_STATE_OUT_PREFIX = "state_out"


def _expected_np_dtype(space: gym.Space) -> Optional[np.dtype]:
    """
    The ONNX tensor dtype Schola exports for a given gymnasium space.

    Parameters
    ----------
    space : gymnasium.Space
        The space to map.

    Returns
    -------
    Optional[numpy.dtype]
        The expected ONNX tensor dtype, or ``None`` for unsupported spaces.
    """
    if isinstance(space, gym.spaces.Box):
        # The exporter feeds torch.as_tensor(space.sample()) to the graph, so
        # the graph dtype mirrors the space's declared dtype.
        return np.dtype(space.dtype)
    if isinstance(space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return np.dtype(np.int64)
    if isinstance(space, gym.spaces.MultiBinary):
        return np.dtype(np.bool_)
    return None


def _parse_shape(dims) -> Tuple[Optional[int], ...]:
    """
    Extract a shape tuple from ONNX ``TensorShapeProto`` dims.

    Parameters
    ----------
    dims : iterable of onnx.TensorShapeProto.Dimension
        The dims to parse.

    Returns
    -------
    Tuple[Optional[int], ...]
        Static dims as ints, dynamic dims as ``None``.
    """
    shape = []
    for dim in dims:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        else:
            shape.append(None)
    return tuple(shape)


@dataclass(frozen=True)
class OnnxTensorSpec:
    """
    Static description of an ONNX graph input or output tensor.

    Parameters
    ----------
    name : str
        The tensor name (an observation or action space key).
    dtype : numpy.dtype
        The tensor data type.
    shape : Tuple[Optional[int], ...]
        The tensor shape; ``None`` marks a dynamic dimension (e.g. the batch
        dimension of exported policies).
    """

    name: str
    dtype: np.dtype
    shape: Tuple[Optional[int], ...]

    @property
    def static_shape(self) -> Tuple[int, ...]:
        """
        Only the statically-known dims of this tensor.

        Returns
        -------
        Tuple[int, ...]
            The static dims.
        """
        return tuple(d for d in self.shape if d is not None)


class OnnxScholaModel:
    """
    ONNX inference runtime that speaks the Schola ONNX export contract.

    Loads an exported ``.onnx`` policy with onnxruntime, validates it against
    the environment's observation and action spaces, and computes batched
    actions from batched raw observations.

    Parameters
    ----------
    onnx_path : str or pathlib.Path
        Path to the exported ONNX policy.
    providers : Optional[Sequence[str]], optional
        onnxruntime execution providers; ``None`` uses the runtime defaults.

    Attributes
    ----------
    inputs : Dict[str, OnnxTensorSpec]
        Graph inputs keyed by tensor name (observation space keys).
    outputs : Dict[str, OnnxTensorSpec]
        Graph outputs keyed by tensor name (action space keys).
    """

    def __init__(
        self,
        onnx_path: Any,
        providers: Optional[Sequence[str]] = None,
    ):
        import onnx
        import onnxruntime as ort

        path = str(onnx_path)
        try:
            self._model_proto = onnx.load(path)
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model '{path}': {e}") from e

        try:
            self._session = ort.InferenceSession(path, providers=providers)
        except ImportError as e:  # pragma: no cover - trivial import error path
            raise ImportError(
                "onnxruntime is required to run ONNX inference. "
                "Install it with `pip install 'schola[sb3]'` or "
                "`pip install onnxruntime`."
            ) from e
        except Exception as e:
            raise ValueError(
                f"onnxruntime failed to load ONNX model '{path}'. "
                "The file may be corrupted, or it may require a different "
                "onnxruntime version. Consider re-exporting the policy with "
                "'schola sb3 export'."
            ) from e

        self.inputs: Dict[str, OnnxTensorSpec] = {}
        self.outputs: Dict[str, OnnxTensorSpec] = {}
        self._parse_graph()

        state_inputs = [
            name for name in self.inputs if name.startswith(_STATE_IN_PREFIX)
        ]
        state_outputs = [
            name for name in self.outputs if name.startswith(_STATE_OUT_PREFIX)
        ]
        if state_inputs or state_outputs:
            raise ValueError(
                f"ONNX model '{path}' contains state tensors "
                f"(inputs={state_inputs}, outputs={state_outputs}). "
                "Stateful models are not supported by the SB3 ONNX exporter; "
                "export a stateless policy and retry."
            )

        if not self.inputs:
            raise ValueError(f"ONNX model '{path}' has no graph inputs.")
        if not self.outputs:
            raise ValueError(f"ONNX model '{path}' has no graph outputs.")

        # Populated by validate(); predict() refuses to run without it.
        self._obs_space: Optional[gym.spaces.Dict] = None
        self._action_space: Optional[gym.spaces.Dict] = None
        self._obs_space_is_natively_dict: Optional[bool] = None
        self._action_space_is_natively_dict: Optional[bool] = None
        self._warned_casts: set = set()

    def _parse_graph(self) -> None:
        """
        Extract input/output specs from the loaded ONNX graph.

        Raises
        ------
        ValueError
            If a tensor uses an unsupported data type.
        """
        import onnx
        from onnx.helper import tensor_dtype_to_np_dtype

        graph = self._model_proto.graph
        initializer_names = {initializer.name for initializer in graph.initializer}

        for tensors, destination in (
            (graph.input, self.inputs),
            (graph.output, self.outputs),
        ):
            for tensor in tensors:
                if tensor.name in initializer_names:
                    continue
                elem_type = tensor.type.tensor_type.elem_type
                try:
                    dtype = np.dtype(tensor_dtype_to_np_dtype(elem_type))
                except Exception:
                    # onnx < 1.21 did not always expose the conversion helper.
                    import onnx.mapping

                    dtype = np.dtype(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[elem_type])
                destination[tensor.name] = OnnxTensorSpec(
                    name=tensor.name,
                    dtype=dtype,
                    shape=_parse_shape(tensor.type.tensor_type.shape.dim),
                )

    @staticmethod
    def _check_space_shape(
        spec: OnnxTensorSpec,
        space: gym.Space,
        kind: str,
    ) -> List[str]:
        """
        Compare a graph tensor shape against an observation/action space.

        Parameters
        ----------
        spec : OnnxTensorSpec
            The graph tensor spec (batch dim excluded by the caller).
        space : gymnasium.Space
            The space to compare against.
        kind : str
            "observation" or "action", used in error messages.

        Returns
        -------
        List[str]
            Shape mismatch descriptions (empty when compatible).
        """
        errors = []
        graph_shape = spec.shape[1:]
        space_shape = space.shape
        # Discrete scalars are exported with an explicit size-1 dim.
        if space_shape == () and graph_shape == (1,):
            graph_shape = ()
        if len(graph_shape) != len(space_shape):
            errors.append(
                f"{kind} '{spec.name}': graph shape {graph_shape} does not "
                f"match space shape {space_shape}"
            )
            return errors
        for graph_dim, space_dim in zip(graph_shape, space_shape):
            if graph_dim is not None and graph_dim != space_dim:
                errors.append(
                    f"{kind} '{spec.name}': graph shape {graph_shape} does "
                    f"not match space shape {space_shape}"
                )
                break
        return errors

    def validate(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        """
        Validate the graph against the environment's spaces.

        The observation and action spaces are wrapped in ``Dict`` with a single
        ``"obs"`` / ``"action"`` key when they are not already Dicts, mirroring
        the exporter. Raises a single ``ValueError`` listing every mismatch so
        a wrong model/environment pairing fails loudly before any episodes run.

        Parameters
        ----------
        observation_space : gymnasium.Space
            The environment's observation space.
        action_space : gymnasium.Space
            The environment's action space.

        Raises
        ------
        ValueError
            If any input/output name, dtype, or shape mismatches its space.
        """
        self._obs_space_is_natively_dict = isinstance(observation_space, gym.spaces.Dict)
        self._action_space_is_natively_dict = isinstance(action_space, gym.spaces.Dict)

        if not isinstance(observation_space, gym.spaces.Dict):
            observation_space = gym.spaces.Dict({"obs": observation_space})
        if not isinstance(action_space, gym.spaces.Dict):
            action_space = gym.spaces.Dict({"action": action_space})

        errors: List[str] = []

        graph_inputs = set(self.inputs.keys())
        expected_inputs = set(observation_space.spaces.keys())
        for missing in sorted(expected_inputs - graph_inputs):
            errors.append(f"missing input '{missing}'")
        for extra in sorted(graph_inputs - expected_inputs):
            errors.append(f"unexpected input '{extra}'")
        for name, space in observation_space.spaces.items():
            spec = self.inputs.get(name)
            if spec is None:
                continue
            if spec.shape[0] is not None and spec.shape[0] != 1:
                errors.append(
                    f"observation '{name}': graph has a fixed batch size "
                    f"{spec.shape[0]}; expected a dynamic batch dimension"
                )
            expected_dtype = _expected_np_dtype(space)
            if expected_dtype is not None and spec.dtype != expected_dtype:
                errors.append(
                    f"observation '{name}': graph dtype {spec.dtype} does not "
                    f"match expected {expected_dtype} for "
                    f"{type(space).__name__} space"
                )
            errors.extend(
                self._check_space_shape(spec, space, "observation")
            )

        graph_outputs = set(self.outputs.keys())
        expected_outputs = set(action_space.spaces.keys())
        for missing in sorted(expected_outputs - graph_outputs):
            errors.append(f"missing output '{missing}'")
        for extra in sorted(graph_outputs - expected_outputs):
            errors.append(f"unexpected output '{extra}'")
        for name, space in action_space.spaces.items():
            spec = self.outputs.get(name)
            if spec is None:
                continue
            if spec.shape[0] is not None and spec.shape[0] != 1:
                errors.append(
                    f"action '{name}': graph has a fixed batch size "
                    f"{spec.shape[0]}; expected a dynamic batch dimension"
                )
            expected_dtype = _expected_np_dtype(space)
            if expected_dtype is not None and spec.dtype != expected_dtype:
                errors.append(
                    f"action '{name}': graph dtype {spec.dtype} does not "
                    f"match expected {expected_dtype} for "
                    f"{type(space).__name__} space"
                )
            errors.extend(self._check_space_shape(spec, space, "action"))

        if errors:
            raise ValueError(
                "ONNX model does not match the environment spaces:\n- "
                + "\n- ".join(errors)
            )

        self._obs_space = observation_space
        self._action_space = action_space

    def _warn_once(self, name: str, source_dtype: np.dtype) -> None:
        """
        Warn (once per input) about an observation dtype cast.

        Parameters
        ----------
        name : str
            The observation input name.
        source_dtype : numpy.dtype
            The dtype the environment provided.
        """
        if name in self._warned_casts:
            return
        self._warned_casts.add(name)
        logger.warning(
            "Casting observation '%s' from %s to the ONNX input dtype %s. "
            "The Unreal environment is expected to provide observations in "
            "the exported dtype.",
            name,
            source_dtype,
            self.inputs[name].dtype,
        )

    def predict(self, observations: Any) -> Any:
        """
        Run one batched inference step.

        Parameters
        ----------
        observations : numpy.ndarray or Dict[str, numpy.ndarray]
            Raw environment observations with a leading batch dimension
            (e.g. directly from a Schola ``VecEnv``).

        Returns
        -------
        numpy.ndarray or Dict[str, numpy.ndarray]
            Batched actions in the same layout as the (unwrapped) action space:
            a dict keyed by action name for Dict action spaces, a single array
            otherwise. Actions are returned verbatim (no clipping, no
            post-processing), matching what the Unreal NNE runtime consumes.

        Raises
        ------
        RuntimeError
            If ``validate`` has not been called yet.
        ValueError
            If observations are incompatible with the graph inputs.
        """
        if self._obs_space is None or self._action_space is None:
            raise RuntimeError(
                "OnnxScholaModel.validate() must be called before predict()."
            )

        batch_size: Optional[int] = None
        feed: Dict[str, np.ndarray] = {}
        for name, spec in self.inputs.items():
            if isinstance(observations, dict):
                obs = observations[name]
            else:
                obs = observations
            obs = np.asarray(obs)

            space_shape = self._obs_space[name].shape
            if obs.shape == space_shape:
                # Unbatched single observation; add a leading batch dim.
                obs = obs[np.newaxis, ...]

            if batch_size is None:
                batch_size = obs.shape[0]
            elif obs.shape[0] != batch_size:
                raise ValueError(
                    f"observation '{name}': batch size {obs.shape[0]} does "
                    f"not match {batch_size} from other inputs"
                )

            if spec.shape[0] is not None and spec.shape[0] != batch_size:
                raise ValueError(
                    f"observation '{name}': graph has a fixed batch size "
                    f"{spec.shape[0]} but {batch_size} observations were "
                    "provided"
                )

            if obs.dtype != spec.dtype:
                self._warn_once(name, obs.dtype)
                obs = obs.astype(spec.dtype)

            # Discrete scalars are exported with an explicit size-1 dim, so
            # the feed layout always follows the graph (not the space).
            rest = tuple(
                graph_dim if graph_dim is not None else space_dim
                for graph_dim, space_dim in zip(spec.shape[1:], space_shape)
            )
            if space_shape == () and spec.shape[1:] == (1,):
                rest = (1,)
            target_shape = (batch_size,) + rest
            if obs.size != int(np.prod(target_shape, dtype=np.int64)):
                raise ValueError(
                    f"observation '{name}': expected {target_shape} elements "
                    f"(shape {target_shape}), got {obs.shape}"
                )
            feed[name] = np.ascontiguousarray(obs.reshape(target_shape))

        raw_outputs = self._session.run(None, feed)
        output_names = [tensor.name for tensor in self._session.get_outputs()]

        actions: Dict[str, np.ndarray] = {}
        for name, arr in zip(output_names, raw_outputs):
            arr = np.asarray(arr)
            if name not in self._action_space.spaces:
                raise RuntimeError(
                    f"Unexpected ONNX output '{name}'; "
                    "validate() should have caught this."
                )
            space_shape = self._action_space[name].shape
            target_shape = (arr.shape[0],) + space_shape
            if arr.size != int(np.prod(target_shape, dtype=np.int64)):
                raise ValueError(
                    f"action '{name}': expected {target_shape} elements, "
                    f"got {arr.shape}"
                )
            actions[name] = arr.reshape(target_shape)

        if self._action_space_is_natively_dict:
            return actions
        return actions["action"]
