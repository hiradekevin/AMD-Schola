# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Base class for connections that use the gRPC server.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
import grpc
from schola.core.protocols.base_protocol import AutoResetType, BaseRLProtocol
from schola.core.protocols.protobuf.deserialize import from_proto, _flatten_training_state
from schola.core.protocols.protobuf.serialize import to_proto, fill_generic, serialize_actions_direct, _get_to_proto_handler, _get_fill_generic_handler
import schola.generated.GymConnector_pb2_grpc as gym_grpc
import schola.generated.GymConnector_pb2 as util_messages
import schola.generated.Definitions_pb2 as env_definitions
import schola.generated.State_pb2 as state
import schola.generated.StateUpdates_pb2 as state_updates
from schola.core.protocols.socket_protocol import SocketProtocolMixin
import gymnasium as gym
from gymnasium.vector.vector_env import AutoresetMode
import logging

logger = logging.getLogger(__name__)


class BaseGrpcProtocol(SocketProtocolMixin):
    """
    Base class for gym gRPC wiring shared by sync and async clients.

    Notes
    -----
    Subclasses implement ``channel_connected`` and own the concrete ``grpc`` channel type.
    """

    VALID_CREDENTIAL_MODES = ("local", "insecure")

    def __init__(
        self,
        url: str,
        port: Optional[int] = None,
        environment_start_timeout: Optional[int] = 45,
        credential_mode: Literal["local", "insecure"] = "local",
    ):
        super().__init__(url, port, client_only=(credential_mode == "insecure"))
        if credential_mode not in self.VALID_CREDENTIAL_MODES:
            raise ValueError(
                f"credential_mode must be one of {self.VALID_CREDENTIAL_MODES}, "
                f"got {credential_mode!r}"
            )
        self._gym_stub: Optional[gym_grpc.GymServiceStub] = None  # type: ignore
        self.environment_start_timeout = environment_start_timeout
        self.credential_mode = credential_mode
        # Pre-allocated protobuf message for reuse across steps (avoids per-step allocation)
        self._cached_state_update = state_updates.StateUpdate()

    @property
    def gym_stub(self) -> gym_grpc.GymServiceStub:
        assert self._gym_stub is not None, "gRPC stub is not initialized"
        return self._gym_stub

    def prepare_start_msg(
        self, auto_reset_type: AutoresetMode
    ) -> util_messages.GymConnectorStartRequest:
        start_msg = util_messages.GymConnectorStartRequest()

        if auto_reset_type == AutoresetMode.DISABLED:
            start_msg.autoreset_type = util_messages.AutoResetType.DISABLED
        elif auto_reset_type == AutoresetMode.SAME_STEP:
            start_msg.autoreset_type = util_messages.AutoResetType.SAME_STEP
        elif auto_reset_type == AutoresetMode.NEXT_STEP:
            start_msg.autoreset_type = util_messages.AutoResetType.NEXT_STEP

        return start_msg

    def prepare_reset_msg(
        self, seeds: Optional[List] = None, options: Optional[List] = None
    ) -> state_updates.StateUpdate:
        state_update = state_updates.StateUpdate(reset=state_updates.Reset())
        reset_msg: state_updates.Reset = state_update.reset

        if seeds is not None:
            for env_id, seed in enumerate(seeds):
                if seed is not None:
                    reset_msg.environments[env_id].seed = seed

        if options is not None:
            for env_id, option_dict in enumerate(options):
                for key, option in option_dict.items():
                    reset_msg.environments[env_id].options[key] = str(option)
        return state_update

    def prepare_action_msg(
        self, actions: Dict[int, Dict[str, Any]], action_space: Dict[str, gym.Space]
    ) -> state_updates.StateUpdate:
        state_update = self._cached_state_update
        state_update.Clear()
        state_update.step  # Initialize the step oneof
        state_update.status = state_updates.CommunicatorStatus.GOOD

        step = state_update.step
        to_proto_cached = None
        fill_cached = None
        sample_space = next(iter(action_space.values())) if action_space else None
        if sample_space is not None:
            sample_type = type(sample_space)
            to_proto_cached = _get_to_proto_handler(sample_type)

        for env_id in actions:
            env_update = step.environments.add()
            env_updates = env_update.updates
            for agent_id, action in actions[env_id].items():
                if to_proto_cached is not None:
                    point_msg = to_proto_cached(sample_space, action)
                else:
                    point_msg = to_proto(action_space[agent_id], action)
                if fill_cached is None:
                    fill_cached = _get_fill_generic_handler(type(point_msg))
                fill_cached(point_msg, env_updates[agent_id])
        return state_update

    @property
    def channel_connected(self) -> bool:
        raise NotImplementedError(
            "channel_connected is not implemented for BaseGrpcProtocol"
        )

    def __bool__(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            True iff the connection is active
        """
        return (self.has_socket or self.is_started) and self.channel_connected

    @property
    def properties(self) -> Dict[str, Any]:
        return self.mixin_properties


class GrpcProtocol(BaseGrpcProtocol, BaseRLProtocol):
    """
    Synchronous ``BaseRLProtocol`` over gRPC.

    See Also
    --------
    BaseGrpcProtocol
    BaseRLProtocol
    """

    def __init__(
        self,
        url: str,
        port: Optional[int] = None,
        environment_start_timeout: Optional[int] = 45,
        credential_mode: Literal["local", "insecure"] = "local",
        grpc_close_timeout: float = 5.0,
    ):
        super().__init__(url, port, environment_start_timeout, credential_mode)
        self.grpc_close_timeout = grpc_close_timeout
        self.channel: Optional[grpc.Channel] = None

    def close(self) -> None:
        """
        Close the Unreal Connection. Method must be safe to call multiple times.
        """
        logger.debug("Close invoked")
        SocketProtocolMixin.on_close(self)

        if self.channel_connected:
            try:
                state_update = state_updates.StateUpdate(
                    status=state_updates.CommunicatorStatus.CLOSED
                )
                self.gym_stub.UpdateState(state_update, timeout=self.grpc_close_timeout)
            except grpc.RpcError as e:
                logger.debug(
                    "gRPC close message failed (connection may already be closed): %s",
                    e,
                )
            finally:
                self.channel.close()
                self.channel = None
                self._gym_stub = None  # type: ignore
        else:
            logger.debug("gRPC channel already closed")

    def start(self) -> None:
        """
        Open the Connection to Unreal Engine.
        """
        SocketProtocolMixin.on_start(self)

        options = [
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
        ]

        if self.credential_mode == "insecure":
            self.channel = grpc.insecure_channel(
                self.address, options=options
            ).__enter__()
        else:
            self.channel = grpc.secure_channel(
                self.address, grpc.local_channel_credentials(), options=options
            ).__enter__()
        self._gym_stub = gym_grpc.GymServiceStub(self.channel)

    def send_startup_msg(
        self, auto_reset_type: AutoresetMode = AutoresetMode.SAME_STEP
    ):

        start_msg = self.prepare_start_msg(auto_reset_type)
        self.gym_stub.StartGymConnector(
            start_msg, timeout=self.environment_start_timeout, wait_for_ready=True
        )

    def get_definition(
        self,
    ) -> Tuple[
        List[List[str]],
        List[Dict[str, str]],
        Dict[int, Dict[str, gym.Space]],
        Dict[int, Dict[str, gym.Space]],
    ]:
        training_defn: env_definitions.TrainingDefinition = (
            self.gym_stub.RequestTrainingDefinition(
                util_messages.TrainingDefinitionRequest()
            )
        )

        uids, agent_types, obs_spaces, act_spaces = from_proto(training_defn)

        return uids, agent_types, obs_spaces, act_spaces

    def send_reset_msg(
        self, seeds: Optional[List] = None, options: Optional[List] = None
    ):

        # abort any inprogress stuff
        state_update = self.prepare_reset_msg(seeds, options)

        response: state.State = self.gym_stub.UpdateState(state_update)
        obs, info = from_proto(response.initial_state)
        # Convert from Dict[Dict[envID, Dict[agentID, Any]]] to list[Dict[agentID, Any]]
        observations = [obs[env_id] for env_id in range(len(obs))]
        infos = [info[env_id] for env_id in range(len(info))]
        return observations, infos

    def send_action_msg(
        self, actions: Dict[int, Dict[str, Any]], action_space: Dict[str, gym.Space]
    ):
        state_update = self.prepare_action_msg(actions, action_space)
        response = self.gym_stub.UpdateState(state_update)

        observations, rewards, terminateds, truncateds, infos = from_proto(
            response.training_state
        )

        if response.HasField("initial_state"):
            initial_obs, initial_info = from_proto(response.initial_state)
        else:
            initial_obs, initial_info = {}, {}

        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
            initial_obs,
            initial_info,
        )

    def send_action_msg_flat(
        self, actions_flat, action_space, id_manager
    ):
        """
        Send actions using a flat array (bypasses intermediate dict creation).
        Returns data in the same format as send_action_msg.
        """
        serialize_actions_direct(actions_flat, id_manager, action_space, self._cached_state_update)
        response = self.gym_stub.UpdateState(self._cached_state_update)

        observations, rewards, terminateds, truncateds, infos = _flatten_training_state(
            response.training_state, id_manager
        )

        if response.HasField("initial_state"):
            initial_obs, initial_info = from_proto(response.initial_state)
        else:
            initial_obs, initial_info = {}, {}

        return (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
            initial_obs,
            initial_info,
        )

    @property
    def channel_connected(self) -> bool:
        """
        Returns whether the connection is active or not

        Returns
        -------
        bool
            Whether the connection is active or not
        """
        return self.channel is not None
