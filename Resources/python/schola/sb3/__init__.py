# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
"""
Support for Stable Baselines 3 environments.
"""

from typing import Any, List, Optional, Sequence, Tuple, Union

from .async_env import AsyncVecEnv, is_iterable
from .env import BaseVecEnv, VecEnv

from schola.core.protocols.async_base_protocol import AsyncBaseRLProtocol

__all__ = ["AsyncVecEnv", "BaseVecEnv", "VecEnv", "make_vec_env"]


def make_vec_env(
    simulators: Union[Any, Sequence[Any]],
    protocols: Union[Any, Sequence[Any]],
    verbosity: int = 0,
):
    """
    Create a Schola SB3 VecEnv, automatically selecting the implementation.

    - A single (simulator, protocol) pair with synchronous protocols returns a
      synchronous :class:`VecEnv`.
    - Multiple pairs, or any asynchronous protocol (``AsyncBaseRLProtocol``),
      returns an :class:`AsyncVecEnv`, which overlaps all per-instance
      ``send_action_msg`` RPCs on one event loop so wall-clock wait time
      scales with ``max(RTT)`` instead of ``sum(RTT)``.

    Parameters
    ----------
    simulators : BaseSimulator or sequence of BaseSimulator
        One simulator per UE5 instance (single object or equal-length sequence).
    protocols : BaseRLProtocol or sequence of BaseRLProtocol
        One protocol per simulator (single object or equal-length sequence).
    verbosity : int, default=0
        Verbosity level for logging.

    Returns
    -------
    VecEnv or AsyncVecEnv
        The appropriate vectorized environment for the given instances.

    Raises
    ------
    ValueError
        If ``simulators`` and ``protocols`` have different lengths.
    """
    sim_list: List[Any] = (
        [simulators] if not is_iterable(simulators) else list(simulators)
    )
    proto_list: List[Any] = (
        [protocols] if not is_iterable(protocols) else list(protocols)
    )
    if len(sim_list) != len(proto_list):
        raise ValueError(
            "simulators and protocols must have the same length "
            f"({len(sim_list)} vs {len(proto_list)})."
        )
    if len(sim_list) == 1 and not isinstance(proto_list[0], AsyncBaseRLProtocol):
        return VecEnv(sim_list[0], proto_list[0], verbosity)
    return AsyncVecEnv(sim_list, proto_list, verbosity)
