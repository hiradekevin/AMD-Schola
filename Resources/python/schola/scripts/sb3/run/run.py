# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Run inference with an ONNX policy against a Schola-backed environment.
"""

import logging
from typing import Dict, List, Tuple, cast

import gymnasium as gym
import numpy as np

from cyclopts import App

from schola.scripts.common.command_template import MetaNoAlgCommand
from schola.scripts.common.settings import ExternalSimulatorConfig
from schola.scripts.sb3.run.settings import Sb3OnnxRunScriptSettings

# Logging setup (idempotent)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def merge_dict_actions(
    actions: Dict[str, np.ndarray], action_space: gym.spaces.Dict
) -> np.ndarray:
    """
    Merge batched per-key actions into the flat layout expected by Schola's
    SB3 ``VecEnv`` for Dict action spaces.

    This is the inverse of ``schola.sb3.utils.split_value``, which the
    environment uses to restore the per-key actions before sending them to the
    simulator. The merged layout mirrors ``VecMergeDictActionWrapper``'s merged
    action space (the space the policy was trained against), so the ONNX
    outputs must be assembled exactly this way before calling ``env.step``.

    Parameters
    ----------
    actions : Dict[str, numpy.ndarray]
        Batched actions keyed by action space name.
    action_space : gymnasium.spaces.Dict
        The environment's (unwrapped) Dict action space.

    Returns
    -------
    numpy.ndarray
        Merged flat actions: concatenated along the last axis for Box and
        MultiBinary branches, stacked per branch for Discrete/MultiDiscrete
        branches.

    Raises
    ------
    TypeError
        If the Dict action space mixes unsupported space types.
    """
    first_space = next(iter(action_space.spaces.values()))
    branch_arrays = [actions[name] for name in action_space.spaces.keys()]
    if isinstance(first_space, gym.spaces.Box):
        return np.concatenate(branch_arrays, axis=-1)
    if isinstance(first_space, (gym.spaces.Discrete, gym.spaces.MultiDiscrete)):
        return np.stack(branch_arrays, axis=-1).astype(np.int64)
    if isinstance(first_space, gym.spaces.MultiBinary):
        return np.concatenate(
            [array.astype(np.bool_) for array in branch_arrays], axis=-1
        )
    raise TypeError(
        f"Unsupported action space type {type(first_space)}; expected Box, "
        "Discrete, MultiDiscrete, or MultiBinary."
    )


def main(args: Sb3OnnxRunScriptSettings) -> Tuple[float, float]:
    """
    Run an exported ONNX policy against a Schola-backed environment.

    The ONNX policy is executed verbatim: raw environment observations are fed
    to the graph and graph outputs are sent back to the environment, exactly
    like the Unreal NNE runtime. Episode statistics are reported in the same
    format as ``schola sb3 eval``.

    Parameters
    ----------
    args : Sb3OnnxRunScriptSettings
        CLI / script configuration.

    Returns
    -------
    tuple[float, float]
        Mean and standard deviation of episodic return over
        ``n_eval_episodes``.
    """
    from stable_baselines3.common.vec_env import VecNormalize

    from schola.core.error_manager import ScholaErrorContextManager
    from schola.onnx.model import OnnxScholaModel
    from schola.sb3.async_env import AsyncVecEnv
    from schola.sb3.env import VecEnv

    env = None
    try:
        # Load and sanity-check the ONNX graph before launching the simulator
        # so a bad export fails fast without starting Unreal.
        model = OnnxScholaModel(args.onnx_path)
        logger.info(
            "Loaded ONNX model '%s' (inputs=%s, outputs=%s)",
            args.onnx_path,
            list(model.inputs),
            list(model.outputs),
        )

        with ScholaErrorContextManager():

            sim_args = args.environment_settings.simulator_settings
            protocol_args = args.environment_settings.protocol_settings
            n_sim = sim_args.num_simulators
            if (
                n_sim > 1
                and not isinstance(sim_args, ExternalSimulatorConfig)
                and getattr(sim_args, "per_process_saved_root", None) is None
            ):
                logger.warning(
                    "num_simulators=%d but per_process_saved_root is unset: parallel "
                    "standalones default to the same staged Saved/ tree (logs, locks). "
                    "Set environment.simulator.(project|executable).per_process_saved_root "
                    "to a directory; Schola adds sim_0 .. sim_%d there via -userdir=.",
                    n_sim,
                    n_sim - 1,
                )

            if n_sim == 1:
                env = VecEnv(
                    sim_args.make(),
                    protocol_args.make(),
                    verbosity=args.logging_settings.schola_verbosity,
                )
            elif isinstance(sim_args, ExternalSimulatorConfig):
                simulators = [sim_args.make() for _ in range(n_sim)]
                async_protocols = protocol_args.make_n_async(n_sim)
                env = AsyncVecEnv(
                    simulators,
                    async_protocols,
                    verbosity=args.logging_settings.schola_verbosity,
                )
            else:
                from schola.core.simulators.unreal.executable_simulator import (
                    UnrealExecutable,
                )

                primary = cast(UnrealExecutable, sim_args.make())
                simulators = [primary] + primary.spawn_executables(n_sim - 1)
                async_protocols = protocol_args.make_n_async(n_sim)
                env = AsyncVecEnv(
                    simulators,
                    async_protocols,
                    verbosity=args.logging_settings.schola_verbosity,
                )

            if args.vecnormalize is not None:
                env = VecNormalize.load(str(args.vecnormalize), env)
                env.training = False
                env.norm_reward = True
                logger.info(
                    "Loaded VecNormalize statistics from %s", args.vecnormalize
                )

            model.validate(env.observation_space, env.action_space)
            logger.info(
                "ONNX model validated against observation space %s and "
                "action space %s",
                env.observation_space,
                env.action_space,
            )

            is_dict_action_space = isinstance(env.action_space, gym.spaces.Dict)

            episode_rewards: List[float] = []
            episode_lengths: List[int] = []
            current_returns = np.zeros(env.num_envs, dtype=np.float64)
            current_lengths = np.zeros(env.num_envs, dtype=np.int64)

            obs = env.reset()
            while len(episode_rewards) < args.n_eval_episodes:
                actions = model.predict(obs)
                if is_dict_action_space:
                    actions = merge_dict_actions(actions, env.action_space)
                obs, rewards, dones, _infos = env.step(actions)

                current_returns += rewards
                current_lengths += 1
                for i in range(env.num_envs):
                    if dones[i]:
                        episode_rewards.append(float(current_returns[i]))
                        episode_lengths.append(int(current_lengths[i]))
                        current_returns[i] = 0.0
                        current_lengths[i] = 0

            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))

            logger.info(
                "Evaluation complete: mean_reward=%.4f +/- %.4f (over %d episodes)",
                mean_reward,
                std_reward,
                len(episode_rewards),
            )
            # print out the per episode rewards
            per_episode_reward_str = "Per episode rewards: \n"
            for episode_reward, episode_length in zip(
                episode_rewards, episode_lengths
            ):
                per_episode_reward_str += (
                    f"\tEpisode reward: {episode_reward:.4f}, "
                    f"Episode length: {episode_length}\n"
                )
            logger.info(per_episode_reward_str)

            env.close()
            env = None
            return mean_reward, std_reward
    except (KeyboardInterrupt, Exception):
        if env is not None:
            env.close()
        raise


_run_app = App(name="run", help="Run an ONNX policy against a Schola environment")


run_app = MetaNoAlgCommand(_run_app, Sb3OnnxRunScriptSettings, main, logger).make()


if __name__ == "__main__":
    run_app.meta()
