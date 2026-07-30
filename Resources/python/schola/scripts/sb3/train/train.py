# Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Script to train a Stable Baselines3 model using Schola.

PATCHED COPY: resolves `network_architecture_settings.features_extractor_class`
(a dotted import path) into `policy_kwargs["features_extractor_class"]` so a
custom SB3 features extractor can be selected from YAML. Replaces
Plugins/Schola/Resources/python/schola/scripts/sb3/train/train.py.
Pair with the patched settings.py in this same folder.
"""

from dataclasses import asdict
import os
import logging
import signal
from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    cast,
)

from schola.scripts.common.settings import (
    ExternalSimulatorConfig,
    get_activation_function,
)
from schola.scripts.common.command_template import MetaAlgCommand
from schola.scripts.sb3.train.settings import (
    PPOTrainSettings,
    SACTrainSettings,
    Sb3TrainScriptSettings,
)
from cyclopts import App
from schola.scripts.common.panel import print_error

# Logging setup (idempotent)
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

logger = logging.getLogger(__name__)


def warn_if_small_image_observation(observation_space, threshold: int = 64):
    """Issue a panel warning if any Box observation that looks image-like has
    a spatial dimension smaller than `threshold`.

    Heuristic: Treat Box spaces with shape len 2 (H, W) or len 3 (C,H,W) / (H,W,C)
    as image-like. For 3D, pick the two largest dims as spatial. If min spatial < threshold
    we warn once.
    """
    import gymnasium as gym

    def _iter_box_spaces(space):
        if isinstance(space, gym.spaces.Box):
            yield space
        elif isinstance(space, gym.spaces.Dict):
            for s in space.spaces.values():
                yield from _iter_box_spaces(s)
        elif isinstance(space, gym.spaces.Tuple):
            for s in space.spaces:
                yield from _iter_box_spaces(s)

    for box in _iter_box_spaces(observation_space):
        shape = box.shape
        if len(shape) == 2:
            h, w = shape
        elif len(shape) == 3:
            # Two largest dims are spatial (robust to (C,H,W) vs (H,W,C)).
            h, w = sorted(shape, reverse=True)[:2]
        else:
            continue

        if min(h, w) < threshold:
            print_error(
                f"Image observation detected with shape {shape}; min dimension < {threshold}. "
                "The default SB3 CNN may fail or produce poor features. Consider resizing "
                "or providing a custom features_extractor."
            )
            break


def _resolve_features_extractor_class(dotted_path: str):
    """
    Import and return the class referenced by `dotted_path` (e.g.
    'my_project.extractors.MyCombinedExtractor').

    Parameters
    ----------
    dotted_path : str
        Fully qualified dotted path to a class implementing SB3's
        `BaseFeaturesExtractor` interface. The containing module must be
        importable (on PYTHONPATH) when this runs.

    Returns
    -------
    type
        The resolved class object.
    """
    import importlib

    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"features_extractor_class must be a dotted path 'module.ClassName' (got '{dotted_path}')."
        )
    module = importlib.import_module(module_path)
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(
            f"Module '{module_path}' has no attribute '{class_name}' (from features_extractor_class='{dotted_path}')."
        ) from e


def main(args: Sb3TrainScriptSettings) -> Optional[Tuple[float, float]]:
    """
    Main function for training a Stable Baselines3 model using Schola.

    Parameters
    ----------
    args : Sb3ScriptSettings
        The arguments for the script.

    Returns
    -------
    Optional[Tuple[float,float]]
        The mean and standard deviation of the rewards if evaluation is enabled, otherwise None.
    """

    if args.training_settings.pbar:
        try:
            import tqdm
        except Exception:
            logger.warning("tqdm not installed. disabling PBar")
            args.training_settings.pbar = False

    if args.training_settings.pbar:
        try:
            import rich
        except Exception:
            logger.warning("rich not installed. disabling PBar")
            args.training_settings.pbar = False

    if args.logging_settings.enable_tensorboard:
        try:
            import tensorboard
        except Exception:
            logger.warning("tensorboard not installed. Disabling tensorboard logging")
            args.logging_settings.enable_tensorboard = False

    from schola.sb3.export import convert_ckpt_to_onnx_for_unreal
    import gymnasium as gym
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import VecNormalize

    from schola.scripts.sb3.utils import RewardCallback, CustomProgressBarCallback
    from schola.sb3.env import VecEnv
    from schola.sb3.async_env import AsyncVecEnv
    from schola.sb3.utils import VecMergeDictActionWrapper
    from schola.core.error_manager import ScholaErrorContextManager
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from stable_baselines3.common.base_class import BaseAlgorithm
    from schola.sb3.export import save_model_as_onnx
    from stable_baselines3.common import utils

    # initialize so we can force closure at the end
    env = None
    model: Optional[BaseAlgorithm] = None
    try:
        # This context manager redirects GRPC errors into custom error types to help debug
        with ScholaErrorContextManager() as err_ctxt:

            # make a gym environment (single or multi-simulator)
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

            if isinstance(env.action_space, gym.spaces.Dict):
                logger.warning(
                    "SB3 doesn't support dictionary action spaces. Attempting to merge into a single non-composite action space (e.g. Box, MultiDiscrete). This will cause issues with the ONNX Export."
                )
                env = VecMergeDictActionWrapper(env)

            model_loaded = False
            if args.resume_settings.resume_from:
                try:
                    model = args.algorithm_settings.constructor.load(
                        args.resume_settings.resume_from, env=env
                    )
                    model_loaded = True
                except Exception as e:
                    logger.warning(
                        "Error loading model '%s': %s. Training from scratch",
                        args.resume_settings.resume_from,
                        e,
                    )

            # When resuming, SB3 restores optimizer / PPO fields from the checkpoint. Without this,
            # YAML `algorithm` blocks (e.g. learning_rate, target_kl) are silently ignored — the
            # most visible symptom is train/learning_rate staying at the old zip value (often 3e-4).
            if model_loaded:
                algo = args.algorithm_settings
                if hasattr(model, "learning_rate"):
                    model.learning_rate = algo.learning_rate  # type: ignore[union-attr]
                    if hasattr(model, "_setup_lr_schedule"):
                        model._setup_lr_schedule()  # type: ignore[attr-defined]
                if hasattr(model, "target_kl"):
                    model.target_kl = getattr(algo, "target_kl", None)  # type: ignore[union-attr]
                logger.info(
                    "Resumed model: applied YAML learning_rate=%s and target_kl=%s from algorithm settings.",
                    getattr(algo, "learning_rate", None),
                    getattr(algo, "target_kl", None),
                )

            if not model_loaded:
                policy_kwargs: Optional[dict[str, Any]] = None
                net_arch_settings = args.network_architecture_settings
                if (
                    net_arch_settings.activation
                    or net_arch_settings.critic_parameters
                    or net_arch_settings.policy_parameters
                    or net_arch_settings.features_extractor_class
                ):
                    if isinstance(env.observation_space, gym.spaces.Dict):
                        policy_kwargs = dict(
                            features_extractor_kwargs={"normalized_image": True}
                        )
                    else:
                        policy_kwargs = dict()

                    if net_arch_settings.features_extractor_class:
                        if not isinstance(env.observation_space, gym.spaces.Dict):
                            logger.warning(
                                "network_architecture_settings.features_extractor_class is set "
                                "but the observation space isn't a Dict; ignoring it (SB3's "
                                "MlpPolicy doesn't take a features_extractor_class override here)."
                            )
                        else:
                            policy_kwargs["features_extractor_class"] = (
                                _resolve_features_extractor_class(
                                    net_arch_settings.features_extractor_class
                                )
                            )
                            if net_arch_settings.features_extractor_kwargs:
                                policy_kwargs["features_extractor_kwargs"].update(
                                    net_arch_settings.features_extractor_kwargs
                                )
                            logger.info(
                                "Using custom features_extractor_class=%s with kwargs=%s",
                                net_arch_settings.features_extractor_class,
                                policy_kwargs["features_extractor_kwargs"],
                            )

                    if net_arch_settings.activation:
                        policy_kwargs["activation_fn"] = get_activation_function(
                            net_arch_settings.activation
                        )

                    if (
                        net_arch_settings.critic_parameters
                        or net_arch_settings.policy_parameters
                    ):
                        # default to nothing
                        policy_kwargs["net_arch"] = dict(vf=[], pi=[], qf=[])

                    if net_arch_settings.critic_parameters:
                        policy_kwargs["net_arch"][
                            args.algorithm_settings.critic_type
                        ] = net_arch_settings.critic_parameters

                    if net_arch_settings.policy_parameters:
                        policy_kwargs["net_arch"][
                            "pi"
                        ] = net_arch_settings.policy_parameters

                model = args.algorithm_settings.constructor(
                    policy=(
                        "MultiInputPolicy"
                        if isinstance(env.observation_space, gym.spaces.Dict)
                        else "MlpPolicy"
                    ),
                    env=env,
                    verbose=args.logging_settings.sb3_verbosity,
                    policy_kwargs=policy_kwargs,
                    **asdict(args.algorithm_settings),
                )

            assert model is not None, "Model is None. This should never happen."

            if args.resume_settings.load_vecnormalize:
                if model.get_vec_normalize_env() is None:
                    try:
                        VecNormalize.load(
                            str(args.resume_settings.load_vecnormalize), env
                        )
                    except Exception:
                        logger.warning(
                            "Error loading saved VecNormalize parameters. Skipping."
                        )
                else:
                    logger.warning(
                        "resume_settings.load_vecnormalize was true but no VecNormalize wrapper exists to load to. Skipping."
                    )

            if args.resume_settings.load_replay_buffer:
                if hasattr(model, "load_replay_buffer"):
                    try:
                        model.load_replay_buffer(
                            args.resume_settings.load_replay_buffer
                        )  # type: ignore
                    except Exception:
                        logger.warning("Error loading saved Replay Buffer. Skipping.")
                else:
                    logger.warning(
                        "resume_settings.load_replay_buffer was true but Model does not have a Replay Buffer to load to. Skipping."
                    )

            callbacks = []

            # grab all loggers that we can find installed in the pc,
            output_formats = []

            # This is a bit of a hack, since output_formats doesn't have a getter/setter but it this is totally safe otherwise
            sb3_logger = utils.configure_logger(
                args.logging_settings.sb3_verbosity,
                (
                    str(args.logging_settings.log_dir)
                    if args.logging_settings.enable_tensorboard
                    else None
                ),
                args.algorithm_settings.name,
                args.resume_settings.reset_timestep,
            )

            # Append CSV output when file logging is enabled (independent of tensorboard)
            if args.logging_settings.enable_csv_logging:
                from stable_baselines3.common.logger import CSVOutputFormat
                # Ensure directory exists (__post_init__ may not have run with CLI-set values)
                args.logging_settings.log_dir.mkdir(parents=True, exist_ok=True)
                csv_path = args.logging_settings.log_dir / f"{args.algorithm_settings.name.lower()}.csv"
                sb3_logger.output_formats.append(CSVOutputFormat(str(csv_path)))

            sb3_logger.output_formats += output_formats
            model.set_logger(sb3_logger)

            if args.logging_settings.enable_tensorboard or args.logging_settings.enable_csv_logging:
                reward_callback = RewardCallback(
                    verbose=args.logging_settings.callback_verbosity,
                    frequency=args.logging_settings.log_freq,
                    num_envs=env.num_envs,
                    info_keys=args.logging_settings.info_log_keys,
                )
                callbacks.append(reward_callback)

            if args.checkpoint_settings.enable_checkpoints:
                ckpt_callback = CheckpointCallback(
                    save_freq=args.checkpoint_settings.save_freq,
                    save_path=str(args.checkpoint_settings.checkpoint_dir),
                    name_prefix=args.name_prefix,
                    save_replay_buffer=args.checkpoint_settings.save_replay_buffer,
                    save_vecnormalize=args.checkpoint_settings.save_vecnormalize,
                )
                callbacks.append(ckpt_callback)

            if args.training_settings.pbar:
                pbar_callback = CustomProgressBarCallback()
                callbacks.append(pbar_callback)

            env_options = getattr(args.environment_settings, "env_options", None)
            if env_options:
                # Inherited from SB3's `set_options`
                env.set_options(options=env_options)

            model.learn(
                total_timesteps=args.training_settings.timesteps,
                callback=callbacks,
                reset_num_timesteps=args.resume_settings.reset_timestep,
                log_interval=args.logging_settings.log_freq,
            )

            if args.checkpoint_settings.save_final_policy:
                logger.info("... Saving final policy checkpoint")
                if isinstance(env, VecMergeDictActionWrapper):
                    setattr(model, "__unmerged_action_space", env.unmerged_action_space)

                model.save(
                    os.path.join(
                        args.checkpoint_settings.checkpoint_dir,
                        f"{args.name_prefix}_final.zip",
                    )
                )

                if (
                    args.checkpoint_settings.save_vecnormalize
                    and model.get_vec_normalize_env() is not None
                ):
                    model.get_vec_normalize_env().save(
                        os.path.join(
                            args.checkpoint_settings.checkpoint_dir,
                            f"{args.name_prefix}_vec_normalize_final.zip",
                        )
                    )

            if args.checkpoint_settings.export_onnx:
                logger.info("... Exporting final policy to ONNX")
                if isinstance(env, VecMergeDictActionWrapper):
                    action_space = env.unmerged_action_space
                else:
                    action_space = model.action_space
                save_model_as_onnx(
                    model,
                    f"{args.checkpoint_settings.checkpoint_dir}/{args.name_prefix}_final.onnx",
                    action_space=action_space,
                )

            if not args.training_settings.disable_eval:
                logger.info("... Evaluating the model")
                env_with_monitor = VecMonitor(env)
                mean_reward, std_reward = evaluate_policy(
                    model, env_with_monitor, n_eval_episodes=10, deterministic=True
                )

                logger.info(
                    "Evaluation complete: mean_reward=%.2f +/- %.2f",
                    mean_reward,
                    std_reward,
                )
                env.close()
                return mean_reward, std_reward  # type: ignore
            else:
                logger.info("Evaluation disabled. Skipping.")
                env.close()
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            logger.info("Ctrl-C received. Shutting down gracefully;")
            signal.signal(signal.SIGINT, signal.SIG_IGN)  # Protect cleanup phase
        if env is not None:
            env.close()
        raise


app = App(name="train", help="Train a model using StableBaselines3")


class MetaTrainSB3Command(MetaAlgCommand[Sb3TrainScriptSettings]):
    """
    ``MetaAlgCommand`` configuration for Stable-Baselines3 (PPO and SAC).

    See Also
    --------
    MetaAlgCommand
    """

    @property
    def algorithm_table(self):
        return {
            "sac": SACTrainSettings,
            "ppo": PPOTrainSettings,
        }

    @property
    def algorithm_help(self):
        return {
            "sac": "Train a model using Soft Actor-Critic(SAC) with StableBaselines3.",
            "ppo": "Train a model using Proximal Policy Optimization(PPO) with StableBaselines3.",
        }


app = MetaTrainSB3Command(app, Sb3TrainScriptSettings, main, logger).make()
