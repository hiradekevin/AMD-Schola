# Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

"""
Cyclopts dataclasses for running ONNX inference with Schola.
"""

from dataclasses import dataclass, field
from typing import Annotated, Optional
from pathlib import Path

from cyclopts import Parameter, validators

from schola.scripts.common.settings import EnvironmentSettings
from schola.scripts.sb3.settings import Sb3BaseLoggingSettings


@dataclass
class Sb3OnnxRunScriptSettings:
    """
    Top-level settings for running ONNX inference with a Schola environment.
    """

    onnx_path: Annotated[
        Optional[Path],
        Parameter(
            group="Inference Arguments",
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
            required=True,
            alias="-m",
        ),
    ] = None
    "Path to the ONNX policy exported with ``schola sb3 export`` (required via ``--onnx-path``)."

    n_eval_episodes: Annotated[
        int, Parameter(validator=validators.Number(gte=1), group="Inference Arguments")
    ] = 10
    "Number of evaluation episodes to run the ONNX policy for."

    vecnormalize: Annotated[
        Optional[Path],
        Parameter(
            group="Inference Arguments",
            validator=validators.Path(exists=True, file_okay=True, dir_okay=False),
        ),
    ] = None
    "Optional ``VecNormalize`` statistics file (``.zip``) saved alongside the policy. Required when the policy was trained inside a ``VecNormalize`` wrapper."

    logging_settings: Annotated[
        Sb3BaseLoggingSettings, Parameter(group="Logging Arguments", name="*")
    ] = field(default_factory=Sb3BaseLoggingSettings)
    "Logging verbosity for Schola and SB3 components."

    environment_settings: Annotated[
        EnvironmentSettings, Parameter(group="Environment Arguments", name="*")
    ] = field(default_factory=EnvironmentSettings)
    "How to launch or attach to the Unreal simulator and gRPC protocol for the environment."
