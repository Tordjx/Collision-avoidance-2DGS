#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 Inria

import gymnasium
import numpy as np
from upkie.envs import UpkieGroundVelocity
from upkie.utils.raspi import on_raspi

from config.settings import EnvSettings
from env.navigation_wrapper import NavigationWrapper


def make_rays_pink_env(
    velocity_env: UpkieGroundVelocity,
    env_settings: EnvSettings,
    eval_mode: bool = False,
) -> gymnasium.Wrapper:
    velocity_env = NavigationWrapper(velocity_env)
    if not on_raspi():
        from foa.rays_sim import RaysSimWrapper

        rescaled_accel_env = RaysSimWrapper(
            velocity_env,
            image_every=env_settings.image_every
        )

    else:
        from foa.rays_raspi import RaysRaspiWrapper
        rescaled_accel_env = RaysRaspiWrapper(
            velocity_env,
            image_every=env_settings.image_every
        )
    return rescaled_accel_env
