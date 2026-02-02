from __future__ import annotations

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
import isaaclab.utils.math as math_utils

from isaaclab.envs.mdp import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformLevelVelocityCommandCfg



class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: UniformLevelVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.low_vel_env_lin_x_ranges)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        high_vel_env_ids = env_ids <= (self.num_envs * self.cfg.rel_high_vel_envs)
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        r_high = torch.empty(len(high_vel_env_ids), device=self.device)
        self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)
        # set y commands of high vel envs to zero
        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min) & \
                            (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range
        
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > \
                                            self.cfg.min_command_norm).unsqueeze(1)
        
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )