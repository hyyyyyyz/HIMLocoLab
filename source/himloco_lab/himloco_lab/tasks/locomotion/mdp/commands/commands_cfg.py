from dataclasses import MISSING

from isaaclab.utils import configclass

from .commands import UniformLevelVelocityCommand

from isaaclab.envs.mdp import UniformVelocityCommandCfg

@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    
    class_type: type = UniformLevelVelocityCommand
    
    curriculums_limit_ranges: tuple[float, float] = MISSING
    
    low_vel_env_lin_x_ranges: tuple[float, float] = MISSING
    
    rel_high_vel_envs: float = MISSING
    
    min_command_norm: float = MISSING