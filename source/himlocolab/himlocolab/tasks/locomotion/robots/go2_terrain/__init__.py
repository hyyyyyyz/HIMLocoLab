import gymnasium as gym

gym.register(
    id="Unitree-Go2-Terrain",
    entry_point="himlocolab.envs:HimlocoManagerBasedRLEnv",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "himloco_rsl_rl_cfg": f"himlocolab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
        "rsl_rl_cfg_entry_point": f"himlocolab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg"
    },
)

gym.register(
    id="Unitree-Go2-Terrain-Play",
    entry_point="himlocolab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "himloco_rsl_rl_cfg": f"himlocolab.tasks.locomotion.agents.himloco_rsl_rl_cfg:PPORunnerCfg",
    },
)
