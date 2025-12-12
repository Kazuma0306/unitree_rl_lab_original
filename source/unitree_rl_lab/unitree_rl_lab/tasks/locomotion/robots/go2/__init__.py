import gymnasium as gym

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Paper1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg2:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg2:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Paper2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg3:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg3:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        "skrl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Unitree-Go2-Proposed",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg4:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg4:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        # "skrl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Unitree-Go2-Proposed2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg5:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg5:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        # "skrl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Unitree-Go2-Proposed3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg6:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg6:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        # "skrl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Unitree-Go2-Proposed4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg7:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg7:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        # "skrl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents:skrl_ppo_cfg.yaml",
    },
)



gym.register(
    id="Unitree-Go2-Upper",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_up:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_up:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        # "skrl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Unitree-Go2-Upper2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg_up2:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg_up2:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
        # "skrl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents:skrl_ppo_cfg.yaml",
    },
)