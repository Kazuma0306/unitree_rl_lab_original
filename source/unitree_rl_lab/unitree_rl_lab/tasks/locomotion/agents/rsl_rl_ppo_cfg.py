# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlRndCfg
from unitree_rl_lab.tasks.locomotion.robots.go2.locotransformer import VisionMLPActorCritic, LocoTransformerActorCritic


@configclass
class VisionMLPActorCriticCfg(RslRlPpoActorCriticCfg):
    """あなたのカスタムモデルVisionMLPActorCritic用の設定クラス。"""
    
    # 1. class_nameを、あなたのカスタムモデルへのパスで上書きします。

    class_name: str = "VisionMLPActorCritic"

    # 2. RslRlPpoActorCriticCfgにはない、あなたのモデル独自の引数をここに追加します。
    prop_obs_keys: list[str] = [
        "base_lin_vel", "base_ang_vel", "projected_gravity",
        "position_commands", "joint_pos_rel", "joint_vel_rel", "last_action"
    ]
    vision_obs_key: str = "camera_image"

    # 3. RslRlPpoActorCriticCfgから継承したデフォルト値を上書きすることもできます。
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [256, 128]  # 例
    critic_hidden_dims: list[int] = [256, 128] # 例
    activation: str = "elu"



@configclass
class LocoTransformerActorCriticCfg(RslRlPpoActorCriticCfg):
    """あなたのカスタムモデルVisionMLPActorCritic用の設定クラス。"""
    
    # 1. class_nameを、あなたのカスタムモデルへのパスで上書きします。

    class_name: str = "LocoTransformerActorCritic"

    # 2. RslRlPpoActorCriticCfgにはない、あなたのモデル独自の引数をここに追加します。
    prop_obs_keys: list[str] = [
        "base_lin_vel", "base_ang_vel", "projected_gravity",
        "position_commands", "joint_pos_rel", "joint_vel_rel", "last_action"
    ]
    vision_obs_key: str = "camera_image"

    # 3. RslRlPpoActorCriticCfgから継承したデフォルト値を上書きすることもできます。
    init_noise_std: float = 1.0
    actor_hidden_dims: list[int] = [256, 128]  # 例
    critic_hidden_dims: list[int] = [256, 128] # 例
    activation: str = "elu"



@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1200 #50000
    save_interval = 100
    experiment_name = ""  # same as task name
    # empirical_normalization = False 
    actor_obs_normalization=False, 
    critic_obs_normalization=False,

    #for paper 1

    # obs_groups =  {
    #     "policy": ["base_lin_vel",
    #             "base_ang_vel",
    #             "projected_gravity",
    #             "position_commands",
    #             "joint_pos_rel",
    #             "joint_vel_rel",
    #             "last_action", "height_scanner"],
    #     "critic":["base_lin_vel",
    #             "base_ang_vel",
    #             "projected_gravity",
    #             "position_commands",
    #             "joint_pos_rel",
    #             "joint_vel_rel",
    #             "joint_effort",
    #             "last_action", "height_scanner"],


    # }

    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    # )

    # for paper2

    obs_groups =  {
        "policy": ["base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "position_commands",
                "joint_pos_rel",
                "joint_vel_rel",
                "last_action", 
                "camera_image"
                ],
        "critic":["base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "position_commands",
                "joint_pos_rel",
                "joint_vel_rel",
                # "joint_effort",
                "last_action",
                 "camera_image"
                 ],



    }
      
    # policy: RslRlPpoActorCriticCfg = VisionMLPActorCriticCfg()
    policy: RslRlPpoActorCriticCfg = LocoTransformerActorCriticCfg()

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        # num_mini_batches=4,
        num_mini_batches=96, # for transformer
        # learning_rate=1.0e-3,
        learning_rate=1.0e-4, # for transformer
        # schedule="adaptive",
        schedule="constant",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    rnd =  RslRlRndCfg(
        weight = 1.0,
        learning_rate = 1.0e-4,
        predictor_hidden_dims = [128,128],
        target_hidden_dims = [256, 256]



    )
