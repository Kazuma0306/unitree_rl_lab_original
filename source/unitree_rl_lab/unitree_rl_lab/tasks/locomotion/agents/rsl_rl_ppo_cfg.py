# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlRndCfg


@configclass
class BasePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1000 #50000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    # policy = RslRlPpoActorCriticCfg(
    #     init_noise_std=1.0,
    #     actor_hidden_dims=[512, 256, 128],
    #     critic_hidden_dims=[512, 256, 128],
    #     activation="elu",
    # )

    #for paper2
    # ▼▼▼ この policy の定義を以下のように書き換えます ▼▼▼
    policy: dict = {
        "init_noise_std": 1.0,
        "activation": "elu",
        # 'module'キーは、RSL-RLにカスタムネットワーククラスを使用するよう指示します
        "module": {
            # あなたが作成したカスタムクラスへの完全なPythonパス
            "_target_": "unitree_rl_lab.tasks.locomotion.robots.go2.locotransformer.VisionMLPActorCritic",
            
            # --- ここに LocoTransformerActorCriticクラスの__init__メソッドに必要な引数を記述 ---

            "prop_obs_keys": [
                "base_lin_vel",
                "base_ang_vel",
                "projected_gravity",
                "position_commands",
                "joint_pos_rel",
                "joint_vel_rel",
                "last_action"
            ],
            # 視覚として扱う観測のキーを渡す
            "vision_obs_key": "camera_image",
            
            "prop_obs_dim": 61,              # 例：あなたの身体感覚の観測データの次元数に置き換えてください
            "vision_obs_shape": (3, 64, 64), # 観測設定と一致するカメラ画像の形状 (チャンネル数, 高さ, 幅)
            "num_actions": 12,               # 例：あなたのロボットのアクションの次元数（関節数）に置き換えてください
            
            # デフォルトから変更したい場合は、他の__init__引数もここに追加できます
            # "transformer_hidden_dim": 256,
            # "transformer_num_layers": 2,
        },

    }
    # ▲▲▲ 修正はここまで ▲▲▲

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
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
