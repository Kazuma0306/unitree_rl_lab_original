import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import gymnasium as gym
# ▼▼▼ この行を追加 ▼▼▼
from gymnasium.spaces import Box
import numpy as np
from torch.distributions import Normal # <-- Add this line
import torchvision

# このファイルを例えば custom_models.pyとして保存します


from rsl_rl.modules import MLP 





# class VisionMLPActorCritic(ActorCritic):
#     """
#     視覚情報と身体感覚を扱う、シンプルなMLPベースのActor-Criticモデル。
#     - 視覚: ConvNetで特徴量を抽出
#     - 身体感覚: MLPで特徴量を抽出
#     - 融合: 抽出された特徴量を結合(Concatenate)
#     - 本体: 結合された特徴量をMLPで処理
#     """
#     def __init__(
#         self,
#         # from runner 
#         obs: dict,
#         obs_groups: dict,
#         num_actions: int,
#         # --- Cfgから「名前」で渡される引数をその後に書く ---
#         prop_obs_keys: list[str],
#         vision_obs_key: str,
#         prop_encoder_dims: list[int] = [256, 128],
#         vision_encoder_channels: list[int] = [32, 64, 64],
#         shared_mlp_dims: list[int] = [256, 128],
#         init_noise_std: float = 1.0,
#         activation: str = "elu",
#         **kwargs,
#     ):
#         # 1. 親クラスに渡すための「無害化」された観測辞書を作成します。
#         #    オリジナルのobs辞書から、問題となる画像データをキーごと取り除きます。
#         sanitized_obs = {
#             key: value for key, value in obs.items()
#             if key != vision_obs_key
#         }

#         #    obs_groupsから、画像データのキーを取り除きます。
#         sanitized_obs_groups = {
#             group: [key for key in keys if key != vision_obs_key]
#             for group, keys in obs_groups.items()
#         }

#         # 3. 両方とも無害化されたバージョンを親クラスに渡します。
#         super().__init__(
#             obs=sanitized_obs,
#             obs_groups=sanitized_obs_groups,
#             num_actions=num_actions,
#             **kwargs
#         )

#         self.prop_obs_keys = prop_obs_keys
#         self.vision_obs_key = vision_obs_key
        
#         if activation == "elu":
#             self.activation = nn.ELU()
#         elif activation == "relu":
#             self.activation = nn.ReLU()
#         else:
#             raise NotImplementedError

#         prop_obs_dim = sum(obs[key].shape[1] for key in self.prop_obs_keys)
#         vision_obs_shape = obs[vision_obs_key].shape[1:]


#         # --- 1. 身体感覚エンコーダ (MLP) ---
#         prop_layers = []
#         in_dim = prop_obs_dim
#         # for dim in prop_encoder_dims:
#         #     prop_layers.append(nn.Linear(in_dim, dim))
#         #     prop_layers.append(self.activation)
#         #     in_dim = dim
#         # self.proprioception_encoder = nn.Sequential(*prop_layers)

#         self.proprioception_encoder = MLP(
#             input_dim=prop_obs_dim,
#             output_dim=prop_encoder_dims[-1],
#             hidden_dims=prop_encoder_dims[:-1],
#             activation=activation
#         )
        
#         # --- 2. 視覚エンコーダ (ConvNet) ---
#         h, w, c = vision_obs_shape
#         conv_layers = []
#         in_c = c
#         for out_c in vision_encoder_channels:
#             conv_layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1))
#             conv_layers.append(self.activation)
#             in_c = out_c
#         conv_layers.append(nn.Flatten())
#         self.vision_encoder = nn.Sequential(*conv_layers)

#         # ConvNetの出力次元数を自動計算
#         with torch.no_grad():
#             dummy_input = torch.zeros(1, c, h ,w)
#             cnn_output_dim = self.vision_encoder(dummy_input).shape[1]

#         self.prop_norm = nn.LayerNorm(prop_encoder_dims[-1])
#         self.vis_norm = nn.LayerNorm(cnn_output_dim)

#         self.prop_proj = nn.Linear(prop_encoder_dims[-1], 256)
#         self.vis_proj  = nn.Linear(cnn_output_dim, 256)

#         # --- 3. 共有ネットワーク (MLP) ---
#         # 身体感覚と視覚の特徴量を結合したものが入力となる
#         # fused_dim = prop_encoder_dims[-1] + cnn_output_dim
#         fused_dim = 256 * 2
#         shared_layers = []
#         in_dim = fused_dim
#         # for dim in shared_mlp_dims:
#         #     shared_layers.append(nn.Linear(in_dim, dim))
#         #     shared_layers.append(self.activation)
#         #     in_dim = dim
#         # self.shared_body = nn.Sequential(*shared_layers)

#         self.shared_body = MLP(
#             input_dim=fused_dim,
#             output_dim=shared_mlp_dims[-1],
#             hidden_dims=shared_mlp_dims[:-1],
#             activation=activation
#         )

#         self.critic_body = MLP(
#             input_dim=256,
#             output_dim=shared_mlp_dims[-1],
#             hidden_dims=shared_mlp_dims[:-1],
#             activation=activation
#         )

#         # --- 4. 出力ヘッド ---
#         # self.actor = nn.Linear(shared_mlp_dims[-1], num_actions)
#         self.actor = nn.Sequential(
#             nn.Linear(shared_mlp_dims[-1], num_actions)
#             )
#         # self.critic = nn.Linear(share d_mlp_dims[-1], 1)

#         self.critic = nn.Sequential(
#             nn.Linear(shared_mlp_dims[-1], 1)
#             )
        
#         # --- RSL-RLで必要なaction_distributionのためのパラメータ ---
#         # self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
#         # self.distribution = None
#         Normal.set_default_validate_args(False)


#     # def forward(self, observations: dict):

#     #     # 1. 身体感覚データのテンソルをリストに集める
#     #     prop_tensors = [observations[key] for key in self.prop_obs_keys]
#     #     # 2. 集めたテンソルを結合して、一つの長いベクトルにする
#     #     prop_obs_vector = torch.cat(prop_tensors, dim=-1)
#     #     # 3. 視覚データを辞書から取り出す
#     #     vis_obs = observations[self.vision_obs_key]
#     #     max_depth = 10.0  # 環境に合わせて
#     #     if vis_obs.dtype.is_floating_point:
#     #         vis_obs = torch.clamp(vis_obs, 0, max_depth) / max_depth   # [0,1]
#     #         vis_obs = vis_obs * 2.0 - 1.0                              # [-1,1]
#     #     else:
#     #         vis_obs = vis_obs.float() / 255.0
#     #         vis_obs = vis_obs * 2.0 - 1.0

#     #     # 1. 各エンコーダで特徴量を抽出
#     #     # prop_features = self.proprioception_encoder(prop_obs_vector)
#     #     # vision_features = self.vision_encoder(vis_obs)

#     #     prop_features = self.prop_proj(self.prop_norm(self.proprioception_encoder(prop_obs_vector)))
#     #     vision_features = self.vis_proj(self.vis_norm(self.vision_encoder(vis_obs)))


        
#     #     # 正規化された特徴量を結合します
#     #     # fused_features = torch.cat([normalized_prop_features, normalized_vision_features], dim=-1)
#     #     # ★★★ 修正はここまで ★★★
        
#     #     # 2. 特徴量を結合 (Concatenate)
#     #     fused_features = torch.cat([prop_features, vision_features], dim=-1)

#     #     # 3. 結合された特徴量を共有ネットワークに通す
#     #     shared_output = self.shared_body(fused_features)

#     #     # 4. ActorとCriticの出力を計算
#     #     actions_mean = self.actor_head(shared_output)
#     #     critic_value = self.critic_head(shared_output)

#     #     return actions_mean, critic_value#.squeeze(-1)

#     # # ----------------------------------------
#     # # --- RSL-RLのRunnerが期待するメソッドをオーバーライド ---
#     # # ----------------------------------------
#     # def act(self, observations: dict, **kwargs):
#     #     actions_mean, _ = self.forward(observations)
#     #     std = torch.exp(self.log_std)
#     #     self.distribution = Normal(actions_mean, std)
#     #     return self.distribution.sample()

#     # def act_inference(self, observations: dict, **kwargs):
#     #     actions_mean, _ = self.forward(observations)
#     #     return actions_mean

#     # def evaluate(self, observations: dict, **kwargs):
#     #     _, critic_value = self.forward(observations)
#     #     return critic_value
  

#     def get_actor_obs(self, obs):
#         """Actor 用の fused features を返す"""
#         prop_tensors = [obs[k] for k in self.prop_obs_keys]
#         prop_vec = torch.cat(prop_tensors, dim=-1)
#         prop_feat = self.proprioception_encoder(prop_vec)

#         vis_obs = obs[self.vision_obs_key] #B, H, W, C

#         # max_depth = 10.0
#         # if vis_obs.dtype.is_floating_point:
#         #     vis_obs = torch.clamp(vis_obs, 0, max_depth) / max_depth
#         #     vis_obs = vis_obs * 2.0 - 1.0
#         # else:
#         #     vis_obs = vis_obs.float() / 255.0
#         #     vis_obs = vis_obs * 2.0 - 1.0

#         # clipping_rangeから最小・最大距離を設定
#         clipping_min = 0.1
#         clipping_max = 3.0
#         # env.step()から得られた元の深度データテンソル
#         # shape: (num_envs, height, width, 1)
#         # 以前の課題：範囲外を示す0を、まず最大距離として扱う
#         vis_obs[vis_obs == 0] = clipping_max

#         # [0, 1]の範囲に正規化
#         normalized_obs = (vis_obs - clipping_min) / (clipping_max - clipping_min)

#         # 念のため、計算誤差などで範囲外になった値を0か1に収める
#         vis_obs = torch.clamp(normalized_obs, 0.0, 1.0)

#         # vis_obs = torch.zeros_like(vis_obs)
#         # vis_obs = vis_obs.float()
#         # vis_obs = vis_obs * 0.0


#         vis_obs = vis_obs.permute(0, 3, 1, 2) #B, C, H, W
#         # print("vision shape:", vis_obs.shape)

#         vis_feat = self.vision_encoder(vis_obs)

#         prop_features = self.prop_proj(self.prop_norm(prop_feat))
#         vision_features = self.vis_proj(self.vis_norm(vis_feat))


#         fused = torch.cat([prop_features, vision_features], dim=-1)

#         return self.shared_body(fused)

#     def get_critic_obs(self, obs):
#         # prop_tensors = [obs[k] for k in self.prop_obs_keys] #use only prop features
#         # prop_vec = torch.cat(prop_tensors, dim=-1)
#         # prop_feat = self.proprioception_encoder(prop_vec)

#         # prop_features = self.prop_proj(self.prop_norm(prop_feat))

#         # return self.critic_body(prop_features)

#         return self.get_actor_obs(obs)




import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class VisionMLPActorCritic(ActorCritic):
    """
    Depth(メートル) + Normals([-1,1]) を単一 Conv 幹で concat して扱う Actor-Critic。
    入力想定:
      - obs[vision_depth_key]   : (B, H, W, Sd)   ※Sd=スタック数(1可)
      - obs[vision_normals_key] : (B, H, W, 3*Sn) ※Sn=スタック数(1可)
    """
    def __init__(
        self,
        # from runner
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        # --- 追加/変更点 ---
        prop_obs_keys: list[str],
        vision_depth_key: str,
        vision_normals_key: str,
        num_depth_stack: int = 1,
        num_normals_stack: int = 1,
        near: float = 0.1,
        far: float = 3.0,
        # 既存パラメタ
        prop_encoder_dims: list[int] = [256, 128],
        vision_encoder_channels: list[int] = [32, 64, 64],
        shared_mlp_dims: list[int] = [256, 128],
        init_noise_std: float = 1.0,
        activation: str = "elu",
        **kwargs,
    ):
        # 画像キーを親へ渡さない
        sanitized_obs = {
            k: v for k, v in obs.items()
            if k not in (vision_depth_key, vision_normals_key)
        }
        sanitized_obs_groups = {
            g: [k for k in ks if k not in (vision_depth_key, vision_normals_key)]
            for g, ks in obs_groups.items()
        }
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_obs_groups, num_actions=num_actions, **kwargs)

        self.prop_obs_keys = prop_obs_keys
        self.vision_depth_key = vision_depth_key
        self.vision_normals_key = vision_normals_key
        self.num_depth_stack = num_depth_stack
        self.num_normals_stack = num_normals_stack
        self.near = near
        self.far = far

        # 活性化
        def _act():
            if activation == "elu":
                return nn.ELU()
            elif activation == "relu":
                return nn.ReLU()
            raise NotImplementedError
        self._act = _act

        # --- Proprio encoder ---
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        self.proprioception_encoder = MLP(
            input_dim=prop_obs_dim,
            output_dim=prop_encoder_dims[-1],
            hidden_dims=prop_encoder_dims[:-1],
            activation=activation
        )

        # --- Conv 幹（入力チャンネル数を Depth/Normals から決定） ---
        # 形状はチャンネル最後 (B,H,W,C*) を想定
        depth_c = obs[vision_depth_key].shape[-1]              # = Sd
        normals_c = obs[vision_normals_key].shape[-1]          # = 3*Sn
        assert depth_c == num_depth_stack, \
            f"depth stack mismatch: tensor C={depth_c}, num_depth_stack={num_depth_stack}"
        assert normals_c % 3 == 0 and normals_c // 3 == num_normals_stack, \
            f"normals stack mismatch: tensor C={normals_c}, num_normals_stack={num_normals_stack}"

        in_c = 2 * num_depth_stack + 3 * num_normals_stack     # invD+valid + normals
        conv_layers, c_in = [], in_c
        for c_out in vision_encoder_channels:
            conv_layers += [nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1), _act()]
            c_in = c_out
        conv_layers.append(nn.Flatten())
        self.vision_encoder = nn.Sequential(*conv_layers)

        # Conv 出力次元を推定
        with torch.no_grad():
            # ダミー入力サイズは元観測の H,W を使う（B=1）
            H, W = obs[vision_depth_key].shape[1], obs[vision_depth_key].shape[2]
            dummy = torch.zeros(1, in_c, H, W)
            cnn_output_dim = self.vision_encoder(dummy).shape[1]

        # 正規化と射影
        self.prop_norm = nn.LayerNorm(prop_encoder_dims[-1])
        self.vis_norm = nn.LayerNorm(cnn_output_dim)
        self.prop_proj = nn.Linear(prop_encoder_dims[-1], 256)
        self.vis_proj  = nn.Linear(cnn_output_dim, 256)

        # --- 共有/価値ネット ---
        fused_dim = 256 * 2
        self.shared_body = MLP(
            input_dim=fused_dim,
            output_dim=shared_mlp_dims[-1],
            hidden_dims=shared_mlp_dims[:-1],
            activation=activation
        )
        self.critic_body = MLP(
            input_dim=256,  # critic を Proprio 専用にしたい場合に差し替え可能
            output_dim=shared_mlp_dims[-1],
            hidden_dims=shared_mlp_dims[:-1],
            activation=activation
        )

        # --- 出力ヘッド ---
        self.actor  = nn.Sequential(nn.Linear(shared_mlp_dims[-1], num_actions))
        self.critic = nn.Sequential(nn.Linear(shared_mlp_dims[-1], 1))

        # --- RSL-RL 向け：アクション分布パラメータ ---
        self.log_std = nn.Parameter(torch.ones(num_actions) * math.log(init_noise_std))
        Normal.set_default_validate_args(False)

    # ---------- 前処理ユーティリティ ----------
    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        # (B,H,W,C) → (B,C,H,W)
        return x.permute(0, 3, 1, 2).contiguous()

    def _prep_depth(self, depth_bhwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        depth_bhwc: (B,H,W,Sd) [meters]
        returns:
            depth_in: (B, 2*Sd, H, W)  [invD, valid] をフレーム毎に並べたチャンネル
            valid   : (B, Sd, H, W)    bool
        """
        B, H, W, Sd = depth_bhwc.shape
        D = self._to_bchw(depth_bhwc).reshape(B, Sd, H, W)  # (B,Sd,H,W)
        valid = torch.isfinite(D) & (D > self.near) & (D < self.far)

        Dc = torch.clamp(D, min=self.near, max=self.far)
        invD = (1.0 / Dc - 1.0 / self.far) / (1.0 / self.near - 1.0 / self.far)  # [0,1]
        invD = torch.where(valid, invD, torch.zeros_like(invD))

        depth_in = torch.cat([invD, valid.float()], dim=1)  # (B, 2*Sd, H, W)
        return depth_in, valid

    def _prep_normals(self, normals_bhwc: torch.Tensor, depth_valid: torch.Tensor) -> torch.Tensor:
        """
        normals_bhwc: (B,H,W,3*Sn) in [-1,1]
        depth_valid : (B,Sd,H,W) bool
        returns:
            normals_in: (B, 3*Sn, H, W), 欠測は 0
        """
        B, H, W, Cn = normals_bhwc.shape
        Sn = Cn // 3
        N = self._to_bchw(normals_bhwc)  # (B, 3*Sn, H, W)
        N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)

        # 深度の valid を可能なら同フレーム数で適用
        if depth_valid.shape[1] == Sn:
            mask = depth_valid.unsqueeze(2).expand(B, Sn, 3, H, W).reshape(B, 3*Sn, H, W)
            N = N * mask.float()
        elif depth_valid.shape[1] == 1:
            mask = depth_valid.expand(B, Sn, H, W).unsqueeze(2).expand(B, Sn, 3, H, W).reshape(B, 3*Sn, H, W)
            N = N * mask.float()
        # それ以外（枚数不一致）は無遮蔽で通す
        return N

    # ---------- 前向き ----------
    def get_actor_obs(self, obs):
        # Proprio
        prop_vec = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        prop_feat = self.proprioception_encoder(prop_vec)

        # Vision: Depth + Normals → concat
        depth_bhwc   = obs[self.vision_depth_key]    # (B,H,W,Sd)
        normals_bhwc = obs[self.vision_normals_key]  # (B,H,W,3*Sn)

        depth_in, valid = self._prep_depth(depth_bhwc)               # (B,2*Sd,H,W)
        normals_in = self._prep_normals(normals_bhwc, valid)         # (B,3*Sn,H,W)
        vis_in = torch.cat([depth_in, normals_in], dim=1)            # (B,2*Sd+3*Sn,H,W)

        vis_feat = self.vision_encoder(vis_in)

        # 射影＆融合
        prop_features   = self.prop_proj(self.prop_norm(prop_feat))
        vision_features = self.vis_proj(self.vis_norm(vis_feat))
        fused = torch.cat([prop_features, vision_features], dim=-1)

        return self.shared_body(fused)

    def get_critic_obs(self, obs):
        # 今回は actor と同じ特徴を使用（必要なら proprio-only へ変更可）
        return self.get_actor_obs(obs)








# class LocoTransformerActorCritic(ActorCritic):
#     """
#     LocoTransformerの論文に基づいたActor-Criticモデル。
#     身体感覚（Proprioception）と視覚（Vision）をTransformerで融合する。
#     """

#     def __init__(
#         self,
#         # from runner 
#         obs: dict,
#         obs_groups: dict,
#         num_actions: int,
#         # --- Cfgから「名前」で渡される引数をその後に書く ---
#         prop_obs_keys: list[str],
#         vision_obs_key: str,
#         prop_encoder_dims: list[int] = [256, 256],
#         vision_channels: int = 128,
#         transformer_hidden_dim: int = 256,
#         transformer_n_heads: int = 4,
#         transformer_num_layers: int = 2,
#         projection_head_dims: list = [256, 256],
#         num_image_stack: int = 4,
#         init_noise_std: float = 1.0,
#         activation: str = "elu",
#         **kwargs,
#     ):
#         # 1. 親クラスに渡すための「無害化」された観測辞書を作成します。
#         #    オリジナルのobs辞書から、問題となる画像データをキーごと取り除きます。
#         sanitized_obs = {
#             key: value for key, value in obs.items()
#             if key != vision_obs_key
#         }
#         #    obs_groupsから、画像データのキーを取り除きます。
#         sanitized_obs_groups = {
#             group: [key for key in keys if key != vision_obs_key]
#             for group, keys in obs_groups.items()
#         }
#         # 3. 両方とも無害化されたバージョンを親クラスに渡します。
#         super().__init__(
#             obs=sanitized_obs,
#             obs_groups=sanitized_obs_groups,
#             num_actions=num_actions,
#             **kwargs
#         )

#         self.prop_obs_keys = prop_obs_keys
#         self.vision_obs_key = vision_obs_key

#         if activation == "elu":
#             self.activation = nn.ELU()
#         elif activation == "relu":
#             self.activation = nn.ReLU()
#         else:
#             raise NotImplementedError

#         prop_obs_dim = sum(obs[key].shape[1] for key in self.prop_obs_keys)
#         vision_obs_shape = obs[vision_obs_key].shape[1:] #flattend

#         print(prop_obs_dim)
#         print(vision_obs_shape[0]) 


#         # 1. 身体感覚エンコーダ (MLP)
#         # 論文: a 2-layer MLP with hidden dimensions (256, 256)
#         prop_layers = []
#         in_dim = prop_obs_dim
#         for dim in prop_encoder_dims:
#             prop_layers.append(nn.Linear(in_dim, dim))
#             prop_layers.append(nn.ELU())
#             in_dim = dim
#         self.proprioception_encoder = nn.Sequential(*prop_layers)
        
#         input_channels = num_image_stack
#         # 2. 視覚エンコーダ (ConvNet)
#         # 論文: 64x64の画像を 4x4 x 128チャンネルの特徴量マップに変換

#         self.vision_encoder = nn.Sequential(
#             # Input: (B, C, 64, 64)
#             nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), # -> (B, 32, 32, 32)
#             self.activation,
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (B, 64, 16, 16)
#             self.activation,
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (B, 128, 8, 8)
#             self.activation,
#             nn.Conv2d(128, vision_channels, kernel_size=3, stride=2, padding=1), # -> (B, 128, 4, 4)
#             self.activation,
#         )
        
#         # 3. Transformerのための準備
#         # 各特徴量をTransformerの次元に射影（Project）する層
#         self.prop_proj = nn.Linear(prop_encoder_dims[-1], transformer_hidden_dim)
#         self.vis_proj = nn.Linear(vision_channels, transformer_hidden_dim)
        
#         # Transformer Encoder層を定義
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=transformer_hidden_dim,
#             nhead=transformer_n_heads,
#             dim_feedforward=512,
#             activation="gelu",
#             batch_first=True # RSL-RLは (batch, seq, dim) の形式を扱うため
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
#         # 論文の 4x4=16個の視覚トークンに対応する位置エンコーディング
#         self.visual_pos_embedding = nn.Parameter(torch.randn(1, 16, transformer_hidden_dim))
#         # 身体感覚トークン用の位置エンコーディング
#         self.prop_pos_embedding = nn.Parameter(torch.randn(1, 1, transformer_hidden_dim))

#         # 4. 出力ヘッド
#         # 論文: 融合された特徴量を処理する2層のMLP
#         fused_feature_dim = transformer_hidden_dim * 2 # 身体感覚と視覚の2種類
#         proj_layers = []
#         in_dim = fused_feature_dim
#         for dim in projection_head_dims:
#             proj_layers.append(nn.Linear(in_dim, dim))
#             proj_layers.append(nn.ELU())
#             in_dim = dim
#         self.projection_head = nn.Sequential(*proj_layers)

#         # Actor (行動決定) と Critic (価値評価) の最終層
#         # self.actor = nn.Linear(projection_head_dims[-1], num_actions)
#         # self.critic = nn.Linear(projection_head_dims[-1], 1)

#         self.actor = nn.Sequential(
#             nn.Linear(projection_head_dims[-1], num_actions)
#             )
#         self.critic = nn.Sequential(
#             nn.Linear(projection_head_dims[-1], 1)
#             )


        
#         # RSL-RLで必要なaction_distributionのためのパラメータ
#         # self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
#         # self.distribution = None
#         Normal.set_default_validate_args(False)


#     # def forward(self, observations: dict):
#     #     prop_tensors = [observations[key] for key in self.prop_obs_keys]
#     #     # 2. 集めたテンソルを結合して、一つの長いベクトルにする
#     #     prop_obs_vector = torch.cat(prop_tensors, dim=-1)
#     #     # 3. 視覚データを辞書から取り出す
#     #     vis_obs_flat = observations[self.vision_obs_key] #flattend observations

#     #     batch_size = vis_obs_flat.shape[0]
#     #     # (Batch, 16384) -> (Batch, 4, 64, 64)
#     #     # 4はスタックされたフレーム数、64, 64は画像の高さと幅
#     #     vis_obs_unflattened = vis_obs_flat.view(batch_size, 4, 64, 64)


#     #     max_depth = 10.0  # 環境に合わせて
#     #     if vis_obs_unflattened.dtype.is_floating_point:
#     #         vis_obs = torch.clamp(vis_obs_unflattened, 0, max_depth) / max_depth   # [0,1]
#     #         vis_obs = vis_obs * 2.0 - 1.0                              # [-1,1]
#     #     else:
#     #         vis_obs = vis_obs_unflattened.float() / 255.0
#     #         vis_obs = vis_obs * 2.0 - 1.0

#     #     # --- 1. 各エンコーダで特徴量を抽出 ---
#     #     prop_features = self.proprioception_encoder(prop_obs_vector)
#     #     vis_features_map = self.vision_encoder(vis_obs)

#     #     # --- 2. トークン化とTransformerによる融合 ---
#     #     # 身体感覚トークン
#     #     prop_token = self.prop_proj(prop_features).unsqueeze(1) # (batch, 1, dim)
        
#     #     # 視覚トークン
#     #     vis_tokens = vis_features_map.flatten(2).permute(0, 2, 1) # (batch, 16, channels)
#     #     vis_tokens = self.vis_proj(vis_tokens)
        
#     #     # 位置エンコーディングを追加
#     #     prop_token += self.prop_pos_embedding
#     #     vis_tokens += self.visual_pos_embedding
        
#     #     # 全トークンを結合してTransformerに入力
#     #     all_tokens = torch.cat([prop_token, vis_tokens], dim=1)
#     #     fused_tokens = self.transformer_encoder(all_tokens)
        
#     #     # --- 3. 融合された特徴量の処理と出力 ---
#     #     # 論文の記述通り、各モダリティの情報を平均化して集約
#     #     fused_prop = fused_tokens[:, 0, :]
#     #     fused_vis = fused_tokens[:, 1:, :].mean(dim=1)
        
#     #     # 2つの特徴量を結合
#     #     fused_features = torch.cat([fused_prop, fused_vis], dim=1)
        
#     #     # Projection Headに通す
#     #     final_features = self.projection_head(fused_features)

#     #     # ActorとCriticの出力を計算
#     #     actions_mean = self.actor(final_features)
#     #     critic_value = self.critic(final_features)

#     #     return actions_mean, critic_value


#     def get_actor_obs(self, obs):
#         """Actor 用の fused features を返す"""
#         prop_tensors = [obs[k] for k in self.prop_obs_keys]
#         prop_vec = torch.cat(prop_tensors, dim=-1)
#         prop_feat = self.proprioception_encoder(prop_vec)

#         vis_obs_flat = obs[self.vision_obs_key] #flattend observations

#         batch_size = vis_obs_flat.shape[0]
#         # (Batch, 16384) -> (Batch, 4, 64, 64)
#         # 4はスタックされたフレーム数、64, 64は画像の高さと幅
#         vis_obs_unflattened = vis_obs_flat.view(batch_size, 4, 64, 64) #B,C,H,W

#         clipping_min = 0.1
#         clipping_max = 3.0
#         # env.step()から得られた元の深度データテンソル
#         # shape: (num_envs, height, width, 1)
#         # 以前の課題：範囲外を示す0を、まず最大距離として扱う
#         vis_obs_unflattened[vis_obs_unflattened == 0] = clipping_max

#         # [0, 1]の範囲に正規化
#         normalized_obs = (vis_obs_unflattened - clipping_min) / (clipping_max - clipping_min)

#         # 念のため、計算誤差などで範囲外になった値を0か1に収める
#         vis_obs = torch.clamp(normalized_obs, 0.0, 1.0)

#         # print(
#         #     "Depth input stats:",
#         #     vis_obs_unflattened.min().item(),
#         #     vis_obs_unflattened.max().item()
#         # )


        
#         # print(
#         #     "Normalized depth stats:",
#         #     vis_obs.min().item(),
#         #     vis_obs.max().item()
#         # )

        

#         vis_feat = self.vision_encoder(vis_obs)
#         # print("vision feature shape:", vis_feat.shape)


#         # 身体感覚トークン
#         prop_token = self.prop_proj(prop_feat).unsqueeze(1) # (batch, 1, dim)
#         # 視覚トークン
#         vis_tokens = vis_feat.flatten(2).permute(0, 2, 1) # (batch, 16, channels)
#         # print("vis_tokens shape:", vis_tokens.shape)  
#         vis_tokens = self.vis_proj(vis_tokens)
        
#         # 位置エンコーディングを追加
#         prop_token += self.prop_pos_embedding
#         vis_tokens += self.visual_pos_embedding
        
#         # 全トークンを結合してTransformerに入力
#         all_tokens = torch.cat([prop_token, vis_tokens], dim=1)
#         # print("all tokens shape:", all_tokens.shape)  
#         fused_tokens = self.transformer_encoder(all_tokens)
        
#         # --- 3. 融合された特徴量の処理と出力 ---
#         # 論文の記述通り、各モダリティの情報を平均化して集約
#         fused_prop = fused_tokens[:, 0, :]
#         fused_vis = fused_tokens[:, 1:, :].mean(dim=1)
        
#         # 2つの特徴量を結合
#         fused_features = torch.cat([fused_prop, fused_vis], dim=1)

#         # Projection Headに通す
#         final_features = self.projection_head(fused_features)

#         return final_features


#     def get_critic_obs(self, obs):
#         """Critic も同じ fused features を使う（場合によっては別設計も可能）"""
#         return self.get_actor_obs(obs)




class LocoTransformerActorCritic(ActorCritic):
    def __init__(
        self,
        # from runner 
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        # --- Cfgから渡す ---
        prop_obs_keys: list[str],
        vision_depth_key: str,                 # 追加: Depthのキー
        vision_normals_key: str,               # 追加: Normalsのキー
        prop_encoder_dims: list[int] = [256, 256],
        vision_channels: int = 128,            # 最終的にDepth+Normalsを合わせた出力ch
        transformer_hidden_dim: int = 256,
        transformer_n_heads: int = 4,
        transformer_num_layers: int = 2,
        projection_head_dims: list = [256, 256],
        num_depth_stack: int = 4,              # 追加: Depthのフレームスタック数
        num_normals_stack: int = 4,            # 追加: Normalsのフレームスタック数（未スタックなら1）
        near: float = 0.1,                     # 追加: Depth近クリップ
        far: float = 3.0,                      # 追加: Depth遠クリップ
        image_size: tuple = (64, 64),          # 追加: (H,W)
        init_noise_std: float = 1.0,
        activation: str = "elu",
        **kwargs,
    ):
        # 画像キーをobsから除外（親クラスへ渡さない）
        sanitized_obs = {k: v for k, v in obs.items() if k not in [vision_depth_key, vision_normals_key]}
        sanitized_obs_groups = {
            g: [k for k in ks if k not in [vision_depth_key, vision_normals_key]]
            for g, ks in obs_groups.items()
        }
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_obs_groups, num_actions=num_actions, **kwargs)

        # 保存
        self.prop_obs_keys = prop_obs_keys
        self.vision_depth_key = vision_depth_key
        self.vision_normals_key = vision_normals_key
        self.num_depth_stack = num_depth_stack
        self.num_normals_stack = num_normals_stack
        self.near = near
        self.far = far
        self.H, self.W = image_size

        # 活性化
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

        # 1) Proprio MLP
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        prop_layers = []
        in_dim = prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers += [nn.Linear(in_dim, dim), nn.ELU()]
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)

        # 2) Vision encoders (two-tower)
        # Depth塔: invD(1) + valid(1) = 2ch/フレーム → 2 * num_depth_stack
        depth_in_ch = 2 * num_depth_stack
        # Normals塔: (Nx,Ny,Nz)=3ch/フレーム → 3 * num_normals_stack
        normals_in_ch = 3 * num_normals_stack

        depth_out = vision_channels // 2
        normals_out = vision_channels - depth_out  # 合計 vision_channels

        def make_stem(cin, c1=32, c2=64, c3=96, cout=128):
            return nn.Sequential(
                nn.Conv2d(cin, c1, kernel_size=3, stride=2, padding=1), self.activation,  # H/2
                nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1), self.activation,  # H/4
                nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1), self.activation,  # H/8
                nn.Conv2d(c3, cout, kernel_size=3, stride=2, padding=1), self.activation, # H/16 (4x4 if 64x64)
            )

        self.depth_encoder = make_stem(depth_in_ch, cout=depth_out)
        self.normals_encoder = make_stem(normals_in_ch, cout=normals_out)

        # 3) Transformer周り
        self.prop_proj = nn.Linear(prop_encoder_dims[-1], transformer_hidden_dim)
        self.vis_proj = nn.Linear(vision_channels, transformer_hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim, nhead=transformer_n_heads,
            dim_feedforward=512, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        # 4x4=16トークン想定の位置埋め込み
        self.visual_pos_embedding = nn.Parameter(torch.randn(1, 16, transformer_hidden_dim))
        self.prop_pos_embedding   = nn.Parameter(torch.randn(1, 1, transformer_hidden_dim))

        # 4) 出力ヘッド
        fused_feature_dim = transformer_hidden_dim * 2
        proj_layers = []
        in_dim = fused_feature_dim
        for dim in projection_head_dims:
            proj_layers += [nn.Linear(in_dim, dim), nn.ELU()]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)
        self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
        self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

    # ---- 前処理ユーティリティ ----
    def _prep_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        """
        depth_flat: (B, num_depth_stack*H*W) or (B, H*W) if stack=1
        return: (B, 2*num_depth_stack, H, W)  # [invD, valid] をフレーム毎に並べる
        """
        B = depth_flat.shape[0]
        D = depth_flat.view(B, self.num_depth_stack, self.H, self.W)  # meters
        valid = torch.isfinite(D) & (D > self.near) & (D < self.far)  # (B,S,H,W)

        Dc = torch.clamp(D, min=self.near, max=self.far)
        invD = (1.0 / Dc - 1.0 / self.far) / (1.0 / self.near - 1.0 / self.far)  # [0,1]
        invD[~valid] = 0.0

        depth_input = torch.cat([invD, valid.float()], dim=1)  # (B, 2*S, H, W)
        return depth_input

    def _prep_normals(self, normals_flat: torch.Tensor, depth_valid_like: torch.Tensor) -> torch.Tensor:
        """
        normals_flat: (B, 3*num_normals_stack*H*W) or (B, 3*H*W)
        depth_valid_like: (B, num_depth_stack, H, W) のbool（DepthのvalidをNormalsにも掛けたい場合）
        return: (B, 3*num_normals_stack, H, W)
        """
        B = normals_flat.shape[0]
        N = normals_flat.view(B, 3*self.num_normals_stack, self.H, self.W)
        N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)

        # Depthと同じvalidを適用（Normalsの欠測定義が曖昧なことが多いため）
        if self.num_normals_stack == self.num_depth_stack:
            mask = depth_valid_like.unsqueeze(2).expand(B, self.num_normals_stack, 3, self.H, self.W)
            mask = mask.reshape(B, 3*self.num_normals_stack, self.H, self.W)
            N = N * mask.float()
        return N

    # ---- 観測の前処理と前向き計算 ----
    def get_actor_obs(self, obs):
        # --- Proprio ---
        prop_vec = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        prop_feat = self.proprioception_encoder(prop_vec)              # (B, P)
        prop_token = self.prop_proj(prop_feat).unsqueeze(1)            # (B,1,D)
        prop_token = prop_token + self.prop_pos_embedding

        # --- Vision（Depth + Normals）---
        depth_flat   = obs[self.vision_depth_key]                      # (B, S*H*W)
        normals_flat = obs[self.vision_normals_key]                    # (B, 3*S'*H*W)

        # Depth: invD + valid
        B = depth_flat.shape[0]
        D = depth_flat.view(B, self.num_depth_stack, self.H, self.W)
        valid = torch.isfinite(D) & (D > self.near) & (D < self.far)   # (B,S,H,W)

        depth_in = self._prep_depth(depth_flat)                        # (B, 2S, H, W)
        # Normals: [-1,1], 無効は0埋め
        normals_in = self._prep_normals(normals_flat, valid)           # (B, 3S', H, W)

        # 二塔エンコード → (B, C_d, 4,4), (B, C_n, 4,4)
        fd = self.depth_encoder(depth_in)
        fn = self.normals_encoder(normals_in)
        fvis = torch.cat([fd, fn], dim=1)                              # (B, vision_channels, 4, 4) concat along channel dimension

        # 視覚トークン化（16トークン）
        vis_tokens = fvis.flatten(2).permute(0, 2, 1)                  # (B, 16, C)
        vis_tokens = self.vis_proj(vis_tokens)                         # (B, 16, D)
        vis_tokens = vis_tokens + self.visual_pos_embedding

        # Transformer
        all_tokens = torch.cat([prop_token, vis_tokens], dim=1)        # (B, 17, D)
        fused_tokens = self.transformer_encoder(all_tokens)

        fused_prop = fused_tokens[:, 0, :]
        fused_vis  = fused_tokens[:, 1:, :].mean(dim=1)
        fused_features = torch.cat([fused_prop, fused_vis], dim=1)
        final_features = self.projection_head(fused_features)
        return final_features

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)









class LocoTransformerFinetune(ActorCritic):
    """
    LocoTransformerActorCriticモデルに、ファインチューニングのための
    重みの凍結（Freeze）および段階的な解凍（Thaw）機能を実装した拡張版。
    """
    def __init__(
        self,
        # ... (元の__init__の引数はすべて同じ) ...
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        prop_obs_keys: list[str],
        vision_depth_key: str,
        vision_normals_key: str,
        prop_encoder_dims: list[int] = [256, 256],
        vision_channels: int = 128,
        transformer_hidden_dim: int = 256,
        transformer_n_heads: int = 4,
        transformer_num_layers: int = 2,
        projection_head_dims: list = [256, 256],
        num_depth_stack: int = 4,
        num_normals_stack: int = 4,
        near: float = 0.1,
        far: float = 3.0,
        image_size: tuple = (64, 64),
        init_noise_std: float = 1.0,
        activation: str = "elu",
        **kwargs,
    ):
        # 親クラスの初期化 (ここは元のコードと同じ)
        sanitized_obs = {k: v for k, v in obs.items() if k not in [vision_depth_key, vision_normals_key]}
        sanitized_obs_groups = {
            g: [k for k in ks if k not in [vision_depth_key, vision_normals_key]]
            for g, ks in obs_groups.items()
        }
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_obs_groups, num_actions=num_actions, **kwargs)

        # モデルの各パーツの定義 (ここは元のコードと同じ)
        self.prop_obs_keys = prop_obs_keys
        self.vision_depth_key = vision_depth_key
        self.vision_normals_key = vision_normals_key
        self.num_depth_stack = num_depth_stack
        self.num_normals_stack = num_normals_stack
        self.near = near
        self.far = far
        self.H, self.W = image_size

        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

        # 1) Proprio MLP
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        prop_layers = []
        in_dim = prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers += [nn.Linear(in_dim, dim), nn.ELU()]
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)

        # 2) Vision encoders (two-tower)
        depth_in_ch = 2 * num_depth_stack
        normals_in_ch = 3 * num_normals_stack
        depth_out = vision_channels // 2
        normals_out = vision_channels - depth_out

        def make_stem(cin, c1=32, c2=64, c3=96, cout=128):
            return nn.Sequential(
                nn.Conv2d(cin, c1, kernel_size=3, stride=2, padding=1), self.activation,
                nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1), self.activation,
                nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1), self.activation,
                nn.Conv2d(c3, cout, kernel_size=3, stride=2, padding=1), self.activation,
            )
        self.depth_encoder = make_stem(depth_in_ch, cout=depth_out)
        self.normals_encoder = make_stem(normals_in_ch, cout=normals_out)

        # 3) Transformer周り
        self.prop_proj = nn.Linear(prop_encoder_dims[-1], transformer_hidden_dim)
        self.vis_proj = nn.Linear(vision_channels, transformer_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim, nhead=transformer_n_heads,
            dim_feedforward=512, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        self.visual_pos_embedding = nn.Parameter(torch.randn(1, 16, transformer_hidden_dim))
        self.prop_pos_embedding   = nn.Parameter(torch.randn(1, 1, transformer_hidden_dim))

        # 4) 出力ヘッド
        fused_feature_dim = transformer_hidden_dim * 2
        proj_layers = []
        in_dim = fused_feature_dim
        for dim in projection_head_dims:
            proj_layers += [nn.Linear(in_dim, dim), nn.ELU()]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)
        self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
        self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

      
        # ★★★ ここからが追加部分 ★★★
        # 学習の便宜上、モデルの主要なパーツをグループ化しておく
        # self.vision_modules = nn.ModuleList([self.depth_encoder, self.normals_encoder, self.vis_proj])
        # self.proprio_modules = nn.ModuleList([self.proprioception_encoder, self.prop_proj])
        # self.transformer_module = self.transformer_encoder
        # self.output_modules = nn.ModuleList([self.projection_head, self.actor, self.critic])

        self._vision_modules   = [self.depth_encoder, self.normals_encoder, self.vis_proj]
        self._proprio_modules  = [self.proprioception_encoder, self.prop_proj]
        # self._transformer_ref  = self.transformer_encoder   # ただの参照（別名登録しない）
        self._output_modules   = [self.projection_head, self.actor, self.critic]
        
        # 初期状態では全てのパラメータを学習対象にする
        self.set_finetuning_mode('full')




    def set_finetuning_mode(self, mode: str, unfreeze_transformer_layers: int = 0, verbose: bool = False):
        """
        ファインチューニング用の勾配制御。
        デフォルトで全て解凍し、必要な部分だけ凍結する方式。
        - mode: 'full' | 'freeze_all_but_heads' | 'freeze_transformer' | 'freeze_vision'
        - unfreeze_transformer_layers: 'freeze_transformer'時に末尾N層だけ解凍（0なら完全凍結）
        """

        # 0) まず全パラメータを学習ON（既存状態をリセット）
        # for p in self.parameters():
        #     p.requires_grad = True

        # ヘルパ：Module / ModuleList / list のどれでも param を回せるように
        def _iter_params(obj):
            import torch.nn as nn
            if obj is None:
                return []
            if isinstance(obj, (list, tuple)):
                for m in obj:
                    for p in _iter_params(m):
                        yield p
            # elif isinstance(obj, nn.Module):
            #     for p in obj.parameters():
            #         yield p
            else:
                return []  # nn.Parameter などの個別は別で扱う

        def _set_params(obj, requires_grad: bool):
            for p in _iter_params(obj):
                p.requires_grad = requires_grad

        # 1) モードごとの指定
        if mode == 'full':
            # 何も凍結しない（ただし下のstd/posの扱いは維持）
            pass

        elif mode == 'freeze_all_but_heads':
            # Vision (CNN+vis_proj), Transformer, PosEmb を凍結
            _set_params([self.depth_encoder, self.normals_encoder, self.vis_proj], False)
            _set_params(self.transformer_encoder, False)
            if hasattr(self, "visual_pos_embedding"):
                self.visual_pos_embedding.requires_grad = False
            if hasattr(self, "prop_pos_embedding"):
                self.prop_pos_embedding.requires_grad = False
            # 残す：proprio + head（既定でONに戻してあるので何もしない）

        elif mode == 'freeze_transformer':
            # Transformer 凍結（末尾N層だけ解凍可）
            _set_params(self.transformer_encoder, False)
            # 位置埋め込みは基本Transformerと一蓮托生
            if hasattr(self, "visual_pos_embedding"):
                self.visual_pos_embedding.requires_grad = False
            if hasattr(self, "prop_pos_embedding"):
                self.prop_pos_embedding.requires_grad = False

            # 末尾N層だけ解凍
            if unfreeze_transformer_layers > 0:
                L = len(self.transformer_encoder.layers)
                n = max(0, min(unfreeze_transformer_layers, L))
                for i in range(L - n, L):
                    _set_params(self.transformer_encoder.layers[i], True)
                # 末尾を解凍するなら視覚のpos埋め込みも学習させた方が安定
                if hasattr(self, "visual_pos_embedding"):
                    self.visual_pos_embedding.requires_grad = True

        elif mode == 'freeze_vision':
            # Vision 凍結（CNN幹 + vis_proj）
            _set_params([self.depth_encoder, self.normals_encoder, self.vis_proj], False)
            # 残す：Transformer, Heads, Proprio, PosEmb（Transformerが動くので visual_pos はONでOK）
            if hasattr(self, "visual_pos_embedding"):
                self.visual_pos_embedding.requires_grad = True
            # prop_pos は保守的に凍結のままでもよい（必要なら下で個別ONに）

        else:
            raise ValueError(f"Unknown finetuning mode: {mode}")

        # # 2) std系パラメータは必ず学習ON（エントロピー固定化を防ぐ）
        # for name in ("log_std_unconstrained", "log_std", "action_std"):
        #     if hasattr(self, name):
        #         getattr(self, name).requires_grad = True

        # 3) 監査（任意）
        if verbose:
            tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
            tot = sum(p.numel() for p in self.parameters())
            print(f"[finetune] mode={mode}, unfreeze_tr_layers={unfreeze_transformer_layers} "
                f"=> trainable {tr}/{tot} ({100*tr/tot:.1f}%)")

    # ... (以降のメソッドは元のコードと同じ) ...
    # ---- 前処理ユーティリティ ----
    def _prep_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        B = depth_flat.shape[0]
        D = depth_flat.view(B, self.num_depth_stack, self.H, self.W)
        valid = torch.isfinite(D) & (D > self.near) & (D < self.far)
        Dc = torch.clamp(D, min=self.near, max=self.far)
        invD = (1.0 / Dc - 1.0 / self.far) / (1.0 / self.near - 1.0 / self.far)
        invD[~valid] = 0.0
        depth_input = torch.cat([invD, valid.float()], dim=1)
        return depth_input

    def _prep_normals(self, normals_flat: torch.Tensor, depth_valid_like: torch.Tensor) -> torch.Tensor:
        B = normals_flat.shape[0]
        N = normals_flat.view(B, 3*self.num_normals_stack, self.H, self.W)
        N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
        if self.num_normals_stack == self.num_depth_stack:
            mask = depth_valid_like.unsqueeze(2).expand(B, self.num_normals_stack, 3, self.H, self.W)
            mask = mask.reshape(B, 3*self.num_normals_stack, self.H, self.W)
            N = N * mask.float()
        return N

    # ---- 観測の前処理と前向き計算 ----
    def get_actor_obs(self, obs):
        prop_vec = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        prop_feat = self.proprioception_encoder(prop_vec)
        prop_token = self.prop_proj(prop_feat).unsqueeze(1)
        prop_token = prop_token + self.prop_pos_embedding

        depth_flat = obs[self.vision_depth_key]
        normals_flat = obs[self.vision_normals_key]
        B = depth_flat.shape[0]
        D = depth_flat.view(B, self.num_depth_stack, self.H, self.W)
        valid = torch.isfinite(D) & (D > self.near) & (D < self.far)
        depth_in = self._prep_depth(depth_flat)
        normals_in = self._prep_normals(normals_flat, valid)
        fd = self.depth_encoder(depth_in)
        fn = self.normals_encoder(normals_in)
        fvis = torch.cat([fd, fn], dim=1)

        vis_tokens = fvis.flatten(2).permute(0, 2, 1)
        vis_tokens = self.vis_proj(vis_tokens)
        vis_tokens = vis_tokens + self.visual_pos_embedding

        all_tokens = torch.cat([prop_token, vis_tokens], dim=1)
        fused_tokens = self.transformer_encoder(all_tokens)
        fused_prop = fused_tokens[:, 0, :]
        fused_vis  = fused_tokens[:, 1:, :].mean(dim=1)
        fused_features = torch.cat([fused_prop, fused_vis], dim=1)
        final_features = self.projection_head(fused_features)
        return final_features

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)
