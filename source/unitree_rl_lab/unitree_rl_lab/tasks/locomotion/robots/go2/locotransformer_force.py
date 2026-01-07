
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from torch.distributions import Normal # <-- Add this line
import torchvision

import torch.nn.functional as F

from rsl_rl.modules import MLP 

from rsl_rl.networks import MLP, EmpiricalNormalization, Memory



class AttnPool(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_heads, d_model)) # 学習可能なクエリ
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x: [Batch, NumTokens, Dim]
        # [H,D]・[B,N,D]^T -> [B,H,N] -> softmax -> [B,H,N] (アテンション重み)
        attn_scores = torch.einsum('hd,bnd->bhn', self.query, x) / (x.shape[-1] ** 0.5)
        attn_weights = attn_scores.softmax(dim=-1)
        # [B,H,N]・[B,N,D] -> [B,H,D] (重み付き和) -> [B,D] (ヘッド間で平均)
        pooled = torch.einsum('bhn,bnd->bhd', attn_weights, x).mean(dim=1)
        return self.proj(pooled)



# class LocoTransformerHFP(ActorCritic):
#     """
#     Heightmap + Force/Torque + Proprio を Transformer で統合するActor-Critic
#     - Heightmap: 2D Conv stem + AdaptiveAvgPool2d(4x4) -> 16トークン
#     - Force/Torque: 各脚の時系列を1D-Conv/TCNで要約 -> 脚トークン(4)
#     - Proprio: MLP -> 1トークン
#     - 最終: 1 + 4 + 16 トークンを Transformer で融合、Actor/Critic へ
#     """
#     def __init__(
#         self,
#         # runner から
#         obs: dict,
#         obs_groups: dict,
#         num_actions: int,

#         # --- Cfgから ---
#         prop_obs_keys: list[str],
#         # heightmap_key: str,                 # e.g. "heightmap"
#         # height_shape: tuple = (32, 32),     # RayCaster: size=[1.6,1.6], res=0.1 -> 16x16
#         height_channels: int = 1,           # 1: height, 2: height+valid など
#         ft_stack_key: str = "ft_stack",     # [B,K,4,6]
#         ft_in_dim: int = 3,                 # Fx,Fy,Fz
#         transformer_hidden_dim: int = 256,
#         transformer_n_heads: int = 4,
#         transformer_num_layers: int = 2,
#         prop_encoder_dims: list[int] = [256, 256],
#         projection_head_dims: list[int] = [256, 256],
#         activation: str = "elu",
#         init_noise_std: float = 1.0,
#         **kwargs,
#     ):
#         # 観測からheight/ftを親に渡さない
#         sanitized_obs = {k: v for k, v in obs.items() if k not in [heightmap_key, ft_stack_key]}
#         sanitized_groups = {g: [k for k in ks if k not in [heightmap_key, ft_stack_key]] for g, ks in obs_groups.items()}
#         super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

#         # 保存
#         self.prop_obs_keys   = prop_obs_keys
#         # self.heightmap_key   = heightmap_key
#         # self.height_shape    = height_shape     # (Hh, Wh)
#         self.height_channels = height_channels
#         self.ft_stack_key    = ft_stack_key
#         self.ft_in_dim       = ft_in_dim

#         # 活性化
#         self.activation = nn.ELU() if activation == "elu" else nn.ReLU()

#         # ===== 1) Proprio encoder =====
#         prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
#         prop_layers, in_dim = [], prop_obs_dim
#         for dim in prop_encoder_dims:
#             prop_layers += [nn.Linear(in_dim, dim), nn.ELU()]
#             in_dim = dim
#         self.proprioception_encoder = nn.Sequential(*prop_layers)
#         self.prop_proj = nn.Linear(prop_encoder_dims[-1], transformer_hidden_dim)

#         # ===== 2) Heightmap encoder（学習可能ダウンサンプリング；情報保持優先） =====
#         # def make_hm_stem(cin):
#         #     return nn.Sequential(
#         #         nn.Conv2d(cin, 32, 3, 1, 1), self.activation,   # 形を保ったまま局所特徴
#         #         nn.Conv2d(32, 64, 3, 2, 1), self.activation,    # /2
#         #         nn.Conv2d(64, 96, 3, 2, 1), self.activation,    # /4
#         #         nn.Conv2d(96, 128, 3, 1, 1), self.activation,   # 出力ch固定
#         #         nn.AdaptiveAvgPool2d((4, 4)),                   # ★常に4x4へ
#         #     )
#         # self.height_encoder = make_hm_stem(self.height_channels)
#         # self.height_proj    = nn.Linear(128, transformer_hidden_dim)
#         # self.height_pos_embedding = nn.Parameter(torch.randn(1, 16, transformer_hidden_dim))  # 4*4=16

#         # ===== 3) Force/Torque encoder（脚×時系列 → 脚トークン8つ） =====
#         # Kは可変でOK（Conv1dは可変長を受けられる）。最後をAvgPool1d(1)で要約。
#         self.ft_encoder = nn.Sequential(
#             nn.Conv1d(self.ft_in_dim, 64, 3, padding=1), self.activation,
#             nn.Conv1d(64, 128, 3, padding=2, dilation=2), self.activation,   # 受容野拡大(TCN風)
#             nn.Conv1d(128, 128, 3, padding=4, dilation=4), self.activation,
#             nn.AdaptiveAvgPool1d(1),   # -> [B*4, 128, 1]
#         )
#         self.ft_proj = nn.Linear(128, transformer_hidden_dim)
#         self.leg_pos_embedding = nn.Parameter(torch.randn(1, 4, transformer_hidden_dim))  # 4脚

#         # ===== 4) Transformer =====
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=transformer_hidden_dim, nhead=transformer_n_heads,
#             dim_feedforward=512, activation="gelu", batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_num_layers)
#         self.prop_pos_embedding  = nn.Parameter(torch.randn(1, 1, transformer_hidden_dim))

#         # ===== 5) 出力ヘッド（Actor/Critic） =====
#         fused_feature_dim = transformer_hidden_dim * 3
#         proj_layers, in_dim = [], fused_feature_dim
#         for dim in projection_head_dims:
#             proj_layers += [nn.Linear(in_dim, dim), nn.ELU()]
#             in_dim = dim
#         self.projection_head = nn.Sequential(*proj_layers)
#         self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
#         self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

#         self.force_pool = AttnPool(transformer_hidden_dim)
#         self.hmap_pool  = AttnPool(transformer_hidden_dim)

#     # ---------- 前処理 ----------
#     # def _prep_heightmap(self, hm_flat: torch.Tensor) -> torch.Tensor:
#     #     """
#     #     hm_flat: [B, C*H*W] or [B, H*W] when C=1
#     #     return:  [B, C, H, W]
#     #     """
#     #     B = hm_flat.shape[0]
#     #     Hh, Wh = self.height_shape
#     #     C = self.height_channels
#     #     if hm_flat.shape[1] == Hh * Wh and C == 1:
#     #         hm = hm_flat.view(B, 1, Hh, Wh)
#     #     else:
#     #         assert hm_flat.shape[1] == C * Hh * Wh, \
#     #             f"heightmap length mismatch: got {hm_flat.shape[1]}, want {C*Hh*Wh}"
#     #         hm = hm_flat.view(B, C, Hh, Wh)
#     #     return torch.nan_to_num(hm, 0.0, 0.0, 0.0)

#     def _prep_ft(self, ft_stack: torch.Tensor) -> torch.Tensor:
#         """
#         ft_stack: [B, K, 4, D] -> [B*4, D, K]
#         """
#         B, K, L, D = ft_stack.shape
#         return ft_stack.permute(0, 2, 3, 1).contiguous().view(B*L, D, K)


#     # ---------- 前向き ----------
#     def get_actor_obs(self, obs):
#         # Proprio -> 1トークン
#         prop_vec  = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
#         prop_feat = self.proprioception_encoder(prop_vec)
#         prop_tok  = self.prop_proj(prop_feat).unsqueeze(1) + self.prop_pos_embedding  # [B,1,D]

#         # Heightmap -> 16トークン
#         # hm_flat   = obs[self.heightmap_key]                 # [B, C*H*W] or [B, H*W]
#         # hm        = self._prep_heightmap(hm_flat)           # [B, C, H, W]
#         # fh        = self.height_encoder(hm)                 # [B, 128, 4, 4]
#         # h_tokens  = fh.flatten(2).permute(0, 2, 1)          # [B, 16, 128]
#         # h_tokens  = self.height_proj(h_tokens)              # [B, 16, D]
#         # h_tokens  = h_tokens + self.height_pos_embedding

#         # Force/Torque -> 脚トークン(4)
#         ft_stack  = obs[self.ft_stack_key]                  # [B, K, 4, D]
#         ft_seq    = self._prep_ft(ft_stack)                 # [B*4, D, K]
#         ff        = self.ft_encoder(ft_seq).squeeze(-1)     # [B*4, 128]
#         # ff        = self.ft_proj(ff).view(ft_stack.shape[0], 4, -1) + self.leg_pos_embedding  # [B,4,D]
#         ff = self.ft_proj(ff)                     # [B*L, D]

#         B = ft_stack.shape[0]
#         L = ft_stack.shape[2] if ft_stack.dim() >= 3 else 4  # 安全
#         D = self.ft_proj.out_features            # or self.D

#         # ★ B=0でも安全な明示リシェイプ
#         ff = ff.view(B, L, D)                    # [B, L, D]
#         # pos-emb は L に合わせてスライス
#         ff = ff + self.leg_pos_embedding[:, :L, :]   # [1, L, D] を想定

#         # すべて結合 → Transformer
#         all_tokens = torch.cat([prop_tok, ff, h_tokens], dim=1)  # [B, 1+4+16, D]
        
#         fused       = self.transformer_encoder(all_tokens)      # [B, 1+4+16, D]
#         prop_tok    = fused[:, 0, :]                            # [B,D]
#         force_tok   = fused[:, 1:1+4, :]                        # [B,4,D]
#         # hmap_tok    = fused[:, 1+4:, :]                         # [B,16,D]

#         force_feat  = self.force_pool(force_tok)                # [B,D]
#         hmap_feat   = self.hmap_pool(hmap_tok)                  # [B,D]

#         feat        = torch.cat([prop_tok, force_feat, hmap_feat], dim=1)  # [B, 3D]
        
#         return self.projection_head(feat)

#     def get_critic_obs(self, obs):
#         return self.get_actor_obs(obs)


    
import torch
from torch import nn
from rsl_rl.modules import ActorCritic  # (RSL-RLのインポートを想定)
# (AttnPool のインポートも必要)
# from .utils import AttnPool 

class LocoTransformerHFP(ActorCritic):
    """
    ★ 修正: Heightmap を削除。
    Force/Torque + Proprio を Transformer で統合するActor-Critic
    - Force/Torque: 各脚の時系列を1D-Conv/TCNで要約 -> 3ステップにプーリング -> 4脚 x 3時間 = 12トークン
    - Proprio: MLP -> 1トークン
    - 最終: 1 + 12 トークンを Transformer で融合、Actor/Critic へ
    """
    def __init__(
        self,
        # runner から
        obs: dict,
        obs_groups: dict,
        num_actions: int,

        # --- Cfgから ---
        prop_obs_keys: list[str],
        ft_stack_key: str = "ft_stack",     # [B,K,4,6]
        ft_in_dim: int = 3,                 # Fx,Fy,Fz
        ft_time_tokens: int = 12,            # ★ 修正: FTの時間軸トークン数 (例: 2)
        heightmap_key: str | None = None,   # ★ 修正: 観測辞書から除くためにキーだけ受け取る
        transformer_hidden_dim: int = 256,
        transformer_n_heads: int = 4,
        transformer_num_layers: int = 2,
        prop_encoder_dims: list[int] = [256, 256],
        projection_head_dims: list[int] = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        # ★ 修正: 観測からftと(存在するなら)heightmapを親に渡さない
        ignore_keys = [ft_stack_key]
        if heightmap_key:
            ignore_keys.append(heightmap_key)
        
        sanitized_obs = {k: v for k, v in obs.items() if k not in ignore_keys}
        sanitized_groups = {g: [k for k in ks if k not in ignore_keys] for g, ks in obs_groups.items()}
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

        # 保存
        self.prop_obs_keys   = prop_obs_keys
        self.ft_stack_key    = ft_stack_key
        self.ft_in_dim       = ft_in_dim
        self.num_ft_time_tokens = ft_time_tokens # (例: 2)
        self.num_legs = 4 # (ハードコード)
        self.transformer_hidden_dim = transformer_hidden_dim


        # 活性化
        self.activation = nn.ELU() if activation == "elu" else nn.ReLU()

        # ===== 1) Proprio encoder =====
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        # prop_layers, in_dim = [], prop_obs_dim
        # for dim in prop_encoder_dims:
        #     prop_layers += [nn.Linear(in_dim, dim), nn.ELU()]
        #     in_dim = dim
        # self.proprioception_encoder = nn.Sequential(*prop_layers)
        # self.prop_proj = nn.Linear(prop_encoder_dims[-1], transformer_hidden_dim)

        self.prop_proj = nn.Linear(prop_obs_dim, transformer_hidden_dim)

        # ===== 2) Heightmap encoder（学習可能ダウンサンプリング；情報保持優先） =====
        # ★ 修正: Heightmap関連のすべてを削除
        # self.height_encoder = ...
        # self.height_proj    = ...
        # self.height_pos_embedding = ...

        # ===== 3) Force/Torque encoder（脚×時系列 → 8トークン） =====
        # self.ft_encoder = nn.Sequential(
        #     nn.Conv1d(self.ft_in_dim, 64, 3, padding=1), self.activation,
        #     nn.Conv1d(64, 128, 3, padding=2, dilation=2), self.activation,
        #     nn.Conv1d(128, 128, 3, padding=4, dilation=4), self.activation,
        #     # ★ 修正: (1) -> (N) に変更し、時間軸のトークンをN個生成
        #     nn.AdaptiveAvgPool1d(self.num_ft_time_tokens),   # -> [B*4, 128, 3]
        # )
        self.ft_encoder = nn.Sequential( # [B*4, 3,  12]
            # Kernel=3, Padding=1 なら、入力12 -> 出力12 
            nn.Conv1d(self.ft_in_dim, transformer_hidden_dim, kernel_size=3, padding=1),
            self.activation # お好みで活性化関数
        ) #[B*4, 256, 12]

        
        # ★ 修正: 4脚 -> (4脚 * N時間トークン) = 48トークン
        num_ft_tokens = self.num_legs * self.num_ft_time_tokens # (4 * 3 = 12)
        # self.leg_pos_embedding = nn.Parameter(torch.randn(1, num_ft_tokens, transformer_hidden_dim)* 0.02)

        # [1, 4, 1, D] ... 脚の埋め込み (4脚分)
        self.leg_embedding = nn.Parameter(torch.randn(1, self.num_legs, 1, transformer_hidden_dim) * 0.02)
        # [1, 1, 3, D] ... 時間の埋め込み (3ステップ分)
        self.time_embedding = nn.Parameter(torch.randn(1, 1, self.num_ft_time_tokens, transformer_hidden_dim) * 0.02)

        # ===== 4) Transformer =====
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim, nhead=transformer_n_heads,
            dim_feedforward=512, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_num_layers)
        self.prop_pos_embedding  = nn.Parameter(torch.randn(1, 1, transformer_hidden_dim)* 0.02)

        # ===== 5) 出力ヘッド（Actor/Critic） =====
        # ★ 修正: 3D -> 2D (Prop + Force のみ)
        # fused_feature_dim = transformer_hidden_dim * 2
        # proj_layers, in_dim = [], fused_feature_dim

        total_tokens = 1 + (self.num_legs * self.num_ft_time_tokens)
        fused_feature_dim = transformer_hidden_dim * total_tokens
        proj_layers, in_dim = [], fused_feature_dim

    

        for dim in projection_head_dims:
            proj_layers += [nn.Linear(in_dim, dim), nn.ELU()]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)
        self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
        self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

        self.force_pool = AttnPool(transformer_hidden_dim)
        # ★ 修正: Heightmap プールを削除
        # self.hmap_pool  = AttnPool(transformer_hidden_dim)

    # ---------- 前処理 ----------
    # ★ 修正: _prep_heightmap メソッドを削除
    # def _prep_heightmap(self, hm_flat: torch.Tensor) -> torch.Tensor:
    #     ...

    def _prep_ft(self, ft_stack: torch.Tensor) -> torch.Tensor:
        """
        ft_stack: [B, K, 4, D] -> [B*4, D, K]
        """
        B, K, L, D = ft_stack.shape
       
            
        # (B, L, D, K) -> (B*L, D, K)
        return ft_stack.permute(0, 2, 3, 1).contiguous().view(B*L, D, K)


    # ---------- 前向き ----------
    def get_actor_obs(self, obs):
        # Proprio -> 1トークン
        prop_vec  = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        # prop_feat = self.proprioception_encoder(prop_vec)
        # prop_tok  = self.prop_proj(prop_feat).unsqueeze(1) + self.prop_pos_embedding  # [B,1,D]
        prop_tok  = self.prop_proj(prop_vec).unsqueeze(1) + self.prop_pos_embedding  # [B,1,D]

        # ★ 修正: Heightmap ブロックを削除
        # hm_flat   = obs[self.heightmap_key]
        # ...
        # h_tokens  = ...

        # Force/Torque -> 12トークン (4脚 x 2時間)
        ft_stack  = obs[self.ft_stack_key]                  # [B, K 12, L 4, D_in 3]
        B = ft_stack.shape[0]
        L = self.num_legs # 4
        T = self.num_ft_time_tokens # 12
        D = self.transformer_hidden_dim

        ft_seq    = self._prep_ft(ft_stack)                 # [B*4, D_in, K]
        ff        = self.ft_encoder(ft_seq)                 # [B*4, 256, 12]
        
        # ★ 修正: 射影の前に(T, D_feat)の形に
        ff        = ff.permute(0, 2, 1).contiguous()        # [B*4, 12, 256]
        # ff        = self.ft_proj(ff)                        # [B*4, 2, D_trans]
        
        # ★ 修正: (B, L, T, D) -> (B, L*T, D) に変形
        ff = ff.view(B, L, T, D)                         # [B, 4, 12, D 256]

        ff = ff + self.leg_embedding + self.time_embedding                        
        
        # ★ 修正: (1, 8, D) の Positional Embedding を加算
        # ff = ff + self.leg_pos_embedding                 # [B, 12, D]

        ff = ff.flatten(1, 2) 

        # ★ 修正: h_tokens を削除
        all_tokens = torch.cat([prop_tok, ff], dim=1)   # [B, 1+48, D]


        
        fused  = self.transformer_encoder(all_tokens)      # [B, 49, D]


        # prop_tok    = fused[:, 0, :]                            # [B,D]
        
        # # ★ 修正: (1:1+4) -> (1:) に変更 (残りのトークンすべて)
        # force_tok   = fused[:, 1:, :]                         # [B, 12, D]
        
        # # ★ 修正: hmap_tok を削除
        # # hmap_tok    = ...

        # force_feat  = self.force_pool(force_tok)                # [B,D]　　　　　　　　　　　　＊＊＊＊＊＊
        # # ★ 修正: hmap_feat を削除
        # # hmap_feat   = self.hmap_pool(hmap_tok)

        # # ★ 修正: hmap_feat を削除
        # feat        = torch.cat([prop_tok, force_feat], dim=1)  # [B, 2D]

        feat = fused.flatten(start_dim=1) # when not use Attention Pool
        
        return self.projection_head(feat) 

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)





from einops import rearrange

# class MlpHFP(ActorCritic):
#     """
#     Proprio + Force/Torque（時系列は時間平均）を単純にConcatして
#     MLPでActor/Criticに入れるベースライン。
#     - ft_stack: [B, K, 4, D_in]  (K=time, legs=4, D_in>=3)
#     - proprioは prop_obs_keys をそのまま連結
#     """
    # def __init__(
    #     self,
    #     obs: dict,
    #     obs_groups: dict,
    #     num_actions: int,
    #     prop_obs_keys: list[str],
    #     ft_stack_key: str = "ft_stack",
    #     ft_in_dim: int = 3,                 # 例: Fx,Fy,Fz のみ使用
    #     ft_time_tokens: int = 3,            # Conv後の時間トークン数(K->この数に集約)
    #     prop_encoder_dims: list[int] = [256, 256],
    #     projection_head_dims: list[int] = [256, 256],
    #     ft_feat_dim: int = 128,             # Conv出力チャンネル
    #     activation: str = "elu",
    #     init_noise_std: float = 1.0,
    #     use_layernorm: bool = True,
    #     **kwargs,
    # ):
    #     # 親へはFTを渡さない（自前で処理）
    #     ignore_keys = [ft_stack_key]
    #     sanitized_obs = {k: v for k, v in obs.items() if k not in ignore_keys}
    #     sanitized_groups = {g: [k for k in ks if k not in ignore_keys] for g, ks in obs_groups.items()}
    #     super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

#         # 保存
#         self.prop_obs_keys = prop_obs_keys
#         self.ft_stack_key  = ft_stack_key
#         self.ft_in_dim     = ft_in_dim
#         self.num_ft_time_tokens = ft_time_tokens
#         self.num_legs = 4

#         act = nn.ELU() if activation == "elu" else nn.ReLU()

#         # ===== Proprio encoder =====
#         prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
#         prop_layers, in_dim = [], prop_obs_dim
#         for dim in prop_encoder_dims:
#             prop_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.proprioception_encoder = nn.Sequential(*prop_layers)
#         self.prop_out_dim = prop_encoder_dims[-1]
#         self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()

#         # ===== Force/Torque encoder (脚×時系列→脚ベクトル) =====
#         # Conv1d は [B*L, C_in, K] を受け取る
#         self.ft_encoder = nn.Sequential(
#             nn.Conv1d(self.ft_in_dim, 64, 3, padding=1), act,
#             nn.Conv1d(64, ft_feat_dim, 3, padding=2, dilation=2), act,
#             nn.AdaptiveAvgPool1d(self.num_ft_time_tokens),     # -> [B*L, ft_feat_dim, T]
#         )
#         # self.ft_encoder = nn.Sequential( # [B*4, 3,  12]
#         #     # Kernel=3, Padding=1 なら、入力12 -> 出力12 
#         #     nn.Conv1d(self.ft_in_dim, ft_feat_dim, kernel_size=3, padding=1),
#         #     act # お好みで活性化関数
#         # )
#         self.ft_feat_dim = ft_feat_dim
#         # 時間平均（必要なら注意に差し替え可）
#         self.ft_out_norm = nn.LayerNorm(self.num_legs * self.ft_feat_dim) if use_layernorm else nn.Identity()

#         # ===== Projection head / Actor-Critic =====
#         fused_feature_dim = self.prop_out_dim + self.num_legs * self.ft_feat_dim
#         proj, in_dim = [], fused_feature_dim
#         for dim in projection_head_dims:
#             proj += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.projection_head = nn.Sequential(*proj)
#         self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
#         self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

#     def _prep_ft(self, ft_stack: torch.Tensor) -> torch.Tensor:
#         """
#         ft_stack: [B, K, 4, D_all] -> [B*4, C_in, K] （C_in=self.ft_in_dim）
#         ※ D_all>=C_in を想定。Fx,Fy,Fz が先頭3でない場合は注意
#         """
#         B, K, L, Dall = ft_stack.shape
#         assert L == self.num_legs, f"legs={L} != {self.num_legs}"
#         if Dall < self.ft_in_dim:
#             raise ValueError(f"ft_in_dim={self.ft_in_dim} but input has {Dall} channels")
#         # 先頭3がFx,Fy,Fzでない場合は、環境に合わせて明示インデックスを切る:
#         # idx_fx, idx_fy, idx_fz = ...
#         ft_stack = ft_stack[..., :self.ft_in_dim]  # [B,K,4,C_in]
#         return ft_stack.permute(0, 2, 3, 1).contiguous().view(B * L, self.ft_in_dim, K)

#     def get_actor_obs(self, obs):
#         # ---- Proprio ----
#         prop_vec  = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
#         prop_feat = self.prop_norm(self.proprioception_encoder(prop_vec))  # [B, P]

#         # ---- Force/Torque ----
#         ft_stack = obs[self.ft_stack_key]                  # [B, K, 4, D_in>=3]
#         B, K, L, _ = ft_stack.shape
#         x = self._prep_ft(ft_stack)                        # [B*L, C_in, K]
#         x = self.ft_encoder(x)                             # [B*L, F, T]
#         x = x.permute(0, 2, 1).contiguous()                # [B*L, T, F]
#         x = x.mean(dim=1)                                  # 時間平均 → [B*L, F]
#         x = x.view(B, L * self.ft_feat_dim)                # 脚を連結 → [B, L*F]
#         x = self.ft_out_norm(x)

#         feat = torch.cat([prop_feat, x], dim=-1)           # [B, fused_feature_dim]

#         # feat = prop_feat
#         return self.projection_head(feat)                  # [B, hidden]

#     def get_critic_obs(self, obs):
#         return self.get_actor_obs(obs)







class PropFtActor(nn.Module):
    def __init__(
        self,
        # prop_keys: list[str],
        prop_dim,
        ft_time_steps: int,   # K
        ft_in_dim: int,       # Fx,Fy,Fz → 3 など
        num_legs: int,
        num_actions: int=12,
        ft_feat_dim: int = 128,
        prop_hidden_dims: list[int] = [256, 256],
        fused_hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        use_layernorm: bool = True,
        # ft_key : str = "ft_stack"
    ):
        super().__init__()
        # self.prop_obs_keys = prop_keys
        self.ft_time_steps = ft_time_steps
        self.ft_in_dim = ft_in_dim
        self.num_legs = num_legs
        self.prop_dim = prop_dim
        # self.ft_stack_key = ft_key
        self.K = ft_time_steps
        self.L = num_legs
        self.D = ft_in_dim

        # ここがONNX用のヒント
        self.input_dim = self.prop_dim + self.K * self.L * self.D

        act = nn.ELU() if activation == "elu" else nn.ReLU()

        # --- Proprio encoder ---
        # prop_obs_dim = 81
        prop_layers = []
        in_dim = prop_dim
        for dim in prop_hidden_dims:
            prop_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.prop_encoder = nn.Sequential(*prop_layers)
        self.prop_out_dim = prop_hidden_dims[-1]
        self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()

        # --- FT encoder (ほぼ元の _prep_ft + ft_encoder 相当) ---
        self.ft_encoder = nn.Sequential(
            nn.Conv1d(self.ft_in_dim, 64, 3, padding=1), act,
            nn.Conv1d(64, ft_feat_dim, 3, padding=2, dilation=2), act,
            nn.AdaptiveAvgPool1d(3),      # 例: 時間トークン数 = 3
        )
        self.ft_feat_dim = ft_feat_dim
        self.ft_out_norm = nn.LayerNorm(self.num_legs * self.ft_feat_dim) if use_layernorm else nn.Identity()

        # --- Fused projection ---
        fused_dim = self.prop_out_dim + self.num_legs * self.ft_feat_dim
        proj_layers = []
        in_dim = fused_dim
        for dim in fused_hidden_dims:
            proj_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)

        # --- 最後の Actor ヘッド ---
        self.linear_out = nn.Linear(fused_hidden_dims[-1], num_actions)



    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:#Obs_flatは辞書


        # ---- Proprio ----
        # prop  = torch.cat([obs_flat[k] for k in self.prop_obs_keys], dim=-1)
        # # ---- Force/Torque ----
        # ft = obs_flat[self.ft_stack_key]                  # [B, K, 4, D_in>=3]
        # B = obs_flat.shape[0]
        B, obs_dim = obs_flat.shape




        prop = obs_flat[:, : self.prop_dim]              # [B, prop_dim]
        ft_flat = obs_flat[:, self.prop_dim:]


        # 再び [B,K,4,D] に戻す

        # B, K, L, _ = ft.shape
        K = self.ft_time_steps
        L = self.num_legs
        D = self.ft_in_dim
        ft = ft_flat.view(B, K, L, D)  # [B,K,L,D]

        # Conv1d 用に [B*L, C_in, K] へ
        ft = ft.permute(0, 2, 3, 1).contiguous()          # [B,L,D,K]
        ft = ft.view(B * L, D, K)                         # [B*L, C_in, K]
        ft = self.ft_encoder(ft)                          # [B*L, F, T]
        ft = ft.permute(0, 2, 1).contiguous().mean(1)     # 時間平均 [B*L, F]
        ft = ft.view(B, L * self.ft_feat_dim)             # [B, L*F]
        ft = self.ft_out_norm(ft)

        # Proprio + FT 融合
        prop_feat = self.prop_norm(self.prop_encoder(prop))
        fused = torch.cat([prop_feat, ft], dim=-1)
        fused = self.projection_head(fused)

        return self.linear_out(fused)




class MlpHFP(ActorCritic):

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        prop_obs_keys: list[str],
        ft_stack_key: str = "ft_stack",
        ft_in_dim: int = 3,                 # 例: Fx,Fy,Fz のみ使用
        ft_time_steps: int = 12,            # Conv後の時間トークン数(K->この数に集約)
        prop_encoder_dims: list[int] = [256, 256],
        projection_head_dims: list[int] = [256, 256],
        ft_feat_dim: int = 128,             # Conv出力チャンネル
        activation: str = "elu",
        init_noise_std: float = 1.0,
        use_layernorm: bool = True,
        num_legs =4,
        **kwargs,
    ):
 
         # 親へはFTを渡さない（自前で処理）
        ignore_keys = [ft_stack_key]
        sanitized_obs = {k: v for k, v in obs.items() if k not in ignore_keys}
        sanitized_groups = {g: [k for k in ks if k not in ignore_keys] for g, ks in obs_groups.items()}
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

        # 親には「フラット観測の次元」だけ渡す
        # super().__init__(num_obs=num_obs, num_actions=num_actions, **kwargs)
        # super().__init__(obs=obs, obs_groups=obs_groups, num_actions=num_actions, **kwargs)

        #観測は辞書型、PropとFtに分けられている


        B = next(iter(obs.values())).shape[0]


        self.prop_dim = sum(
            obs[k].view(B, -1).shape[1]
            for k in prop_obs_keys
        )

        self.prop_obs_keys = prop_obs_keys
        self.ft_stack_key = ft_stack_key

        # 観測レイアウト情報を保持
        self.ft_time_steps = ft_time_steps
        self.ft_in_dim = ft_in_dim
        self.num_legs = num_legs

        # ★ここがポイント：self.actor / self.critic に「自前モジュール」を突っ込む
        self.actor  = PropFtActor(
            prop_dim=self.prop_dim,
            ft_time_steps=self.ft_time_steps,
            ft_in_dim=self.ft_in_dim,
            num_legs=self.num_legs,
            num_actions=num_actions,
        )
        print(f"Actor MLP: {self.actor}")
        self.critic = PropFtActor(
            prop_dim=self.prop_dim,
            ft_time_steps=self.ft_time_steps,
            ft_in_dim=self.ft_in_dim,
            num_legs=self.num_legs,
            num_actions=1,
        )
        print(f"Critic MLP: {self.critic}")

    
    def get_actor_obs(self, obs):
        # return obs      # [B, num_obs]  フラットのまま

    
        """
        obs: {"base_lin_vel": [B,...], "ft_stack": [B,K,4,D], ...}
        を [B, prop_dim + ft_flat_dim] の Tensor に変換する
        """
        # まずバッチサイズ
        B = next(iter(obs.values())).shape[0]

        # Proprio 部分（指定キーを順番に flatten）
        prop_list = []
        for k in self.prop_obs_keys:
            t = obs[k]
            t = t.view(B, -1)
            prop_list.append(t)
        prop = torch.cat(prop_list, dim=-1)  # [B, prop_dim]

        # FT 部分（そのまま flatten）
        ft = obs[self.ft_stack_key].view(B, -1)  # [B, ft_flat_dim]

        return torch.cat([prop, ft], dim=-1)     # [B, prop_dim + ft_flat_dim]

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)








class HiActor(nn.Module):
    def __init__(
        self,
        # prop_keys: list[str],
        prop_dim,
        num_actions: int,
        ft_feat_dim: int = 128,
        hm_shape: tuple[int, int] = (64, 64),   # Heightmap の (H,W
        in_ch : int = 3,
        prop_hidden_dims: list[int] = [256, 256],
        fused_hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        use_layernorm: bool = True,
        # ft_key : str = "ft_stack"
    ):
        super().__init__()
        # self.prop_obs_keys = prop_keys
        
        self.prop_dim = prop_dim
        # self.ft_stack_key = ft_key
        self.hm_H, self.hm_W = hm_shape
        self.in_ch = in_ch
       

        # ここがONNX用のヒント
        self.input_dim = self.prop_dim + self.hm_H * self.hm_W * self.in_ch

        act = nn.ELU() if activation == "elu" else nn.ReLU()

        # --- Proprio encoder ---
        # prop_obs_dim = 81
        prop_layers = []
        in_dim = prop_dim
        for dim in prop_hidden_dims:
            prop_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.prop_encoder = nn.Sequential(*prop_layers)
        self.prop_out_dim = prop_hidden_dims[-1]
        self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()


        self.hm_encoder = nn.Sequential(
            # 入力: [B, 4, 64, 64]
            nn.Conv2d(self.in_ch, 32, kernel_size=5, stride=2, padding=2),  # -> [B,32,32,32]
            act,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> [B,64, 16, 16]
            act,
            # 必要ならもう1段（32x32ならここで止めてもOK）
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), act,# -> [B,64, 8, 8]
        )
        # self.hm_feat_dim = 64   # 最終 Conv のチャンネル数
        self.hm_conv_out_channels = 64
        # self.hm_feat_dim = 128
        self.hm_conv_out_H = 8
        self.hm_conv_out_W = 8

        self.hm_mlp = nn.Sequential(
            nn.Linear(self.hm_conv_out_channels * self.hm_conv_out_H * self.hm_conv_out_W, 512),
            act,
            nn.Linear(512, 256),
            act,
        )

        # self.hm_global_pool = lambda x: x.mean(dim=[2, 3])
        self.hm_feat_dim = 256
        self.hm_norm = nn.LayerNorm(self.hm_feat_dim) if use_layernorm else nn.Identity()
        

        # --- Fused projection ---
        fused_dim = self.prop_out_dim + self.hm_feat_dim
        proj_layers = []
        in_dim = fused_dim
        for dim in fused_hidden_dims:
            proj_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)

        # --- 最後の Actor ヘッド ---
        self.linear_out = nn.Linear(fused_hidden_dims[-1], num_actions)



    def forward(self, obs_flat: torch.Tensor) -> torch.Tensor:#Obs_flatは辞書


        # ---- Proprio ----
        # prop  = torch.cat([obs_flat[k] for k in self.prop_obs_keys], dim=-1)
        # # ---- Force/Torque ----
        # ft = obs_flat[self.ft_stack_key]                  # [B, K, 4, D_in>=3]
        # B = obs_flat.shape[0]
        B, obs_dim = obs_flat.shape


        prop = obs_flat[:, : self.prop_dim]              # [B, prop_dim]
        hm_flat = obs_flat[:, self.prop_dim:]      # [B, H*W] or [B, H, W]

        if hm_flat.ndim == 2:
            B = hm_flat.shape[0]
            hm = hm_flat.view(B, self.in_ch, self.hm_H, self.hm_W)  # [B,1,H,W]
        elif hm_flat.ndim == 3:
            # すでに [B,H,W] の場合2
            B = hm_flat.shape[0]
            hm = hm_flat.view(B, self.in_ch, self.hm_H, self.hm_W)
        else:
            raise ValueError(f"Unexpected heightmap shape: {hm_flat.shape}")

        x = self.hm_encoder(hm)                  # [B,64,h',w']
        # x = self.hm_global_pool(x)               # [B,64]

        x = x.view(B, -1)                # flatten → [B, C*H'*W']
        x = self.hm_mlp(x)                         # [B,128]
        hm =self.hm_norm(x)


        # Proprio + FT 融合
        prop_feat = self.prop_norm(self.prop_encoder(prop))
        fused = torch.cat([prop_feat, hm], dim=-1)
        fused = self.projection_head(fused)

        return self.linear_out(fused)




class VisionHighLevelAC(ActorCritic):

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        prop_obs_keys: list[str],
        heightmap_key: str = "heightmap",
        # ft_stack_key: str = "ft_stack",
        hm_shape: tuple[int, int] = (64, 64),   # Heightmap の (H,W)
        # hm_shape: tuple[int, int] = (32, 32),
        prop_encoder_dims: list[int] = [256, 256],
        projection_head_dims: list[int] = [256, 256],
        # core_hidden_dim=256,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        use_layernorm: bool = True,
        **kwargs,
    ):
 
         # 親へはFTを渡さない（自前で処理）
        ignore_keys = [heightmap_key]
        sanitized_obs = {k: v for k, v in obs.items() if k not in ignore_keys}
        sanitized_groups = {g: [k for k in ks if k not in ignore_keys] for g, ks in obs_groups.items()}
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

        # 親には「フラット観測の次元」だけ渡す
        # super().__init__(num_obs=num_obs, num_actions=num_actions, **kwargs)
        # super().__init__(obs=obs, obs_groups=obs_groups, num_actions=num_actions, **kwargs)

        #観測は辞書型、PropとFtに分けられている


        B = next(iter(obs.values())).shape[0]


        self.prop_dim = sum(
            obs[k].view(B, -1).shape[1]
            for k in prop_obs_keys
        )

        self.prop_obs_keys = prop_obs_keys
        self.heightmap_key= heightmap_key
        self.hm_shape = hm_shape


        # ★ここがポイント：self.actor / self.critic に「自前モジュール」を突っ込む
        self.actor  = HiActor(
            prop_dim=self.prop_dim,
            hm_shape = self.hm_shape,
            num_actions=num_actions,
        )
        print(f"Actor MLP: {self.actor}")
        self.critic = HiActor(
            prop_dim=self.prop_dim,
            hm_shape = self.hm_shape,
            num_actions=1,
        )
        print(f"Critic MLP: {self.critic}")

    
    def get_actor_obs(self, obs):
        # return obs      # [B, num_obs]  フラットのまま

    
        """
        obs: {"base_lin_vel": [B,...], "ft_stack": [B,K,4,D], ...}
        を [B, prop_dim + ft_flat_dim] の Tensor に変換する
        """
        # まずバッチサイズ
        B = next(iter(obs.values())).shape[0]

        # Proprio 部分（指定キーを順番に flatten）
        prop_list = []
        for k in self.prop_obs_keys:
            t = obs[k]
            t = t.view(B, -1)
            prop_list.append(t)
        prop = torch.cat(prop_list, dim=-1)  # [B, prop_dim]

        # FT 部分（そのまま flatten）
        hm = obs[self.heightmap_key].view(B, -1)  # [B, ft_flat_dim]

        return torch.cat([prop, hm], dim=-1)     # [B, prop_dim + ft_flat_dim]

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)



# class VisionHighLevelAC(ActorCritic):
#     """
#     上位ポリシー:
#     - 観測:
#         - Heightmap:   env 側の ObservationTerm で生成した 2D 高さマップ
#         - Proprio:     prop_obs_keys で指定したベクトル観測（下位と似た構成）
#     - モデル:
#         - Heightmap → CNN で z_img
#         - Proprio  → MLP で z_prop
#         - [z_img, z_prop] → Projection head → Actor / Critic
#     """
#     def __init__(
#         self,
#         obs: dict,
#         obs_groups: dict,
#         num_actions: int,
#         prop_obs_keys: list[str],
#         heightmap_key: str = "heightmap",
#         # ft_stack_key: str = "ft_stack",

#         # ft_stack_key: str = "ft_stack",
#         hm_shape: tuple[int, int] = (64, 64),   # Heightmap の (H,W)
#         # hm_shape: tuple[int, int] = (32, 32),
#         prop_encoder_dims: list[int] = [256, 256],
#         projection_head_dims: list[int] = [256, 256],
#         # core_hidden_dim=256,
#         activation: str = "elu",
#         init_noise_std: float = 1.0,
#         use_layernorm: bool = True,
#         **kwargs,
#     ):
#         # Heightmap は自前で処理するので、親には渡すがそのままでもOK（無視しても良い）
#         # 観測からheight/ftを親に渡さない
#         sanitized_obs = {k: v for k, v in obs.items() if k not in [heightmap_key, ]}
#         sanitized_groups = {g: [k for k in ks if k not in [heightmap_key]] for g, ks in obs_groups.items()}

#         # print("=== env.cfg.observations.policy.term_cfgs ===")
#         # print(env.cfg.observations.policy.term_cfgs.keys())

#         # # 2) 実際に ActorCritic に渡ってきた obs のキー
#         # print("=== obs keys passed to ActorCritic ===")
#         # print(list(obs.keys()))
#         # for k, v in obs.items():
#         #     print(k, v.shape)

#         # print("=== sanitized_obs shapes ===")
#         # for k, v in sanitized_obs.items():
#         #     print(k, v.shape, "ndim=", len(v.shape))

#         #     # デバッグ用: どれが3次元/4次元か強制チェック
#         #     assert len(v.shape) == 2, f"BAD OBS SHAPE: key={k}, shape={v.shape}"
#         super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

#         # super().__init__(obs=obs, obs_groups=obs_groups, num_actions=num_actions,
#         #                  init_noise_std=init_noise_std, **kwargs)

       

#         self.prop_obs_keys = prop_obs_keys
#         self.heightmap_key = heightmap_key
#         self.hm_H, self.hm_W = hm_shape

#         act = nn.ELU() if activation == "elu" else nn.ReLU()

#         # ===== Proprio encoder =====
#         prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
#         prop_layers, in_dim = [], prop_obs_dim
#         for dim in prop_encoder_dims:
#             prop_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.proprioception_encoder = nn.Sequential(*prop_layers)
#         self.prop_out_dim = prop_encoder_dims[-1]
#         self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()

#         # ===== Heightmap encoder (2D Conv) =====
#         # 入力: [B, 1, H, W]
#         # self.hm_encoder = nn.Sequential(
#         #     nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), act,   # -> [B,16,H/2,W/2]
#         #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), act,  # -> [B,32,H/4,W/4]
#         #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), act,  # -> [B,64,H/8,W/8]
#         # )

#         # self.hm_encoder = nn.Sequential(
#         #     nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), act,   # 1→32
#         #     nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1), act,   # 32→64
#         #     nn.Conv2d(64,128,kernel_size=3, stride=2, padding=1), act,   # 64→128
#         # )

#         self.hm_encoder = nn.Sequential(
#             # 入力: [B, 4, 64, 64]
#             nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # -> [B,32,32,32]
#             act,
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> [B,64, 16, 16]
#             act,
#             # 必要ならもう1段（32x32ならここで止めてもOK）
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), act,# -> [B,64, 8, 8]
#         )
#         # self.hm_feat_dim = 64   # 最終 Conv のチャンネル数
#         self.hm_conv_out_channels = 64
#         # self.hm_feat_dim = 128
#         self.hm_conv_out_H = 8
#         self.hm_conv_out_W = 8

#         self.hm_mlp = nn.Sequential(
#             nn.Linear(self.hm_conv_out_channels * self.hm_conv_out_H * self.hm_conv_out_W, 512),
#             act,
#             nn.Linear(512, 256),
#             act,
#         )

#         # self.hm_global_pool = lambda x: x.mean(dim=[2, 3])
#         self.hm_feat_dim = 256
#         self.hm_norm = nn.LayerNorm(self.hm_feat_dim) if use_layernorm else nn.Identity()


#         # ===== Projection head / Actor-Critic =====
#         fused_feature_dim = self.prop_out_dim + self.hm_feat_dim #512
#         proj, in_dim = [], fused_feature_dim
#         for dim in projection_head_dims:
#             proj += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.projection_head = nn.Sequential(*proj)


#         self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
#         self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))


#         # self.actor  = nn.Sequential(nn.Linear(core_hidden_dim, num_actions))
#         # self.critic = nn.Sequential(nn.Linear(core_hidden_dim, 1))

#     # ------------------------------------------------------------------
#     #  観測のエンコード
#     # ------------------------------------------------------------------
#     def _encode_proprio(self, obs):
#         # 下位と同様、指定された key を concat。
#         # 例: ["base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", ...]
#         prop_vec  = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
#         prop_feat = self.proprioception_encoder(prop_vec)       # [B, P]
#         return self.prop_norm(prop_feat)

#     def _encode_heightmap(self, obs):
#         # ObservationTerm 側で heightmap をフラット (2H*W) で返している想定。
#         # 例: depth_heightmap() が (N,H,W,1) → (N,H*W) で返している場合、
#         #     ここで (B,1,H,W) に reshape して Conv に入れる。
#         hm_flat = obs[self.heightmap_key]        # [B, H*W] or [B, H, W]
#         if hm_flat.ndim == 2:
#             B = hm_flat.shape[0]
#             hm = hm_flat.view(B, 3, self.hm_H, self.hm_W)  # [B,1,H,W]
#         elif hm_flat.ndim == 3:
#             # すでに [B,H,W] の場合2
#             B = hm_flat.shape[0]
#             hm = hm_flat.view(B, 3, self.hm_H, self.hm_W)
#         else:
#             raise ValueError(f"Unexpected heightmap shape: {hm_flat.shape}")

#         x = self.hm_encoder(hm)                  # [B,64,h',w']
#         # x = self.hm_global_pool(x)               # [B,64]

#         x = x.view(B, -1)                # flatten → [B, C*H'*W']
#         x = self.hm_mlp(x)                         # [B,128]


#         return self.hm_norm(x)

#     # ------------------------------------------------------------------
#     #  Actor / Critic 入力
#     # ------------------------------------------------------------------
#     def get_actor_obs(self, obs):
#         prop_feat = self._encode_proprio(obs)    # [B, P]
#         hm_feat   = self._encode_heightmap(obs)  # [B, 64]
#         feat = torch.cat([prop_feat, hm_feat], dim=-1)  # [B, fused_feature_dim]　256＋256=512
        
#         return self.projection_head(feat)

#     def get_critic_obs(self, obs):
#         # 今回は Actor/Critic 同じ特徴を使う（MlpHFP と同じポリシー）
#         return self.get_actor_obs(obs)


    


   



import torch
import torch.nn as nn
from rsl_rl.modules import ActorCriticRecurrent  # ★ここを継承元にする


# class VisionHighLevelAC(ActorCriticRecurrent):
#     """
#     B案:
#       - env からは 1 本のフラット観測ベクトルを受け取る
#         obs = [ proprio (prop_obs_dim), heightmap (hm_channels * H * W) ]
#       - まず CNN+MLP で埋め込み
#       - その埋め込みを ActorCriticRecurrent の Memory に通す
#     """

#     def __init__(
#         self,
#         num_actor_obs: int ,
#         num_critic_obs: int,
#         num_actions: int,
#         # ---- ここから先はこのクラス専用の引数 ----
#         prop_obs_dim: int = 37,                    # Proprio の次元数
#         hm_shape: tuple[int, int] = (64, 64),
#         hm_channels: int = 3,                 # {height, mask} なら 2
#         prop_encoder_dims: list[int] = [256, 256],
#         hm_mlp_dims: list[int] = [512, 256],
#         fused_mlp_dims: list[int] = [256, 256],   # Proprio+Heightmap を結合後のMLP
#         # ActorCriticRecurrent 自体の設定
#         actor_hidden_dims: list[int] = [256, 256, 256],
#         critic_hidden_dims: list[int] = [256, 256, 256],
#         activation: str = "elu",
#         rnn_type: str = "gru",               # or "gru"
#         rnn_hidden_size: int = 256,
#         rnn_num_layers: int = 1,
#         init_noise_std: float = 1.0,
#         **kwargs,
#     ):
#         self.prop_obs_dim = prop_obs_dim
#         self.hm_H, self.hm_W = hm_shape
#         self.hm_channels = hm_channels

#         # --- 生観測の次元チェック ---
#         expected_obs_dim = prop_obs_dim + hm_channels * self.hm_H * self.hm_W
#         # assert num_actor_obs == expected_obs_dim, \
#         #     f"num_actor_obs({num_actor_obs}) != prop_obs_dim + hm_channels*H*W ({expected_obs_dim})"

#         current_obs_dim = sum(v.flatten(1).shape[1] for v in num_actor_obs.values())
#         assert current_obs_dim == expected_obs_dim, \
#             f"Obs dim mismatch: {current_obs_dim} != {expected_obs_dim}"
#         assert num_critic_obs == expected_obs_dim, \
#             f"num_critic_obs({num_critic_obs}) != prop_obs_dim + hm_channels*H*W ({expected_obs_dim})"

#         # --- 活性化関数 ---
#         act = nn.ELU() if activation == "elu" else nn.ReLU()

#         # ===========================
#         # 1) Proprio Encoder (MLP)
#         # ===========================
#         prop_layers = []
#         in_dim = prop_obs_dim
#         for dim in prop_encoder_dims:
#             prop_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.proprio_encoder = nn.Sequential(*prop_layers)
#         self.prop_out_dim = prop_encoder_dims[-1]
#         self.prop_norm = nn.LayerNorm(self.prop_out_dim)

#         # ===========================
#         # 2) Heightmap Encoder (CNN + MLP)
#         #    入力: [*, hm_channels, H, W]
#         # ===========================
#         self.hm_encoder = nn.Sequential(
#             nn.Conv2d(hm_channels, 32, kernel_size=5, stride=2, padding=2), act,   # -> [*, 32, H/2, W/2]
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), act,            # -> [*, 64, H/4, W/4]
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), act,            # -> [*, 64, H/8, W/8]
#         )
#         # 64x64 → 8x8 になる前提
#         self.hm_conv_out_C = 64
#         self.hm_conv_out_H = self.hm_H // 8
#         self.hm_conv_out_W = self.hm_W // 8
#         hm_flat_dim = self.hm_conv_out_C * self.hm_conv_out_H * self.hm_conv_out_W

#         hm_mlp_layers = []
#         in_dim = hm_flat_dim
#         for dim in hm_mlp_dims:
#             hm_mlp_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.hm_mlp = nn.Sequential(*hm_mlp_layers)
#         self.hm_out_dim = hm_mlp_dims[-1]
#         self.hm_norm = nn.LayerNorm(self.hm_out_dim)

#         # ===========================
#         # 3) Fused MLP (Projection Head)
#         # ===========================
#         fused_in_dim = self.prop_out_dim + self.hm_out_dim
#         fused_layers = []
#         in_dim = fused_in_dim
#         for dim in fused_mlp_dims:
#             fused_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.fused_mlp = nn.Sequential(*fused_layers)
#         self.fused_out_dim = fused_mlp_dims[-1]  # ★ これが RNN への入力次元になる

#         # ===========================
#         # 4) ActorCriticRecurrent 本体の初期化
#         #     num_actor_obs / num_critic_obs には
#         #     「エンコード後の次元」を渡すのがミソ
#         # ===========================
#         super().__init__(
#             num_actor_obs=self.fused_out_dim,
#             num_critic_obs=self.fused_out_dim,
#             num_actions=num_actions,
#             actor_hidden_dims=actor_hidden_dims,
#             critic_hidden_dims=critic_hidden_dims,
#             activation=activation,
#             rnn_type=rnn_type,
#             rnn_hidden_size=rnn_hidden_size,
#             rnn_num_layers=rnn_num_layers,
#             init_noise_std=init_noise_std,
#             **kwargs,
#         )

#     # =========================================================
#     # ヘルパ: 生観測ベクトル → 埋め込み（2D / 3D 両対応）
#     # =========================================================
#     def _encode_flat(self, obs_flat: torch.Tensor) -> torch.Tensor:
#         """
#         obs_flat: [B, raw_dim] の 2D テンソルを想定
#         戻り値: [B, fused_out_dim]
#         """
#         B, D = obs_flat.shape
#         assert D == self.prop_obs_dim + self.hm_channels * self.hm_H * self.hm_W

#         # Proprio 部分
#         prop = obs_flat[:, : self.prop_obs_dim]
#         prop_feat = self.proprio_encoder(prop)
#         prop_feat = self.prop_norm(prop_feat)

#         # Heightmap 部分
#         hm_flat = obs_flat[:, self.prop_obs_dim:]
#         hm = hm_flat.view(B, self.hm_channels, self.hm_H, self.hm_W)  # [B, C, H, W]
#         x = self.hm_encoder(hm)                    # [B, C', H', W']
#         x = x.view(B, -1)                          # [B, C'*H'*W']
#         hm_feat = self.hm_mlp(x)                   # [B, hm_out_dim]
#         hm_feat = self.hm_norm(hm_feat)

#         # 結合 → Fused MLP
#         fused = torch.cat([prop_feat, hm_feat], dim=-1)  # [B, prop_out + hm_out]
#         fused = self.fused_mlp(fused)                    # [B, fused_out_dim]
#         return fused

#     def _encode_obs_maybe_sequence(self, obs: torch.Tensor) -> torch.Tensor:
#         """
#         obs が [B, D] / [T, B, D] の両方を扱う。
#         - RNN の batch 学習時: [T, B, D]
#         - Rollout(通常時):     [B, D]
#         戻り値は obs と同じ次元数を持つ (最後の次元だけ fused_out_dim)。
#         """
#         if obs.ndim == 2:
#             # [B, D]
#             return self._encode_flat(obs)  # [B, F]
#         elif obs.ndim == 3:
#             # [T, B, D]
#             T, B, D = obs.shape
#             obs_2d = obs.view(T * B, D)              # [T*B, D]
#             fused_2d = self._encode_flat(obs_2d)     # [T*B, F]
#             return fused_2d.view(T, B, -1)           # [T, B, F]
#         else:
#             raise ValueError(f"Unexpected obs shape {obs.shape}")

#     # =========================================================
#     # ActorCriticRecurrent のインタフェースをオーバーライド
#     # =========================================================
#     def act(self, observations, masks=None, hidden_states=None):
#         """
#         - env からは「生」obs（Proprio + Heightmap のフラット）を受け取る
#         - ここでエンコードしてから ActorCriticRecurrent に渡す
#         """
#         encoded = self._encode_obs_maybe_sequence(observations)
#         return super().act(encoded, masks=masks, hidden_states=hidden_states)

#     def act_inference(self, observations):
#         """
#         推論用（hidden_states は内部で持つモード）
#         """
#         encoded = self._encode_obs_maybe_sequence(observations)
#         return super().act_inference(encoded)

#     def evaluate(self, critic_observations, masks=None, hidden_states=None):
#         """
#         Critic 用の観測も同じくエンコードしてから ActorCriticRecurrent に渡す。
#         （今回は Actor/Critic 同じ obs を想定）
#         """
#         encoded = self._encode_obs_maybe_sequence(critic_observations)
#         return super().evaluate(encoded, masks=masks, hidden_states=hidden_states)




# import torch
import torch.nn as nn
from rsl_rl.modules import ActorCriticRecurrent  # ★ここを使う

# class VisionHighRNN(ActorCriticRecurrent):
#     """
#     上位ポリシー（再帰版）:
#       - env からは dict 観測を受け取る (元の ActorCritic と同じ)
#       - このクラス内で
#           Proprio → MLP
#           Heightmap → CNN+MLP
#         で埋め込んでから、ActorCriticRecurrent の RNN に渡す
#     """

#     def __init__(
#         self,
#         obs: dict,
#         obs_groups: dict,
#         num_actions: int,
#         prop_obs_keys: list[str],
#         heightmap_key: str = "heightmap",
#         hm_shape: tuple[int, int] = (64, 64),
#         hm_channels: int = 2,  # Height + Mask なら 2。Heightだけなら 1 にする
#         prop_encoder_dims: list[int] = [256, 256],
#         hm_mlp_dims: list[int] = [256, 256],
#         projection_head_dims: list[int] = [256, 128],
#         actor_hidden_dims=[128, 256],
#         critic_hidden_dims=[128, 256],
#         activation: str = "elu",
#         use_layernorm: bool = True,
#         rnn_type="gru",
#         rnn_hidden_dim=128,
#         rnn_num_layers=1,
#         **kwargs,
#     ):
#         # --- 1) ベースクラスに渡す obs から heightmap だけ抜いておく ---
#         #     （heightmap はここで自前処理するので、ActorCriticRecurrent には渡さない）
#         # sanitized_obs = {k: v for k, v in obs.items() if k != heightmap_key}
#         # sanitized_groups = {
#         #     g: [k for k in ks if k != heightmap_key]
#         #     for g, ks in obs_groups.items()
#         # }

#         sanitized_obs = {k: v for k, v in obs.items() if k not in [heightmap_key]}
#         sanitized_groups = {g: [k for k in ks if k not in [heightmap_key]] for g, ks in obs_groups.items()}

#         # ★ ここで RNN 付き ActorCritic を初期化
#         super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, rnn_type="gru",**kwargs)
#         # super().__init__(obs=obs, obs_groups=obs_groups, num_actions=num_actions, **kwargs)

#         self.prop_obs_keys = prop_obs_keys          # Proprioに使うキー
#         self.heightmap_key = heightmap_key
#         self.hm_H, self.hm_W = hm_shape
#         self.hm_channels = hm_channels

#         act = nn.ELU() if activation == "elu" else nn.ReLU()

#         # ===== 2) Proprio Encoder =====
#         prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
#         self.prop_obs_dim = prop_obs_dim
#         prop_layers, in_dim = [], prop_obs_dim
#         for dim in prop_encoder_dims:
#             prop_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.proprioception_encoder = nn.Sequential(*prop_layers)
#         self.prop_out_dim = prop_encoder_dims[-1]
#         self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()

#         # ===== 3) Heightmap Encoder (CNN + MLP) =====
#         # 入力: [B, hm_channels, H, W]
#         self.hm_encoder = nn.Sequential(
#             nn.Conv2d(self.hm_channels, 32, kernel_size=5, stride=2, padding=2), act,  # -> [B,32,32,32]
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), act,               # -> [B,64,16,16]
#             nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), act,               # -> [B,64, 8, 8]
#         )
#         self.hm_conv_out_channels = 64
#         self.hm_conv_out_H = self.hm_H // 8
#         self.hm_conv_out_W = self.hm_W // 8

#         hm_flat_dim = self.hm_conv_out_channels * self.hm_conv_out_H * self.hm_conv_out_W
#         hm_layers, in_dim = [], hm_flat_dim
#         for dim in hm_mlp_dims:
#             hm_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.hm_mlp = nn.Sequential(*hm_layers)
#         self.hm_out_dim = hm_mlp_dims[-1]
#         self.hm_norm = nn.LayerNorm(self.hm_out_dim) if use_layernorm else nn.Identity()

#         # ===== 4) Projection head (Proprio + Heightmap 結合) =====
#         fused_dim = self.prop_out_dim + self.hm_out_dim
#         proj_layers, in_dim = [], fused_dim
#         for dim in projection_head_dims:
#             proj_layers += [nn.Linear(in_dim, dim), act]
#             in_dim = dim
#         self.projection_head = nn.Sequential(*proj_layers)
#         self.fused_out_dim = projection_head_dims[-1]
#         # ※ この fused_out_dim が実質「RNNへの入力次元」になるイメージ


#         # ==== ★RNNを上書き（再定義）====
#         # self.obs_groups = sanitized_groups
#         # num_actor_obs = 0
#         # for obs_group in obs_groups["policy"]:
#         #     assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
#         #     num_actor_obs += obs[obs_group].shape[-1]
#         # num_critic_obs = 0
#         # for obs_group in obs_groups["critic"]:
#         #     assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
#         #     num_critic_obs += obs[obs_group].shape[-1]

#         # self.memory_a = Memory(256, type="gru", num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
#         # self.memory_c = Memory(256, type="gru", num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)


#         # # self.actor  = nn.Sequential(nn.Linear(core_hidden_dim, num_actions))
#         # # self.critic = nn.Sequential(nn.Linear(core_hidden_dim, 1))
#         # self.actor = MLP(rnn_hidden_dim, 12, actor_hidden_dims, activation)
#         # self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, activation)


#     # ------------------------------------------------------------
#     # 内部エンコード用ヘルパ
#     # ------------------------------------------------------------
#     def _encode_proprio(self, obs: dict) -> torch.Tensor:
#         # Proprio: 指定キーを concat
#         prop_vec = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)  # [B, D_prop]
#         prop_feat = self.proprioception_encoder(prop_vec)                   # [B, prop_out_dim]
#         return self.prop_norm(prop_feat)

#     def _encode_heightmap(self, obs: dict) -> torch.Tensor:
#         hm_flat = obs[self.heightmap_key]           # [B, hm_channels*H*W] を想定


#         # ---- 最後の次元 = C*H*W であることを確認 ----
#         *leading_dims, feat_dim = hm_flat.shape   # 例: [traj, T, C*H*W] → leading_dims=[traj,T]
#         expected_dim = self.hm_channels * self.hm_H * self.hm_W
#         assert feat_dim == expected_dim, \
#             f"heightmap dim mismatch: got {feat_dim}, expected {expected_dim} (= {self.hm_channels}*{self.hm_H}*{self.hm_W})"

#         # ---- 先頭の軸を全部まとめてバッチにする ----
#         # 例: leading_dims=(traj,T) → B_total = traj*T
#         if len(leading_dims) == 0:
#             B_total = 1
#         else:
#             B_total = 1
#             for d in leading_dims:
#                 B_total *= d

#         # [*, C*H*W] → [B_total, C*H*W]
#         hm_2d = hm_flat.reshape(B_total, feat_dim)

#         # [B_total, C*H*W] → [B_total, C, H, W]
#         hm_img = hm_2d.view(B_total, self.hm_channels, self.hm_H, self.hm_W)

#          # Conv + MLP
#         x = self.hm_encoder(hm_img)   # [B_total, C', h', w']
#         x = x.view(B_total, -1)       # flatten
#         x = self.hm_mlp(x)            # [B_total, hm_feat_dim]
#         x = self.hm_norm(x)

#         # 元の leading_dims に戻す: [leading..., hm_feat_dim]
#         hm_feat = x.view(*leading_dims, -1) if len(leading_dims) > 0 else x

#         return hm_feat


#     # ------------------------------------------------------------
#     # ★ ActorCriticRecurrent が呼ぶフック
#     #    - ここで RNN に渡す特徴ベクトルを定義する
#     # ------------------------------------------------------------
#     def get_actor_obs(self, obs: dict) -> torch.Tensor:
#         """
#         RNN + Actor に渡す「1ステップぶんの特徴」を返す。
#         obs は dict のまま渡ってくる（ベースクラスがそう呼んでくれる前提）。
#         """
#         prop_feat = self._encode_proprio(obs)       # [B, prop_out_dim]
#         hm_feat   = self._encode_heightmap(obs)     # [B, hm_out_dim]
#         fused     = torch.cat([prop_feat, hm_feat], dim=-1)  # [B, fused_dim]
#         return self.projection_head(fused)          # [B, fused_out_dim]

#     def get_critic_obs(self, obs: dict) -> torch.Tensor:
#         # Actor / Critic で同じ特徴を使うならそのまま返す
#         return self.get_actor_obs(obs)


    



import torch
import torch.nn as nn




   




from rsl_rl.modules.actor_critic_recurrent import ActorCriticRecurrent
import torch
import torch.nn as nn







class VisionHighRNN(ActorCriticRecurrent):
    """
    上位ポリシー（再帰版）:
      - env からは dict 観測を受け取る (元の ActorCritic と同じ)
      - このクラス内で
          Proprio → MLP
          Heightmap → CNN+MLP
        で埋め込んでから、ActorCriticRecurrent の RNN に渡す
    """

    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        prop_obs_keys: list[str],
        heightmap_key: str = "heightmap",
        # hm_shape: tuple[int, int] = (64, 64),
        # hm_channels: int = 2,  # Height + Mask なら 2。Heightだけなら 1 にする
        prop_encoder_dims: list[int] = [256, 256],
        hm_mlp_dims: list[int] = [1024, 512, 256],
        projection_head_dims: list[int] = [256, 128],
        actor_hidden_dims=[128, 256],
        critic_hidden_dims=[128, 256],
        activation: str = "elu",
        use_layernorm: bool = True,
        rnn_type="gru",
        rnn_hidden_dim=128,
        rnn_num_layers=1,
        **kwargs,
    ):
        # --- 1) ベースクラスに渡す obs から heightmap だけ抜いておく ---
        #     （heightmap はここで自前処理するので、ActorCriticRecurrent には渡さない）
        # sanitized_obs = {k: v for k, v in obs.items() if k != heightmap_key}
        # sanitized_groups = {
        #     g: [k for k in ks if k != heightmap_key]
        #     for g, ks in obs_groups.items()
        # }

        sanitized_obs = {k: v for k, v in obs.items() if k not in [heightmap_key]}
        sanitized_groups = {g: [k for k in ks if k not in [heightmap_key]] for g, ks in obs_groups.items()}

        # ★ ここで RNN 付き ActorCritic を初期化
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, rnn_type="gru",**kwargs)
        # super().__init__(obs=obs, obs_groups=obs_groups, num_actions=num_actions, **kwargs)

        self.prop_obs_keys = prop_obs_keys          # Proprioに使うキー
        self.heightmap_key = heightmap_key
        # self.hm_H, self.hm_W = hm_shape
        # self.hm_channels = hm_channels

        act = nn.ELU() if activation == "elu" else nn.ReLU()

        # ===== 2) Proprio Encoder =====
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        self.prop_obs_dim = prop_obs_dim
        prop_layers, in_dim = [], prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)
        self.prop_out_dim = prop_encoder_dims[-1]
        self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()

        # ===== 3) Heightmap Encoder (CNN + MLP) =====
        # 入力: [B, hm_channels, H, W]
        # self.hm_encoder = nn.Sequential(
        #     nn.Conv2d(self.hm_channels, 32, kernel_size=5, stride=2, padding=2), act,  # -> [B,32,32,32]
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), act,               # -> [B,64,16,16]
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), act,               # -> [B,64, 8, 8]
        # )
        # self.hm_conv_out_channels = 64
        # self.hm_conv_out_H = self.hm_H // 8
        # self.hm_conv_out_W = self.hm_W // 8

        # hm_flat_dim = self.hm_conv_out_channels * self.hm_conv_out_H * self.hm_conv_out_W
        # hm_layers, in_dim = [], hm_flat_dim
        # for dim in hm_mlp_dims:
        #     hm_layers += [nn.Linear(in_dim, dim), act]
        #     in_dim = dim
        # self.hm_mlp = nn.Sequential(*hm_layers)
        # self.hm_out_dim = hm_mlp_dims[-1]
        # self.hm_norm = nn.LayerNorm(self.hm_out_dim) if use_layernorm else nn.Identity()
     




        # hm_dim = obs[self.heightmap_key].shape[1] 

        # self.hm_in_norm = nn.LayerNorm(hm_dim)

        # prop_layers, in_dim = [], hm_dim
        # for dim in hm_mlp_dims:
        #     prop_layers += [nn.Linear(in_dim, dim), act]
        #     in_dim = dim
        # self.hm_encoder = nn.Sequential(*prop_layers)
        # self.hm_out_dim = prop_encoder_dims[-1]
        # self.hm_norm = nn.LayerNorm(self.hm_out_dim) if use_layernorm else nn.Identity()




        self.hm_channels = 1
        # self.hm_H = 61
        # self.hm_W = 61

        self.hm_H = 31
        self.hm_W = 61

        # hm_dim = self.hm_channels * self.hm_H * self.hm_W  # = 3721

        hm_dim = obs[self.heightmap_key].shape[-1]  # ← shape[1] じゃなく shape[-1]


        # 入力正規化：どっちか片方でOK
        # (A) learnable LayerNorm（入力が毎回怪しいなら便利）
        self.hm_in_norm = nn.LayerNorm(hm_dim)

        # (B) すでに「hm = (hm_flat - 2)/3」みたいな固定正規化を使うなら、
        # self.hm_in_norm は Identity にしてもOK
        # self.hm_in_norm = nn.Identity()

        # ---- Conv encoder ----
        # act は既存の act をそのまま使う想定（ReLU/ELU/LeakyReLUなど）
        conv_ch1, conv_ch2, conv_ch3 = 32, 64, 64

        # self.hm_conv = nn.Sequential(
        #     nn.Conv2d(self.hm_channels, conv_ch1, kernel_size=3, stride=2, padding=1),  # 61->31
        #     act,
        #     nn.Conv2d(conv_ch1, conv_ch2, kernel_size=3, stride=2, padding=1),          # 31->16
        #     act,
        #     nn.Conv2d(conv_ch2, conv_ch3, kernel_size=3, stride=2, padding=1),          # 16->8
        #     act,
        #     # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), 
        #     # act,

        #     # 空間サイズ(8x8)は変わらず、チャネルだけ 64 -> 16 になる
        #     nn.Conv2d(64, 16, kernel_size=1), 
        #     act,
        #     # nn.AdaptiveAvgPool2d((1, 1)),                                               # -> [B,64,1,1]
        # )

    

        C = self.hm_channels
        act = act  # 既存の活性化

        # self.hm_conv = nn.Sequential(
        #     # 61 -> 31
        #     nn.Conv2d(C, 32, 3, 2, 1), act,

        #     # 31x31 （stride=1）
        #     nn.Conv2d(32, 64, 3, 1, 1), act,
        #     nn.Conv2d(64, 64, 3, 1, 1), act,

        #     # ★チャネル圧縮（64 -> 8）
        #     nn.Conv2d(64, 1, 1, 1, 0), #act,
        # )


        self.hm_conv = nn.Sequential(
            nn.Conv2d(C, 8, 3, stride=1, padding=1), act,            # 61->61
            nn.Conv2d(8, 8, 3, stride=1, padding=1, groups=8), act,  # depthwise（超軽い）
            nn.Conv2d(8, 8, 1, stride=1, padding=0), act,            # pointwise
            nn.Conv2d(8, 1, 1, stride=1, padding=0),                 # ★最後は活性化なし
        )


        # min+avg concat → (B, 16, 16, 16) → flatten 4096
        self.hm_fc = nn.Sequential(
            nn.Linear(1 * 61 * 31, 256))


        # ---- Conv出力をベクトル化して 256 次元へ ----
        hm_out_dim = 256
        # self.hm_fc = nn.Sequential(
        #     nn.Linear(conv_ch3, hm_out_dim),
        #     # ここは好み：最後は act 無しの方が安定しがち
        #     # act,
        # )

        # self.hm_fc = nn.Sequential(
        #     nn.Flatten(),                      # 64*8*8 = 4096
        #     nn.Linear(64*4*4, 256),
        # )

        # self.hm_fc = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(2048, 256), 
        # )

        self.hm_out_dim = hm_out_dim
        self.hm_norm = nn.LayerNorm(self.hm_out_dim) if use_layernorm else nn.Identity()






        # ===== 4) Projection head (Proprio + Heightmap 結合) =====
        fused_dim = self.prop_out_dim + self.hm_out_dim
        proj_layers, in_dim = [], fused_dim
        for dim in projection_head_dims:
            proj_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)
        self.fused_out_dim = projection_head_dims[-1]
        # ※ この fused_out_dim が実質「RNNへの入力次元」になるイメージ


        # ==== ★RNNを上書き（再定義）====
        # self.obs_groups = sanitized_groups
        # num_actor_obs = 0
        # for obs_group in obs_groups["policy"]:
        #     assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
        #     num_actor_obs += obs[obs_group].shape[-1]
        # num_critic_obs = 0
        # for obs_group in obs_groups["critic"]:
        #     assert len(obs[obs_group].shape) == 2, "The ActorCriticRecurrent module only supports 1D observations."
        #     num_critic_obs += obs[obs_group].shape[-1]

        # self.memory_a = Memory(256, type="gru", num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        # self.memory_c = Memory(256, type="gru", num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)


        # # self.actor  = nn.Sequential(nn.Linear(core_hidden_dim, num_actions))
        # # self.critic = nn.Sequential(nn.Linear(core_hidden_dim, 1))
        # self.actor = MLP(rnn_hidden_dim, 12, actor_hidden_dims, activation)
        # self.critic = MLP(rnn_hidden_dim, 1, critic_hidden_dims, activation)


    # ------------------------------------------------------------
    # 内部エンコード用ヘルパ
    # ------------------------------------------------------------
    def _encode_proprio(self, obs: dict) -> torch.Tensor:
        # Proprio: 指定キーを concat
        prop_vec = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)  # [B, D_prop]
        prop_feat = self.proprioception_encoder(prop_vec)                   # [B, prop_out_dim]
        return self.prop_norm(prop_feat)

    # def _encode_heightmap(self, obs: dict) -> torch.Tensor:
    #     hm_flat = obs[self.heightmap_key]           # [B, hm_channels*H*W] を想定


    #     # ---- 最後の次元 = C*H*W であることを確認 ----
    #     # *leading_dims, feat_dim = hm_flat.shape   # 例: [traj, T, C*H*W] → leading_dims=[traj,T]
    #     # expected_dim = self.hm_channels * self.hm_H * self.hm_W
    #     # assert feat_dim == expected_dim, \
    #     #     f"heightmap dim mismatch: got {feat_dim}, expected {expected_dim} (= {self.hm_channels}*{self.hm_H}*{self.hm_W})"

    #     # # ---- 先頭の軸を全部まとめてバッチにする ----
    #     # # 例: leading_dims=(traj,T) → B_total = traj*T
    #     # if len(leading_dims) == 0:
    #     #     B_total = 1
    #     # else:
    #     #     B_total = 1
    #     #     for d in leading_dims:
    #     #         B_total *= d

    #     # # [*, C*H*W] → [B_total, C*H*W]
    #     # hm_2d = hm_flat.reshape(B_total, feat_dim)

    #     # # [B_total, C*H*W] → [B_total, C, H, W]
    #     # hm_img = hm_2d.view(B_total, self.hm_channels, self.hm_H, self.hm_W)

    #     #  # Conv + MLP
    #     # x = self.hm_encoder(hm_img)   # [B_total, C', h', w']
    #     # x = x.view(B_total, -1)       # flatten

    #     # x = self.hm_mlp(x)            # [B_total, hm_feat_dim]
    #     # hm_flat = self.hm_in_norm(hm_flat)
    #     hm = (hm_flat - 2.0) / 3.0
    #     x = self.hm_encoder(hm) 
    #     x = self.hm_norm(x)

    #     # 元の leading_dims に戻す: [leading..., hm_feat_dim]
    #     # hm_feat = x.view(*leading_dims, -1) if len(leading_dims) > 0 else x

    #     return x


    def _encode_heightmap(self, obs: dict) -> torch.Tensor:
        hm = obs[self.heightmap_key]  # shape: [B,3721] or [T,B,3721] or [B,T,3721] etc.

        feat_dim = hm.shape[-1]
        leading = hm.shape[:-1]       # 先頭の次元全部（例: (T,B)）

        # 期待するdimチェック（ここ重要）
        expected = self.hm_channels * self.hm_H * self.hm_W
        assert feat_dim == expected, f"hm dim mismatch: got {feat_dim}, expected {expected}"

        # [*, feat] -> [N, feat]
        hm_2d = hm.reshape(-1, feat_dim)

        # ---- 入力正規化（どっちか片方）----
        # hm_2d = self.hm_in_norm(hm_2d)          # LayerNorm(3721)
        # もしくは固定スケールなら:
        # hm_2d = (hm_2d - 2.0) / 3.0

        # [N, feat] -> [N, C, H, W]
        hm_img = hm_2d.view(-1, self.hm_channels, self.hm_H, self.hm_W)

        # Conv -> [N, conv_ch, 1, 1]
        x = self.hm_conv(hm_img)
        # x = x.flatten(1)             # [N, conv_ch]

        # x = -torch.nn.functional.adaptive_max_pool2d(-x, (24,24))  # min相当
        # x_avg = torch.nn.functional.adaptive_avg_pool2d(x, (24,24))

        # zmin = -F.max_pool2d(-x, 3, 1, 1)      # 低い部分を太らせる（min-pool）
        # x = -F.adaptive_max_pool2d(-x, (48,48))  # ビン内最小
        # x = -F.adaptive_max_pool2d(-x, (32, 64))  # (B,1,32,48)


        # x = torch.cat([x_min, x_avg], dim=1)  # (B,32,8,8)
        x = x.flatten(1)


        x = self.hm_fc(x)            # [N, hm_out_dim]
        x = self.hm_norm(x)

        # [N, hm_out_dim] -> [leading..., hm_out_dim]
        x = x.view(*leading, -1)
        return x




    # ------------------------------------------------------------
    # ★ ActorCriticRecurrent が呼ぶフック
    #    - ここで RNN に渡す特徴ベクトルを定義する
    # ------------------------------------------------------------
    def get_actor_obs(self, obs: dict) -> torch.Tensor:
        """
        RNN + Actor に渡す「1ステップぶんの特徴」を返す。
        obs は dict のまま渡ってくる（ベースクラスがそう呼んでくれる前提）。
        """
        prop_feat = self._encode_proprio(obs)       # [B, prop_out_dim]
        hm_feat   = self._encode_heightmap(obs)     # [B, hm_out_dim]
        fused     = torch.cat([prop_feat, hm_feat], dim=-1)  # [B, fused_dim]
        return self.projection_head(fused)          # [B, fused_out_dim]

    def get_critic_obs(self, obs: dict) -> torch.Tensor:
        # Actor / Critic で同じ特徴を使うならそのまま返す
        return self.get_actor_obs(obs)