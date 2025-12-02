
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from torch.distributions import Normal # <-- Add this line
import torchvision



from rsl_rl.modules import MLP 


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

class MlpHFP(ActorCritic):
    """
    Proprio + Force/Torque（時系列は時間平均）を単純にConcatして
    MLPでActor/Criticに入れるベースライン。
    - ft_stack: [B, K, 4, D_in]  (K=time, legs=4, D_in>=3)
    - proprioは prop_obs_keys をそのまま連結
    """
    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        prop_obs_keys: list[str],
        ft_stack_key: str = "ft_stack",
        ft_in_dim: int = 3,                 # 例: Fx,Fy,Fz のみ使用
        ft_time_tokens: int = 3,            # Conv後の時間トークン数(K->この数に集約)
        prop_encoder_dims: list[int] = [256, 256],
        projection_head_dims: list[int] = [256, 256],
        ft_feat_dim: int = 128,             # Conv出力チャンネル
        activation: str = "elu",
        init_noise_std: float = 1.0,
        use_layernorm: bool = True,
        **kwargs,
    ):
        # 親へはFTを渡さない（自前で処理）
        ignore_keys = [ft_stack_key]
        sanitized_obs = {k: v for k, v in obs.items() if k not in ignore_keys}
        sanitized_groups = {g: [k for k in ks if k not in ignore_keys] for g, ks in obs_groups.items()}
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

        # 保存
        self.prop_obs_keys = prop_obs_keys
        self.ft_stack_key  = ft_stack_key
        self.ft_in_dim     = ft_in_dim
        self.num_ft_time_tokens = ft_time_tokens
        self.num_legs = 4

        act = nn.ELU() if activation == "elu" else nn.ReLU()

        # ===== Proprio encoder =====
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        prop_layers, in_dim = [], prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)
        self.prop_out_dim = prop_encoder_dims[-1]
        self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()

        # ===== Force/Torque encoder (脚×時系列→脚ベクトル) =====
        # Conv1d は [B*L, C_in, K] を受け取る
        self.ft_encoder = nn.Sequential(
            nn.Conv1d(self.ft_in_dim, 64, 3, padding=1), act,
            nn.Conv1d(64, ft_feat_dim, 3, padding=2, dilation=2), act,
            nn.AdaptiveAvgPool1d(self.num_ft_time_tokens),     # -> [B*L, ft_feat_dim, T]
        )
        # self.ft_encoder = nn.Sequential( # [B*4, 3,  12]
        #     # Kernel=3, Padding=1 なら、入力12 -> 出力12 
        #     nn.Conv1d(self.ft_in_dim, ft_feat_dim, kernel_size=3, padding=1),
        #     act # お好みで活性化関数
        # )
        self.ft_feat_dim = ft_feat_dim
        # 時間平均（必要なら注意に差し替え可）
        self.ft_out_norm = nn.LayerNorm(self.num_legs * self.ft_feat_dim) if use_layernorm else nn.Identity()

        # ===== Projection head / Actor-Critic =====
        fused_feature_dim = self.prop_out_dim + self.num_legs * self.ft_feat_dim
        proj, in_dim = [], fused_feature_dim
        for dim in projection_head_dims:
            proj += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj)
        self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
        self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

    def _prep_ft(self, ft_stack: torch.Tensor) -> torch.Tensor:
        """
        ft_stack: [B, K, 4, D_all] -> [B*4, C_in, K] （C_in=self.ft_in_dim）
        ※ D_all>=C_in を想定。Fx,Fy,Fz が先頭3でない場合は注意
        """
        B, K, L, Dall = ft_stack.shape
        assert L == self.num_legs, f"legs={L} != {self.num_legs}"
        if Dall < self.ft_in_dim:
            raise ValueError(f"ft_in_dim={self.ft_in_dim} but input has {Dall} channels")
        # 先頭3がFx,Fy,Fzでない場合は、環境に合わせて明示インデックスを切る:
        # idx_fx, idx_fy, idx_fz = ...
        ft_stack = ft_stack[..., :self.ft_in_dim]  # [B,K,4,C_in]
        return ft_stack.permute(0, 2, 3, 1).contiguous().view(B * L, self.ft_in_dim, K)

    def get_actor_obs(self, obs):
        # ---- Proprio ----
        prop_vec  = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        prop_feat = self.prop_norm(self.proprioception_encoder(prop_vec))  # [B, P]

        # ---- Force/Torque ----
        ft_stack = obs[self.ft_stack_key]                  # [B, K, 4, D_in>=3]
        B, K, L, _ = ft_stack.shape
        x = self._prep_ft(ft_stack)                        # [B*L, C_in, K]
        x = self.ft_encoder(x)                             # [B*L, F, T]
        x = x.permute(0, 2, 1).contiguous()                # [B*L, T, F]
        x = x.mean(dim=1)                                  # 時間平均 → [B*L, F]
        x = x.view(B, L * self.ft_feat_dim)                # 脚を連結 → [B, L*F]
        x = self.ft_out_norm(x)

        feat = torch.cat([prop_feat, x], dim=-1)           # [B, fused_feature_dim]

        # feat = prop_feat
        return self.projection_head(feat)                  # [B, hidden]

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)







class VisionHighLevelAC(ActorCritic):
    """
    上位ポリシー:
    - 観測:
        - Heightmap:   env 側の ObservationTerm で生成した 2D 高さマップ
        - Proprio:     prop_obs_keys で指定したベクトル観測（下位と似た構成）
    - モデル:
        - Heightmap → CNN で z_img
        - Proprio  → MLP で z_prop
        - [z_img, z_prop] → Projection head → Actor / Critic
    """
    def __init__(
        self,
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        prop_obs_keys: list[str],
        heightmap_key: str = "heightmap",
        ft_stack_key: str = "ft_stack",
        hm_shape: tuple[int, int] = (64, 64),   # Heightmap の (H,W)
        # hm_shape: tuple[int, int] = (32, 32),
        prop_encoder_dims: list[int] = [256, 128],
        projection_head_dims: list[int] = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        use_layernorm: bool = True,
        **kwargs,
    ):
        # Heightmap は自前で処理するので、親には渡すがそのままでもOK（無視しても良い）
        # 観測からheight/ftを親に渡さない
        sanitized_obs = {k: v for k, v in obs.items() if k not in [heightmap_key, ft_stack_key]}
        sanitized_groups = {g: [k for k in ks if k not in [heightmap_key]] for g, ks in obs_groups.items()}

        # print("=== env.cfg.observations.policy.term_cfgs ===")
        # print(env.cfg.observations.policy.term_cfgs.keys())

        # # 2) 実際に ActorCritic に渡ってきた obs のキー
        # print("=== obs keys passed to ActorCritic ===")
        # print(list(obs.keys()))
        # for k, v in obs.items():
        #     print(k, v.shape)

        # print("=== sanitized_obs shapes ===")
        # for k, v in sanitized_obs.items():
        #     print(k, v.shape, "ndim=", len(v.shape))

        #     # デバッグ用: どれが3次元/4次元か強制チェック
        #     assert len(v.shape) == 2, f"BAD OBS SHAPE: key={k}, shape={v.shape}"
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

        # super().__init__(obs=obs, obs_groups=obs_groups, num_actions=num_actions,
        #                  init_noise_std=init_noise_std, **kwargs)

       


        self.prop_obs_keys = prop_obs_keys
        self.heightmap_key = heightmap_key
        self.hm_H, self.hm_W = hm_shape

        act = nn.ELU() if activation == "elu" else nn.ReLU()

        # ===== Proprio encoder =====
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        prop_layers, in_dim = [], prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)
        self.prop_out_dim = prop_encoder_dims[-1]
        self.prop_norm = nn.LayerNorm(self.prop_out_dim) if use_layernorm else nn.Identity()

        # ===== Heightmap encoder (2D Conv) =====
        # 入力: [B, 1, H, W]
        # self.hm_encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2), act,   # -> [B,16,H/2,W/2]
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), act,  # -> [B,32,H/4,W/4]
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), act,  # -> [B,64,H/8,W/8]
        # )

        # self.hm_encoder = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), act,   # 1→32
        #     nn.Conv2d(32,64, kernel_size=3, stride=2, padding=1), act,   # 32→64
        #     nn.Conv2d(64,128,kernel_size=3, stride=2, padding=1), act,   # 64→128
        # )

        self.hm_encoder = nn.Sequential(
            # 入力: [B, 1, 64, 64]
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # -> [B,32,32,32]
            act,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> [B,64, 16, 16]
            act,
            # 必要ならもう1段（32x32ならここで止めてもOK）
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), act,
        )
        # self.hm_feat_dim = 64   # 最終 Conv のチャンネル数
        self.hm_conv_out_channels = 64
        # self.hm_feat_dim = 128
        self.hm_conv_out_H = 8
        self.hm_conv_out_W = 8

        self.hm_mlp = nn.Sequential(
            nn.Linear(self.hm_conv_out_channels * self.hm_conv_out_H * self.hm_conv_out_W, 256),
            act,
            nn.Linear(256, 128),
            act,
        )

        # self.hm_global_pool = lambda x: x.mean(dim=[2, 3])
        self.hm_feat_dim = 128
        self.hm_norm = nn.LayerNorm(self.hm_feat_dim) if use_layernorm else nn.Identity()

        # ===== Projection head / Actor-Critic =====
        fused_feature_dim = self.prop_out_dim + self.hm_feat_dim
        proj, in_dim = [], fused_feature_dim
        for dim in projection_head_dims:
            proj += [nn.Linear(in_dim, dim), act]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj)

        self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
        self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

    # ------------------------------------------------------------------
    #  観測のエンコード
    # ------------------------------------------------------------------
    def _encode_proprio(self, obs):
        # 下位と同様、指定された key を concat。
        # 例: ["base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", ...]
        prop_vec  = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        prop_feat = self.proprioception_encoder(prop_vec)       # [B, P]
        return self.prop_norm(prop_feat)

    def _encode_heightmap(self, obs):
        # ObservationTerm 側で heightmap をフラット (2H*W) で返している想定。
        # 例: depth_heightmap() が (N,H,W,1) → (N,H*W) で返している場合、
        #     ここで (B,1,H,W) に reshape して Conv に入れる。
        hm_flat = obs[self.heightmap_key]        # [B, H*W] or [B, H, W]
        if hm_flat.ndim == 2:
            B = hm_flat.shape[0]
            hm = hm_flat.view(B, 1, self.hm_H, self.hm_W)  # [B,1,H,W]
        elif hm_flat.ndim == 3:
            # すでに [B,H,W] の場合2
            B = hm_flat.shape[0]
            hm = hm_flat.view(B, 1, self.hm_H, self.hm_W)
        else:
            raise ValueError(f"Unexpected heightmap shape: {hm_flat.shape}")

        x = self.hm_encoder(hm)                  # [B,64,h',w']
        # x = self.hm_global_pool(x)               # [B,64]

        x = x.view(B, -1)                # flatten → [B, C*H'*W']
        x = self.hm_mlp(x)                         # [B,128]


        return self.hm_norm(x)

    # ------------------------------------------------------------------
    #  Actor / Critic 入力
    # ------------------------------------------------------------------
    def get_actor_obs(self, obs):
        prop_feat = self._encode_proprio(obs)    # [B, P]
        hm_feat   = self._encode_heightmap(obs)  # [B, 64]
        feat = torch.cat([prop_feat, hm_feat], dim=-1)  # [B, fused_feature_dim]　128＋128=256
        return self.projection_head(feat)        # [B, hidden]

    def get_critic_obs(self, obs):
        # 今回は Actor/Critic 同じ特徴を使う（MlpHFP と同じポリシー）
        return self.get_actor_obs(obs)