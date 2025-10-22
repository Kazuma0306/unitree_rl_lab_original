
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



class LocoTransformerHFP(ActorCritic):
    """
    Heightmap + Force/Torque + Proprio を Transformer で統合するActor-Critic
    - Heightmap: 2D Conv stem + AdaptiveAvgPool2d(4x4) -> 16トークン
    - Force/Torque: 各脚の時系列を1D-Conv/TCNで要約 -> 脚トークン(4)
    - Proprio: MLP -> 1トークン
    - 最終: 1 + 4 + 16 トークンを Transformer で融合、Actor/Critic へ
    """
    def __init__(
        self,
        # runner から
        obs: dict,
        obs_groups: dict,
        num_actions: int,

        # --- Cfgから ---
        prop_obs_keys: list[str],
        heightmap_key: str,                 # e.g. "heightmap"
        height_shape: tuple = (32, 32),     # RayCaster: size=[1.6,1.6], res=0.1 -> 16x16
        height_channels: int = 1,           # 1: height, 2: height+valid など
        ft_stack_key: str = "ft_stack",     # [B,K,4,6]
        ft_in_dim: int = 3,                 # Fx,Fy,Fz
        transformer_hidden_dim: int = 256,
        transformer_n_heads: int = 4,
        transformer_num_layers: int = 2,
        prop_encoder_dims: list[int] = [256, 256],
        projection_head_dims: list[int] = [256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        # 観測からheight/ftを親に渡さない
        sanitized_obs = {k: v for k, v in obs.items() if k not in [heightmap_key, ft_stack_key]}
        sanitized_groups = {g: [k for k in ks if k not in [heightmap_key, ft_stack_key]] for g, ks in obs_groups.items()}
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, **kwargs)

        # 保存
        self.prop_obs_keys   = prop_obs_keys
        self.heightmap_key   = heightmap_key
        self.height_shape    = height_shape     # (Hh, Wh)
        self.height_channels = height_channels
        self.ft_stack_key    = ft_stack_key
        self.ft_in_dim       = ft_in_dim

        # 活性化
        self.activation = nn.ELU() if activation == "elu" else nn.ReLU()

        # ===== 1) Proprio encoder =====
        prop_obs_dim = sum(obs[k].shape[1] for k in self.prop_obs_keys)
        prop_layers, in_dim = [], prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers += [nn.Linear(in_dim, dim), nn.ELU()]
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)
        self.prop_proj = nn.Linear(prop_encoder_dims[-1], transformer_hidden_dim)

        # ===== 2) Heightmap encoder（学習可能ダウンサンプリング；情報保持優先） =====
        def make_hm_stem(cin):
            return nn.Sequential(
                nn.Conv2d(cin, 32, 3, 1, 1), self.activation,   # 形を保ったまま局所特徴
                nn.Conv2d(32, 64, 3, 2, 1), self.activation,    # /2
                nn.Conv2d(64, 96, 3, 2, 1), self.activation,    # /4
                nn.Conv2d(96, 128, 3, 1, 1), self.activation,   # 出力ch固定
                nn.AdaptiveAvgPool2d((4, 4)),                   # ★常に4x4へ
            )
        self.height_encoder = make_hm_stem(self.height_channels)
        self.height_proj    = nn.Linear(128, transformer_hidden_dim)
        self.height_pos_embedding = nn.Parameter(torch.randn(1, 16, transformer_hidden_dim))  # 4*4=16

        # ===== 3) Force/Torque encoder（脚×時系列 → 脚トークン4つ） =====
        # Kは可変でOK（Conv1dは可変長を受けられる）。最後をAvgPool1d(1)で要約。
        self.ft_encoder = nn.Sequential(
            nn.Conv1d(self.ft_in_dim, 64, 3, padding=1), self.activation,
            nn.Conv1d(64, 128, 3, padding=2, dilation=2), self.activation,   # 受容野拡大(TCN風)
            nn.Conv1d(128, 128, 3, padding=4, dilation=4), self.activation,
            nn.AdaptiveAvgPool1d(1),   # -> [B*4, 128, 1]
        )
        self.ft_proj = nn.Linear(128, transformer_hidden_dim)
        self.leg_pos_embedding = nn.Parameter(torch.randn(1, 4, transformer_hidden_dim))  # 4脚

        # ===== 4) Transformer =====
        enc_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim, nhead=transformer_n_heads,
            dim_feedforward=512, activation="gelu", batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=transformer_num_layers)
        self.prop_pos_embedding  = nn.Parameter(torch.randn(1, 1, transformer_hidden_dim))

        # ===== 5) 出力ヘッド（Actor/Critic） =====
        fused_feature_dim = transformer_hidden_dim * 3
        proj_layers, in_dim = [], fused_feature_dim
        for dim in projection_head_dims:
            proj_layers += [nn.Linear(in_dim, dim), nn.ELU()]
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)
        self.actor  = nn.Sequential(nn.Linear(projection_head_dims[-1], num_actions))
        self.critic = nn.Sequential(nn.Linear(projection_head_dims[-1], 1))

        self.force_pool = AttnPool(transformer_hidden_dim)
        self.hmap_pool  = AttnPool(transformer_hidden_dim)

    # ---------- 前処理 ----------
    def _prep_heightmap(self, hm_flat: torch.Tensor) -> torch.Tensor:
        """
        hm_flat: [B, C*H*W] or [B, H*W] when C=1
        return:  [B, C, H, W]
        """
        B = hm_flat.shape[0]
        Hh, Wh = self.height_shape
        C = self.height_channels
        if hm_flat.shape[1] == Hh * Wh and C == 1:
            hm = hm_flat.view(B, 1, Hh, Wh)
        else:
            assert hm_flat.shape[1] == C * Hh * Wh, \
                f"heightmap length mismatch: got {hm_flat.shape[1]}, want {C*Hh*Wh}"
            hm = hm_flat.view(B, C, Hh, Wh)
        return torch.nan_to_num(hm, 0.0, 0.0, 0.0)

    def _prep_ft(self, ft_stack: torch.Tensor) -> torch.Tensor:
        """
        ft_stack: [B, K, 4, D] -> [B*4, D, K]
        """
        B, K, L, D = ft_stack.shape
        return ft_stack.permute(0, 2, 3, 1).contiguous().view(B*L, D, K)


    # ---------- 前向き ----------
    def get_actor_obs(self, obs):
        # Proprio -> 1トークン
        prop_vec  = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        prop_feat = self.proprioception_encoder(prop_vec)
        prop_tok  = self.prop_proj(prop_feat).unsqueeze(1) + self.prop_pos_embedding  # [B,1,D]

        # Heightmap -> 16トークン
        hm_flat   = obs[self.heightmap_key]                 # [B, C*H*W] or [B, H*W]
        hm        = self._prep_heightmap(hm_flat)           # [B, C, H, W]
        fh        = self.height_encoder(hm)                 # [B, 128, 4, 4]
        h_tokens  = fh.flatten(2).permute(0, 2, 1)          # [B, 16, 128]
        h_tokens  = self.height_proj(h_tokens)              # [B, 16, D]
        h_tokens  = h_tokens + self.height_pos_embedding

        # Force/Torque -> 脚トークン(4)
        ft_stack  = obs[self.ft_stack_key]                  # [B, K, 4, D]
        ft_seq    = self._prep_ft(ft_stack)                 # [B*4, D, K]
        ff        = self.ft_encoder(ft_seq).squeeze(-1)     # [B*4, 128]
        # ff        = self.ft_proj(ff).view(ft_stack.shape[0], 4, -1) + self.leg_pos_embedding  # [B,4,D]
        ff = self.ft_proj(ff)                     # [B*L, D]

        B = ft_stack.shape[0]
        L = ft_stack.shape[2] if ft_stack.dim() >= 3 else 4  # 安全
        D = self.ft_proj.out_features            # or self.D

        # ★ B=0でも安全な明示リシェイプ
        ff = ff.view(B, L, D)                    # [B, L, D]
        # pos-emb は L に合わせてスライス
        ff = ff + self.leg_pos_embedding[:, :L, :]   # [1, L, D] を想定

        # すべて結合 → Transformer
        all_tokens = torch.cat([prop_tok, ff, h_tokens], dim=1)  # [B, 1+4+16, D]
        
        fused       = self.transformer_encoder(all_tokens)      # [B, 1+4+16, D]
        prop_tok    = fused[:, 0, :]                            # [B,D]
        force_tok   = fused[:, 1:1+4, :]                        # [B,4,D]
        hmap_tok    = fused[:, 1+4:, :]                         # [B,16,D]

        force_feat  = self.force_pool(force_tok)                # [B,D]
        hmap_feat   = self.hmap_pool(hmap_tok)                  # [B,D]

        feat        = torch.cat([prop_tok, force_feat, hmap_feat], dim=1)  # [B, 3D]
        
        return self.projection_head(feat)

    def get_critic_obs(self, obs):
        return self.get_actor_obs(obs)