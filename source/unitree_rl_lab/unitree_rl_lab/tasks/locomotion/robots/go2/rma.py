# ---- Phase 1: PPO で π, μ を学習（rsl-rl PPOをそのまま使用）
# rollout時: z = mu(et) を特権観測から計算して policy obs に結合

# ---- Phase 2: φ を教師あり
# rollout時: z_hat = phi(hist(x,a)), a = pi([x,a_prev,z_hat])
# バッファに (z_hat_target=z=mu(et)) を保存 → MSE 学習



# rma_actor_critic.py
from __future__ import annotations
from typing import List, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

# rsl_rl の ActorCritic をインポート（あなたの環境に合わせてパス調整）
from rsl_rl.modules import ActorCritic


# ---------- 汎用MLP ----------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], out_dim: int, layernorm=True, final=None):
        super().__init__()
        dims = (in_dim, *hidden, out_dim)
        layers = []
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1])]
            if layernorm:
                layers += [nn.LayerNorm(dims[i+1])]
            layers += [nn.SiLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        if final is not None:
            layers += [final]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)


# ---------- μ: 特権情報 et -> z ----------
class EnvEncoderMu(nn.Module):
    def __init__(self, et_dim=17, z_dim=8, hidden: Tuple[int, ...]=(256,128), tanh_bound=True):
        super().__init__()
        self.tanh_bound = tanh_bound
        self.mlp = MLP(et_dim, hidden, z_dim, layernorm=True)
    def forward(self, et: torch.Tensor) -> torch.Tensor:
        z = self.mlp(et)
        return torch.tanh(z) if self.tanh_bound else z




# ---------- φ: 履歴 (x_hist, a_hist) -> z_hat ----------
class AdapterPhi(nn.Module):
    def __init__(self, x_dim=30, a_dim=12, z_dim=8, T=50, step_embed_dim=32, conv_channels=32):
        super().__init__()
        self.x_dim, self.a_dim, self.T = x_dim, a_dim, T
        in_step = x_dim + a_dim
        self.step_emb = MLP(in_step, (64,), step_embed_dim, layernorm=True)
        C = conv_channels
        self.temporal = nn.Sequential(
            nn.Conv1d(step_embed_dim, C, kernel_size=8, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(C, C, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.Conv1d(C, C, kernel_size=5, stride=1, padding=2), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Flatten(), nn.LayerNorm(C), nn.Linear(C, 64), nn.SiLU(), nn.Linear(64, z_dim))
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=0.2); nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2); nn.init.zeros_(m.bias)

    def forward(self, x_hist: torch.Tensor, a_hist: torch.Tensor) -> torch.Tensor:
        # x_hist, a_hist: [B, T, *]
        assert x_hist.shape[1] == self.T and a_hist.shape[1] == self.T, "history length mismatch"
        h = torch.cat([x_hist, a_hist], dim=-1)           # [B,T,x+a]
        B, T, D = h.shape
        e = self.step_emb(h.view(B*T, D)).view(B, T, -1)  # [B,T,E]
        t = self.temporal(e.transpose(1, 2).contiguous()) # [B,C,1]
        z_hat = self.proj(t)                               # [B,z]
        return torch.tanh(z_hat)




class RMAActorCritic(ActorCritic):
    """
    RMA: Actor-Critic with μ(et) or φ(history) to provide extrinsics z to policy.
    - Phase1 (train π+μ):   use_phi=False -> z = μ(et)
    - Phase2 (train φ / run): use_phi=True  -> z = φ(x_hist, a_hist)
    """
    def __init__(
        self,
        # --- rsl-rl base ---
        obs: Dict[str, torch.Tensor],
        obs_groups: Dict[str, List[str]],
        num_actions: int,

        # --- keys / dims ---
        prop_obs_keys: List[str],   # e.g., ["j_pos", "j_vel", "base_rp", "feet_contact"]
        a_prev_key: str,            # e.g., "act_prev"
        et_key: str | None = None,  # e.g., "obs_priv" (Phase1で使用)
        x_hist_key: str | None = None,  # e.g., "x_hist"  [B,T,x_dim]
        a_hist_key: str | None = None,  # e.g., "a_hist"  [B,T,a_dim]

        # --- switches ---
        use_phi: bool = False,      # False->μ, True->φ
        x_dim: int = 30,
        a_dim: int = 12,
        z_dim: int = 8,

        # --- nets ---
        actor_hidden: Tuple[int, ...] = (256,256,128),
        critic_hidden: Tuple[int, ...] = (256,256,128),
        mu_hidden: Tuple[int, ...] = (256,128),
        phi_T: int = 50,
        phi_step_embed: int = 32,
        phi_channels: int = 32,

        # --- PPO ---
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        # μ/φ に渡す観測は親に渡さない（ベースは使わないため）
        ignore_keys = [k for k in [et_key, x_hist_key, a_hist_key] if k]
        sanitized_obs = {k: v for k, v in obs.items() if k not in ignore_keys}
        sanitized_groups = {g: [k for k in ks if k not in ignore_keys] for g, ks in obs_groups.items()}
        super().__init__(obs=sanitized_obs, obs_groups=sanitized_groups, num_actions=num_actions, init_noise_std=init_noise_std, **kwargs)

        # 保存
        self.prop_obs_keys = prop_obs_keys
        self.a_prev_key    = a_prev_key
        self.et_key        = et_key
        self.x_hist_key    = x_hist_key
        self.a_hist_key    = a_hist_key
        self.use_phi       = use_phi

        self.x_dim, self.a_dim, self.z_dim = x_dim, a_dim, z_dim

        # μ / φ
        self.mu  = EnvEncoderMu(et_dim=obs[et_key].shape[1] if et_key else 17, z_dim=z_dim, hidden=mu_hidden) if et_key else None
        self.phi = AdapterPhi(x_dim=x_dim, a_dim=a_dim, z_dim=z_dim, T=phi_T,
                              step_embed_dim=phi_step_embed, conv_channels=phi_channels) if (x_hist_key and a_hist_key) else None

        # --- Policy/Critic heads (obs = [prop, a_prev, z]) ---
        policy_in = self._prop_dim(obs, prop_obs_keys) + obs[a_prev_key].shape[1] + z_dim
        self.actor_mlp  = MLP(policy_in, actor_hidden, num_actions, layernorm=True)
        self.critic_mlp = MLP(policy_in, critic_hidden, 1, layernorm=True)

        # 学習可能な状態非依存 log_std（rsl-rl の buffer と互換）
        self.log_std = nn.Parameter(torch.ones(num_actions) * -0.7)

    # ===== 必須: 親クラスが呼ぶ =====
    def get_actor_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        prop = torch.cat([obs[k] for k in self.prop_obs_keys], dim=-1)
        a_prev = obs[self.a_prev_key]
        z = self._z_from_mu_or_phi(obs)
        return torch.cat([prop, a_prev, z], dim=-1)

    def get_critic_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 同じ入力でOK（共有 trunk）
        return self.get_actor_obs(obs)

    # ===== rsl-rl 互換API =====
    def act(self, obs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.get_actor_obs(obs)
        mean = self.actor_mlp(x)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(-1)

    def evaluate(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.get_actor_obs(obs)
        mean = self.actor_mlp(x)
        value = self.critic_mlp(x).squeeze(-1)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return value, log_prob, entropy

    # ===== 内部ヘルパ =====
    def _z_from_mu_or_phi(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.use_phi:
            assert self.phi is not None, "use_phi=True ですが x_hist/a_hist が未設定です"
            x_hist = obs[self.x_hist_key]  # [B,T,x_dim]
            a_hist = obs[self.a_hist_key]  # [B,T,a_dim]
            return self.phi(x_hist, a_hist)
        else:
            assert self.mu is not None and self.et_key is not None, "use_phi=False ですが et_key/mu が未設定です"
            et = obs[self.et_key]         # [B,et_dim]
            with torch.no_grad():         # μの出力は勾配不要（RMA Phase1でもOK）
                return self.mu(et)

    @staticmethod
    def _prop_dim(obs: Dict[str, torch.Tensor], keys: List[str]) -> int:
        return sum(obs[k].shape[1] for k in keys)


# ===== 動作テスト（ダミー） =====
if __name__ == "__main__":
    B, T = 32, 50
    x_dim, a_dim, z_dim = 30, 12, 8
    dummy_obs = {
        "j_pos": torch.randn(B, 12),
        "j_vel": torch.randn(B, 12),
        "base_rp": torch.randn(B, 2),
        "feet_contact": torch.randint(0, 2, (B, 4)).float(),
        "act_prev": torch.randn(B, a_dim),
        "obs_priv": torch.randn(B, 17),
        "x_hist": torch.randn(B, T, x_dim),
        "a_hist": torch.randn(B, T, a_dim),
    }
    groups = {"proprio": ["j_pos","j_vel","base_rp","feet_contact","act_prev"]}

    # Phase1: μを使う
    ac1 = RMAActorCritic(
        obs=dummy_obs, obs_groups=groups, num_actions=a_dim,
        prop_obs_keys=["j_pos","j_vel","base_rp","feet_contact"],
        a_prev_key="act_prev", et_key="obs_priv",
        use_phi=False, x_dim=x_dim, a_dim=a_dim, z_dim=z_dim,
    )
    a, lp = ac1.act(dummy_obs); v, lp2, ent = ac1.evaluate(dummy_obs, a)
    print("[Phase1]", a.shape, lp.shape, v.shape, ent.shape)

    # Phase2: φを使う
    ac2 = RMAActorCritic(
        obs=dummy_obs, obs_groups=groups, num_actions=a_dim,
        prop_obs_keys=["j_pos","j_vel","base_rp","feet_contact"],
        a_prev_key="act_prev", x_hist_key="x_hist", a_hist_key="a_hist",
        use_phi=True, x_dim=x_dim, a_dim=a_dim, z_dim=z_dim,
    )
    a, lp = ac2.act(dummy_obs); v, lp2, ent = ac2.evaluate(dummy_obs, a)
    print("[Phase2]", a.shape, lp.shape, v.shape, ent.shape)