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





class VisionMLPActorCritic(ActorCritic):
    """
    視覚情報と身体感覚を扱う、シンプルなMLPベースのActor-Criticモデル。
    - 視覚: ConvNetで特徴量を抽出
    - 身体感覚: MLPで特徴量を抽出
    - 融合: 抽出された特徴量を結合(Concatenate)
    - 本体: 結合された特徴量をMLPで処理
    """
    def __init__(
        self,
        # from runner 
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        # --- Cfgから「名前」で渡される引数をその後に書く ---
        prop_obs_keys: list[str],
        vision_obs_key: str,
        prop_encoder_dims: list[int] = [256, 128],
        vision_encoder_channels: list[int] = [32, 64, 64],
        shared_mlp_dims: list[int] = [256, 128],
        init_noise_std: float = 1.0,
        activation: str = "elu",
        **kwargs,
    ):
        # 1. 親クラスに渡すための「無害化」された観測辞書を作成します。
        #    オリジナルのobs辞書から、問題となる画像データをキーごと取り除きます。
        sanitized_obs = {
            key: value for key, value in obs.items()
            if key != vision_obs_key
        }

        #    obs_groupsから、画像データのキーを取り除きます。
        sanitized_obs_groups = {
            group: [key for key in keys if key != vision_obs_key]
            for group, keys in obs_groups.items()
        }

        # 3. 両方とも無害化されたバージョンを親クラスに渡します。
        super().__init__(
            obs=sanitized_obs,
            obs_groups=sanitized_obs_groups,
            num_actions=num_actions,
            **kwargs
        )

        self.prop_obs_keys = prop_obs_keys
        self.vision_obs_key = vision_obs_key
        
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

        prop_obs_dim = sum(obs[key].shape[1] for key in self.prop_obs_keys)
        vision_obs_shape = obs[vision_obs_key].shape[1:]


        # --- 1. 身体感覚エンコーダ (MLP) ---
        prop_layers = []
        in_dim = prop_obs_dim
        # for dim in prop_encoder_dims:
        #     prop_layers.append(nn.Linear(in_dim, dim))
        #     prop_layers.append(self.activation)
        #     in_dim = dim
        # self.proprioception_encoder = nn.Sequential(*prop_layers)

        self.proprioception_encoder = MLP(
            input_dim=prop_obs_dim,
            output_dim=prop_encoder_dims[-1],
            hidden_dims=prop_encoder_dims[:-1],
            activation=activation
        )
        
        # --- 2. 視覚エンコーダ (ConvNet) ---
        h, w, c = vision_obs_shape
        conv_layers = []
        in_c = c
        for out_c in vision_encoder_channels:
            conv_layers.append(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1))
            conv_layers.append(self.activation)
            in_c = out_c
        conv_layers.append(nn.Flatten())
        self.vision_encoder = nn.Sequential(*conv_layers)

        # ConvNetの出力次元数を自動計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h ,w)
            cnn_output_dim = self.vision_encoder(dummy_input).shape[1]

        self.prop_norm = nn.LayerNorm(prop_encoder_dims[-1])
        self.vis_norm = nn.LayerNorm(cnn_output_dim)

        self.prop_proj = nn.Linear(prop_encoder_dims[-1], 256)
        self.vis_proj  = nn.Linear(cnn_output_dim, 256)

        # --- 3. 共有ネットワーク (MLP) ---
        # 身体感覚と視覚の特徴量を結合したものが入力となる
        # fused_dim = prop_encoder_dims[-1] + cnn_output_dim
        fused_dim = 256 * 2
        shared_layers = []
        in_dim = fused_dim
        # for dim in shared_mlp_dims:
        #     shared_layers.append(nn.Linear(in_dim, dim))
        #     shared_layers.append(self.activation)
        #     in_dim = dim
        # self.shared_body = nn.Sequential(*shared_layers)

        self.shared_body = MLP(
            input_dim=fused_dim,
            output_dim=shared_mlp_dims[-1],
            hidden_dims=shared_mlp_dims[:-1],
            activation=activation
        )

        self.critic_body = MLP(
            input_dim=256,
            output_dim=shared_mlp_dims[-1],
            hidden_dims=shared_mlp_dims[:-1],
            activation=activation
        )

        # --- 4. 出力ヘッド ---
        # self.actor = nn.Linear(shared_mlp_dims[-1], num_actions)
        self.actor = nn.Sequential(
            nn.Linear(shared_mlp_dims[-1], num_actions)
            )
        # self.critic = nn.Linear(share d_mlp_dims[-1], 1)

        self.critic = nn.Sequential(
            nn.Linear(shared_mlp_dims[-1], 1)
            )
        
        # --- RSL-RLで必要なaction_distributionのためのパラメータ ---
        # self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        # self.distribution = None
        Normal.set_default_validate_args(False)


    # def forward(self, observations: dict):

    #     # 1. 身体感覚データのテンソルをリストに集める
    #     prop_tensors = [observations[key] for key in self.prop_obs_keys]
    #     # 2. 集めたテンソルを結合して、一つの長いベクトルにする
    #     prop_obs_vector = torch.cat(prop_tensors, dim=-1)
    #     # 3. 視覚データを辞書から取り出す
    #     vis_obs = observations[self.vision_obs_key]
    #     max_depth = 10.0  # 環境に合わせて
    #     if vis_obs.dtype.is_floating_point:
    #         vis_obs = torch.clamp(vis_obs, 0, max_depth) / max_depth   # [0,1]
    #         vis_obs = vis_obs * 2.0 - 1.0                              # [-1,1]
    #     else:
    #         vis_obs = vis_obs.float() / 255.0
    #         vis_obs = vis_obs * 2.0 - 1.0

    #     # 1. 各エンコーダで特徴量を抽出
    #     # prop_features = self.proprioception_encoder(prop_obs_vector)
    #     # vision_features = self.vision_encoder(vis_obs)

    #     prop_features = self.prop_proj(self.prop_norm(self.proprioception_encoder(prop_obs_vector)))
    #     vision_features = self.vis_proj(self.vis_norm(self.vision_encoder(vis_obs)))


        
    #     # 正規化された特徴量を結合します
    #     # fused_features = torch.cat([normalized_prop_features, normalized_vision_features], dim=-1)
    #     # ★★★ 修正はここまで ★★★
        
    #     # 2. 特徴量を結合 (Concatenate)
    #     fused_features = torch.cat([prop_features, vision_features], dim=-1)

    #     # 3. 結合された特徴量を共有ネットワークに通す
    #     shared_output = self.shared_body(fused_features)

    #     # 4. ActorとCriticの出力を計算
    #     actions_mean = self.actor_head(shared_output)
    #     critic_value = self.critic_head(shared_output)

    #     return actions_mean, critic_value#.squeeze(-1)

    # # ----------------------------------------
    # # --- RSL-RLのRunnerが期待するメソッドをオーバーライド ---
    # # ----------------------------------------
    # def act(self, observations: dict, **kwargs):
    #     actions_mean, _ = self.forward(observations)
    #     std = torch.exp(self.log_std)
    #     self.distribution = Normal(actions_mean, std)
    #     return self.distribution.sample()

    # def act_inference(self, observations: dict, **kwargs):
    #     actions_mean, _ = self.forward(observations)
    #     return actions_mean

    # def evaluate(self, observations: dict, **kwargs):
    #     _, critic_value = self.forward(observations)
    #     return critic_value
  

    def get_actor_obs(self, obs):
        """Actor 用の fused features を返す"""
        prop_tensors = [obs[k] for k in self.prop_obs_keys]
        prop_vec = torch.cat(prop_tensors, dim=-1)
        prop_feat = self.proprioception_encoder(prop_vec)

        vis_obs = obs[self.vision_obs_key] #B, H, W, C

        # max_depth = 10.0
        # if vis_obs.dtype.is_floating_point:
        #     vis_obs = torch.clamp(vis_obs, 0, max_depth) / max_depth
        #     vis_obs = vis_obs * 2.0 - 1.0
        # else:
        #     vis_obs = vis_obs.float() / 255.0
        #     vis_obs = vis_obs * 2.0 - 1.0

        # clipping_rangeから最小・最大距離を設定
        clipping_min = 0.1
        clipping_max = 3.0
        # env.step()から得られた元の深度データテンソル
        # shape: (num_envs, height, width, 1)
        # 以前の課題：範囲外を示す0を、まず最大距離として扱う
        vis_obs[vis_obs == 0] = clipping_max

        # [0, 1]の範囲に正規化
        normalized_obs = (vis_obs - clipping_min) / (clipping_max - clipping_min)

        # 念のため、計算誤差などで範囲外になった値を0か1に収める
        vis_obs = torch.clamp(normalized_obs, 0.0, 1.0)

        # vis_obs = torch.zeros_like(vis_obs)
        # vis_obs = vis_obs.float()
        # vis_obs = vis_obs * 0.0


        vis_obs = vis_obs.permute(0, 3, 1, 2) #B, C, H, W
        # print("vision shape:", vis_obs.shape)

        vis_feat = self.vision_encoder(vis_obs)

        prop_features = self.prop_proj(self.prop_norm(prop_feat))
        vision_features = self.vis_proj(self.vis_norm(vis_feat))


        fused = torch.cat([prop_features, vision_features], dim=-1)

        return self.shared_body(fused)

    def get_critic_obs(self, obs):
        # prop_tensors = [obs[k] for k in self.prop_obs_keys] #use only prop features
        # prop_vec = torch.cat(prop_tensors, dim=-1)
        # prop_feat = self.proprioception_encoder(prop_vec)

        # prop_features = self.prop_proj(self.prop_norm(prop_feat))

        # return self.critic_body(prop_features)

        return self.get_actor_obs(obs)






class LocoTransformerActorCritic(ActorCritic):
    """
    LocoTransformerの論文に基づいたActor-Criticモデル。
    身体感覚（Proprioception）と視覚（Vision）をTransformerで融合する。
    """

    def __init__(
        self,
        # from runner 
        obs: dict,
        obs_groups: dict,
        num_actions: int,
        # --- Cfgから「名前」で渡される引数をその後に書く ---
        prop_obs_keys: list[str],
        vision_obs_key: str,
        prop_encoder_dims: list[int] = [256, 256],
        vision_channels: int = 128,
        transformer_hidden_dim: int = 256,
        transformer_n_heads: int = 4,
        transformer_num_layers: int = 2,
        projection_head_dims: list = [256, 256],
        num_image_stack: int = 4,
        init_noise_std: float = 1.0,
        activation: str = "elu",
        **kwargs,
    ):
        # 1. 親クラスに渡すための「無害化」された観測辞書を作成します。
        #    オリジナルのobs辞書から、問題となる画像データをキーごと取り除きます。
        sanitized_obs = {
            key: value for key, value in obs.items()
            if key != vision_obs_key
        }
        #    obs_groupsから、画像データのキーを取り除きます。
        sanitized_obs_groups = {
            group: [key for key in keys if key != vision_obs_key]
            for group, keys in obs_groups.items()
        }
        # 3. 両方とも無害化されたバージョンを親クラスに渡します。
        super().__init__(
            obs=sanitized_obs,
            obs_groups=sanitized_obs_groups,
            num_actions=num_actions,
            **kwargs
        )

        self.prop_obs_keys = prop_obs_keys
        self.vision_obs_key = vision_obs_key

        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

        prop_obs_dim = sum(obs[key].shape[1] for key in self.prop_obs_keys)
        vision_obs_shape = obs[vision_obs_key].shape[1:] #flattend

        print(prop_obs_dim)
        print(vision_obs_shape[0]) 


        # 1. 身体感覚エンコーダ (MLP)
        # 論文: a 2-layer MLP with hidden dimensions (256, 256)
        prop_layers = []
        in_dim = prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers.append(nn.Linear(in_dim, dim))
            prop_layers.append(nn.ELU())
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)
        
        input_channels = num_image_stack
        # 2. 視覚エンコーダ (ConvNet)
        # 論文: 64x64の画像を 4x4 x 128チャンネルの特徴量マップに変換

        self.vision_encoder = nn.Sequential(
            # Input: (B, C, 64, 64)
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1), # -> (B, 32, 32, 32)
            self.activation,
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (B, 64, 16, 16)
            self.activation,
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (B, 128, 8, 8)
            self.activation,
            nn.Conv2d(128, vision_channels, kernel_size=3, stride=2, padding=1), # -> (B, 128, 4, 4)
            self.activation,
        )
        
        # 3. Transformerのための準備
        # 各特徴量をTransformerの次元に射影（Project）する層
        self.prop_proj = nn.Linear(prop_encoder_dims[-1], transformer_hidden_dim)
        self.vis_proj = nn.Linear(vision_channels, transformer_hidden_dim)
        
        # Transformer Encoder層を定義
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_hidden_dim,
            nhead=transformer_n_heads,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True # RSL-RLは (batch, seq, dim) の形式を扱うため
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
        # 論文の 4x4=16個の視覚トークンに対応する位置エンコーディング
        self.visual_pos_embedding = nn.Parameter(torch.randn(1, 16, transformer_hidden_dim))
        # 身体感覚トークン用の位置エンコーディング
        self.prop_pos_embedding = nn.Parameter(torch.randn(1, 1, transformer_hidden_dim))

        # 4. 出力ヘッド
        # 論文: 融合された特徴量を処理する2層のMLP
        fused_feature_dim = transformer_hidden_dim * 2 # 身体感覚と視覚の2種類
        proj_layers = []
        in_dim = fused_feature_dim
        for dim in projection_head_dims:
            proj_layers.append(nn.Linear(in_dim, dim))
            proj_layers.append(nn.ELU())
            in_dim = dim
        self.projection_head = nn.Sequential(*proj_layers)

        # Actor (行動決定) と Critic (価値評価) の最終層
        # self.actor = nn.Linear(projection_head_dims[-1], num_actions)
        # self.critic = nn.Linear(projection_head_dims[-1], 1)

        self.actor = nn.Sequential(
            nn.Linear(projection_head_dims[-1], num_actions)
            )
        self.critic = nn.Sequential(
            nn.Linear(projection_head_dims[-1], 1)
            )


        
        # RSL-RLで必要なaction_distributionのためのパラメータ
        # self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        # self.distribution = None
        Normal.set_default_validate_args(False)


    # def forward(self, observations: dict):
    #     prop_tensors = [observations[key] for key in self.prop_obs_keys]
    #     # 2. 集めたテンソルを結合して、一つの長いベクトルにする
    #     prop_obs_vector = torch.cat(prop_tensors, dim=-1)
    #     # 3. 視覚データを辞書から取り出す
    #     vis_obs_flat = observations[self.vision_obs_key] #flattend observations

    #     batch_size = vis_obs_flat.shape[0]
    #     # (Batch, 16384) -> (Batch, 4, 64, 64)
    #     # 4はスタックされたフレーム数、64, 64は画像の高さと幅
    #     vis_obs_unflattened = vis_obs_flat.view(batch_size, 4, 64, 64)


    #     max_depth = 10.0  # 環境に合わせて
    #     if vis_obs_unflattened.dtype.is_floating_point:
    #         vis_obs = torch.clamp(vis_obs_unflattened, 0, max_depth) / max_depth   # [0,1]
    #         vis_obs = vis_obs * 2.0 - 1.0                              # [-1,1]
    #     else:
    #         vis_obs = vis_obs_unflattened.float() / 255.0
    #         vis_obs = vis_obs * 2.0 - 1.0

    #     # --- 1. 各エンコーダで特徴量を抽出 ---
    #     prop_features = self.proprioception_encoder(prop_obs_vector)
    #     vis_features_map = self.vision_encoder(vis_obs)

    #     # --- 2. トークン化とTransformerによる融合 ---
    #     # 身体感覚トークン
    #     prop_token = self.prop_proj(prop_features).unsqueeze(1) # (batch, 1, dim)
        
    #     # 視覚トークン
    #     vis_tokens = vis_features_map.flatten(2).permute(0, 2, 1) # (batch, 16, channels)
    #     vis_tokens = self.vis_proj(vis_tokens)
        
    #     # 位置エンコーディングを追加
    #     prop_token += self.prop_pos_embedding
    #     vis_tokens += self.visual_pos_embedding
        
    #     # 全トークンを結合してTransformerに入力
    #     all_tokens = torch.cat([prop_token, vis_tokens], dim=1)
    #     fused_tokens = self.transformer_encoder(all_tokens)
        
    #     # --- 3. 融合された特徴量の処理と出力 ---
    #     # 論文の記述通り、各モダリティの情報を平均化して集約
    #     fused_prop = fused_tokens[:, 0, :]
    #     fused_vis = fused_tokens[:, 1:, :].mean(dim=1)
        
    #     # 2つの特徴量を結合
    #     fused_features = torch.cat([fused_prop, fused_vis], dim=1)
        
    #     # Projection Headに通す
    #     final_features = self.projection_head(fused_features)

    #     # ActorとCriticの出力を計算
    #     actions_mean = self.actor(final_features)
    #     critic_value = self.critic(final_features)

    #     return actions_mean, critic_value


    def get_actor_obs(self, obs):
        """Actor 用の fused features を返す"""
        prop_tensors = [obs[k] for k in self.prop_obs_keys]
        prop_vec = torch.cat(prop_tensors, dim=-1)
        prop_feat = self.proprioception_encoder(prop_vec)

        vis_obs_flat = obs[self.vision_obs_key] #flattend observations

        batch_size = vis_obs_flat.shape[0]
        # (Batch, 16384) -> (Batch, 4, 64, 64)
        # 4はスタックされたフレーム数、64, 64は画像の高さと幅
        vis_obs_unflattened = vis_obs_flat.view(batch_size, 4, 64, 64) #B,C,H,W

        clipping_min = 0.1
        clipping_max = 3.0
        # env.step()から得られた元の深度データテンソル
        # shape: (num_envs, height, width, 1)
        # 以前の課題：範囲外を示す0を、まず最大距離として扱う
        vis_obs_unflattened[vis_obs_unflattened == 0] = clipping_max

        # [0, 1]の範囲に正規化
        normalized_obs = (vis_obs_unflattened - clipping_min) / (clipping_max - clipping_min)

        # 念のため、計算誤差などで範囲外になった値を0か1に収める
        vis_obs = torch.clamp(normalized_obs, 0.0, 1.0)

        # print(
        #     "Depth input stats:",
        #     vis_obs_unflattened.min().item(),
        #     vis_obs_unflattened.max().item()
        # )


        
        # print(
        #     "Normalized depth stats:",
        #     vis_obs.min().item(),
        #     vis_obs.max().item()
        # )

        

        vis_feat = self.vision_encoder(vis_obs)
        # print("vision feature shape:", vis_feat.shape)


        # 身体感覚トークン
        prop_token = self.prop_proj(prop_feat).unsqueeze(1) # (batch, 1, dim)
        # 視覚トークン
        vis_tokens = vis_feat.flatten(2).permute(0, 2, 1) # (batch, 16, channels)
        # print("vis_tokens shape:", vis_tokens.shape)  
        vis_tokens = self.vis_proj(vis_tokens)
        
        # 位置エンコーディングを追加
        prop_token += self.prop_pos_embedding
        vis_tokens += self.visual_pos_embedding
        
        # 全トークンを結合してTransformerに入力
        all_tokens = torch.cat([prop_token, vis_tokens], dim=1)
        # print("all tokens shape:", all_tokens.shape)  
        fused_tokens = self.transformer_encoder(all_tokens)
        
        # --- 3. 融合された特徴量の処理と出力 ---
        # 論文の記述通り、各モダリティの情報を平均化して集約
        fused_prop = fused_tokens[:, 0, :]
        fused_vis = fused_tokens[:, 1:, :].mean(dim=1)
        
        # 2つの特徴量を結合
        fused_features = torch.cat([fused_prop, fused_vis], dim=1)

        # Projection Headに通す
        final_features = self.projection_head(fused_features)

        return final_features


    def get_critic_obs(self, obs):
        """Critic も同じ fused features を使う（場合によっては別設計も可能）"""
        return self.get_actor_obs(obs)



    # def act(self, observations: dict, **kwargs):
    #     """訓練中に、現在の観測から行動をサンプリングします。"""
    #     actions_mean, _ = self.forward(observations)
    #     # 行動の分布を更新
    #     if self.noise_std_type == "scalar":
    #         std = self.std.expand_as(actions_mean)
    #     elif self.noise_std_type  == "log" :
    #         std = torch.exp(self.log_std).expand_as(actions_mean)
    #     self.distribution = Normal(actions_mean, std)
    #     # 分布からサンプリング
    #     return self.distribution.sample()

    # def act_inference(self, observations: dict, **kwargs):
    #     """テスト（推論）中に、決定論的な行動を返します。"""
    #     actions_mean, _ = self.forward(observations)
    #     return actions_mean

    # def evaluate(self, observations: dict, **kwargs):
    #     """現在の観測の価値を評価します。"""
    #     _, critic_value = self.forward(observations)
    #     return critic_value
