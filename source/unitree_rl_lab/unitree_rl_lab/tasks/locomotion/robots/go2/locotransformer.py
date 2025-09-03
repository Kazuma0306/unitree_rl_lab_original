import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import gymnasium as gym
# ▼▼▼ この行を追加 ▼▼▼
from gymnasium.spaces import Box
import numpy as np

# このファイルを例えば custom_models.pyとして保存します


# class MultiModalEnvWrapper(RslRlVecEnvWrapper):
#     """
#     辞書形式の観測空間を、古いRSL-RLのOnPolicyRunnerで扱えるようにするラッパー。
#     """
#     def __init__(self, env, clip_actions):
#         # まず親クラスの初期化を呼び出す
#         super().__init__(env)
        
#         # --- OnPolicyRunnerの__init__を騙すための偽装工作 ---
#         # OnPolicyRunnerは、初期化時にnum_obsという平坦なベクトル次元数しか見ない。
#         # そこで、観測空間がまるで身体感覚（proprioception）だけであるかのように見せかける。
#         prop_space = self.env.observation_space["proprioception"]
        
#         # このラッパーが見せる表向きの観測空間と次元数を上書き
#         self.observation_space = prop_space
#         self.num_obs = prop_space.shape[0]


# class MultiModalEnvWrapper(RslRlVecEnvWrapper):
#     """
#     フラットな辞書形式の観測空間を、古いRSL-RLのOnPolicyRunnerで扱えるようにするラッパー。
#     ベクトル観測を自動的に識別して、それらを結合した偽装用の観測空間を作成する。
#     """
#     def __init__(self, env, clip_actions):
#         # まず親クラスの初期化を呼び出す
#         super().__init__(env)
        
#         # --- OnPolicyRunnerの__init__を騙すための偽装工作 ---
        
#         total_prop_dims = 0
#         # 環境が持つ観測空間の辞書をループ
#         for space in self.env.observation_space.spaces.values():
#             # その空間が1次元のBox（つまりベクトル）であるかをチェック
#             if isinstance(space, Box) and len(space.shape) == 1:
#                 # ベクトルであれば、その次元数を合計に加える
#                 total_prop_dims += space.shape[0]


#         # 計算した値を、ラッパーの内部変数として保存する
#         self._num_obs = total_prop_dims
#         self._observation_space = Box(
#             low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32
#         )

#         # ▼▼▼ ここからが重要な修正点 ▼▼▼
        
#         @property
#         def num_obs(self) -> int:
#             """OnPolicyRunnerが参照するnum_obsをオーバーライドし、偽装した値を返す。"""
#             return self._num_obs

#         @property
#         def observation_space(self) -> gym.spaces.Space:
#             """OnPolicyRunnerが参照するobservation_spaceをオーバーライドし、偽装した空間を返す。"""
#             return self._observation_space


class MultiModalEnvWrapper(RslRlVecEnvWrapper):
    """
    辞書形式の観測空間を、古いRSL-RLのOnPolicyRunnerで扱えるようにする最終版ラッパー。
    初期化時と実行時で、返す観測データの形式を変える。
    """
    def __init__(self, env, clip_actions: bool = False):
        # clip_actionsを親クラスに正しく渡す
        super().__init__(env, clip_actions=clip_actions)
        
        # --- OnPolicyRunnerの__init__を騙すための偽装工作 ---
        
        # 観測空間の中からベクトルデータだけを探し出し、その合計次元数を計算する
        self._prop_keys = []
        total_prop_dims = 0
        for key, space in self.env.observation_space.spaces.items():
            if isinstance(space, Box) and len(space.shape) == 1:
                self._prop_keys.append(key)
                total_prop_dims += space.shape[0]

        # 計算した値を、ラッパーの内部変数として保存する
        self._num_obs = total_prop_dims
        self._observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self._num_obs,), dtype=np.float32
        )

    # --- OnPolicyRunnerの__init__から呼ばれるプロパティとメソッドをオーバーライド ---

    @property
    def num_obs(self) -> int:
        """OnPolicyRunnerの初期化時に参照されるnum_obs。偽装した値を返す。"""
        return self._num_obs

    @property
    def observation_space(self) -> gym.spaces.Space:
        """OnPolicyRunnerの初期化時に参照されるobservation_space。偽装した空間を返す。"""
        return self._observation_space

    def get_observations(self) -> torch.Tensor:
        """OnPolicyRunnerの初期化時に一度だけ呼ばれる。偽装したベクトルデータを返す。"""
        # 環境から完全な辞書形式の観測データを取得
        obs_dict = self.env.get_observations()
        # 辞書の中から、身体感覚のデータだけを集めて結合し、単一のテンソルとして返す
        prop_tensors = [obs_dict[key] for key in self._prop_keys]
        return torch.cat(prop_tensors, dim=-1)


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
        prop_obs_keys: list[str],
        vision_obs_key: str,

        # prop_obs_dim: int,
        # vision_obs_shape: tuple, # (channels, height, width)
        num_actions: int,
        # --- モデルのサイズに関するパラメータ ---
        prop_encoder_dims: list = [256, 128],
        vision_encoder_channels: list = [32, 64, 64],
        shared_mlp_dims: list = [256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs,
    ):
        # 親クラスの__init__はMLP前提なので呼び出さず、nn.Moduleとして初期化
        nn.Module.__init__(self)
        
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

        # --- 1. 身体感覚エンコーダ (MLP) ---
        prop_layers = []
        in_dim = 61
        for dim in prop_encoder_dims:
            prop_layers.append(nn.Linear(in_dim, dim))
            prop_layers.append(self.activation)
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)
        
        # --- 2. 視覚エンコーダ (ConvNet) ---
        c, h, w = vision_obs_shape
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
            dummy_input = torch.zeros(1, *vision_obs_shape)
            cnn_output_dim = self.vision_encoder(dummy_input).shape[1]

        # --- 3. 共有ネットワーク (MLP) ---
        # 身体感覚と視覚の特徴量を結合したものが入力となる
        fused_dim = prop_encoder_dims[-1] + cnn_output_dim
        shared_layers = []
        in_dim = fused_dim
        for dim in shared_mlp_dims:
            shared_layers.append(nn.Linear(in_dim, dim))
            shared_layers.append(self.activation)
            in_dim = dim
        self.shared_body = nn.Sequential(*shared_layers)

        # --- 4. 出力ヘッド ---
        self.actor_head = nn.Linear(shared_mlp_dims[-1], num_actions)
        self.critic_head = nn.Linear(shared_mlp_dims[-1], 1)
        
        # --- RSL-RLで必要なaction_distributionのためのパラメータ ---
        self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        self.distribution = None
        Normal.set_default_validate_args(False)


    def forward(self, observations: dict):
        """モデルの主要なデータフローを定義します。"""
        # 1. 身体感覚データのテンソルをリストに集める
        prop_tensors = [observations[key] for key in self.prop_obs_keys]
        # 2. 集めたテンソルを結合して、一つの長いベクトルにする
        prop_obs_vector = torch.cat(prop_tensors, dim=-1)
        # 3. 視覚データを辞書から取り出す
        vis_obs = observations[self.vision_obs_key]

        # 1. 各エンコーダで特徴量を抽出
        prop_features = self.proprioception_encoder(prop_obs_vector)
        vision_features = self.vision_encoder(vis_obs)
        
        # 2. 特徴量を結合 (Concatenate)
        fused_features = torch.cat([prop_features, vision_features], dim=-1)

        # 3. 結合された特徴量を共有ネットワークに通す
        shared_output = self.shared_body(fused_features)

        # 4. ActorとCriticの出力を計算
        actions_mean = self.actor_head(shared_output)
        critic_value = self.critic_head(shared_output)

        return actions_mean, critic_value.squeeze(-1)

    # ----------------------------------------
    # --- RSL-RLのRunnerが期待するメソッドをオーバーライド ---
    # ----------------------------------------
    def act(self, observations: dict, **kwargs):
        actions_mean, _ = self.forward(observations)
        std = torch.exp(self.log_std)
        self.distribution = Normal(actions_mean, std)
        return self.distribution.sample()

    def act_inference(self, observations: dict, **kwargs):
        actions_mean, _ = self.forward(observations)
        return actions_mean

    def evaluate(self, observations: dict, **kwargs):
        _, critic_value = self.forward(observations)
        return critic_value
        
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)






class LocoTransformerActorCritic(ActorCritic):
    """
    LocoTransformerの論文に基づいたActor-Criticモデル。
    身体感覚（Proprioception）と視覚（Vision）をTransformerで融合する。
    """
    def __init__(
        self,
        prop_obs_dim: int,
        vision_obs_shape: tuple, # (channels, height, width)
        num_actions: int,
        # --- モデルのサイズに関するパラメータ（論文に基づいた値） ---
        prop_encoder_dims: list = [256, 256],
        vision_channels: int = 128,
        transformer_hidden_dim: int = 256,
        transformer_n_heads: int = 4,
        transformer_num_layers: int = 2,
        activation: str = "elu",
        projection_head_dims: list = [256, 256],
        **kwargs,
    ):
        # rsl_rlのActorCriticの__init__はMLP前提なので呼び出さず、
        # PyTorchの基本モジュールとして初期化する
        nn.Module.__init__(self)

        # 1. 身体感覚エンコーダ (MLP)
        # 論文: a 2-layer MLP with hidden dimensions (256, 256)
        prop_layers = []
        in_dim = prop_obs_dim
        for dim in prop_encoder_dims:
            prop_layers.append(nn.Linear(in_dim, dim))
            prop_layers.append(nn.ELU())
            in_dim = dim
        self.proprioception_encoder = nn.Sequential(*prop_layers)
        
        # 2. 視覚エンコーダ (ConvNet)
        # 論文: 64x64の画像を 4x4 x 128チャンネルの特徴量マップに変換
        # これは一般的なCNNアーキテクチャで実装
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(vision_obs_shape[0], 32, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ELU(),
            nn.Conv2d(64, vision_channels, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # この時点で 4x4 x 128 の特徴量マップになっているはず
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
            activation="elu",
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
        self.actor_head = nn.Linear(projection_head_dims[-1], num_actions)
        self.critic_head = nn.Linear(projection_head_dims[-1], 1)
        
        # RSL-RLで必要なaction_distributionのためのパラメータ
        # (この部分は元のActorCriticからコピー)
        self.std = nn.Parameter(torch.ones(num_actions))
        # ...

    def forward(self, observations: dict):
        # 観測はグループ分けされた辞書として渡ってくる
        prop_obs = observations["proprioception"]
        vis_obs = observations["vision"]

        # --- 1. 各エンコーダで特徴量を抽出 ---
        prop_features = self.proprioception_encoder(prop_obs)
        vis_features_map = self.vision_encoder(vis_obs)

        # --- 2. トークン化とTransformerによる融合 ---
        # 身体感覚トークン
        prop_token = self.prop_proj(prop_features).unsqueeze(1) # (batch, 1, dim)
        
        # 視覚トークン
        batch_size = vis_features_map.shape[0]
        vis_tokens = vis_features_map.flatten(2).permute(0, 2, 1) # (batch, 16, channels)
        vis_tokens = self.vis_proj(vis_tokens)
        
        # 位置エンコーディングを追加
        prop_token += self.prop_pos_embedding
        vis_tokens += self.visual_pos_embedding
        
        # 全トークンを結合してTransformerに入力
        all_tokens = torch.cat([prop_token, vis_tokens], dim=1)
        fused_tokens = self.transformer_encoder(all_tokens)
        
        # --- 3. 融合された特徴量の処理と出力 ---
        # 論文の記述通り、各モダリティの情報を平均化して集約
        fused_prop = fused_tokens[:, 0, :]
        fused_vis = fused_tokens[:, 1:, :].mean(dim=1)
        
        # 2つの特徴量を結合
        fused_features = torch.cat([fused_prop, fused_vis], dim=1)
        
        # Projection Headに通す
        final_features = self.projection_head(fused_features)

        # ActorとCriticの出力を計算
        actions_mean = self.actor_head(final_features)
        critic_value = self.critic_head(final_features)

        return actions_mean, critic_value

    # act, evaluate, get_actions_log_prob などのメソッドは、
    # このforwardメソッドを呼び出すように親クラスからオーバーライドする必要があります。

    def act(self, observations: dict, **kwargs):
        """訓練中に、現在の観測から行動をサンプリングします。"""
        actions_mean, _ = self.forward(observations)
        # 行動の分布を更新
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type  == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(actions_mean, std)
        # 分布からサンプリング
        return self.distribution.sample()

    def act_inference(self, observations: dict, **kwargs):
        """テスト（推論）中に、決定論的な行動を返します。"""
        actions_mean, _ = self.forward(observations)
        return actions_mean

    def evaluate(self, observations: dict, **kwargs):
        """現在の観測の価値を評価します。"""
        _, critic_value = self.forward(observations)
        return critic_value

    def get_actions_log_prob(self, actions):
        """与えられた行動の対数確率を返します。"""
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def entropy(self):
        """行動分布のエントロピーを返します。"""
        return self.distribution.entropy().sum(dim=-1)