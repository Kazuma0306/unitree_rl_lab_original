from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

import torch
from .velocity_env_cfg2 import RobotEnvCfg

class Go2PositionEnv(ManagerBasedRLEnv):
    """
    Go2歩行学習用のカスタム環境。
    テスト時に目標地点のマーカーを可視化する機能を持つ。
    """
    # この環境が受け取るCfgクラスの型ヒントです
    cfg: "RobotEnvCfg"

    def __init__(self, cfg: RobotEnvCfg, **kwargs):
        """初期化時に、設定からマーカーオブジェクトを取得します。"""
        super().__init__(cfg, **kwargs)
        self._goal_markers: VisualizationMarkers | None = self.scene.get("visualization_markers", None)

    def _pre_physics_step(self, actions: torch.Tensor):
        """物理計算の直前に呼ばれる関数で、マーカーの位置を更新します。"""
        super()._pre_physics_step(actions)
        
        # マーカーの更新はテスト(Play)時のみ実行します
        if self.is_playing() and self._goal_markers is not None:
            self._update_goal_markers()

    def _update_goal_markers(self):
        """コマンドに基づいて目標地点マーカーの位置を更新します。"""
        # コマンドマネージャーから現在の目標位置を取得します
        target_pos_local = self.command_manager.get_command("base_position")
        
        # ワールド座標系での絶対目標位置を計算します
        target_pos_world = self.scene.env_origins + target_pos_local[:, :3]
        
        # マーカーが地面に埋まらないように、高さを少し上げます
        target_pos_world[:, 2] = 0.15

        # 計算した位置にマーカーを表示します
        self._goal_markers.visualize(positions=target_pos_world, marker_indices=0)
