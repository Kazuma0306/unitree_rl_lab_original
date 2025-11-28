#proposed method (multiple legs on block)



import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, CameraCfg, TiledCameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers



from isaaclab.utils.math import quat_apply_inverse
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf
import omni.usd


from isaaclab.envs.mdp import StepFRToBlockCommandCfg, MultiLegBaseCommandCfg


# ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
#     size=(8.0, 8.0),
#     border_width=20.0,
#     num_rows=10,
#     num_cols=20,
#     horizontal_scale=0.1,
#     vertical_scale=0.005,
#     slope_threshold=0.75,
#     use_cache=False,
#     sub_terrains={
#         "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.05, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
#             proportion=0.2,
#             step_height_range=(0.05, 0.23),
#             step_width=0.3,
#             platform_width=3.0,
#             border_width=1.0,
#             holes=False,
#         ),
#         "boxes": terrain_gen.MeshRandomGridTerrainCfg(
#             proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
#         ),
#         "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
#             proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
#         ),
#         "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#         "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
#             proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
#         ),
#     },
# )

DESCRETE_OBSTACLES_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),

        "descrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
             proportion=0.2, border_width=0.25, obstacle_width_range =(0.8,  0.9) , obstacle_height_range = (0.0, 0.02), num_obstacles = 80
        ),
        # more difficult version
        "descrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
             proportion=0.6, border_width=0.25, obstacle_width_range =(0.8,  0.9) , obstacle_height_range = (0.01, 0.05), num_obstacles = 80
        ),
    },
)


STEPPING_STONES_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),

        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
             proportion=0.2, border_width=0.25,  horizontal_scale = 0.01, stone_height_max = 0.01, stone_width_range = (1.0, 1.5), stone_distance_range = (0.05, 0.08),  holes_depth = -5.0, platform_width = 1.5,

        ),

        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
             proportion=0.7, border_width=0.25,  horizontal_scale = 0.01, stone_height_max = 0.01, stone_width_range = (0.7, 1.5), stone_distance_range = (0.05, 0.09),  holes_depth = -5.0, platform_width = 1.5,

        ),


        #difficult version
        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #      proportion=0.9, border_width=0.25,  horizontal_scale = 0.05, stone_height_max = 0.01, stone_width_range = (0.5, 0.9), stone_distance_range = (0.06, 0.09),  holes_depth = -5.0, platform_width = 1.5,

        # ),

        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #      proportion=1.0, border_width=0.25,  stone_height_max = 0.05, stone_width_range = (0.2, 0.9), stone_distance_range = (0.08, 0.26),  holes_depth = -5.0, platform_width = 1.5,

        # ),
    },
)



COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),
        # "descrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        #      proportion=0.1, border_width=0.25, obstacle_width_range =(0.8,  0.9) , obstacle_height_range = (0.01, 0.05), num_obstacles = 80
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.05, 0.23),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
    },
)


MOAT_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    # border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={

        "moat_terrain": terrain_gen.MeshMoatTerrainCfg(
             proportion=0.7, platform_width = 0.8,

        ),

    },
)


BLOCK_CFG = terrain_gen.TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    # border_width=20.0,
    num_rows=1,
    num_cols=2,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={

        "block_terrain": terrain_gen.MeshBlockTerrainCfg(
             proportion=0.7, 

        ),

    },
)



import numpy as np


# def make_ring_xy(stone_w, gap, inner_half, outer_half):
#     pitch = stone_w + gap
#     xs = np.arange(-outer_half + stone_w/2, outer_half, pitch)
#     ys = np.arange(-outer_half + stone_w/2, outer_half, pitch)
#     xs, ys = np.meshgrid(xs, ys, indexing="xy")
#     m = (np.maximum(np.abs(xs), np.abs(ys)) > inner_half) & (np.maximum(np.abs(xs), np.abs(ys)) < outer_half)
#     xs, ys = xs[m], ys[m]
#     return [(float(x), float(y)) for x, y in zip(xs.ravel(), ys.ravel())]


# def make_ring_xy2(stone_w, gap, inner_half, outer_half):
#     import numpy as np
#     pitch = stone_w + gap
#     # ★ 内側の石の“端”が inner_half に一致するように開始位相を合わせる
#     #    具体的には「石中心 = inner_half - stone_w/2」が格子点になる s を選ぶ
#     s = (inner_half - stone_w/2) % pitch
#     xs = np.arange(-outer_half + s,  outer_half + 1e-6, pitch)
#     ys = np.arange(-outer_half + s,  outer_half + 1e-6, pitch)
#     xs, ys = np.meshgrid(xs, ys, indexing="xy")

#     # L∞ノルムの“正方枠”で抽出（あなたの元実装と同じ）
#     m = (np.maximum(np.abs(xs), np.abs(ys)) > inner_half) & \
#         (np.maximum(np.abs(xs), np.abs(ys)) < outer_half)
#     xs, ys = xs[m], ys[m]
#     return [(float(x), float(y)) for x, y in zip(xs.ravel(), ys.ravel())]


# def make_ring_xy3(stone_w, gap, inner_half, outer_half):
#         pitch = stone_w + gap

#         # ★ 修正点: 原点 0.0 を基準に座標を生成
        
#         # 1. 正の座標（0.0 を含む）を生成
#         # (例: pitch=1.5, outer_half=5.0 -> [0.0, 1.5, 3.0, 4.5])
#         coords_pos = np.arange(0, outer_half, pitch)
        
#         # 2. 負の座標を生成 (0.0 は含めない)
#         # (例: pitch=1.5, outer_half=5.0 -> [-1.5, -3.0, -4.5])
#         coords_neg = np.arange(-pitch, -outer_half, -pitch)
        
#         # 3. 結合して対称な座標リストを作成
#         # (例: [-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5])
#         #    np.abs() を取ると [4.5, 3.0, 1.5, 0.0, 1.5, 3.0, 4.5] となり完璧に対称
#         xs = np.concatenate((coords_neg, coords_pos))
#         ys = np.concatenate((coords_neg, coords_pos))

#         # --- 以下は元のコードのままでOK ---
        
#         # 4. グリッドを生成
#         xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")
        
#         # 5. マスク（フィルタリング）
#         #    (このロジックは、xs, ys が 0.0 対称である前提で正しく機能する)
#         m = (np.maximum(np.abs(xs_grid), np.abs(ys_grid)) > inner_half) & \
#             (np.maximum(np.abs(xs_grid), np.abs(ys_grid)) < outer_half)
        
#         # 6. マスクがTrueの要素だけを抽出
#         xs_flat, ys_flat = xs_grid[m], ys_grid[m]
        
#         return [(float(x), float(y)) for x, y in zip(xs_flat, ys_flat)]


def make_ring_xy4(stone_w, gap, inner_half, outer_half):
    """
    グリッドを使いつつ、「はみ出し」を防ぐため、
    フィルタリング時にブロックの幅を考慮する。
    """
    pitch = stone_w + gap
    half_w = stone_w / 2.0  # ★ ブロックの半分の幅

    # 1. 原点(0,0)に対称なグリッドを生成 (No. 52のロジック)
    #    (これが「均等」の基礎となります)
    coords_pos = np.arange(0, outer_half, pitch)
    coords_neg = np.arange(-pitch, -outer_half, -pitch)
    xs = np.concatenate((coords_neg, coords_pos))
    ys = np.concatenate((coords_neg, coords_pos))

    if xs.size == 0 or ys.size == 0:
        return []  # 格子点が無ければ空リストを返す

    xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")

    # 2. フィルタリング (★ここが修正点)
    
    # グリッドの各点(石の中心)のL∞ノルム(チェビシェフ距離)
    max_dist_center = np.maximum(np.abs(xs_grid), np.abs(ys_grid))

    # 3. 内側の境界チェック
    #    石の「中心」が「(内側の境界 + ブロックの半幅)」より外側にあるか
    m_inner = (max_dist_center > inner_half + half_w)

    # 4. 外側の境界チェック
    #    石の「中心」が「(外側の境界 - ブロックの半幅)」より内側にあるか
    m_outer = (max_dist_center < outer_half - half_w)

    m_positive_x = (xs_grid - half_w > 0)

    # ブロックの上端が y_limit より下
    m_y_upper = (ys_grid + half_w < 1.55)
    # ブロックの下端が -y_limit より上
    m_y_lower = (ys_grid - half_w > -1.55)

    # 5. 両方を満たすもの
    #    (これにより、ブロック全体がリングの内側に収まる)
    m = m_inner & m_outer & m_positive_x & m_y_upper & m_y_lower
    
    xs_flat, ys_flat = xs_grid[m], ys_grid[m]
    
    return [(float(x), float(y)) for x, y in zip(xs_flat, ys_flat)]

# def make_ring_xy2(
#     stone_w: float,           # タイル一辺
#     gap: float,               # タイル間の隙間（中心間ピッチは stone_w+gap）
#     inner_half: float,        # 中央安全帯の「半幅」(例: 1.5m四方なら 0.75)
#     outer_half: float,        # 外縁の「半幅」（堀の内側など）
#     center_xy=(0.0, 0.0),     # 格子の中心（env原点と一致させるのが基本）
#     jitter: float = 0.0,      # 角度ではなく微小な位置ランダム(<= gap/2 推奨)
# ):
#     """
#     原点（center_xy）に対して左右対称な格子を作り、各タイルの「外接正方形」が
#     ・内側境界（inner_half）に食い込まない
#     ・外側境界（outer_half）からはみ出さない
#     ようにフィルタします（Chebyshev距離で判定）。
#     """
#     pitch = stone_w + gap
#     cx, cy = center_xy

#     # 外縁に“全面が入る”最大の個数を左右対称に計算
#     # 有効幅は outer_half*2 から「左右の余白 stone_w」を引いた長さ
#     usable = 2.0 * outer_half - stone_w
#     if usable < 0:
#         return []  # outer_half が小さすぎる
#     n_axis = int(np.floor(usable / pitch)) + 1          # 片軸方向の個数
#     # 中心対称に並ぶように原点を決める（左右に同数だけ並ぶ）
#     start = -0.5 * pitch * (n_axis - 1)
#     xs = start + np.arange(n_axis) * pitch + cx
#     ys = start + np.arange(n_axis) * pitch + cy

#     # グリッド化
#     XX, YY = np.meshgrid(xs, ys, indexing="xy")
#     XX = XX.ravel()
#     YY = YY.ravel()

#     # 各タイル中心に対し、タイル半径（=stone_w/2）を考慮して
#     # 外側： |x-cx| or |y-cy| <= outer_half - stone_w/2
#     # 内側： |x-cx| or |y-cy| >= inner_half + stone_w/2
#     # で判定（Chebyshev距離＝外接正方形で判定）
#     dx = np.abs(XX - cx)
#     dy = np.abs(YY - cy)
#     cheb = np.maximum(dx, dy)

#     keep_outer = cheb <= (outer_half - stone_w * 0.5 + 1e-9)
#     keep_inner = cheb >= (inner_half + stone_w * 0.5 - 1e-9)
#     mask = keep_outer & keep_inner

#     XX, YY = XX[mask], YY[mask]

#     # 軽いゆらぎ（はみ出さないよう outer/inner を再チェックして抑制）
#     if jitter > 0.0:
#         jx = np.random.uniform(-jitter, jitter, size=XX.shape)
#         jy = np.random.uniform(-jitter, jitter, size=YY.shape)
#         XX2 = XX + jx
#         YY2 = YY + jy
#         dx2 = np.abs(XX2 - cx)
#         dy2 = np.abs(YY2 - cy)
#         cheb2 = np.maximum(dx2, dy2)
#         ok = (cheb2 <= (outer_half - stone_w * 0.5)) & (cheb2 >= (inner_half + stone_w * 0.5))
#         XX[ok] = XX2[ok]
#         YY[ok] = YY2[ok]

#     return [(float(x), float(y)) for x, y in zip(XX, YY)]


STONE_W, STONE_H, GAP = 0.2, 0.3, 0.004
stone_xy_list = make_ring_xy4(STONE_W, GAP, inner_half=0.7, outer_half=3.37)

# stone_xy_list = [(0.6, 0.0), (0.9, 0.15), (1.2, -0.1)]



# blocks_cfg, xy_list = build_block_cfgs_for_scene(
#     stone_w=0.2, stone_h=0.3, gap=0.06,
#     inner_half=0.75, outer_half=1.75, z=0.03, mass=3.0
# )

def make_stone_cfg(i, pos_xyz):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Stone_{i:04d}",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(

                collision_enabled=True,
               
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.8, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos_xyz)  # ここで初期配置
    )


z0 = 0.1  # 石の厚みに応じた天板高さ


# for proposed terrain
# stones_dict = {
#     f"stone_{i:04d}": make_stone_cfg(i, (x, y, z0))
#     for i, (x, y) in enumerate(stone_xy_list)
# }



sizex = 0.16
sizey = 0.16
MASS_B = 1.0

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        # terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        # terrain_generator=ROUGH_TERRAINS_CFG,
        # terrain_generator=DESCRETE_OBSTACLES_CFG,
        # terrain_generator= STEPPING_STONES_CFG, #TODO
        # terrain_generator= MOAT_CFG, # proposed env
        terrain_generator= BLOCK_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    #for proposed terrain
    # stones: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
    #         rigid_objects=stones_dict
    #     )



    #左前
    stone1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_1",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.4, 0.2, -0.14)
        )
    )

    #右前
    stone2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_2",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.4, -0.2, -0.14)
        )
    )

    #ML
    stone3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_3",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0, 0.2, -0.14)
        )
    )


    #MR
    stone4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_4",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0, -0.2, -0.14)
        )
    )


    #FL
    stone5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_5",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.2, 0.2, -0.14)
        )
    )


    #FR
    stone6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_6",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.2, -0.2, -0.14)
        )
    )

    #RL
    stone7 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_7",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(-0.2, 0.2, -0.14)
        )
    )

    #RR
    stone8 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_8",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(-0.2, -0.2, -0.14)
        )
    )

    #左前2
    stone9 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_9",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.6, 0.2, -0.14)
        )
    )

    #右前2
    stone10 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Stone_10",
        spawn=sim_utils.CuboidCfg(
            size=(sizex, sizey, 0.3),  # 天板サイズ
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
            pos=(0.6, -0.2, -0.14)
        )
    )


   


    

    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[3.1, 3.1]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/ground"],
    # )

    # camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     debug_vis = True,
    #     update_period=0.02,
    #     height=64,
    #     width=64,
    #     data_types=["depth","normals"],
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
    #     # ),
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=10.0, horizontal_aperture=20.955, clipping_range=(0.1, 3.0)
    #     ),
    #     # depth_clipping_behavior = "max",
    #     depth_clipping_behavior = "zero",
    #     # offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.2), rot=(0.418761, -0.612372, 0.581238, -0.327329), convention="ros"),


    # )

    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
    #     debug_vis = True,
    #     update_period=0.0,
    #     height=64,
    #     width=64,
    #     data_types=["depth"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    # )



    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=12, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    


@configclass
class EventCfg:
    """Configuration for events."""

    reset_scene = EventTerm(
        func = mdp.reset_scene_to_default, 
        mode = "reset",
    )


    update_stone1_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone1"),  # 対象のアセット名
        }
    )

    update_stone2_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone2"),  # 対象のアセット名
        }
    )


    update_stone3_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone3"),  # 対象のアセット名
        }
    )

    update_stone4_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone4"),  # 対象のアセット名
        }
    )

    update_stone5_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone5"),  # 対象のアセット名
        }
    )


    update_stone6_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone6"),  # 対象のアセット名
        }
    )

    update_stone7_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone7"),  # 対象のアセット名
        }
    )

    update_stone8_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone8"),  # 対象のアセット名
        }
    )

    update_stone9_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone9"),  # 対象のアセット名
        }
    )

    update_stone10_mass = EventTerm(
        func=mdp.apply_mass_curriculum,
        mode="reset",  # リセット時に発火
        params={
            "asset_cfg": SceneEntityCfg("stone10"),  # 対象のアセット名
        }
    )

    # stone_1 = EventTerm(

    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("stone1"),
    #         "mass_distribution_params": (30.0, 30.0),  # 初期ステージ
    #         "operation": "abs",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     },
    # )

    # stone_2 = EventTerm(

    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("stone2"),
    #         "mass_distribution_params": (30.0, 30.0),  # 初期ステージ
    #         "operation": "abs",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     },
    # )

    # stone_3 = EventTerm(

    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("stone3"),
    #         "mass_distribution_params": (30.0, 30.0),  # 初期ステージ
    #         "operation": "abs",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     },
    # )

    # stone_4 = EventTerm(

    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("stone4"),
    #         "mass_distribution_params": (30.0, 30.0),  # 初期ステージ
    #         "operation": "abs",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     },
    # )

    # stone_5 = EventTerm(

    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("stone5"),
    #         "mass_distribution_params": (30.0, 30.0),  # 初期ステージ
    #         "operation": "abs",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     },
    # )

    # stone_6 = EventTerm(

    #     func = mdp.randomize_rigid_body_mass,
    #     mode = "reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("stone6"),
    #         "mass_distribution_params": (30.0, 30.0),  # 初期ステージ
    #         "operation": "abs",
    #         "distribution": "uniform",
    #         "recompute_inertia": True,
    #     },
    # )


    #proposed

    # reset_objects = EventTerm(
    #     func = mdp.reset_collection_to_default,
    #     mode = "reset",
    # )

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.3, 1.2),
    #         "dynamic_friction_range": (0.3, 1.2),
    #         "restitution_range": (0.0, 0.15),
    #         "num_buckets": 64,
    #     },
    # )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-1.0, 3.0),
    #         "operation": "add",
    #     },
    # )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "force_range": (0.0, 0.0),
    #         "torque_range": (-0.0, 0.0),
    #     },
    # )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         # "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    #         "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-0.5, 0.5)},
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     },
    # )

    # reset_robot_joints = EventTerm(
    #     func=mdp.reset_joints_by_scale,
    #     mode="reset",
    #     params={
    #         "position_range": (1.0, 1.0),
    #         "velocity_range": (-1.0, 1.0),
    #     },
    # )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(5.0, 10.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformLevelVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(20.0, 20.0),
    #     rel_standing_envs=0.1,
    #     debug_vis=False,
    #     # ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #     #     lin_vel_x=(0.25, 1.0), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-1, 1)
    #     # ),

    #     # ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #     #     lin_vel_x=(0.25, 1.0), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.3, 0.3)
    #     # ),

    #     ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(0.6, 1.0), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.3, 0.3)
    #     ),

    #     limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
    #     ),
    # )



    # step_fr_to_block = mdp.StepFRToBlockCommandCfg(
    #     resampling_time_range=(2.0, 3.0),
    #     local_offset_range=(-0.02, 0.02),
    #     debug_vis=True,
    # )


    step_fr_to_block = mdp.MultiLegBaseCommand3Cfg(
        resampling_time_range=(10000.0, 10000.0),
        debug_vis=True,
    )




@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        # position_commands = ObsTerm(
        #     func = mdp.generated_commands, params={"command_name": "base_velocity"}
        # )


        position_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "step_fr_to_block"}) #ベース座標系での目標

        # fr_target_xy_rel = ObsTerm(func=mdp.fr_target_xy_rel_single_block, clip=(-1.0, 1.0)) 
        # fr_target_xy_rel = ObsTerm(func=mdp.leg_xy_err) #ベース座標系での目標誤差

        fr_target_xy_rel = ObsTerm(func=mdp.legs_err_base_all) #ベース座標系での目標誤差



        leg_position = ObsTerm(func = mdp.ee_pos_base_obs)#ベース座標系での脚位置


        # position_commands = ObsTerm(
        #     func = mdp.generated_commands, params={"command_name": "goal_position"}
        # )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        ft_stack = ObsTerm(
            func=mdp.contact_ft_stack,   # 下の関数
            params=dict(
                sensor_cfg=SceneEntityCfg("contact_forces",
                # body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                body_names=".*_foot"),
                mass_kg=15.0,
                return_shape="bkl3",
            ),
            clip=(-3.0, 3.0),
        )



       


        # front_depth = ObsTerm(
        #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
        #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
        # )

        # front_normals = ObsTerm(
        #         func=mdp.image,
        #         params=dict(
        #             sensor_cfg=SceneEntityCfg("camera"),
        #             data_type="normals",
        #             normalize=False,            # ← [-1,1]のままもらう
        #         ),
        # )
        

        def __post_init__(self):
            # self.history_length = 3
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

     
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        # position_commands = ObsTerm(
        #     func = mdp.generated_commands, params={"command_name": "base_velocity"}
        # )


        position_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "step_fr_to_block"})#ベース座標系での目標

        # fr_target_xy_rel = ObsTerm(func=mdp.fr_target_xy_rel_single_block, clip=(-1.0, 1.0)) 

        # fr_target_xy_rel = ObsTerm(func=mdp.leg_xy_err) 

        fr_target_xy_rel = ObsTerm(func=mdp.legs_err_base_all) #ベース座標系での目標誤差

        leg_position = ObsTerm(func = mdp.ee_pos_base_obs)

        





        # position_commands = ObsTerm(
        #     func = mdp.generated_commands, params={"command_name": "goal_position"}
        # )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
        )
        # joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        ft_stack = ObsTerm(
            func=mdp.contact_ft_stack,   # 下の関数
            params=dict(
                sensor_cfg=SceneEntityCfg("contact_forces",
                # body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                body_names=".*_foot"),
                mass_kg=15.0,
                return_shape="bkl3",
            ),
            clip=(-3.0, 3.0),
        )

    
        # front_depth = ObsTerm(
        #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
        #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
        # )

        # front_normals = ObsTerm(
        #         func=mdp.image,
        #         params=dict(
        #             sensor_cfg=SceneEntityCfg("camera"),
        #             data_type="normals",
        #             normalize=False,            # ← [-1,1]のままもらう
        #         ),
        # )
    
    

        def __post_init__(self):
            self.concatenate_terms = False
            # self.history_length = 3

    # privileged observations
    critic: CriticCfg = CriticCfg()






@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # track_lin_vel_xy = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

    # track_lin_vel_xy = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=2.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )

  

    # position_tracking = RewTerm(
    #     func=mdp.position_command_error_tanh,
    #     weight=15.0,
    #     params={"std": 2.0, "command_name": "goal_position"},
    # )

    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_error_abs,
    #     weight=-1.0,
    #     params={"command_name": "goal_position"},
    # )



    # orientation_tracking = RewTerm(
    #     func=mdp.heading_endgame_reward,
    #     weight=5.0,
    #     params={"command_name": "goal_position"},
    # )



    # position_tracking = RewTerm(
    #     func=mdp.position_endgame_reward,
    #     weight=6.0,
    #     params={"command_name": "goal_position"}
    # )

    # move_dir_early = RewTerm(  # 初期フェーズだけ2→後で1へ
    #     func=mdp.move_in_direction_early, weight=3.0,
    #     params=dict(command_name="goal_position")
    # )


    #指定位置へ脚を置くbonus
    # fr_on_block = RewTerm(func=mdp.fr_on_block_rect, weight=0.05, params=dict(margin=0.01)) #ブロック座標系で、脚がブロック範囲内かどうか
    # fr_on_block_bonus = RewTerm(func=mdp.FROnBlockBonusOnce, weight= 1.0) #ブロック座標系、連続タッチ評価

    # bonus for all legs contact
    # fr_on_block_bonus = RewTerm(func=mdp.MultiLegHoldBonusOnce2, weight= 0.001)
    fr_on_block_bonus = RewTerm(func=mdp.MultiLegHoldBonusPhase, weight= 0.001)
    
    # そっと置く
    # impact_spike_fr = RewTerm(func=mdp.impact_spike_penalty_fr, weight=-0.3, params=dict(dfz_thresh=0.15))
    # gentle_band = RewTerm(func=mdp.support_force_band_reward, weight=0.6, params=dict(
    #         sensor_cfg=SceneEntityCfg(
    #             "contact_forces",
    #             body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],  # センサ側の定義と一致させる
    #         ),)  # 体重比基準
    # )

    # ブロックを動かさない
    # block_angvel = RewTerm(func=mdp.ang_vel, weight=-0.35)

    block_stability_penalty = RewTerm(
    func=mdp.BlocksMovementPenalty,
    weight=-1.0,  # ペナルティなのでマイナス
)

    # progress_to_stone = RewTerm(func=mdp.fr_target_progress_reward3, weight=2.5,#ワールド座標系だが内積を撮っているので問題なし、FRProgressToStoneBaseと役割がかぶる
    # )　しかも報酬内でDtをかけるようになっているので二重がけになる

    # distance_to_stone = RewTerm(func= mdp.fr_target_distance_reward_3d4, weight = 0.5) #ベース座標系での距離
    distance_to_stone = RewTerm(func= mdp.legs_reward_gaussian, weight = 0.2) #ベース座標系での距離 , all legs


    distance_progress = RewTerm(func= mdp.LegsProgressToTargetsBase, weight = 8)#ベース座標系での進捗, all legs weighted sum

    wrong_penalty = RewTerm(func = mdp.WrongPlacePenalty, weight = -0.01)


  

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    # base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-3e-5)

    # -- robot
    # dont_wait = RewTerm(
    #     func=mdp.dont_wait_rel, weight=-1.0, # 前回実装した自作関数 -2.0 for simple walking
    #     params={"velocity_threshold": 0.2, "distance_threshold": 1.0, "command_name": "goal_position"}
    # )



    # base_acc = RewTerm(func = mdp.base_accel_l2, weight = -0.0005)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0) #-5.0 for simple walking

    # for navigation
    # joint_pos = RewTerm(
    #     func=mdp.joint_position_penalty_nav,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.3,
    #         "distance_threshold": 0.5,
    #         "command_name": "goal_position",
    #     },
    # )

    # joint_pos = RewTerm(
    #     func=mdp.joint_position_penalty,
    #     weight=-0.4,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stand_still_scale": 5.0,
    #         "velocity_threshold": 0.3,
    #     },
    # )

    # -- feet
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.1,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.5,
    #     },
    # )
    # air_time_variance = RewTerm(
    #     func=mdp.air_time_variance_penalty,
    #     # weight=-1.0,
    #     weight=-0.2,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )
    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )


    # feet_contact_forces = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-0.02,
    #     params={
    #         "threshold": 100.0,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight= -2.0)

    # stand_still = RewTerm(
    #     func=mdp.stand_still,
    #     weight=0.5,  # この重みは調整が必要です
    #     params={
    #         "distance_threshold": 0.1,  # 25cm以内
    #         "heading_threshold": 0.8,    # 0.5ラジアン以内
    #         "time_threshold": 2.0,       # 最後の2秒間
    #     }
    # )

    # === 到達後の静止（負項） ===
    # stand_still_neg = RewTerm(
    #     func=mdp.stand_still_negative, weight=-1.0,
    #     params=dict(last_seconds=1.0, lin_coef=2.5, ang_coef=1.0,
    #                 command_name="goal_position", reach_radius=0.2)
    # )





    # # ==== 踏み外し（石ゾーン外 or 上面帯から下方に外れた接触） ====

    # feet_gap_pen = RewTerm(
    #     func=mdp.feet_gap_contact_penalty,
    #     weight=-20.0,   # まずはこのくらいから
    #     params={
    #         "asset_cfg":        SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg":       SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "hole_z":           -5.0,   # 固定
    #         "gap_tol":          4.75,   # 穴面+4.7 以内で接地→減点
    #         "min_contact_time": 0.02,
    #         "force_z_thresh":   60.0,   # 任意（無ければ None）
    #         "foot_sole_offset": 0.0,
    #         "normalize_by_feet": False,
    #     },
    # )
















    # feet_gap_pen = RewTerm(
    #     func=mdp.feet_gap_discrete_penalty,
    #     weight=-8.0,   # まずはこのくらいから
    #     params={
    #         "asset_cfg":        SceneEntityCfg("robot", body_names=".*_foot"),
    #         "sensor_cfg":       SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #     },
    # )

#     mass_logger = RewTerm(
#     func=mdp.LogObjectMass, # 上で作ったクラス
#     weight=0.0,         # 学習には影響させない
# )





@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})


    # ★ 追加：成功（保持できたら終了）
    # success_hold = DoneTerm(
    #     func=mdp.HoldFROnBlockWithContact,
    #     # time_outフラグは付けない（成功エピソードとしてカウントしたい）
    # )




    # success_hold = DoneTerm(
    #     func=mdp.HoldAllFeetWithContact2,
    #     # time_outフラグは付けない（成功エピソードとしてカウントしたい）
    # )

    # # ★ 追加：失敗（ブロックを動かし過ぎた）
    # block_overtilt = DoneTerm(
    #     func=mdp.block_overtilt_single,
    #     params={"limit_angle": 0.30}
    # )
    # # 任意：角速度による失敗
    # block_high_angvel = DoneTerm(
    #     func=mdp.block_high_angvel_single,
    #     params={"limit_w": 2.0}
    # )




    # block_moved = DoneTerm(
    # func=mdp.AllBlocksMovedTermination,
    # # params={
    # #     "block_names": ["stone1", "stone2", "stone3", "stone4", "stone5", "stone6"],
    # #     "pos_limit": 0.20,  # 20cm動いたらアウト
    # #     "ori_limit": 0.52,  # 30度(約0.52rad)以上回転したらアウト（傾きor回転）
    # # },

    # )


MIN_EPS=2000
UP=0.7
DOWN=0.25
D_MIN_EPS=2000
COOL_DOWN=1000
ALPHA=0.0001
# OLD=(11.0, 11.0)
OLD=(1.0, 1.0)
# OLD=(20.0, 20.0)


# STAGE=[(30.0, 30.0),
#        (28.0, 28.0),
#         (26.0, 26.0),
#         (24.0, 24.0),
#         (22.0, 22.0),
#         (20.0, 20.0)]
# STAGE=[(20.0, 20.0),
#        (19.0, 19.0),
#         (18.0, 18.0),
#         (17.0, 17.0),
#         (16.0, 16.0),
#         (15.0, 15.0),
#         (14.0, 14.0),
#         (13.0, 13.0),
#         (12.0, 12.0),
#         (11.0, 11.0)
#         ]

STAGE=[(11.0, 11.0),
       (10.0, 10.0),
        (9.0, 9.0),
        (8.0, 8.0),
        (7.0, 7.0),
        (6.0, 6.0),
        (5.0, 5.0),
        (4.0, 4.0),
        (3.0, 3.0),
        (2.0, 2.0),
        (1.0, 1.0),
        ]

# STAGE=[(1.0, 1.0),
#         (0.9, 0.9),
#         (0.8, 0.8),
#         (0.7, 0.7),
#         (0.6, 0.6),
#         (0.5, 0.5),
#         (0.4, 0.4),
#         (0.3, 0.3),
#         (0.2, 0.2),
#         (0.1, 0.1),
#         ]



@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)





    # terrain_levels = CurrTerm(func=mdp.terrain_level_curriculum) #new Curriculum

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_nav) 



    # schedule_lin = CurrTerm(
    #     func = mdp.schedule_reward_weight,
    #     params = {
    #         "term_name" : "track_lin_vel_xy",
    #         "weight": 0.8,
    #         "num_steps": 3000

    #     }
    # )

    # schedule_ang = CurrTerm(
    #     func = mdp.schedule_reward_weight,
    #     params = {
    #         "term_name" : "track_ang_vel_z",
    #         "weight": 0.5,
    #         "num_steps": 3000

    #     }
    # )

    # schedule_stone_on = CurrTerm(
    #     func = mdp.schedule_reward_weight,
    #     params = {
    #         "term_name" : "feet_on_stone",
    #         "weight": 0.5,
    #         "num_steps": 3000

    #     }
    # )

    # schedule_stone_off = CurrTerm(
    #     func = mdp.schedule_reward_weight,
    #     params = {
    #         "term_name" : "feet_gap_pen",
    #         "weight": -1.0,
    #         "num_steps": 3000

    #     }
    # )


    # schedule_lin = CurrTerm(
    #     func = mdp.schedule_reward_weight,
    #     params = {
    #         "term_name" : "distance_progress",
    #         "weight": 0,
    #         "num_steps": 20000
    #     }
    # )




    stone1_difficulty_update = CurrTerm(
    # 直接計算関数を指定する
    func=mdp.shared_mass_curriculum,
    params=dict(
        # ここに必要なパラメータを全部書く
        old_range=OLD,
        # stages=[(30.0, 30.0),
        #             (28.0, 28.0),
        #             (26.0, 26.0),
        #             (24.0, 24.0),
        #             (22.0, 22.0),
        #             (20.0, 20.0)],

        stages=STAGE,
        master=False, # ★ 1つだけ登録すればいいので True にする
        min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
    ),
)



    stone2_difficulty_update = CurrTerm(
    # 直接計算関数を指定する
    func=mdp.shared_mass_curriculum,
    params=dict(
        # ここに必要なパラメータを全部書く
        old_range=OLD,
        # stages=[(30.0, 30.0),
        #             (28.0, 28.0),
        #             (26.0, 26.0),
        #             (24.0, 24.0),
        #             (22.0, 22.0),
        #             (20.0, 20.0)],
        stages=STAGE,
        master=False, # ★ 1つだけ登録すればいいので True にする
        min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
    ),
)





    stone3_difficulty_update = CurrTerm(
    # 直接計算関数を指定する
    func=mdp.shared_mass_curriculum,
    params=dict(
        # ここに必要なパラメータを全部書く
        old_range=OLD,
        # stages=[(30.0, 30.0),
        #             (28.0, 28.0),
        #             (26.0, 26.0),
        #             (24.0, 24.0),
        #             (22.0, 22.0),
        #             (20.0, 20.0)],
        stages=STAGE,
        master=True, # ★ 1つだけ登録すればいいので True にする
        min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
    ),
)


    stone4_difficulty_update = CurrTerm(
        # 直接計算関数を指定する
        func=mdp.shared_mass_curriculum,
        params=dict(
            # ここに必要なパラメータを全部書く
            old_range=OLD,
            # stages=[(30.0, 30.0),
            #             (28.0, 28.0),
            #             (26.0, 26.0),
            #             (24.0, 24.0),
            #             (22.0, 22.0),
            #             (20.0, 20.0)],
            stages=STAGE,
            master=False, # ★ 1つだけ登録すればいいので True にする
            min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,       
        ),


    )

    stone5_difficulty_update = CurrTerm(
        # 直接計算関数を指定する
        func=mdp.shared_mass_curriculum,
        params=dict(
            # ここに必要なパラメータを全部書く
            old_range=OLD,
            # stages=[(30.0, 30.0),
            #             (28.0, 28.0),
            #             (26.0, 26.0),
            #             (24.0, 24.0),
            #             (22.0, 22.0),
            #             (20.0, 20.0)],
            stages=STAGE,
            master=False, # ★ 1つだけ登録すればいいので True にする
            min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
        ),
    )

    stone6_difficulty_update = CurrTerm(
        # 直接計算関数を指定する
        func=mdp.shared_mass_curriculum,
        params=dict(
            # ここに必要なパラメータを全部書く
            old_range=OLD,
            # stages=[(30.0, 30.0),
            #             (28.0, 28.0),
            #             (26.0, 26.0),
            #             (24.0, 24.0),
            #             (22.0, 22.0),
            #             (20.0, 20.0)],
            stages=STAGE,
            master=False, # ★ 1つだけ登録すればいいので True にする
            min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
        ),
    )

    stone7_difficulty_update = CurrTerm(
        # 直接計算関数を指定する
        func=mdp.shared_mass_curriculum,
        params=dict(
            # ここに必要なパラメータを全部書く
            old_range=OLD,
            # stages=[(30.0, 30.0),
            #             (28.0, 28.0),
            #             (26.0, 26.0),
            #             (24.0, 24.0),
            #             (22.0, 22.0),
            #             (20.0, 20.0)],
            stages=STAGE,
            master=False, # ★ 1つだけ登録すればいいので True にする
            min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
        ),
    )

    stone8_difficulty_update = CurrTerm(
        # 直接計算関数を指定する
        func=mdp.shared_mass_curriculum,
        params=dict(
            # ここに必要なパラメータを全部書く
            old_range=OLD,
            # stages=[(30.0, 30.0),
            #             (28.0, 28.0),
            #             (26.0, 26.0),
            #             (24.0, 24.0),
            #             (22.0, 22.0),
            #             (20.0, 20.0)],
            stages=STAGE,
            master=False, # ★ 1つだけ登録すればいいので True にする
            min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
        ),
    )


    stone9_difficulty_update = CurrTerm(
        # 直接計算関数を指定する
        func=mdp.shared_mass_curriculum,
        params=dict(
            # ここに必要なパラメータを全部書く
            old_range=OLD,
            # stages=[(30.0, 30.0),
            #             (28.0, 28.0),
            #             (26.0, 26.0),
            #             (24.0, 24.0),
            #             (22.0, 22.0),
            #             (20.0, 20.0)],
            stages=STAGE,
            master=False, # ★ 1つだけ登録すればいいので True にする
            min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
        ),
    )

    stone10_difficulty_update = CurrTerm(
        # 直接計算関数を指定する
        func=mdp.shared_mass_curriculum,
        params=dict(
            # ここに必要なパラメータを全部書く
            old_range=OLD,
            # stages=[(30.0, 30.0),
            #             (28.0, 28.0),
            #             (26.0, 26.0),
            #             (24.0, 24.0),
            #             (22.0, 22.0),
            #             (20.0, 20.0)],
            stages=STAGE,
            master=False, # ★ 1つだけ登録すればいいので True にする
            min_eps=MIN_EPS,
            up_rate=UP,
            down_rate=DOWN,
            down_min_eps=D_MIN_EPS,
            cooldown_steps=COOL_DOWN,
            alpha=ALPHA,      
        ),
    )




#     stone1_mass_curriculum = CurrTerm(
#     func=mdp.modify_term_cfg,
#     params=dict(
#         address="events.stone_1.params.mass_distribution_params",
#         modify_fn=mdp.shared_mass_curriculum,
#         modify_params=dict(
#             stages=[(30.0, 30.0),
#                     (28.0, 28.0),
#                     (26.0, 26.0),
#                     (24.0, 24.0),
#                     (22.0, 22.0),
#                     (20.0, 20.0)],
#             master=True,        # ★ ここだけ True
#             up_successes=64,
#             min_eps=1000,
#             up_rate=0.7,
#             down_rate=0.25,
#             down_min_eps=1000,
#             cooldown_steps=10,
#             alpha=0.2,         # 連続的な変化にしたければ調整
#         ),
#     ),
# )

# Stone2 用（master=False）
# stone2_mass_curriculum = CurrTerm(
#     func=mdp.modify_term_cfg,
#     params=dict(
#         address="events.stone_2.params.mass_distribution_params",
#         modify_fn=mdp.shared_mass_curriculum,
#         modify_params=dict(
#             stages=[(30.0, 30.0),
#                     (28.0, 28.0),
#                     (26.0, 26.0),
#                     (24.0, 24.0),
#                     (22.0, 22.0),
#                     (20.0, 20.0)],
#             master=False,       # ★ こちらは False
#             # 他のパラメータは master 側と同じで OK
#             up_successes=64,
#             min_eps=100,
#             up_rate=0.7,
#             down_rate=0.25,
#             down_min_eps=100,
#             cooldown_steps=0,
#             alpha=0.2,
#         ),
#     ),
# )

# stone3_mass_curriculum = CurrTerm(
#     func=mdp.modify_term_cfg,
#     params=dict(
#         address="events.stone_3.params.mass_distribution_params",
#         modify_fn=mdp.shared_mass_curriculum,
#         modify_params=dict(
#             stages=[(30.0, 30.0),
#                     (28.0, 28.0),
#                     (26.0, 26.0),
#                     (24.0, 24.0),
#                     (22.0, 22.0),
#                     (20.0, 20.0)],
#             master=False,       # ★ こちらは False
#             # 他のパラメータは master 側と同じで OK
#             up_successes=64,
#             min_eps=100,
#             up_rate=0.7,
#             down_rate=0.25,
#             down_min_eps=100,
#             cooldown_steps=0,
#             alpha=0.2,
#         ),
#     ),
# )

# stone4_mass_curriculum = CurrTerm(
#     func=mdp.modify_term_cfg,
#     params=dict(
#         address="events.stone_4.params.mass_distribution_params",
#         modify_fn=mdp.shared_mass_curriculum,
#         modify_params=dict(
#             stages=[(30.0, 30.0),
#                     (28.0, 28.0),
#                     (26.0, 26.0),
#                     (24.0, 24.0),
#                     (22.0, 22.0),
#                     (20.0, 20.0)],
#             master=False,       # ★ こちらは False
#             # 他のパラメータは master 側と同じで OK
#             up_successes=64,
#             min_eps=100,
#             up_rate=0.7,
#             down_rate=0.25,
#             down_min_eps=100,
#             cooldown_steps=0,
#             alpha=0.2,
#         ),
#     ),
# )

# stone5_mass_curriculum = CurrTerm(
#     func=mdp.modify_term_cfg,
#     params=dict(
#         address="events.stone_5.params.mass_distribution_params",
#         modify_fn=mdp.shared_mass_curriculum,
#         modify_params=dict(
#             stages=[(30.0, 30.0),
#                     (28.0, 28.0),
#                     (26.0, 26.0),
#                     (24.0, 24.0),
#                     (22.0, 22.0),
#                     (20.0, 20.0)],
#             master=False,       # ★ こちらは False
#             # 他のパラメータは master 側と同じで OK
#             up_successes=64,
#             min_eps=100,
#             up_rate=0.7,
#             down_rate=0.25,
#             down_min_eps=100,
#             cooldown_steps=0,
#             alpha=0.2,
#         ),
#     ),
# )

# stone6_mass_curriculum = CurrTerm(
#     func=mdp.modify_term_cfg,
#     params=dict(
#         address="events.stone_6.params.mass_distribution_params",
#         modify_fn=mdp.shared_mass_curriculum,
#         modify_params=dict(
#             stages=[(30.0, 30.0),
#                     (28.0, 28.0),
#                     (26.0, 26.0),
#                     (24.0, 24.0),
#                     (22.0, 22.0),
#                     (20.0, 20.0)],
#             master=False,       # ★ こちらは False
#             # 他のパラメータは master 側と同じで OK
#             up_successes=64,
#             min_eps=100,
#             up_rate=0.7,
#             down_rate=0.25,
#             down_min_eps=100,
#             cooldown_steps=0,
#             alpha=0.2,
#         ),
#     ),
# )

    

   




@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""
    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # scene: RobotSceneCfg = RobotSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # self.render_on_reset = True

        # エラーが ~820,000 を要求しているので、それより大きい2のべき乗（例: 2**20）に設定するのが一般的です。
        self.sim.physx.gpu_max_rigid_patch_count = 900000 # 例：約100万 (1,048,576) に設定

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        # self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training


       


        # self.scene.terrain.terrain_generator.curriculum = False




        # if getattr(self.curriculum, "terrain_levels", None) is not None:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = True
        # else:
        #     if self.scene.terrain.terrain_generator is not None:
        #         self.scene.terrain.terrain_generator.curriculum = False



        
        
        




# @configclass
# class RobotPlayEnvCfg(RobotEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()
#         self.scene.num_envs = 32
#         self.scene.terrain.terrain_generator.num_rows = 2
#         self.scene.terrain.terrain_generator.num_cols = 1
#         self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        # 親クラスの設定をまず継承する
        super().__post_init__()

        # --- 以下は再生（テスト）時専用の設定 ---
        
        # 表示する環境の数を減らす
        self.scene.num_envs = 2

        # self.scene.env_spacing = 8.0
        
        # 表示する地形のサイズを小さくする
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
        
        # [最終修正] 古い速度コマンドの行を、新しい位置コマンドの設定に変更
        # hasattrで "base_position" が存在するか安全にチェック

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # remove random pushing
        # self.events.base_external_force_torque = None
        # self.events.push_robot = None

        self.scene.terrain.terrain_generator.curriculum = False

        # self.scene.terrain.terrain_levels = 0