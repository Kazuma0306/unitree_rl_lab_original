#Upper Model  Environment

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


from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg7 import RobotEnvCfg

LOW_LEVEL_ENV_CFG = RobotEnvCfg()







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
        # "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        # ),

        # "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
        #      proportion=0.2, border_width=0.25,  horizontal_scale = 0.01, stone_height_max = 0.01, stone_width_range = (1.0, 1.5), stone_distance_range = (0.05, 0.08),  holes_depth = -5.0, platform_width = 1.5,

        # ),

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
             proportion=0.7, platform_width = 1.0,

        ),

    },
)



sizex = 0.2
sizey = 0.4
MASS_B = 100.0

# MASS_B = 0.1



import numpy as np

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




# def make_ring_xy5(stone_w, inner_half, outer_half, gap=1e-4, margin=1e-3, y_limit=1.55):
#     """
#     - stone_w: ブロックの一辺
#     - inner_half: 中央台の半サイズ（ここでは 0.5）
#     - outer_half: 外側の半径（どこまで石を敷き詰めるか）
#     - gap: ブロック同士の間隔（ほぼ 0）
#     - margin: 台の縁からブロックまでのすき間（ほぼ 0）
#     """
#     pitch = stone_w + gap
#     half_w = stone_w / 2.0

#     # 台の縁 (inner_half) から見て
#     # 「ブロック内側の縁」がほぼ接する位置にブロック中心を置きたい：
#     #   inner_edge ≒ inner_half + margin
#     #   center = inner_edge + half_w
#     # => r0: 最初のリングの中心半径
#     r0 = inner_half + margin + half_w  # 0.5 + margin + 0.15 ≒ 0.65

#     # 正の側の中心座標を r0 から pitch 間隔で生成
#     # ブロックの外側までが outer_half を越えない範囲で。
#     max_center = outer_half - half_w
#     if r0 > max_center:
#         return []

#     coords_pos = np.arange(r0, max_center + 1e-6, pitch)
#     coords_neg = -coords_pos[::-1]  # 原点対称に

#     if coords_pos.size == 0:
#         return []

#     xs = np.concatenate((coords_neg, coords_pos))
#     ys = np.concatenate((coords_neg, coords_pos))

#     xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")

#     # L∞ノルム（チェビシェフ距離）でリング帯を取る
#     max_dist_center = np.maximum(np.abs(xs_grid), np.abs(ys_grid))

#     # 中央台＋マージンより外側
#     m_inner = (max_dist_center >= r0 - half_w)   # だいたい inner_half + margin

#     # outer_half からはみ出さない
#     m_outer = (max_dist_center <= outer_half - half_w)

#     # 前方（+x）側だけ使うなら：
#     m_positive_x = (xs_grid - half_w > 0.0)

#     # y 範囲制限（必要なら調整）
#     m_y_upper = (ys_grid + half_w < y_limit)
#     m_y_lower = (ys_grid - half_w > -y_limit)

#     m = m_inner & m_outer & m_positive_x & m_y_upper & m_y_lower

#     xs_flat, ys_flat = xs_grid[m], ys_grid[m]
#     return [(float(x), float(y)) for x, y in zip(xs_flat, ys_flat)]


# STONE_W, STONE_H, GAP = 0.2, 0.3, 0.004
# stone_xy_list = make_ring_xy4(STONE_W, GAP, inner_half=0.7, outer_half=3.37)

# stone_xy_list = [(0.6, 0.0), (0.9, 0.15), (1.2, -0.1)]



# blocks_cfg, xy_list = build_block_cfgs_for_scene(
#     stone_w=0.2, stone_h=0.3, gap=0.06,
#     inner_half=0.75, outer_half=1.75, z=0.03, mass=3.0
# )

# def make_stone_cfg(i, pos_xyz):
#     return RigidObjectCfg(
#         prim_path=f"{{ENV_REGEX_NS}}/Stone_{i:04d}",
#         spawn=sim_utils.CuboidCfg(
#             size=(0.2, 0.2, 0.3),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
#             mass_props=sim_utils.MassPropertiesCfg(mass=100),
#             collision_props=sim_utils.CollisionPropertiesCfg(

#                 collision_enabled=True,
               
#             ),
#             physics_material=sim_utils.RigidBodyMaterialCfg(
#                 static_friction=0.9, dynamic_friction=0.8, restitution=0.0
#             ),
#             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=pos_xyz)  # ここで初期配置
#     )


def make_stone_cfg(i, pos_xyz, size_xy=(0.2, 0.2), size_z=0.3):
    sx, sy = size_xy
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Stone_{i:04d}",
        spawn=sim_utils.CuboidCfg(
            size=(float(sx), float(sy), float(size_z)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=100),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9, dynamic_friction=0.8, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos_xyz)
    )


z0 = 0.3  # 石の厚みに応じた天板高さ



STONE_W, STONE_H, GAP = 0.2, 0.3, 0.004


# STONE_W = 0.3       # 0.3 x 0.3 のブロック
# STONE_H = 0.3
# GAP     = 0.01      # ピッチ = 0.31m → そこそこ詰めて並ぶ

# CENTER_HALF = 0.5   # 中央の台の半分の大きさ
# MARGIN      = 0.05  # 台の縁からブロックまでの隙間 (~5cm)

# INNER_HALF  = CENTER_HALF + MARGIN  # = 0.55
# OUTER_HALF  = 1.5                   # 好きな広さに調整

stone_xy_list = make_ring_xy4(
    stone_w=STONE_W,
    gap=GAP,
    inner_half=0.7,
    outer_half=3.37,
)


# 中央の台
# PLATFORM_SIZE = 1.0          # 1.0 x 1.0
# PLATFORM_HALF = PLATFORM_SIZE * 0.5

# # Go2 のサイズ感（大雑把）
# GO2_LENGTH = 0.70            # だいたい 70 cm
# GO2_WIDTH  = 0.31            # だいたい 31 cm

# # 石幅のレンジ [m]（例：簡単な時は 0.35、大変になると 0.22 まで細く）
# STONE_WIDTH_RANGE = (0.22, 0.35)     # (min, max)

# # 石同士のギャップのレンジ [m]（例：最初ほぼ 0、難しくなると ~0.3）
# STONE_GAP_RANGE   = (0.0, 0.30)      # (min, max)

# # リングの内側半径（台の縁+数 mm）
# INNER_MARGIN = 0.01                  # 台とのギャップほぼ 0
# INNER_HALF   = PLATFORM_HALF + INNER_MARGIN   # ≒ 0.51

# # リングの厚さ（どこまで外側に石を置くか）
# RING_WIDTH   = GO2_LENGTH * 2     # 体長の8割くらい外に広げる
# OUTER_HALF   = INNER_HALF + RING_WIDTH
# OUTER_HALF   = 5


# Y_LIMIT      = 1.2                   # 左右にどこまで石を出すか
# STONE_H      = 0.30                  # 高さは一定




# def generate_stepping_stone_ring_xy(
#     stone_w: float,
#     inner_half: float,
#     outer_half: float,
#     gap: float = 1e-4,
#     margin: float = 1e-3,
#     y_limit: float = 1.5,
#     front_only: bool = True,
# ):
#     """
#     - stone_w   : ブロック一辺
#     - inner_half: 中央台の半サイズ＋α (この内側には石を置かない)
#     - outer_half: リングの外側半径
#     - gap       : ブロック同士の隙間
#     - margin    : inner_half から石までの隙間（ほぼ0でOK）
#     - y_limit   : |y| <= y_limit の範囲だけ使う
#     - front_only: True なら +x 側だけ（ロボット前方だけ）石を置く
#     戻り値: [(x, y), ...]  （env原点基準のローカル座標）
#     """
#     pitch = stone_w + gap
#     half_w = stone_w * 0.5

#     # 台の縁 inner_half から margin だけ外側に、
#     # ブロックの内側がほぼ接するように中心半径 r0 を決める
#     #   inner_edge ≒ inner_half + margin
#     #   center    = inner_edge + half_w
#     r0 = inner_half + margin + half_w

#     max_center = outer_half - half_w
#     if r0 > max_center:
#         return []

#     # 正側の中心位置を r0 から pitch ごとに並べる
#     coords_pos = np.arange(r0, max_center + 1e-6, pitch)
#     if coords_pos.size == 0:
#         return []

#     coords_neg = -coords_pos[::-1]

#     xs = np.concatenate((coords_neg, coords_pos))
#     ys = np.concatenate((coords_neg, coords_pos))

#     xs_grid, ys_grid = np.meshgrid(xs, ys, indexing="xy")

#     # L∞ノルムでリング判定
#     max_dist_center = np.maximum(np.abs(xs_grid), np.abs(ys_grid))

#     m_inner = (max_dist_center >= r0 - half_w)   # 中央台＋margin から外側
#     m_outer = (max_dist_center <= outer_half - half_w)

#     # 前方だけにするかどうか
#     if front_only:
#         m_x = (xs_grid - half_w > 0.0)
#     else:
#         m_x = np.ones_like(xs_grid, dtype=bool)

#     # y 範囲制限
#     m_y_upper = (ys_grid + half_w < y_limit)
#     m_y_lower = (ys_grid - half_w > -y_limit)

#     m = m_inner & m_outer & m_x & m_y_upper & m_y_lower

#     xs_flat, ys_flat = xs_grid[m], ys_grid[m]
#     return [(float(x), float(y)) for x, y in zip(xs_flat, ys_flat)]







# def resolve_stone_params(difficulty: float):
#     """
#     IsaacLab公式 stepping_stones_terrain と同じ補間で
#     石幅とギャップを決める。
#     difficulty: 0.0(易) ～ 1.0(難)
#     """
#     d = float(np.clip(difficulty, 0.0, 1.0))

#     w_min, w_max = STONE_WIDTH_RANGE
#     g_min, g_max = STONE_GAP_RANGE

#     # stone_width = max - d*(max-min)
#     stone_w = w_max - d * (w_max - w_min)
#     # stone_distance = min + d*(max-min)
#     stone_gap = g_min + d * (g_max - g_min)

#     return stone_w, stone_gap




# def generate_ring_xy_isaac(
#     difficulty: float,
#     inner_half: float = INNER_HALF,
#     outer_half: float = OUTER_HALF,
#     y_limit: float = Y_LIMIT,
#     front_only: bool = True,
# ):
#     """
#     IsaacLab公式と同じ difficulty 補間で
#     stone_w / gap を決めてリング座標を返す。
#     戻り値: [(x, y), ...] （env原点基準）
#     """
#     stone_w, stone_gap = resolve_stone_params(difficulty)
#     return generate_stepping_stone_ring_xy(
#         stone_w=stone_w,
#         inner_half=inner_half,
#         outer_half=outer_half,
#         gap=stone_gap,
#         margin=1e-4,      # 台との隙間ほぼ0
#         y_limit=y_limit,
#         front_only=front_only,
#     )



# difficulty = 0.1   # カリキュラム等で決める [0,1]

# stone_xy_list = generate_ring_xy_isaac(difficulty)



# stone_xy_list = make_ring_xy5(
#     stone_w=STONE_W,
#     inner_half=CENTER_HALF,
#     outer_half=OUTER_HALF,
#     gap=1e-4,     # ブロック同士のギャップ ≒ 0
#     margin=1e-3,  # 台の縁とのギャップ ≒ 0
# )





import numpy as np

def quantize(v: float, h: float) -> float:
    """horizontal_scale の格子に合わせて丸める"""
    return round(v / h) * h

def rects_intersect(ax0, ax1, ay0, ay1, bx0, bx1, by0, by1) -> bool:
    return (ax0 < bx1) and (ax1 > bx0) and (ay0 < by1) and (ay1 > by0)



def snap_up(v, h):   return np.ceil(v / h) * h
def snap_down(v, h): return np.floor(v / h) * h
def snap_near(v, h): return np.round(v / h) * h


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def params_from_difficulty(
    difficulty: float,
    horizontal_scale: float,
    stone_width_range_m: tuple[float, float],      # (max, min) 例: (0.40, 0.20)
    stone_distance_range_m: tuple[float, float],   # (min, max) 例: (0.10, 0.35)
) -> tuple[int, int, float, float]:
    """
    IsaacLab の stepping-stones と同じ発想:
      - 幅は difficulty↑で小さく: max -> min
      - 距離は difficulty↑で大きく: min -> max
    その後 int(.../h) でピクセル化（切り捨て）する。
    """
    d = clamp01(float(difficulty))
    w_max, w_min = stone_width_range_m
    dist_min, dist_max = stone_distance_range_m

    w_m    = lerp(w_max,  w_min,  d)      # 幅は減る
    dist_m = lerp(dist_min, dist_max, d)  # 距離は増える

    w_px    = max(1, int(w_m / horizontal_scale))
    dist_px = max(0, int(dist_m / horizontal_scale))

    # “実際に使われる（=ピクセルに落ちた）”メートル値
    w_eff_m    = w_px * horizontal_scale
    dist_eff_m = dist_px * horizontal_scale
    return w_px, dist_px, w_eff_m, dist_eff_m




# def generate_xy_list_front_isaac(
#     terrain_size_xy=(8.0, 8.0),     # (Lx, Ly) [m] そのタイルの大きさ
#     horizontal_scale=0.05,          # [m] Terrain HF と揃えたい格子
#     stone_size_xy=(0.35, 0.25),     # (sx, sy) [m] ブロック天板サイズ（XY）
#     gap_xy=(0.15, 0.15),            # (gx, gy) [m] ブロック間ギャップ
#     platform_size=1.2,              # 中央台の一辺 [m]（正方形を仮定）
#     platform_center=(0.0, 0.0),     # 台中心（普通は (0,0)）
#     x_front_ratio=0.5,              # 前半のみ = 0.5（x>0 側）
#     margin=0.10,                    # 端からの余白 [m]
#     clearance=0.02,                 # 台との追加クリアランス [m]
#     per_row_phase=True,             # 行ごとに位相ずらし（stepping-stones風）
#     jitter_xy=(0.0, 0.0),           # (jx, jy) [m] 追加ランダム
#     seed=0,
# ):
#     rng = np.random.default_rng(seed)

#     Lx, Ly = terrain_size_xy
#     sx, sy = stone_size_xy
#     gx, gy = gap_xy
#     jx, jy = jitter_xy

#     # ピッチ（中心間距離）
#     px = sx + gx
#     py = sy + gy

#     # 台（platform）のAABB
#     pcx, pcy = platform_center
#     half_p = platform_size * 0.5
#     plat_x0 = pcx - half_p - clearance
#     plat_x1 = pcx + half_p + clearance
#     plat_y0 = pcy - half_p - clearance
#     plat_y1 = pcy + half_p + clearance

#     # 配置領域（中心座標で安全に収まる範囲）
#     x_min = 0.0 + margin + sx * 0.5
#     x_max = (Lx * x_front_ratio) - margin - sx * 0.5  # x>0 側だけ使う想定（原点が中心なら Lx/2 が前端）
#     y_min = -Ly * 0.5 + margin + sy * 0.5
#     y_max = +Ly * 0.5 - margin - sy * 0.5

#     # 格子に量子化（HFと揃えるなら推奨）
#     # x_min = quantize(x_min, horizontal_scale)
#     # x_max = quantize(x_max, horizontal_scale)
#     # y_min = quantize(y_min, horizontal_scale)
#     # y_max = quantize(y_max, horizontal_scale)

#     # スナップは min=ceil / max=floor
#     x_min = snap_up(x_min, horizontal_scale)
#     x_max = snap_down(x_max, horizontal_scale)
#     y_min = snap_up(y_min, horizontal_scale)
#     y_max = snap_down(y_max, horizontal_scale)

#     # px_q  = max(horizontal_scale, quantize(px, horizontal_scale))
#     # py_q  = max(horizontal_scale, quantize(py, horizontal_scale))

#     px_q = max(horizontal_scale, snap_near(px, horizontal_scale))
#     py_q = max(horizontal_scale, snap_near(py, horizontal_scale))

#     points = []

#     # y の帯（row）を走査
#     y = y_min
#     row = 0
#     while y <= y_max + 1e-9:
#         # 行ごとに x の開始位相をずらす（完全格子にしたいなら 0 に固定）
#         phase = rng.uniform(0.0, sx) if per_row_phase else 0.0
#         x = x_min + quantize(phase, horizontal_scale)

#         while x <= x_max + 1e-9:
#             # ジッター（必要なら）
#             xx = x + (rng.uniform(-jx, jx) if jx > 0 else 0.0)
#             yy = y + (rng.uniform(-jy, jy) if jy > 0 else 0.0)

#             # 格子に戻す（HFと揃える）
#             xx = quantize(xx, horizontal_scale)
#             yy = quantize(yy, horizontal_scale)

#             # 石のAABB（中心から）
#             stone_x0 = xx - sx * 0.5
#             stone_x1 = xx + sx * 0.5
#             stone_y0 = yy - sy * 0.5
#             stone_y1 = yy + sy * 0.5

#             # 中央台と交差する石は除外
#             if not rects_intersect(stone_x0, stone_x1, stone_y0, stone_y1,
#                                    plat_x0, plat_x1, plat_y0, plat_y1):
#                 points.append((xx, yy))

#             x += px_q

#         y += py_q
#         row += 1

#     # return np.asarray(points, dtype=np.float32)

#     return points



import math
import random
from typing import List, Tuple, Optional

def rect_intersect_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return (a0 < b1) and (a1 > b0)

def stepping_stones_xy_front_half_pixelwise(
    size_x_m: float,
    size_y_m: float,
    horizontal_scale: float,
    platform_width_m: float,
    difficulty: float,
    stone_width_range_m: tuple[float, float] = (0.40, 0.20),     # (max, min)
    stone_distance_range_m: tuple[float, float] = (0.10, 0.35),  # (min, max)
    margin_m: float = 0.2,              # 外周の安全マージン
    platform_clearance_m: float = 0.00,  # 台からさらに離したいなら +（0で“IsaacLabの台境界ぴったり”）
    per_row_phase: bool = True,          # 行ごとに開始位相をランダム化（IsaacLab風）
    seed: int = 0,
    max_points: Optional[int] = None,    # 石数を固定したいなら指定（足りない場合は返り値が少なくなる）
) -> tuple[List[Tuple[float, float]], dict]:
    """
    返り値:
      - xy: [(x_local, y_local), ...]  ※すべて Python float（OmegaConfでも安全）
      - meta: 実効 stone_size/gap などデバッグ情報
    """
    h = float(horizontal_scale)
    rng = random.Random(seed)

    # --- 1) IsaacLab と同じくピクセルで実効サイズを決める（intで切り捨て）
    W = int(size_x_m / h)   # x方向ピクセル数
    H = int(size_y_m / h)   # y方向ピクセル数
    size_x_eff = W * h
    size_y_eff = H * h

    cx = W // 2
    cy = H // 2

    # --- 2) difficulty から stone_width / stone_distance を決めてピクセル化
    w_px, dist_px, w_eff_m, dist_eff_m = params_from_difficulty(
        difficulty, h, stone_width_range_m, stone_distance_range_m
    )
    pitch_px = max(1, w_px + dist_px)

    # --- 3) platform も IsaacLab と同じピクセル境界で切る
    pf_px = int(platform_width_m / h)
    # IsaacLabっぽい中心切り出し（整数境界）
    px1 = (W - pf_px) // 2
    px2 = (W + pf_px) // 2
    py1 = (H - pf_px) // 2
    py2 = (H + pf_px) // 2

    # clearance を “ピクセル” で拡張（台を避けすぎるのが嫌なら 0 推奨）
    clear_px = int(platform_clearance_m / h)
    px1c, px2c = px1 - clear_px, px2 + clear_px
    py1c, py2c = py1 - clear_px, py2 + clear_px

    # --- 4) 外周マージンもピクセルで（はみ出しゼロを保証するため）
    margin_px = int(margin_m / h)

    # --- 5) まず「石パッチ左下(x0,y0)」をピクセルで走査
    #   y0 は下から上へ、x0 は “前半(x>=0)” 側だけ
    #
    # 重要: RigidObject は欠けられないので、パッチが完全に入る範囲だけにする
    #
    # ピクセルiのx座標(m) は (i - cx)*h とする（cxがx=0近辺）
    # “前半”は中心より右（i >= cx）だが、パッチ幅があるので中心側を少し避ける
    #
    # パッチが完全に入る条件: x0 >= 0側境界 かつ x0+w_px <= W-margin_px
    y0_min = margin_px
    y0_max = H - margin_px - w_px
    # x0_min = max(cx, margin_px)          # x>=0側へ
    # x0_max = W - margin_px - w_px

    # x0_min = max(px2c, cx, margin_px)
    x0_min = max(px2c + 1, cx, margin_px)

    x0_max = W - margin_px - w_px

    xy: List[Tuple[float, float]] = []

    y0 = y0_min
    row = 0
    while y0 <= y0_max:
        # 行ごとに “開始位相” をランダムにズラす（0〜w_px-1）
        phase = rng.randrange(0, w_px) if (per_row_phase and w_px > 1) else 0
        x0 = x0_min + phase

        while x0 <= x0_max:
            # --- 台との交差（ピクセル矩形で判定）
            sx0, sx1 = x0, x0 + w_px
            sy0, sy1 = y0, y0 + w_px
            hit_platform = rect_intersect_1d(sx0, sx1, px1c, px2c) and rect_intersect_1d(sy0, sy1, py1c, py2c)

            if not hit_platform:
                # 石の中心（ピクセル）→ メートル
                # ここは “ピクセル中心”に寄せておくと分かりやすい
                xc = x0 + w_px * 0.5
                yc = y0 + w_px * 0.5
                x_m = (xc - cx) * h
                y_m = (yc - cy) * h

                # --- 最終安全チェック（メートルAABBで “はみ出しゼロ”）
                # 前半のみ: x - w/2 >= 0
                if (x_m - w_eff_m * 0.5) >= 0.0 + margin_px * h and \
                   (x_m + w_eff_m * 0.5) <= (size_x_eff * 0.5 - margin_px * h) and \
                   (abs(y_m) + w_eff_m * 0.5) <= (size_y_eff * 0.5 - margin_px * h):
                    xy.append((float(x_m), float(y_m)))
                    if max_points is not None and len(xy) >= max_points:
                        meta = dict(
                            W=W, H=H, size_x_eff=size_x_eff, size_y_eff=size_y_eff,
                            stone_w_px=w_px, stone_dist_px=dist_px, pitch_px=pitch_px,
                            stone_w_eff_m=w_eff_m, stone_dist_eff_m=dist_eff_m,
                            platform_pf_px=pf_px, platform_bbox_px=(px1c, px2c, py1c, py2c),
                        )
                        return xy, meta

            x0 += pitch_px

        y0 += pitch_px
        row += 1

    meta = dict(
        W=W, H=H, size_x_eff=size_x_eff, size_y_eff=size_y_eff,
        stone_w_px=w_px, stone_dist_px=dist_px, pitch_px=pitch_px,
        stone_w_eff_m=w_eff_m, stone_dist_eff_m=dist_eff_m,
        platform_pf_px=pf_px, platform_bbox_px=(px1c, px2c, py1c, py2c),
    )
    return xy, meta



# stone_xy_list = generate_xy_list_front_isaac(
#     terrain_size_xy=(8.0, 8.0),
#     horizontal_scale=0.05,
#     stone_size_xy=(0.3, 0.3),
#     gap_xy=(0.1, 0.1),
#     platform_size=1.0,
#     x_front_ratio=0.5,     # 前半（x>0 側）だけ
#     margin=0.10,
#     clearance=0.0,
#     per_row_phase=False,
#     # seed=42,
# )


stone_xy_list, meta = stepping_stones_xy_front_half_pixelwise(
    size_x_m=8.0,
    size_y_m=8.0,
    horizontal_scale=0.02,
    platform_width_m=1.0,
    difficulty=0.67,
    stone_width_range_m=(0.50, 0.20),
    stone_distance_range_m=(0.02, 0.05),
    margin_m=0.2,
    platform_clearance_m=0.03,   # まずは 0 推奨（避けすぎを防ぐ）
    per_row_phase=False,
    # seed=123,
    max_points=None,            # num_stonesに合わせるなら apply 側で切る/退避が安全
)


actual_stone_width = meta['stone_w_eff_m']

# 床が z=0 で、石を床の上に置くなら
# z_center = STONE_H * 0.5   # 中心 = 高さの半分
z_center = z0 - STONE_H * 0.5

stones_dict = {
    f"stone{i:04d}": make_stone_cfg(
        i,
        pos_xyz=(x, y, z_center),
        size_xy = (actual_stone_width, actual_stone_width)

    )
    for i, (x, y) in enumerate(stone_xy_list, start=1)
}


stones = RigidObjectCollectionCfg(
        # prim_path="{ENV_REGEX_NS}/Stones",  # コレクションの親
        rigid_objects=stones_dict,
)
from dataclasses import field



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
        # terrain_generator= STEPPING_STONES_CFG, 
        terrain_generator= MOAT_CFG, # proposed env
        # terrain_generator= BLOCK_CFG,
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


    stones: RigidObjectCollectionCfg = field(
        default_factory=lambda: stones   # 上で作った stones を渡す
    )



    # #左前
    # stone1 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_1",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(0.4, 0.2, -0.14)
    #         pos=(1.2, 0.2, -0.14)
    #     )
    # )

    # #右前
    # stone2 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_2",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(0.4, -0.2, -0.14)
    #         pos=(1.2, -0.2, -0.14)
    #     )
    # )

    # #ML
    # stone3 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_3",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(0.8, 0.2, -0.14)
    #     )
    # )


    # #MR
    # stone4 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_4",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(0.8, -0.2, -0.14)
    #     )
    # )


    # #FL
    # stone5 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_5",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(0.2, 0.2, -0.14)
    #         pos=(1.0, 0.2, -0.14)
    #     )
    # )


    # #FR
    # stone6 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_6",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(0.2, -0.2, -0.14)
    #         pos=(1.0, -0.2, -0.14)
    #     )
    # )

    # #RL
    # stone7 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_7",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(-0.2, 0.2, -0.14)
    #         pos=(0.6, 0.2, -0.14)
    #     )
    # )

    # #RR
    # stone8 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_8",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(-0.2, -0.2, -0.14)
    #         pos=(0.6, -0.2, -0.14)
    #     )
    # )

    # #左前2
    # stone9 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_9",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(0.6, 0.2, -0.14)
    #         pos=(1.4, 0.2, -0.14)
    #     )
    # )

    # #右前2
    # stone10 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_10",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         # pos=(0.6, -0.2, -0.14)
    #         pos=(1.4, -0.2, -0.14)
    #     )
    # )


    
    # #左前3
    # stone11 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_11",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(1.6, 0.2, -0.14)
    #     )
    # )

    # #右前3
    # stone12 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_12",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(1.6, -0.2, -0.14)
    #     )
    # )


    # #左前4
    # stone13 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_13",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(1.8, 0.2, -0.14)
    #     )
    # )


    # #右前4
    # stone14= RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_14",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(1.8, -0.2, -0.14)
    #     )
    # )



    # #左前4
    # stone15 = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_15",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(2.0, 0.2, -0.14)
    #     )
    # )


    # #右前4
    # stone16= RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Stone_16",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(sizex, sizey, 0.3),  # 天板サイズ
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=MASS_B),   # ランダム化候補
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(
    #             static_friction=1.0, dynamic_friction=1.0, restitution=0.0
    #         ),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.6, 0.8))
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         # ここは各ENVの原点からの相対。与えられた足置き位置に合わせて配置する
    #         pos=(2.0, -0.2, -0.14)
    #     )
    # )









   


    

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
    #     update_period=0.02, #50Hz, Same as Low Policy, Ideal 30Hz TODO
    #     height=64,
    #     width=64,
    #     data_types=["depth"],
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5.0)
    #     # ),
    #     # spawn=sim_utils.PinholeCameraCfg(
    #     #     focal_length=16.0, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.1, 4.0)
    #     # ),

    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=10.5, focus_distance=3.0, horizontal_aperture=20.955, clipping_range=(0.1, 3.0)
    #     ),
    #     depth_clipping_behavior = "max",
    #     # depth_clipping_behavior = "zero",
    #     # offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.32, 0.0, 0.15), rot=(0.2418, -0.6645,  0.6645, -0.2418), convention="ros"),
    #     offset=CameraCfg.OffsetCfg(pos=(0.37, 0.0, 0.15), rot=(0.0616, -0.7044, 0.7044, -0.0616), convention="ros"),
    #     # offset=CameraCfg.OffsetCfg(pos=(0.05, 0.0, 1), rot = (0.0, -0.7071, 0.7071, 0.0), convention="ros"),



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



    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=12 , track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )





@configclass
class ActionsCfg:
    """Action terms for the MDP."""

    pre_trained_policy_action: mdp.FootstepPolicyActionCfg = mdp.FootstepPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt", #TODO
        policy_path=f"/home/digital/isaac_ws/unitree_rl_lab/logs/rsl_rl/unitree_go2_proposed4/2025-12-12_10-36-24/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.JointPositionAction, #lower's action
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy, # lower's observation
    )



@configclass
class HighLevelPolicyObsCfg(ObsGroup):

    # observation terms (order preserved)
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100)) #last command

    leg_position = ObsTerm(func = mdp.ee_pos_base_obs)#ベース座標系での脚位置


    # front_depth = ObsTerm(
    #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
    #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
    # )

    # heightmap = ObsTerm(
    # func=mdp.depth_heightmap4,
    # params=dict(
    #     sensor_cfg=SceneEntityCfg("camera"),  # 定義したカメラ名
    #     asset_cfg=SceneEntityCfg("robot"),
    #     x_range=(0.1, 3.0),
    #     y_range=(-2.0, 2.0),
    #     grid_shape=(64, 64),
    #     default_height=-4,
    # ),
    # )

    # heightmap = ObsTerm(
    #     func=mdp.image,
    #     # env.scene.sensors["camera"] に合わせて名前を変える
    #     params={
    #         "sensor_cfg": SceneEntityCfg("camera"),
    #         "data_type": "depth",   # TiledCameraCfg.data_types=["depth"] と合わせる
    #         "normalize": True,      # depthの inf を0にするなどの処理をしてくれる
    #     },
    #     # ↓ここがキモ：このtermだけ履歴を持たせる
    #     history_length=2,           # 4フレーム分スタック
    #     flatten_history_dim=True,   # (B,4,H,W,C) → (B, 4*H*W*C)にflatten
    # )

    heightmap = ObsTerm(
        func = mdp.obs_near_blocks_col

    )

    ft_stack = ObsTerm(
            func=mdp.contact_ft_stack,   # 下の関数
            params=dict(
                sensor_cfg=SceneEntityCfg("contact_forces",
                # body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
                body_names=".*_foot"),
                mass_kg=15.0,
                return_shape="flat",
                components="z"
            ),
            clip=(-3.0, 3.0),
        )


    def __post_init__(self):
            # self.history_length = 3
            # self.enable_corruption = True
            self.concatenate_terms = False




@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    policy: mdp.ObservationGroupCfg = HighLevelPolicyObsCfg()
    critic: mdp.ObservationGroupCfg = HighLevelPolicyObsCfg()


    # @configclass
    # class PolicyCfg(ObsGroup):
    #     """Observations for policy group."""

    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity)
    #     pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    #     # front_depth = ObsTerm(
    #     #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
    #     #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
    #     # )

    #     heightmap = ObsTerm(
    #     func=mdp.depth_heightmap,
    #     params=dict(
    #         sensor_cfg=SceneEntityCfg("tiled_camera"),  # 定義したカメラ名
    #         asset_cfg=SceneEntityCfg("robot"),
    #         x_range=(-1.0, 1.0),
    #         y_range=(-0.8, 0.8),
    #         grid_shape=(32, 32),
    #         default_height=0.0,
    #     ),
    #     )


    #     # def __post_init__(self):
    #     #     # self.history_length = 3
    #     #     self.enable_corruption = True
    #     #     self.concatenate_terms = False


    # # observation groups
    # policy: PolicyCfg = PolicyCfg()


    # @configclass
    # class CriticCfg(ObsGroup):
    #     """Observations for critic group."""
    #     # observation terms (order preserved)
    #     base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    #     base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
    #     projected_gravity = ObsTerm(func=mdp.projected_gravity)
    #     pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

    #     # front_depth = ObsTerm(
    #     #     func=mdp.image, # mdpに関数がある場合。なければ自作関数
    #     #     params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "depth"}
    #     # )

    #     heightmap = ObsTerm(
    #     func=mdp.depth_heightmap,
    #     params=dict(
    #         sensor_cfg=SceneEntityCfg("tiled_camera"),  # 定義したカメラ名
    #         asset_cfg=SceneEntityCfg("robot"),
    #         x_range=(-1.0, 1.0),
    #         y_range=(-0.8, 0.8),
    #         grid_shape=(32, 32),
    #         default_height=0.0,
    #     ),
    #     )

    #     # def __post_init__(self):
    #     #     # self.history_length = 3
    #     #     # self.enable_corruption = True
    #     #     self.concatenate_terms = False


    # critic: CriticCfg = CriticCfg()






# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
#     position_tracking = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.4,
#         params={"std": 0.2, "command_name": "pose_command"},
#     )
#     position_tracking_fine_grained = RewTerm(
#         func=mdp.position_command_error_tanh,
#         weight=0.4,
#         params={"std": 2.0, "command_name": "pose_command"},
#     )
#     orientation_tracking = RewTerm(
#         func=mdp.heading_command_error_abs,
#         weight=-0.6,
#         params={"command_name": "pose_command"},
#     )

#     distance_progress = RewTerm(func= mdp.BaseProgressToTargetRel, weight = 15)#ベース座標系での進捗, all legs weighted sum



#     # joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
#     # # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
#     # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
#     # # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
#     # energy = RewTerm(func=mdp.energy, weight=-3e-5)

#     # -- robot
#     # dont_wait = RewTerm(
#     #     func=mdp.dont_wait_rel, weight=-1.0, # 前回実装した自作関数 -2.0 for simple walking
#     #     params={"velocity_threshold": 0.2, "distance_threshold":0.8, "command_name": "pose_command"}
#     # )

#     dont_wait = RewTerm(
#         func=mdp.dont_wait_rel3, weight=-1, 
#         params={"distance_threshold": 0.2, "max_distance":0.8, "command_name": "pose_command"}
#     )




#     # base_acc = RewTerm(func = mdp.base_accel_l2, weight = -0.0005)

#     flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0) 

#     undesired_contacts = RewTerm(
#         func=mdp.undesired_contacts,
#         weight=-1.5,
#         params={
#             "threshold": 1,
#             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
#         },
#     )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.3,
        params={"std": 0.2, "command_name": "pose_command"},
    )
    position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.4,
        params={"std": 2.0, "command_name": "pose_command"},
    )
    orientation_tracking = RewTerm(
        func=mdp.heading_command_error_abs,
        weight=-0.8,
        params={"command_name": "pose_command"},
    )

    distance_progress = RewTerm(func= mdp.BaseProgressToTargetRel, weight = 15)#ベース座標系での進捗, all legs weighted sum



    # joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-2e-4)
    # # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    # # dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # energy = RewTerm(func=mdp.energy, weight=-3e-5)

    # -- robot
    # dont_wait = RewTerm(
    #     func=mdp.dont_wait_rel, weight=-1.0, # 前回実装した自作関数 -2.0 for simple walking
    #     params={"velocity_threshold": 0.2, "distance_threshold":0.8, "command_name": "pose_command"}
    # )

    dont_wait = RewTerm(
        func=mdp.dont_wait_rel3, weight=-1, 
        params={"distance_threshold": 0.2, "max_distance":0.8, "command_name": "pose_command"}
    )




    # base_acc = RewTerm(func = mdp.base_accel_l2, weight = -0.0005)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0) 

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*", ".*_hip", ".*_thigh", ".*_calf"]),
        },
    )





@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(24.0, 24.0),
        debug_vis=True,
        # ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-3.0, 3.0), pos_y=(-3.0, 3.0), heading=(-math.pi, math.pi)),
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(0.5, 2.5), pos_y=(-0.0, 0.0), heading=(-0, 0)),
    )

    step_fr_to_block = mdp.FootstepFromHighLevelCfg(
        debug_vis = True
    )







@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})




@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""


    # schedule_lin = CurrTerm(
    #     func = mdp.schedule_reward_weight,
    #     params = {
    #         "term_name" : "position_tracking_fine_grained",
    #         "weight": 0.1,
    #         "num_steps": 25000
    #     }
    # )

    schedule_lin = CurrTerm(
        func = mdp.schedule_reward_weight,
        params = {
            "term_name" : "distance_progress",
            "weight": 1,
            "num_steps": 25000
        }
    )





@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the walking environment."""

    # environment settings
    # scene: SceneEntityCfg = LOW_LEVEL_ENV_CFG.scene
    # scene: SceneEntityCfg = RobotSceneCfg(num_envs=2048, env_spacing=2.5)

    # scene: SceneEntityCfg = RobotSceneCfg(num_envs=1024, env_spacing=2.5)
    # scene: SceneEntityCfg = RobotSceneCfg(num_envs=512, env_spacing=2.5)

    scene: SceneEntityCfg = RobotSceneCfg(num_envs=256, env_spacing=2.5)
    # scene: SceneEntityCfg = RobotSceneCfg(num_envs=2, env_spacing=2.5)

    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    # events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()


    def __post_init__(self):
        """Post initialization."""

        self.sim.dt = LOW_LEVEL_ENV_CFG.sim.dt
        self.sim.render_interval = LOW_LEVEL_ENV_CFG.decimation
        self.decimation = LOW_LEVEL_ENV_CFG.decimation * 5#TODO　５Hz 10Hz
        self.episode_length_s = self.commands.pose_command.resampling_time_range[1]
        self.sim.physx.gpu_max_rigid_patch_count = 1000000 # 例：約100万 (1,048,576) に設定


       
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        
        # if self.scene.terrain.terrain_generator is not None:
        #     self.scene.terrain.terrain_generator.num_rows = 2
        #     self.scene.terrain.terrain_generator.num_cols = 2





class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2

        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False