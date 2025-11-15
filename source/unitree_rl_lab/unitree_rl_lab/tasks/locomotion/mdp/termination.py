


from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.utils.math import quat_apply_inverse


from .helpers_single_block import _block_pos_w, _block_quat_w, _block_ang_vel_w, _yaw_from_quat
from .helpers_single_block import _base_pos_xy, _base_yaw



def _block_center_xy(env, key="stone2"):
    return _block_pos_w(env, key)[..., :2]  # [B,2]

def _block_yaw_w(env, key="stone2"):
    return _yaw_from_quat(_block_quat_w(env, key))      # [B]

def _block_theta(env, key="stone2"):
    q = _block_quat_w(env, key)                         # [B,4]
    w,x,y,z = q.unbind(-1)
    zc = 1.0 - 2.0*(x*x + y*y)
    return torch.arccos(torch.clamp(zc, -1.0, 1.0))     # [B]

def _block_wmag(env, key="stone2"):
    w = _block_ang_vel_w(env, key)                      # [B,3]
    return torch.linalg.norm(w, dim=-1)


    


def success_hold_fr_single_block(env, T_hold_s=0.5):
    return (env._buf.hold_t >= T_hold_s)

# 失敗：ブロックが大きく傾いたら True（学習を切り上げ）
def block_overtilt_single(env, limit_angle=0.30):  # 例: 0.30rad ≈ 17°
    theta = _block_theta(env)
    return (theta > limit_angle)

# （任意）ブロックの角速度が高すぎたら失敗
def block_high_angvel_single(env, limit_w=2.0):    # rad/s
    return (_block_wmag(env) > limit_w)