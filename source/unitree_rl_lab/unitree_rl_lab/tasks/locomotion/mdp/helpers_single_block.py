import torch

def _block_pos_w(env, key="stone2"):
    # returns [B,3]
    p = env.scene.rigid_objects[key].data.root_pos_w
    if p.ndim == 3:   # [B,1,3] → [B,3]
        p = p[:, 0, :]
    return p  # [B,3]

def _block_quat_w(env, key="stone2"):
    # returns [B,4] (w,x,y,z)
    q = env.scene.rigid_objects[key].data.root_quat_w
    if q.ndim == 3:   # [B,1,4] → [B,4]
        q = q[:, 0, :]
    return q

def _block_ang_vel_w(env, key="stone2"):
    # returns [B,3]
    w = env.scene.rigid_objects[key].data.root_ang_vel_w
    if w.ndim == 3:   # [B,1,3] → [B,3]
        w = w[:, 0, :]
    return w

def _yaw_from_quat(q):  # q: [B,4] (w,x,y,z)
    w,x,y,z = q.unbind(-1)
    return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))  # [B]




# --- block（単一インスタンス "stone2"） ---
def _block_pos_w(env, key="stone2"):
    p = env.scene.rigid_objects[key].data.root_pos_w  # [B,3] or [B,1,3]
    return p[:, 0, :] if p.ndim == 3 else p          # [B,3]

def _block_quat_w(env, key="stone2"):
    q = env.scene.rigid_objects[key].data.root_quat_w # [B,4] or [B,1,4]
    return q[:, 0, :] if q.ndim == 3 else q          # [B,4]

def _block_ang_vel_w(env, key="stone2"):
    w = env.scene.rigid_objects[key].data.root_ang_vel_w # [B,3] or [B,1,3]
    return w[:, 0, :] if w.ndim == 3 else w              # [B,3]

# --- base（ロボットroot） ---
def _base_pos_xy(env):
    return env.scene.articulations["robot"].data.root_pos_w[..., :2]  # [B,2]

def _base_quat_w(env):
    return env.scene.articulations["robot"].data.root_quat_w          # [B,4]

def _base_yaw(env):
    return _yaw_from_quat(_base_quat_w(env))     

