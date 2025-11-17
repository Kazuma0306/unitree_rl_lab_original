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


# def _yaw_from_quat(q, order="Auto"):
#     if order == "xyzw":
#         x, y, z, w = q.unbind(-1)
#     elif order == "wxyz":
#         w, x, y, z = q.unbind(-1)
#     else:  # auto: w成分が先頭か末尾かで推定（多くの姿勢で有効）
#         # いちばん大きい成分が w であることが多い性質を利用
#         idx = q.abs().argmax(dim=-1)  # [...], 先頭or末尾が多い
#         is_wxyz = (idx == 0)
#         # 先頭をwとみなすケース
#         w1, x1, y1, z1 = q.unbind(-1)
#         # 末尾をwとみなすケース
#         x2, y2, z2, w2 = q.unbind(-1)
#         yaw_wxyz = torch.atan2(2*(w1*z1 + x1*y1), 1 - 2*(y1*y1 + z1*z1))
#         yaw_xyzw = torch.atan2(2*(w2*z2 + x2*y2), 1 - 2*(y2*y2 + z2*z2))
#         # 先頭がwと推定した要素は yaw_wxyz、そうでない要素は yaw_xyzw を採用
#         return torch.where(is_wxyz, yaw_wxyz, yaw_xyzw)

#     return torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))




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

