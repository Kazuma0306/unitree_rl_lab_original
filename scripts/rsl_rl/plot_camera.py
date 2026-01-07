import h5py
import matplotlib.pyplot as plt

# path = "teacher_static.h5"  # ←あなたのファイル名に合わせて
# with h5py.File(path, "r") as f:
#     print("keys:", list(f.keys()))
#     rgb = f["rgb"]  # (N,H,W,3) uint8

#     print("rgb shape:", rgb.shape, "dtype:", rgb.dtype)

#     # 例：先頭5枚を表示
#     for i in range(5):
#         img = rgb[i]  # HWC
#         plt.figure()
#         plt.title(f"rgb[{i}]")
#         plt.imshow(img)
#         plt.axis("off")
#     plt.show()




import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------
# ここだけあなたの設定に合わせる
# ---------------------------
PATH = "/home/digital/isaac_ws/unitree_rl_lab/teacher_static.h5"

# BEV（raycasterと合わせる）
L = 1.2
W = 0.6
Z_PLANE = 0.0   # worldの地面高さ（あなたの前提）

# カメラ intrinsics（Isaacの PinholeCameraCfg から近似）
IMG_H = 64
IMG_W = 64
FOCAL_LEN = 10.5          # focal_length
H_APERTURE = 20.955       # horizontal_aperture
# vertical_aperture は未指定なら横とアスペクトで決まる想定（正方形なら同じ）
V_APERTURE = H_APERTURE * (IMG_H / IMG_W)

# 近似 pinhole: fx = Wpx * f / sensor_width, fy = Hpx * f / sensor_height
fx = IMG_W * FOCAL_LEN / H_APERTURE
fy = IMG_H * FOCAL_LEN / V_APERTURE
cx = (IMG_W - 1) * 0.5
cy = (IMG_H - 1) * 0.5

# カメラ offset（base -> cam）
# OffsetCfg.pos / rot をそのまま入れる（rotは wxyz）
t_bc = torch.tensor([0.370, 0.0, 0.15], dtype=torch.float32)
q_bc_wxyz = torch.tensor([0.4056, -0.5792, 0.4056, -0.5792], dtype=torch.float32)

# ---------------------------
# 回転・姿勢ユーティリティ
# ---------------------------
def quat_to_rotmat_wxyz(q):
    """q: (...,4) in (w,x,y,z) -> R: (...,3,3)"""
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1] + (3,3))
    return R

def yaw_from_quat_wxyz(q):
    """world Z-yaw from quaternion wxyz"""
    w,x,y,z = q.unbind(-1)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    return torch.atan2(siny, cosy)

def rotz(yaw):
    c = torch.cos(yaw); s = torch.sin(yaw)
    R = torch.zeros(yaw.shape + (3,3), dtype=yaw.dtype, device=yaw.device)
    R[...,0,0]=c; R[...,0,1]=-s
    R[...,1,0]=s; R[...,1,1]=c
    R[...,2,2]=1
    return R

@torch.no_grad()
def rgb_to_bev_ipm(rgb_bchw, base_pos_w, base_quat_wxyz, Hb, Wb):
    """
    rgb_bchw: (1,3,H,W) float [0,1]
    base_pos_w: (3,)
    base_quat_wxyz: (4,)
    Hb,Wb: BEV grid size
    returns: bev (1,3,Hb,Wb)
    """

    dev = rgb_bchw.device
    dtype = rgb_bchw.dtype

    # BEVセル中心 (x_b,y_b) を作る（raycasterと同じ範囲）
    xs = torch.linspace(-L/2, L/2, Hb, device=dev, dtype=dtype)
    ys = torch.linspace(-W/2, W/2, Wb, device=dev, dtype=dtype)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")  # (Hb,Wb)
    pw = torch.stack([xg, yg, torch.zeros_like(xg)], dim=-1)  # (Hb,Wb,3)

    # world上の平面点（yawのみで回す：attach_yaw_only相当）
    yaw = yaw_from_quat_wxyz(base_quat_wxyz[None])[0]
    R_yaw = rotz(yaw)  # (3,3)

    p0 = base_pos_w.clone()
    p0[2] = Z_PLANE
    Xw = (R_yaw[None,None,:,:] @ pw[...,None]).squeeze(-1) + p0[None,None,:]  # (Hb,Wb,3)

    # カメラの world pose を合成（フル姿勢で）
    R_wb = quat_to_rotmat_wxyz(base_quat_wxyz[None])[0]  # (3,3)
    R_bc = quat_to_rotmat_wxyz(q_bc_wxyz[None].to(dev))[0].to(dtype)         # (3,3)
    t_bc_ = t_bc.to(dev, dtype)

    R_wc = R_wb @ R_bc
    t_wc = base_pos_w + (R_wb @ t_bc_)  # world位置

    # world -> cam
    R_cw = R_wc.T
    t_cw = -(R_cw @ t_wc)

    Xc = (R_cw[None,None,:,:] @ Xw[...,None]).squeeze(-1) + t_cw[None,None,:]  # (Hb,Wb,3)

    Z = Xc[...,2].clamp(min=1e-6)   # ROS optical: +Z forward として投影
    u = fx * (Xc[...,0] / Z) + cx
    v = fy * (Xc[...,1] / Z) + cy

    Himg, Wimg = rgb_bchw.shape[-2], rgb_bchw.shape[-1]
    grid_x = 2.0 * (u / (Wimg - 1.0)) - 1.0
    grid_y = 2.0 * (v / (Himg - 1.0)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)[None, ...]  # (1,Hb,Wb,2)

    bev = F.grid_sample(rgb_bchw, grid, mode="bilinear",
                        padding_mode="zeros", align_corners=True)
    return bev




# with h5py.File(PATH, "r") as f:
#     def show(name, obj):
#         if isinstance(obj, h5py.Dataset):
#             print(name, obj.shape, obj.dtype)
#     f.visititems(show)



# # ---------------------------
# # 可視化
# # ---------------------------
# with h5py.File(PATH, "r") as f:
#     print("keys:", list(f.keys()))
#     rgb = f["rgb"]            # (N,H,W,3) 想定
#     root = f["root_state_w"]          # (N,13) 想定
#     hits_z = f["hits_z"]      # (N,Hb,Wb) があるなら BEVサイズに使う

#     N = rgb.shape[0]
#     Hb, Wb = hits_z.shape[1], hits_z.shape[2]

#     for i in range(min(5, N)):
#         img = rgb[i]  # HWC uint8
#         # torchへ
#         img_t = torch.from_numpy(img).float() / 255.0  # (H,W,3)
#         img_t = img_t.permute(2,0,1).unsqueeze(0)      # (1,3,H,W)

#         r = np.array(root[i], dtype=np.float32)
#         base_pos_w = torch.from_numpy(r[0:3])
#         base_quat_wxyz = torch.from_numpy(r[3:7])  # IsaacLab root_state_w は通常 wxyz

#         bev = rgb_to_bev_ipm(img_t, base_pos_w, base_quat_wxyz, Hb, Wb)  # (1,3,Hb,Wb)
#         bev_np = bev[0].permute(1,2,0).clamp(0,1).cpu().numpy()

#         plt.figure(figsize=(8,4))
#         plt.subplot(1,2,1)
#         plt.title(f"rgb[{i}]")
#         plt.imshow(img)
#         plt.axis("off")

#         plt.subplot(1,2,2)
#         plt.title(f"BEV z={Z_PLANE}  HbxWb={Hb}x{Wb}")
#         plt.imshow(bev_np)
#         plt.axis("off")

#         plt.tight_layout()

#     plt.show()





import h5py, numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- 固定パラメータ（あなたの設定）----
PATH = "teacher_static.h5"
L, W = 1.2, 0.6
Z_PLANE = 0.0

IMG_H = 64; IMG_W = 64
FOCAL_LEN = 10.5
H_APERTURE = 20.955
V_APERTURE = H_APERTURE * (IMG_H / IMG_W)

fx = IMG_W * FOCAL_LEN / H_APERTURE
fy = IMG_H * FOCAL_LEN / V_APERTURE
cx = (IMG_W - 1) * 0.5
cy = (IMG_H - 1) * 0.5

t_off = torch.tensor([0.370, 0.0, 0.15], dtype=torch.float32)
q_off_wxyz = torch.tensor([0.4056, -0.5792, 0.4056, -0.5792], dtype=torch.float32)

def quat_to_R_wxyz(q):
    w,x,y,z = q.unbind(-1)
    ww,xx,yy,zz = w*w,x*x,y*y,z*z
    wx,wy,wz = w*x,w*y,w*z
    xy,xz,yz = x*y,x*z,y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1]+(3,3))
    return R

def quat_to_R_xyzw(q):
    x,y,z,w = q.unbind(-1)
    return quat_to_R_wxyz(torch.stack([w,x,y,z], dim=-1))

def yaw_from_quat_wxyz(q):
    w,x,y,z = q.unbind(-1)
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    return torch.atan2(siny, cosy)

def rotz(yaw):
    c = torch.cos(yaw); s = torch.sin(yaw)
    R = torch.zeros(yaw.shape+(3,3), dtype=yaw.dtype, device=yaw.device)
    R[...,0,0]=c; R[...,0,1]=-s
    R[...,1,0]=s; R[...,1,1]=c
    R[...,2,2]=1
    return R

def make_bev_world_points(base_pos_w, base_quat_wxyz, Hb, Wb):
    # BEV grid in base-yaw frame
    xs = torch.linspace(-L/2, L/2, Hb, dtype=torch.float32)
    ys = torch.linspace(-W/2, W/2, Wb, dtype=torch.float32)
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    pw = torch.stack([xg, yg, torch.zeros_like(xg)], dim=-1)  # (Hb,Wb,3)

    yaw = yaw_from_quat_wxyz(base_quat_wxyz[None])[0]
    R_y = rotz(yaw)
    p0 = base_pos_w.clone()
    p0[2] = Z_PLANE
    Xw = (R_y[None,None,:,:] @ pw[...,None]).squeeze(-1) + p0[None,None,:]
    return Xw  # (Hb,Wb,3)

def compose_Twc(base_pos_w, base_R_wb, off_R, off_t, off_is_cam_to_base: bool):
    # off: either cam->base or base->cam
    if off_is_cam_to_base:
        R_wc = base_R_wb @ off_R
        t_wc = base_pos_w + (base_R_wb @ off_t)
    else:
        # invert base->cam to cam->base
        R_cb = off_R
        t_cb = off_t
        R_bc = R_cb.transpose(-1,-2)
        t_bc = -(R_bc @ t_cb)
        R_wc = base_R_wb @ R_bc
        t_wc = base_pos_w + (base_R_wb @ t_bc)
    return R_wc, t_wc  # cam->world

def project_uv(Xc, depth_axis: str):
    # depth_axis: "Z" (optical) or "X" (ros-forward)
    if depth_axis == "Z":
        Z = Xc[...,2].clamp(min=1e-6)
        u = fx*(Xc[...,0]/Z) + cx
        v = fy*(Xc[...,1]/Z) + cy
        depth = Xc[...,2]
    else:  # "X"
        Z = Xc[...,0].clamp(min=1e-6)
        u = fx*(Xc[...,1]/Z) + cx
        v = fy*(Xc[...,2]/Z) + cy
        depth = Xc[...,0]
    return u, v, depth

@torch.no_grad()
def try_one(img_bchw, root13, Hb, Wb, quat_order, off_is_cam_to_base, depth_axis):
    # root
    base_pos_w = torch.from_numpy(root13[0:3]).float()
    q = torch.from_numpy(root13[3:7]).float()
    if quat_order == "wxyz":
        base_R = quat_to_R_wxyz(q[None])[0]
        base_q_wxyz = q
    else:
        base_R = quat_to_R_xyzw(q[None])[0]
        # yaw用にwxyzへ
        base_q_wxyz = torch.stack([q[3],q[0],q[1],q[2]])

    # off
    off_R = quat_to_R_wxyz(q_off_wxyz[None])[0]
    off_t = t_off

    # points on world plane
    Xw = make_bev_world_points(base_pos_w, base_q_wxyz, Hb, Wb)  # (Hb,Wb,3)

    # cam pose
    R_wc, t_wc = compose_Twc(base_pos_w, base_R, off_R, off_t, off_is_cam_to_base)
    R_cw = R_wc.transpose(-1,-2)
    t_cw = -(R_cw @ t_wc)


    # Camera offset (wxyz) base->cam （反転しない）
    # t_bc = torch.tensor([0.370, 0.0, 0.15], dtype=torch.float32)
    # q_bc_wxyz = torch.tensor([0.4056, -0.5792, 0.4056, -0.5792], dtype=torch.float32)

    # # 1) base world回転は「カメラ姿勢」にはフル回転を使う（ロール/ピッチ含む）
    # R_wb = quat_to_R_wxyz(base_q_wxyz[None])[0]
    # R_bc = quat_to_R_wxyz(q_bc_wxyz[None])[0]
    # R_wc = R_wb @ R_bc
    # t_wc = base_pos_w.float() + (R_wb @ t_bc)

    # # world->cam
    # R_cw = R_wc.T
    # t_cw = -(R_cw @ t_wc)

    # Xw = make_bev_world_points(base_pos_w, base_q_wxyz, Hb, Wb)  # (Hb,Wb,3)


    # world->cam
    Xc = (R_cw[None,None,:,:] @ Xw[...,None]).squeeze(-1) + t_cw[None,None,:]

    u, v, depth = project_uv(Xc, depth_axis)
    inb = (u >= 0) & (u <= IMG_W-1) & (v >= 0) & (v <= IMG_H-1) & (depth > 0.05)
    ratio = inb.float().mean().item()

    # sample
    grid_x = 2.0*(u/(IMG_W-1.0)) - 1.0
    grid_y = 2.0*(v/(IMG_H-1.0)) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)[None]  # (1,Hb,Wb,2)

    bev = F.grid_sample(img_bchw, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return ratio, bev

# with h5py.File(PATH, "r") as f:
#     rgb = f["rgb"]
#     root = f["root_state_w"]
#     Hb, Wb = f["hits_z"].shape[1], f["hits_z"].shape[2]

#     i = 0
#     img = rgb[i]  # HWC uint8
#     img_t = torch.from_numpy(img).float()/255.0
#     img_t = img_t.permute(2,0,1).unsqueeze(0)  # (1,3,H,W)
#     r = np.array(root[i], dtype=np.float32)

#     best = None
#     for quat_order in ["wxyz"]:
#         for off_is_cam_to_base in [False]:
#             for depth_axis in ["Z"]:
#                 ratio, bev = try_one(img_t, r, Hb, Wb, quat_order, off_is_cam_to_base, depth_axis)
#                 cfg = (quat_order, off_is_cam_to_base, depth_axis)
#                 if (best is None) or (ratio > best[0]):
#                     best = (ratio, cfg, bev)

#     ratio, cfg, bev = best
#     print("best in-bounds ratio:", ratio, "cfg:", cfg)

#     bev_np = bev[0].permute(1,0,2).clamp(0,1).numpy()
#     # bev_show = bev.permute(1, 0, 2)  # (Wb,Hb,3)
#     # bev0 = bev[0]                       # (3,Hb,Wb)
#     # bev_rgb = bev0.permute(1,2,0)       # (Hb,Wb,3)  <- imshow向け
#     # bev_show = bev_rgb.permute(1,0,2)   # (Wb,Hb,3)  <- 横=xにしたい表示

#     bev0 = bev[0].permute(1, 2, 0)      # (Hb,Wb,3)  i=x, j=y
#     bev_show = torch.flip(bev0, dims=[0, 1])  # ★xもyも反転

#     x_min, x_max = -L/2, L/2


#     plt.imshow(bev_show.clamp(0,1).cpu().numpy(),
#             origin="upper",
#             extent=[W/2, -W/2, x_min, x_max],   # 横軸= y, 縦軸= x
#             aspect="equal")
#     plt.xlabel("y left [m]")
#     plt.ylabel("x forward [m]")
#     plt.title("Top-down BEV (top=+x, left=+y)")
#     plt.show()




#     plt.figure(figsize=(8,4))
#     plt.subplot(1,2,1); plt.title("RGB"); plt.imshow(img); plt.axis("off")
#     # plt.imshow(bev_show, origin="lower", extent=[0, L, W/2, -W/2], aspect="auto")

#     # plt.subplot(1,2,2); plt.title(f"BEV ratio={ratio:.3f}\n{cfg}"); plt.imshow(bev_np); plt.axis("off")
#     plt.tight_layout()
#     plt.show()



# import h5py
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# # ---- 固定パラメータ ----
# PATH = "teacher_static.h5"
# L, W = 1.2, 0.6
# Z_PLANE = 0.0

# IMG_H = 64; IMG_W = 64
# FOCAL_LEN = 10.5
# H_APERTURE = 20.955
# V_APERTURE = H_APERTURE * (IMG_H / IMG_W)

# fx = IMG_W * FOCAL_LEN / H_APERTURE
# fy = IMG_H * FOCAL_LEN / V_APERTURE
# cx = (IMG_W - 1) * 0.5
# cy = (IMG_H - 1) * 0.5

# # ベース(ロボット中心)から見たカメラの位置と姿勢 (T_bc)
# # これらは通常 "Base to Camera" の変換として定義されます
# t_off = torch.tensor([0.370, 0.0, 0.15], dtype=torch.float32) # Base前方37cm, 高さ15cm
# q_off_wxyz = torch.tensor([0.4056, -0.5792, 0.4056, -0.5792], dtype=torch.float32)

# def quat_to_R_wxyz(q):
#     w,x,y,z = q.unbind(-1)
#     ww,xx,yy,zz = w*w,x*x,y*y,z*z
#     wx,wy,wz = w*x,w*y,w*z
#     xy,xz,yz = x*y,x*z,y*z
#     R = torch.stack([
#         ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
#         2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
#         2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
#     ], dim=-1).reshape(q.shape[:-1]+(3,3))
#     return R

# def make_bev_world_points(base_pos_w, base_quat_wxyz, Hb, Wb):
#     """
#     BEV画像の各ピクセルに対応するワールド座標を計算します。
#     """
#     # 【修正点1】
#     # 画像の行(0行目)を「前方(+L/2)」に、最終行を「後方(-L/2)」に対応させる
#     # これにより画像の上側がロボットの前方になります
#     xs = torch.linspace(L/2, -L/2, Hb, dtype=torch.float32) 
    
#     # 画像の列(0列目)は「左(-W/2)」、最終列は「右(W/2)」
#     ys = torch.linspace(-W/2, W/2, Wb, dtype=torch.float32)
    
#     xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    
#     # ベースフレームでのグリッド座標 (Hb, Wb, 3)
#     p_base = torch.stack([xg, yg, torch.zeros_like(xg)], dim=-1)

#     # Base -> World 変換
#     # 1. Baseの回転行列を取得 (Yawのみ考慮する場合と、全姿勢考慮する場合があるが、BEV平面固定ならYawのみで良いことが多い)
#     # ここでは元のコードの意図を汲み、Yaw回転だけ適用してZ平面に貼り付けます
#     base_R = quat_to_R_wxyz(base_quat_wxyz)
    
#     # シンプルに World = R_wb @ p_base + t_wb で計算
#     # ※元コードのyaw計算とrotzは正しいですが、行列演算で一括処理します
#     Xw = (base_R[None, None, :, :] @ p_base[..., None]).squeeze(-1) + base_pos_w[None, None, :]
    
#     # Z座標を強制的に固定平面にする場合（地面投影）
#     Xw[..., 2] = Z_PLANE
    
#     return Xw

# def get_cam_pose_cw(base_pos_w, base_quat_wxyz):
#     """
#     World -> Camera の変換行列 (R_cw, t_cw) を計算
#     """
#     # 1. T_wb (World -> Base)
#     R_wb = quat_to_R_wxyz(base_quat_wxyz)
#     t_wb = base_pos_w

#     # 2. T_bc (Base -> Camera) 固定オフセット
#     R_bc = quat_to_R_wxyz(q_off_wxyz)
#     t_bc = t_off

#     # 3. T_wc (World -> Camera) = T_wb * T_bc
#     # R_wc = R_wb @ R_bc
#     # t_wc = R_wb @ t_bc + t_wb
#     R_wc = R_wb @ R_bc
#     t_wc = (R_wb @ t_bc) + t_wb

#     # 4. T_cw (Camera -> World) = inv(T_wc)
#     # 直交行列の逆行列は転置
#     R_cw = R_wc.t()
#     t_cw = -(R_cw @ t_wc)
    
#     return R_cw, t_cw

# def project_uv(Xc):
#     """
#     カメラ座標系 (Z-forward) の点を画像平面 (u, v) に投影
#     """
#     # 典型的なカメラ座標系: X=Right, Y=Down, Z=Forward
#     # 深度は Z 軸
#     eps = 1e-6
#     Z = Xc[..., 2].clamp(min=eps)
#     X = Xc[..., 0]
#     Y = Xc[..., 1]
    
#     u = fx * (X / Z) + cx
#     v = fy * (Y / Z) + cy
#     return u, v, Z

# @torch.no_grad()
# def generate_bev(img_tensor, root_state, Hb, Wb):
#     # root_state から位置と姿勢を抽出
#     base_pos_w = torch.from_numpy(root_state[0:3]).float()
#     # wxyzに並べ替え (h5の並び順が[pos, quat(wxyz)]であると仮定)
#     # もしh5がxyzwなら修正が必要ですが、元コードに従いwxyzとして扱います
#     base_q_wxyz = torch.from_numpy(root_state[3:7]).float() 

#     # 1. BEVグリッドのWorld座標計算
#     Xw = make_bev_world_points(base_pos_w, base_q_wxyz, Hb, Wb) # (Hb, Wb, 3)

#     # 2. World -> Camera 変換行列の計算
#     R_cw, t_cw = get_cam_pose_cw(base_pos_w, base_q_wxyz)

#     # 3. World座標をCamera座標へ変換
#     # Xc = R_cw * Xw + t_cw
#     Xc = (R_cw[None, None, :, :] @ Xw[..., None]).squeeze(-1) + t_cw[None, None, :]

#     # 4. UV投影
#     u, v, depth = project_uv(Xc)

#     # 5. grid_sample 用に正規化 (-1 ~ 1)
#     # grid_x 対応 u (幅方向), grid_y 対応 v (高さ方向)
#     grid_x = 2.0 * (u / (IMG_W - 1.0)) - 1.0
#     grid_y = 2.0 * (v / (IMG_H - 1.0)) - 1.0
    
#     # カメラの裏側(depth<=0)や画角外をマスクするための準備
#     mask = (u >= 0) & (u <= IMG_W-1) & (v >= 0) & (v <= IMG_H-1) & (depth > 0.1)
    
#     # グリッド生成 (N, H, W, 2)
#     grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    
#     # サンプリング
#     # border_paddingだと画角外が引き伸ばされるので zeros が無難
#     bev = F.grid_sample(img_tensor, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    
#     return bev, mask

# # ---- メイン処理 ----
# with h5py.File(PATH, "r") as f:
#     rgb = f["rgb"]
#     root = f["root_state_w"]
#     Hb, Wb = f["hits_z"].shape[1], f["hits_z"].shape[2]

#     i = 0  # 任意のインデックス
#     img_np = rgb[i]
#     img_t = torch.from_numpy(img_np).float() / 255.0
#     img_t = img_t.permute(2, 0, 1).unsqueeze(0) # (1, 3, H, W)
#     r = np.array(root[i], dtype=np.float32)

#     bev, mask = generate_bev(img_t, r, Hb, Wb)
    
#     # 表示
#     bev_np = bev[0].permute(1, 2, 0).clamp(0, 1).numpy()
#     mask_np = mask.float().numpy()

#     plt.figure(figsize=(10, 5))
    
#     plt.subplot(1, 3, 1)
#     plt.title("Front Camera Input")
#     plt.imshow(img_np)
#     plt.axis("off")

#     plt.subplot(1, 3, 2)
#     plt.title("BEV Output")
#     plt.imshow(bev_np)
#     # ロボット位置の図示（画像の下中央）
#     plt.plot(Wb/2, Hb, 'r^', markersize=10, label="Robot") 
#     plt.legend(loc='lower center')
#     plt.axis("off")
    
#     plt.subplot(1, 3, 3)
#     plt.title("Valid Projection Mask")
#     plt.imshow(mask_np, cmap="gray")
#     plt.axis("off")

#     plt.tight_layout()
#     plt.show()


import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---- パラメータ設定 ----
PATH = "teacher_static.h5"

# 【重要】BEVの表示範囲設定
# ロボット座標系 (Base Link) での範囲を指定します
# カメラが x=0.37, z=0.15 にあるため、死角を避けて x=0.5 あたりから描画します
X_MIN = 0.4   # 手前 (m)
X_MAX = 1.2  # 奥 (m)
Y_WIDTH = 0.8 # 横幅 (m) -> Y: +1.5 ~ -1.5

BEV_H = 128# 出力画像の高さ
BEV_W = 64  # 出力画像の幅
Z_GROUND = 0.0

# カメラ内部パラメータ
IMG_H = 64; IMG_W = 64
FOCAL_LEN = 10.5
H_APERTURE = 20.955
V_APERTURE = H_APERTURE * (IMG_H / IMG_W)
fx = IMG_W * FOCAL_LEN / H_APERTURE
fy = IMG_H * FOCAL_LEN / V_APERTURE
cx = (IMG_W - 1) * 0.5
cy = (IMG_H - 1) * 0.5

# ベース(Robot) -> カメラ(Camera) の変換
# t_off: Baseから見たCameraの位置
t_bc = torch.tensor([0.370, 0.0, 0.15], dtype=torch.float32)
# q_off: Base座標系をCamera座標系に合わせる回転 (あるいはその逆)
# 通常、TFデータは "Parent to Child" (Base -> Camera) の回転を表します
q_bc_wxyz = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=torch.float32)

def quat_to_R_wxyz(q):
    w,x,y,z = q.unbind(-1)
    ww,xx,yy,zz = w*w,x*x,y*y,z*z
    wx,wy,wz = w*x,w*y,w*z
    xy,xz,yz = x*y,x*z,y*z
    R = torch.stack([
        ww+xx-yy-zz, 2*(xy-wz),     2*(xz+wy),
        2*(xy+wz),   ww-xx+yy-zz,   2*(yz-wx),
        2*(xz-wy),   2*(yz+wx),     ww-xx-yy+zz
    ], dim=-1).reshape(q.shape[:-1]+(3,3))
    return R

def make_bev_grid_points(Hb, Wb):
    """
    BEV画像の各ピクセルに対応する「ロボット座標系(Base)」での位置(x,y,0)を作成
    画像上(row=0) = X_MAX (遠く)
    画像下(row=H) = X_MIN (手前)
    画像左(col=0) = Y_MAX (+Y, 左)
    画像右(col=W) = Y_MIN (-Y, 右)
    """
    xs = torch.linspace(X_MAX, X_MIN, Hb, dtype=torch.float32)
    ys = torch.linspace(Y_WIDTH/2, -Y_WIDTH/2, Wb, dtype=torch.float32)
    
    xg, yg = torch.meshgrid(xs, ys, indexing="ij")
    zg = torch.full_like(xg, Z_GROUND)
    
    # (Hb, Wb, 3) -> Base座標系での点群
    points_base = torch.stack([xg, yg, zg], dim=-1)
    return points_base

def transform_points_base_to_cam(points_base, R_bc, t_bc):
    """
    点群を Base座標系 -> Camera座標系 へ変換
    P_cam = R_bc^T * (P_base - t_bc)
    """
    # 1. 並進を引く
    p_centered = points_base - t_bc
    
    # 2. 回転（R_bc の逆行列＝転置行列を掛ける）
    # 行列積: (N, 3) @ (3, 3) の形にするため工夫
    # R_bc.t() @ vector
    
    R_cb = R_bc.t() # Base->Cam の逆は Cam->Base ではなく、逆回転
    
    # einsumで (Hb, Wb, 3) x (3, 3) -> (Hb, Wb, 3)
    # p_centered[... , j] * R_cb[j, i] -> out[... , i]
    points_cam = torch.einsum('...j,ij->...i', p_centered, R_cb)
    
    return points_cam

def project_to_uv(points_cam):
    """
    Camera座標系の点を画像平面へ投影
    Camera座標系定義: Z=Forward(Depth), X=Right, Y=Down (OpenCV/Standard)
    """
    eps = 1e-6
    X = points_cam[..., 0]
    Y = points_cam[..., 1]
    Z = points_cam[..., 2]
    
    # カメラの後ろにある点は無効
    valid_mask = Z > 0.1
    
    u = fx * (X / Z.clamp(min=eps)) + cx
    v = fy * (Y / Z.clamp(min=eps)) + cy
    
    return u, v, Z, valid_mask

@torch.no_grad()
def generate_bev(img_t, Hb, Wb):
    # 1. BEVグリッド（ロボット座標系）作成
    points_base = make_bev_grid_points(Hb, Wb)
    
    # 2. Base -> Camera 変換
    # 今回はRobotの移動(World座標)は無視し、Robot座標系だけで考えればOKです
    # （カメラ画像はRobotについているカメラから見たものなので）
    R_bc = quat_to_R_wxyz(q_bc_wxyz)
    points_cam = transform_points_base_to_cam(points_base, R_bc, t_bc)
    
    # 3. 投影 (UV計算)
    u, v, depth, valid_mask = project_to_uv(points_cam)
    
    # デバッグ: 投影されたUVの中心付近の値を見てみる
    center_u = u[Hb//2, Wb//2].item()
    center_v = v[Hb//2, Wb//2].item()
    center_z = depth[Hb//2, Wb//2].item()
    print(f"Grid Center -> Cam Depth: {center_z:.2f}m, UV: ({center_u:.1f}, {center_v:.1f})")

    # 4. Grid Sample用正規化 (-1 ~ 1)
    grid_x = 2.0 * (u / (IMG_W - 1.0)) - 1.0
    grid_y = 2.0 * (v / (IMG_H - 1.0)) - 1.0
    
    # マスク適用（画角外を飛ばす）
    # grid_sampleは範囲外(-1~1の外)を参照するとpadding_modeに従う
    # ここでは明示的に範囲外の値を設定する必要はないが、Z<0等はケアが必要
    # Zが負（カメラ裏）の場所は grid_x, grid_y を -2 (範囲外) に飛ばす
    invalid = ~valid_mask
    grid_x[invalid] = -2.0
    grid_y[invalid] = -2.0
    
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0) # (1, H, W, 2)
    
    # サンプリング
    bev = F.grid_sample(img_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return bev

# ---- メイン処理 ----
with h5py.File(PATH, "r") as f:
    rgb = f["rgb"]
    
    # データを1枚ロード
    i = 0
    img_np = rgb[i] # (64, 64, 3)
    img_t = torch.from_numpy(img_np).float() / 255.0
    img_t = img_t.permute(2, 0, 1).unsqueeze(0) # (1, 3, 64, 64)
    
    # 実行
    bev_out = generate_bev(img_t, BEV_H, BEV_W)
    
    # 表示
    bev_np = bev_out[0].permute(1, 2, 0).clamp(0, 1).numpy()
    
    plt.figure(figsize=(10, 5))
    
    # 元画像
    plt.subplot(1, 2, 1)
    plt.title(f"Front Camera\n(Size: {IMG_W}x{IMG_H})")
    plt.imshow(img_np)
    # 中心点目安
    plt.plot(cx, cy, 'r+', label="Center")
    plt.axis("off")
    
    # BEV
    plt.subplot(1, 2, 2)
    plt.title(f"BEV (Top View)\nUp=+X(Forward), Left=+Y")
    plt.imshow(bev_np, extent=[Y_WIDTH/2, -Y_WIDTH/2, X_MIN, X_MAX])
    plt.xlabel("Y (Left+, Right-)")
    plt.ylabel("X (Forward)")
    # ロボット位置の方向を示す矢印
    # plt.arrow(0, X_MIN, 0, 0.2, head_width=0.1, head_length=0.1, fc='red', ec='red')
    # plt.text(0, X_MIN-0.1, "Robot", ha='center', color='red')
    
    plt.show()





    import h5py
import torch
import matplotlib.pyplot as plt

# PATH, generate_bev, BEV_H, BEV_W, IMG_W, IMG_H, cx, cy, X_MIN, X_MAX, Y_WIDTH は
# あなたの既存定義をそのまま使う前提

STRIDE = 10
N_SHOW = 5
IDX0 = 0

# with h5py.File(PATH, "r") as f:
#     rgb = f["rgb"]  # (N,64,64,3) uint8
#     N = rgb.shape[0]

#     idxs = [IDX0 + k * STRIDE for k in range(N_SHOW)]
#     idxs = [i for i in idxs if i < N]
#     n = len(idxs)

#     plt.figure(figsize=(10, 3 * n))

#     for row, i in enumerate(idxs):
#         img_np = rgb[i]  # (64,64,3) uint8

#         # (1,3,64,64) float
#         img_t = torch.from_numpy(img_np).float() / 255.0
#         img_t = img_t.permute(2, 0, 1).unsqueeze(0)

#         # BEV (1,3,BEV_H,BEV_W)
#         bev_out = generate_bev(img_t, BEV_H, BEV_W)

#         # (BEV_H,BEV_W,3)
#         bev_np = bev_out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

#         # --- RGB ---
#         ax1 = plt.subplot(n, 2, row * 2 + 1)
#         ax1.set_title(f"RGB idx={i}")
#         ax1.imshow(img_np)
#         ax1.plot(cx, cy, "r+")
#         ax1.axis("off")

#         # --- BEV ---
#         ax2 = plt.subplot(n, 2, row * 2 + 2)
#         ax2.set_title(f"BEV idx={i}  (Up=+X, Left=+Y)")
#         ax2.imshow(bev_np, extent=[Y_WIDTH/2, -Y_WIDTH/2, X_MIN, X_MAX])
#         ax2.set_xlabel("Y (Left+, Right-)")
#         ax2.set_ylabel("X (Forward)")

#     plt.tight_layout()
#     plt.show()


import h5py, numpy as np
from scipy.spatial.transform import Rotation as R

PATH="teacher_static.h5"
with h5py.File(PATH,"r") as f:
    q = f["root_state_w"][:,3:7]  # (N,4) wxyz のはず
    # scipy は xyzw なので並べ替え
    q_xyzw = np.stack([q[:,1],q[:,2],q[:,3],q[:,0]], axis=1)
    eul = R.from_quat(q_xyzw).as_euler("xyz", degrees=True)  # roll,pitch,yaw
    roll = eul[:,0]; pitch = eul[:,1]; yaw = eul[:,2]
    print("roll deg mean/std/min/max:", roll.mean(), roll.std(), roll.min(), roll.max())
    print("pitch deg mean/std/min/max:", pitch.mean(), pitch.std(), pitch.min(), pitch.max())


import h5py, numpy as np
from scipy.spatial.transform import Rotation as R

with h5py.File(PATH,"r") as f:
    q = f["root_state_w"][:,3:7]  # wxyz
    q_xyzw = np.stack([q[:,1],q[:,2],q[:,3],q[:,0]], axis=1)
    eul = R.from_quat(q_xyzw).as_euler("xyz", degrees=True)
    roll, pitch, yaw = eul[:,0], eul[:,1], eul[:,2]
    print("corr(roll,yaw)=", np.corrcoef(roll, yaw)[0,1])


import numpy as np
from scipy.spatial.transform import Rotation as R

q_bc_wxyz = np.array([0.4056, -0.5792, 0.4056, -0.5792])
q_bc_xyzw = np.array([q_bc_wxyz[1], q_bc_wxyz[2], q_bc_wxyz[3], q_bc_wxyz[0]])
eul = R.from_quat(q_bc_xyzw).as_euler("xyz", degrees=True)
print("camera offset euler xyz [deg] roll,pitch,yaw =", eul)
