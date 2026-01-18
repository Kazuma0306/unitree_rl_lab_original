import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# def show_one(h5_path, idx=0, show_hits=False, show_trav=True):
#     with h5py.File(h5_path, "r") as f:
#         L, W = f.attrs["ray_size_xy"]          # [1.2, 0.6]
#         res = float(f.attrs["ray_resolution"]) # 0.02
#         hits_z = f["hits_z"][idx]              # (Hb,Wb)
#         trav  = f["trav"][idx]                 # (Hb,Wb) uint8
#         foot_xy = f["foot_xy"][idx]            # (4,2)

#     Hb, Wb = hits_z.shape
#     extent = [-L/2, L/2, -W/2, W/2]            # x:[-0.6,0.6], y:[-0.3,0.3]

#     plt.figure()
#     if show_hits:
#         # hits_z は (x_index, y_index) で入ってることが多いので transpose 推奨
#         plt.imshow(hits_z.T, origin="lower", extent=extent, aspect="auto")
#         plt.title(f"hits_z + foot_xy (idx={idx})")
#     elif show_trav:
#         plt.imshow(trav.T, origin="lower", extent=extent, aspect="auto")
#         plt.title(f"trav + foot_xy (idx={idx})")
#     else:
#         plt.xlim(extent[0], extent[1]); plt.ylim(extent[2], extent[3])
#         plt.title(f"foot_xy only (idx={idx})")

#     # 4脚をマーカーだけ変えて重ねる（色はデフォルトでOK）
#     markers = ["o", "s", "^", "x"]  # FL, FR, RL, RR のつもり
#     labels  = ["FL", "FR", "RL", "RR"]
#     for i in range(4):
#         plt.scatter(foot_xy[i,0], foot_xy[i,1], marker=markers[i])
#         plt.text(foot_xy[i,0], foot_xy[i,1], labels[i])

#     plt.xlabel("x (base yaw frame) [m]")
#     plt.ylabel("y (base yaw frame) [m]")
#     plt.show()

# # 例
# show_one("teacher_static.h5", idx=0, show_hits=False, show_trav=True)




import h5py
import numpy as np
import matplotlib.pyplot as plt





# PATH   = "/home/digital/isaac_ws/unitree_rl_lab/teacher_static.h5"          # ★



PATH = "/home/digital/isaac_ws/unitree_rl_lab/teacher_walk2.h5"





def xy_to_ij(xy, L, W, res, Hb, Wb):
    # xy: (...,2) in [-L/2,L/2],[-W/2,W/2]
    x = xy[...,0]; y = xy[...,1]
    i = np.rint((x + L/2) / res).astype(np.int64)  # 0..Hb-1
    j = np.rint((y + W/2) / res).astype(np.int64)  # 0..Wb-1
    i = np.clip(i, 0, Hb-1)
    j = np.clip(j, 0, Wb-1)
    return i, j

def show_distribution(h5_path, max_n=None):
    with h5py.File(h5_path, "r") as f:
        L, W = f.attrs["ray_size_xy"]
        res = float(f.attrs["ray_resolution"])
        foot_xy = f["cmd_xy"][:]  # (N,4,2)
        Hb, Wb = f["hits_z"].shape[1:3]
        N = foot_xy.shape[0]
        if max_n is not None:
            N = min(N, max_n)
            foot_xy = foot_xy[:N]

    extent = [-L/2, L/2, -W/2, W/2]
    H = [np.zeros((Hb, Wb), dtype=np.int32) for _ in range(4)]

    for leg in range(4):
        i, j = xy_to_ij(foot_xy[:,leg,:], L, W, res, Hb, Wb)
        np.add.at(H[leg], (i, j), 1)

    titles = ["FL", "FR", "RL", "RR"]
    for leg in range(4):
        plt.figure()
        plt.imshow(H[leg].T, origin="lower", extent=extent, aspect="auto")
        plt.title(f"foot placement density: {titles[leg]} (N={N})")
        plt.xlabel("x [m]"); plt.ylabel("y [m]")
        plt.show()

# 例








import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------
# 1) ここだけあなたの実装に合わせて import を直してください
# -------------------------
# 例: student_train.py に Model と Dataset がある前提
from student_train import TeacherH5Dataset, StudentNetInBEV  # ★要調整: 実ファイル/クラス名に合わせる


# -------------------------
# 2) チェックポイント読み込み（よくある形式に対応）
# -------------------------
def load_model(ckpt_path: str, device: str = "cuda", **model_kwargs):
    model = StudentNetInBEV(**model_kwargs).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)

    # 形式の違いを吸収
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and any(k.startswith("backbone.") or k.startswith("head.") for k in ckpt.keys()):
        sd = ckpt
    else:
        sd = ckpt

    # DDPなどで "module." が付いている場合に除去
    sd2 = {}
    for k, v in sd.items():
        if k.startswith("module."):
            sd2[k[len("module."):]] = v
        else:
            sd2[k] = v

    missing, unexpected = model.load_state_dict(sd2, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    model.eval()
    return model


# -------------------------
# 3) logits -> pred_xy 変換
# -------------------------
@torch.no_grad()
def logits_to_pred_xy(foot_logits: torch.Tensor, L: float, W: float):
    """
    foot_logits: (B,4,Hb,Wb) or (B,4,Hb*Wb)
    return pred_xy: (B,4,2), prob_maps: (B,4,Hb,Wb)
    """
    if foot_logits.ndim == 4:
        B, F, Hb, Wb = foot_logits.shape
        flat = foot_logits.reshape(B, F, Hb * Wb)
        prob = torch.softmax(flat, dim=-1).reshape(B, F, Hb, Wb)
        idx = flat.argmax(dim=-1)  # (B,4)
    elif foot_logits.ndim == 3:
        B, F, K = foot_logits.shape
        # K=Hb*Wb を推定できないので、後で呼ぶ側で reshape してから渡すのが安全
        raise ValueError("foot_logits is (B,4,Hb*Wb). reshape to (B,4,Hb,Wb) before calling.")
    else:
        raise ValueError(f"Unexpected foot_logits shape: {foot_logits.shape}")

    # idx -> (i,j)
    i = (idx // Wb).long()
    j = (idx %  Wb).long()

    x_coords = torch.linspace(-L/2, L/2, Hb, device=foot_logits.device)
    y_coords = torch.linspace(-W/2, W/2, Wb, device=foot_logits.device)

    x = x_coords[i]  # (B,4)
    y = y_coords[j]  # (B,4)
    pred_xy = torch.stack([x, y], dim=-1)  # (B,4,2)
    return pred_xy, prob


# -------------------------
# 4) show_one: 背景 + GT + Pred を重ね描き
# -------------------------
def show_one(h5_path: str, idx: int, pred_xy=None, bg="trav"):
    with h5py.File(h5_path, "r") as f:
        L, W = f.attrs["ray_size_xy"]
        hits_z = f["hits_z"][idx]          # (Hb,Wb)
        trav  = f["trav"][idx]             # (Hb,Wb)
        gt_xy = f["cmd_xy"][idx]          # (4,2)

    extent = [-L/2, L/2, -W/2, W/2]

    plt.figure(figsize=(6, 4))
    if bg == "hits":
        plt.imshow(hits_z.T, origin="lower", extent=extent, aspect="auto")
        plt.title(f"hits_z + GT/PRED (idx={idx})")
    else:
        plt.imshow(trav.T, origin="lower", extent=extent, aspect="auto")
        plt.title(f"trav + GT/PRED (idx={idx})")

    plt.xlabel("x [m] (yaw-base)")
    plt.ylabel("y [m] (yaw-base)")

    markers = ["o", "s", "^", "x"]
    names = ["FL", "FR", "RL", "RR"]

    # GT
    for k in range(4):
        plt.scatter(gt_xy[k, 0], gt_xy[k, 1], marker=markers[k])
        plt.text(gt_xy[k, 0], gt_xy[k, 1], f"GT {names[k]}", fontsize=8)

    # Pred
    if pred_xy is not None:
        if torch.is_tensor(pred_xy):
            pred_xy = pred_xy.detach().cpu().numpy()
        for k in range(4):
            plt.scatter(pred_xy[k, 0], pred_xy[k, 1], marker=markers[k])
            plt.text(pred_xy[k, 0], pred_xy[k, 1], f"P {names[k]}", fontsize=8)

    plt.tight_layout()
    plt.show()


# -------------------------
# 5) 実行: モデル推論して show_one に重ねる
# -------------------------
@torch.no_grad()
def visualize_prediction(h5_path: str, ckpt_path: str = None, sample_idx: int = 0, device: str = "cuda"):
    # Dataset は学習時と同じものを使う（前処理の食い違い防止）
    ds = TeacherH5Dataset(h5_path)  # ★学習時と同じ引数にする（use_trav/no_proprio などがあるなら揃える）

    # attrs
    with h5py.File(h5_path, "r") as f:
        L, W = f.attrs["ray_size_xy"]
        Hb, Wb = f["hits_z"].shape[1:3]

    if ckpt_path is not None :
        # model
        model = load_model(
            ckpt_path,
            device=device,
            # ★必要ならここにモデル構築の引数（use_trav / proprio_dim 等）を入れる
        )

    batch = ds[sample_idx]
    rgb = batch["rgb"]                      # 期待: (3,H,W) float (0..1) など
    proprio = batch.get("proprio", None)    # 期待: (D,) or None

    # tensor化
    rgb_t = rgb.unsqueeze(0).to(device)  # (1,3,H,W)
    if proprio is not None:
        proprio_t = proprio.unsqueeze(0).to(device)
    else:
        proprio_t = None


    if ckpt_path is not None :
        # forward（あなたのモデルの返り値に合わせて取り出し）
        out = model(rgb_t, proprio_t) if proprio_t is not None else model(rgb_t)

        # out の形式に応じて foot_logits を取り出す
        if isinstance(out, dict):
            foot_logits = out["foot_logits"]
        elif isinstance(out, (tuple, list)):
            foot_logits = out[0]
        else:
            foot_logits = out

        # (1,4,Hb,Wb) になってなければ reshape（実装によっては (1,4,Hb*Wb) のことがある）
        if foot_logits.ndim == 3 and foot_logits.shape[-1] == Hb * Wb:
            foot_logits = foot_logits.view(1, 4, Hb, Wb)

        pred_xy, prob = logits_to_pred_xy(foot_logits, float(L), float(W))  # (1,4,2)
        pred_xy0 = pred_xy[0]  # (4,2)

        # 可視化
        show_one(h5_path, sample_idx, pred_xy=pred_xy0, bg="trav")
    
    else :
        show_one(h5_path, sample_idx, pred_xy=None, bg="trav")

    
    show_distribution(h5_path, max_n=1000)




if __name__ == "__main__":
    ckpt_path = "student_ckpt.pt"            # ★学習後の重み
    # visualize_prediction(h5_path, ckpt_path, sample_idx=0, device="cuda")
    visualize_prediction(PATH, sample_idx=0, device="cuda")

