"""Trích đặc trưng (notebook 03: truyền thống; notebook 04: deep, phần "Deep" cuối file).

# Traditional
Đặc trưng: hog, sift_bovw, color_hist, color_moments, lbp và đặc trưng ghép hog_color.
Đầu vào là kết quả của 02_data_prep: data/processed/meta.csv (đường dẫn, nhãn, cột split = train/val/test)
và data/processed/config.json (seed, img_size).
Mọi đặc trưng được lưu thành artifacts/features/<tên>.npy (float32), dòng i ứng với dòng i của meta.csv
(cột idx = i), kèm <tên>.json ghi cấu hình prep + trích đặc trưng đã dùng.
Ảnh được nạp bằng data_prep.load_rgb (RGB, nền trong suốt -> trắng) rồi đưa về img_size x img_size theo [RESIZE].

CÁCH ĐỔI PHƯƠNG ÁN (để so sánh độ chính xác giữa các lần chạy)
---------------------------------------------------------------
Mỗi điểm lựa chọn là một bộ khối có tiêu đề
    # ===== [TÊN] CÁCH n: mô tả (ĐANG DÙNG | đang tắt) =====
Chỉ MỘT khối mỗi bộ ở dạng code, các khối còn lại nằm trong comment (mỗi dòng bắt đầu bằng "# ").
Muốn đổi: comment khối đang dùng, bỏ comment khối muốn dùng, sửa chữ ĐANG DÙNG/đang tắt ở tiêu đề,
rồi chạy lại notebook 03 (chỉ đặc trưng bị ảnh hưởng mới cần sinh lại).

    [RESIZE]      _resize            : đưa ảnh về img_size (1: như eval_tf của 02 - cạnh ngắn 256 rồi cắt giữa 224
                                       | 2: cạnh ngắn 224 rồi cắt giữa, giữ nhiều ảnh hơn)
    [HOG]         _hog_vector        : cỡ ảnh/cỡ ô (1: 128px ô 8 | 2: 128px ô 16 | 3: 224px ô 16)
    [SIFT_POINTS] _sift_descriptors  : điểm lấy descriptor (1: dense lưới 8px | 2: keypoint SIFT)
    [BOVW_FIT]    extract_sift_bovw  : fit từ điển trên (1: tập train | 2: train + val)
    [COLOR_HIST]  _color_hist_vector : histogram HSV (1: 3D 8x4x4 = 128 chiều | 2: từng kênh 16+8+8 = 32)
    [COLOR_REGION] _region           : vùng tính màu (1: cả ảnh | 2: 60% ở giữa, bớt ảnh hưởng của nền)
    [LBP]         _lbp_vector        : (1: P8 R1 lưới 4x4 = 160 chiều | 2: toàn ảnh P8 R1 = 10 | 3: toàn ảnh P16 R2 = 18)
    [COMBINE]     combine_features   : ghép đặc trưng (1: L2-chuẩn hóa từng phần rồi nối | 2: nối thẳng)

# Deep
Đặc trưng: resnet50 (2048), efficientnet_b0 (1280), vit_b16 (768), lấy từ mạng pretrain ImageNet của torchvision,
đóng băng hoàn toàn (eval + inference_mode, không augmentation), bỏ head phân loại, giữ vector lớp áp chót.
Ảnh nạp qua cùng load_image như trên rồi chuẩn hóa mean/std ImageNet, tức đúng eval_tf của 02.
Đổi phương án theo cùng cách như trên, rồi chạy lại notebook 04.

    [RESNET_WEIGHTS] _deep_weights   : trọng số ResNet-50 (1: IMAGENET1K_V1 | 2: IMAGENET1K_V2)
    [DEEP_SIZE]      _deep_size      : cỡ ảnh vào mạng (1: 224 | 2: 128, ViT bị bỏ qua)
    [DEEP_NORM]      _deep_norm      : hậu xử lý (1: lưu thô | 2: L2-chuẩn hóa từng dòng)

Đừng đặt ghi chú tự do bên trong các khối: công cụ kiểm thử bật/tắt khối bằng cách thêm/bỏ tiền tố "# ".
"""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from modules import data_prep as prep

# Không để OpenCV tạo thread pool ở tiến trình cha: nếu cha đã dùng OpenCV nhiều luồng rồi mới fork
# tiến trình con thì các con bị treo (deadlock). Song song hóa được làm bằng nhiều tiến trình, không phải nhiều luồng.
cv2.setNumThreads(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # thư mục gốc repo
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")              # đầu ra của 02_data_prep

# Lựa chọn đang có hiệu lực (các khối bên dưới tự điền khi chạy; xem fill_active)
ACTIVE = {}

_CFG = None        # cấu hình, đặt trong từng tiến trình con qua _init_worker
_CENTERS = None    # từ điển BoVW (k, 128) cho bước mã hóa
_SIFT = None
_BLAS_LIMIT = []   # giữ tham chiếu để giới hạn luồng BLAS còn hiệu lực trong suốt đời tiến trình con


# --------------------------------------------------------------------------- #
# Nạp cấu hình và kết quả của 02_data_prep
# --------------------------------------------------------------------------- #
def load_config():
    """config.yaml (đường dẫn, tham số BoVW) + data/processed/config.json của 02 (seed, img_size)."""
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(PROCESSED_DIR, "config.json"), encoding="utf-8") as f:
        cfg["prep"] = json.load(f)
    cfg["seed"], cfg["img_size"] = cfg["prep"]["seed"], cfg["prep"]["img_size"]
    for k in ["features_dir", "results_dir", "figures_dir"]:
        cfg[k] = os.path.join(ROOT, cfg[k])
    return cfg


def resolve_path(p):
    """Đường dẫn trong meta.csv được ghi tương đối so với thư mục notebooks/ (vd ../data/raw/...)."""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(ROOT, "notebooks", p))


def load_meta():
    """meta.csv của 02 + cột idx (= số dòng) và dict splits {"train"|"val"|"test": list idx}."""
    meta = pd.read_csv(os.path.join(PROCESSED_DIR, "meta.csv"))
    meta.insert(0, "idx", np.arange(len(meta)))
    splits = {s: meta.loc[meta["split"] == s, "idx"].tolist() for s in ["train", "val", "test"]}
    return meta, splits


def _resize(img, size):
    """Đưa ảnh PIL RGB về size x size, không kéo giãn (theo EDA, tỷ lệ/kích thước ảnh gắn với lớp)."""
    # ===== [RESIZE] CÁCH 1: giống eval_tf của 02 - cạnh ngắn về size*256/224 rồi cắt giữa size x size (ĐANG DÙNG) =====
    ACTIVE["resize"] = "short256_center_crop"
    short = int(size * 256 / 224)
    # ===== [RESIZE] CÁCH 2: cạnh ngắn về size rồi cắt giữa (giữ được nhiều phần ảnh hơn) (đang tắt) =====
    # ACTIVE["resize"] = "short224_center_crop"
    # short = size
    # ===== hết [RESIZE] =====
    # Làm tròn đúng như torchvision T.Resize + T.CenterCrop để ảnh trùng từng pixel với eval_tf của 02
    w, h = img.size
    if w <= h:
        new_w, new_h = short, int(short * h / w)
    else:
        new_w, new_h = int(short * w / h), short
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left, top = int(round((new_w - size) / 2.0)), int(round((new_h - size) / 2.0))
    return img.crop((left, top, left + size, top + size))


def load_image(path, size):
    """Đọc 1 ảnh theo đường dẫn trong meta.csv -> PIL RGB size x size."""
    return _resize(prep.load_rgb(resolve_path(path)), size)


# --------------------------------------------------------------------------- #
# Hạ tầng: chạy song song, lưu/nạp, kiểm tra
# --------------------------------------------------------------------------- #
def _init_worker(cfg, centers):
    global _CFG, _CENTERS
    _CFG, _CENTERS = cfg, centers
    cv2.setNumThreads(1)
    # Mỗi tiến trình con chỉ 1 luồng BLAS: 20 tiến trình x nhiều luồng sẽ tranh CPU và chậm đi rất nhiều
    _BLAS_LIMIT.append(threadpool_limits(1))


def _load(rel):
    return load_image(rel, _CFG["img_size"])


def _run(func, items, cfg, centers=None, n_jobs=None):
    """Chạy func trên từng phần tử của items song song, giữ nguyên thứ tự."""
    with ProcessPoolExecutor(n_jobs or os.cpu_count(), initializer=_init_worker,
                             initargs=(cfg, centers)) as ex:
        return list(ex.map(func, items, chunksize=32))


def save_features(name, X, cfg, extra=None):
    """Lưu <features_dir>/<name>.npy (float32) và <name>.json (cấu hình đã dùng)."""
    fill_active()
    os.makedirs(cfg["features_dir"], exist_ok=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    np.save(os.path.join(cfg["features_dir"], f"{name}.npy"), X)
    meta = {"name": name, "shape": list(X.shape), "prep": cfg["prep"], "extract": dict(ACTIVE),
            "img_size": cfg["img_size"], "seed": cfg["seed"], **(extra or {})}
    with open(os.path.join(cfg["features_dir"], f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False, default=float)


def load_features(name, cfg):
    return np.load(os.path.join(cfg["features_dir"], f"{name}.npy"))


def check_features(X, meta):
    """Kiểm tra cơ bản; ném AssertionError nếu sai. Trả về dict thống kê."""
    assert X.ndim == 2 and len(X) == len(meta), f"số dòng {len(X)} != meta {len(meta)}"
    assert np.isfinite(X).all(), "có NaN/inf"
    return {"dims": X.shape[1], "zero_rows": int((~X.any(axis=1)).sum()),
            "min": float(X.min()), "max": float(X.max()), "MB": round(X.nbytes / 1e6, 1)}


# --------------------------------------------------------------------------- #
# Traditional: HOG
# --------------------------------------------------------------------------- #
def _hog_vector(img):
    """HOG trên ảnh xám của `img` (PIL RGB)."""
    # ===== [HOG] CÁCH 1: ảnh 128x128, ô 8x8 -> 8100 chiều (ĐANG DÙNG) =====
    ACTIVE["hog"] = "128px_cell8"
    size, cell = 128, 8
    # ===== [HOG] CÁCH 2: ảnh 128x128, ô 16x16 -> 1764 chiều (đang tắt) =====
    # ACTIVE["hog"] = "128px_cell16"
    # size, cell = 128, 16
    # ===== [HOG] CÁCH 3: ảnh 224x224, ô 16x16 -> 6084 chiều (đang tắt) =====
    # ACTIVE["hog"] = "224px_cell16"
    # size, cell = 224, 16
    # ===== hết [HOG] =====
    g = np.asarray(img.convert("L").resize((size, size), Image.BICUBIC))
    return hog(g, orientations=9, pixels_per_cell=(cell, cell), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


def _hog_one(rel):
    return _hog_vector(_load(rel))


# --------------------------------------------------------------------------- #
# Traditional: SIFT + Bag of Visual Words
# --------------------------------------------------------------------------- #
def _get_sift():
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    return _SIFT


def _sift_descriptors(gray):
    """Descriptor RootSIFT (n, 128) của ảnh xám uint8; n có thể bằng 0."""
    sift = _get_sift()
    # ===== [SIFT_POINTS] CÁCH 1: dense - lưới đều bước 8px, cỡ 16px (ĐANG DÙNG) =====
    ACTIVE["sift_points"] = "dense_step8"
    step = 8
    kps = [cv2.KeyPoint(float(x), float(y), 16.0)
           for y in range(step, gray.shape[0], step) for x in range(step, gray.shape[1], step)]
    _, desc = sift.compute(gray, kps)
    # ===== [SIFT_POINTS] CÁCH 2: keypoint do bộ phát hiện SIFT chọn (đang tắt) =====
    # ACTIVE["sift_points"] = "keypoint"
    # _, desc = sift.detectAndCompute(gray, None)
    # ===== hết [SIFT_POINTS] =====
    if desc is None or len(desc) == 0:
        return np.zeros((0, 128), dtype=np.float32)
    desc = desc / (desc.sum(axis=1, keepdims=True) + 1e-7)  # RootSIFT: L1 rồi căn bậc hai
    return np.sqrt(desc).astype(np.float32)


def _gray(img):
    return np.asarray(img.convert("L"))


def _sift_sample(item):
    """Lấy mẫu ngẫu nhiên tối đa bovw_desc_per_image descriptor của một ảnh (để fit từ điển)."""
    idx, rel = item
    desc = _sift_descriptors(_gray(_load(rel)))
    n = _CFG["bovw_desc_per_image"]
    if len(desc) > n:
        desc = desc[np.random.RandomState(_CFG["seed"] * 100003 + idx).choice(len(desc), n, replace=False)]
    return desc


def _sift_encode(rel):
    """Histogram BoVW (k chiều, tổng = 1) của một ảnh; ảnh không có descriptor -> histogram toàn 0."""
    desc = _sift_descriptors(_gray(_load(rel)))
    k = len(_CENTERS)
    if len(desc) == 0:
        return np.zeros(k, dtype=np.float32), 0
    c2 = (_CENTERS ** 2).sum(axis=1)
    words = np.argmin(c2[None, :] - 2.0 * desc @ _CENTERS.T, axis=1)
    hist = np.bincount(words, minlength=k).astype(np.float32)
    return hist / hist.sum(), len(desc)


def _fit_vocab(meta, idx_list, cfg):
    """KMeans trên descriptor lấy mẫu từ các ảnh có idx trong idx_list; trả về tâm cụm (k, 128)."""
    rows = meta.set_index("idx").loc[idx_list]
    items = list(zip(rows.index.tolist(), rows["path"].tolist()))
    desc = np.vstack([d for d in _run(_sift_sample, items, cfg) if len(d)])
    km = MiniBatchKMeans(cfg["bovw_k"], random_state=cfg["seed"], batch_size=4096, n_init=3, max_iter=100)
    km.fit(desc)
    return km.cluster_centers_.astype(np.float32)


def _encode_all(meta, centers, cfg):
    out = _run(_sift_encode, meta["path"].tolist(), cfg, centers=centers)
    X = np.stack([h for h, _ in out])
    n = np.array([c for _, c in out])
    return X, {"descriptors_mean": float(n.mean()), "images_without_descriptors": int((n == 0).sum()),
               "vocab_size": int(len(centers))}


def extract_sift_bovw(meta, splits, cfg):
    """SIFT + BoVW. Trả về dict {tên đặc trưng: (X, thống kê)}.

    Final test không bao giờ được dùng để fit từ điển.
    """
    # ===== [BOVW_FIT] CÁCH 1: fit từ điển chỉ trên tập train (ĐANG DÙNG) =====
    ACTIVE["bovw_fit"] = "train"
    centers = _fit_vocab(meta, splits["train"], cfg)
    # ===== [BOVW_FIT] CÁCH 2: fit từ điển trên train + val (đang tắt) =====
    # ACTIVE["bovw_fit"] = "train_val"
    # centers = _fit_vocab(meta, splits["train"] + splits["val"], cfg)
    # ===== hết [BOVW_FIT] =====
    return {"sift_bovw": _encode_all(meta, centers, cfg)}


# --------------------------------------------------------------------------- #
# Traditional: màu và kết cấu
# --------------------------------------------------------------------------- #
def _region(arr):
    """Vùng ảnh dùng để tính đặc trưng màu."""
    # ===== [COLOR_REGION] CÁCH 1: cả ảnh (ĐANG DÙNG) =====
    ACTIVE["color_region"] = "whole"
    return arr
    # ===== [COLOR_REGION] CÁCH 2: 60% ở giữa, bỏ 20% mỗi cạnh (đang tắt) =====
    # ACTIVE["color_region"] = "center60"
    # h, w = arr.shape[:2]
    # return arr[h // 5: h - h // 5, w // 5: w - w // 5]
    # ===== hết [COLOR_REGION] =====


def _color_hist_vector(img):
    """Histogram màu HSV (OpenCV: H 0-179, S và V 0-255), chuẩn hóa tổng = 1."""
    hsv = np.ascontiguousarray(_region(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2HSV)))
    # ===== [COLOR_HIST] CÁCH 1: histogram 3D chung 8x4x4 -> 128 chiều (ĐANG DÙNG) =====
    ACTIVE["color_hist"] = "hsv3d_8x4x4"
    h = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel()
    h = h / h.sum()
    # ===== [COLOR_HIST] CÁCH 2: histogram riêng từng kênh H16 + S8 + V8 -> 32 chiều (đang tắt) =====
    # ACTIVE["color_hist"] = "hsv_per_channel_16_8_8"
    # parts = [cv2.calcHist([hsv], [c], None, [b], [0, r]).ravel() for c, b, r in [(0, 16, 180), (1, 8, 256), (2, 8, 256)]]
    # h = np.concatenate([p / p.sum() for p in parts])
    # ===== hết [COLOR_HIST] =====
    return h.astype(np.float32)


def _color_hist_one(rel):
    return _color_hist_vector(_load(rel))


def _color_moments_vector(img):
    """Mean, std, skewness của 3 kênh Lab (9 chiều). Lab tránh được vấn đề Hue quay vòng."""
    lab = _region(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)).reshape(-1, 3).astype(np.float64)
    mean = lab.mean(axis=0)
    std = lab.std(axis=0)
    skew = np.cbrt(((lab - mean) ** 3).mean(axis=0))
    return np.concatenate([mean, std, skew]).astype(np.float32)


def _color_moments_one(rel):
    return _color_moments_vector(_load(rel))


def _lbp_vector(gray):
    """Histogram LBP "uniform" của ảnh xám uint8, mỗi ô một histogram chuẩn hóa tổng = 1."""
    # ===== [LBP] CÁCH 1: P=8, R=1, lưới 4x4 ô -> 160 chiều (ĐANG DÙNG) =====
    ACTIVE["lbp"] = "p8r1_grid4x4"
    P, R, grid = 8, 1, 4
    # ===== [LBP] CÁCH 2: P=8, R=1, toàn ảnh -> 10 chiều (đang tắt) =====
    # ACTIVE["lbp"] = "p8r1_global"
    # P, R, grid = 8, 1, 1
    # ===== [LBP] CÁCH 3: P=16, R=2, toàn ảnh -> 18 chiều (đang tắt) =====
    # ACTIVE["lbp"] = "p16r2_global"
    # P, R, grid = 16, 2, 1
    # ===== hết [LBP] =====
    lbp = local_binary_pattern(gray, P, R, method="uniform").astype(np.int64)
    nb = P + 2
    hs = [np.bincount(cell.ravel(), minlength=nb) / cell.size
          for row in np.array_split(lbp, grid, axis=0) for cell in np.array_split(row, grid, axis=1)]
    return np.concatenate(hs).astype(np.float32)


def _lbp_one(rel):
    return _lbp_vector(_gray(_load(rel)))


def combine_features(parts):
    """Ghép các ma trận đặc trưng (cùng số dòng) thành một."""
    # ===== [COMBINE] CÁCH 1: L2-chuẩn hóa từng dòng của mỗi phần rồi nối (ĐANG DÙNG) =====
    ACTIVE["combine"] = "l2_each_then_concat"
    parts = [p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-12) for p in parts]
    # ===== [COMBINE] CÁCH 2: nối thẳng, giữ nguyên thang đo từng phần (đang tắt) =====
    # ACTIVE["combine"] = "plain_concat"
    # ===== hết [COMBINE] =====
    return np.hstack(parts)


# --------------------------------------------------------------------------- #
# Điều phối
# --------------------------------------------------------------------------- #
_SIMPLE = {"hog": _hog_one, "color_hist": _color_hist_one, "color_moments": _color_moments_one, "lbp": _lbp_one}


def extract_simple(name, meta, cfg, n_jobs=None):
    """Trích một trong các đặc trưng hog / color_hist / color_moments / lbp cho mọi ảnh trong meta.

    Trả về (X, thống kê). Dòng i của X ứng với dòng i của meta.
    """
    fill_active()
    t = time.time()
    X = np.stack(_run(_SIMPLE[name], meta["path"].tolist(), cfg, n_jobs=n_jobs))
    return X, {"seconds": round(time.time() - t, 1)}


def feature_summary(cfg, names=None):
    """Bảng tóm tắt các đặc trưng đã lưu trong features_dir: số chiều, dung lượng, dòng toàn 0, cấu hình đã dùng.

    names: chỉ lấy các đặc trưng này (vd list(DEEP_MODELS)); bỏ trống = mọi đặc trưng.
    """
    rows = []
    for fn in sorted(os.listdir(cfg["features_dir"])):
        if not fn.endswith(".json") or (names is not None and fn[:-5] not in names):
            continue
        with open(os.path.join(cfg["features_dir"], fn), encoding="utf-8") as f:
            meta = json.load(f)
        X = load_features(meta["name"], cfg)
        rows.append({"đặc trưng": meta["name"], "số dòng": X.shape[0], "số chiều": X.shape[1],
                     "MB": round(X.nbytes / 1e6, 1), "dòng toàn 0": int((~X.any(axis=1)).sum()),
                     "giây": meta.get("seconds"), "ảnh không có descriptor": meta.get("images_without_descriptors"),
                     "resize": meta["extract"].get("resize")})
    df = pd.DataFrame(rows).set_index("đặc trưng")
    return df if names is None else df.dropna(axis=1, how="all")


def fill_active():
    """Chạy thử từng hàm vector trên ảnh trống để ACTIVE được điền (tiến trình con không ghi ngược về cha)."""
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    _hog_vector(img)
    _color_hist_vector(img)
    _color_moments_vector(img)
    _lbp_vector(_gray(img))
    _sift_descriptors(_gray(img))
    _resize(img, 8)
    _fill_deep_active()


# --------------------------------------------------------------------------- #
# Minh họa (chỉ để giải thích trong notebook 03, không ảnh hưởng đặc trưng đã lưu)
# --------------------------------------------------------------------------- #
def pick_examples(meta, labels, seed=42):
    """Mỗi lớp trong `labels` lấy ngẫu nhiên 1 ảnh; trả về list (idx, path, label)."""
    out = []
    for lab in labels:
        row = meta[meta["label"] == lab].sample(1, random_state=seed).iloc[0]
        out.append((int(row["idx"]), row["path"], lab))
    return out


def _load_cfg_image(cfg, rel):
    return load_image(rel, cfg["img_size"])


def show_hog_demo(cfg, rel):
    """Ảnh gốc -> ảnh xám -> độ lớn gradient -> HOG (mũi tên hướng trội mỗi ô)."""
    import matplotlib.pyplot as plt
    fill_active()
    tag = ACTIVE["hog"]                              # vd "128px_cell8": đọc lại đúng cấu hình [HOG] đang dùng
    size, cell = int(tag.split("px")[0]), int(tag.split("cell")[1])
    img = _load_cfg_image(cfg, rel)
    g = np.asarray(img.convert("L").resize((size, size), Image.BICUBIC)).astype(np.float32)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    vec, vis = hog(g, orientations=9, pixels_per_cell=(cell, cell), cells_per_block=(2, 2),
                   block_norm="L2-Hys", visualize=True, feature_vector=True)
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
    ax[0].imshow(img); ax[0].set_title("Ảnh RGB")
    ax[1].imshow(g, cmap="gray"); ax[1].set_title(f"Ảnh xám {size}x{size}")
    for k in range(0, size + 1, cell):
        ax[1].axhline(k - 0.5, color="yellow", lw=0.3); ax[1].axvline(k - 0.5, color="yellow", lw=0.3)
    ax[2].imshow(np.hypot(gx, gy), cmap="magma"); ax[2].set_title("Độ lớn gradient")
    ax[3].imshow(vis, cmap="gray"); ax[3].set_title("HOG: hướng cạnh trội mỗi ô")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"HOG ({tag}) -> vector {len(vec)} chiều", fontsize=10)
    fig.tight_layout()
    return fig


def show_sift_demo(cfg, rel, idx, X_bovw):
    """Lưới điểm dense trên ảnh xám, descriptor 128 chiều của 1 điểm, histogram BoVW của ảnh."""
    import matplotlib.pyplot as plt
    gray = _gray(_load_cfg_image(cfg, rel))
    step = 8
    pts = [(x, y) for y in range(step, gray.shape[0], step) for x in range(step, gray.shape[1], step)]
    desc = _sift_descriptors(gray)
    ncols = len(range(step, gray.shape[1], step))
    j = (len(pts) // ncols // 2) * ncols + ncols // 2          # điểm ở giữa ảnh
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1.5, 2]})
    ax[0].imshow(gray, cmap="gray")
    ax[0].scatter(*zip(*pts), s=1.5, c="cyan")
    x, y = pts[j]
    sub = 3 * 16 / 2                                   # OpenCV: cạnh ô con = 3 * (cỡ keypoint / 2) = 24px
    for k in range(5):                                 # vùng descriptor 4x4 ô con quanh điểm giữa
        ax[0].plot([x - 2 * sub, x + 2 * sub], [y - 2 * sub + k * sub] * 2, c="red", lw=0.8)
        ax[0].plot([x - 2 * sub + k * sub] * 2, [y - 2 * sub, y + 2 * sub], c="red", lw=0.8)
    ax[0].plot(x, y, "o", c="red", ms=4)
    ax[0].set_title(f"{len(pts)} điểm dense (bước {step}px)"); ax[0].axis("off")
    ax[1].bar(range(128), desc[j], width=1.0, color=[plt.cm.tab20(i // 8 % 20) for i in range(128)])
    ax[1].set_title("Descriptor RootSIFT của điểm đỏ\n(4x4 ô con, mỗi ô 8 hướng = 128)")
    ax[1].set_xlim(-1, 128); ax[1].set_xticks(range(0, 129, 32))
    h = X_bovw[idx]
    ax[2].bar(range(len(h)), h, width=1.0, color="tab:blue")
    ax[2].set_title(f"Histogram BoVW của ảnh ({len(h)} từ thị giác, tổng = 1)")
    ax[2].set_xlabel("chỉ số từ thị giác (cụm KMeans)")
    fig.tight_layout()
    return fig


def show_color_hist_demo(cfg, examples):
    """Mỗi ví dụ một hàng: ảnh + histogram HSV, mỗi cột tô bằng màu tâm của bin."""
    import matplotlib.pyplot as plt
    fill_active()
    bins = (8, 4, 4)                                   # khớp [COLOR_HIST] cách 1
    centers = np.stack(np.meshgrid(*[(np.arange(b) + 0.5) / b for b in bins], indexing="ij"), -1).reshape(-1, 3)
    hsv = np.stack([centers[:, 0] * 180, centers[:, 1] * 255, np.maximum(centers[:, 2], 0.25) * 255], -1)
    rgb = cv2.cvtColor(hsv.astype(np.uint8)[None], cv2.COLOR_HSV2RGB)[0] / 255.0
    fig, ax = plt.subplots(len(examples), 2, figsize=(13, 2.3 * len(examples)),
                           gridspec_kw={"width_ratios": [1, 6]}, squeeze=False)
    for r, (_, rel, lab) in enumerate(examples):
        img = _load_cfg_image(cfg, rel)
        h = _color_hist_vector(img)
        ax[r][0].imshow(img); ax[r][0].set_title(lab, fontsize=9); ax[r][0].axis("off")
        if ACTIVE["color_hist"] == "hsv3d_8x4x4":
            ax[r][1].bar(range(len(h)), h, width=1.0, color=rgb, edgecolor="0.4", linewidth=0.3)
        else:
            ax[r][1].bar(range(len(h)), h, width=1.0)
        ax[r][1].set_xlim(-1, len(h)); ax[r][1].set_ylabel("tỷ lệ pixel")
    ax[-1][1].set_xlabel("bin (H chạy chậm nhất: 8 khoảng màu; trong mỗi khoảng là 4 S x 4 V)")
    fig.suptitle(f"Histogram màu HSV ({ACTIVE['color_hist']}, vùng: {ACTIVE['color_region']})", fontsize=10)
    fig.tight_layout()
    return fig


def color_moments_by_class(X_cm, meta):
    """Trung bình 9 color moments theo lớp (để thấy màu phân biệt lớp nào)."""
    cols = [f"{s}_{c}" for s in ["mean", "std", "skew"] for c in ["L", "a", "b"]]
    df = pd.DataFrame(X_cm.astype(np.float64), columns=cols)
    df["lớp"] = meta["label"].values
    return df.groupby("lớp")[cols].mean().round(1)


def show_lbp_demo(cfg, rel):
    """Ảnh xám -> ảnh mã LBP (kèm lưới ô) -> histogram LBP của vài ô."""
    import matplotlib.pyplot as plt
    fill_active()
    P, R = (16, 2) if ACTIVE["lbp"].startswith("p16") else (8, 1)
    grid = 4 if ACTIVE["lbp"].endswith("grid4x4") else 1
    gray = _gray(_load_cfg_image(cfg, rel))
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    vec = _lbp_vector(gray)
    nb = P + 2
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 2]})
    ax[0].imshow(gray, cmap="gray"); ax[0].set_title("Ảnh xám"); ax[0].axis("off")
    ax[1].imshow(lbp, cmap="nipy_spectral", vmin=0, vmax=nb - 1); ax[1].axis("off")
    ax[1].set_title(f"Mã LBP mỗi pixel (0..{nb - 1})")
    for k in range(1, grid):
        ax[1].axhline(k * gray.shape[0] / grid, color="white", lw=1)
        ax[1].axvline(k * gray.shape[1] / grid, color="white", lw=1)
    cells = [0, grid * grid // 2 + grid // 2] if grid > 1 else [0]
    w = 0.8 / len(cells)
    for n, c in enumerate(cells):
        ax[2].bar(np.arange(nb) + n * w, vec[c * nb:(c + 1) * nb], width=w, label=f"ô {c}")
    ax[2].set_xticks(range(nb))
    ax[2].set_xticklabels([str(i) for i in range(P + 1)] + ["không\nuniform"], fontsize=8)
    ax[2].set_xlabel("mã uniform = số bit 1 trong vòng lân cận (0 = điểm sáng, P = điểm tối/phẳng)")
    ax[2].set_ylabel("tỷ lệ pixel"); ax[2].legend()
    ax[2].set_title(f"Histogram LBP từng ô (P={P}, R={R}, lưới {grid}x{grid} -> {len(vec)} chiều)")
    fig.tight_layout()
    return fig


def norm_table(named_parts):
    """Độ dài (chuẩn L2) trung bình mỗi dòng của từng ma trận đặc trưng, trước khi ghép."""
    return pd.DataFrame({name: {"số chiều": X.shape[1], "chuẩn L2 trung bình": float(np.linalg.norm(X, axis=1).mean())}
                         for name, X in named_parts.items()}).T.round(3)


# =========================================================================== #
# Deep (notebook 04): đặc trưng từ mạng pretrain ImageNet, đóng băng (không huấn luyện)
# =========================================================================== #
# Tên đặc trưng -> (kiến trúc torchvision, tên thuộc tính của head phân loại, số chiều đặc trưng)
DEEP_MODELS = {
    "resnet50": ("resnet50", "fc", 2048),               # đầu ra global average pooling
    "efficientnet_b0": ("efficientnet_b0", "classifier", 1280),  # đầu ra global average pooling
    "vit_b16": ("vit_b_16", "heads", 768),              # token [CLS] sau LayerNorm cuối
}


def _deep_weights(name):
    """Tên bộ trọng số torchvision dùng cho từng mô hình."""
    if name == "resnet50":
        # ===== [RESNET_WEIGHTS] CÁCH 1: IMAGENET1K_V1 - công thức train gốc, top-1 ImageNet 76.1% (ĐANG DÙNG) =====
        ACTIVE["resnet_weights"] = "IMAGENET1K_V1"
        # ===== [RESNET_WEIGHTS] CÁCH 2: IMAGENET1K_V2 - công thức mới nhiều augmentation/regularization, top-1 80.9% (đang tắt) =====
        # ACTIVE["resnet_weights"] = "IMAGENET1K_V2"
        # ===== hết [RESNET_WEIGHTS] =====
        return "ResNet50_Weights." + ACTIVE["resnet_weights"]
    return {"efficientnet_b0": "EfficientNet_B0_Weights.IMAGENET1K_V1",
            "vit_b16": "ViT_B_16_Weights.IMAGENET1K_V1"}[name]


def _deep_size():
    """Cỡ ảnh đưa vào mạng."""
    # ===== [DEEP_SIZE] CÁCH 1: 224 - đúng độ phân giải lúc pretrain (ĐANG DÙNG) =====
    ACTIVE["deep_size"] = 224
    # ===== [DEEP_SIZE] CÁCH 2: 128 - ablation độ phân giải thấp; ViT-B/16 không hỗ trợ nên bị bỏ qua (đang tắt) =====
    # ACTIVE["deep_size"] = 128
    # ===== hết [DEEP_SIZE] =====
    return ACTIVE["deep_size"]


def _deep_norm(X):
    """Hậu xử lý ma trận đặc trưng deep trước khi lưu."""
    # ===== [DEEP_NORM] CÁCH 1: lưu vector thô (ĐANG DÙNG) =====
    ACTIVE["deep_norm"] = "raw"
    # ===== [DEEP_NORM] CÁCH 2: L2-chuẩn hóa từng dòng (đang tắt) =====
    # ACTIVE["deep_norm"] = "l2"
    # X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    # ===== hết [DEEP_NORM] =====
    return X


def _fill_deep_active():
    _deep_weights("resnet50")
    _deep_size()
    _deep_norm(np.zeros((1, 1), np.float32))


def get_device():
    """mps (Mac) -> cuda -> cpu."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_tensor(img, cfg):
    """PIL RGB -> tensor (3, H, W) chuẩn hóa mean/std ImageNet, đúng như phần cuối eval_tf của 02."""
    import torch
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    mean = torch.tensor(cfg["prep"]["mean"]).view(3, 1, 1)
    std = torch.tensor(cfg["prep"]["std"]).view(3, 1, 1)
    return (x - mean) / std


def _uint8_one(rel):
    return np.asarray(_load(rel), dtype=np.uint8)


def image_cache(meta, cfg, size, n_jobs=None):
    """Mảng uint8 (N, size, size, 3) của mọi ảnh trong meta sau load_image, lưu ở artifacts/cache/ và mở bằng memmap.

    Giải mã + resize ảnh chỉ làm MỘT lần (song song nhiều tiến trình như notebook 03), sau đó cả ba mạng
    đọc thẳng từng batch từ đĩa. Cách này nhanh hơn DataLoader nhiều worker, vốn làm máy RAM ít bị swap.
    Cache tự sinh lại nếu danh sách ảnh, cỡ ảnh hoặc phương án [RESIZE] thay đổi.
    """
    import hashlib
    from tqdm.auto import tqdm
    _resize(Image.new("RGB", (8, 8)), 8)             # điền ACTIVE["resize"]
    cache_dir = os.path.join(ROOT, "artifacts", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.join(cache_dir, f"images_{size}_{ACTIVE['resize']}")
    key = {"n": len(meta), "size": size, "resize": ACTIVE["resize"],
           "paths_sha1": hashlib.sha1("\n".join(meta["path"]).encode()).hexdigest()}
    if os.path.exists(base + ".json") and os.path.exists(base + ".npy"):
        with open(base + ".json", encoding="utf-8") as f:
            if json.load(f) == key:
                return np.load(base + ".npy", mmap_mode="r")
    arr = np.lib.format.open_memmap(base + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(len(meta), size, size, 3))
    with ProcessPoolExecutor(n_jobs or os.cpu_count(), initializer=_init_worker,
                             initargs=({**cfg, "img_size": size}, None)) as pool:
        for i, a in enumerate(tqdm(pool.map(_uint8_one, meta["path"].tolist(), chunksize=32),
                                   total=len(meta), desc=f"cache ảnh {size}px", unit="ảnh")):
            arr[i] = a
    arr.flush()
    del arr
    os.replace(base + ".tmp.npy", base + ".npy")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(key, f, indent=1)
    return np.load(base + ".npy", mmap_mode="r")


def build_backbone(name, device=None):
    """Nạp mạng pretrain, tách head phân loại. Trả về (backbone, head, weights).

    backbone(x) -> vector đặc trưng; head(vector) -> logits 1000 lớp ImageNet (chỉ dùng để minh họa).
    """
    import torch
    import torchvision
    arch, head_attr, _ = DEEP_MODELS[name]
    weights = torchvision.models.get_weight(_deep_weights(name))
    model = torchvision.models.get_model(arch, weights=weights)
    head = getattr(model, head_attr)
    setattr(model, head_attr, torch.nn.Identity())
    device = device or get_device()
    return model.eval().to(device), head.eval().to(device), weights


def extract_deep(name, meta, cfg, batch_size=64, device=None):
    """Trích đặc trưng deep cho mọi ảnh trong meta (giữ thứ tự dòng), có thanh tiến trình.

    Trả về (X, thống kê); X = None nếu cấu hình hiện tại không hỗ trợ mô hình này (ViT với ảnh khác 224).
    """
    import torch
    from tqdm.auto import tqdm
    _fill_deep_active()
    size = _deep_size()
    if name == "vit_b16" and size != 224:
        return None, {"skipped": f"ViT-B/16 của torchvision chỉ nhận ảnh 224x224, đang đặt [DEEP_SIZE] = {size}"}
    device = device or get_device()
    torch.manual_seed(cfg["seed"])
    imgs = image_cache(meta, cfg, size)
    model, _, weights = build_backbone(name, device)
    mean = torch.tensor(cfg["prep"]["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["prep"]["std"], device=device).view(1, 3, 1, 1)
    t = time.time()
    out = []
    with torch.inference_mode(), tqdm(total=len(imgs), desc=name, unit="ảnh") as bar:
        for s in range(0, len(imgs), batch_size):
            x = torch.from_numpy(np.ascontiguousarray(imgs[s:s + batch_size])).to(device)
            x = (x.permute(0, 3, 1, 2).float() / 255.0 - mean) / std      # = ToTensor + Normalize của eval_tf
            out.append(model(x).float().cpu().numpy())
            bar.update(len(x))
    X = _deep_norm(np.concatenate(out))
    return X, {"seconds": round(time.time() - t, 1), "device": device, "weights": str(weights),
               "input_size": size, "dims": int(X.shape[1]),
               "preprocess": f"load_rgb -> cạnh ngắn {int(size * 256 / 224)} (bilinear) -> cắt giữa {size} "
                             "-> chuẩn hóa mean/std ImageNet (eval_tf của 02)"}


def run_deep(name, meta, cfg, **kw):
    """Trích + kiểm tra + lưu một đặc trưng deep, in thống kê. Trả về X (None nếu bị bỏ qua)."""
    X, st = extract_deep(name, meta, cfg, **kw)
    if X is None:
        print(f"BỎ QUA {name}: {st['skipped']}")
        return None
    print(name, check_features(X, meta), st)
    save_features(name, X, cfg, st)
    return X


def show_deep_demo(cfg, rel, name, topk=5):
    """Ảnh đầu vào (sau tiền xử lý) -> vector đặc trưng -> top-k lớp ImageNet khi gắn lại head gốc."""
    import matplotlib.pyplot as plt
    import torch
    _fill_deep_active()
    size = 224 if name == "vit_b16" else _deep_size()
    model, head, weights = build_backbone(name, "cpu")
    img = load_image(rel, size)
    with torch.inference_mode():
        feat = model(_to_tensor(img, cfg)[None])
        prob = head(feat).softmax(1)[0]
    feat = feat[0].numpy()
    p, i = prob.topk(topk)
    cats = weights.meta["categories"]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4), gridspec_kw={"width_ratios": [1, 2.2, 1.4]})
    ax[0].imshow(img); ax[0].set_title(f"Ảnh đầu vào {size}x{size}"); ax[0].axis("off")
    ax[1].bar(range(len(feat)), feat, width=1.0, color="tab:blue")
    ax[1].set_xlim(-1, len(feat))
    ax[1].set_title(f"Vector đặc trưng {len(feat)} chiều (min {feat.min():.2f}, max {feat.max():.2f})")
    ax[1].set_xlabel("chiều")
    ax[2].barh(range(topk)[::-1], p.numpy(), color="tab:orange")
    ax[2].set_yticks(range(topk)[::-1]); ax[2].set_yticklabels([cats[k] for k in i.tolist()], fontsize=8)
    ax[2].set_xlim(0, 1); ax[2].set_title(f"Top-{topk} ImageNet (head gốc)")
    fig.suptitle(f"{name} ({weights})", fontsize=10)
    fig.tight_layout()
    return fig
