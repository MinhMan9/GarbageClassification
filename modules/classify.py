"""
modules/classify.py
===================
Huấn luyện, tuning siêu tham số (5-fold CV) và ablation PCA cho các mô hình
phân loại cổ điển (DummyClassifier, Logistic Regression, SVM, Random Forest).

Module chia làm hai phần giống modules/extract.py:
- # Traditional (notebook 05): phân loại trên đặc trưng truyền thống (hog, sift_bovw, ...).
- # Deep        (notebook 06): phần của đặc trưng học sâu sẽ thêm ở cuối file.

Cách dùng (notebook 05)
-----------------------
    import modules.extract as ex
    import modules.evaluation as ev
    import modules.classify as clf

    cfg = ex.load_config()
    data = clf.load_dev_data(["hog", "sift_bovw"], cfg)
    cv = ev.get_cv(n_splits=cfg["n_splits"], seed=cfg["seed"])

    df_base = clf.run_baselines(data["X_dev"], data["y_dev"], cv, seed=cfg["seed"])
    df_lr, best_lr = clf.tune_model("hog", data["X_dev"]["hog"], data["y_dev"], cv, "lr", [...])
"""

import json
import os
import time
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from modules import extract as ex
from modules import evaluation as ev

from hashlib import file_digest
from pathlib import Path
from time import perf_counter

# =========================================================================== #
# Traditional (notebook 05)
# =========================================================================== #

# Tên hiển thị đẹp cho từng mã mô hình
MODEL_LABELS = {
    "dummy_freq": "Dummy (most_frequent)",
    "dummy_strat": "Dummy (stratified)",
    "lr": "Logistic Regression",
    "linear_svm": "SVM (Linear)",
    "sgd_svm": "SVM (Linear-SGD)",
    "rbf_svm": "SVM (RBF)",
    "rf": "Random Forest",
}


# --------------------------------------------------------------------------- #
# 1. Nạp dữ liệu Development và đặc trưng đã trích xuất
# --------------------------------------------------------------------------- #

def load_dev_data(feature_names: list, cfg: dict) -> dict:
    """
    Nạp meta.csv, ánh xạ nhãn và các file đặc trưng .npy, sau đó tách riêng
    tập Development (train + val = 80%) để chạy 5-fold CV.

    Tập Final Test (20%) được giữ riêng chỉ số `test_idx` và không chạm tới
    trong bước chọn mô hình / tuning siêu tham số.

    Parameters
    ----------
    feature_names : list[str]
        Danh sách tên đặc trưng cần nạp (vd: ['hog', 'sift_bovw']).
    cfg : dict
        Cấu hình đọc từ ex.load_config().

    Returns
    -------
    dict
        Gồm: meta, splits, dev_idx, test_idx, label2id, id2label, class_names,
        y_dev, X_dev (dict {tên_đặc_trưng: np.ndarray}).
    """
    meta, splits = ex.load_meta()
    with open(os.path.join(ex.PROCESSED_DIR, "label2id.json"), encoding="utf-8") as f:
        label2id = json.load(f)
    id2label = {int(v): k for k, v in label2id.items()}
    class_names = [id2label[i] for i in range(len(id2label))]

    dev_idx = splits["train"] + splits["val"]
    test_idx = splits["test"]

    y_all = meta["label"].map(label2id).to_numpy(dtype=np.int64)
    y_dev = y_all[dev_idx]

    X_dev = {}
    for name in feature_names:
        X_full = ex.load_features(name, cfg)
        X_dev[name] = np.ascontiguousarray(X_full[dev_idx], dtype=np.float32)

    print(f"Tổng số ảnh: {len(meta)} | Development (5-fold CV): {len(dev_idx)} | Final Test (niêm phong): {len(test_idx)}")
    for name, X in X_dev.items():
        l2_mean = float(np.linalg.norm(X, axis=1).mean())
        print(f"  - {name:14s}: shape={X.shape}, min={X.min():.3f}, max={X.max():.3f}, L2_mean={l2_mean:.3f}")

    return {
        "meta": meta,
        "splits": splits,
        "dev_idx": dev_idx,
        "test_idx": test_idx,
        "label2id": label2id,
        "id2label": id2label,
        "class_names": class_names,
        "y_dev": y_dev,
        "X_dev": X_dev,
    }


# --------------------------------------------------------------------------- #
# 2. Khởi tạo Pipeline mô hình (chống rò rỉ dữ liệu giữa các fold)
# --------------------------------------------------------------------------- #

def _format_params(params: dict) -> str:
    """Chuyển dict tham số thành chuỗi ngắn gọn để hiển thị trong bảng kết quả."""
    if not params:
        return "default"
    return ", ".join(f"{k}={v}" for k, v in params.items())


def make_pipeline(
    model_type: str,
    params: dict = None,
    scale: bool = True,
    pca_dims: int = None,
    seed: int = 42,
) -> Pipeline:
    """
    Lắp ráp sklearn Pipeline: [StandardScaler] -> [PCA] -> Classifier.

    Mọi bước tiền xử lý (chuẩn hóa thang đo, giảm chiều PCA) được đặt BÊN TRONG
    Pipeline để khi chạy 5-fold CV, chúng chỉ học thống kê trên 4 phần train
    của từng fold, tuyệt đối không rò rỉ thông tin từ phần validation.

    Parameters
    ----------
    model_type : str
        Một trong: 'dummy_freq', 'dummy_strat', 'lr', 'linear_svm', 'rbf_svm', 'rf'.
    params : dict, optional
        Siêu tham số ghi đè cho bộ phân loại (vd: {'C': 1.0, 'class_weight': 'balanced'}).
    scale : bool, default=True
        Có chuẩn hóa StandardScaler trước khi vào mô hình/PCA hay không.
        (Lưu ý: Dummy và Random Forest không cần chuẩn hóa nếu không dùng PCA.)
    pca_dims : int, optional
        Số chiều cần nén bằng PCA. Bỏ trống (None) nếu dùng đặc trưng gốc.
    seed : int, default=42
        Random seed cố định.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """
    p = dict(params or {})
    steps = []

    # Chuẩn hóa thang đo khi dùng LR / SVM hoặc khi có bước PCA (PCA nhạy với thang đo)
    need_scale = scale and (model_type not in ("dummy_freq", "dummy_strat", "rf") or pca_dims is not None)
    if need_scale:
        steps.append(("scaler", StandardScaler()))

    if pca_dims is not None:
        # svd_solver='randomized' giúp PCA chạy nhanh hơn nhiều khi dữ liệu 8100 chiều (HOG)
        steps.append(("pca", PCA(n_components=pca_dims, svd_solver="randomized", random_state=seed)))

    if model_type == "dummy_freq":
        clf = DummyClassifier(strategy="most_frequent")
    elif model_type == "dummy_strat":
        clf = DummyClassifier(strategy="stratified", random_state=seed)
    elif model_type == "lr":
        cfg_lr = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "random_state": seed}
        cfg_lr.update(p)
        clf = LogisticRegression(**cfg_lr)
    elif model_type == "linear_svm":
        cfg_svm = {"C": 0.1, "class_weight": "balanced", "max_iter": 1500, "tol": 1e-3, "dual": True, "random_state": seed}
        cfg_svm.update(p)
        clf = LinearSVC(**cfg_svm)
    elif model_type == "sgd_svm":
        cfg_sgd = {"loss": "hinge", "penalty": "l2", "alpha": 0.1, "class_weight": "balanced", "max_iter": 1000, "tol": 1e-3, "random_state": seed}
        cfg_sgd.update(p)
        clf = SGDClassifier(**cfg_sgd)
    elif model_type == "rbf_svm":
        cfg_rbf = {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "random_state": seed}
        cfg_rbf.update(p)
        clf = SVC(**cfg_rbf)
    elif model_type == "rf":
        cfg_rf = {
            "n_estimators": 200,
            "max_depth": None,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": seed,
        }
        cfg_rf.update(p)
        clf = RandomForestClassifier(**cfg_rf)
    else:
        raise ValueError(f"Không hỗ trợ model_type='{model_type}'. Chọn trong: {list(MODEL_LABELS)}")

    steps.append(("clf", clf))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# 3. Baseline sàn (Dummy) & Baseline chính (Logistic Regression)
# --------------------------------------------------------------------------- #

def run_baselines(
    X_dev_dict: dict,
    y_dev: np.ndarray,
    cv,
    seed: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Chạy 5-fold CV cho baseline sàn (DummyClassifier) và baseline chính
    (Logistic Regression mặc định C=1.0, class_weight='balanced') trên các đặc trưng.
    """
    rows = []
    first_feat = next(iter(X_dev_dict))
    X_first = X_dev_dict[first_feat]

    # 1. Baseline sàn (không phụ thuộc đặc trưng ảnh)
    for m_type in ("dummy_freq", "dummy_strat"):
        pipe = make_pipeline(m_type, seed=seed)
        row = ev.evaluate_cv(pipe, X_first, y_dev, cv, "(none)", MODEL_LABELS[m_type], "baseline sàn", n_jobs=n_jobs)
        rows.append(row)
        print(f"[Baseline sàn] {MODEL_LABELS[m_type]:22s} | Macro-F1: {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f} "
              f"| Acc: {row['val_acc_mean']:.4f}")

    # 2. Baseline chính: Logistic Regression mặc định trên từng đặc trưng
    for feat_name, X in X_dev_dict.items():
        params = {"C": 1.0, "class_weight": "balanced"}
        pipe = make_pipeline("lr", params=params, scale=True, seed=seed)
        feat_jobs = 1 if X.shape[1] > 2000 else n_jobs
        row = ev.evaluate_cv(pipe, X, y_dev, cv, feat_name, MODEL_LABELS["lr"], _format_params(params), n_jobs=feat_jobs)
        rows.append(row)
        print(f"[Baseline LR ] {feat_name:14s} | Macro-F1: {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f} "
              f"| Acc: {row['val_acc_mean']:.4f} | Fit: {row['fit_time_mean']:.1f}s/fold")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Tuning siêu tham số qua 5-fold Cross Validation
# --------------------------------------------------------------------------- #

def tune_model(
    feature_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    model_type: str,
    param_list: list,
    scale: bool = True,
    pca_dims: int = None,
    seed: int = 42,
    n_jobs: int = -1,
):
    """
    Khảo sát danh sách các bộ siêu tham số `param_list` trên 5-fold CV,
    in tiến độ từng cấu hình và trả về (df_results, best_config).

    Lưu ý về đa luồng (n_jobs):
    - Random Forest đã tự bật n_jobs=-1 bên trong, nên tự động đặt n_jobs=1 ở vòng
      cross_validate để tránh xung đột luồng trên Windows.

    Parameters
    ----------
    feature_name : str
        Tên đặc trưng (vd: 'hog', 'sift_bovw').
    X, y : np.ndarray
        Đặc trưng và nhãn tập Development.
    cv : StratifiedKFold
    model_type : str
        Mã mô hình ('lr', 'linear_svm', 'rbf_svm', 'rf').
    param_list : list[dict]
        Danh sách các dict tham số cần thử (vd: [{'C': 0.01}, {'C': 0.1}, {'C': 1.0}]).

    Returns
    -------
    (df, best_info) : tuple[pd.DataFrame, dict]
        df: bảng kết quả tất cả cấu hình đã thử (kèm cột từng tham số để dễ vẽ biểu đồ).
        best_info: dict chứa cấu hình đạt val_f1_mean cao nhất.
    """
    # RF đã tự đa luồng; dữ liệu nhiều chiều (>2000) chạy 1 luồng CV để tránh nhân bản RAM gây swap
    cv_jobs = 1 if (model_type == "rf" or X.shape[1] > 2000) else n_jobs
    model_label = MODEL_LABELS[model_type]
    rows = []

    print(f"=== Tuning {model_label} trên `{feature_name}` ({len(param_list)} cấu hình × 5 fold) ===")
    for i, params in enumerate(param_list, 1):
        t0 = time.time()
        pipe = make_pipeline(model_type, params=params, scale=scale, pca_dims=pca_dims, seed=seed)
        p_str = _format_params(params)
        if pca_dims is not None:
            p_str = f"pca={pca_dims}, {p_str}"

        row = ev.evaluate_cv(pipe, X, y, cv, feature_name, model_label, p_str, n_jobs=cv_jobs)
        # Lưu kèm từng tham số riêng lẻ thành cột để plot_tuning_curve đọc trực tiếp
        for k, v in params.items():
            row[f"param_{k}"] = "None" if v is None else v
        row["model_type"] = model_type
        row["params_dict"] = json.dumps(params)
        rows.append(row)

        elapsed = time.time() - t0
        print(f"  [{i}/{len(param_list)}] {p_str:38s} -> "
              f"Val F1: {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f} | "
              f"Train F1: {row['train_f1_mean']:.4f} | ({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    best_idx = int(df["val_f1_mean"].idxmax())
    best_row = df.loc[best_idx]
    best_info = {
        "feature": feature_name,
        "model_type": model_type,
        "model": model_label,
        "params": json.loads(best_row["params_dict"]),
        "params_str": best_row["params"],
        "val_f1_mean": float(best_row["val_f1_mean"]),
        "val_f1_std": float(best_row["val_f1_std"]),
    }
    print(f"--> Tốt nhất: {best_info['params_str']} | Macro-F1 = {best_info['val_f1_mean']:.4f} ± {best_info['val_f1_std']:.4f}\n")
    return df, best_info


# --------------------------------------------------------------------------- #
# 5. Ablation Study: Giảm chiều đặc trưng bằng PCA
# --------------------------------------------------------------------------- #

def compute_pca_variance(X: np.ndarray, max_components: int = 1024, seed: int = 42) -> np.ndarray:
    """
    Tính mảng phương sai tích lũy (cumulative explained variance ratio) khi nén
    ma trận X (sau StandardScaler) bằng PCA tối đa `max_components` chiều.
    Chỉ dùng để minh họa mức độ giữ thông tin của PCA.
    """
    k = min(max_components, X.shape[0], X.shape[1])
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    pca.fit(X_scaled)
    return np.cumsum(pca.explained_variance_ratio_)


def run_pca_ablation(
    feature_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    model_specs: list,
    k_list: list,
    seed: int = 42,
    n_jobs: int = -1,
    include_original: bool = True,
) -> pd.DataFrame:
    """
    Thí nghiệm Ablation PCA: so sánh Macro-F1 (mean ± std) và thời gian huấn luyện
    khi nén đặc trưng xuống từng mức số chiều trong `k_list` so với đặc trưng gốc.

    PCA luôn nằm trong Pipeline để tránh data leakage giữa các fold.

    Parameters
    ----------
    feature_name : str
        Tên đặc trưng ('hog' hoặc 'sift_bovw').
    X, y : np.ndarray
        Đặc trưng và nhãn tập Development.
    cv : StratifiedKFold
    model_specs : list[dict]
        Danh sách mô hình muốn thử, mỗi phần tử có dạng:
        {'model_type': 'lr', 'params': {'C': 0.1, 'class_weight': 'balanced'}}.
    k_list : list[int]
        Các mốc số chiều PCA cần khảo sát (vd: [32, 64, 128, 256, 512]).
    include_original : bool, default=True
        Có chạy kèm mốc gốc (pca_dims=None) để đối chiếu hay không.

    Returns
    -------
    pd.DataFrame
    """
    dims_to_run = ([None] if include_original else []) + [k for k in k_list if k < X.shape[1]]
    rows = []

    print(f"=== Ablation PCA trên `{feature_name}` (gốc {X.shape[1]} chiều) ===")
    for spec in model_specs:
        m_type = spec["model_type"]
        params = spec.get("params", {})
        m_label = spec.get("label", MODEL_LABELS[m_type])
        cv_jobs = 1 if (m_type == "rf" or X.shape[1] > 2000) else n_jobs

        for k in dims_to_run:
            t0 = time.time()
            pipe = make_pipeline(m_type, params=params, scale=True, pca_dims=k, seed=seed)
            p_str = _format_params(params)
            tag = f"gốc ({X.shape[1]}d)" if k is None else f"pca={k}, {p_str}"
            row = ev.evaluate_cv(pipe, X, y, cv, feature_name, m_label, tag, n_jobs=cv_jobs)
            row["pca_dims"] = np.nan if k is None else int(k)
            row["model_type"] = m_type
            row["params_dict"] = json.dumps(params)
            rows.append(row)

            elapsed = time.time() - t0
            print(f"  {m_label:20s} | {tag:12s} -> "
                  f"Val F1: {row['val_f1_mean']:.4f} ± {row['val_f1_std']:.4f} | "
                  f"Fit: {row['fit_time_mean']:.2f}s/fold (tổng {elapsed:.1f}s)")

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 6. Lưu/nạp cache từng bước & Tự động mở rộng biên Tuning (Coarse-to-Fine)
# --------------------------------------------------------------------------- #

def save_cache(df: pd.DataFrame, name: str, cfg: dict) -> str:
    """Lưu nháp kết quả từng cell ra artifacts/results/cache/<name>.csv."""
    cdir = os.path.join(cfg["results_dir"], "cache")
    os.makedirs(cdir, exist_ok=True)
    path = os.path.join(cdir, f"{name}.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_cache(name: str, cfg: dict) -> pd.DataFrame:
    """Nạp lại bảng kết quả đã lưu trong cache."""
    path = os.path.join(cfg["results_dir"], "cache", f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Chưa có cache '{path}'. Hãy chạy cell tương ứng trước.")
    return pd.read_csv(path)


def best_from_df(df: pd.DataFrame) -> dict:
    """Trích thông tin cấu hình đạt val_f1_mean cao nhất từ DataFrame tuning."""
    best_row = df.loc[int(df["val_f1_mean"].idxmax())]
    return {
        "feature": best_row["feature"],
        "model_type": best_row["model_type"],
        "model": best_row["model"],
        "params": json.loads(best_row["params_dict"]),
        "params_str": best_row["params"],
        "val_f1_mean": float(best_row["val_f1_mean"]),
        "val_f1_std": float(best_row["val_f1_std"]),
    }


def expand_tuning_edge(
    df_prev: pd.DataFrame,
    feature_name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    model_type: str,
    candidate_params: list,
    sort_col: str = None,
    scale: bool = True,
    seed: int = 42,
    n_jobs: int = -1,
    tol_drop: float = 0.002,
):
    """
    Tự động mở rộng biên (Coarse-to-Fine Iterative Grid Refinement):
    Chạy tiếp các bộ tham số mới trong `candidate_params` (bỏ qua cái đã có trong `df_prev`),
    hễ thấy Val Macro-F1 bắt đầu giảm quá `tol_drop` so với đỉnh đang giữ thì DỪNG NGAY.
    """
    existing = set(df_prev["params_dict"].tolist())
    to_try = [p for p in candidate_params if json.dumps(p) not in existing]

    best_f1 = float(df_prev["val_f1_mean"].max())
    cv_jobs = 1 if (model_type == "rf" or X.shape[1] > 2000) else n_jobs
    model_label = MODEL_LABELS[model_type]
    new_rows = []

    print(f"=== Mở rộng biên {model_label} trên `{feature_name}` (đỉnh hiện tại: {best_f1:.4f}) ===")
    for i, params in enumerate(to_try, 1):
        t0 = time.time()
        pipe = make_pipeline(model_type, params=params, scale=scale, seed=seed)
        p_str = _format_params(params)
        row = ev.evaluate_cv(pipe, X, y, cv, feature_name, model_label, p_str, n_jobs=cv_jobs)
        for k, v in params.items():
            row[f"param_{k}"] = "None" if v is None else v
        row["model_type"] = model_type
        row["params_dict"] = json.dumps(params)
        new_rows.append(row)

        vf1 = row["val_f1_mean"]
        print(f"  [+{i}] {p_str:38s} -> Val F1: {vf1:.4f} ± {row['val_f1_std']:.4f} | "
              f"Train F1: {row['train_f1_mean']:.4f} | ({time.time() - t0:.1f}s)")

        if vf1 > best_f1:
            best_f1 = vf1
        elif vf1 < best_f1 - tol_drop:
            print(f"  --> Điểm đã đi xuống ({vf1:.4f} < {best_f1:.4f}), đã bắt trọn đỉnh. Dừng mở rộng!")
            break

    df_merged = pd.concat([df_prev, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else df_prev.copy()
    if sort_col and sort_col in df_merged.columns:
        df_merged = df_merged.sort_values(sort_col).reset_index(drop=True)
    best_info = best_from_df(df_merged)
    print(f"--> Đỉnh cuối cùng: {best_info['params_str']} | Macro-F1 = {best_info['val_f1_mean']:.4f} ± {best_info['val_f1_std']:.4f}\n")
    return df_merged, best_info


# =========================================================================== #
# Deep (notebook 06)
# =========================================================================== #
DEEP_FEATURE_DIMS = {"resnet50": 2048, "efficientnet_b0": 1280, "vit_b16": 768}


def load_deep_metadata(repo_root):
    """Trả về metadata development, ánh xạ nhãn, cấu hình prep và tổng số hàng."""
    processed = Path(repo_root) / "data" / "processed"
    meta = pd.read_csv(processed / "meta.csv", dtype={"md5": str})
    label2id = json.loads((processed / "label2id.json").read_text(encoding="utf-8"))
    prep = json.loads((processed / "config.json").read_text(encoding="utf-8"))

    columns = ["path", "md5", "label", "split"]
    if not set(columns).issubset(meta.columns) or meta[columns].isna().any().any():
        raise ValueError("meta.csv thiếu cột hoặc thiếu giá trị bắt buộc.")
    if not meta["md5"].is_unique or not meta["path"].is_unique:
        raise ValueError("meta.csv có MD5 hoặc đường dẫn trùng.")
    if set(meta["split"]) != {"train", "val", "test"}:
        raise ValueError("meta.csv phải có đủ train, val và test.")
    if set(meta["label"]) != set(label2id) or sorted(label2id.values()) != list(range(len(label2id))):
        raise ValueError("Nhãn trong meta.csv không khớp label2id.json hoặc ID không liên tục.")

    development = meta.loc[meta["split"].isin(["train", "val"]), columns].copy()
    # row_id là vị trí trong .npy; chỉ số sau reset_index là vị trí trong development.
    development.insert(0, "row_id", development.index)
    development["label_id"] = development["label"].map(label2id)
    if development["label_id"].nunique() != len(label2id):
        raise ValueError("Development thiếu ít nhất một lớp.")
    return development.reset_index(drop=True), label2id, prep, len(meta)


def load_deep_embedding(repo_root, name, development, prep, n_rows):
    """Nạp ma trận embedding theo thứ tự của development."""
    if name not in DEEP_FEATURE_DIMS:
        raise ValueError(f"Không nhận diện được embedding {name!r}.")

    folder = Path(repo_root) / "artifacts" / "features"
    info = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
    expected_shape = (n_rows, DEEP_FEATURE_DIMS[name])
    if info.get("name") != name or info.get("shape") != list(expected_shape) or info.get("prep") != prep:
        raise ValueError(f"{name}.json không khớp cấu hình hoặc kích thước dự kiến.")

    features = np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if features.shape != expected_shape or features.dtype != np.float32:
        raise ValueError(f"{name}.npy phải có shape {expected_shape} và dtype float32.")
    X = features[development["row_id"].to_numpy()]
    if not np.isfinite(X).all() or not X.any(axis=1).all():
        raise ValueError(f"{name} có NaN, Inf hoặc vector toàn 0 trong development.")
    return X


def make_deep_folds(development, seed):
    """Trả về các cặp vị trí train/val và bảng MD5 với fold_id từ 1 đến 5."""
    labels = development["label_id"]
    if labels.empty or labels.value_counts().min() < 5:
        raise ValueError("Mỗi lớp development phải có ít nhất 5 ảnh để chia 5 fold.")

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(development)), labels))
    fold_table = development[["md5"]].copy()
    fold_ids = np.empty(len(development), dtype=int)
    for fold_id, (_, val_indices) in enumerate(folds, start=1):
        fold_ids[val_indices] = fold_id
    fold_table["fold_id"] = fold_ids
    return folds, fold_table


def evaluate_deep_cv(estimator, X, y, folds, *, return_search=False, pca_step=None):
    """Trả về điểm OOF, thêm lịch sử search hoặc thông số bước PCA khi được yêu cầu."""
    if return_search and not isinstance(estimator, GridSearchCV):
        raise TypeError("return_search=True yêu cầu estimator là GridSearchCV.")
    class_ids = np.unique(y)
    predictions = np.full(len(y), -1, dtype=int)
    rows = []
    search_rows = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        if np.any(predictions[val_idx] != -1):
            raise ValueError("Một ảnh xuất hiện ở validation của nhiều fold.")

        model = clone(estimator)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        start = perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = perf_counter() - start

        start = perf_counter()
        y_pred = model.predict(X_val)
        predict_seconds = perf_counter() - start

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_pred, labels=class_ids, average="macro", zero_division=0
        )
        predictions[val_idx] = y_pred
        rows.append({
            "fold_id": fold_id,
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "macro_f1": f1,
            "accuracy": accuracy_score(y_val, y_pred),
            "macro_precision": precision,
            "macro_recall": recall,
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
        })
        if pca_step is not None:
            fitted_model = model.best_estimator_ if return_search else model
            pca = fitted_model.named_steps[pca_step]
            rows[-1].update({
                "pca_components": pca.n_components_,
                "pca_training_samples": pca.n_samples_,
                "pca_variance_retained": float(pca.explained_variance_ratio_.sum()),
                "pca_components_bytes": pca.components_.nbytes,
            })
        if return_search:
            rows[-1].update({
                "best_params": json.dumps(model.best_params_, sort_keys=True),
                "best_inner_macro_f1": model.best_score_,
                "refit_seconds": model.refit_time_,
                "tuning_seconds": max(0.0, fit_seconds - model.refit_time_),
            })
            search_rows.append(pd.DataFrame({
                "fold_id": fold_id,
                "params": [json.dumps(p, sort_keys=True) for p in model.cv_results_["params"]],
                "inner_macro_f1": model.cv_results_["mean_test_score"],
                "inner_fit_seconds": model.cv_results_["mean_fit_time"],
                "selected": np.arange(len(model.cv_results_["params"])) == model.best_index_,
            }))
            print(f"Fold {fold_id}: {model.best_params_}, outer Macro F1={f1:.5f}", flush=True)

    if np.any(predictions == -1):
        raise ValueError("Có ảnh development chưa nhận dự đoán OOF.")

    if return_search:
        return pd.DataFrame(rows), predictions, pd.concat(search_rows, ignore_index=True)
    return pd.DataFrame(rows), predictions


def file_sha256(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def save_deep_results(results_dir, prefix, metrics, oof, run_info):
    """Tổng hợp theo cấu hình và lưu cùng định dạng cho các thí nghiệm deep."""
    metric_names = [
        "macro_f1", "accuracy", "macro_precision", "macro_recall",
        "fit_seconds", "predict_seconds",
    ]
    summary = (
        metrics.groupby(["model", "embedding"], sort=False)[metric_names]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    tables = {
        f"{prefix}_cv_metrics.csv": metrics,
        f"{prefix}_cv_summary.csv": summary,
        f"{prefix}_oof_predictions.csv": oof,
    }
    for filename, table in tables.items():
        table.to_csv(results_dir / filename, index=False, lineterminator="\n")
    manifest = {
        **run_info,
        "output_sha256": {
            filename: file_sha256(results_dir / filename) for filename in tables
        },
    }
    (results_dir / f"{prefix}_run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def summarize_deep_oof(predictions, label2id, *, by_fold=False):
    """Tính metric từng lớp và confusion matrix từ các dự đoán OOF."""
    class_names = sorted(label2id, key=label2id.get)
    class_ids = [label2id[name] for name in class_names]
    if not predictions[["y_true", "y_pred"]].isin(class_ids).all().all():
        raise ValueError("Dự đoán chứa ID nhãn ngoài label2id.")

    group_columns = ["stage", "model", "embedding"]
    if by_fold:
        group_columns.append("fold_id")
    class_tables = []
    confusion_tables = []
    for keys, rows in predictions.groupby(group_columns, sort=False):
        group = dict(zip(group_columns, keys))
        cm = confusion_matrix(rows["y_true"], rows["y_pred"], labels=class_ids)
        precision, recall, f1, support = precision_recall_fscore_support(
            rows["y_true"], rows["y_pred"], labels=class_ids, zero_division=0
        )
        true_positive = cm.diagonal()
        class_tables.append(pd.DataFrame({
            **group, "class_id": class_ids, "label": class_names,
            "support": support, "predicted_count": cm.sum(axis=0),
            "tp": true_positive, "fp": cm.sum(axis=0) - true_positive,
            "fn": support - true_positive, "precision": precision,
            "recall": recall, "f1": f1,
        }))
        rates = np.divide(
            cm, support[:, None], out=np.zeros_like(cm, dtype=float),
            where=support[:, None] != 0,
        )
        confusion_tables.append(pd.DataFrame({
            **group, "true_id": np.repeat(class_ids, len(class_ids)),
            "predicted_id": np.tile(class_ids, len(class_ids)),
            "true_label": np.repeat(class_names, len(class_ids)),
            "predicted_label": np.tile(class_names, len(class_ids)),
            "count": cm.ravel(), "row_rate": rates.ravel(),
        }))
    return pd.concat(class_tables, ignore_index=True), pd.concat(confusion_tables, ignore_index=True)
