"""
modules/evaluation.py
=====================
Đánh giá mô hình phân loại bằng 5-fold Cross Validation, tính mean ± std
cho các chỉ số (macro-F1, accuracy, macro precision/recall, thời gian chạy),
lưu bảng kết quả (.csv) và vẽ biểu đồ phục vụ notebook 05, 06, 07.

Cách dùng
---------
    import modules.evaluation as ev

    cv = ev.get_cv(n_splits=5, seed=42)
    row = ev.evaluate_cv(model, X_dev, y_dev, cv, "hog", "Logistic Regression", "C=1.0")
    display(ev.format_results_table(df))
    y_oof = ev.get_oof_predictions(model, X_dev, y_dev, cv)
    fig, rep = ev.plot_confusion_matrix_and_f1(y_dev, y_oof, class_names)
    ev.save_results(df, "trad_best_models_cv.csv", cfg)
    ev.save_figure(fig, "cm_best_trad.png", cfg)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
)


# Bộ chỉ số đánh giá chuẩn theo mô tả BTL
# - Chỉ số chính: macro_f1 (phản ánh công bằng cả 12 lớp dù dữ liệu mất cân bằng)
# - Chỉ số phụ: accuracy, macro_precision, macro_recall
SCORING = {
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
}


# --------------------------------------------------------------------------- #
# 1. Chia fold và đánh giá Cross Validation
# --------------------------------------------------------------------------- #

def get_cv(n_splits: int = 5, seed: int = 42) -> StratifiedKFold:
    """
    Tạo bộ chia StratifiedKFold cố định.

    Stratified đảm bảo tỷ lệ 12 lớp rác giống hệt nhau ở mọi fold;
    shuffle=True + random_state=seed giúp kết quả lặp lại được.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def evaluate_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    feature_name: str,
    model_name: str,
    params_str: str = "",
    n_jobs: int = 1,
) -> dict:
    """
    Chạy 5-fold Cross Validation cho một mô hình và tổng hợp thống kê mean ± std.

    Bên trong, cross_validate tự động chạy model.fit() trên 4 fold và chấm điểm
    trên 1 fold còn lại, lặp đủ 5 lần.

    Parameters
    ----------
    model : estimator hoặc sklearn Pipeline
    X : np.ndarray, shape (n_samples, n_features)
        Ma trận đặc trưng tập Development.
    y : np.ndarray, shape (n_samples,)
        Mảng nhãn số nguyên (0..11).
    cv : StratifiedKFold
        Bộ chia fold tạo từ get_cv().
    feature_name : str
        Tên đặc trưng (vd: 'hog', 'sift_bovw').
    model_name : str
        Tên mô hình (vd: 'LogisticRegression', 'LinearSVC', 'RandomForest').
    params_str : str, optional
        Ghi chú ngắn bộ siêu tham số đang thử (vd: 'C=1.0, balanced').
    n_jobs : int, default=1
        Số luồng CPU chạy song song các fold (đặt -1 nếu model không tự đa luồng).

    Returns
    -------
    dict
        Từ điển chứa mean, std của các chỉ số trên Val & Train và thời gian chạy.
    """
    res = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=n_jobs,
    )

    val_f1 = res["test_macro_f1"]
    train_f1 = res["train_macro_f1"]
    val_acc = res["test_accuracy"]
    val_prec = res["test_macro_precision"]
    val_rec = res["test_macro_recall"]
    fit_t = res["fit_time"]
    score_t = res["score_time"]

    return {
        "feature": feature_name,
        "model": model_name,
        "params": params_str,
        "val_f1_mean": float(np.mean(val_f1)),
        "val_f1_std": float(np.std(val_f1)),
        "train_f1_mean": float(np.mean(train_f1)),
        "train_f1_std": float(np.std(train_f1)),
        "overfit_gap": float(np.mean(train_f1) - np.mean(val_f1)),
        "val_acc_mean": float(np.mean(val_acc)),
        "val_acc_std": float(np.std(val_acc)),
        "val_prec_mean": float(np.mean(val_prec)),
        "val_prec_std": float(np.std(val_prec)),
        "val_rec_mean": float(np.mean(val_rec)),
        "val_rec_std": float(np.std(val_rec)),
        "fit_time_mean": float(np.mean(fit_t)),
        "score_time_mean": float(np.mean(score_t)),
        "total_time": float(np.sum(fit_t) + np.sum(score_t)),
    }


def get_oof_predictions(model, X: np.ndarray, y: np.ndarray, cv, n_jobs: int = 1) -> np.ndarray:
    """
    Lấy dự đoán Out-of-Fold (OOF) cho toàn bộ tập Development qua 5 fold.

    Mỗi mẫu chỉ được dự đoán bởi mô hình KHÔNG nhìn thấy nó lúc train,
    nên dùng để vẽ Confusion Matrix và tính F1 từng lớp một cách trung thực.
    """
    return cross_val_predict(model, X, y, cv=cv, n_jobs=n_jobs)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Tính nhanh 4 chỉ số chuẩn từ mảng nhãn thật và nhãn dự đoán."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# 2. Định dạng bảng và lưu kết quả
# --------------------------------------------------------------------------- #

def _pm(mean_s: pd.Series, std_s: pd.Series, digits: int = 4) -> pd.Series:
    """Định dạng cột dạng 'mean ± std' (vd: '0.6123 ± 0.0085')."""
    return [f"{m:.{digits}f} ± {s:.{digits}f}" for m, s in zip(mean_s, std_s)]


def format_results_table(df: pd.DataFrame, sort_by: str = "val_f1_mean") -> pd.DataFrame:
    """
    Chuyển DataFrame kết quả thô thành bảng gọn đẹp để hiển thị trong notebook/báo cáo.
    """
    d = df.sort_values(sort_by, ascending=False).reset_index(drop=True) if sort_by else df.copy()
    out = pd.DataFrame({
        "Đặc trưng": d["feature"],
        "Mô hình": d["model"],
        "Tham số": d["params"],
        "Macro-F1 (Val)": _pm(d["val_f1_mean"], d["val_f1_std"]),
        "Accuracy (Val)": _pm(d["val_acc_mean"], d["val_acc_std"]),
        "Macro-Prec": d["val_prec_mean"].round(4),
        "Macro-Rec": d["val_rec_mean"].round(4),
        "Train F1": d["train_f1_mean"].round(4),
        "Fit (s/fold)": d["fit_time_mean"].round(2),
    })
    return out


def save_results(df: pd.DataFrame, filename: str, cfg: dict) -> str:
    """Lưu bảng kết quả (.csv) vào thư mục results_dir (artifacts/results/)."""
    os.makedirs(cfg["results_dir"], exist_ok=True)
    if not filename.endswith(".csv"):
        filename += ".csv"
    path = os.path.join(cfg["results_dir"], filename)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Đã lưu bảng kết quả: {path} ({len(df)} dòng)")
    return path


def save_json(data: dict, filename: str, cfg: dict) -> str:
    """Lưu thông tin cấu hình/siêu tham số tốt nhất (.json) vào results_dir."""
    os.makedirs(cfg["results_dir"], exist_ok=True)
    if not filename.endswith(".json"):
        filename += ".json"
    path = os.path.join(cfg["results_dir"], filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Đã lưu JSON: {path}")
    return path


def save_figure(fig, filename: str, cfg: dict, dpi: int = 150) -> str:
    """Lưu biểu đồ (.png) vào thư mục figures_dir (artifacts/figures/)."""
    os.makedirs(cfg["figures_dir"], exist_ok=True)
    if not filename.endswith(".png"):
        filename += ".png"
    path = os.path.join(cfg["figures_dir"], filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Đã lưu biểu đồ: {path}")
    return path


# --------------------------------------------------------------------------- #
# 3. Trực quan hóa: Tuning, So sánh mô hình, Confusion Matrix, Ablation PCA
# --------------------------------------------------------------------------- #

def plot_tuning_curve(
    df: pd.DataFrame,
    param_col: str,
    title: str,
    x_label: str = None,
    log_x: bool = False,
):
    """
    Vẽ đường cong Train Macro-F1 và Val Macro-F1 (kèm vùng ± std) theo siêu tham số.
    Giúp nhìn rõ vùng Underfitting và Overfitting.
    """
    d = df.copy()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = d[param_col]

    # Nếu tham số là chuỗi (vd 'None', 'sqrt') thì dùng chỉ số rời rạc
    is_numeric = pd.api.types.is_numeric_dtype(x)
    x_vals = x.values if is_numeric else np.arange(len(d))

    ax.plot(x_vals, d["train_f1_mean"], "o--", color="tab:orange", label="Train Macro-F1")
    ax.plot(x_vals, d["val_f1_mean"], "o-", color="tab:blue", lw=2, label="Val Macro-F1 (5-fold)")
    ax.fill_between(
        x_vals,
        d["val_f1_mean"] - d["val_f1_std"],
        d["val_f1_mean"] + d["val_f1_std"],
        color="tab:blue",
        alpha=0.18,
    )

    # Đánh dấu điểm tốt nhất trên Val
    best_i = int(d["val_f1_mean"].idxmax())
    best_x = x_vals[d.index.get_loc(best_i)]
    best_y = d.loc[best_i, "val_f1_mean"]
    ax.scatter([best_x], [best_y], color="red", s=80, zorder=5, label=f"Tốt nhất ({best_y:.4f})")

    if is_numeric and log_x:
        ax.set_xscale("log")
    elif not is_numeric:
        ax.set_xticks(x_vals)
        ax.set_xticklabels([str(v) for v in x], rotation=15)

    ax.set_xlabel(x_label or param_col)
    ax.set_ylabel("Macro-F1")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_model_comparison(df_best: pd.DataFrame, title: str = "So sánh 5-Fold CV Macro-F1 trên đặc trưng truyền thống"):
    """
    Vẽ biểu đồ cột nhóm so sánh Macro-F1 (có thanh sai số ± std) giữa các mô hình
    trên từng bộ đặc trưng.
    """
    features = list(df_best["feature"].unique())
    models = list(df_best["model"].unique())

    x = np.arange(len(features))
    width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(max(7.5, 3.2 * len(features)), 4.6))
    for i, m in enumerate(models):
        sub = df_best[df_best["model"] == m].set_index("feature").reindex(features)
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            sub["val_f1_mean"],
            width=width * 0.92,
            yerr=sub["val_f1_std"],
            capsize=4,
            label=m,
        )
        for bar, val in zip(bars, sub["val_f1_mean"]):
            if pd.notna(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.012,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=10)
    ax.set_ylabel("Macro-F1 (mean ± std qua 5 fold)")
    ax.set_ylim(0, min(1.05, float(df_best["val_f1_mean"].max()) + 0.12))
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="Mô hình")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_and_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix & F1 theo từng lớp",
):
    """
    Vẽ 2 hình cạnh nhau:
    - Trái: Confusion Matrix chuẩn hóa theo hàng (%) cho 12 lớp.
    - Phải: Biểu đồ cột ngang F1-score của từng lớp (sắp xếp từ cao đến thấp).
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true") * 100.0
    rep = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    per_class_f1 = pd.Series({c: rep[c]["f1-score"] for c in class_names}).sort_values()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 6.2), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Tỷ lệ (%)"},
        annot_kws={"size": 8},
        ax=ax1,
    )
    ax1.set_xlabel("Nhãn dự đoán (Predicted)")
    ax1.set_ylabel("Nhãn thực tế (True)")
    ax1.set_title("Confusion Matrix (% theo từng lớp thực tế)")
    ax1.tick_params(axis="x", rotation=45)

    colors = ["tab:red" if v < per_class_f1.mean() else "tab:blue" for v in per_class_f1.values]
    bars = ax2.barh(per_class_f1.index, per_class_f1.values, color=colors, alpha=0.85)
    ax2.axvline(
        per_class_f1.mean(),
        color="black",
        linestyle="--",
        lw=1.2,
        label=f"Macro-F1 = {per_class_f1.mean():.4f}",
    )
    for bar, v in zip(bars, per_class_f1.values):
        ax2.text(v + 0.01, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8.5)
    ax2.set_xlim(0, 1.08)
    ax2.set_xlabel("F1-score")
    ax2.set_title("F1-score từng lớp (đỏ = thấp hơn trung bình)")
    ax2.grid(axis="x", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right")

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    return fig, pd.DataFrame(rep).T


def plot_pca_ablation(
    df_pca: pd.DataFrame,
    feature_name: str,
    cum_var: np.ndarray = None,
    orig_dims: int = None,
):
    """
    Vẽ biểu đồ cho thí nghiệm Ablation PCA trên một đặc trưng:
    - Trái: Số chiều PCA vs Macro-F1 (mean ± std) cho các mô hình (kèm đường gốc không PCA).
    - Phải: Số chiều PCA vs Thời gian huấn luyện trung bình mỗi fold (giây).
    """
    sub = df_pca[df_pca["feature"] == feature_name].copy()
    models = list(sub["model"].unique())

    ncols = 3 if cum_var is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.8 * ncols, 4.4))

    ax_f1, ax_time = axes[0], axes[1]
    for m in models:
        dm = sub[sub["model"] == m]
        base = dm[dm["pca_dims"].isna()]
        pca_rows = dm[dm["pca_dims"].notna()].sort_values("pca_dims")

        ks = pca_rows["pca_dims"].astype(int).values
        line, = ax_f1.plot(ks, pca_rows["val_f1_mean"], "o-", lw=2, label=m)
        ax_f1.fill_between(
            ks,
            pca_rows["val_f1_mean"] - pca_rows["val_f1_std"],
            pca_rows["val_f1_mean"] + pca_rows["val_f1_std"],
            color=line.get_color(),
            alpha=0.15,
        )
        ax_time.plot(ks, pca_rows["fit_time_mean"], "o-", lw=2, color=line.get_color(), label=m)

        # Đường nét đứt ngang biểu thị kết quả gốc (không dùng PCA)
        if len(base) > 0:
            b_f1 = float(base["val_f1_mean"].iloc[0])
            b_t = float(base["fit_time_mean"].iloc[0])
            ax_f1.axhline(b_f1, color=line.get_color(), linestyle="--", alpha=0.65,
                          label=f"{m} (gốc: {b_f1:.3f})")
            ax_time.axhline(b_t, color=line.get_color(), linestyle="--", alpha=0.65,
                            label=f"{m} (gốc: {b_t:.1f}s)")

    ax_f1.set_xlabel("Số chiều sau PCA (n_components)")
    ax_f1.set_ylabel("Macro-F1 (5-fold CV)")
    ax_f1.set_title(f"{feature_name}: Số chiều PCA vs. Macro-F1")
    ax_f1.grid(True, linestyle="--", alpha=0.4)
    ax_f1.legend(fontsize=8)

    ax_time.set_xlabel("Số chiều sau PCA (n_components)")
    ax_time.set_ylabel("Thời gian fit trung bình / fold (giây)")
    ax_time.set_title(f"{feature_name}: Số chiều PCA vs. Tốc độ huấn luyện")
    ax_time.grid(True, linestyle="--", alpha=0.4)
    ax_time.legend(fontsize=8)

    if cum_var is not None:
        ax_v = axes[2]
        xs = np.arange(1, len(cum_var) + 1)
        ax_v.plot(xs, cum_var * 100.0, color="tab:green", lw=2)
        for thresh in (80, 90, 95):
            idx = np.searchsorted(cum_var * 100.0, thresh)
            if idx < len(cum_var):
                ax_v.axhline(thresh, color="gray", linestyle=":", lw=1)
                ax_v.scatter([idx + 1], [cum_var[idx] * 100.0], color="red", s=30, zorder=5)
                ax_v.text(idx + 1, thresh - 4, f"{thresh}% @ {idx + 1}d", fontsize=8)
        ax_v.set_xlabel("Số thành phần chính (k)")
        ax_v.set_ylabel("Phương sai tích lũy (%)")
        dim_txt = f" (gốc {orig_dims}d)" if orig_dims else ""
        ax_v.set_title(f"{feature_name}{dim_txt}: Phương sai giữ lại")
        ax_v.set_ylim(0, 102)
        ax_v.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    return fig
