import json
from hashlib import file_digest
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# Traditional

# Deep
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
