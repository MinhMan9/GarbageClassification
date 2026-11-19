"""PCA 512-dim extension: fixed ViT-B/16 + StandardScaler + LR(C=1).
Run from project root: .venv\Scripts\python.exe scripts\pca_512_ablation.py
No test access. Writes independent pca512_* artifacts; preserves pca_* 768/256 experiment.
"""
import json
import platform
import sys
import warnings
from pathlib import Path
from hashlib import sha256

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.classify import (
    evaluate_deep_cv, load_deep_embedding, load_deep_metadata,
    make_deep_folds, summarize_deep_oof, file_sha256, save_deep_results,
)

def main():
    out = ROOT / "artifacts/results/classify_deep"
    dev, label2id, prep, n_total = load_deep_metadata(ROOT)
    folds, fold_table = make_deep_folds(dev, prep["seed"])
    y = dev["label_id"].to_numpy()
    assert len(dev) == 12692 and n_total == 15866
    assert set(dev["split"]) <= {"train", "val"} and "test" not in set(dev["split"])
    stored_fold_table = pd.read_csv(out / "folds.csv", dtype={"md5": str})
    pd.testing.assert_frame_equal(stored_fold_table, fold_table)

    parent = json.loads((out / "pca_run.json").read_text(encoding="utf-8"))
    if parent["label2id"] if "label2id" in parent else False:
        raise ValueError("Unexpected experimental metadata format")
    assert parent["embedding"] == "vit_b16"
    assert parent["logistic_regression"]["C"] == 1.0
    assert parent["pca"]["n_components"] == 256
    assert parent["pca"]["svd_solver"] == "randomized"
    assert parent["pca"]["whiten"] is False
    for name, digest in parent["input_sha256"].items():
        if file_sha256(ROOT / name) != digest:
            raise ValueError(f"Input has changed: {name}")
    for name, digest in parent["output_sha256"].items():
        if file_sha256(out / name) != digest:
            raise ValueError(f"Original PCA results changed: {name}")
    old_scores = pd.read_csv(out / "pca_cv_metrics.csv")
    old_pred = pd.read_csv(out / "pca_oof_predictions.csv", dtype={"md5": str})
    assert len(old_pred) == len(dev) * 2

    pca_params = dict(parent["pca"])
    pca_params["n_components"] = 512
    estimator = make_pipeline(
        StandardScaler(),
        PCA(**pca_params),
        LogisticRegression(**parent["logistic_regression"]),
    )
    print("Starting PCA 512 / LR(C=1): five independent training folds...", flush=True)
    X = load_deep_embedding(ROOT, "vit_b16", dev, prep, n_total)
    with threadpool_limits(limits=4, user_api="blas"), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        scores, predicted = evaluate_deep_cv(estimator, X, y, folds, pca_step="pca")
    metrics = scores.assign(model="logistic_regression_pca512", embedding="vit_b16")
    oof = fold_table.assign(
        model="logistic_regression_pca512", embedding="vit_b16",
        y_true=y, y_pred=predicted, stage="pca_ablation"
    )
    assert len(oof) == len(dev) and oof["md5"].is_unique
    ref = old_pred.loc[old_pred["model"].eq("logistic_regression_pca256"),
                       ["md5","fold_id","y_true"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        oof[["md5","fold_id","y_true"]].reset_index(drop=True), ref
    )
    class_table, conf_table = summarize_deep_oof(oof, label2id)
    class_folds, _ = summarize_deep_oof(oof, label2id, by_fold=True)
    resources = pd.DataFrame([{
        "model": "logistic_regression_pca512",
        "classifier_dimension": 512,
        "vector_storage_dtype": "float32",
        "vector_payload_bytes": 512 * 4,
        "pca_components_bytes": int(scores["pca_components_bytes"].iloc[0]),
        "retained_variance_mean": float(scores["pca_variance_retained"].mean()),
        "retained_variance_std": float(scores["pca_variance_retained"].std(ddof=1)),
    }])
    extras = {
        "pca512_class_metrics.csv": class_table,
        "pca512_class_metrics_folds.csv": class_folds,
        "pca512_confusion.csv": conf_table,
        "pca512_resources.csv": resources,
    }
    for filename, frame in extras.items():
        frame.to_csv(out / filename, index=False, lineterminator="\n")
    manifest = {
        "purpose": "PCA 512 dimension extension to frozen ViT B16 / LR C=1.",
        "source_pca_run_sha256": file_sha256(out / "pca_run.json"),
        "source_pca_metrics_sha256": file_sha256(out / "pca_cv_metrics.csv"),
        "cv": parent["cv"],
        "pca": pca_params,
        "logistic_regression": parent["logistic_regression"],
        "preprocessing": parent["preprocessing"],
        "sample_count": len(dev),
        "seed": prep["seed"],
        "versions": {"python": platform.python_version(),
                     "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__},
        "input_sha256": parent["input_sha256"],
        "additional_output_sha256":
            {name: file_sha256(out / name) for name in extras},
    }
    summary = save_deep_results(out, "pca512", metrics, oof, manifest)
    print(summary.to_string(index=False), flush=True)
    print(resources.to_string(index=False), flush=True)
    print("Result verification: same 5 folds, no test rows, own artifacts saved.", flush=True)

if __name__ == "__main__":
    main()
