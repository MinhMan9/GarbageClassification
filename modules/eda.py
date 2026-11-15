"""
modules/eda.py
==============
Exploratory Data Analysis (EDA) — phân tích dữ liệu khám phá cho bài toán
phân loại ảnh Garbage Classification (12 classes, bản ENHANCED).

Module này được dùng cho notebook 01_eda.ipynb. Nhãn của ảnh được lấy từ
tên thư mục lớp, ví dụ: <img_dir>/<class_name>/<image>.jpg

Cách dùng
---------
    import modules.eda as eda

    img_dirs, csv_files = eda.find_dataset("../data/raw")
    df = eda.build_image_df(img_dirs["garbage_classification_enhanced"])
    df = eda.add_image_properties(df)
    eda.plot_class_distribution(df)
    eda.compare_class_distribution({"original": df_orig, "enhanced": df})
    eda.plot_correction_matrix(corr, old_col='old_label', new_col='new_label')
    eda.show_sample_images(df, n_per_class=5)
    eda.plot_image_size_distribution(df)
    df = eda.find_duplicates(df)
    stats = eda.compute_image_stats(df, n_per_class=150)
    eda.plot_image_stats(stats)
"""

import hashlib
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
sns.set_theme(style="whitegrid")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Load dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

def find_dataset(root: str, min_classes: int = 5):
    """
    Tìm các thư mục ảnh và file CSV trong thư mục dữ liệu.

    Một thư mục được xem là thư mục ảnh nếu bên trong có ít nhất
    `min_classes` thư mục con (mỗi thư mục con là một lớp) chứa ảnh.

    Parameters
    ----------
    root : str
        Thư mục gốc chứa dữ liệu đã tải về (ví dụ '../data/raw').
    min_classes : int
        Số thư mục lớp tối thiểu để coi là một thư mục ảnh.

    Returns
    -------
    (img_dirs, csv_files) : tuple[dict[str, Path], list[Path]]
        img_dirs: {tên thư mục: đường dẫn}, csv_files: danh sách file CSV.
    """
    root = Path(root)
    img_dirs = {}
    for d in [root, *[p for p in root.rglob("*") if p.is_dir()]]:
        class_dirs = [
            s for s in d.iterdir()
            if s.is_dir() and any(f.suffix.lower() in IMG_EXTS for f in s.iterdir() if f.is_file())
        ]
        if len(class_dirs) >= min_classes:
            img_dirs[d.name] = d
    csv_files = sorted(root.rglob("*.csv"))

    print("Thư mục ảnh:")
    for name, d in img_dirs.items():
        print(f"  - {name}: {d}")
    print("File CSV:")
    for f in csv_files:
        print(f"  - {f}")
    return img_dirs, csv_files


def build_image_df(img_dir: str) -> pd.DataFrame:
    """
    Quét thư mục ảnh, tạo DataFrame mô tả từng ảnh và in tổng quan.

    Nhãn (label) được lấy từ tên thư mục chứa ảnh.

    Parameters
    ----------
    img_dir : str
        Thư mục ảnh, bên trong chia theo thư mục từng lớp.

    Returns
    -------
    pd.DataFrame
        Gồm các cột: path, label, filename, ext.
    """
    img_dir = Path(img_dir)
    rows = [
        {"path": str(p), "label": p.parent.name, "filename": p.name, "ext": p.suffix.lower()}
        for p in sorted(img_dir.rglob("*"))
        if p.suffix.lower() in IMG_EXTS
    ]
    df = pd.DataFrame(rows)
    print("Dataset shape:", df.shape)
    print(f"Số lớp  : {df['label'].nunique()}")
    print(f"Các lớp : {sorted(df['label'].unique())}")
    print("\nFile extension:")
    print(df["ext"].value_counts())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Kiểm tra chất lượng dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

def add_image_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Đọc từng ảnh để lấy kích thước, mode, dung lượng và đánh dấu ảnh lỗi.

    Ảnh không đọc được sẽ có is_valid = False (các hàm vẽ biểu đồ phía sau
    tự động bỏ qua những ảnh này).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame từ build_image_df.

    Returns
    -------
    pd.DataFrame
        Bản sao của df, thêm các cột: width, height, mode, is_valid,
        aspect_ratio, size_kb.
    """
    props = []
    for p in df["path"]:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                props.append((im.size[0], im.size[1], im.mode, True))
        except Exception:
            props.append((np.nan, np.nan, None, False))

    out = df.copy()
    out[["width", "height", "mode", "is_valid"]] = pd.DataFrame(props, index=df.index)
    out["aspect_ratio"] = out["width"] / out["height"]
    out["size_kb"] = [os.path.getsize(p) / 1024 for p in out["path"]]

    n_bad = (~out["is_valid"]).sum()
    print(f"Ảnh lỗi / không đọc được: {n_bad}")
    if n_bad:
        print(out.loc[~out["is_valid"], "path"].head(10).to_string(index=False))
    print("\nImage mode:")
    print(out["mode"].value_counts(dropna=False))
    return out


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phát hiện ảnh trùng lặp hoàn toàn bằng mã băm MD5.

    Ngoài số ảnh trùng, hàm còn đếm các nhóm ảnh giống hệt nhau nhưng
    mang nhãn khác nhau (dấu hiệu của nhãn sai). Không phát hiện được
    ảnh gần trùng (đã resize hoặc nén lại).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame từ build_image_df.

    Returns
    -------
    pd.DataFrame
        Bản sao của df, thêm cột md5.
    """
    out = df.copy()
    out["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in out["path"]]

    in_group = out.duplicated("md5", keep=False)
    print(f"Ảnh nằm trong nhóm trùng : {in_group.sum()}")
    print(f"Ảnh dư thừa (có thể loại): {out.duplicated('md5').sum()}")

    label_per_hash = out[in_group].groupby("md5")["label"].nunique()
    print(f"Nhóm trùng nhưng khác nhãn: {(label_per_hash > 1).sum()}")

    if in_group.any():
        print("\nẢnh trùng theo lớp:")
        print(out.loc[in_group, "label"].value_counts())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 3. Phân phối các lớp
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
    title: str = "Class distribution"
):
    """
    In thống kê và vẽ biểu đồ phân phối số ảnh theo lớp.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Tên cột nhãn.
    title : str
        Tiêu đề biểu đồ.
    """
    counts = df[label_col].value_counts()
    pct = (counts / len(df) * 100).round(2)
    print("Class counts & percentages:")
    print(pd.DataFrame({"count": counts, "percent (%)": pct}))
    print(f"\nImbalance ratio (max/min): {counts.max() / counts.min():.2f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=counts.values, y=counts.index, color="#4C78A8", ax=ax)
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v}", va="center", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Number of images")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.show()


def compare_class_distribution(dfs: dict):
    """
    So sánh số ảnh mỗi lớp giữa các phiên bản dataset.

    Nếu có đúng hai phiên bản, bảng kết quả có thêm cột diff
    (phiên bản thứ hai trừ phiên bản thứ nhất).

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        {tên phiên bản: DataFrame}, ví dụ {"original": df_orig, "enhanced": df}.
    """
    names = list(dfs)
    table = pd.DataFrame({n: d["label"].value_counts() for n, d in dfs.items()}).fillna(0).astype(int)
    if len(names) == 2:
        table["diff"] = table[names[1]] - table[names[0]]
    print(table)
    print("\nTổng:", {n: int(table[n].sum()) for n in names})

    table[names].plot(kind="bar", figsize=(11, 5), width=0.8)
    plt.title("Images per class: " + " vs ".join(names))
    plt.xlabel("")
    plt.ylabel("Number of images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 4. Phân tích nhãn đã được sửa
# ──────────────────────────────────────────────────────────────────────────────

def plot_correction_matrix(corr: pd.DataFrame, old_col: str, new_col: str):
    """
    Vẽ heatmap số ảnh bị sửa nhãn: nhãn cũ (hàng) sang nhãn mới (cột).

    Parameters
    ----------
    corr : pd.DataFrame
        DataFrame đọc từ file CSV các nhãn đã sửa.
    old_col : str
        Tên cột chứa nhãn cũ.
    new_col : str
        Tên cột chứa nhãn mới.
    """
    ct = pd.crosstab(corr[old_col], corr[new_col])
    print(f"Tổng số ảnh được sửa nhãn: {len(corr)}\n")
    print("Top các cặp sửa nhãn:")
    print(ct.stack().sort_values(ascending=False).loc[lambda s: s > 0].head(10))

    plt.figure(figsize=(9, 7))
    sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
    plt.title("Label corrections (old -> new)")
    plt.ylabel("Old label")
    plt.xlabel("New label")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Trực quan ảnh và kích thước
# ──────────────────────────────────────────────────────────────────────────────

def show_sample_images(df: pd.DataFrame, n_per_class: int = 5, seed: int = 42):
    """
    Hiển thị lưới ảnh ngẫu nhiên, mỗi hàng là một lớp.

    Parameters
    ----------
    df : pd.DataFrame
    n_per_class : int
        Số ảnh hiển thị cho mỗi lớp.
    seed : int
        Random seed để kết quả lặp lại được.
    """
    df = df[df["is_valid"]] if "is_valid" in df else df
    classes = sorted(df["label"].unique())
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 2.2, len(classes) * 2.2))
    for i, c in enumerate(classes):
        sub = df[df["label"] == c]
        picks = sub.sample(min(n_per_class, len(sub)), random_state=seed)["path"].tolist()
        for j in range(n_per_class):
            ax = axes[i, j]
            ax.axis("off")
            if j < len(picks):
                ax.imshow(Image.open(picks[j]).convert("RGB"))
            if j == 0:
                ax.set_title(c, fontsize=10, loc="left")
    plt.tight_layout()
    plt.show()


def plot_image_size_distribution(df: pd.DataFrame):
    """
    Vẽ phân phối width, height, aspect ratio và dung lượng file.

    Cần chạy add_image_properties trước. Giúp chọn kích thước resize
    và cách xử lý tỷ lệ khung hình ở bước preprocessing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame từ add_image_properties.
    """
    d = df[df["is_valid"]]
    print(d[["width", "height", "aspect_ratio", "size_kb"]].describe().round(2))
    print("\nTop 5 kích thước phổ biến nhất (width x height):")
    sizes = d["width"].astype(int).astype(str) + "x" + d["height"].astype(int).astype(str)
    print(sizes.value_counts().head(5))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (col, title) in zip(
        axes.ravel(),
        [("width", "Width (px)"), ("height", "Height (px)"),
         ("aspect_ratio", "Aspect ratio (W/H)"), ("size_kb", "File size (KB)")],
    ):
        sns.histplot(d[col], bins=40, ax=ax, color="#4C78A8")
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 6. Thống kê màu sắc và chất lượng ảnh
# ──────────────────────────────────────────────────────────────────────────────

def compute_image_stats(
    df: pd.DataFrame,
    n_per_class: int = 150,
    size: int = 128,
    seed: int = 42
) -> pd.DataFrame:
    """
    Tính brightness, contrast, sharpness và mean RGB trên mẫu ảnh theo lớp.

    Đồng thời in mean/std theo kênh RGB của toàn bộ mẫu, dùng cho bước
    normalization khi huấn luyện mô hình.

    Parameters
    ----------
    df : pd.DataFrame
    n_per_class : int
        Số ảnh lấy mẫu cho mỗi lớp.
    size : int
        Kích thước resize (size x size) khi tính thống kê.
    seed : int
        Random seed để kết quả lặp lại được.

    Returns
    -------
    pd.DataFrame
        Thống kê theo từng ảnh: label, mean_r, mean_g, mean_b,
        brightness, contrast, sharpness.
    """
    d = df[df["is_valid"]] if "is_valid" in df else df
    sample = pd.concat(
        [g.sample(min(n_per_class, len(g)), random_state=seed) for _, g in d.groupby("label")]
    ).reset_index(drop=True)

    rows = []
    ch_sum = np.zeros(3)
    ch_sq = np.zeros(3)
    n_pix = 0
    for _, r in sample.iterrows():
        arr = np.asarray(Image.open(r["path"]).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        gray = arr.mean(axis=2)
        lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
               + gray[1:-1, :-2] + gray[1:-1, 2:])
        ch_sum += arr.sum(axis=(0, 1))
        ch_sq += (arr ** 2).sum(axis=(0, 1))
        n_pix += size * size
        rows.append({
            "label": r["label"],
            "mean_r": arr[..., 0].mean(),
            "mean_g": arr[..., 1].mean(),
            "mean_b": arr[..., 2].mean(),
            "brightness": gray.mean(),
            "contrast": gray.std(),
            "sharpness": lap.var(),
        })

    mean = ch_sum / n_pix
    std = np.sqrt(ch_sq / n_pix - mean ** 2)
    print(f"Số ảnh dùng để thống kê: {len(sample)}")
    print(f"Mean (R, G, B): {np.round(mean, 4)}")
    print(f"Std  (R, G, B): {np.round(std, 4)}")
    return pd.DataFrame(rows)


def plot_image_stats(stats: pd.DataFrame):
    """
    Vẽ boxplot brightness / sharpness theo lớp và mean RGB trung bình mỗi lớp.

    Parameters
    ----------
    stats : pd.DataFrame
        DataFrame từ compute_image_stats.
    """
    order = sorted(stats["label"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, col in zip(axes, ["brightness", "sharpness"]):
        sns.boxplot(data=stats, x=col, y="label", order=order, color="#4C78A8", fliersize=2, ax=ax)
        ax.set_title(f"{col.capitalize()} by class")
        ax.set_ylabel("")
    plt.tight_layout()
    plt.show()

    rgb = stats.groupby("label")[["mean_r", "mean_g", "mean_b"]].mean().loc[order]
    rgb.plot(kind="bar", figsize=(11, 4), color=["#E45756", "#54A24B", "#4C78A8"], width=0.8)
    plt.title("Mean RGB by class")
    plt.xlabel("")
    plt.ylabel("Mean intensity (0-1)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["R", "G", "B"])
    plt.tight_layout()
    plt.show()