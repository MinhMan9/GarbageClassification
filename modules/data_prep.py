"""
modules/data_prep.py
====================
Data Preparation — chuẩn bị dữ liệu cho bài toán phân loại ảnh
Garbage Classification (12 classes, bản ENHANCED).

Module này được dùng cho notebook 02_data_prep.ipynb. Thứ tự các bước:
loại ảnh trùng -> chia development / final test -> thống kê chỉ học từ
tập train (class weight) -> transform -> DataLoader -> lưu kết quả.

Cách dùng
---------
    import modules.data_prep as prep

    df = prep.remove_duplicates(df)
    train, val, test = prep.split_dataset(df, test_size=0.2, val_size=0.2)
    prep.check_split(train, val, test)
    class_weights = prep.compute_class_weights(train, label2id)
    train_tf, eval_tf = prep.get_transforms(img_size=224)
    loaders = prep.make_loaders(train, val, test, label2id, train_tf, eval_tf)
    prep.show_batch(loaders['train'], id2label)
    prep.save_artifacts(train, val, test, label2id, class_weights, config, out_dir)
"""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Thống kê chuẩn hóa của ImageNet (dùng khi fine-tune mô hình pretrained)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Làm sạch dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại ảnh trùng lặp hoàn toàn (theo mã băm MD5 của file).

    - Các bản trùng cùng nhãn: giữ lại một bản.
    - Các bản trùng nhưng khác nhãn: không biết nhãn nào đúng nên bỏ cả nhóm.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame từ eda.build_image_df (cần cột 'path' và 'label').

    Returns
    -------
    pd.DataFrame
        DataFrame đã làm sạch, có thêm cột md5.
    """
    out = df.copy()
    out["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in out["path"]]

    g = out.groupby("md5")["label"].agg(n_label="nunique", n_img="size")
    conflict = g.index[g["n_label"] > 1]
    clean = out[~out["md5"].isin(conflict)].drop_duplicates("md5").reset_index(drop=True)

    print(f"Nhóm ảnh trùng: {int((g['n_img'] > 1).sum())} | trong đó khác nhãn: {len(conflict)}")
    print(f"Giữ {len(clean)} ảnh; bỏ {len(out) - len(clean)} ảnh.")
    return clean


# ──────────────────────────────────────────────────────────────────────────────
# 2. Chia dữ liệu
# ──────────────────────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    label_col: str = "label",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42
):
    """
    Chia dữ liệu thành development (train + val) và final test, stratified theo lớp.

    Final test được cất riêng, chỉ dùng ở cuối để chấm mô hình đã chọn.
    Development được chia tiếp thành train (học) và val (chọn mô hình, chỉnh
    siêu tham số).

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Tên cột nhãn dùng để stratify.
    test_size : float
        Tỷ lệ final test so với toàn bộ dữ liệu.
    val_size : float
        Tỷ lệ val so với development.
    seed : int
        Random seed để kết quả lặp lại được.

    Returns
    -------
    (train, val, test) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    dev, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=seed)
    train, val = train_test_split(dev, test_size=val_size, stratify=dev[label_col], random_state=seed)
    train, val, test = (d.reset_index(drop=True) for d in (train, val, test))

    print(f"Development: {len(dev)} (train {len(train)} + val {len(val)}) | Final test: {len(test)}")
    return train, val, test


def _img_size(path: str) -> str:
    """Đọc kích thước ảnh dạng 'WxH' (chỉ đọc header nên nhanh)."""
    with Image.open(path) as im:
        return f"{im.width}x{im.height}"


def check_split(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "label"
):
    """
    Kiểm tra việc chia: phân phối lớp, rò rỉ ảnh trùng và phân bố kích thước ảnh.

    Parameters
    ----------
    train, val, test : pd.DataFrame
        Kết quả của split_dataset (cần cột 'md5' từ remove_duplicates).
    label_col : str
        Tên cột nhãn.
    """
    parts = {"train": train, "val": val, "test": test}

    print("=== Phân phối lớp (%) ===")
    print(pd.DataFrame({n: d[label_col].value_counts(normalize=True) for n, d in parts.items()})
          .mul(100).round(2))

    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert set(parts[a]["md5"]).isdisjoint(parts[b]["md5"]), f"Có ảnh trùng giữa {a} và {b}"
    print("\nKhông có ảnh trùng giữa các tập.")

    sizes = {n: d["path"].map(_img_size) for n, d in parts.items()}
    top5 = sizes["train"].value_counts().head(5).index
    print("\n=== Top 5 kích thước phổ biến nhất (%) ===")
    print(pd.DataFrame({n: s.value_counts(normalize=True).reindex(top5) for n, s in sizes.items()})
          .rename_axis("size").mul(100).round(2))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Xử lý mất cân bằng lớp
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(train: pd.DataFrame, label2id: dict, label_col: str = "label") -> dict:
    """
    Tính class weight (nghịch đảo tần suất) chỉ từ tập train.

    weight_c = N / (số_lớp * n_c): lớp càng ít ảnh thì weight càng lớn.
    Dùng làm tham số `weight` của hàm loss khi huấn luyện.

    Parameters
    ----------
    train : pd.DataFrame
        Tập train (không dùng val/test để tránh rò rỉ thông tin).
    label2id : dict
        Ánh xạ tên lớp -> chỉ số.
    label_col : str
        Tên cột nhãn.

    Returns
    -------
    dict[int, float]
        {chỉ số lớp: weight}.
    """
    counts = train[label_col].value_counts()
    weights = len(train) / (len(label2id) * counts)
    out = {label2id[c]: float(weights[c]) for c in label2id}

    print(pd.DataFrame({"count": counts, "weight": weights.round(3)}).loc[list(label2id)])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. Transform, Dataset và DataLoader
# ──────────────────────────────────────────────────────────────────────────────

def load_rgb(path: str) -> Image.Image:
    """
    Đọc ảnh và chuyển về RGB.

    Ảnh RGBA/P (có thể có nền trong suốt) được ghép lên nền trắng trước khi
    bỏ kênh alpha, tránh vùng trong suốt bị chuyển thành màu đen.
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def get_transforms(img_size: int = 224, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Tạo transform cho tập train (có augmentation) và cho val/test (không augmentation).

    Các lựa chọn dựa trên EDA:
    - Không kéo giãn hoặc pad ảnh: dùng RandomResizedCrop (train) và
      Resize cạnh ngắn + CenterCrop (val/test) để không để lộ nguồn ảnh qua kích thước.
    - Chỉ đổi độ sáng/tương phản, không đổi hue/saturation vì ba lớp thủy tinh
      phân biệt bằng màu.
    - Chỉ lật ngang và xoay nhẹ.

    Parameters
    ----------
    img_size : int
        Kích thước ảnh đầu vào của mô hình.
    mean, std : tuple
        Thống kê chuẩn hóa theo kênh RGB.

    Returns
    -------
    (train_tf, eval_tf) : tuple[T.Compose, T.Compose]
    """
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_tf, eval_tf


class GarbageDataset(Dataset):
    """
    Dataset đọc ảnh từ đường dẫn trong DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Cần cột 'path' và cột nhãn.
    label2id : dict
        Ánh xạ tên lớp -> chỉ số.
    transform : callable, optional
        Transform áp dụng lên ảnh PIL.
    label_col : str
        Tên cột nhãn.
    """

    def __init__(self, df: pd.DataFrame, label2id: dict, transform=None, label_col: str = "label"):
        self.paths = df["path"].tolist()
        self.labels = df[label_col].map(label2id).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def make_loaders(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label2id: dict,
    train_tf,
    eval_tf,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42
) -> dict:
    """
    Tạo DataLoader cho train, val và test.

    Chỉ train được shuffle và augmentation; val/test giữ thứ tự cố định.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    label2id : dict
    train_tf, eval_tf : callable
        Transform từ get_transforms.
    batch_size : int
    num_workers : int
        Số tiến trình đọc ảnh. Mặc định 0 để chạy được trên mọi hệ điều hành;
        tăng lên 2-4 khi chạy trên Linux/Colab để đọc nhanh hơn.
    seed : int
        Random seed cho việc shuffle tập train.

    Returns
    -------
    dict[str, DataLoader]
        Khóa: 'train', 'val', 'test'.
    """
    g = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(GarbageDataset(train, label2id, train_tf), batch_size=batch_size,
                            shuffle=True, num_workers=num_workers, generator=g),
        "val": DataLoader(GarbageDataset(val, label2id, eval_tf), batch_size=batch_size,
                          shuffle=False, num_workers=num_workers),
        "test": DataLoader(GarbageDataset(test, label2id, eval_tf), batch_size=batch_size,
                           shuffle=False, num_workers=num_workers),
    }
    print({k: f"{len(v.dataset)} ảnh, {len(v)} batch" for k, v in loaders.items()})
    return loaders


def show_batch(loader, id2label: dict, mean=IMAGENET_MEAN, std=IMAGENET_STD, n: int = 8):
    """
    In shape của một batch và hiển thị một số ảnh (đã bỏ chuẩn hóa) kèm nhãn.

    Parameters
    ----------
    loader : DataLoader
    id2label : dict
        Ánh xạ chỉ số -> tên lớp.
    mean, std : tuple
        Thống kê chuẩn hóa đã dùng, để khôi phục ảnh khi hiển thị.
    n : int
        Số ảnh hiển thị.
    """
    images, labels = next(iter(loader))
    print("Batch images:", tuple(images.shape), "| labels:", tuple(labels.shape))

    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    shown = (images[:n] * std_t + mean_t).clamp(0, 1)

    fig, axes = plt.subplots(1, len(shown), figsize=(len(shown) * 2, 2.4))
    for ax, img, y in zip(axes, shown, labels[:n]):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(id2label[int(y)], fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Lưu kết quả
# ──────────────────────────────────────────────────────────────────────────────

def save_artifacts(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label2id: dict,
    class_weights: dict,
    config: dict,
    out_dir: str
):
    """
    Lưu kết quả chuẩn bị dữ liệu để các notebook huấn luyện dùng lại.

    Gồm: meta.csv (đường dẫn, nhãn và cột split = train/val/test),
    label2id.json, class_weights.json và config.json.

    Parameters
    ----------
    train, val, test : pd.DataFrame
    label2id : dict
    class_weights : dict
    config : dict
        Cấu hình đã dùng (seed, kích thước ảnh, mean/std, tỷ lệ chia).
    out_dir : str
        Thư mục lưu kết quả.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.concat(
        [train.assign(split="train"), val.assign(split="val"), test.assign(split="test")],
        ignore_index=True,
    )
    meta.to_csv(out / "meta.csv", index=False)
    (out / "label2id.json").write_text(json.dumps(label2id, indent=2, ensure_ascii=False))
    (out / "class_weights.json").write_text(json.dumps({str(k): v for k, v in class_weights.items()}, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print("Đã lưu:", sorted(p.name for p in out.iterdir()))