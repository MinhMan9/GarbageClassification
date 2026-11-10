# Phân loại rác thải gia dụng từ ảnh

Bài tập lớn môn CO3135 Lập trình cho AI và Khoa học dữ liệu.

**Bài toán:** phân loại ảnh rác gia dụng vào 12 nhóm (paper, cardboard, biological, metal, plastic, green-glass, brown-glass, white-glass, clothes, shoes, batteries, trash).
**Dataset:** Garbage Classification (Kaggle, Mostafa Abla).

**Cách làm:** trích đặc trưng từ ảnh (truyền thống: HOG, SIFT + Bag of Visual Words; học sâu đóng băng: ResNet, EfficientNet, ViT), lưu thành `.npy`, rồi huấn luyện Logistic Regression (baseline), SVM, Random Forest trên các đặc trưng đó (5 đặc trưng × 3 mô hình = 15 tổ hợp). Có tuning siêu tham số và ablation.

## Cài đặt

### Cách 1: Sử dụng `uv` (Khuyên dùng - tốc độ cài cực nhanh)

```bash
# Tạo môi trường ảo
uv venv

# Kích hoạt môi trường ảo:
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Cài đặt thư viện
uv pip install -r requirements.txt
```

### Cách 2: Sử dụng `pip` truyền thống

```bash
python -m venv .venv
source .venv/bin/activate  # Trên Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu trúc thư mục

```
garbage-classification/
├── README.md
├── requirements.txt
├── config.yaml                        # seed, đường dẫn raw_dir, kích thước resize (không tự ý sửa)
├── .gitignore
│
├── data/                              # KHÔNG commit
│   ├── raw/                           # ảnh gốc tải từ Kaggle (mỗi lớp một thư mục)
│   └── manifest.csv                   # SINH RA bởi 02_data_prep
├── splits.json                        # SINH RA bởi 02_data_prep (dev / final test / 5 fold, commit được)
│
├── modules/                           # hàm dùng chung
│   ├── data_prep.py                   # quét ảnh, kiểm tra hỏng, bỏ trùng, chia tập, nạp manifest/splits
│   ├── eda.py                         # hàm phục vụ EDA
│   ├── extract.py                     # trích đặc trưng, lưu/nạp .npy
│   ├── classify.py                    # mô hình LR / SVM / RF, tuning
│   └── evaluation.py                  # CV 5 fold, macro-F1 mean ± std, biểu đồ
│
├── notebooks/                         # đánh số theo thứ tự chạy
│   ├── 01_eda.ipynb                   # EDA trên ảnh gốc
│   ├── 02_data_prep.ipynb             # làm sạch, bỏ trùng, chia tập
│   ├── 03_extract_traditional.ipynb   # HOG, SIFT + BoVW
│   ├── 04_extract_deep.ipynb          # ResNet, EfficientNet, ViT
│   ├── 05_classify_traditional.ipynb  # LR, SVM, RF trên HOG, SIFT
│   ├── 06_classify_deep.ipynb         # LR, SVM, RF trên 3 embedding
│   └── 07_final_evaluation.ipynb      # tổng hợp, final test, ablation
│
└── artifacts/                         # KHÔNG commit
    ├── features/                      # đặc trưng .npy
    ├── results/                       # file kết quả .csv
    └── figures/                       # biểu đồ
```
