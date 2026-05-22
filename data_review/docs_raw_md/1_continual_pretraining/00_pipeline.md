# 00 — Continual Pre-Training Pipeline

## Mục tiêu

Thực hiện Continual Pre-Training (CPT) trên tập văn bản pháp luật Việt Nam nhằm bổ sung domain knowledge pháp lý vào một SLM nền tảng (base model), mà không fine-tune cho tác vụ cụ thể nào.

---

## Tổng quan luồng dữ liệu

```
HuggingFace Dataset
  VLSP2025-LegalSML/legal-pretrain
          │
          ▼
  [scripts/clone.py]
  Download parquet shards
  → continual_training/data/raw/data/
          │
          ▼
  [scripts/view.py]
  Thống kê nhanh: rows, columns, token estimate
  → continual_training/report/data_view.md
          │
          ▼
  [scripts/clean.py]          [scripts/clean_v2.py]
  Dedup → Normalize           Dedup → HTML→Markdown
  → Filter → Format           (markitdown) → Filter → Format
  → processed/train.parquet   → processed/train_v2.parquet
          │                           │
          └──────────┬────────────────┘
                     ▼
  CPT Training
  (script chưa implement — xem TODO)
```

---

## Cấu trúc thư mục

```
continual_training/
├── data/
│   ├── raw/
│   │   └── data/          # parquet shards gốc từ HuggingFace
│   └── processed/
│       └── train.parquet  # output sau clean, sẵn sàng train
├── docs/                  # tài liệu (bạn đang đọc)
├── report/
│   └── data_view.md       # báo cáo thống kê raw data
└── scripts/
    ├── clone.py           # download data từ HuggingFace
    ├── view.py            # generate report thống kê
    ├── clean.py           # cleaning pipeline (regex normalize)
    └── clean_v2.py        # cleaning pipeline (markitdown HTML→Markdown)
```

---

## Scripts

### `clone.py` — Download data

Download 2 shard đầu (train-00000, train-00001) từ HuggingFace dataset `VLSP2025-LegalSML/legal-pretrain`.

```bash
python continual_training/scripts/clone.py
```

Cấu hình trong file:
- `REPO_ID` — HuggingFace dataset repo
- `PARQUET_FILES` — danh sách shard cần download

---

### `view.py` — Thống kê raw data

Đọc toàn bộ parquet trong `raw/data/`, in thống kê ra console và ghi report ra `report/data_view.md`.

```bash
python continual_training/scripts/view.py
```

Output bao gồm: số rows, columns, metadata keys, word/token estimate.

---

### `clean.py` — Cleaning pipeline (v1, regex)

Pipeline 6 bước trên raw parquet, output ra `processed/train.parquet`.

```bash
python continual_training/scripts/clean.py
```

#### Các bước xử lý

| Bước | Tên | Mô tả |
|------|-----|--------|
| 1 | Load | Đọc tất cả `.parquet` trong `raw/data/` |
| 2 | Dedup | Loại trùng theo `Id` → `DocIdentity` → content prefix |
| 3 | Normalize | Unescape HTML, strip tags bằng regex, NFC unicode, collapse whitespace |
| 4 | Filter | Lọc theo token length, blank line ratio, non-VN char ratio |
| 5 | Format | Ghép header metadata vào đầu mỗi document |
| 6 | Save | Ghi columns `text`, `metadata` ra `processed/train.parquet` |

---

### `clean_v2.py` — Cleaning pipeline (v2, markitdown)

Giống v1 nhưng thay bước 3 bằng **markitdown** để convert HTML → Markdown thay vì chỉ strip tags. Phù hợp hơn vì `doc_content` là HTML thực sự (chứa `<table>`, `<span style=...>`, `<br/>`, ...).

```bash
# Cài dependency trước
pip install markitdown[all]

python continual_training/scripts/clean_v2.py
```

#### So sánh v1 vs v2

| | `clean.py` (v1) | `clean_v2.py` (v2) |
|---|---|---|
| Bước normalize | Regex strip tags | markitdown HTML→Markdown |
| Giữ cấu trúc (bảng, đề mục) | Không | Có (Markdown tables, headings) |
| Tốc độ | Nhanh | Chậm hơn (~500 doc/batch) |
| Dependency thêm | Không | `markitdown[all]` |
| Output file | `train.parquet` | `train_v2.parquet` |

#### Các bước xử lý

| Bước | Tên | Mô tả |
|------|-----|--------|
| 1 | Load | Đọc tất cả `.parquet` trong `raw/data/` |
| 2 | Dedup | Loại trùng theo `Id` → `DocIdentity` → content prefix |
| 3 | HTML→Markdown | Convert HTML → Markdown qua markitdown, fallback regex nếu lỗi |
| 4 | Filter | Lọc theo token length, blank line ratio, non-VN char ratio |
| 5 | Format | Ghép header metadata vào đầu mỗi document |
| 6 | Save | Ghi columns `text`, `metadata` ra `processed/train_v2.parquet` |

#### Thresholds mặc định (dùng chung cả v1 và v2)

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| `MIN_TOKENS` | 100 | Bỏ doc lỗi scrape / tiêu đề rỗng |
| `MAX_TOKENS` | 50,000 | Bỏ doc bị concat nhầm hoặc bất thường |
| `MAX_BLANK_LINE_RATIO` | 0.6 | Bỏ doc toàn dòng trống |
| `MAX_NON_VN_RATIO` | 0.3 | Bỏ doc chứa quá nhiều ký tự lạ |

#### Format output — column `text`

```
NGHỊ ĐỊNH 12/2021/NĐ-CP | Chính phủ | 2021-02-18
<nội dung văn bản dạng Markdown...>
```

Header giúp model học được metadata của văn bản pháp lý trong quá trình CPT.

---

## Thống kê dataset (2 shards)

| Metric | Giá trị |
|--------|---------|
| Shards đã download | 2 / 24 |
| Tổng số văn bản | 8,066 |
| Avg tokens / văn bản | ~11,413 |
| Tổng tokens ước tính | ~92M |

> Dataset gốc đầy đủ (24 shards) ước tính ~1.1B tokens.
> Dự án này chỉ dùng 2 shards (~92M tokens) cho CPT thử nghiệm.

---

## TODO

- [ ] Viết script training CPT (Unsloth / HuggingFace Trainer)
- [ ] Đánh giá perplexity trên held-out legal text sau CPT
- [ ] So sánh model trước / sau CPT trên legal QA benchmark
