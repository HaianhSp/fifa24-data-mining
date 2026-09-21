# FIFA24 Player Data Mining

Đồ án môn **Khai phá dữ liệu** — phân tích bộ dữ liệu chỉ số cầu thủ FIFA24 bằng các kỹ thuật Classification và Clustering.

## Mục tiêu

Từ các chỉ số kỹ năng của cầu thủ (tốc độ, dứt điểm, chuyền bóng, phòng ngự...), dự án giải quyết 2 bài toán:

1. **Phân loại nhóm giá trị chuyển nhượng** (Value-Tier Classification) — dự đoán một cầu thủ thuộc nhóm giá trị nào (Thấp / Trung bình / Cao / Ngôi sao) dựa trên chỉ số kỹ năng.
2. **Phân cụm phong cách thi đấu** (Playstyle Clustering) — khám phá các nhóm phong cách tự nhiên tồn tại trong dữ liệu (Tấn công / Phòng ngự / Trung tuyến).

> **Lưu ý quan trọng**: Dataset gốc không có cột vị trí thi đấu (LB, RB, ST...), nên dự án không phân loại vị trí cầu thủ theo tên gọi cụ thể. Toàn bộ nhãn dùng trong dự án đều xuất phát khách quan từ chính dữ liệu (quantile trên `value`, hoặc cụm tự phát hiện qua GMM) — không tự đặt ngưỡng chủ quan theo cảm tính bóng đá.

## Cấu trúc thư mục

```
fifa24-data-mining/
├── data/
│   └── player_stats.csv          # Dữ liệu gốc: 5682 cầu thủ, 41 chỉ số
├── notebooks/
│   ├── 1_preprocessing.ipynb          # Đọc, làm sạch, chuẩn hóa dữ liệu
│   ├── 2_kmeans_clustering.ipynb      # Phân cụm K-Means (bản đầu, cứng)
│   └── 3_value_tier_classification.ipynb  # Classification + GMM (bản chính)
├── src/
├── .gitignore
└── README.md
```

## Dữ liệu

- **Nguồn**: `player_stats.csv` — chỉ số cầu thủ FIFA24 dạng scout thực tế (tốc độ, kỹ thuật, thể chất...).
- **Kích thước**: 5682 cầu thủ × 41 cột.
- **Vấn đề dữ liệu đã xử lý**:
  - Cột `marking` bị khuyết 100% → loại bỏ.
  - File bị lỗi mã hóa hỗn hợp (một số dòng lưu UTF-8, một số dòng lưu Latin-1) → xử lý bằng cách decode từng dòng, fallback UTF-8 → Latin-1.
  - Cột `value` ở dạng chuỗi (`"$153.500.000"`) → chuyển về số thực (`value_numeric`).
  - Một số dòng bị trùng lặp hoàn toàn → cần `drop_duplicates()` trước khi phân tích.

## Phương pháp

### Hướng A — Value-Tier Classification (có giám sát)

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Nhãn | `pd.qcut(value, q=4)` → Thấp/Trung bình/Cao/Ngôi sao | Chia khách quan theo phân vị, không tự đặt ngưỡng |
| Feature | Toàn bộ chỉ số kỹ năng + thể chất (36 cột) | Loại bỏ định danh và chính cột tạo nhãn (tránh data leakage) |
| Mô hình | `RandomForestClassifier(n_estimators=300)` | Không cần chuẩn hóa, xử lý tốt feature tương quan, diễn giải được qua `feature_importances_` |
| Đánh giá | Accuracy, F1, Confusion Matrix | Nhãn thật nên số liệu đáng tin cậy |

**Kết quả**: Accuracy 83%, F1 macro 0.83. Nhóm "Cao" dễ phân biệt nhất (F1 0.90); nhóm "Thấp" và "Ngôi sao" hay bị nhầm hơn (F1 ~0.77-0.79).

### Hướng C — Playstyle Clustering (không giám sát)

| Bước | Mô tả |
|---|---|
| 1. K-Means (baseline) | K=4 chọn theo cảm tính (notebook 2) — dùng để so sánh |
| 2. GMM + chọn K bằng Silhouette Score | Phát hiện K=2 luôn thắng vì ranh giới mạnh nhất là **thủ môn vs. cầu thủ ngoài sân** |
| 3. Loại thủ môn, phân cụm lại | Trên 5057 cầu thủ ngoài sân, chọn K=3 làm điểm cân bằng (Silhouette ~0.27) |
| 4. Diễn giải cụm | Dựa trên centroid → Tấn công / Phòng ngự / Trung tuyến-Đa năng |

**Kết quả**: 3 nhóm phong cách với xác suất mềm theo từng cụm (`predict_proba`), kiểm chứng qua các cầu thủ nổi tiếng (Mbappé/Ronaldo → Tấn công, Van Dijk → Phòng ngự, Sergio Ramos → Trung tuyến/Đa năng).

> Tên gọi cụm là diễn giải thủ công dựa trên centroid — số cụm (K) và ranh giới cụm mới là phần do dữ liệu quyết định.

## Cách chạy

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Chạy tuần tự từng notebook trong `notebooks/`:

```bash
jupyter notebook notebooks/1_preprocessing.ipynb
jupyter notebook notebooks/2_kmeans_clustering.ipynb
jupyter notebook notebooks/3_value_tier_classification.ipynb
```

> Mỗi notebook tự đọc lại `../data/player_stats.csv` một cách độc lập — không cần chạy theo thứ tự file để có dữ liệu, nhưng trong từng file cần **Restart Kernel → Run All** để tránh lỗi `KeyError` do chạy cell không theo thứ tự.

## Kỹ thuật khai phá dữ liệu áp dụng

- **Data Cleaning**: xử lý cột khuyết, sửa lỗi mã hóa, loại trùng lặp.
- **Data Transformation**: Min-Max Scaling, rời rạc hóa bằng quantile.
- **Classification**: Random Forest (multi-class).
- **Clustering**: K-Means, Gaussian Mixture Model (soft clustering), chọn K bằng Silhouette Score.

## Tác giả

Lê Nguyễn Hải Anh — Phenikaa University
