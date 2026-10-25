# Biến động loại rừng trong các khu bảo tồn Việt Nam (ForTy V2 Beta 2017–2025)

Notebook `notebooks/forty_pa_timeseries.ipynb` lấy các khu bảo tồn của Việt Nam từ WDPA và làm các việc sau cho từng khu:

- vẽ bản đồ loại rừng năm 2017 và 2025 (ảnh tĩnh), kèm bản đồ thay đổi 2017 → 2025 tính ở 10 m;
- tính diện tích từng loại rừng cho mỗi năm 2017–2025 và vẽ biểu đồ đường (line chart): diện tích (ha) và thay đổi so với 2017 (ha);
- nhờ Gemini viết đoạn nhận xét ngắn về biến động.

Kết quả hiển thị trong notebook (10 khu đầu tiên) và được xuất ra báo cáo HTML đầy đủ `outputs/bao_cao_ForTy_PA_VN.html`.

## Dữ liệu

- **ForTy V2 Beta:** `projects/nature-trace/assets/forest_typology/v2_beta/<năm>`. Mỗi năm là một ImageCollection 10 m chia ô, có 7 band uint8 là xác suất từng lớp (0–250). Lớp = argmax + 1:

  | Lớp | Band | Tên |
  |---|---|---|
  | 1 | B0 | Primary forest: mature |
  | 2 | B1 | Naturally regenerating forest |
  | 3 | B2 | Planted forest |
  | 4 | B3 | Plantation forest |
  | 5 | B4 | Tree crops and agroforestry |
  | 6 | B5 | Other land (không phải rừng) |
  | 7 | B6 | Primary forest: regrowing |

  V2 Beta là bản thử nghiệm, thứ tự band và định nghĩa lớp có thể thay đổi.

  Tài liệu tham khảo:
  - Neumann, M., Raichuk, A., Potapov, P. et al. Global forest typology at 10-meter resolution for forest and land-use monitoring. *Sci Data* 13, 1434 (2026). https://doi.org/10.1038/s41597-026-08299-5
  - Google DeepMind. [Beyond tree cover: mapping the world’s forest types](https://deepmind.google/blog/beyond-tree-cover-mapping-the-worlds-forest-types/).
- **Khu bảo tồn:** `WCMC/WDPA/current/polygons`. Bảng này dùng schema mới: `SITE_ID` (trước là `WDPAID`), `REALM` (trước là `MARINE`).
  - Mặc định lấy: `ISO3 = VNM`, `STATUS` là Designated hoặc Inscribed, bỏ khu thuần biển, phần trên đất liền ≥ 1 km².
  - Khu quốc gia và khu quốc tế (UNESCO-MAB, Ramsar, Di sản) đều được tính riêng, nên một vùng rừng có thể thuộc nhiều khu. Bảng `outputs/chong_lan.csv` liệt kê các cặp khu giao nhau. Để loại bớt khu, sửa `PA_FILTER` hoặc `SITE_IDS` trong ô cấu hình.

## Phương pháp

- **Diện tích:** mỗi khu một request cho cả 9 năm (`frequencyHistogram` trên 9 band lớp). Phép tính chạy trong hệ UTM của khu ở 10 m, nên mỗi pixel đúng 100 m². Pixel ở mép ranh giới được tính theo phần nằm trong khu. Nếu một khu quá lớn làm 10 m bị lỗi, notebook tự chuyển sang 20 m rồi 30 m và ghi chú mức đã dùng trong báo cáo.
- **Bản đồ:** `getThumbURL` (EPSG:3857). Ảnh hiển thị lớp rừng bên trong ranh giới khu, lớp 6 (Đất khác) tô xám nhạt.
- **Bản đồ thay đổi 2017 → 2025:** so lớp của từng pixel 10 m giữa hai năm, gộp thành 4 loại (`gee.CHANGES`):
  - rừng tự nhiên → rừng trồng / cây công nghiệp;
  - rừng / cây → đất khác;
  - tăng rừng hoặc phục hồi (đất khác → rừng, rừng trồng → tự nhiên);
  - đổi lớp trong cùng nhóm.

  Diện tích từng cặp "lớp cũ → lớp mới" tính ở 10 m (`outputs/chuyen_doi_lop_2017_2025.csv`). Ảnh cũng tính từ 10 m rồi gộp dần theo bậc 4×4 xuống cỡ điểm ảnh của hình: màu là loại thay đổi chiếm ưu thế, độ đậm là tỷ lệ pixel 10 m đổi lớp. Ảnh được tải theo từng ô bằng `ee.data.computePixels`, vì một request cho cả khu lớn ở 10 m vượt giới hạn bộ nhớ của GEE.
  - "Đổi lớp" là thay đổi gộp (cả hai chiều), thường lớn hơn nhiều so với thay đổi ròng trên biểu đồ đường. Ví dụ ở Tây Nghệ An: khoảng 6% diện tích đổi lớp, nhưng thay đổi ròng chỉ khoảng 1%. Phần lớn là vùng dao động qua lại giữa rừng và đất khác.
- **Nhận xét AI:**
  - Model: mặc định `"auto"`, tức model `gemini-X.Y-flash` cao nhất mà key dùng được (lúc viết là `gemini-3.8-flash`).
  - Gemini chỉ nhận bảng số liệu đã tính và được yêu cầu không dùng thông tin ngoài bảng.
  - Câu trả lời lưu trong `outputs/llm_cache/` và `outputs/nhan_xet_AI.csv`.
  - **Mặc định notebook không gọi Gemini:** ô đầu mục 7 có `RUN_GEMINI = False`, khi đó notebook đọc nhận xét đã lưu trong `outputs/nhan_xet_AI.csv`. Muốn tạo hoặc tạo lại nhận xét thì sửa tay thành `True`.

Mọi kết quả trung gian (diện tích trong `outputs/areas/`, ảnh trong `outputs/thumbs/`, nhận xét AI) đều được lưu lại. Khi chạy lại, notebook chỉ tính những phần còn thiếu. Xóa thư mục tương ứng để tính lại từ đầu.

## Cài đặt và chạy

```bash
uv venv .venv --python 3.12
```

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

- **Earth Engine:** đã đăng nhập sẵn (dùng chung credential với dự án Vegetation-Monitoring, project `ee-hoangvietanh`).
- **Gemini:** biến môi trường `GEMINI_API_KEY`. Không ghi key vào notebook.

Mở notebook, chọn kernel `.venv`, rồi chạy tất cả các ô. Để chạy thử nhanh, đặt `PA_LIMIT = 3` hoặc `SITE_IDS = [...]` trong ô cấu hình.

## Đầu ra (`outputs/`)

| File | Nội dung |
|---|---|
| `bao_cao_ForTy_PA_VN.html` | Báo cáo đầy đủ (ảnh nhúng sẵn, một file duy nhất) |
| `khu_bao_ton.csv` | Danh sách khu đã chọn |
| `dien_tich_loai_rung.csv` | Diện tích (ha): khu × năm × loại rừng (bảng rộng) |
| `dien_tich_loai_rung_dai.csv` | Như trên, dạng dài |
| `chong_lan.csv` | Các cặp khu giao nhau |
| `chuyen_doi_lop_2017_2025.csv` | Diện tích (ha) từng cặp lớp 2017 → 2025 của mỗi khu, kèm loại thay đổi |
| `nhan_xet_AI.csv` | Nhận xét Gemini của từng khu |
| `maps/`, `charts/` | Bản đồ 2017/2025, bản đồ thay đổi và biểu đồ của từng khu (PNG) |
