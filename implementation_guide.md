# Lộ trình Triển khai: Multi-Level Mask Denoiser (Satellite Version)

Tài liệu này được thiết kế để giúp bạn triển khai ý tưởng "Nhiễu đa cấp dựa trên hình ảnh" vào SegRefiner một cách dễ hiểu và có hệ thống nhất.

---

## 1. Hiểu "Cốt lõi" của ý tưởng
Thay vì tạo ra những lỗi ngẫu nhiên như SegRefiner gốc, chúng ta sẽ tạo ra những **lỗi "thông minh"** giống hệt lỗi mà các mô hình AI thường mắc phải khi nhìn ảnh vệ tinh:
- **Lỗi thiếu tòa nhà:** Những tòa nhà nhỏ, mờ sẽ bị xóa đi.
- **Lỗi biên không sắc nét:** Dựa vào độ tương phản của ảnh RGB để làm nhòe biên.

---

## 2. Các thành phần cần xây dựng
Chúng ta sẽ chia công việc thành 3 phần chính:

| Tên gọi | Nhiệm vụ | Logic triển khai |
| :--- | :--- | :--- |
| **GMM Weak Classifier** | Đo độ "khó" của ảnh | Chạy GMM trên pixel RGB -> Tính Entropy -> Ra bản đồ độ bất định $U_{score}$. |
| **Object Processor** | Xử lý từng tòa nhà | Tìm các vùng liên thông -> Tính độ mờ trung bình -> Quyết định xóa hay giữ. |
| **Edge Detector (Kornia)** | Tìm biên thực tế | Dùng thư viện **Kornia** để chạy Canny trực tiếp trên GPU từ ảnh RGB. |

---

## 3. Cấu hình Dataloader (OpenEarthMap)
Vì SegRefiner sử dụng MMDetection, bạn cần cấu hình bộ nạp dữ liệu để đọc được OpenEarthMap.

**Thông tin Dataset:**
- Đường dẫn: `/home/ubuntu/vy/Denoiser/OpenEarthMap_wo_xBD`
- Cấu trúc: `city/images/` và `city/labels/`
- Phân tách (Splits): `train.txt`, `val.txt`, `test.txt`
- **Quy mô dữ liệu:** 
    - Tập **Train**: ~3000 ảnh.
    - Tập **Val**: ~500 ảnh.

**Các bước thực hiện:**
1. **Tạo Dataset Class:** Đăng ký một module mới `OpenEarthMapDataset` trong `mmdet/datasets/`.
2. **Định nghĩa Pipeline:** Trong file cấu hình (config), thiết lập pipeline sử dụng `LoadAnnotations` và một class mới (hoặc cập nhật) là `LoadCoarseMasks`.

---

## 4. Chỉnh sửa Pipeline (`loading.py`)
Mục tiêu là thay thế cách tạo nhiễu cũ của SegRefiner bằng phương pháp đa cấp.

**Vị trí:** `mmdet/datasets/pipelines/loading.py`
- Tìm class `LoadCoarseMasks`.
- Hàm quan trọng: `modify_boundary`. Đây là nơi SegRefiner gốc tạo ra nhiễu từ mặt nạ chuẩn.
- **Nhiệm vụ:** Bạn cần cập nhật logic tại đây để nhận thêm ảnh RGB và áp dụng logic GMM + Kornia Canny để tạo ra mặt nạ nhiễu thông minh thay vì chỉ dựa vào hình khối mặt nạ.

---

## 5. Thiết lập Thí nghiệm (Ablation Study)
Để tái lập hoàn chỉnh bảng **Table III**, bạn cần chuẩn bị **8 file cấu hình (config)** khác nhau tương ứng với 8 tổ hợp nhiễu.

| STT | File Config | Obj | Bnd | Unc | Mô tả |
| :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | `exp1_base.py` | False | False | False | SegRefiner gốc |
| 2 | `exp2_obj.py` | **True** | False | False | Chỉ dùng nhiễu Đối tượng |
| 3 | `exp3_bnd.py` | False | **True** | False | Chỉ dùng nhiễu Biên RGB |
| 4 | `exp4_unc.py` | False | False | **True** | Chỉ dùng nhiễu GMM |
| ... | ... | ... | ... | ... | ... |
| 8 | `exp8_all.py` | **True** | **True** | **True** | **Phương pháp đầy đủ** |

---

## 6. Các thông số kỹ thuật quan trọng (Hyperparameters)
Để đạt kết quả như bài báo, hãy tuân thủ các thông số sau:

- **GMM:** Thiết lập `n_components=2`. Một cụm đại diện cho vật thể, một cụm cho nền.
- **Canny:** Ngưỡng thấp (low_threshold) khoảng 0.1, ngưỡng cao (high_threshold) khoảng 0.2 (Áp dụng cho **ảnh RGB** đã chuẩn hóa 0-1 để tìm biên thực tế).
- **Object Noise:** Tòa nhà nào có độ bất định trung bình $U_{score} > 0.5$ sẽ có xác suất bị xóa cao hơn.
- **Inference Steps:** Giữ nguyên **6 bước** như mặc định của SegRefiner.
- **Training:** Huấn luyện từ đầu (Train from scratch) để tối ưu hoàn toàn cho ảnh vệ tinh.

---

## 7. Lộ trình Triển khai (Step-by-Step)

### Pha 1: Xây dựng "Kho vũ khí" (`noise_utils.py`)
Tạo file tại `mmdet/models/utils/noise_utils.py` chứa:
1.  `compute_gmm_uncertainty(img)`: Trả về bản đồ độ bất định.
2.  `extract_building_instances(mask)`: Trả về danh sách các tòa nhà.
3.  `get_rgb_edges_gpu(img)`: Sử dụng **Kornia** để lấy biên từ ảnh.

### Pha 2: Chỉnh sửa Model chính (`segrefiner_base.py`)
- Cập nhật hàm `q_sample` để nhận tham số từ `noise_components`.
- Cập nhật `forward_train` để truyền ảnh RGB vào quá trình lấy mẫu.

---

## 8. Giải chuyên sâu về Cơ chế Suy luận (Inference)
Cơ chế "đi lùi" ($x_t \to x_{t-1}$) là quá trình tích lũy niềm tin qua 3 bước: Dự đoán $x_0 \to$ Tính xác suất lật bài $p_{c \to f} \to$ Cập nhật $P_{fine}$.

---

## 9. Cấu hình Validation và Bảng kết quả (IoU & Pseudo IoU)

Để theo dõi quá trình huấn luyện và lưu log giống như yêu cầu, bạn cần thực hiện các thiết lập sau:

### A. Thiết lập trong file Config
Trong các file config (`exp1.py`, `exp2.py`,...), hãy thêm/sửa đoạn code sau:

```python
# Tần suất chạy validation và lưu log
evaluation = dict(interval=5000, metric='mIoU') # Chạy val mỗi 5000 steps

log_config = dict(
    interval=50, # In log training mỗi 50 steps
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False)
    ])
```

### B. Định dạng bảng Log mong muốn
Bảng kết quả khi chạy Validation cần hiển thị rõ quy mô dữ liệu và các cột IoU.

**Log Header mẫu:**
```text
2026-04-28 - INFO - Training set: 3000 images
2026-04-28 - INFO - Validation set: 500 images
```

**Bảng kết quả:**
| Class | IoU | Pseudo IoU |
| :--- | :---: | :---: |
| background | 93.71 | 92.68 |
| building | 73.03 | 69.14 |
| **Summary** | **mIoU: 83.37** | **mIoU: 80.91** |

---

## 10. Danh sách kiểm tra (Checklist)
- [ ] Cài đặt `scikit-learn` (cho GMM).
- [ ] Cài đặt `kornia` (cho Canny trên GPU).
- [ ] Chuẩn bị đủ 8 file config ứng với Table III.
- [ ] Thiết lập `evaluation` interval trong config (mặc định 5000).
- [ ] Đảm bảo Log hiển thị số lượng ảnh Train (3000) và Val (500).
- [ ] Kiểm tra các cờ hiệu `use_obj`, `use_bnd`, `use_unc` trong từng file config.
- [ ] Đảm bảo hàm `p_sample` thực hiện đúng logic tích lũy $P_{fine}$.
