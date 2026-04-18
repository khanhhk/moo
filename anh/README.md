# Thư mục `anh/`

Phần mã nguồn chính của đề tài — nghiên cứu tối ưu đa mục tiêu (MOO) và học đa nhiệm (MTL) theo hướng Min-Max/Pareto.

---

## Cấu trúc

```
anh/
├── configs.json
├── process.ipynb                        # Notebook hậu xử lý kết quả
├── Toy_example/
│   ├── VD1.py, VD1_1.py, VD2.py – VD5.py   # Bài toán tối ưu tổng hợp
│   ├── test.py                          # Kiểm tra chuẩn hóa hàm
│   ├── images/                          # Ảnh kết quả
│   └── constraint/minmax/               # Phiên bản có ràng buộc
│       ├── VD1.py, VD1_1.py, VD2.py, VD3.py
│       └── VD1.ipynb
└── Pareto-MTL/
    ├── synthetic_example/               # Toy không cần dữ liệu — chạy thử nhanh
    │   ├── run_synthetic_example.py
    │   └── min_norm_solvers_numpy.py
    ├── multiMNIST/                      # Pareto-MTL cơ sở (NeurIPS 2019)
    │   ├── train.py
    │   ├── model_lenet.py, model_resnet.py
    │   ├── min_norm_solvers.py
    │   └── data/                        # Đặt file pickle dữ liệu vào đây
    └── multiMNIST_MinMax/               # Các biến thể MinMax do tác giả phát triển
        ├── train.py
        ├── train_ver1.py, train_ver2.py
        ├── train_non_monotone_1.py, train_non_monotone_2.py
        ├── model_lenet.py, model_resnet.py
        ├── min_norm_solvers.py
        └── data/                        # Đặt file pickle dữ liệu vào đây
```

---

## Thuật toán

### Toy_example

Mỗi file `VD*.py` thực hiện tối ưu Min-Max có tham chiếu:

1. Định nghĩa hàm mục tiêu và `F(x, ref) = max_i f(x, i, ref)`
2. Xác định tập chỉ số active `J_delta(x, ref)`
3. Giải bài toán con tìm hướng di chuyển bằng `scipy.optimize.minimize`
4. Sinh preference vector bằng `pymoo`, chạy lặp theo từng vector
5. Vẽ Pareto front kết quả

| File | Biến | Mục tiêu |
|---|---|---|
| `VD1.py`, `VD1_1.py` | 2 | 2 |
| `VD2.py` | 2 | 2 |
| `VD3.py`, `VD4.py` | nhiều | 2 |
| `VD5.py` | 3 | 6 |

### multiMNIST_MinMax

Các biến thể phát triển từ Pareto-MTL cơ sở:

| File | Mô tả |
|---|---|
| `train.py` | MinMax với vòng `while` điều chỉnh bước thích nghi |
| `train_ver1.py` | Xử lý rõ theo reference vector |
| `train_ver2.py` | Phiên bản 2 cập nhật theo reference |
| `train_non_monotone_1.py` | Non-monotone variant 1 |
| `train_non_monotone_2.py` | Non-monotone variant 2 |

---

## Dữ liệu

Đặt các file pickle vào đúng thư mục `data/` (xem hướng dẫn tải ở `../../README.md`):

```
Pareto-MTL/multiMNIST/data/multi_mnist.pickle
Pareto-MTL/multiMNIST/data/multi_fashion.pickle
Pareto-MTL/multiMNIST/data/multi_fashion_and_mnist.pickle
```

Tương tự cho `multiMNIST_MinMax/data/`.

Các file `train*.py` đang hardcode đường dẫn cũ — cần sửa trước khi chạy:

| Cần sửa | Đường dẫn cũ |
|---|---|
| Đọc dữ liệu | `/home/ubuntu/workspace/dataset/DANC/` |
| Ghi log | `/home/ubuntu/workspace/DANC/Pareto-MTL/` |
| Lưu model | `/home/ubuntu/workspace/dataset/DANC/Result/` |

---

## Cài đặt

```bash
# Toy_example
pip install numpy scipy matplotlib autograd pymoo pytictoc

# Pareto-MTL
pip install torch torchvision
```

---

## Cách chạy

### 1. Toy example (không cần dữ liệu)

```bash
cd anh/Toy_example
python VD1.py
```

Kết quả ảnh lưu vào `images/`.

### 2. Synthetic example (không cần dữ liệu ảnh)

```bash
cd anh/Pareto-MTL/synthetic_example
python run_synthetic_example.py
```

### 3. Thí nghiệm đầy đủ

```bash
# Bản cơ sở
cd anh/Pareto-MTL/multiMNIST
python train.py

# Các biến thể MinMax
cd anh/Pareto-MTL/multiMNIST_MinMax
python train.py
python train_ver1.py
python train_ver2.py
python train_non_monotone_1.py
python train_non_monotone_2.py
```

---

## Lưu ý

- Yêu cầu GPU để chạy thí nghiệm MTL.
- `synthetic_example/` là điểm bắt đầu tốt nhất nếu chưa có dữ liệu ảnh.
- Sửa `plt.savefig(...)` trong Toy_example sang đường dẫn tương đối trước khi chạy.
