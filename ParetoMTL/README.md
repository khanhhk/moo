# ParetoMTL

Cài đặt chính thức của **Pareto Multi-Task Learning** (NeurIPS 2019) — thuật toán tìm tập nghiệm Pareto-optimal cho học đa nhiệm bằng cách sử dụng preference vector để kiểm soát sự đánh đổi giữa các tác vụ.

> Lin et al., *Pareto Multi-Task Learning*, NeurIPS 2019.

---

## Cấu trúc

```
ParetoMTL/
├── synthetic_example/
│   ├── run_synthetic_example.py     # Ví dụ toy 2D, không cần dữ liệu
│   └── min_norm_solvers_numpy.py    # QP solver (NumPy)
└── multiMNIST/
    ├── train.py                     # Script huấn luyện chính
    ├── model_lenet.py               # LeNet (ảnh 36×36, 2 tác vụ)
    ├── model_resnet.py              # ResNet18 (2 tác vụ)
    ├── min_norm_solvers.py          # QP solver (PyTorch)
    └── data/                        # Thư mục chứa file pickle dữ liệu
```

---

## Thuật toán

**Pareto MTL** mở rộng MOO-MTL bằng cách thêm preference vector để kiểm soát vị trí nghiệm trên Pareto front:

1. **Sinh preference vectors**: phân bố đều trên cung 1/4 đường tròn `[0, π/2]`
2. **Phase khởi tạo** (~2 epoch): tìm nghiệm khả thi ban đầu theo từng preference
3. **Phase tối ưu chính** (niter epoch): descent dọc theo Pareto front với trọng số tính từ QP

**QP solver** — tìm tổ hợp tuyến tính tối ưu của gradient:
```
min ||Σ cᵢ·gᵢ||₂²    s.t.  Σ cᵢ = 1,  cᵢ ≥ 0
```

**Điểm khác biệt so với MOO-MTL**: Pareto MTL bổ sung gradient của các ràng buộc active (các tác vụ đang vi phạm preference) vào bài toán QP, đảm bảo nghiệm di chuyển đúng hướng preference.

---

## Cài đặt

```bash
# Synthetic example
pip install autograd matplotlib numpy

# MultiMNIST
pip install torch torchvision numpy
```

---

## Cách chạy

### 1. Synthetic example (không cần dữ liệu)

```bash
cd /home/khanhhk/moo/ParetoMTL/synthetic_example
python run_synthetic_example.py
```

Kết quả: cửa sổ matplotlib hiển thị 10 nghiệm (đỏ) so với Pareto front chuẩn (xanh).

---

### 2. MultiMNIST

**Yêu cầu**: file pickle dữ liệu trong thư mục `data/` (xem hướng dẫn tải ở `../README.md`):

```
multiMNIST/data/multi_mnist.pickle
multiMNIST/data/multi_fashion.pickle
multiMNIST/data/multi_fashion_and_mnist.pickle
```

**Tạo thư mục lưu model:**
```bash
mkdir -p /home/khanhhk/moo/ParetoMTL/multiMNIST/saved_model
```

**Chạy:**
```bash
cd /home/khanhhk/moo/ParetoMTL/multiMNIST
python train.py
```

**Thay đổi cấu hình** — sửa dòng cuối `train.py`:

```python
# Mặc định
run(dataset='mnist', base_model='lenet', niter=100, npref=5)

# Các tùy chọn khác
run(dataset='fashion',           base_model='lenet',   niter=100, npref=5)
run(dataset='fashion_and_mnist', base_model='lenet',   niter=100, npref=5)
run(dataset='mnist',             base_model='resnet18', niter=20,  npref=5)
```

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| `dataset` | `mnist`, `fashion`, `fashion_and_mnist` | Bộ dữ liệu |
| `base_model` | `lenet`, `resnet18` | Kiến trúc mô hình |
| `niter` | int | Số epoch phase chính |
| `npref` | int | Số preference vector (số nghiệm Pareto) |

---

## Đầu ra

| Loại | Vị trí | Nội dung |
|---|---|---|
| Model checkpoint | `saved_model/{dataset}_{model}_niter_{N}_npref_{P}_prefidx_{I}.pickle` | 1 file / preference vector |
| Console log | stdout | Mỗi 2 epoch: `weights=, train_loss=, train_acc=` |

Ví dụ tên file: `mnist_lenet_niter_100_npref_5_prefidx_0.pickle`

---

## Thời gian chạy ước tính

| Cấu hình | 1 preference vector | Tổng (npref=5) |
|---|---|---|
| LeNet + MNIST | ~10–20 phút | ~1–2 giờ |
| ResNet18 + MNIST | ~30–60 phút | ~3–5 giờ |

---

## Lưu ý

- Yêu cầu GPU (code gọi `.cuda()` cứng).
- Thư mục `saved_model/` phải tồn tại trước khi chạy.
- `Variable` (PyTorch cũ) vẫn hoạt động trên PyTorch ≥ 1.0 nhưng có thể xuất hiện warning.
