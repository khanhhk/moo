# mtl-moo

Framework học đa nhiệm (MTL) sử dụng MGDA (Multiple Gradient Descent Algorithm) để tìm nghiệm Pareto-optimal trên nhiều tác vụ cùng lúc. Hỗ trợ ba dataset: MultiMNIST, CelebA, Cityscapes.

---

## Cấu trúc

```
mtl-moo/
├── configure.sh                      # Thiết lập môi trường (PYTHONPATH)
├── configs.json                      # Đường dẫn dataset
├── sample.json                       # Ví dụ tham số huấn luyện (CelebA + MGDA)
└── multi_task/
    ├── train_multi_task.py           # Script huấn luyện chính (entry point)
    ├── min_norm_solvers.py           # MGDA solver (PyTorch)
    ├── min_norm_solvers_numpy.py     # MGDA solver (NumPy, không dùng trong train)
    ├── model_selector.py             # Chọn encoder/decoder theo dataset
    ├── datasets.py                   # Dispatcher tải dữ liệu
    ├── losses.py                     # Hàm loss cho từng tác vụ
    ├── metrics.py                    # Metrics: ACC, L1, IOU
    ├── models/
    │   ├── multi_lenet.py            # LeNet cho MultiMNIST
    │   ├── multi_faces_resnet.py     # ResNet18 cho CelebA
    │   ├── segnet.py                 # SegNet decoder (Cityscapes)
    │   ├── pspnet.py                 # PSPNet encoder (Cityscapes)
    │   └── resnet_mit.py             # ResNet50/101 dilated (backbone)
    └── loaders/
        ├── multi_mnist_loader.py     # Loader MultiMNIST
        ├── celeba_loader.py          # Loader CelebA (40 thuộc tính)
        └── cityscapes_loader.py      # Loader Cityscapes (S, I, D)
```

---

## Thuật toán: MGDA

Trong MTL thông thường, gradient của các tác vụ thường xung đột nhau. MGDA giải quyết bằng cách tìm tổ hợp tuyến tính của các gradient sao cho chuẩn nhỏ nhất:

```
min ||Σ cᵢ·gᵢ||₂²    s.t.  Σ cᵢ = 1,  cᵢ ∈ [0,1]
```

**MGDA_UB** (mặc định) — xấp xỉ hiệu quả hơn:
- Chỉ tính gradient theo representation (output encoder), không theo toàn bộ tham số encoder
- Kết quả là upper bound của MGDA đầy đủ, nhưng nhanh hơn nhiều

**Chuẩn hóa gradient** (`normalization_type`):
| Giá trị | Ý nghĩa |
|---|---|
| `none` | Không chuẩn hóa |
| `l2` | Chia theo chuẩn L2 |
| `loss` | Chia theo giá trị loss |
| `loss+` | Chia theo `loss × L2` (dùng trong sample.json) |

---

## Dataset

Sửa đường dẫn thực trong `configs.json` trước khi chạy:

| Dataset | Tác vụ | Kích thước ảnh | Ghi chú |
|---|---|---|---|
| **MultiMNIST** | L, R (phân loại 2 chữ số) | 36×36 | File: `multi_mnist.pickle` |
| **CelebA** | 40 thuộc tính khuôn mặt (binary) | 64×64 | Cần `img_align_celeba_png/`, `list_attr_celeba.txt` |
| **Cityscapes** | S (segmentation), I (instance), D (depth) | 256×512 | Cần `leftImg8bit/`, `gtFine/`, `disparity/` |

---

## Cài đặt

```bash
pip install torch torchvision numpy tensorboardX tqdm scipy pillow click
```

---

## Cách chạy

### 1. Thiết lập môi trường

```bash
cd /home/khanhhk/moo/mtl-moo
source configure.sh
```

### 2. Cập nhật `configs.json`

```json
{
    "mnist": {
        "path": "/đường/dẫn/đến/multimnist",
        "all_tasks": ["L", "R"]
    },
    "celeba": {
        "path": "/đường/dẫn/đến/celeba",
        "all_tasks": ["0", "1", "2", "..."],
        "img_rows": 64, "img_cols": 64
    },
    "cityscapes": {
        "path": "/đường/dẫn/đến/cityscapes",
        "all_tasks": ["S", "I", "D"],
        "img_rows": 256, "img_cols": 512
    }
}
```

### 3. Tạo file params.json

```json
{
    "optimizer": "Adam",
    "batch_size": 256,
    "lr": 0.001,
    "dataset": "mnist",
    "tasks": ["L", "R"],
    "algorithm": "mgda",
    "normalization_type": "loss+",
    "use_approximation": true
}
```

Ví dụ đầy đủ cho CelebA (40 tác vụ): xem `sample.json`.

### 4. Chạy huấn luyện

```bash
python -m multi_task.train_multi_task --param_file params.json
```

### 5. Theo dõi kết quả (TensorBoard)

```bash
tensorboard --logdir runs/
```

---

## Đầu ra

| Loại | Vị trí | Nội dung |
|---|---|---|
| Model checkpoint | `saved_models/{exp_id}_{epoch}_model.pkl` | Lưu mỗi 3 epoch |
| TensorBoard log | `runs/{exp_id}_{timestamp}/` | Loss, accuracy, IOU theo epoch |

---

## Lưu ý

- Yêu cầu GPU (CUDA) — code gọi `.cuda()` trực tiếp.
- Tạo thư mục `saved_models/` trước khi chạy nếu chưa có.
- `cityscapes_loader.py` cần file `depth_mean.npy` trong thư mục chạy.
- Code dùng API cũ của PyTorch (`Variable`) — nếu lỗi, cần nâng cấp sang `.detach()`.
