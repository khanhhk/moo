# Thư mục `minh/`

Phần mã nguồn chính của đề tài — nghiên cứu tối ưu đa mục tiêu (MOO) và học đa nhiệm (MTL) theo hướng Pareto MTL với learning rate thích nghi (Armijo line search).

---

## Cấu trúc

```
minh/
├── MOP/                              # Bài toán tối ưu đa mục tiêu tổng hợp
│   ├── VD_1.ipynb                    # 2D, có ràng buộc, so sánh Pareto MTL vs MOO-MTL
│   ├── VD_2.ipynb                    # 20D, hàm mũ, phân tích hội tụ
│   ├── VD_4.ipynb                    # 2D, hàm bậc hai lồi
│   ├── VD_5_3_chieu.ipynb            # 3 mục tiêu, 3 biến, không ràng buộc
│   ├── VD_5_3_chieu_co_rang_buoc.ipynb  # 3 mục tiêu, có ràng buộc hộp
│   └── VD_5_3_chieu_cons_ver2.ipynb  # Biến thể khác của bài toán 3D
└── MTL/                              # Học đa nhiệm trên dữ liệu thực
    ├── drug-review-proposed.ipynb    # Phân loại đánh giá thuốc (2 tác vụ, TextCNN)
    ├── Pareto_multiMNIST.ipynb       # Phân loại chữ số MultiMNIST (LeNet / ResNet18)
    └── Drug_Result.ipynb             # Phân tích và vẽ kết quả thí nghiệm thuốc
```

---

## Thuật toán cốt lõi

**MinNormSolver** — tìm tổ hợp tuyến tính tối ưu của các gradient:
- Giải bài toán QP: `min ||Σ c_i g_i||²`
- Trường hợp 2D: có lời giải giải tích; n>2: dùng projected gradient descent

**ParetoMTL (phiên bản đề xuất)** — mở rộng với learning rate thích nghi:
- Sinh các preference vector phân bố đều trên đường tròn/mặt cầu
- Với mỗi preference: cập nhật gradient có trọng số theo MinNormSolver
- Điều chỉnh bước theo điều kiện Armijo: `loss_new ≤ loss_old + σ × energy`
- Bước giảm theo hệ số κ (0.85–0.99) khi điều kiện bị vi phạm

---

## Cài đặt môi trường

```bash
# Cho MOP (tối ưu tổng hợp)
pip install numpy scipy matplotlib plotly

# Cho MTL (học đa nhiệm)
pip install torch torchvision
pip install nltk contractions  # cho drug-review
```

---

## Cách chạy

Tất cả các notebook chạy theo luồng từ trên xuống dưới (`Run All`).

### MOP — Chạy từ đơn giản đến phức tạp

| Notebook | Bài toán | Ghi chú |
|---|---|---|
| `VD_1.ipynb` | 2D, 2 mục tiêu, có ràng buộc | So sánh Pareto MTL vs MOO-MTL |
| `VD_4.ipynb` | 2D, 2 mục tiêu, lồi | Benchmark bậc hai |
| `VD_2.ipynb` | 20D, 2 mục tiêu | Phân tích hội tụ theo hypervolume |
| `VD_5_3_chieu.ipynb` | 3D, 3 mục tiêu | Trực quan hóa 3D bằng plotly |
| `VD_5_3_chieu_co_rang_buoc.ipynb` | 3D + ràng buộc | Ràng buộc hộp và cầu |
| `VD_5_3_chieu_cons_ver2.ipynb` | 3D variant | Bước khởi tạo 1.5, κ=0.9 |

Không cần dữ liệu ngoài — tất cả bài toán MOP đều tự sinh.

### MTL — Yêu cầu dữ liệu

**1. MultiMNIST** (`Pareto_multiMNIST.ipynb`)

Cần file pickle dữ liệu (xem hướng dẫn tải ở `../README.md`):
```
multi_mnist.pickle       # 120K train / 20K test, ảnh 36x36
```

Chọn model ở đầu notebook:
```python
model_name = "lenet"    # hoặc "resnet"
```

**2. Drug Review** (`drug-review-proposed.ipynb`)

Cần:
- Dataset drug review từ Kaggle (`drugsComTrain_raw.tsv`, `drugsComTest_raw.tsv`)
- GloVe embeddings 840B 300d (`glove.840B.300d.txt`)

Sửa đường dẫn trong notebook trước khi chạy.

**3. Xem kết quả** (`Drug_Result.ipynb`)

Chạy sau `drug-review-proposed.ipynb` để vẽ biểu đồ và tính hypervolume.

---

## Đầu ra

| File/Thư mục | Nội dung |
|---|---|
| `*.txt` (trong notebook) | Log loss/accuracy từng epoch theo preference vector |
| `*.pkl` / `*.pickle` | Model weights đã lưu |
| Plot inline | Pareto front, loss curve, learning rate schedule |

---

## Lưu ý

- Các notebook MTL có hardcode đường dẫn dữ liệu — cần sửa trước khi chạy.
- `VD_5_3_chieu.ipynb` (5.2 MB) nặng vì chứa output 3D plotly — mở chậm là bình thường.
- `Drug_Result.ipynb` phụ thuộc vào kết quả của `drug-review-proposed.ipynb`.
