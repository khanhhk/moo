# Google Drive Downloader

Download folder từ Google Drive.

## Cài đặt

```bash
pip install gdown
```

## Cách dùng

```bash
python download_gdrive.py <folder_url> [--output <path>]
```

| Tham số | Mô tả |
|---|---|
| `url` | Google Drive folder share link |
| `--output`, `-o` | Thư mục lưu (mặc định: tên folder gốc) |

## Ví dụ

```bash
# Download vào thư mục hiện tại (tên folder gốc)
python download_gdrive.py "https://drive.google.com/drive/folders/1eaArbVRlCG8cxWcncYbP_oCGqpiN08gt?usp=sharing"

# Chỉ định thư mục output
ppython download_gdrive.py "https://drive.google.com/drive/folders/1eaArbVRlCG8cxWcncYbP_oCGqpiN08gt?usp=sharing" --output data/
```
