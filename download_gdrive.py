"""
Download folder từ Google Drive với thanh tiến trình.

Usage:
    python download_gdrive.py <folder_url> [--output <path>]

Example:
    python download_gdrive.py "https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing"
    python download_gdrive.py "https://drive.google.com/drive/folders/FOLDER_ID" --output data/

Requires:
    pip install gdown
"""

import argparse
import gdown


def main():
    parser = argparse.ArgumentParser(description="Download folder từ Google Drive")
    parser.add_argument("url", help="Google Drive folder URL")
    parser.add_argument("--output", "-o", default=None, help="Thư mục lưu (mặc định: tên folder gốc)")
    args = parser.parse_args()

    gdown.download_folder(args.url, output=args.output, quiet=False)


if __name__ == "__main__":
    main()
