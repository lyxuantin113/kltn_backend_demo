"""
Script để xóa các file tạm cũ trong thư mục temp_videos
Chạy: python cleanup_temp.py
"""

import os
import time
from pathlib import Path

# Thư mục temp trong project
temp_dir = Path(__file__).parent / "temp_videos"

if not temp_dir.exists():
    print(f"Thư mục {temp_dir} không tồn tại. Không có file nào để xóa.")
    exit(0)

# Xóa tất cả file trong thư mục temp
deleted_count = 0
deleted_size = 0

for file_path in temp_dir.iterdir():
    if file_path.is_file():
        try:
            size = file_path.stat().st_size
            file_path.unlink()
            deleted_count += 1
            deleted_size += size
            print(f"Đã xóa: {file_path.name} ({size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"Không thể xóa {file_path.name}: {e}")

print(f"\nTổng kết:")
print(f"  - Số file đã xóa: {deleted_count}")
print(f"  - Dung lượng giải phóng: {deleted_size / 1024 / 1024:.2f} MB")

