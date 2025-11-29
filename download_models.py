import os
from huggingface_hub import snapshot_download, list_repo_files

# Tên Space của bạn
REPO_ID = "kinly26/kltn_fastapi_demo"

print(f"Checking repository: {REPO_ID}...")

try:
    # List files in the repo to verify existence
    repo_files = list_repo_files(repo_id=REPO_ID, repo_type="space")
    print(f"Files in repo {REPO_ID}:")
    for f in repo_files:
        print(f" - {f}")

    print(f"Downloading models from Space: {REPO_ID}...")
    
    # Tải thư mục models từ Space về thư mục hiện tại
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="space",
        local_dir=".",
        allow_patterns=["models/*"], # Chỉ tải thư mục models
        ignore_patterns=["*.git*"],
        local_dir_use_symlinks=False # Tải file thực, không dùng symlink
    )
    print("Download completed successfully!")
    
    # Kiểm tra lại file
    models_dir = "models"
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        print(f"Files in local models directory: {files}")
        for f in files:
            p = os.path.join(models_dir, f)
            print(f" - {f}: {os.path.getsize(p) / 1024 / 1024:.2f} MB")
    else:
        print("Error: models directory not found after download.")

except Exception as e:
    print(f"Error downloading models: {e}")
    # Không raise error để build process không bị dừng (có thể file đã được copy từ COPY . .)
    # Nhưng nếu COPY thất bại (do LFS pointer), thì script này là cứu cánh.
