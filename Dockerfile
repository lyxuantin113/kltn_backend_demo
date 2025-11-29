# Sử dụng Python 3.10
FROM python:3.10-slim

# Thêm cache bust để ép build lại mới nhất
RUN echo "Cache bust check 2: 2025-11-29"

# Cài đặt thư viện hệ thống
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- GIAI ĐOẠN 1: DOWNLOAD MODEL ---
WORKDIR /model_cache
COPY download_models.py .
RUN pip install huggingface_hub

# Chạy script download
RUN python download_models.py

# [DEBUG QUAN TRỌNG] In ra xem script đã tải cái gì về và nằm ở đâu
RUN echo "=== KIỂM TRA THƯ MỤC /model_cache ===" && ls -R /model_cache

# --- GIAI ĐOẠN 2: SETUP APP ---
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo user (Bắt buộc cho HF Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy code
COPY --chown=user . $HOME/app

# --- GIAI ĐOẠN 3: COPY MODEL VÀO APP ---
USER root

# Tạo thư mục đích
RUN mkdir -p $HOME/app/models

# [FIX LỖI COPY]
# Logic: Copy tất cả file .pt và .pth từ /model_cache (bất kể nằm trong subfolder hay root) vào đích
# Dùng lệnh find để tìm và copy cho chắc ăn, tránh bị sai đường dẫn folder
RUN find /model_cache -name "*.pt" -exec cp {} $HOME/app/models/ \; && \
    find /model_cache -name "*.pth" -exec cp {} $HOME/app/models/ \;

# Phân quyền lại cho user
RUN chown -R user:user $HOME/app/models

# [DEBUG QUAN TRỌNG] In ra xem cuối cùng thư mục app/models có gì không
RUN echo "=== KIỂM TRA THƯ MỤC FINAL $HOME/app/models ===" && ls -R $HOME/app/models

USER user

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]