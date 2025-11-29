# Sử dụng Python 3.10
FROM python:3.10-slim

# Force rebuild to pick up new apt packages
RUN echo "Cache bust 2025-11-29"

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Download Models Layer ---
# Download models vào một thư mục cache riêng để không bị overwrite bởi COPY .
WORKDIR /model_cache
COPY download_models.py .
# Cài đặt huggingface_hub để chạy script download
RUN pip install huggingface_hub
RUN python download_models.py

# --- Application Layer ---
# Thiết lập thư mục làm việc
WORKDIR /app

# Copy requirements và cài đặt dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo user non-root để chạy app (Yêu cầu của Hugging Face Spaces)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Thiết lập biến môi trường cho thư mục làm việc của user
WORKDIR $HOME/app

# Copy toàn bộ source code vào thư mục của user
COPY --chown=user . $HOME/app

# Copy models từ cache vào thư mục app
# Cần chuyển về root để copy từ /model_cache (do root tạo) sang $HOME/app (của user)
USER root
RUN mkdir -p $HOME/app/models && \
    cp -r /model_cache/models/. $HOME/app/models/ && \
    chown -R user:user $HOME/app/models

# Chuyển lại về user
USER user

# Expose port 7860 (Port mặc định của Hugging Face Spaces)
EXPOSE 7860

# Lệnh chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
