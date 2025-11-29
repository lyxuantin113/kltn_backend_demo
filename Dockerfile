# Sử dụng Python 3.10
FROM python:3.10-slim

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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

# Expose port 7860 (Port mặc định của Hugging Face Spaces)
EXPOSE 7860

# Lệnh chạy ứng dụng
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
