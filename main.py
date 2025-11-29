# main.py
import os
import io
import json
from typing import List, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from torchvision import models, transforms
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
import torchvision.transforms.functional as TF


# ============================================================
# 0) CONFIG CHUNG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Task A (distraction) -----
_script_dir = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(_script_dir, "models")

# Cấu hình Task A (Phát hiện hành vi tài xế)
TASK_A_CONFIGS = {
    "ResNet18": {
        "path": os.path.join(MODELS_DIR, "dd_resnet18_unique_subjects.pt"),
        "type": "classification"
    },
    "YOLOv8": {
        "path": os.path.join(MODELS_DIR, "driver_distraction_best_yolov8.pt"),
        "type": "detection"
    },
    "YOLOv11": {
        "path": os.path.join(MODELS_DIR, "driver_distraction_best_yolov11.pt"),
        "type": "detection"
    }
}

# Danh sách class cho ResNet18 (State Farm Dataset - 10 classes)
TASK_A_CLASSES_RESNET = [f"c{i}" for i in range(10)]
TASK_A_SAFE_CLASSES_RESNET = ["c0"]

# Danh sách class cho YOLO (Driver Behavior Dataset - 12 classes)
TASK_A_CLASSES_YOLO = [
    "Safe Driving", "Texting", "Talking on the phone", "Operating the Radio",
    "Drinking", "Reaching Behind", "Hair and Makeup", "Talking to Passenger",
    "Eyes Closed", "Yawning", "Nodding Off", "Eyes Open"
]
TASK_A_SAFE_CLASSES_YOLO = ["Safe Driving", "Eyes Open"]

# Preprocess cho ResNet18 (ảnh)
preprocess_img_A = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ----- Task B (attack) – đa model -----
# 3 mô hình khác nhau: custom_cabin_attack, rwf2000, hockey_fight
TASK_B_CONFIGS = {
    "custom_cabin_attack": {
        "model_path": "models/r21_bilstm_custom.pth",
        "thr_path":   "models/r21_bilstm_custom_best_threshold.json",
        "classes":    ("attack", "non_attack"),
        "pos_class":  "attack",
        "T":          24,  # số frame/clip khi train
    },
    "rwf2000": {
        "model_path": "models/r21_bilstm_rwf.pth",
        "thr_path":   "models/r21_bilstm_rwf_best_threshold.json",
        "classes":    ("Fight", "NonFight"),
        "pos_class":  "Fight",
        "T":          40,  # số frame/clip khi train
    },
    "hockey_fight": {
        "model_path": "models/r21_bilstm_hockey.pth",
        "thr_path":   "models/r21_bilstm_hockey_best_threshold.json",
        "classes":    ("fight", "nonfight"),
        "pos_class":  "fight",
        "T":          16,  # số frame/clip khi train
    },
}

# Các tham số dùng chung cho mọi model Task B:
SIZE_B = 224
CHANNELS_LAST_3D = True

# ---- Tham số cho mô phỏng realtime video ----
# Stage 1: quét thưa
STAGE1_STRIDE = 16   # detect mỗi 16 frame

# Stage 2: quét dày
STAGE2_STRIDE = 4   # detect mỗi 4 frame

# Cửa sổ xét nghi vấn
SUSPICIOUS_WINDOW = 5        # trong 5 lần detect gần nhất
SUSPICIOUS_RATIO = 0.5       # > 50% nghi vấn -> cảnh báo

# Threshold cho Task A (Distraction) - chỉ coi là suspicious nếu prob đủ cao
TASK_A_SUSPICIOUS_PROB_THR = 0.7  # Chỉ coi là suspicious nếu prob của predicted class >= 0.6
TASK_A_DISTRACTION_RATIO = 0.6    # Tỷ lệ nghi vấn cần thiết để trigger Distraction Detected (cao hơn Task B)

# Cooldown cho ATTACK_DETECTED (để có thể trigger lại sau một khoảng thời gian)
ATTACK_COOLDOWN_SECONDS = 2.0  # Khoảng cách tối thiểu giữa các lần trigger (giây)
# Cooldown cho Distraction Detected (để không cảnh báo liên tục)
DISTRACTION_COOLDOWN_SECONDS = 3.0

# Threshold nghi vấn cho Task B
# Tab 3: dùng best_thr * SUSPICIOUS_THR_RATIO (linh hoạt theo từng model)
SUSPICIOUS_THR_RATIO = 0.8   # Hệ số nhân với best_thr để tính threshold nghi vấn


# ============================================================
# 1) MODEL TASK A (RESNET18 & YOLO)
# ============================================================
def load_model_A(model_name="ResNet18"):
    if model_name not in TASK_A_CONFIGS:
        print(f"[WARN] Model {model_name} không tồn tại, fallback ResNet18")
        model_name = "ResNet18"
    
    config = TASK_A_CONFIGS[model_name]
    model_path = config["path"]
    model_type = config["type"]

    if not os.path.exists(model_path):
        print(f"[ERR] Không tìm thấy model tại {model_path}")
        return None, None

    if model_type == "classification":
        try:
            model = models.resnet18(weights=None)
            # ResNet18 output layer size = 10 (cho dataset cũ)
            model.fc = nn.Linear(model.fc.in_features, 10) 
            checkpoint = torch.load(model_path, map_location=DEVICE)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model = checkpoint
            model.to(DEVICE)
            model.eval()
            return model, "classification"
        except Exception as e:
            print(f"[ERR] Load ResNet failed: {e}")
            return None, None

    elif model_type == "detection":
        try:
            model = YOLO(model_path)
            return model, "detection"
        except Exception as e:
            print(f"[ERR] Load YOLO failed: {e}")
            return None, None
    return None, None

def process_frame_taskA_resnet(model, frame_rgb):
    try:
        pil_img = Image.fromarray(frame_rgb)
        input_tensor = preprocess_img_A(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            prob_max, preds = torch.max(probs, 1)
        
        idx = preds.item()
        prob = prob_max.item()
        
        # Sử dụng danh sách class của ResNet
        label = TASK_A_CLASSES_RESNET[idx]
        
        # Logic Suspicious cho ResNet
        # Nếu label KHÔNG nằm trong SAFE list VÀ prob > threshold
        is_suspicious = (label not in TASK_A_SAFE_CLASSES_RESNET) and (prob > TASK_A_SUSPICIOUS_PROB_THR)
        
        return {
            "pred_label": label,
            "suspicious": is_suspicious,
            "probs": [{"label": l, "score": 0.0} for l in TASK_A_CLASSES_RESNET] 
        }
    except Exception as e:
        print(f"ResNet Error: {e}")
        return {"pred_label": "Error", "suspicious": False, "probs": []}

def process_frame_taskA_yolo(model, frame_rgb):
    # YOLO predict
    results = model.predict(frame_rgb, verbose=False, conf=0.4)
    result = results[0]
    
    detected_labels = []
    is_suspicious = False
    
    if result.boxes:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            # Lấy tên class từ model YOLO (nó tự chứa names)
            label = result.names[cls_id]
            detected_labels.append(label)
            
            # Kiểm tra với danh sách SAFE của YOLO
            if label not in TASK_A_SAFE_CLASSES_YOLO:
                is_suspicious = True
    
    label_text = ", ".join(list(set(detected_labels))) if detected_labels else "Safe Driving"
    
    return {
        "pred_label": label_text,
        "suspicious": is_suspicious,
        "probs": []
    }


# ============================================================
# 2) MODEL TASK B (R2PLUS1D + BiLSTM – VIDEO)
# ============================================================
class R2Plus1D_BiLSTM(nn.Module):
    def __init__(self, num_classes=2, use_pretrain=False,
                 lstm_hidden=256, lstm_layers=2,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        base = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT if use_pretrain else None)
        self.stem, self.layer1 = base.stem, base.layer1
        self.layer2, self.layer3, self.layer4 = base.layer2, base.layer3, base.layer4
        self.out_channels = 512
        self.lstm = nn.LSTM(
            input_size=self.out_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=(dropout if lstm_layers > 1 else 0.0),
        )
        feat_dim = lstm_hidden * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):  # (B,3,T,H,W)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)              # (B,512,T',H',W')
        x = x.mean(dim=[3, 4])          # (B,512,T')
        x = x.transpose(1, 2)           # (B,T',512)
        x, _ = self.lstm(x)             # (B,T',H*)
        x = x[:, -1, :]                 # (B,H*)
        x = self.dropout(x)
        logits = self.fc(x)             # (B,num_classes)
        return logits


def load_model_taskB(path: str, device: torch.device, num_classes: int) -> nn.Module:
    model = R2Plus1D_BiLSTM(num_classes=num_classes, use_pretrain=False)
    state = torch.load(path, map_location=device)
    if isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state
    model.to(device)
    if CHANNELS_LAST_3D:
        model = model.to(memory_format=torch.channels_last_3d)
    model.eval()
    return model


def load_best_thr_B(thr_path: str) -> float:
    if not os.path.exists(thr_path):
        return 0.5
    with open(thr_path, "r") as f:
        data = json.load(f)
    return float(data.get("best_thr_pos", 0.5))


def get_frame_count(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    return n


def preprocess_clip_B(frames: List[np.ndarray], T: int) -> torch.Tensor:
    """
    frames: list length T, mỗi phần tử RGB (H,W,3), uint8
    -> tensor (1,3,T,H,W) normalized như lúc train.
    """
    if len(frames) == 0:
        raise ValueError("No frames in clip")

    clip_tensors = []
    for f in frames:
        img = TF.to_tensor(f)  # (3,H,W), [0,1]
        img = TF.resize(img, (SIZE_B, SIZE_B), antialias=True)
        clip_tensors.append(img)
    clip = torch.stack(clip_tensors, dim=1)  # (3,T,H,W)

    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    clip = (clip - mean) / std
    clip = clip.unsqueeze(0)  # (1,3,T,H,W)
    return clip

def predict_clip_taskB(
    frames: List[np.ndarray],
    model: nn.Module,
    class_names: List[str],
    pos_idx: int,
    best_thr: float,
    T: int,
) -> Dict[str, Any]:
    """
    frames: list frame RGB
    model: 1 trong các model B
    class_names: tuple/tên class của model đó
    pos_idx: index lớp dương (attack / Fight / fight)
    best_thr: threshold F1 tốt nhất cho lớp dương
    T: số frame/clip cho model này
    """
    if len(frames) < T and len(frames) > 0:
        frames = frames + [frames[-1]] * (T - len(frames))
    elif len(frames) > T:
        frames = frames[-T:]

    clip = preprocess_clip_B(frames, T).to(
        DEVICE,
        non_blocking=True,
        memory_format=(torch.channels_last_3d if CHANNELS_LAST_3D else torch.contiguous_format),
    )

    with torch.no_grad():
        logits = model(clip)
        probs = F.softmax(logits, dim=1)[0]

    probs_list = probs.cpu().numpy().tolist()
    prob_pos = float(probs_list[pos_idx])
    is_pos_thr = prob_pos >= best_thr

    pred_idx = int(np.argmax(probs_list))
    pred_label = class_names[pred_idx]

    return {
        "prob_pos": prob_pos,
        "is_pos_thr": is_pos_thr,
        "pos_class": class_names[pos_idx],
        "pred_idx_argmax": pred_idx,
        "pred_label_argmax": pred_label,
        "probs": [
            {"label": class_names[i], "score": float(p)}
            for i, p in enumerate(probs_list)
        ],
    }

# Pool chứa tất cả model Task B đã load:
# key = tên (custom_cabin_attack / rwf2000 / hockey_fight)
# value = {model, classes, pos_class, pos_idx, best_thr}
MODEL_B_POOL: Dict[str, Dict[str, Any]] = {}

for name, cfg in TASK_B_CONFIGS.items():
    mpath = cfg["model_path"]
    tpath = cfg["thr_path"]
    classes = list(cfg["classes"])
    pos_class = cfg["pos_class"]

    if not os.path.exists(mpath):
        continue

    try:
        model = load_model_taskB(mpath, DEVICE, num_classes=len(classes))
        best_thr = load_best_thr_B(tpath)
        pos_idx = classes.index(pos_class)
        T = cfg.get("T", 24)
        MODEL_B_POOL[name] = {
            "model": model,
            "classes": classes,
            "pos_class": pos_class,
            "pos_idx": pos_idx,
            "best_thr": best_thr,
            "T": T,
        }
    except Exception as e:
        print(f"[ERROR] Failed to load Task B model '{name}': {e}")

# ============================================================
# 3) FASTAPI APP
# ============================================================
app = FastAPI(title="Driver Behavior & Attack Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_A_CACHE = {}

# Preload ResNet18
try:
    m, t = load_model_A("ResNet18")
    if m: MODEL_A_CACHE["ResNet18"] = (m, t)
except Exception as e:
    print(f"Init load error: {e}")


# ============================================================
# 4) ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    # Debug: List files in models dir
    model_files = []
    if os.path.exists(MODELS_DIR):
        model_files = os.listdir(MODELS_DIR)
        
    return {
        "status": "ok",
        "device": str(DEVICE),
        "taskA_loaded": "ResNet18" in MODEL_A_CACHE,
        "taskB_loaded": len(MODEL_B_POOL) > 0,
        "taskB_models": list(MODEL_B_POOL.keys()),
        "taskA_models_cached": list(MODEL_A_CACHE.keys()),
        "debug_models_dir": MODELS_DIR,
        "debug_model_files": model_files
    }


@app.post("/predict_image")
async def predict_image(
    file: UploadFile = File(...),
    run_taskA: bool = Query(True),
    model_name_A: str = Query("ResNet18"),
    run_taskB: bool = Query(False),
    run_both: bool = Query(False),
    taskB_name: str = Query("custom_cabin_attack"),
):
    """
    Tab Image Detect dùng endpoint này.
    - Chủ yếu dành cho Task A (ResNet18).
    - Task B với ảnh chỉ là demo: coi ảnh như 1 frame, nhân lên T_B frame.
    """
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        raise HTTPException(status_code=400, detail="Vui lòng upload ảnh (.jpg/.png/.bmp)")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không đọc được ảnh: {e}")

    results: Dict[str, Any] = {}

    if run_both:
        run_taskA = True
        run_taskB = True

    # ---- Task A ----
    # ---- Task A ----
    if run_taskA:
        # Load/Get model from cache
        modelA = None
        typeA = None
        
        if model_name_A in MODEL_A_CACHE:
            modelA, typeA = MODEL_A_CACHE[model_name_A]
        else:
            modelA, typeA = load_model_A(model_name_A)
            if modelA:
                MODEL_A_CACHE[model_name_A] = (modelA, typeA)
        
        if modelA is None:
            raise HTTPException(status_code=500, detail=f"Không load được model Task A: {model_name_A}")

        # Convert PIL to numpy for processing functions
        frame_rgb = np.array(img)
        
        if typeA == "classification":
            resA = process_frame_taskA_resnet(modelA, frame_rgb)
        else:
            resA = process_frame_taskA_yolo(modelA, frame_rgb)
            
        results["taskA"] = resA

    # ---- Task B (demo ảnh) ----
    if run_taskB:
        if taskB_name not in MODEL_B_POOL:
            raise HTTPException(status_code=400, detail=f"Task B model '{taskB_name}' không tồn tại")

        infoB = MODEL_B_POOL[taskB_name]
        modelB = infoB["model"]
        classesB = infoB["classes"]
        pos_idxB = infoB["pos_idx"]
        best_thrB = infoB["best_thr"]
        T_B = infoB["T"]

        frame = np.array(img)   # (H,W,3)
        frames = [frame] * T_B  # giả clip T_B frame giống nhau

        resB = predict_clip_taskB(frames, modelB, classesB, pos_idxB, best_thrB, T_B)
        results["taskB"] = {"model_name": taskB_name, **resB}

    # ---- Quy tắc ưu tiên (nếu cả 2) ----
    final_alert = None
    if "taskB" in results and results["taskB"]["is_pos_thr"]:
        final_alert = "ATTACK"
    elif "taskA" in results:
        final_alert = results["taskA"]["pred_label"]

    results["final_alert"] = final_alert

    return results





@app.post("/predict_video_stream_sse")
async def predict_video_stream_sse(
    file: UploadFile = File(...),
    run_taskA: bool = Query(True),
    model_name_A: str = Query("ResNet18"),
    run_taskB: bool = Query(True),
    taskB_name: str = Query("custom_cabin_attack"),
):
    """
    Stream video processing results realtime using Server-Sent Events (SSE).
    FE sẽ nhận events dần dần khi đang xử lý, không phải đợi xong mới nhận.
    """
    # 1) Kiểm tra định dạng file
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Vui lòng upload video (.mp4/.avi/.mov/.mkv)")

    # 2) Load model Task A
    modelA = None
    typeA = None
    if run_taskA:
        if model_name_A in MODEL_A_CACHE:
            modelA, typeA = MODEL_A_CACHE[model_name_A]
        else:
            modelA, typeA = load_model_A(model_name_A)
            if modelA:
                MODEL_A_CACHE[model_name_A] = (modelA, typeA)
        
        if modelA is None:
             raise HTTPException(status_code=500, detail=f"Không load được model Task A: {model_name_A}")

    # 3) Chuẩn bị model Task B theo tên (multi-model)
    modelB = None
    classesB: List[str] = []
    pos_idxB: int = 0
    best_thrB: float = 0.5
    T_B = 24  # Khởi tạo mặc định

    if run_taskB:
        if taskB_name not in MODEL_B_POOL:
            raise HTTPException(status_code=400, detail=f"Task B model '{taskB_name}' không tồn tại")
        infoB = MODEL_B_POOL[taskB_name]
        modelB = infoB["model"]
        classesB = infoB["classes"]
        pos_idxB = infoB["pos_idx"]
        best_thrB = infoB["best_thr"]
        T_B = infoB["T"]

    # 4) Lưu video tạm để OpenCV đọc
    suffix = os.path.splitext(file.filename)[1]
    try:
        # Đọc file theo chunks để tránh lỗi memory
        contents = b""
        chunk_size = 1024 * 1024  # 1MB chunks
        max_size = 500 * 1024 * 1024  # 500MB limit
        
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            contents += chunk
            if len(contents) > max_size:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File quá lớn ({len(contents) / 1024 / 1024:.1f}MB). Vui lòng upload file nhỏ hơn 500MB"
                )
        
        # Ghi file tạm vào thư mục project (ổ E) thay vì ổ C
        # Tạo thư mục temp trong project nếu chưa có
        temp_dir = os.path.join(_script_dir, "temp_videos")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Tạo file tạm với tên unique
        import time
        temp_filename = f"video_{int(time.time() * 1000)}{suffix}"
        tmp_path = os.path.join(temp_dir, temp_filename)
        
        with open(tmp_path, "wb") as tmp:
            tmp.write(contents)
    except OSError as e:
        if "No space left" in str(e) or e.errno == 28:
            raise HTTPException(
                status_code=507,
                detail="Không đủ dung lượng ổ đĩa để xử lý video. Vui lòng giải phóng dung lượng hoặc upload file nhỏ hơn."
            )
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu file tạm: {e}")

    def generate_events():
        """Generator function để stream events về FE"""
        cap = None
        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                try:
                    yield f"data: {json.dumps({'error': 'Không mở được video'})}\n\n"
                except Exception:
                    pass  # Client đã disconnect
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

            # Gửi metadata trước
            try:
                yield f"data: {json.dumps({'type': 'metadata', 'total_frames': total_frames, 'fps': fps})}\n\n"
            except Exception:
                return  # Client đã disconnect

            stage = 1
            stride = STAGE1_STRIDE
            frame_buffer: List[np.ndarray] = []
            suspicious_history: List[bool] = []
            suspicious_A_history: List[bool] = []
            suspicious_B_history: List[bool] = []
            # Thêm detected_B_history để track frame vượt ngưỡng phát hiện (best_thr)
            detected_B_history: List[bool] = []
            
            last_attack_alert_time: Optional[float] = None  # Thời gian của lần trigger ATTACK_DETECTED cuối cùng
            last_distraction_alert_time: Optional[float] = None
            distraction_detected = False
            suspicious_frames_for_attack: List[int] = []

            frame_idx = 0
            detect_count = 0

            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_idx += 1
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                if run_taskB and modelB is not None:
                    frame_buffer.append(frame_rgb)
                    if len(frame_buffer) > T_B:
                        frame_buffer.pop(0)

                # Chỉ detect ở các frame cách nhau "stride"
                if frame_idx % stride != 0:
                    continue

                detect_count += 1
                event: Dict[str, Any] = {
                    "type": "event",
                    "frame_idx": frame_idx,
                    "time_s": frame_idx / fps if fps > 0 else None,
                    "stage": stage,
                }

                suspicious = False
                suspicious_A = False
                suspicious_B = False
                detected_B = False  # Track xem frame này có vượt ngưỡng phát hiện không

                # Task A
                if run_taskA and modelA is not None:
                    if typeA == "classification":
                        resA = process_frame_taskA_resnet(modelA, frame_rgb)
                    else:
                        resA = process_frame_taskA_yolo(modelA, frame_rgb)
                    
                    event["taskA"] = resA
                    if resA["suspicious"]:
                        suspicious = True
                        suspicious_A = True

                # Task B
                if run_taskB and modelB is not None and len(frame_buffer) > 0:
                    resB = predict_clip_taskB(
                        frames=list(frame_buffer),
                        model=modelB,
                        class_names=classesB,
                        pos_idx=pos_idxB,
                        best_thr=best_thrB,
                        T=T_B,
                    )
                    event["taskB"] = {
                        "model_name": taskB_name,
                        **resB
                    }

                    # Tab 3: Dùng best_thr * hệ số (linh hoạt theo từng model)
                    suspicious_thr_b = best_thrB * SUSPICIOUS_THR_RATIO
                    
                    # Check suspicious (Nghi vấn)
                    if resB["prob_pos"] >= suspicious_thr_b:
                        suspicious = True
                        suspicious_B = True
                        suspicious_frames_for_attack.append(frame_idx)
                        
                    # Check detected (Phát hiện)
                    if resB["is_pos_thr"]:
                        detected_B = True

                # Cập nhật history
                suspicious_history.append(suspicious)
                suspicious_A_history.append(suspicious_A)
                suspicious_B_history.append(suspicious_B)
                detected_B_history.append(detected_B)
                
                if len(suspicious_history) > SUSPICIOUS_WINDOW:
                    suspicious_history.pop(0)
                    suspicious_A_history.pop(0)
                    suspicious_B_history.pop(0)
                    detected_B_history.pop(0)

                event["suspicious"] = suspicious
                event["suspicious_A"] = suspicious_A
                event["suspicious_B"] = suspicious_B

                # Chuyển stage
                if stage == 1 and suspicious:
                    stage = 2
                    stride = STAGE2_STRIDE
                    event["stage_switch"] = "1->2"

                # Kiểm tra trigger cảnh báo
                if len(suspicious_history) < SUSPICIOUS_WINDOW:
                    event["window_suspicious_ratio"] = None
                    event["window_has_A"] = None
                    event["window_has_B"] = None
                    event["window_suspicious_ratio_A"] = None
                    event["distraction_detected"] = False
                else:
                    num_susp = sum(suspicious_history)
                    ratio = num_susp / SUSPICIOUS_WINDOW
                    has_A = any(suspicious_A_history)
                    has_B = any(suspicious_B_history)
                    num_susp_A = sum(suspicious_A_history)
                    ratio_A = num_susp_A / SUSPICIOUS_WINDOW
                    
                    # Task B Ratios
                    num_susp_B = sum(suspicious_B_history)
                    ratio_susp_B = num_susp_B / SUSPICIOUS_WINDOW
                    
                    num_detected_B = sum(detected_B_history)
                    ratio_detected_B = num_detected_B / SUSPICIOUS_WINDOW
                    
                    event["window_suspicious_ratio"] = ratio
                    event["window_has_A"] = has_A
                    event["window_has_B"] = has_B
                    event["window_suspicious_ratio_A"] = ratio_A

                    # Cảnh báo Task B (attack) - Logic mới:
                    # 1. 80% là nghi vấn hoặc phát hiện (ratio_susp_B >= 0.8)
                    # 2. HOẶC 50% là phát hiện (ratio_detected_B >= 0.5)
                    current_time = event.get("time_s") if event.get("time_s") is not None else (frame_idx / fps if fps > 0 else None)
                    
                    trigger_condition_B = (ratio_susp_B >= 0.8) or (ratio_detected_B >= 0.5)
                    
                    can_trigger_attack = (
                        trigger_condition_B and has_B and
                        (last_attack_alert_time is None or 
                         current_time is None or 
                         (current_time - last_attack_alert_time) >= ATTACK_COOLDOWN_SECONDS)
                    )
                    if can_trigger_attack:
                        last_attack_alert_time = current_time
                        event["alert"] = "ATTACK_DETECTED"
                    
                    # Phát hiện Task A (distraction) - dùng TASK_A_DISTRACTION_RATIO (cao hơn)
                    can_trigger_distraction = (
                        ratio_A >= TASK_A_DISTRACTION_RATIO and has_A and
                        (last_distraction_alert_time is None or
                         current_time is None or
                         (current_time - last_distraction_alert_time) >= DISTRACTION_COOLDOWN_SECONDS)
                    )
                    if can_trigger_distraction:
                        last_distraction_alert_time = current_time
                        distraction_detected = True
                        event["distraction_detected"] = True
                    else:
                        event["distraction_detected"] = False

                # Stream event về FE ngay lập tức
                try:
                    yield f"data: {json.dumps(event)}\n\n"
                except (GeneratorExit, BrokenPipeError, ConnectionResetError, OSError):
                    break
                except Exception:
                    break

            if cap is not None:
                cap.release()

            # Gửi summary cuối cùng (nếu client còn kết nối)
            try:
                summary = {
                    "type": "summary",
                    "total_frames": total_frames,
                    "fps": fps,
                    "detect_count": detect_count,
                    "alert_triggered": last_attack_alert_time is not None,
                    "distraction_detected": distraction_detected,
                    "taskB_model_used": taskB_name if run_taskB else None,
                    "suspicious_frames_for_attack": suspicious_frames_for_attack,
                }
                yield f"data: {json.dumps(summary)}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            except (GeneratorExit, BrokenPipeError, ConnectionResetError, OSError):
                pass
            except Exception:
                pass

        except Exception as e:
            try:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            except Exception:
                pass
            print(f"[ERROR] Error in video processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Đảm bảo release resources
            if cap is not None:
                cap.release()
            # Xoá file tạm ngay lập tức
            try:
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    print(f"[INFO] Đã xóa file tạm: {tmp_path}")
            except Exception as e:
                print(f"[WARN] Không thể xóa file tạm {tmp_path}: {e}")

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Tắt buffering cho nginx
        }
    )
