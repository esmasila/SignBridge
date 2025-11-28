"""
FastAPI Web Servisi - Türk İşaret Dili → Türkçe Metin Çevirisi

AKTİF ÖZELLİKLER:
1. RAKAM TANIMA (0-9): 
   - Model: multimodal_digits_best.pt
   - Accuracy: 99.79%
   - FPS: ~27 (gerçek zamanlı)
   - Preprocessing: MediaPipe Holistic landmark + 64x64 görüntü
   - Endpoint: POST /predict (model_mode="digits")

2. TTS (Text-to-Speech):
   - gTTS ile Türkçe ses sentezi
   - Endpoint: POST /tts

3. WEB UI:
   - GET / → web/index.html
   - Gerçek zamanlı webcam görüntüsü
   - Model seçici (digits/alphabet)
   - Otomatik TTS

PASSİF/LEGACY:
- Alfabe tanıma (A-Z): legacy_alphabet/ klasörüne taşındı
- Model: kaggle_alphabet_best.pt (99.63% test ama gerçek dünya performansı zayıf)

ENDPOINTS:
- GET /health - API sağlık kontrolü
- POST /predict - Görüntüden tahmin (digits veya alphabet modu)
- POST /tts - Metni sese çevir
- GET /model/info - Model bilgileri
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import numpy as np
import cv2
import torch
from typing import Optional, List, Dict
import base64
from pathlib import Path
import sys
from datetime import datetime
from gtts import gTTS
import io
import tempfile
from ultralytics import YOLO

# Proje modüllerini import et
sys.path.append(str(Path(__file__).parent.parent.parent))

from signbridge.config import (
    API_CONFIG,
    CLASS_TO_TURKISH,
    CHECKPOINT_DIR
)
from signbridge.models.cnn_digits import load_digits_model
from signbridge.models.islr_sequence_model import load_sequence_model
from signbridge.models.multimodal_cnn import MultimodalDigitsCNN
from signbridge.data.preprocess import LandmarkExtractor
from signbridge.utils.logging_utils import setup_logger


# Logger
logger = setup_logger("api_service")

# FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware (frontend için)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Production'da belirli domainler yazılmalı
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model (başlangıçta None)
model = None
model_type = None
device = None
landmark_extractor = None
current_model_mode = "digits"  # "digits" veya "alphabet"

# YOLO model (işaret kelime tanıma)
yolo_model = None
YOLO_CLASSES = ['Anne', 'Arkadas', 'Baba', 'Dur', 'Ev', 'Evet', 'Hayir', 'Kardes', 
                'Merhaba', 'Nasil', 'Nerede', 'Ozur-Dilemek', 'Tamam', 'Telefon', 
                'Tesekkurler', 'Tuvalet', 'Yemek', 'icmek', 'iyi', 'kotu']

# Türk alfabesi harfleri (26 sınıf - 23 harf + del, nothing, space)
ALPHABET_LETTERS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'Y', 'Z', 'del', 'nothing', 'space']


class PredictionRequest(BaseModel):
    """Tahmin isteği modeli"""
    image_base64: Optional[str] = None
    landmarks: Optional[List[float]] = None
    model_mode: Optional[str] = "digits"  # "digits" veya "alphabet"


class PredictionResponse(BaseModel):
    """Tahmin yanıt modeli"""
    success: bool
    prediction_id: int
    prediction_text: str
    confidence: float
    message: Optional[str] = None


class TTSRequest(BaseModel):
    """TTS isteği modeli"""
    text: str
    lang: str = "tr"
    slow: bool = False


def load_model(mode="digits"):
    """Model yükle (digits veya alphabet)"""
    global model, model_type, current_model_mode, device
    
    if mode == "alphabet":
        model_path = CHECKPOINT_DIR / "kaggle_alphabet_best.pt"
        num_classes = 26  # 23 harf + del, nothing, space
    else:  # digits
        model_path = CHECKPOINT_DIR / "multimodal_digits_best.pt"
        num_classes = 10
    
    if not model_path.exists():
        logger.warning(f"Model bulunamadı: {model_path}")
        return False
    
    try:
        model = MultimodalDigitsCNN(num_classes=num_classes).to(device)
        checkpoint = torch.load(str(model_path), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # CPU'da quantization
        if device == "cpu":
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        model_type = "multimodal"
        current_model_mode = mode
        logger.info(f"✅ {mode.upper()} modeli yüklendi ({num_classes} sınıf)")
        return True
    except Exception as e:
        logger.error(f"Model yükleme hatası: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    global device, landmark_extractor, yolo_model
    
    logger.info("=" * 60)
    logger.info("SignBridge API Başlatılıyor")
    logger.info("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # Varsayılan model yükle (digits)
    load_model("digits")
    
    # YOLOv11 model yükle (işaret kelime tanıma)
    try:
        yolo_path = Path(__file__).parent.parent.parent / "yolo_runs" / "turk_isaret_v1" / "weights" / "best.pt"
        if yolo_path.exists():
            yolo_model = YOLO(str(yolo_path))
            logger.info(f"✅ YOLOv11 modeli yüklendi: {yolo_path.name}")
            logger.info(f"   20 kelime: {', '.join(YOLO_CLASSES[:5])}...")
        else:
            logger.warning(f"YOLOv11 modeli bulunamadı: {yolo_path}")
            yolo_model = None
    except Exception as e:
        logger.error(f"YOLOv11 yükleme hatası: {e}")
        yolo_model = None
    
    # MediaPipe
    try:
        landmark_extractor = LandmarkExtractor()
        logger.info("MediaPipe hazır")
    except Exception as e:
        logger.error(f"MediaPipe başlatılamadı: {e}")
        landmark_extractor = None
    
    logger.info("API hazır")


@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapanışında çalışır"""
    global landmark_extractor
    
    logger.info("API kapatılıyor...")
    
    if landmark_extractor:
        landmark_extractor.close()
    
    logger.info("API kapatıldı")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Ana sayfa - Web UI"""
    web_path = Path(__file__).parent.parent.parent / "web" / "index.html"
    if web_path.exists():
        return FileResponse(web_path)
    else:
        # Fallback JSON response
        return JSONResponse({
            "message": "SignBridge API - Türk İşaret Dili Çevirici",
            "version": API_CONFIG["version"],
            "endpoints": {
                "health": "/health",
                "predict": "/predict (POST)",
                "predict_image": "/predict/image (POST)",
                "model_info": "/model/info"
            }
        })


@app.get("/health")
async def health_check():
    """
    Sağlık kontrolü endpoint'i
    API'nin çalışıp çalışmadığını kontrol eder
    """
    global model, model_type, device
    
    status = {
        "status": "healthy",
        "api_version": API_CONFIG["version"],
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": device
    }
    
    return JSONResponse(content=status)


@app.get("/model/info")
async def model_info():
    """Model bilgileri"""
    global model, model_type, yolo_model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil")
    
    info = {
        "digits_model": {
            "model_type": model_type,
            "num_classes": len(CLASS_TO_TURKISH),
            "classes": CLASS_TO_TURKISH,
            "device": device
        },
        "yolo_model": {
            "loaded": yolo_model is not None,
            "num_classes": len(YOLO_CLASSES) if yolo_model else 0,
            "classes": YOLO_CLASSES if yolo_model else [],
            "architecture": "YOLOv11-nano",
            "mAP50": 0.99596
        }
    }
    
    return JSONResponse(content=info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Tahmin endpoint'i
    Base64 görüntü veya landmark dizisi kabul eder
    """
    global model, landmark_extractor, current_model_mode
    
    # Model değiştirme
    if request.model_mode and request.model_mode != current_model_mode:
        load_model(request.model_mode)
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model yüklü değil. Lütfen önce modeli eğitin."
        )
    
    try:
        # Görüntüden tahmin
        if request.image_base64:
            # Base64'ü decode et
            image_bytes = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Geçersiz görüntü")
            
            # Model tipine göre tahmin
            if model_type == "multimodal":
                # Multimodal: hem görüntü hem landmark kullan
                if landmark_extractor is None:
                    raise HTTPException(status_code=503, detail="MediaPipe hazır değil")
                
                # Grayscale'e çevir, sonra adaptive threshold uygula (ışığa uyumlu)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                    cv2.THRESH_BINARY, 11, 2)
                # Tekrar BGR'a çevir (MediaPipe BGR bekliyor)
                image = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
                
                # Görüntüyü küçült (MediaPipe için, daha hızlı)
                h, w = image.shape[:2]
                if w > 320:  # Eğer çok büyükse küçült
                    scale = 320 / w
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                
                landmarks = landmark_extractor.extract_from_image(image)
                if landmarks is None:
                    raise HTTPException(status_code=400, detail="Landmark çıkarılamadı")
                
                prediction_text, confidence = predict_multimodal(image, landmarks)
            elif model_type == "cnn":
                prediction_text, confidence = predict_from_image_cnn(image)
            else:
                # Sekans modeli için landmark çıkar
                if landmark_extractor is None:
                    raise HTTPException(status_code=503, detail="MediaPipe hazır değil")
                
                landmarks = landmark_extractor.extract_from_image(image)
                if landmarks is None:
                    raise HTTPException(status_code=400, detail="Landmark çıkarılamadı")
                
                prediction_text, confidence = predict_from_landmarks(landmarks)
        
        # Landmark'tan tahmin
        elif request.landmarks:
            landmarks = np.array(request.landmarks, dtype=np.float32)
            prediction_text, confidence = predict_from_landmarks(landmarks)
        
        else:
            raise HTTPException(
                status_code=400,
                detail="image_base64 veya landmarks gerekli"
            )
        
        # Sınıf ID'yi bul
        prediction_id = -1
        for class_id, text in CLASS_TO_TURKISH.items():
            if text == prediction_text:
                prediction_id = class_id
                break
        
        return PredictionResponse(
            success=True,
            prediction_id=prediction_id,
            prediction_text=prediction_text,
            confidence=float(confidence),
            message="Tahmin başarılı"
        )
    
    except Exception as e:
        logger.error(f"Tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """
    Görüntü dosyası upload ederek tahmin
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model yüklü değil")
    
    try:
        # Dosyayı oku
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Geçersiz görüntü dosyası")
        
        # Tahmin
        if model_type == "multimodal":
            if landmark_extractor is None:
                raise HTTPException(status_code=503, detail="MediaPipe hazır değil")
            
            landmarks = landmark_extractor.extract_from_image(image)
            if landmarks is None:
                raise HTTPException(status_code=400, detail="Landmark çıkarılamadı")
            
            prediction_text, confidence = predict_multimodal(image, landmarks)
        elif model_type == "cnn":
            prediction_text, confidence = predict_from_image_cnn(image)
        else:
            if landmark_extractor is None:
                raise HTTPException(status_code=503, detail="MediaPipe hazır değil")
            
            landmarks = landmark_extractor.extract_from_image(image)
            if landmarks is None:
                raise HTTPException(status_code=400, detail="Landmark çıkarılamadı")
            
            prediction_text, confidence = predict_from_landmarks(landmarks)
        
        # Sınıf ID
        prediction_id = -1
        for class_id, text in CLASS_TO_TURKISH.items():
            if text == prediction_text:
                prediction_id = class_id
                break
        
        return PredictionResponse(
            success=True,
            prediction_id=prediction_id,
            prediction_text=prediction_text,
            confidence=float(confidence),
            message="Tahmin başarılı"
        )
    
    except Exception as e:
        logger.error(f"Görüntü tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def predict_from_image_cnn(image: np.ndarray) -> tuple:
    """CNN ile görüntüden tahmin"""
    from torchvision import transforms
    from signbridge.config import DIGITS_MODEL_CONFIG
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(DIGITS_MODEL_CONFIG["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    with torch.inference_mode():  # no_grad() yerine inference_mode() (10-15% hızlı)
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = probabilities.max(1)
    
    class_id = predicted_class.item()
    confidence_score = confidence.item()
    prediction_text = CLASS_TO_TURKISH.get(class_id, f"Bilinmeyen ({class_id})")
    
    return prediction_text, confidence_score


def predict_from_landmarks(landmarks: np.ndarray) -> tuple:
    """Landmark'tan tahmin"""
    # TODO: Sekans modeli implementasyonu
    # Şimdilik basit bir tahmin döndür
    return "Landmark tahmini henüz desteklenmiyor", 0.0


def predict_multimodal(image: np.ndarray, landmarks: np.ndarray) -> tuple:
    """Multimodal model ile tahmin (görüntü + landmark)"""
    from torchvision import transforms
    from signbridge.config import DIGITS_MODEL_CONFIG
    
    # Görüntü preprocessing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(DIGITS_MODEL_CONFIG["input_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).unsqueeze(0).to(device)
    
    # Landmark preprocessing (flatten ve normalize)
    landmarks_flat = landmarks.flatten()
    
    # Alphabet modeli 324-dim landmark kullanıyor, 1629'a pad et
    if len(landmarks_flat) < 1629:
        landmarks_flat = np.pad(landmarks_flat, (0, 1629 - len(landmarks_flat)), mode='constant')
    elif len(landmarks_flat) > 1629:
        landmarks_flat = landmarks_flat[:1629]
    
    landmark_tensor = torch.from_numpy(landmarks_flat).float().unsqueeze(0).to(device)
    
    # Tahmin
    with torch.inference_mode():  # no_grad() yerine inference_mode() (10-15% hızlı)
        output = model(image_tensor, landmark_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_class = probabilities.max(1)
    
    class_id = predicted_class.item()
    confidence_score = confidence.item()
    
    # Model moduna göre text çevir
    if current_model_mode == "alphabet":
        prediction_text = ALPHABET_LETTERS[class_id] if class_id < len(ALPHABET_LETTERS) else f"Bilinmeyen ({class_id})"
    else:  # digits
        prediction_text = CLASS_TO_TURKISH.get(class_id, f"Bilinmeyen ({class_id})")
    
    return prediction_text, confidence_score


@app.post("/predict_sign")
async def predict_sign(request: PredictionRequest):
    """
    YOLOv11 ile işaret kelime tanıma endpoint'i
    Base64 görüntü kabul eder, 20 kelimeden birini döndürür
    """
    global yolo_model
    
    if yolo_model is None:
        raise HTTPException(
            status_code=503,
            detail="YOLOv11 modeli yüklü değil"
        )
    
    if not request.image_base64:
        raise HTTPException(
            status_code=400,
            detail="image_base64 gerekli"
        )
    
    try:
        # Base64'ü decode et
        image_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Geçersiz görüntü")
        
        # YOLO inference
        results = yolo_model(image, conf=0.5, verbose=False)
        
        # En yüksek confidence'a sahip tespit
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            
            # En iyi tespit
            best_idx = confidences.argmax()
            best_class = classes[best_idx]
            best_conf = float(confidences[best_idx])
            
            prediction_text = YOLO_CLASSES[best_class]
            
            return PredictionResponse(
                success=True,
                prediction_id=best_class,
                prediction_text=prediction_text,
                confidence=best_conf,
                message="İşaret kelime tanıma başarılı"
            )
        else:
            # Tespit yok
            return PredictionResponse(
                success=False,
                prediction_id=-1,
                prediction_text="Tespit yok",
                confidence=0.0,
                message="Kare içinde işaret algılanamadı"
            )
    
    except Exception as e:
        logger.error(f"YOLO tahmin hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Türkçe metni sese dönüştürür (gTTS ile)
    
    Args:
        request: TTS isteği (text, lang, slow)
    
    Returns:
        MP3 ses dosyası stream
    """
    try:
        # gTTS ile ses oluştur
        tts = gTTS(text=request.text, lang=request.lang, slow=request.slow)
        
        # Memory'de oluştur
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # MP3 stream olarak döndür
        return StreamingResponse(
            audio_buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"inline; filename=tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            }
        )
    
    except Exception as e:
        logger.error(f"TTS hatası: {e}")
        raise HTTPException(status_code=500, detail=f"TTS oluşturulamadı: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Server başlatılıyor: {API_CONFIG['host']}:{API_CONFIG['port']}")
    
    uvicorn.run(
        "signbridge.api.service:app",
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        reload=API_CONFIG["reload"]
    )
