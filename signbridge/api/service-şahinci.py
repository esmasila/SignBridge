"""
FastAPI Web Servisi - Türk İşaret Dili → Türkçe Metin Çevirisi

AKTİF ÖZELLİKLER:
1. KELİME TANIMA (20 sınıf): 
   - Model: YOLOv11 - best.pt
   - Sınıflar: Anne, Arkadas, Baba, Dur, Ev, Evet, Hayir, vb.
   - Endpoint: POST /predict (model_mode="words")

2. RAKAM TANIMA (0-9): 
   - Model: YOLOv11 - digits_yolo.pt (eğitilecek)
   - Sınıflar: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
   - Endpoint: POST /predict (model_mode="digits")

3. TTS (Text-to-Speech):
   - gTTS ile Türkçe ses sentezi
   - Endpoint: POST /tts

4. WEB UI:
   - GET / → web/index.html
   - Gerçek zamanlı webcam görüntüsü
   - Model seçici (words/digits)
   - Otomatik TTS

ENDPOINTS:
- GET /health - API sağlık kontrolü
- POST /predict - Görüntüden tahmin (words veya digits modu)
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

# Global variables
device = None
current_model_mode = "words"  # "words" veya "digits"

# YOLO modelleri
yolo_words_model = None  # Kelimeler için
yolo_digits_model = None  # Rakamlar için

YOLO_WORDS_CLASSES = ['Anne', 'Arkadas', 'Baba', 'Dur', 'Ev', 'Evet', 'Hayir', 'Kardes', 
                      'Merhaba', 'Nasil', 'Nerede', 'Ozur-Dilemek', 'Tamam', 'Telefon', 
                      'Tesekkurler', 'Tuvalet', 'Yemek', 'icmek', 'iyi', 'kotu']

YOLO_DIGITS_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


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


def load_yolo_models():
    """YOLO modellerini yükle (words ve digits)"""
    global yolo_words_model, yolo_digits_model
    
    # Kelimeler modeli
    try:
        words_path = Path(__file__).parent.parent.parent / "yolo_runs" / "turk_isaret_v1" / "weights" / "best.pt"
        if words_path.exists():
            yolo_words_model = YOLO(str(words_path))
            logger.info(f"✅ YOLO WORDS modeli yüklendi: {words_path.name}")
            logger.info(f"   20 kelime: {', '.join(YOLO_WORDS_CLASSES[:5])}...")
        else:
            logger.warning(f"YOLO words modeli bulunamadı: {words_path}")
            yolo_words_model = None
    except Exception as e:
        logger.error(f"YOLO words yükleme hatası: {e}")
        yolo_words_model = None
    
    # Rakamlar modeli (yeni eğitilecek)
    try:
        digits_path = Path(__file__).parent.parent.parent / "yolo_runs" / "digits_v1" / "weights" / "best.pt"
        if digits_path.exists():
            yolo_digits_model = YOLO(str(digits_path))
            logger.info(f"✅ YOLO DIGITS modeli yüklendi: {digits_path.name}")
            logger.info(f"   10 rakam: {', '.join(YOLO_DIGITS_CLASSES)}")
        else:
            logger.warning(f"YOLO digits modeli bulunamadı: {digits_path}")
            logger.warning("   ⚠️ Rakam modelini eğitmek için: 'yolo_runs/digits_v1/weights/best.pt'")
            yolo_digits_model = None
    except Exception as e:
        logger.error(f"YOLO digits yükleme hatası: {e}")
        yolo_digits_model = None
    
    return (yolo_words_model is not None) or (yolo_digits_model is not None)


def load_model(mode="words"):
    """Eski API uyumluluğu için (artık kullanılmıyor)"""
    global current_model_mode
    current_model_mode = mode
    logger.info(f"Model modu değiştirildi: {mode}")
    return True


def predict_yolo(image: np.ndarray, mode: str = "words", conf_threshold: float = 0.25):
    """
    YOLO ile tahmin yap
    Args:
        image: BGR formatında NumPy array
        mode: "words" veya "digits"
        conf_threshold: Güven eşiği
    Returns:
        (prediction_text, confidence) veya (None, 0.0)
    """
    try:
        # Model seç
        if mode == "words" and yolo_words_model:
            results = yolo_words_model.predict(image, conf=conf_threshold, verbose=False)
            classes = YOLO_WORDS_CLASSES
        elif mode == "digits" and yolo_digits_model:
            results = yolo_digits_model.predict(image, conf=conf_threshold, verbose=False)
            classes = YOLO_DIGITS_CLASSES
        else:
            logger.warning(f"{mode} modu için model yüklü değil")
            return None, 0.0
        
        # Sonuçları işle
        if len(results) > 0 and len(results[0].boxes) > 0:
            # En yüksek güvene sahip tespit
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            
            max_idx = np.argmax(confidences)
            prediction_id = int(class_ids[max_idx])
            confidence = float(confidences[max_idx])
            
            if 0 <= prediction_id < len(classes):
                prediction_text = classes[prediction_id]
                return prediction_text, confidence
        
        return None, 0.0
        
    except Exception as e:
        logger.error(f"YOLO tahmin hatası: {e}")
        return None, 0.0


@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    global device, current_model_mode
    
    logger.info("=" * 60)
    logger.info("SignBridge API Başlatılıyor (YOLO-Only Mode)")
    logger.info("=" * 60)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # YOLO modellerini yükle
    load_yolo_models()
    
    # Varsayılan mod
    current_model_mode = "words"
    
    logger.info("API hazır (YOLO-Only)")


@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapanışında çalışır"""
    logger.info("API kapatılıyor...")
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
    global yolo_words_model, yolo_digits_model, device
    
    status = {
        "status": "healthy",
        "api_version": API_CONFIG["version"],
        "words_model_loaded": yolo_words_model is not None,
        "digits_model_loaded": yolo_digits_model is not None,
        "device": device
    }
    
    return JSONResponse(content=status)


@app.get("/model/info")
async def model_info():
    """Model bilgileri (YOLO-Only)"""
    info = {
        "mode": "YOLO-Only",
        "words_model": {
            "loaded": yolo_words_model is not None,
            "num_classes": len(YOLO_WORDS_CLASSES),
            "classes": YOLO_WORDS_CLASSES,
            "architecture": "YOLOv11-nano",
            "path": "yolo_runs/turk_isaret_v1/weights/best.pt"
        },
        "digits_model": {
            "loaded": yolo_digits_model is not None,
            "num_classes": len(YOLO_DIGITS_CLASSES),
            "classes": YOLO_DIGITS_CLASSES,
            "architecture": "YOLOv11-nano",
            "path": "yolo_runs/digits_v1/weights/best.pt",
            "status": "Eğitilmesi gerekiyor" if yolo_digits_model is None else "Hazır"
        },
        "device": device,
        "current_mode": current_model_mode
    }
    
    return JSONResponse(content=info)


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Tahmin endpoint'i (YOLO-Only)
    Base64 görüntü kabul eder, YOLO ile tespit yapar
    """
    global current_model_mode
    
    # Model modu değiştirme
    if request.model_mode:
        current_model_mode = request.model_mode
    
    # Model kontrolü
    if current_model_mode == "words" and yolo_words_model is None:
        raise HTTPException(status_code=503, detail="Kelime modeli yüklü değil")
    elif current_model_mode == "digits" and yolo_digits_model is None:
        raise HTTPException(status_code=503, detail="Rakam modeli yüklü değil")
    
    try:
        # Görüntüden tahmin
        if request.image_base64:
            # Base64'ü decode et
            image_bytes = base64.b64decode(request.image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise HTTPException(status_code=400, detail="Geçersiz görüntü")
            
            # YOLO ile tahmin yap
            prediction_text, confidence = predict_yolo(image, mode=current_model_mode, conf_threshold=0.25)
            
            if prediction_text is None:
                return PredictionResponse(
                    success=False,
                    prediction_id=-1,
                    prediction_text="",
                    confidence=0.0,
                    message="Tespit yapılamadı"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="image_base64 gerekli"
            )
        
        # Sınıf ID'yi bul
        if current_model_mode == "words":
            classes = YOLO_WORDS_CLASSES
        else:
            classes = YOLO_DIGITS_CLASSES
        
        prediction_id = classes.index(prediction_text) if prediction_text in classes else -1
        
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
            # Sadece görüntü ile çalış (MediaPipe yok)
            prediction_text, confidence = predict_multimodal(image, None)
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
