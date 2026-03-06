"""
FastAPI Web Servisi - Türk İşaret Dili → Türkçe Metin Çevirisi

AKTİF ÖZELLİKLER:
1. AUTSL TRANSFORMER (226 Türkçe kelime) - ANA MODEL:
   - Model: autsl_transformer/model/best_model.pt
   - Accuracy: %83.9 val, %81.2 test
   - Architecture: SignTransformerPro (6 layer, 384 d_model, 12 head)
   - Input: 30-frame sequence, 718 feature (raw+velocity+acceleration+distances)
   - Endpoint: POST /predict/frame (tek frame gönder, 30 frame dolunca tahmin döner)
   - Endpoint: POST /predict/sequence (30 frame'lik base64 dizisi gönder)

2. YOLO İŞARET KELIME TANIMA (20 kelime):
   - Model: yolo_runs/turk_isaret_v1/weights/best.pt
   - Endpoint: POST /predict_sign

3. TTS (Text-to-Speech):
   - gTTS ile Türkçe ses sentezi
   - Endpoint: POST /tts

4. WEB UI:
   - GET / → web/index.html

ENDPOINTS:
- GET /health                  - API sağlık kontrolü
- GET /model/info              - Model bilgileri
- POST /predict/frame          - Tek webcam frame'i gönder (AUTSL rolling buffer)
- POST /predict/sequence       - 30 frame'lik dizi gönder (AUTSL)
- POST /predict_sign           - YOLOv11 ile 20 kelime tanıma
- POST /tts                    - Metni sese çevir
- POST /reset/buffer           - AUTSL frame buffer'ını sıfırla
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import torch
from typing import Optional, List
import base64
from pathlib import Path
import sys
from datetime import datetime
from gtts import gTTS
import io

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# Proje modüllerini import et
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from signbridge.config import API_CONFIG, CHECKPOINT_DIR
from signbridge.utils.logging_utils import setup_logger

# AUTSL Transformer
from autsl_transformer.inference import AUTSLPredictor


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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global değişkenler ──────────────────────────────────────────────────────
autsl_predictor: Optional[AUTSLPredictor] = None  # ANA MODEL

yolo_model = None  # YOLO (20 kelime)
YOLO_CLASSES = ['Anne', 'Arkadas', 'Baba', 'Dur', 'Ev', 'Evet', 'Hayir', 'Kardes',
                'Merhaba', 'Nasil', 'Nerede', 'Ozur-Dilemek', 'Tamam', 'Telefon',
                'Tesekkurler', 'Tuvalet', 'Yemek', 'icmek', 'iyi', 'kotu']


# ── Pydantic modelleri ──────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    """Tahmin isteği modeli"""
    image_base64: Optional[str] = None
    landmarks: Optional[List[float]] = None
    model_mode: Optional[str] = "autsl"


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


@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    global autsl_predictor, yolo_model

    logger.info("=" * 60)
    logger.info("SignBridge API Başlatılıyor")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── AUTSL Transformer (ANA MODEL) ──────────────────────────────
    try:
        autsl_dir = ROOT_DIR / "autsl_transformer" / "model"
        autsl_predictor = AUTSLPredictor(model_dir=str(autsl_dir), device=device)
        logger.info(f"✅ AUTSL Transformer yüklendi (226 sınıf, %83.9 val acc)")
    except Exception as e:
        logger.error(f"AUTSL modeli yüklenemedi: {e}")
        autsl_predictor = None

    # ── YOLOv11 (20 kelime) ────────────────────────────────────────
    if _YOLO_AVAILABLE:
        try:
            yolo_path = ROOT_DIR / "yolo_runs" / "turk_isaret_v1" / "weights" / "best.pt"
            if yolo_path.exists():
                yolo_model = YOLO(str(yolo_path))
                logger.info(f"✅ YOLOv11 yüklendi: {yolo_path}")
            else:
                logger.warning(f"YOLOv11 ağırlıkları bulunamadı: {yolo_path}")
        except Exception as e:
            logger.error(f"YOLOv11 yükleme hatası: {e}")
    else:
        logger.warning("ultralytics kurulu değil, YOLOv11 devre dışı")

    logger.info("API hazır")


@app.on_event("shutdown")
async def shutdown_event():
    global autsl_predictor
    logger.info("API kapatılıyor...")
    if autsl_predictor:
        autsl_predictor.close()
    logger.info("API kapatıldı")


@app.get("/", response_class=HTMLResponse)
async def root():
    web_path = ROOT_DIR / "web" / "index.html"
    if web_path.exists():
        return FileResponse(web_path)
    return JSONResponse({
        "message": "SignBridge API - Türk İşaret Dili Çevirici",
        "version": API_CONFIG["version"],
        "docs": "/api/docs",
    })


@app.get("/health")
async def health_check():
    return JSONResponse({
        "status": "healthy",
        "api_version": API_CONFIG["version"],
        "autsl_loaded": autsl_predictor is not None,
        "yolo_loaded": yolo_model is not None,
    })


@app.get("/model/info")
async def model_info():
    """Model bilgileri"""
    if autsl_predictor is None:
        raise HTTPException(status_code=503, detail="AUTSL modeli yüklü değil")

    feat_cfg = autsl_predictor.feat_config or {}
    info = {
        "autsl_transformer": {
            "loaded": True,
            "num_classes": len(autsl_predictor.label_map),
            "feature_size": feat_cfg.get("feature_size", 718),
            "seq_length": autsl_predictor.seq_length,
            "val_acc": feat_cfg.get("best_val_acc"),
            "test_acc": feat_cfg.get("test_acc_no_tta"),
            "buffer_size": len(autsl_predictor.frame_buffer),
        },
        "yolo": {
            "loaded": yolo_model is not None,
            "num_classes": len(YOLO_CLASSES),
            "classes": YOLO_CLASSES,
        },
    }
    return JSONResponse(content=info)


@app.post("/predict/frame", response_model=PredictionResponse)
async def predict_frame(request: PredictionRequest):
    """
    Tek webcam frame'i gönder.
    Sunucu-taraflı rolling buffer (30 frame) dolar dolmaz tahmin döner.

    Kullanım:
      - Her webcam karesini base64 olarak gönder
      - Buffer dolmadan: prediction_text="", confidence=0, message="{N}/30 frame"
      - Buffer dolunca: AUTSL tahmini döner (her frame'de güncellenir)
    """
    if autsl_predictor is None:
        raise HTTPException(status_code=503, detail="AUTSL modeli yüklü değil")

    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 gerekli")

    try:
        img_bytes = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Geçersiz görüntü")

        autsl_predictor.process_frame(frame)
        label, conf, buf_len = autsl_predictor.predict()

        if label is None:
            return PredictionResponse(
                success=False,
                prediction_id=-1,
                prediction_text="",
                confidence=0.0,
                message=f"{buf_len}/{autsl_predictor.seq_length} frame toplandı"
            )

        pred_id = next(
            (k for k, v in autsl_predictor.label_map.items() if v == label), -1
        )
        return PredictionResponse(
            success=True,
            prediction_id=pred_id,
            prediction_text=label,
            confidence=float(conf),
            message="AUTSL tahmini"
        )
    except Exception as e:
        logger.error(f"/predict/frame hatası: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/sequence", response_model=PredictionResponse)
async def predict_sequence(request: PredictionRequest):
    """
    30 frame'lik dizi gönder (images_base64 listesi).
    Sunucu buffer'ını etkilemez.
    """
    if autsl_predictor is None:
        raise HTTPException(status_code=503, detail="AUTSL modeli yüklü değil")
    raise HTTPException(
        status_code=501,
        detail="Henüz implemente edilmedi. /predict/frame endpoint'ini kullanın."
    )


@app.post("/reset/buffer")
async def reset_buffer():
    """AUTSL frame buffer'ını sıfırla"""
    if autsl_predictor:
        autsl_predictor.reset_buffer()
    return JSONResponse({"message": "Buffer sıfırlandı"})


# ── Eski /predict endpoint'i (geriye dönük uyumluluk) ──────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Geriye dönük uyumluluk için /predict/frame'e yönlendir.
    """
    return await predict_frame(request)


def _legacy_predict_id(prediction_text: str) -> int:
    """Sınıf ID'si bul (eski kod uyumu)"""
    # Sınıf ID'yi bul
    prediction_id = -1
    for class_id, text in {}.items():
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
