"""
YOLOv11 API Test Script - İşaret Kelime Tanıma
Webcam'den görüntü alıp API'ye gönderir, sonucu ekrana yazdırır
"""

import cv2
import requests
import base64
import time

API_URL = "http://127.0.0.1:8000"

def test_predict_sign():
    """Webcam'den frame alıp /predict_sign endpoint'ini test et"""
    print("=" * 70)
    print("YOLOv11 API TEST - İşaret Kelime Tanıma")
    print("=" * 70)
    print("20 kelime: MERHABA, EVET, HAYIR, TEŞEKKÜRLER, LÜTFEN, vb.")
    print("Kamera açılıyor... Q tuşu ile çıkış yapın")
    print("=" * 70)
    
    # Kamera aç
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return
    
    print("✅ Kamera açıldı")
    print("\n🎯 İşaret yapın ve SPACE tuşuna basarak test edin")
    print("   Q: Çıkış\n")
    
    frame_count = 0
    last_prediction = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Ekranda göster
        display_frame = frame.copy()
        
        # Bilgi yaz
        cv2.putText(display_frame, "SPACE: Test | Q: Cikis", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if last_prediction:
            # Son tahmini göster
            text = f"{last_prediction['text']} ({last_prediction['conf']:.1%})"
            cv2.putText(display_frame, text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow('YOLOv11 API Test', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Q ile çıkış
        if key == ord('q'):
            break
        
        # SPACE ile test
        if key == ord(' '):
            print(f"\n🔄 Frame {frame_count} test ediliyor...")
            
            # Frame'i encode et
            _, buffer = cv2.imencode('.jpg', frame)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            
            try:
                # API'ye gönder
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/predict_sign",
                    json={"image_base64": base64_image},
                    timeout=5
                )
                elapsed = (time.time() - start_time) * 1000  # ms
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data['success']:
                        last_prediction = {
                            'text': data['prediction_text'],
                            'conf': data['confidence']
                        }
                        
                        print(f"✅ Tespit: {data['prediction_text']}")
                        print(f"   Güven: {data['confidence']:.1%}")
                        print(f"   Süre: {elapsed:.0f}ms")
                    else:
                        print(f"⚠️  {data['message']}")
                        last_prediction = None
                else:
                    print(f"❌ API hatası: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ Hata: {e}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test tamamlandı!")


def test_health():
    """API sağlık kontrolü"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        data = response.json()
        
        print("\n📊 API Sağlık Durumu:")
        print(f"   Status: {data['status']}")
        print(f"   Model yüklü: {data['model_loaded']}")
        print(f"   Device: {data['device']}")
        
        return data['status'] == 'healthy'
    except Exception as e:
        print(f"❌ API'ye erişilemiyor: {e}")
        return False


def test_model_info():
    """Model bilgilerini göster"""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=2)
        data = response.json()
        
        print("\n📦 Model Bilgileri:")
        
        # Digits model
        digits = data.get('digits_model', {})
        print(f"\n   🔢 Digits Model:")
        print(f"      Sınıf sayısı: {digits.get('num_classes', 0)}")
        print(f"      Device: {digits.get('device', 'N/A')}")
        
        # YOLO model
        yolo = data.get('yolo_model', {})
        print(f"\n   ✋ YOLO Model:")
        print(f"      Yüklü: {'✅' if yolo.get('loaded') else '❌'}")
        print(f"      Mimari: {yolo.get('architecture', 'N/A')}")
        print(f"      mAP50: {yolo.get('mAP50', 0):.1%}")
        print(f"      Sınıf sayısı: {yolo.get('num_classes', 0)}")
        
        if yolo.get('classes'):
            print(f"      Kelimeler: {', '.join(yolo['classes'][:5])}...")
        
        return True
    except Exception as e:
        print(f"❌ Model bilgisi alınamadı: {e}")
        return False


if __name__ == "__main__":
    # 1. API sağlık kontrolü
    if not test_health():
        print("\n⚠️  API çalışmıyor! Önce API'yi başlatın:")
        print("   python -m uvicorn signbridge.api.service:app --reload")
        exit(1)
    
    # 2. Model bilgilerini göster
    test_model_info()
    
    # 3. Webcam testi
    print("\n" + "=" * 70)
    input("Webcam testine başlamak için ENTER'a basın...")
    test_predict_sign()
