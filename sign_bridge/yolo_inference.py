"""
YOLOv11 Real-time Inference
Webcam'den canlı işaret tanıma
"""

from ultralytics import YOLO
import cv2


def run_inference(model_path='yolo_runs/turk_isaret_v1/weights/best.pt'):
    """Webcam'den real-time inference"""
    
    print("=" * 70)
    print("YOLOv11 - REAL-TIME İŞARET TANIMA")
    print("=" * 70)
    
    # Model yükle
    print(f"\n📦 Model yükleniyor: {model_path}")
    model = YOLO(model_path)
    
    # Class names
    class_names = model.names
    print(f"🎯 Sınıflar ({len(class_names)}): {list(class_names.values())[:5]}...")
    
    print("\n🎥 Kamera açılıyor...")
    print("Kontroller: Q veya ESC - Çıkış\n")
    print("=" * 70 + "\n")
    
    # Kamera aç
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # YOLO inference
        results = model(frame, conf=0.5, verbose=False)
        
        # Sonuçları çiz
        annotated_frame = results[0].plot()
        
        # FPS göster
        cv2.putText(annotated_frame, "Q/ESC: Quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('YOLOv11 - Turk Isaret Dili', annotated_frame)
        
        # Çıkış
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Inference tamamlandı!")


if __name__ == "__main__":
    run_inference()
