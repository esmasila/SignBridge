import mediapipe as mp
import traceback

try:
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic()
    print("✅ Holistic başlatıldı")
    holistic.close()
except Exception as e:
    print(f"❌ Hata: {e}")
    traceback.print_exc()
