"""
TİD Veri Toplama v2 — 225 Özellik (Pose + Eller, Yüz Yok)
5 kelime: MERHABA, TEŞEKKÜR, EVET, HAYIR, LÜTFEN

Her örnek: 30 frame × 225 özellik (.npy)
Web inference ile birebir uyumlu.

KULLANIM:
    python tid_sequence/scripts/collect_v2.py
    Etiket seç → S tuşu → 50 örnek otomatik toplanır

İPUÇLARI:
  - Koyu tişört giy (ten rengine yakın olmasın)
  - Düz, tek renk arka plan kullan
  - Işık yüzüne gelsin, arkandan değil
  - Göğüs-baş arası görünsün, kamera göz hizasında
  - Her işareti aynı hız ve şekilde yap
"""

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from datetime import datetime

# ── MediaPipe Holistic ──────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ── Ayarlar ─────────────────────────────────────────────────────────────────
BASE_DATA_PATH = Path(__file__).parent.parent / "data_v2"
BASE_DATA_PATH.mkdir(parents=True, exist_ok=True)

MAX_FRAMES = 30          # Her örnek 30 frame (web inference ile aynı)
TARGET_SAMPLES = 50      # Her kelime için 50 örnek
COUNTDOWN_SECONDS = 3    # İlk kayıt öncesi geri sayım
FPS_ESTIMATE = 30        # Tahmini kamera FPS

WORDS = ["MERHABA", "TEŞEKKÜR", "EVET", "HAYIR", "LÜTFEN"]


def extract_landmarks(results):
    """
    MediaPipe sonuçlarından 225 özellik çıkar (Pose + Eller).
    Yüz dahil DEĞİL — web inference ile birebir aynı format.

    Pose:      33 × 3 =  99
    Sol El:    21 × 3 =  63
    Sağ El:    21 × 3 =  63
    Toplam:    75 × 3 = 225
    """
    landmarks = []
    has_hand = False

    # Pose (33 × 3 = 99)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)

    # Sol El (21 × 3 = 63)
    if results.left_hand_landmarks:
        has_hand = True
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    # Sağ El (21 × 3 = 63)
    if results.right_hand_landmarks:
        has_hand = True
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    return np.array(landmarks, dtype=np.float32), has_hand


def draw_landmarks(image, results):
    """Landmark'ları görüntü üzerine çiz"""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )


def save_sequence(sequence, label, counter):
    """Sekansı .npy olarak kaydet — shape: (30, 225)"""
    label_path = BASE_DATA_PATH / label
    label_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{counter:03d}.npy"
    filepath = label_path / filename

    np.save(filepath, sequence)
    print(f"  ✅ {counter + 1}/{TARGET_SAMPLES} — {filepath.name} — Shape: {sequence.shape}")


def select_label():
    """Kullanıcıdan etiket seç"""
    print("\n📋 Kelime Listesi:")
    for i, w in enumerate(WORDS, 1):
        print(f"  {i}. {w}")
    print(f"  0. Özel kelime gir")

    while True:
        choice = input("\nNumara seç (1-5) veya 0: ").strip()
        if choice == "0":
            custom = input("Kelime: ").strip().upper()
            if custom:
                return custom
        elif choice.isdigit() and 1 <= int(choice) <= len(WORDS):
            return WORDS[int(choice) - 1]
        print("❌ Geçersiz seçim, tekrar dene.")


def count_existing(label):
    """Mevcut örnek sayısını say"""
    label_path = BASE_DATA_PATH / label
    if not label_path.exists():
        return 0
    return len(list(label_path.glob("*.npy")))


def main():
    print("=" * 70)
    print("  TİD VERİ TOPLAMA v2 — 225 Özellik (Pose + Eller)")
    print("  30 frame × 225 feature — Web inference uyumlu")
    print("=" * 70)
    print("\nKONTROLLER:")
    print("  S : Otomatik 50 örnek toplamaya başla")
    print("  E : Yeni etiket (kelime) seç")
    print("  Q / ESC : Çıkış")
    print("=" * 70)

    label = select_label()
    existing = count_existing(label)
    print(f"\n✅ Etiket: {label}")
    print(f"📁 Konum: {BASE_DATA_PATH / label}")
    if existing > 0:
        print(f"📂 Mevcut: {existing} örnek var, üstüne eklenecek")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Kamera açılamadı!")
        return

    # Kamera çözünürlüğünü ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:

        recording = False
        auto_mode = False
        sequence = []
        counter = existing  # Mevcut örneklerden devam et
        countdown = 0
        hand_frames_in_seq = 0

        print(f"\n🎥 Kamera açıldı! S tuşuna basarak başla.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Ayna efekti
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            landmarks, has_hand = extract_landmarks(results)

            # ── Geri sayım ──────────────────────────────────────────────
            if auto_mode and not recording and countdown > 0:
                seconds_left = countdown // FPS_ESTIMATE + 1
                cv2.putText(frame, f"BASLIYOR: {seconds_left}",
                            (frame.shape[1] // 2 - 120, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
                countdown -= 1
                if countdown <= 0:
                    recording = True
                    sequence = []
                    hand_frames_in_seq = 0

            # ── Kayıt ──────────────────────────────────────────────────
            if recording:
                sequence.append(landmarks)
                if has_hand:
                    hand_frames_in_seq += 1

                progress = len(sequence)
                color = (0, 0, 255) if has_hand else (0, 100, 255)
                cv2.putText(frame, f"KAYIT: {progress}/{MAX_FRAMES}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

                # El uyarısı
                if not has_hand:
                    cv2.putText(frame, "EL BULUNAMADI!", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 30 frame doldu
                if len(sequence) >= MAX_FRAMES:
                    seq_array = np.array(sequence, dtype=np.float32)  # (30, 225)
                    save_sequence(seq_array, label, counter)
                    counter += 1
                    recording = False
                    sequence = []
                    hand_frames_in_seq = 0

                    if auto_mode and counter < existing + TARGET_SAMPLES:
                        # Sonraki örnek: kısa bekleme sonra devam
                        recording = True
                        sequence = []
                    elif auto_mode:
                        print(f"\n🎉 TAMAMLANDI! {TARGET_SAMPLES} örnek toplandı!")
                        print(f"📁 Toplam: {counter} örnek ({BASE_DATA_PATH / label})\n")
                        auto_mode = False

            # ── Görselleştirme ──────────────────────────────────────────
            draw_landmarks(frame, results)

            # Durum bilgisi
            status_color = (0, 255, 0) if has_hand else (100, 100, 255)
            cv2.putText(frame, f"Etiket: {label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            target_total = existing + TARGET_SAMPLES if auto_mode else counter
            cv2.putText(frame, f"Ornek: {counter}/{target_total}",
                        (10, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            hand_status = "EL: OK" if has_hand else "EL: YOK"
            cv2.putText(frame, hand_status, (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            if not recording and countdown <= 0:
                cv2.putText(frame, "S:Baslat  E:Etiket  Q:Cikis",
                            (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("TID Veri Toplama v2 (225 ozellik)", frame)

            # ── Tuş kontrolleri ─────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('s'), ord('S')):
                if not auto_mode and not recording:
                    auto_mode = True
                    countdown = COUNTDOWN_SECONDS * FPS_ESTIMATE
                    print(f"\n🚀 Otomatik mod: {TARGET_SAMPLES} örnek toplanacak!")
                    print(f"   {COUNTDOWN_SECONDS} saniye sonra başlıyor...\n")

            elif key in (ord('e'), ord('E')):
                if not recording:
                    cv2.destroyAllWindows()
                    label = select_label()
                    existing = count_existing(label)
                    counter = existing
                    auto_mode = False
                    print(f"\n✅ Yeni etiket: {label}")
                    if existing > 0:
                        print(f"📂 Mevcut: {existing} örnek var")

            elif key in (ord('q'), ord('Q'), 27):
                break

    cap.release()
    cv2.destroyAllWindows()

    # Özet
    print("\n" + "=" * 70)
    print("📊 VERİ ÖZETİ:")
    print("=" * 70)
    for folder in sorted(BASE_DATA_PATH.iterdir()):
        if folder.is_dir():
            n = len(list(folder.glob("*.npy")))
            print(f"  {folder.name:15s} : {n} örnek")
    print("=" * 70)


if __name__ == "__main__":
    main()
