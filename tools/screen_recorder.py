"""
Ekran Kaydı Aracı - İşaret Dili Dataset Toplama
================================================
Sözlük siteleri veya YouTube'daki işaret dili videolarını
kaydedip MediaPipe ile keypoint extraction yapar.

Kullanım:
    python tools/screen_recorder.py

Kontroller:
    R   - Kayıt başlat/durdur
    S   - Snapshot al (tek kare keypoint kaydet)
    Q   - Çıkış
    1-9 - Kayıt bölgesi seç (önceden tanımlı)
"""

import cv2
import numpy as np
import time
import json
import threading
from pathlib import Path
from datetime import datetime

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("[UYARI] mss bulunamadı. pip install mss")

try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[UYARI] mediapipe bulunamadı.")

# ─── Ayarlar ──────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path("tid_sequence/data_screen")

# Önceden tanımlı ekran bölgeleri (x, y, genişlik, yükseklik)
REGIONS = {
    "1": {"name": "Sol yarı",        "x": 0,    "y": 0,   "w": 960,  "h": 1080},
    "2": {"name": "Sağ yarı",        "x": 960,  "y": 0,   "w": 960,  "h": 1080},
    "3": {"name": "Tam ekran",       "x": 0,    "y": 0,   "w": 1920, "h": 1080},
    "4": {"name": "Orta pencere",    "x": 400,  "y": 100, "w": 1120, "h": 880},
    "5": {"name": "YouTube oynatıcı","x": 280,  "y": 130, "w": 1360, "h": 765},
    "6": {"name": "Tarayıcı içeriği","x": 0,    "y": 80,  "w": 1920, "h": 960},
}

DISPLAY_SCALE  = 0.5    # Önizleme küçültme oranı
FPS_TARGET     = 30     # Hedef FPS
SEQUENCE_LEN   = 30     # Kayıt uzunluğu (kare)
AUTO_STOP_SEC  = 2.0    # R'ye basınca kaç saniye sonra otomatik dur (0 = kapalı)


class ScreenRecorder:
    def __init__(self):
        self.region = REGIONS["3"]  # Varsayılan: tam ekran
        self.recording = False
        self.frames = []            # Ham video kareleri
        self.keypoints_seq = []     # MediaPipe keypoint dizisi
        self.label = "LABEL"        # Kayıt etiketi
        self.save_video = True      # Video da kaydet
        self.save_keypoints = True  # Keypoint de kaydet
        self.auto_extract = True    # Kayıt bitince otomatik extract

        self.mp_holistic = None
        self.holistic = None
        self._init_mediapipe()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _init_mediapipe(self):
        if not MP_AVAILABLE:
            return
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        self.mp_draw = mp.solutions.drawing_utils

    def _extract_keypoints(self, results):
        """MediaPipe sonuçlarından 543 boyutlu keypoint vektörü çıkar."""
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                         for lm in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33 * 4)

        face = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468 * 3)

        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21 * 3)

        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21 * 3)

        return np.concatenate([pose, face, lh, rh])

    def _draw_landmarks(self, frame, results):
        """Keypoint çizimi (yalnızca eller + omuzlar)."""
        if not MP_AVAILABLE:
            return frame
        mp_draw = self.mp_draw
        mp_h   = self.mp_holistic

        if results.left_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.left_hand_landmarks,
                                   mp_h.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_draw.draw_landmarks(frame, results.right_hand_landmarks,
                                   mp_h.HAND_CONNECTIONS)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(frame, results.pose_landmarks,
                                   mp_h.POSE_CONNECTIONS)
        return frame

    def capture_loop(self):
        """Ana yakalama döngüsü — ayrı thread'de çalışır."""
        if not MSS_AVAILABLE:
            print("[HATA] mss yüklü değil, ekran yakalama yapılamıyor.")
            return

        with mss.mss() as sct:
            mon = {
                "left":   self.region["x"],
                "top":    self.region["y"],
                "width":  self.region["w"],
                "height": self.region["h"],
            }

            print(f"[Bölge] {self.region['name']} — {mon}")

            interval = 1.0 / FPS_TARGET
            t_last   = time.perf_counter()

            while self._running:
                now = time.perf_counter()
                if now - t_last < interval:
                    time.sleep(0.001)
                    continue
                t_last = now

                # Ekran yakala
                img = np.array(sct.grab(mon))          # BGRA
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                # MediaPipe
                results = None
                if MP_AVAILABLE and self.holistic:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = self.holistic.process(rgb)
                    rgb.flags.writeable = True
                    frame = self._draw_landmarks(frame, results)

                # Kayıt
                if self.recording:
                    self.frames.append(frame.copy())
                    if results is not None and MP_AVAILABLE:
                        kp = self._extract_keypoints(results)
                        self.keypoints_seq.append(kp)

                    elapsed = time.perf_counter() - self._rec_start

                    # Otomatik durdur
                    if AUTO_STOP_SEC > 0 and elapsed >= AUTO_STOP_SEC:
                        self.recording = False
                        self._auto_save = True  # ana loop kaydedecek

                    # Geri sayım HUD
                    remaining = max(0.0, AUTO_STOP_SEC - elapsed) if AUTO_STOP_SEC > 0 else elapsed
                    bar_pct   = min(1.0, elapsed / AUTO_STOP_SEC) if AUTO_STOP_SEC > 0 else 0
                    bar_w     = int(bar_pct * 300)
                    cv2.rectangle(frame, (10, 50), (310, 70), (50, 50, 50), -1)
                    cv2.rectangle(frame, (10, 50), (10 + bar_w, 70), (0, 80, 255), -1)
                    label_txt = (f"● REC  {elapsed:.1f}s  kalan: {remaining:.1f}s"
                                 if AUTO_STOP_SEC > 0 else
                                 f"● REC  {elapsed:.1f}s  [{len(self.frames)} kare]")
                    cv2.putText(frame, label_txt,
                                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (0, 0, 255), 2)
                else:
                    cv2.putText(frame,
                                f"[R] Kayit Baslat  |  Bolge: {self.region['name']}",
                                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 200, 0), 2)

                # Etiket
                cv2.putText(frame,
                            f"Etiket: {self.label}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 200, 0), 2)

                # Önizleme
                h, w = frame.shape[:2]
                preview = cv2.resize(frame, (int(w * DISPLAY_SCALE),
                                             int(h * DISPLAY_SCALE)))
                self._preview_frame = preview

    def save_recording(self):
        """Kaydı diske yaz."""
        if not self.frames:
            print("[!] Kaydedilecek kare yok.")
            return

        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / self.label
        out.mkdir(parents=True, exist_ok=True)

        # Video kaydet
        if self.save_video:
            vid_path = out / f"{ts}.mp4"
            h, w     = self.frames[0].shape[:2]
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(str(vid_path), fourcc, FPS_TARGET, (w, h))
            for f in self.frames:
                writer.write(f)
            writer.release()
            print(f"[Kayıt] Video: {vid_path}  ({len(self.frames)} kare)")

        # Keypoint .npy kaydet
        if self.save_keypoints and self.keypoints_seq:
            seq = np.array(self.keypoints_seq)  # (T, 1662)
            # Pad / crop → SEQUENCE_LEN
            if len(seq) >= SEQUENCE_LEN:
                seq = seq[:SEQUENCE_LEN]
            else:
                pad = np.zeros((SEQUENCE_LEN - len(seq), seq.shape[1]))
                seq = np.vstack([seq, pad])

            idx  = len(list(out.glob("*.npy")))
            npy_path = out / f"{ts}_{idx:03d}.npy"
            np.save(str(npy_path), seq)
            print(f"[Kayıt] Keypoint: {npy_path}  shape={seq.shape}")

        self.frames = []
        self.keypoints_seq = []

    def run(self):
        self._running = True
        self._preview_frame = None
        self._rec_start = 0.0
        self._auto_save = False

        # Yakalama thread'i
        t = threading.Thread(target=self.capture_loop, daemon=True)
        t.start()

        print("\n══════════════════════════════════════════")
        print("  Ekran Kaydedici — İşaret Dili Dataset")
        print("══════════════════════════════════════════")
        print("  R      → Kayıt başlat / durdur")
        print("  E      → Etiketi değiştir")
        print("  1-6    → Ekran bölgesi seç")
        print("  Q/ESC  → Çıkış")
        print("══════════════════════════════════════════\n")
        print(f"Geçerli etiket: {self.label}")
        print("Etiket girmek için terminale bakın.\n")

        while True:
            # Önizleme göster
            if self._preview_frame is not None:
                cv2.imshow("Ekran Kaydedici", self._preview_frame)

            # Otomatik kayıt tamamlandı mı?
            if self._auto_save:
                self._auto_save = False
                print("[■] Otomatik durduruldu — kaydediliyor...")
                self.save_recording()

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

            elif key == ord("r"):
                if not self.recording:
                    self.recording  = True
                    self._rec_start = time.perf_counter()
                    print(f"[●] Kayıt başladı  ({self.label})")
                else:
                    self.recording = False
                    print("[■] Kayıt durdu — kaydediliyor...")
                    self.save_recording()

            elif key == ord("e"):
                # Etiket değiştir (pencereyi küçült, terminale yaz)
                cv2.destroyAllWindows()
                new_label = input("Yeni etiket girin (örn: MERHABA): ").strip().upper()
                if new_label:
                    self.label = new_label
                    print(f"[Etiket] → {self.label}")
                cv2.namedWindow("Ekran Kaydedici")

            elif chr(key) in REGIONS if key != 255 else False:
                region_key = chr(key)
                if region_key in REGIONS:
                    self.region = REGIONS[region_key]
                    print(f"[Bölge] → {self.region['name']}")

        self._running = False
        t.join(timeout=2)
        cv2.destroyAllWindows()
        print("Çıkış.")


# ─── YouTube / Web yardımcı fonksiyonlar ─────────────────────────────────────

def detect_label_from_title(window_title: str) -> str:
    """
    Tarayıcı pencere başlığından otomatik etiket çıkar.
    Örn: 'MERHABA - TİD Sözlük' → 'MERHABA'
    """
    known = [
        "MERHABA", "EVET", "HAYIR", "LÜTFEN", "TEŞEKKÜR",
        "SU", "YEMEK", "ANNE", "BABA", "EV", "OKUL",
    ]
    title_up = window_title.upper()
    for word in known:
        if word in title_up:
            return word
    return "UNKNOWN"


def extract_keypoints_from_video(video_path: str, label: str,
                                  out_dir: Path = OUTPUT_DIR):
    """
    Kaydedilmiş bir video dosyasından keypoint çıkarır.
    Zaten ekran kaydı yapıldıysa bu fonksiyon çalışır.
    """
    if not MP_AVAILABLE:
        print("[HATA] mediapipe yüklü değil.")
        return

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[HATA] Video açılamadı: {video_path}")
        return

    out = out_dir / label
    out.mkdir(parents=True, exist_ok=True)

    sequence = []
    frame_count = 0

    with mp_holistic.Holistic(min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility]
                              for lm in results.pose_landmarks.landmark]).flatten() \
                   if results.pose_landmarks else np.zeros(33 * 4)
            face = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.face_landmarks.landmark]).flatten() \
                   if results.face_landmarks else np.zeros(468 * 3)
            lh   = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.left_hand_landmarks.landmark]).flatten() \
                   if results.left_hand_landmarks else np.zeros(21 * 3)
            rh   = np.array([[lm.x, lm.y, lm.z]
                              for lm in results.right_hand_landmarks.landmark]).flatten() \
                   if results.right_hand_landmarks else np.zeros(21 * 3)

            kp = np.concatenate([pose, face, lh, rh])
            sequence.append(kp)

            # Her SEQUENCE_LEN karede bir kaydet (kayan pencere)
            if len(sequence) == SEQUENCE_LEN:
                idx = len(list(out.glob("*.npy")))
                ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = out / f"{ts}_{idx:03d}.npy"
                np.save(str(path), np.array(sequence))
                print(f"[Extract] {path}")
                sequence = sequence[10:]  # 10 kare ileri kaydır (overlap)

    cap.release()
    print(f"[Tamam] {frame_count} kare işlendi. Çıktı: {out}")


def batch_extract(video_folder: str, out_dir: Path = OUTPUT_DIR):
    """
    Bir klasördeki tüm videoları işle.
    Klasör adı etiket olarak kullanılır.
    Örn: videos/MERHABA/clip1.mp4 → etiket: MERHABA
    """
    folder = Path(video_folder)
    videos = list(folder.rglob("*.mp4")) + list(folder.rglob("*.avi")) + \
             list(folder.rglob("*.mov")) + list(folder.rglob("*.mkv"))

    print(f"[Batch] {len(videos)} video bulundu.")
    for vid in videos:
        label = vid.parent.name.upper()
        print(f"\n→ {vid.name}  [{label}]")
        extract_keypoints_from_video(str(vid), label, out_dir)


# ─── Ana giriş ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]

        if cmd == "extract" and len(sys.argv) >= 4:
            # python screen_recorder.py extract video.mp4 MERHABA
            extract_keypoints_from_video(sys.argv[2], sys.argv[3])

        elif cmd == "batch" and len(sys.argv) >= 3:
            # python screen_recorder.py batch /path/to/videos/
            batch_extract(sys.argv[2])

        else:
            print("Kullanım:")
            print("  python screen_recorder.py                        # Canlı kayıt")
            print("  python screen_recorder.py extract vid.mp4 LABEL  # Video → keypoint")
            print("  python screen_recorder.py batch /videos/          # Toplu işlem")
    else:
        # Canlı ekran kaydı modu
        recorder = ScreenRecorder()
        label = input("Başlangıç etiketi (BÜYÜK HARF, örn: MERHABA): ").strip().upper()
        if label:
            recorder.label = label
        recorder.run()
