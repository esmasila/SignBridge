"""Ekran goruntulerine telefon/laptop cercevesi ekle - Canva icin hazir PNG'ler."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path

IMG = Path(r"C:/Users/leven/Downloads/poster-images")
OUT = Path(r"C:/Users/leven/Downloads/poster-images/framed")
OUT.mkdir(exist_ok=True)

def round_corners(img, radius):
    """Resmi yuvarlak koselerle maskele."""
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, img.size[0], img.size[1]), radius=radius, fill=255)
    img = img.convert("RGBA")
    img.putalpha(mask)
    return img

def make_phone_mockup(screenshot_path, out_path):
    """Telefon cercevesi: koyu bezel + yuvarlak koseler + notch + golge."""
    ss = Image.open(screenshot_path).convert("RGBA")
    sw, sh = ss.size
    # Bezel kalinligi (piksel)
    bezel = max(int(sw * 0.04), 30)
    bezel_top = bezel + 10  # ust biraz daha kalin (notch icin)
    bezel_bot = bezel + 10

    # Toplam telefon boyutu
    pw = sw + 2 * bezel
    ph = sh + bezel_top + bezel_bot

    # Outer canvas - shadow icin biraz buyuk
    shadow_pad = 60
    cw = pw + 2 * shadow_pad
    ch = ph + 2 * shadow_pad

    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))

    # Golge (saydam siyah, blur)
    shadow_layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_layer)
    sd.rounded_rectangle(
        (shadow_pad + 20, shadow_pad + 30, shadow_pad + pw + 10, shadow_pad + ph + 30),
        radius=80, fill=(0, 0, 0, 100)
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=20))
    canvas = Image.alpha_composite(canvas, shadow_layer)

    # Telefon govdesi (koyu bezel)
    body = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    bd.rounded_rectangle((0, 0, pw, ph), radius=80, fill=(20, 20, 30, 255))
    # Iç ekran alani (cikartilacak)
    inner_radius = 50

    # Notch (ust ortada kucuk siyah cizgi)
    notch_w = int(pw * 0.32)
    notch_h = 22
    nx = (pw - notch_w) // 2
    ny = 15
    bd.rounded_rectangle((nx, ny, nx + notch_w, ny + notch_h),
                         radius=12, fill=(8, 8, 12, 255))

    canvas.paste(body, (shadow_pad, shadow_pad), body)

    # Ekran goruntusunu yuvarlak koselerle ekle
    ss_rounded = round_corners(ss, inner_radius)
    canvas.paste(ss_rounded,
                 (shadow_pad + bezel, shadow_pad + bezel_top),
                 ss_rounded)

    canvas.save(out_path, "PNG")
    print(f"  ✓ {out_path.name}")

def make_browser_mockup(screenshot_path, out_path):
    """Tarayici cercevesi: ust bar + url bar + 3 nokta + golge."""
    ss = Image.open(screenshot_path).convert("RGBA")
    sw, sh = ss.size
    pad = 8         # ic padding
    top_bar = 90    # tarayici ust bar yuksekligi
    shadow_pad = 80

    pw = sw + 2 * pad
    ph = sh + top_bar + pad

    cw = pw + 2 * shadow_pad
    ch = ph + 2 * shadow_pad

    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))

    # Golge
    shadow_layer = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow_layer)
    sd.rounded_rectangle(
        (shadow_pad + 25, shadow_pad + 35, shadow_pad + pw + 15, shadow_pad + ph + 35),
        radius=30, fill=(0, 0, 0, 90)
    )
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=22))
    canvas = Image.alpha_composite(canvas, shadow_layer)

    # Tarayici govdesi - acik gri arka plan
    body = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    bd = ImageDraw.Draw(body)
    bd.rounded_rectangle((0, 0, pw, ph), radius=24, fill=(245, 245, 250, 255))

    # Ust bar (acik gri)
    bd.rounded_rectangle((0, 0, pw, top_bar), radius=24, fill=(232, 232, 240, 255))
    # Alt kose duzeltme (top_bar ile govde birlesimi)
    bd.rectangle((0, top_bar - 24, pw, top_bar), fill=(232, 232, 240, 255))

    # 3 nokta (kapatma butonlari) - sol ust
    circle_y = top_bar // 2
    circle_r = 14
    circle_x_start = 36
    colors = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]  # kirmizi, sari, yesil
    for i, c in enumerate(colors):
        cx = circle_x_start + i * (circle_r * 2 + 12)
        bd.ellipse((cx - circle_r, circle_y - circle_r,
                    cx + circle_r, circle_y + circle_r), fill=c + (255,))

    # URL bar (ortada)
    url_w = int(pw * 0.55)
    url_x = (pw - url_w) // 2
    url_y = circle_y - 22
    url_h = 44
    bd.rounded_rectangle((url_x, url_y, url_x + url_w, url_y + url_h),
                         radius=22, fill=(255, 255, 255, 255))
    # Lock icon
    lock_size = 20
    bd.ellipse((url_x + 16, url_y + url_h//2 - lock_size//2,
                url_x + 16 + lock_size, url_y + url_h//2 + lock_size//2),
               fill=(150, 150, 150, 255))
    # Fake URL text - draw simple line (yazi yuklemek karmasik)
    bd.rectangle((url_x + 50, circle_y - 4, url_x + url_w - 30, circle_y + 4),
                 fill=(200, 200, 200, 255))

    canvas.paste(body, (shadow_pad, shadow_pad), body)

    # Ekran goruntusu
    ss_inner = ss
    canvas.paste(ss_inner,
                 (shadow_pad + pad, shadow_pad + top_bar),
                 ss_inner)

    canvas.save(out_path, "PNG")
    print(f"  ✓ {out_path.name}")

# Mobil ekran goruntuleri -> telefon cercevesi
print("📱 Telefon mockup'lari olusturuluyor:")
mobile_files = list(IMG.glob("mobil_*.jpeg"))
for f in mobile_files:
    out_name = "phone_" + f.stem + ".png"
    make_phone_mockup(f, OUT / out_name)

# Web ekran goruntuleri -> tarayici cercevesi
print("\n💻 Tarayici mockup'lari olusturuluyor:")
web_files = list(IMG.glob("web_*.jpeg"))
for f in web_files:
    out_name = "browser_" + f.stem + ".png"
    make_browser_mockup(f, OUT / out_name)

print(f"\n✅ Tamamlandi: {OUT}")
print(f"Toplam: {len(list(OUT.glob('*.png')))} dosya")
