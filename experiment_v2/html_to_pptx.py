"""HTML posteri tek slide'lık 70x100cm pptx'e çevir.
Yöntem: Playwright ile HTML'i 70x100cm @ 300dpi PNG'ye render, sonra pptx'e tam slide gömü."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from playwright.sync_api import sync_playwright
from pptx import Presentation
from pptx.util import Cm
from pathlib import Path

HTML = Path(r"C:/Users/leven/Downloads/poster_design.html")
PNG  = Path(r"C:/Users/leven/Downloads/poster_design_full.png")
PPTX = Path(r"C:/Users/leven/Downloads/SignBridge_poster.pptx")

# 70x100cm @ 96dpi = 2645 x 3780 px, daha yüksek için 2x = 5290 x 7559
W_CM, H_CM = 70, 100
DPI_SCALE = 2.5  # 96 * 2.5 ≈ 240 dpi - matbaaya yetecek netlik
W_PX = int(W_CM * 37.795 * DPI_SCALE)
H_PX = int(H_CM * 37.795 * DPI_SCALE)
print(f"Render boyutu: {W_PX} x {H_PX} px ({W_CM}x{H_CM}cm @ {int(96*DPI_SCALE)}dpi)")

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={'width': W_PX, 'height': H_PX},
                            device_scale_factor=1)
    page.goto(f'file:///{HTML.as_posix()}')
    # Scale 1.0 ve marginBottom kaldır - poster gerçek boyutta
    page.evaluate('document.getElementById("poster").style.transform="scale(1)"')
    page.evaluate('document.getElementById("poster").style.marginBottom="0"')
    page.evaluate('document.body.style.background="white"')
    page.evaluate('document.body.style.padding="0"')
    # Toolbar'ı gizle (önizleme amaçlı)
    page.evaluate('const t=document.querySelector(".toolbar"); if(t) t.style.display="none";')
    page.wait_for_timeout(2000)
    # Poster elementinin tam boyutunda screenshot al
    poster = page.locator('#poster')
    poster.screenshot(path=str(PNG))
    browser.close()

print(f"PNG kaydedildi: {PNG}")

# Şimdi pptx oluştur: tek slide 70x100cm, içinde tam boy PNG
prs = Presentation()
prs.slide_width  = Cm(W_CM)
prs.slide_height = Cm(H_CM)
blank = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank)
slide.shapes.add_picture(str(PNG), 0, 0, Cm(W_CM), Cm(H_CM))
prs.save(str(PPTX))
print(f"PPTX kaydedildi: {PPTX}")
print(f"Slide boyutu: {W_CM}x{H_CM} cm")
