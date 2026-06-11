const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
        ShadingType, PageNumber, PageBreak, LevelFormat, ImageRun } = require('docx');
const fs = require('fs');
const path = require('path');

const IMG_DIR = path.join(__dirname, 'report_images');

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const headerShading = { fill: "D5E8F0", type: ShadingType.CLEAR };
const cellMargins = { top: 80, bottom: 80, left: 120, right: 120 };

// Sabit satır aralığı (line spacing) - farklı PC'lerde kayma olmasın
const LINE_SPACING = { line: 276, lineRule: "auto" }; // 1.15 satır aralığı (276 twips = 1.15x)
const FIXED_FONT = { font: "Calibri", size: 22 }; // 11pt Calibri her yerde aynı

function headerCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: headerShading, margins: cellMargins,
    children: [new Paragraph({ spacing: { ...LINE_SPACING, after: 0 }, children: [new TextRun({ text, bold: true, ...FIXED_FONT })] })]
  });
}
function cell(text, width, bold) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, margins: cellMargins,
    children: [new Paragraph({ spacing: { ...LINE_SPACING, after: 0 }, children: [new TextRun({ text, bold: bold || false, ...FIXED_FONT })] })]
  });
}
function bold(text) { return new TextRun({ text, bold: true, ...FIXED_FONT }); }
function normal(text) { return new TextRun({ text, ...FIXED_FONT }); }
function italic(text) { return new TextRun({ text, italics: true, color: "808080", ...FIXED_FONT }); }
function heading(text) {
  return new Paragraph({ spacing: { ...LINE_SPACING, before: 300, after: 150 },
    children: [new TextRun({ text, bold: true, font: "Calibri", size: 28 })] });
}
function subheading(text) {
  return new Paragraph({ spacing: { ...LINE_SPACING, before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, font: "Calibri", size: 24 })] });
}
function p(...runs) { return new Paragraph({ spacing: { ...LINE_SPACING, after: 80 }, children: runs }); }
function bullet(text) { return p(normal("\u2022 " + text)); }
function emptyP() { return new Paragraph({ spacing: { ...LINE_SPACING } }); }
function screenshotPlaceholder(text) {
  return new Paragraph({ spacing: { before: 100, after: 100 },
    children: [italic("[EKRAN G\u00D6R\u00DCN\u00DCT\u00DCS\u00DC: " + text + "]")]
  });
}
function tryImage(filename, caption, widthPx, heightPx) {
  // Try multiple extensions
  const exts = ['', '.png', '.jpeg', '.jpg'];
  let filePath = null;
  for (const ext of exts) {
    const tryPath = path.join(IMG_DIR, filename + ext);
    if (fs.existsSync(tryPath)) { filePath = tryPath; break; }
    // Also try without extension if filename already has one
    if (ext === '' && fs.existsSync(path.join(IMG_DIR, filename))) { filePath = path.join(IMG_DIR, filename); break; }
  }
  // Try swapping extension
  if (!filePath) {
    const base = filename.replace(/\.\w+$/, '');
    for (const ext of ['.png', '.jpeg', '.jpg']) {
      const tryPath = path.join(IMG_DIR, base + ext);
      if (fs.existsSync(tryPath)) { filePath = tryPath; break; }
    }
  }
  if (!filePath) {
    console.warn("  EKSIK: " + filename + " -> placeholder kalacak");
    return [screenshotPlaceholder(caption)];
  }
  const imgData = fs.readFileSync(filePath);
  const isJpeg = filePath.endsWith('.jpeg') || filePath.endsWith('.jpg');
  console.log("  EKLENDI: " + path.basename(filePath) + " (" + (imgData.length/1024).toFixed(0) + " KB)");
  return [
    new Paragraph({ spacing: { before: 150, after: 50 }, alignment: AlignmentType.CENTER,
      children: [new ImageRun({ data: imgData, transformation: { width: widthPx, height: heightPx }, type: isJpeg ? 'jpg' : 'png' })]
    }),
    new Paragraph({ spacing: { after: 150 }, alignment: AlignmentType.CENTER,
      children: [italic(caption)]
    }),
  ];
}
function separator() {
  return new Paragraph({ spacing: { before: 200, after: 200 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "A0A0A0", space: 1 } },
    children: [] });
}

// ========== KISIM 1 ==========
const kisim1 = [
  new Paragraph({ alignment: AlignmentType.CENTER, children: [bold("B\u0130LG\u0130SAYAR M\u00DCHEND\u0130SL\u0130\u011E\u0130")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [bold("B\u0130LG\u0130SAYAR TASARIM VE UYGULAMALARI DERS\u0130")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [bold("PROJE RAPORU (Geni\u015Fletilmi\u015F)")] }),
  emptyP(),
  p(bold("\u00D6\u011Fretim Y\u0131l\u0131 ve D\u00F6nemi")),
  p(bold("2025\u20132026 G\u00FCz + Bahar Yar\u0131y\u0131l\u0131")),
  emptyP(),
  p(bold("Proje Sahibi")),
  p(bold("Esma S\u0131la \u015EAH\u0130NC\u0130 | 22260810048 | esmasilasahincii@gmail.com")),
  emptyP(),
  p(bold("Teslim Tarihi")),
  p(bold("26.03.2026")),
  emptyP(),
  p(bold("Proje Ba\u015Fl\u0131\u011F\u0131")),
  p(bold("SignBridge \u2013 T\u00FCrk \u0130\u015Faret Dili (T\u0130D) Ger\u00E7ek Zamanl\u0131 Yapay Zek\u00E2 Destekli \u00C7eviri Uygulamas\u0131")),
  separator(),
  heading("Proje \u00D6zeti"),
  p(normal("SignBridge, T\u00FCrk \u0130\u015Faret Dili (T\u0130D) kullanan bireylerle i\u015Faret dili bilmeyenler aras\u0131nda ger\u00E7ek zamanl\u0131, \u00E7ift y\u00F6nl\u00FC (T\u0130D\u2192Metin/Ses ve Metin\u2192T\u0130D Avatar) bir ileti\u015Fim k\u00F6pr\u00FCs\u00FC sunan web ve mobil tabanl\u0131 bir yapay zeka uygulamas\u0131d\u0131r. \u00C7\u00F6z\u00FCm, mahremiyet ve d\u00FC\u015F\u00FCk gecikme i\u00E7in on-device (cihaz-\u00FCst\u00FC) i\u015Flemeyi tercih eder. Sistem T\u0130D\u2019e \u00F6zg\u00FC tasarlanm\u0131\u015Ft\u0131r.")),
  separator(),
  heading("KISIM 1: G\u00DCZ D\u00D6NEM\u0130 \u00C7ALI\u015EMALARI"),

  subheading("1.1 Problem Tan\u0131m\u0131 ve Motivasyon"),
  p(normal("\u0130\u015Faret dilleri yaln\u0131z manuel (el/kol) bile\u015Fenlerden de\u011Fil; dilbilgisel ve ezgisel i\u015Flevler ta\u015F\u0131yan non-manual (a\u011F\u0131z, ba\u015F/g\u00F6vde) ipu\u00E7lar\u0131ndan olu\u015Fan \u00E7ok kanall\u0131 bir yap\u0131ya sahiptir. Her \u00FClkenin i\u015Faret dili farkl\u0131d\u0131r; T\u0130D (T\u00FCrk \u0130\u015Faret Dili), ASL (Amerikan \u0130\u015Faret Dili) veya DGS (Alman \u0130\u015Faret Dili) ile bire bir \u00F6rt\u00FC\u015Fmez. Bu nedenle uluslararas\u0131 veri setleri ve modeller do\u011Frudan T\u0130D i\u00E7in kullan\u0131lamamaktad\u0131r.")),
  p(normal("\u0130\u015Faret dillerine eri\u015Fim, BM Engelli Haklar\u0131 S\u00F6zle\u015Fmesi (CRPD) taraf\u0131ndan vurgulanan ve T\u00FCrkiye\u2019de 5378 say\u0131l\u0131 Kanun ile kamusal hizmetlerde tan\u0131nan evrensel bir hakt\u0131r. T\u00FCrkiye\u2019de yakla\u015F\u0131k 3 milyon i\u015Fitme engelli birey bulunmakta olup, bu bireylerin g\u00FCnl\u00FCk ya\u015Famda ileti\u015Fim kurabilmeleri i\u00E7in teknolojik \u00E7\u00F6z\u00FCmlere ihtiya\u00E7 duyulmaktad\u0131r.")),
  emptyP(),
  p(bold("Tespit Edilen Problemler:")),
  bullet("Mevcut uygulamalarda T\u00FCrk \u0130\u015Faret Dili deste\u011Fi yetersizdir; \u00E7o\u011Fu uygulama ASL odakl\u0131d\u0131r"),
  bullet("T\u00FCrkiye\u2019deki \u00E7\u00F6z\u00FCmler \u00E7o\u011Funlukla Metin\u2192T\u0130D y\u00F6n\u00FCne yo\u011Funla\u015F\u0131r; T\u0130D\u2192Metin/Ses y\u00F6n\u00FC zay\u0131f kal\u0131r"),
  bullet("Bulut tabanl\u0131 mimariler gecikme olu\u015Fturur; ham g\u00F6r\u00FCnt\u00FC g\u00F6nderimi mahremiyet endi\u015Fesi yarat\u0131r"),
  bullet("T\u0130D\u2019e \u00F6zg\u00FC a\u00E7\u0131k kaynak veri seti say\u0131s\u0131 olduk\u00E7a s\u0131n\u0131rl\u0131d\u0131r"),
  bullet("\u00C7ift y\u00F6nl\u00FC (T\u0130D\u2192Metin ve Metin\u2192T\u0130D) ileti\u015Fim sunan bir T\u00FCrk\u00E7e uygulama bulunmamaktad\u0131r"),
  emptyP(),

  subheading("1.2 Geli\u015Ftirilen Modeller"),
  p(normal("G\u00FCz d\u00F6neminde proje kapsam\u0131nda "), bold("iki ayr\u0131 derin \u00F6\u011Frenme modeli"), normal(" ba\u015Far\u0131yla geli\u015Ftirilmi\u015F ve production ortam\u0131na al\u0131nm\u0131\u015Ft\u0131r:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [1800, 3000, 2400, 2160],
    rows: [
      new TableRow({ children: [headerCell("Model", 1800), headerCell("Hedef", 3000), headerCell("Mimari", 2400), headerCell("Do\u011Fruluk", 2160)] }),
      new TableRow({ children: [cell("Model 1", 1800, true), cell("Rakam Tan\u0131ma (0-9)", 3000), cell("Multimodal CNN", 2400), cell("%99.79", 2160)] }),
      new TableRow({ children: [cell("Model 2", 1800, true), cell("Kelime Tan\u0131ma (20 kelime)", 3000), cell("YOLOv11-nano", 2400), cell("%99.6 mAP50", 2160)] }),
    ]
  }),
  emptyP(),
  p(bold("Model 1 \u2013 Multimodal CNN (Rakam Tan\u0131ma):")),
  p(normal("MediaPipe Hands ile ger\u00E7ek zamanl\u0131 el landmark\u2019lar\u0131 \u00E7\u0131kar\u0131lm\u0131\u015F, bu landmark verisi ile ham g\u00F6r\u00FCnt\u00FC birle\u015Ftirilerek \u00E7ift kanall\u0131 (multimodal) bir CNN mimarisi olu\u015Fturulmu\u015Ftur. Model, 0\u20139 aras\u0131 T\u0130D rakam i\u015Faretlerini %99.79 do\u011Fruluk ile tan\u0131maktad\u0131r. E\u011Fitim s\u00FCrecinde cross-entropy loss fonksiyonu ve Adam optimizer kullan\u0131lm\u0131\u015Ft\u0131r.")),
  emptyP(),
  p(normal("CNN mimarisi \u00E7ift kanall\u0131 (dual-branch) bir yap\u0131 kullanmaktad\u0131r. Birinci kanal, 64\u00D764 boyutlu RGB g\u00F6r\u00FCnt\u00FCy\u00FC 4 adet Conv2D + MaxPool katman\u0131ndan ge\u00E7irir (BatchNorm ve ReLU aktivasyon ile, Dropout=0.3). \u0130kinci kanal, MediaPipe\u2019tan elde edilen 1629 boyutlu landmark vekt\u00F6r\u00FCn\u00FC 3 adet Dense (tam ba\u011Flant\u0131l\u0131) katman\u0131ndan ge\u00E7irir (BatchNorm + ReLU + Dropout). \u0130ki kanal\u0131n \u00E7\u0131kt\u0131lar\u0131 fusion (concatenate) katman\u0131nda birle\u015Ftirilir ve son classifier katman\u0131 Softmax(10) ile 0-9 aras\u0131 s\u0131n\u0131fland\u0131rma yapar.")),
  emptyP(),
  p(bold("Model 1 Mimarisi Detay Tablosu:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Parametre", 3120), headerCell("De\u011Fer", 6240)] }),
      new TableRow({ children: [cell("Girdi 1 (G\u00F6r\u00FCnt\u00FC)", 3120), cell("64\u00D764\u00D73 RGB g\u00F6r\u00FCnt\u00FC", 6240)] }),
      new TableRow({ children: [cell("Girdi 2 (Landmark)", 3120), cell("1629 boyutlu vekt\u00F6r (543 landmark \u00D7 3)", 6240)] }),
      new TableRow({ children: [cell("CNN Kanal\u0131", 3120), cell("4\u00D7Conv2D + MaxPool + BatchNorm + ReLU", 6240)] }),
      new TableRow({ children: [cell("MLP Kanal\u0131", 3120), cell("3\u00D7Dense Layer + BatchNorm + ReLU", 6240)] }),
      new TableRow({ children: [cell("Fusion", 3120), cell("Concatenate (iki kanal birle\u015Ftirme)", 6240)] }),
      new TableRow({ children: [cell("Dropout", 3120), cell("0.3", 6240)] }),
      new TableRow({ children: [cell("Classifier", 3120), cell("Softmax(10)", 6240)] }),
      new TableRow({ children: [cell("Optimizer", 3120), cell("Adam", 6240)] }),
      new TableRow({ children: [cell("Loss", 3120), cell("Cross-Entropy", 6240)] }),
      new TableRow({ children: [cell("Test Accuracy", 3120), cell("%99.79", 6240)] }),
    ]
  }),
  emptyP(),
  p(bold("Model 2 \u2013 YOLOv11-nano (Kelime Tan\u0131ma):")),
  p(normal("Ultralytics YOLOv11-nano mimarisi ile 20 farkl\u0131 T\u0130D kelimesi i\u00E7in object detection modeli e\u011Fitilmi\u015Ftir. Roboflow platformu \u00FCzerinden etiketlenen veri seti kullan\u0131lm\u0131\u015Ft\u0131r. Model, %99.6 mAP50 de\u011Ferine ula\u015Fm\u0131\u015F ve ger\u00E7ek zamanl\u0131 inference i\u00E7in optimize edilmi\u015Ftir. YOLO yakla\u015F\u0131m\u0131, her frame\u2019de ba\u011F\u0131ms\u0131z olarak nesne tespiti yapmas\u0131 nedeniyle statik pozlar i\u00E7in ba\u015Far\u0131l\u0131 olmu\u015F ancak dinamik hareketlerin temporal \u00F6r\u00FCnt\u00FClerini yakalayamam\u0131\u015Ft\u0131r.")),
  emptyP(),
  p(normal("YOLO modeli, tek ge\u00E7i\u015Fli (single-pass) nesne tespiti yakla\u015F\u0131m\u0131 kullanarak h\u0131zl\u0131 inference sa\u011Flamaktad\u0131r. Nano varyant\u0131 se\u00E7ilmesinin sebebi, d\u00FC\u015F\u00FCk hesaplama maliyeti ile ger\u00E7ek zamanl\u0131 (\u226525 FPS) \u00E7al\u0131\u015Fabilmesidir. E\u011Fitim s\u00FCrecinde transfer learning uygulanm\u0131\u015F ve COCO \u00F6n-e\u011Fitimli a\u011F\u0131rl\u0131klar kullan\u0131lm\u0131\u015Ft\u0131r.")),
  emptyP(),
  p(bold("YOLO Taraf\u0131ndan Tan\u0131nan 20 Kelime (G\u00FCz D\u00F6nemi):")),
  p(normal("Anne, Arkadas, Baba, Dur, Ev, Evet, Hayir, Kardes, Merhaba, Nasil, Nerede, Ozur-Dilemek, Tamam, Telefon, Tesekkurler, Tuvalet, Yemek, icmek, iyi, kotu")),
  emptyP(),

  subheading("1.3 Denenen Ancak Terk Edilen Yakla\u015F\u0131mlar"),
  p(normal("Proje s\u00FCrecinde her yakla\u015F\u0131m dikkatle de\u011Ferlendirilmi\u015F, ba\u015Far\u0131s\u0131z olan y\u00F6ntemler analiz edilerek alternatiflere ge\u00E7ilmi\u015Ftir:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [2400, 3480, 3480],
    rows: [
      new TableRow({ children: [headerCell("Yakla\u015F\u0131m", 2400), headerCell("Sonu\u00E7", 3480), headerCell("Terk Sebebi", 3480)] }),
      new TableRow({ children: [cell("Alfabe Tan\u0131ma (A-Z)", 2400), cell("%99.83 test accuracy", 3480), cell("Ger\u00E7ek d\u00FCnyada ba\u015Far\u0131s\u0131z (overfitting)", 3480)] }),
      new TableRow({ children: [cell("LSTM Sequence (5 kelime)", 2400), cell("%62 val accuracy", 3480), cell("Yetersiz veri (250 sample), class imbalance", 3480)] }),
    ]
  }),
  emptyP(),
  p(bold("Alfabe Tan\u0131ma Deneyimi: "), normal("A-Z harflerini tan\u0131yan model, test setinde %99.83 do\u011Fruluk g\u00F6stermi\u015F ancak ger\u00E7ek d\u00FCnyada (farkl\u0131 \u0131\u015F\u0131k ko\u015Fullar\u0131, farkl\u0131 eller, farkl\u0131 arka planlar) ciddi performans d\u00FC\u015F\u00FC\u015F\u00FC ya\u015Fanm\u0131\u015Ft\u0131r. Bu durum, e\u011Fitim verisinin \u00E7e\u015Fitlili\u011Finin yetersiz olmas\u0131ndan (overfitting) kaynaklanm\u0131\u015Ft\u0131r.")),
  p(bold("LSTM Sequence Deneyimi: "), normal("5 kelime i\u00E7in LSTM tabanl\u0131 sequence modeli denenmi\u015Ftir. Ancak her kelime i\u00E7in yaln\u0131zca 50 sample (toplam 250 sequence) toplanabilmi\u015F ve bu durum ciddi class imbalance sorununa yol a\u00E7m\u0131\u015Ft\u0131r. Model %62 validation accuracy\u2019de kalm\u0131\u015Ft\u0131r. Bu deneyim, bahar d\u00F6neminde daha kapsaml\u0131 veri toplama stratejisinin belirlenmesinde yol g\u00F6sterici olmu\u015Ftur.")),
  emptyP(),

  subheading("1.4 Veri Seti (G\u00FCz D\u00F6nemi)"),
  p(normal("G\u00FCz d\u00F6neminde YOLO modeli e\u011Fitimi i\u00E7in Roboflow platformu \u00FCzerinden etiketlenmi\u015F kapsaml\u0131 bir veri seti olu\u015Fturulmu\u015Ftur:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Parametre", 3120), headerCell("De\u011Fer", 6240)] }),
      new TableRow({ children: [cell("Platform", 3120), cell("Roboflow (etiketleme ve augmentation)", 6240)] }),
      new TableRow({ children: [cell("Toplam G\u00F6r\u00FCnt\u00FC", 3120), cell("21,928 etiketli g\u00F6r\u00FCnt\u00FC", 6240)] }),
      new TableRow({ children: [cell("S\u0131n\u0131f Say\u0131s\u0131", 3120), cell("20 T\u0130D kelimesi + background", 6240)] }),
      new TableRow({ children: [cell("E\u011Fitim/Val/Test", 3120), cell("%70 / %20 / %10 split", 6240)] }),
      new TableRow({ children: [cell("Etiketleme Format\u0131", 3120), cell("YOLO bounding box (x_center, y_center, w, h)", 6240)] }),
      new TableRow({ children: [cell("S\u0131n\u0131f Ba\u015F\u0131na Ortalama", 3120), cell("~1096 g\u00F6r\u00FCnt\u00FC", 6240)] }),
    ]
  }),
  p(normal("Veri seti, farkl\u0131 a\u00E7\u0131lardan, farkl\u0131 \u0131\u015F\u0131k ko\u015Fullar\u0131nda ve farkl\u0131 arka planlarda \u00E7ekilmi\u015F g\u00F6r\u00FCnt\u00FClerden olu\u015Fmaktad\u0131r. Roboflow \u00FCzerinde augmentation (d\u00F6nd\u00FCrme, \u00E7evirme, parlakl\u0131k de\u011Fi\u015Ftirme) uygulanarak veri \u00E7e\u015Fitlili\u011Fi art\u0131r\u0131lm\u0131\u015Ft\u0131r.")),
  emptyP(),

  subheading("1.5 G\u00FCz D\u00F6nemi API ve Backend"),
  p(normal("G\u00FCz d\u00F6neminde FastAPI tabanl\u0131 bir REST API geli\u015Ftirilmi\u015Ftir. Uvicorn ASGI sunucusu ile async i\u015Fleme deste\u011Fi sa\u011Flanm\u0131\u015Ft\u0131r.")),
  emptyP(),
  p(bold("REST Endpoints:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [1560, 2340, 5460],
    rows: [
      new TableRow({ children: [headerCell("Metod", 1560), headerCell("Endpoint", 2340), headerCell("A\u00E7\u0131klama", 5460)] }),
      new TableRow({ children: [cell("GET", 1560), cell("/", 2340), cell("Web UI ana sayfa", 5460)] }),
      new TableRow({ children: [cell("GET", 1560), cell("/health", 2340), cell("Sa\u011Fl\u0131k kontrol\u00FC (health check)", 5460)] }),
      new TableRow({ children: [cell("GET", 1560), cell("/model/info", 2340), cell("Model bilgileri (s\u0131n\u0131f say\u0131s\u0131, tip)", 5460)] }),
      new TableRow({ children: [cell("POST", 1560), cell("/predict", 2340), cell("Rakam tan\u0131ma (0-9)", 5460)] }),
      new TableRow({ children: [cell("POST", 1560), cell("/predict_sign", 2340), cell("Kelime tan\u0131ma (20 kelime)", 5460)] }),
      new TableRow({ children: [cell("POST", 1560), cell("/tts", 2340), cell("Text-to-Speech (T\u00FCrk\u00E7e)", 5460)] }),
    ]
  }),
  emptyP(),
  p(bold("WebSocket Endpoints:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 7020],
    rows: [
      new TableRow({ children: [headerCell("Endpoint", 2340), headerCell("A\u00E7\u0131klama", 7020)] }),
      new TableRow({ children: [cell("/ws/avatar", 2340), cell("Avatar animasyon streaming", 7020)] }),
      new TableRow({ children: [cell("/ws/realtime", 2340), cell("Ger\u00E7ek zamanl\u0131 i\u015Faret tan\u0131ma", 7020)] }),
      new TableRow({ children: [cell("/ws/stats", 2340), cell("Ba\u011Flant\u0131 istatistikleri", 7020)] }),
    ]
  }),
  emptyP(),

  subheading("1.6 G\u00FCz D\u00F6nemi Web Aray\u00FCz\u00FC"),
  p(normal("HTML5/JavaScript ile ger\u00E7ek zamanl\u0131 web aray\u00FCz\u00FC olu\u015Fturulmu\u015Ftur. Kullan\u0131c\u0131, web kameras\u0131 \u00FCzerinden i\u015Faret yapabilmekte ve sistem ger\u00E7ek zamanl\u0131 olarak tahmin \u00FCretmektedir.")),
  emptyP(),
  p(bold("Web Sayfalar\u0131:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Sayfa", 3120), headerCell("A\u00E7\u0131klama", 6240)] }),
      new TableRow({ children: [cell("Ana Sayfa", 3120), cell("Kamera aray\u00FCz\u00FC, ger\u00E7ek zamanl\u0131 tahmin, istatistikler, Top-3 tahmin", 6240)] }),
      new TableRow({ children: [cell("Hastane \u0130leti\u015Fim", 3120), cell("Sa\u011Fl\u0131k sekt\u00F6r\u00FC i\u00E7in haz\u0131r mesaj butonlar\u0131 ve T\u0130D animasyonu", 6240)] }),
      new TableRow({ children: [cell("Unity Avatar", 3120), cell("3D avatar sayfas\u0131 (Unity WebGL ile render)", 6240)] }),
    ]
  }),
  p(normal("Aray\u00FCz, MediaPipe ile landmark g\u00F6rselle\u015Ftirmesi, tahmin sonucunu ekranda g\u00F6sterme ve istatistik paneli \u00F6zelliklerine sahiptir. Responsive tasar\u0131m ile farkl\u0131 ekran boyutlar\u0131na uyum sa\u011Flamaktad\u0131r.")),
  emptyP(),

  subheading("1.7 G\u00FCz D\u00F6nemi Teknoloji Y\u0131\u011F\u0131n\u0131"),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Teknoloji", 3120), headerCell("Kullan\u0131m Amac\u0131", 6240)] }),
      new TableRow({ children: [cell("Python 3.10", 3120), cell("Ana geli\u015Ftirme dili", 6240)] }),
      new TableRow({ children: [cell("PyTorch", 3120), cell("Derin \u00F6\u011Frenme model e\u011Fitimi ve inference", 6240)] }),
      new TableRow({ children: [cell("MediaPipe", 3120), cell("El ve v\u00FCcut landmark \u00E7\u0131kar\u0131m\u0131", 6240)] }),
      new TableRow({ children: [cell("Ultralytics YOLOv11", 3120), cell("Object detection tabanl\u0131 i\u015Faret tan\u0131ma", 6240)] }),
      new TableRow({ children: [cell("FastAPI", 3120), cell("REST API backend sunucu", 6240)] }),
      new TableRow({ children: [cell("OpenCV", 3120), cell("G\u00F6r\u00FCnt\u00FC i\u015Fleme ve kamera eri\u015Fimi", 6240)] }),
      new TableRow({ children: [cell("Roboflow", 3120), cell("Veri seti etiketleme ve y\u00F6netimi", 6240)] }),
    ]
  }),
  emptyP(),

  subheading("1.8 Geli\u015Ftirme Ortam\u0131 ve Donan\u0131m"),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Bile\u015Fen", 3120), headerCell("Detay", 6240)] }),
      new TableRow({ children: [cell("GPU", 3120), cell("NVIDIA RTX 3050 (4 GB VRAM)", 6240)] }),
      new TableRow({ children: [cell("CUDA", 3120), cell("CUDA 12.4 + cuDNN", 6240)] }),
      new TableRow({ children: [cell("RAM", 3120), cell("16 GB DDR4", 6240)] }),
      new TableRow({ children: [cell("\u0130\u015Fletim Sistemi", 3120), cell("Windows 11", 6240)] }),
      new TableRow({ children: [cell("Python", 3120), cell("3.10", 6240)] }),
      new TableRow({ children: [cell("IDE", 3120), cell("VS Code, Android Studio", 6240)] }),
      new TableRow({ children: [cell("Versiyon Kontrol", 3120), cell("Git + GitHub", 6240)] }),
    ]
  }),
  emptyP(),

  subheading("1.9 Ba\u015Far\u0131 \u00D6l\u00E7\u00FCtleri (G\u00FCz D\u00F6nemi)"),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 1560, 2340, 1560],
    rows: [
      new TableRow({ children: [headerCell("Metrik", 2340), headerCell("Hedef", 1560), headerCell("Ger\u00E7ekle\u015Fen", 2340), headerCell("Durum", 1560)] }),
      new TableRow({ children: [cell("Do\u011Fruluk (Rakam)", 2340), cell("\u2265%75", 1560), cell("%99.79", 2340), cell("\u2705 A\u015F\u0131ld\u0131", 1560)] }),
      new TableRow({ children: [cell("Do\u011Fruluk (Kelime)", 2340), cell("\u2265%75", 1560), cell("%99.6 mAP50", 2340), cell("\u2705 A\u015F\u0131ld\u0131", 1560)] }),
      new TableRow({ children: [cell("FPS", 2340), cell("\u226525", 1560), cell("~27 FPS", 2340), cell("\u2705", 1560)] }),
      new TableRow({ children: [cell("Gecikme", 2340), cell("\u226430ms", 1560), cell("~37ms", 2340), cell("\u26A0\uFE0F Yak\u0131n", 1560)] }),
    ]
  }),
  emptyP(),

  subheading("1.10 G\u00FCz D\u00F6nemi De\u011Ferlendirme"),
  p(normal("G\u00FCz d\u00F6nemi sonunda projenin temel altyap\u0131s\u0131 ba\u015Far\u0131yla kurulmu\u015F, iki \u00E7al\u0131\u015Fan model (rakam + 20 kelime) elde edilmi\u015Ftir. Ancak \u015Fu eksiklikler tespit edilmi\u015Ftir:")),
  bullet("Kelime say\u0131s\u0131 (20) g\u00FCnl\u00FCk ileti\u015Fim i\u00E7in yetersizdir; en az 100 kelimeye \u00E7\u0131k\u0131lmal\u0131d\u0131r"),
  bullet("YOLO yakla\u015F\u0131m\u0131 statik pozlarla s\u0131n\u0131rl\u0131d\u0131r; dinamik hareketler i\u00E7in sequence tabanl\u0131 model gereklidir"),
  bullet("Yaln\u0131zca T\u0130D\u2192Metin y\u00F6n\u00FC vard\u0131r; Metin\u2192T\u0130D y\u00F6n\u00FC eksiktir"),
  bullet("Mobil uygulama yoktur; yaln\u0131zca web taray\u0131c\u0131 \u00FCzerinden eri\u015Fim m\u00FCmk\u00FCnd\u00FCr"),
  bullet("C\u00FCmle olu\u015Fturma ve T\u00FCrk\u00E7e NLP deste\u011Fi bulunmamaktad\u0131r"),
  p(normal("Bu tespitler, bahar d\u00F6nemi \u00E7al\u0131\u015Fmalar\u0131n\u0131n y\u00F6n\u00FCn\u00FC belirlemi\u015Ftir.")),
  emptyP(),
  new Paragraph({ children: [new PageBreak()] }),
];

// ========== KISIM 2: PLANLANAN ÇALIŞMALAR ==========
const kisim2 = [
  separator(),
  heading("KISIM 2: PLANLANAN \u00C7ALI\u015EMALAR (G\u00FCz D\u00F6neminde Belirlenen Hedefler)"),
  p(normal("G\u00FCz d\u00F6nemi sonunda, bahar d\u00F6nemi i\u00E7in a\u015Fa\u011F\u0131daki hedefler belirlendi ve bunlar\u0131n ger\u00E7ekle\u015Ftirilme durumlar\u0131 KISIM 3\u2019te detayl\u0131 olarak a\u00E7\u0131klanm\u0131\u015Ft\u0131r:")),
  emptyP(),

  subheading("2.1 Model Geli\u015Ftirme Hedefleri"),
  bullet("Kelime say\u0131s\u0131n\u0131 20\u2019den en az 100\u2019e \u00E7\u0131karmak"),
  bullet("Statik YOLO modelinden sequence tabanl\u0131 (LSTM/GRU) modele ge\u00E7mek"),
  bullet("Temporal \u00F6r\u00FCnt\u00FCleri yakalayarak dinamik hareketleri tan\u0131mak"),
  bullet("Daha kapsaml\u0131 veri toplama ve augmentation stratejisi geli\u015Ftirmek"),
  emptyP(),

  subheading("2.2 Avatar ve \u00C7ift Y\u00F6nl\u00FC \u0130leti\u015Fim"),
  bullet("Metin \u2192 T\u0130D y\u00F6n\u00FCn\u00FC ekleyerek \u00E7ift y\u00F6nl\u00FC ileti\u015Fim sa\u011Flamak"),
  bullet("3D avatar animasyon sistemi geli\u015Ftirmek"),
  bullet("T\u00FCrk\u00E7e NLP mod\u00FCl\u00FC ile c\u00FCmle \u00E7evirisi yapabilmek"),
  emptyP(),

  subheading("2.3 Mobil Uygulama"),
  bullet("Flutter ile Android mobil uygulama geli\u015Ftirmek"),
  bullet("Kamera ve avatar modlar\u0131n\u0131 mobilde \u00E7al\u0131\u015F\u0131r hale getirmek"),
  bullet("Offline veya yerel a\u011F \u00FCzerinden \u00E7al\u0131\u015Fma deste\u011Fi"),
  emptyP(),

  subheading("2.4 \u0130\u015F Plan\u0131 (\u00D6ng\u00F6r\u00FClen)"),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [1560, 7800],
    rows: [
      new TableRow({ children: [headerCell("Hafta", 1560), headerCell("Planlanan \u00C7al\u0131\u015Fma", 7800)] }),
      new TableRow({ children: [cell("1-2", 1560), cell("Veri toplama (100 kelime), MediaPipe Holistic entegrasyonu", 7800)] }),
      new TableRow({ children: [cell("3-4", 1560), cell("GRU/LSTM model e\u011Fitimi, augmentation pipeline", 7800)] }),
      new TableRow({ children: [cell("5-6", 1560), cell("3D avatar geli\u015Ftirme, poz kaydetme sistemi, NLP mod\u00FCl\u00FC", 7800)] }),
      new TableRow({ children: [cell("7-8", 1560), cell("Flutter mobil uygulama, web/mobil entegrasyon", 7800)] }),
      new TableRow({ children: [cell("9-10", 1560), cell("Test, optimizasyon, rapor yaz\u0131m\u0131", 7800)] }),
    ]
  }),
  p(normal("Bu planlar\u0131n b\u00FCy\u00FCk \u00E7o\u011Funlu\u011Fu ba\u015Far\u0131yla ger\u00E7ekle\u015Ftirilmi\u015Ftir. Detaylar KISIM 3\u2019te yer almaktad\u0131r.")),
  emptyP(),
  new Paragraph({ children: [new PageBreak()] }),
];

// ========== KISIM 3: BAHAR ==========
const kisim3 = [
  separator(),
  heading("KISIM 3: GER\u00C7EKLE\u015ET\u0130R\u0130LEN \u00C7ALI\u015EMALAR (Bahar D\u00F6nemi 2026)"),
  p(normal("G\u00FCz d\u00F6neminde olu\u015Fturulan temel altyap\u0131 \u00FCzerine, bahar d\u00F6neminde projenin kapsam\u0131 \u00F6nemli \u00F6l\u00E7\u00FCde geni\u015Fletilmi\u015Ftir. 20 kelimelik YOLO modelinden 100 kelimelik GRU tabanl\u0131 sequence modeline ge\u00E7ilmi\u015F, 3D avatar animasyon sistemi geli\u015Ftirilmi\u015F ve Flutter mobil uygulama olu\u015Fturulmu\u015Ftur. Sistem art\u0131k hem web hem de mobil platformlarda \u00E7al\u0131\u015Fmakta olup, \u00E7ift y\u00F6nl\u00FC ileti\u015Fim (T\u0130D\u2192Metin ve Metin\u2192T\u0130D Avatar) deste\u011Fi sunmaktad\u0131r.")),
  emptyP(),
  subheading("3.1 Model Geli\u015Ftirme: GRU Tabanl\u0131 100 Kelime Tan\u0131ma"),
  p(normal("G\u00FCz d\u00F6neminde YOLOv11-nano ile 20 kelime tan\u0131ma ba\u015Far\u0131yla ger\u00E7ekle\u015Ftirilmi\u015Fti. Ancak YOLO yakla\u015F\u0131m\u0131 (object detection) her frame\u2019i ba\u011F\u0131ms\u0131z olarak de\u011Ferlendirdi\u011Fi i\u00E7in statik poz tan\u0131ma ile s\u0131n\u0131rl\u0131 kalm\u0131\u015F, dinamik hareketlerin temporal \u00F6r\u00FCnt\u00FClerini yakalayamam\u0131\u015Ft\u0131r. \u00D6rne\u011Fin \u201CMERHABA\u201D ve \u201CHO\u015E\u00C7A KAL\u201D gibi hareket i\u00E7eren i\u015Faretler, YOLO ile tutarl\u0131 bi\u00E7imde tan\u0131namam\u0131\u015Ft\u0131r. Bu nedenle bahar d\u00F6neminde sequence-based bir mimariye ge\u00E7ilmi\u015Ftir.")),
  emptyP(),
  p(bold("Neden GRU?")),
  p(normal("Recurrent Neural Network (RNN) ailesinden GRU (Gated Recurrent Unit) mimarisi, LSTM\u2019e g\u00F6re daha az parametre ile benzer performans sa\u011Flamaktad\u0131r. GRU\u2019nun iki kap\u0131s\u0131 vard\u0131r: reset gate (ne kadar ge\u00E7mi\u015F bilginin unutulaca\u011F\u0131) ve update gate (ne kadar yeni bilginin al\u0131naca\u011F\u0131). LSTM\u2019deki cell state ayr\u0131 bir yap\u0131 olarak bulunmazken, GRU bu i\u015Flevi hidden state \u00FCzerinden ger\u00E7ekle\u015Ftirir. Bu sayede daha az hesaplama maliyeti ile mobil cihazlarda daha h\u0131zl\u0131 inference s\u00FCresi sunmaktad\u0131r.")),
  emptyP(),
  p(bold("MediaPipe Holistic Landmark \u00C7\u0131kar\u0131m\u0131:")),
  p(normal("Her video frame\u2019inden Google MediaPipe Holistic kullan\u0131larak toplam 543 landmark noktas\u0131 \u00E7\u0131kar\u0131lmaktad\u0131r. Bu landmark\u2019lar 4 farkl\u0131 v\u00FCcut b\u00F6lgesinden elde edilir:")),
  bullet("Y\u00FCz: 468 nokta (y\u00FCz ifadeleri ve a\u011F\u0131z hareketleri i\u00E7in)"),
  bullet("V\u00FCcut (Pose): 33 nokta (omuz, dirsek, bilek pozisyonlar\u0131)"),
  bullet("Sol El: 21 nokta (parmak eklemleri ve avu\u00E7 i\u00E7i)"),
  bullet("Sa\u011F El: 21 nokta (parmak eklemleri ve avu\u00E7 i\u00E7i)"),
  p(normal("Her noktan\u0131n (x, y, z) koordinatlar\u0131 al\u0131nd\u0131\u011F\u0131nda toplam girdi boyutu 543 \u00D7 3 = 1629 \u00F6zellik olmaktad\u0131r. 30 frame\u2019lik bir pencere (SEQ_LEN=30) \u00FCzerinden temporal \u00F6r\u00FCnt\u00FCler yakalanmaktad\u0131r.")),
  emptyP(),
  p(bold("Model Mimarisi Detaylar\u0131:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Parametre", 3120), headerCell("De\u011Fer", 6240)] }),
      new TableRow({ children: [cell("Mimari", 3120), cell("GRU, 2 katman, 256 hidden units", 6240)] }),
      new TableRow({ children: [cell("Girdi Boyutu", 3120), cell("1629 \u00F6zellik (543 landmark \u00D7 3 koordinat)", 6240)] }),
      new TableRow({ children: [cell("Sequence Uzunlu\u011Fu", 3120), cell("30 frame (SEQ_LEN=30)", 6240)] }),
      new TableRow({ children: [cell("Frame Stride", 3120), cell("3 (her 3 frame\u2019de 1 \u00F6rnekleme)", 6240)] }),
      new TableRow({ children: [cell("S\u0131n\u0131f Say\u0131s\u0131", 3120), cell("100 kelime", 6240)] }),
      new TableRow({ children: [cell("Classifier", 3120), cell("Linear(256\u2192128) \u2192 ReLU \u2192 Dropout \u2192 Linear(128\u2192100)", 6240)] }),
      new TableRow({ children: [cell("Optimizer", 3120), cell("Adam (lr=0.001, weight_decay=1e-4)", 6240)] }),
      new TableRow({ children: [cell("Scheduler", 3120), cell("ReduceLROnPlateau (factor=0.5, patience=8)", 6240)] }),
      new TableRow({ children: [cell("Loss", 3120), cell("CrossEntropyLoss (class weights ile dengeleme)", 6240)] }),
      new TableRow({ children: [cell("Dropout", 3120), cell("0.3 (GRU) + 0.15 (classifier)", 6240)] }),
      new TableRow({ children: [cell("Batch Size", 3120), cell("16", 6240)] }),
      new TableRow({ children: [cell("Max Epoch", 3120), cell("100", 6240)] }),
      new TableRow({ children: [cell("Train/Val Split", 3120), cell("%80 / %20 (stratified)", 6240)] }),
      new TableRow({ children: [cell("Toplam Veri", 3120), cell("70392 sequence (augmentation sonras\u0131)", 6240)] }),
      new TableRow({ children: [cell("Validation Accuracy", 3120), cell("%80.3 (14079 val sample)", 6240)] }),
    ]
  }),
  emptyP(),
  p(bold("Tan\u0131nan 100 Kelime:")),
  p(normal("AB\u0130, ABLA, AFER\u0130N, AGUSTOS, AKILLI, AKRABA, AKSAM, ALERJI, ALLAH, ALTI, AMCA, AMIN, ANNE, APTAL, ARALIK, ARAP, ARKADAS, ASK, ATAT\u00DCRK, AYIP, AYRAN, BABA, BAKLAVA, BARIS, BEN, BES, BILGISAYAR, BIR, BOS VERMEK, BU KADAR, BUGUN, BUYUK, CARSAMBA, CAY, CD, CEP TELEFONU, CEZA, CIKOLATA, CUMA, CUMARTESI, CUNKU, DIKKAT, DOKTOR, DOKUZ, DORT, DOYMAK, DUN, EKIM, EKMEK, ELHAMDULILLAH, ERIK, EV, EVET, EVLENMEK, EYLUL, FINAL, FUTBOL, GUN, HANGI, HAZIRAN, HEMSIRE, HENTBOL, HOSCA KAL, IKI, IYI, KAHVE, KASIM, KIS, KOTU, KUCUK, MART, MAYIS, MERHABA, NISAN, OCAK, OGRETMEN, OKUL, ORUC, PAZAR, PAZARTESI, PERSEMBE, SABAH, SALI, SEKIZ, SEN, SIFIR, SU, SUBAT, TEMMUZ, TESEKKUR, TUVALET, TV, UC, VOLEYBOL, YARDIM, YATMAK, YEDI, YEMEK, YETER, ZENGIN")),
  emptyP(),
  p(bold("S\u0131n\u0131f Bazl\u0131 Performans Analizi:")),
  p(normal("100 kelime aras\u0131ndaki performans de\u011Fi\u015Fkenlik g\u00F6stermektedir. En y\u00FCksek performansl\u0131 kelimeler (F1 \u2265 0.95):")),
  bullet("DOYMAK: F1=1.00 (m\u00FCkemmel tan\u0131ma)"),
  bullet("HANGI: F1=1.00, KIS: F1=1.00, KASIM: F1=1.00"),
  bullet("YARDIM: F1=1.00, EV: F1=0.99, CD: F1=0.99"),
  bullet("SABAH: F1=0.99, PAZAR: F1=0.99, PAZARTESI: F1=0.97"),
  emptyP(),
  p(normal("D\u00FC\u015F\u00FCk performansl\u0131 kelimeler (F1 < 0.60) ve nedenleri:")),
  bullet("BIR: F1=0.36 \u2013 k\u0131sa ve basit hareket, di\u011Fer rakamlarla kar\u0131\u015F\u0131yor"),
  bullet("MAYIS: F1=0.36 \u2013 di\u011Fer ay isimleriyle benzer hareket \u00F6r\u00FCnt\u00FCs\u00FC"),
  bullet("DOKUZ: F1=0.46, SEKIZ: F1=0.47 \u2013 rakam i\u015Faretleri birbirine yak\u0131n"),
  bullet("KAHVE: F1=0.52 \u2013 hareket benzerli\u011Fi sorunu"),
  bullet("KOTU: F1=0.55 \u2013 IYI kelimesiyle kar\u0131\u015Fma"),
  p(normal("Genel olarak, hareket \u00F6r\u00FCnt\u00FCs\u00FC birbirine benzeyen kelimeler (rakamlar, ay isimleri) daha d\u00FC\u015F\u00FCk do\u011Fruluk g\u00F6sterirken, benzersiz ve belirgin hareketlere sahip kelimeler (DOYMAK, HANGI, KIS) m\u00FCkemmel do\u011Fruluk elde etmi\u015Ftir.")),
  emptyP(),

  subheading("3.2 Veri Toplama ve Augmentation S\u00FCreci"),
  p(normal("G\u00FCz d\u00F6neminde LSTM i\u00E7in 5 kelime \u00D7 50 sample = 250 sequence toplanm\u0131\u015F ve bu yetersiz kalm\u0131\u015Ft\u0131. Bahar d\u00F6neminde tamamen farkl\u0131 ve \u00E7ok daha kapsaml\u0131 bir veri toplama stratejisi uygulanm\u0131\u015Ft\u0131r.")),
  emptyP(),
  p(bold("Video Tabanl\u0131 Toplama Pipeline\u2019\u0131:")),
  p(normal("1. Webcam \u00FCzerinden her kelime i\u00E7in 15-30 saniyelik video kayd\u0131 yap\u0131l\u0131r")),
  p(normal("2. Video, OpenCV ile frame\u2019lere ayr\u0131l\u0131r")),
  p(normal("3. Her frame MediaPipe Holistic ile i\u015Flenerek 1629 boyutlu landmark vekt\u00F6r\u00FC \u00E7\u0131kar\u0131l\u0131r")),
  p(normal("4. Kayan pencere (sliding window) ile 30 frame\u2019lik sequence\u2019ler olu\u015Fturulur (stride=3)")),
  p(normal("5. Her sequence .npy format\u0131nda (NumPy array) kaydedilir")),
  p(normal("Bu pipeline sayesinde tek bir 20 saniyelik videodan yakla\u015F\u0131k 150-200 sequence otomatik olarak \u00E7\u0131kar\u0131labilmektedir. Manuel tek tek toplaman\u0131n aksine bu y\u00F6ntem \u00E7ok daha h\u0131zl\u0131 ve tutarl\u0131 veri elde edilmesini sa\u011Flam\u0131\u015Ft\u0131r.")),
  emptyP(),
  p(bold("Data Augmentation Teknikleri:")),
  p(normal("Toplanan ham veri \u00FCzerinde 6 farkl\u0131 augmentation tekni\u011Fi uygulanarak veri \u00E7e\u015Fitlili\u011Fi art\u0131r\u0131lm\u0131\u015Ft\u0131r:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [2400, 6960],
    rows: [
      new TableRow({ children: [headerCell("Teknik", 2400), headerCell("A\u00E7\u0131klama", 6960)] }),
      new TableRow({ children: [cell("Time Warp", 2400), cell("Zamansal geni\u015Fletme/s\u0131k\u0131\u015Ft\u0131rma \u2013 sequence h\u0131z\u0131n\u0131 rastgele de\u011Fi\u015Ftirerek farkl\u0131 tempo varyasyonlar\u0131 olu\u015Fturur", 6960)] }),
      new TableRow({ children: [cell("Spatial Scale", 2400), cell("Uzamsal \u00F6l\u00E7ekleme \u2013 landmark koordinatlar\u0131n\u0131 rastgele \u00F6l\u00E7ekleyerek farkl\u0131 v\u00FCcut boyutlar\u0131n\u0131 sim\u00FCle eder", 6960)] }),
      new TableRow({ children: [cell("Translation", 2400), cell("Uzamsal kayma \u2013 t\u00FCm landmark\u2019lar\u0131 rastgele \u00F6teleyerek farkl\u0131 konumlar\u0131 sim\u00FCle eder", 6960)] }),
      new TableRow({ children: [cell("Noise Injection", 2400), cell("Gaussian g\u00FCr\u00FClt\u00FC ekleme \u2013 landmark koordinatlar\u0131na k\u00FC\u00E7\u00FCk rastgele de\u011Ferler ekler", 6960)] }),
      new TableRow({ children: [cell("Mirror", 2400), cell("Ayna yans\u0131mas\u0131 \u2013 sol/sa\u011F eli de\u011Fi\u015Ftirerek veri \u00E7e\u015Fitlili\u011Fini art\u0131r\u0131r", 6960)] }),
      new TableRow({ children: [cell("Rotation", 2400), cell("Uzamsal rotasyon \u2013 landmark\u2019lar\u0131 rastgele d\u00F6nd\u00FCrerek a\u00E7\u0131 varyasyonlar\u0131 olu\u015Fturur", 6960)] }),
    ]
  }),
  emptyP(),
  p(bold("Veri \u0130statistikleri:")),
  bullet("Ham veri: 100 kelime \u00D7 ortalama ~100 sequence = ~10000 ham sequence"),
  bullet("Augmentation sonras\u0131: 70392 toplam sequence"),
  bullet("E\u011Fitim seti: 56313 sequence (%80)"),
  bullet("Validation seti: 14079 sequence (%20, stratified split)"),
  emptyP(),

  subheading("3.3 3D Avatar Animasyon Sistemi (Metin \u2192 T\u0130D)"),
  p(normal("G\u00FCz d\u00F6neminde yaln\u0131zca T\u0130D\u2192Metin y\u00F6n\u00FC geli\u015Ftirilmi\u015Fti. Bahar d\u00F6neminde projeye Metin\u2192T\u0130D y\u00F6n\u00FC eklenerek \u00E7ift y\u00F6nl\u00FC ileti\u015Fim sa\u011Flanm\u0131\u015Ft\u0131r. Bu ama\u00E7la Three.js tabanl\u0131 bir 3D avatar animasyon sistemi geli\u015Ftirilmi\u015Ftir. Kullan\u0131c\u0131 T\u00FCrk\u00E7e bir c\u00FCmle yazd\u0131\u011F\u0131nda, sistem bu c\u00FCmleyi T\u0130D gloss formuna d\u00F6n\u00FC\u015Ft\u00FCr\u00FCr ve 3D avatar her kelimeyi s\u0131rayla canland\u0131r\u0131r.")),
  emptyP(),
  p(bold("3D Model ve Render Sistemi:")),
  p(normal("Avatar olarak Blender\u2019da olu\u015Fturulmu\u015F bir humanoid karakter (rain.glb) kullan\u0131lmaktad\u0131r. Model, CloudRig kemik sistemi ile donat\u0131lm\u0131\u015F olup toplam 100+ kemik i\u00E7ermektedir. Three.js k\u00FCt\u00FCphanesi ve GLTFLoader ile WebGL tabanl\u0131 ger\u00E7ek zamanl\u0131 render ger\u00E7ekle\u015Ftirilmektedir. Sahne, DirectionalLight ve AmbientLight ile ayd\u0131nlat\u0131lm\u0131\u015F, OrbitControls ile kullan\u0131c\u0131 kameray\u0131 d\u00F6nd\u00FCrebilmektedir.")),
  emptyP(),
  p(bold("Poz Kay\u0131t Sistemi:")),
  p(normal("Her kelime i\u00E7in avatara ait kemik rotasyonlar\u0131 (quaternion) elle kaydedilmi\u015Ftir. Toplam 101 kelime i\u00E7in ayr\u0131 ayr\u0131 poz veritaban\u0131 olu\u015Fturulmu\u015F ve saved_poses.json dosyas\u0131nda saklanmaktad\u0131r. Her poz, kemik ad\u0131 \u2192 {x, y, z, w} quaternion rotasyonu e\u015Fle\u015Ftirmesi bi\u00E7imindedir. Poz kay\u0131t arac\u0131 (step2_recorder.html) ile yeni kelimeler kolayca eklenebilmektedir.")),
  emptyP(),
  p(bold("Animasyon Ge\u00E7i\u015F Sistemi:")),
  p(normal("Pozlar aras\u0131 ge\u00E7i\u015Flerde Smootherstep easing fonksiyonu kullan\u0131lmaktad\u0131r: f(t) = t\u00B3(t(6t\u221215)+10). Bu fonksiyon, linear interpolation\u2019a g\u00F6re \u00E7ok daha do\u011Fal ve ak\u0131c\u0131 ge\u00E7i\u015Fler sa\u011Flamaktad\u0131r. Quaternion interpolasyonu (slerp) ile kemik rotasyonlar\u0131 yumusat\u0131lm\u0131\u015Ft\u0131r. Her kelime aras\u0131nda idle (bo\u015Fta) pozuna d\u00F6n\u00FC\u015F yap\u0131larak do\u011Fal bir g\u00F6r\u00FCn\u00FCm elde edilmektedir.")),
  emptyP(),
  p(bold("C\u00FCmle Animasyonu:")),
  p(normal("Kullan\u0131c\u0131 bir c\u00FCmle girdi\u011Finde \u015Fu ad\u0131mlar uygulan\u0131r:")),
  p(normal("1. T\u00FCrk\u00E7e NLP mod\u00FCl\u00FC ile c\u00FCmle gloss dizisine \u00E7evrilir (\u00F6rn: \u201CBug\u00FCn evdeydim\u201D \u2192 [BUGUN, EV])")),
  p(normal("2. Her kelime s\u0131rayla animasyon kuyru\u011Funa eklenir")),
  p(normal("3. Avatar, idle \u2192 kelime1 \u2192 idle \u2192 kelime2 \u2192 ... \u015Feklinde s\u0131ral\u0131 animasyon yapar")),
  p(normal("4. Animasyon s\u0131ras\u0131nda alt\u00E7ubukta hangi kelimenin oynat\u0131ld\u0131\u011F\u0131 g\u00F6sterilir")),
  emptyP(),
  p(bold("H\u0131z Kontrol\u00FC: "), normal("0.3x\u20132.0x aras\u0131 ayarlanabilir animasyon h\u0131z\u0131. Yava\u015F h\u0131zda \u00F6\u011Frenme ama\u00E7l\u0131 kullan\u0131m, h\u0131zl\u0131 modda do\u011Fal konu\u015Fma temposu elde edilmektedir.")),
  emptyP(),

  subheading("3.4 T\u00FCrk\u00E7e NLP (Do\u011Fal Dil \u0130\u015Fleme) Mod\u00FCl\u00FC"),
  p(normal("Avatar sisteminin en kritik bile\u015Fenlerinden biri, kullan\u0131c\u0131n\u0131n yazd\u0131\u011F\u0131 T\u00FCrk\u00E7e c\u00FCmleleri T\u0130D gloss formuna d\u00F6n\u00FC\u015Ft\u00FCren NLP mod\u00FCl\u00FCD\u00FCr. T\u0130D, T\u00FCrk\u00E7eden farkl\u0131 bir c\u00FCmle yap\u0131s\u0131na sahiptir ve ekler kullan\u0131lmaz. \u00D6rne\u011Fin T\u00FCrk\u00E7ede \u201CEvdeydim\u201D tek kelime iken, T\u0130D\u2019de bu \u201CEV\u201D olarak ifade edilir. T\u00FCrk\u00E7e sondan eklemeli (agglutinative) bir dil oldu\u011Fu i\u00E7in, kelimelerin k\u00F6k\u00FCne ula\u015Fmak i\u00E7in kapsaml\u0131 bir ek at\u0131lma (suffix stripping) sistemi geli\u015Ftirilmi\u015Ftir.")),
  emptyP(),
  p(bold("NLP \u0130\u015Flem Ad\u0131mlar\u0131:")),
  p(normal("1. "), bold("T\u00FCrk\u00E7e karakter normalizasyonu: "), normal("\u0130\u2192I, \u015E\u2192S, \u00D6\u2192O, \u00DC\u2192U, \u00C7\u2192C, \u011E\u2192G d\u00F6n\u00FC\u015F\u00FCmleri yap\u0131l\u0131r. Bu, poz veritaban\u0131ndaki ASCII tabanl\u0131 anahtar kelimelerle e\u015Fle\u015Ftirme i\u00E7in gereklidir.")),
  p(normal("2. "), bold("\u00C7ok kelimeli poz e\u015Fle\u015Ftirme: "), normal("T\u0130D\u2019de baz\u0131 kavramlar birden fazla kelime ile ifade edilir. Sistem, 2-3 kelimelik bile\u015Fikleri \u00F6nce kontrol eder (\u00F6rn: HO\u015E\u00C7A KAL, BO\u015E VERMEK, BU KADAR, CEP TELEFONU, GE\u00C7M\u0130\u015E OLSUN).")),
  p(normal("3. "), bold("Suffix stripping (ek at\u0131lma): "), normal("Toplam 115+ T\u00FCrk\u00E7e ek desteklenmektedir. Ekler uzunluklar\u0131na g\u00F6re s\u0131ralanarak en uzun e\u015Fle\u015Fme \u00F6nceliklidir. Desteklenen ek kategorileri:")),
  bullet("Ge\u00E7mi\u015F zaman: -DAYDIM, -DEYDIM, -MISTIM, -MUSTUM, -DI, -DU, -TI, -TU"),
  bullet("\u015Eimdiki zaman: -IYORUM, -IYORSUN, -UYORUM, -UYOR, -YOR"),
  bullet("Gelecek zaman: -ACAK, -ECEK, -ACAGIM, -ECEGIM"),
  bullet("\u00C7o\u011Ful/\u0130yelik: -LARI, -LERI, -LARINA, -LERINE, -LARIMIZ, -LERIMIZ"),
  bullet("Hal ekleri: -DA, -DE, -DAN, -DEN, -INA, -INE, -INDA, -INDE"),
  bullet("\u0130yelik: -IM, -IN, -IMIZI, -INIZI, -UMUZ, -UNUZ"),
  p(normal("4. "), bold("K\u00F6k minimum uzunluk korumas\u0131: "), normal("K\u00F6k en az 2 karakter olmal\u0131d\u0131r (EV, SU gibi k\u0131sa kelimeler i\u00E7in). Bu, a\u015F\u0131r\u0131 ek at\u0131lmas\u0131n\u0131 \u00F6nler.")),
  p(normal("5. "), bold("Soru tespiti: "), normal("C\u00FCmlede mi/m\u0131/mu/m\u00FC ekleri veya ? i\u015Fareti varsa, sonu\u00E7 dizisine SORU kelimesi eklenir.")),
  emptyP(),
  p(bold("\u00D6rnek D\u00F6n\u00FC\u015F\u00FCmler:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 3120, 3120],
    rows: [
      new TableRow({ children: [headerCell("T\u00FCrk\u00E7e Giri\u015F", 3120), headerCell("T\u0130D Gloss", 3120), headerCell("A\u00E7\u0131klama", 3120)] }),
      new TableRow({ children: [cell("Evdeydim", 3120), cell("EV", 3120), cell("Ek at\u0131lma: -DEYDIM", 3120)] }),
      new TableRow({ children: [cell("Ho\u015F\u00E7a kal\u0131n", 3120), cell("HOSCA KAL", 3120), cell("\u00C7ok kelimeli + ek at\u0131lma: -IN", 3120)] }),
      new TableRow({ children: [cell("Bug\u00FCn evdeydim", 3120), cell("BUGUN EV", 3120), cell("\u0130ki kelime ayr\u0131 i\u015Flenir", 3120)] }),
      new TableRow({ children: [cell("Okula gidiyorum", 3120), cell("OKUL", 3120), cell("-A ve -IYORUM ekleri at\u0131l\u0131r", 3120)] }),
      new TableRow({ children: [cell("Doktor nerede?", 3120), cell("DOKTOR NEREDE SORU", 3120), cell("Soru i\u015Fareti tespiti", 3120)] }),
    ]
  }),
  emptyP(),
  p(normal("NLP mod\u00FCl\u00FC hem JavaScript (web avatar) hem de Dart (Flutter mobil) dillerinde ayr\u0131 ayr\u0131 implement edilmi\u015Ftir. Her iki implementasyon da ayn\u0131 suffix listesini ve \u00E7ok kelimeli e\u015Fle\u015Ftirme mant\u0131\u011F\u0131n\u0131 kullanmaktad\u0131r.")),
  emptyP(),

  subheading("3.5 Web Uygulamas\u0131 (Backend + Frontend)"),
  p(normal("G\u00FCz d\u00F6neminde FastAPI ile olu\u015Fturulan REST API, bahar d\u00F6neminde Flask + SocketIO tabanl\u0131 bir mimariye ge\u00E7irilmi\u015Ftir. Bu ge\u00E7i\u015Fin temel sebebi, ger\u00E7ek zamanl\u0131 \u00E7ift y\u00F6nl\u00FC ileti\u015Fim i\u00E7in WebSocket protokol\u00FCn\u00FCn gerekli olmas\u0131d\u0131r. HTTP request/response modeli, s\u00FCrekli video ak\u0131\u015F\u0131 i\u00E7in verimsiz kalmaktayd\u0131.")),
  emptyP(),
  p(bold("Backend Mimarisi (web_app.py):")),
  p(normal("Sunucu, ayn\u0131 anda hem web hem de mobil istemcilere hizmet vermektedir:")),
  bullet("Flask + Flask-SocketIO ile WebSocket ve HTTP endpoint\u2019leri"),
  bullet("WebSocket: Web taray\u0131c\u0131dan gelen base64 frame\u2019leri ger\u00E7ek zamanl\u0131 i\u015Fler"),
  bullet("HTTP POST /predict_frame: Mobil uygulamadan gelen tek frame tahminleri"),
  bullet("MediaPipe Holistic ile ger\u00E7ek zamanl\u0131 landmark \u00E7\u0131kar\u0131m\u0131"),
  bullet("GRU model inference (PyTorch, GPU deste\u011Fi)"),
  bullet("Sliding window buffer: 30 frame biriktirilir, her yeni frame\u2019de tahmin g\u00FCncellenir"),
  bullet("Confidence threshold (%90) ve smoothing (10 ard\u0131\u015F\u0131k ayn\u0131 tahmin) ile g\u00FCr\u00FClt\u00FC filtreleme"),
  bullet("C\u00FCmle olu\u015Fturma: Kelimeler otomatik biriktirilir, 2 saniye el alg\u0131lanmazsa c\u00FCmle tamamlan\u0131r"),
  bullet("gTTS ile T\u00FCrk\u00E7e seslendirme deste\u011Fi (tamamlanan c\u00FCmleleri sesli okuma)"),
  bullet("CORS deste\u011Fi ile cross-origin eri\u015Fim (mobil uygulama i\u00E7in gerekli)"),
  emptyP(),
  p(bold("Ger\u00E7ek Zamanl\u0131 Inference Pipeline\u2019\u0131:")),
  p(normal("1. \u0130stemci (web/mobil) base64 kodlanm\u0131\u015F frame g\u00F6nderir")),
  p(normal("2. Frame decode edilir, BGR\u2019ye \u00E7evrilir, gerekirse ayna yans\u0131mas\u0131 yap\u0131l\u0131r")),
  p(normal("3. MediaPipe Holistic ile 543 landmark \u00E7\u0131kar\u0131l\u0131r")),
  p(normal("4. Landmark vekt\u00F6r\u00FC 30 frame\u2019lik sliding window buffer\u2019a eklenir")),
  p(normal("5. Buffer doldu\u011Funda GRU modeli ile tahmin yap\u0131l\u0131r")),
  p(normal("6. Tahmin sonucu confidence score ile birlikte istemciye d\u00F6nd\u00FCr\u00FCl\u00FCr")),
  emptyP(),
  p(bold("Frontend:")),
  p(normal("Responsive HTML5 aray\u00FCz\u00FC, koyu tema tasar\u0131m\u0131 ile geli\u015Ftirilmi\u015Ftir. Sol panelde kamera g\u00F6r\u00FCnt\u00FCs\u00FC ve MediaPipe landmark overlay\u2019i, sa\u011F panelde tan\u0131nan kelimeler, T\u00FCrk\u00E7e c\u00FCmle \u00E7evirisi ve ge\u00E7mi\u015F c\u00FCmleler g\u00F6sterilmektedir. Buffer doluluk \u00E7ubu\u011Fu ile kullan\u0131c\u0131ya g\u00F6rsel geri bildirim verilmektedir.")),
  emptyP(),

  subheading("3.6 Flutter Mobil Uygulama"),
  p(normal("Projenin en \u00F6nemli bahar d\u00F6nemi geli\u015Fmelerinden biri, t\u00FCm sistemin Flutter ile mobil uygulamaya ta\u015F\u0131nmas\u0131d\u0131r. Uygulama Android platformunda \u00E7al\u0131\u015Fmakta olup, iki ana mod i\u00E7ermektedir:")),
  emptyP(),
  p(bold("Kamera Modu (T\u0130D \u2192 Metin):")),
  p(normal("Telefon kameras\u0131n\u0131 kullanarak ger\u00E7ek zamanl\u0131 i\u015Faret dili tan\u0131ma ger\u00E7ekle\u015Ftirir. CameraController ile \u00F6n kameradan frame\u2019ler yakalan\u0131r, JPEG format\u0131nda base64\u2019e encode edilir ve HTTP POST ile sunucuya g\u00F6nderilir. Sunucu yan\u0131t\u0131 (tahmin, confidence, el tespiti durumu) al\u0131nd\u0131\u011F\u0131nda ekranda g\u00F6sterilir.")),
  p(normal("Kamera modunda web aray\u00FCz\u00FCyle ayn\u0131 smoothing mant\u0131\u011F\u0131 uygulanmaktad\u0131r: confidence threshold (%80), smooth needed (4 ard\u0131\u015F\u0131k ayn\u0131 tahmin), same word max (3), min word gap (600ms). Bu parametreler, yanl\u0131\u015F pozitifleri \u00F6nlerken ger\u00E7ek tahminlerin h\u0131zl\u0131ca g\u00F6sterilmesini sa\u011Flamaktad\u0131r.")),
  emptyP(),
  p(bold("Avatar Modu (Metin \u2192 T\u0130D):")),
  p(normal("Flutter WebView bile\u015Feni ile 3D avatar sayfas\u0131 mobil cihazda render edilmektedir. Kullan\u0131c\u0131 T\u00FCrk\u00E7e c\u00FCmle yazd\u0131\u011F\u0131nda, Dart ile yaz\u0131lm\u0131\u015F NLP mod\u00FCl\u00FC c\u00FCmleyi gloss dizisine \u00E7evirir ve JavaScript bridge \u00FCzerinden WebView\u2019a g\u00F6nderir. Avatar, kelimeleri s\u0131rayla canland\u0131r\u0131r.")),
  emptyP(),
  p(bold("Mobil Uygulama \u00D6zellikleri:")),
  bullet("\u00D6n/arka kamera deste\u011Fi ile ger\u00E7ek zamanl\u0131 capture (Timer.periodic, 150ms aral\u0131k)"),
  bullet("Flutter WebView ile 3D avatar g\u00F6r\u00FCnt\u00FCleme (Three.js WebGL render)"),
  bullet("Dart ile T\u00FCrk\u00E7e NLP c\u00FCmle \u00E7evirisi (web versiyonuyla ayn\u0131 115+ suffix deste\u011Fi)"),
  bullet("H\u0131zl\u0131 ifade butonlar\u0131: Merhaba, Te\u015Fekk\u00FCr, Evet, Ho\u015F\u00E7a Kal, Su"),
  bullet("101 kelime listesi paneli (slide-up panel ile g\u00F6r\u00FCnt\u00FCleme)"),
  bullet("H\u0131z ayar\u0131 slider (0.3x \u2013 2.0x)"),
  bullet("Sunucu ba\u011Flant\u0131 durumu g\u00F6stergesi (Ba\u011Fl\u0131/Ba\u011Flant\u0131 Yok)"),
  bullet("Otomatik sunucu ke\u015Ffi (mDNS + manuel IP yap\u0131land\u0131rmas\u0131)"),
  bullet("Lifecycle y\u00F6netimi: Uygulama arka plana at\u0131ld\u0131\u011F\u0131nda kamera durdurulur, geri d\u00F6n\u00FCld\u00FC\u011F\u00FCnde otomatik ba\u015Flat\u0131l\u0131r"),
  bullet("Debug konsol (a\u011F istekleri ve hatalar\u0131 g\u00F6sterir)"),
  emptyP(),
  p(bold("Teknik Mimari:")),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Bile\u015Fen", 3120), headerCell("Teknoloji", 6240)] }),
      new TableRow({ children: [cell("UI Framework", 3120), cell("Flutter 3.x (Dart)", 6240)] }),
      new TableRow({ children: [cell("Kamera", 3120), cell("camera package (CameraController, takePicture)", 6240)] }),
      new TableRow({ children: [cell("3D Avatar", 3120), cell("webview_flutter (Three.js WebGL i\u00E7erir)", 6240)] }),
      new TableRow({ children: [cell("HTTP \u0130stemci", 3120), cell("http package (POST /predict_frame)", 6240)] }),
      new TableRow({ children: [cell("NLP", 3120), cell("Dart NlpService (suffix stripping, gloss \u00E7evirisi)", 6240)] }),
      new TableRow({ children: [cell("Yerel Sunucu", 3120), cell("shelf package (avatar HTML dosyalar\u0131 serve eder)", 6240)] }),
      new TableRow({ children: [cell("State Y\u00F6netimi", 3120), cell("StatefulWidget + setState", 6240)] }),
    ]
  }),
  emptyP(),

  subheading("3.7 Sistem Mimarisi Genel Bak\u0131\u015F"),
  p(normal("SignBridge sistemi, a\u015Fa\u011F\u0131daki bile\u015Fenlerden olu\u015Fmaktad\u0131r:")),
  emptyP(),
  p(bold("T\u0130D \u2192 Metin Y\u00F6n\u00FC:")),
  p(normal("Kamera \u2192 Frame Capture \u2192 Base64 Encode \u2192 Sunucu \u2192 MediaPipe Landmark \u2192 GRU Tahmin \u2192 Smoothing \u2192 Kelime \u2192 C\u00FCmle Olu\u015Fturma \u2192 T\u00FCrk\u00E7e NLP \u2192 Ekranda G\u00F6sterim")),
  emptyP(),
  p(bold("Metin \u2192 T\u0130D Y\u00F6n\u00FC:")),
  p(normal("T\u00FCrk\u00E7e C\u00FCmle Giri\u015Fi \u2192 NLP Suffix Stripping \u2192 Gloss Dizisi \u2192 Poz Veritaban\u0131 E\u015Fle\u015Ftirme \u2192 Three.js Quaternion Animasyon \u2192 3D Avatar G\u00F6r\u00FCnt\u00FCleme")),
  emptyP(),
  p(bold("Platform Deste\u011Fi:")),
  bullet("Web: Chrome/Firefox/Edge taray\u0131c\u0131lar\u0131nda WebSocket ile d\u00FC\u015F\u00FCk gecikmeli ileti\u015Fim"),
  bullet("Mobil: Flutter Android uygulamas\u0131 ile HTTP/REST API \u00FCzerinden ileti\u015Fim"),
  bullet("Sunucu: Python Flask + SocketIO, yerel a\u011Fda (LAN) \u00E7al\u0131\u015F\u0131r"),
  emptyP(),

  subheading("3.8 G\u00FCncellenen Ba\u015Far\u0131 \u00D6l\u00E7\u00FCtleri (G\u00FCz vs Bahar)"),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [2340, 1800, 3000, 2220],
    rows: [
      new TableRow({ children: [headerCell("Metrik", 2340), headerCell("G\u00FCz", 1800), headerCell("Bahar", 3000), headerCell("Durum", 2220)] }),
      new TableRow({ children: [cell("Kelime Say\u0131s\u0131", 2340), cell("20", 1800), cell("100", 3000), cell("5x art\u0131\u015F", 2220)] }),
      new TableRow({ children: [cell("Model Mimarisi", 2340), cell("YOLOv11", 1800), cell("GRU (Sequence)", 3000), cell("Temporal", 2220)] }),
      new TableRow({ children: [cell("Avatar", 2340), cell("Yok", 1800), cell("3D Avatar (101 poz)", 3000), cell("Yeni", 2220)] }),
      new TableRow({ children: [cell("Mobil Uygulama", 2340), cell("Yok", 1800), cell("Flutter (Android)", 3000), cell("Yeni", 2220)] }),
      new TableRow({ children: [cell("NLP Deste\u011Fi", 2340), cell("Yok", 1800), cell("T\u00FCrk\u00E7e c\u00FCmle \u00E7eviri", 3000), cell("Yeni", 2220)] }),
      new TableRow({ children: [cell("\u0130leti\u015Fim Y\u00F6n\u00FC", 2340), cell("Tek y\u00F6n", 1800), cell("\u00C7ift y\u00F6n", 3000), cell("Geli\u015Fme", 2220)] }),
    ]
  }),
  emptyP(),

  subheading("3.9 Kar\u015F\u0131la\u015F\u0131lan Zorluklar ve \u00C7\u00F6z\u00FCmler"),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Zorluk", 3120), headerCell("\u00C7\u00F6z\u00FCm", 6240)] }),
      new TableRow({ children: [cell("YOLO ile dinamik hareketlerin yakalanamamas\u0131", 3120), cell("GRU sequence modeline ge\u00E7i\u015F", 6240)] }),
      new TableRow({ children: [cell("Manuel veri toplaman\u0131n yetersiz kalmas\u0131", 3120), cell("Video tabanl\u0131 otomatik toplama + augmentation", 6240)] }),
      new TableRow({ children: [cell("T\u00FCrk\u00E7e ek yap\u0131s\u0131n\u0131n karma\u015F\u0131kl\u0131\u011F\u0131", 3120), cell("115+ ek destekleyen suffix stripping sistemi", 6240)] }),
      new TableRow({ children: [cell("Mobil kamera y\u00F6n sorunlar\u0131", 3120), cell("EXIF orientation d\u00FCzeltmesi + letterbox padding", 6240)] }),
      new TableRow({ children: [cell("Avatar poz ge\u00E7i\u015Flerinin sert olmas\u0131", 3120), cell("Smootherstep easing fonksiyonu", 6240)] }),
      new TableRow({ children: [cell("WebView\u2013Flutter veri aktar\u0131m\u0131", 3120), cell("LocalServer proxy + JavaScript bridge", 6240)] }),
    ]
  }),
  emptyP(),

  subheading("3.10 Teknoloji Y\u0131\u011F\u0131n\u0131 (Bahar D\u00F6nemi Eklemeleri)"),
  new Table({
    width: { size: 9360, type: WidthType.DXA }, columnWidths: [3120, 6240],
    rows: [
      new TableRow({ children: [headerCell("Teknoloji", 3120), headerCell("Kullan\u0131m Amac\u0131", 6240)] }),
      new TableRow({ children: [cell("Flutter (Dart)", 3120), cell("Android mobil uygulama geli\u015Ftirme", 6240)] }),
      new TableRow({ children: [cell("Three.js", 3120), cell("3D avatar render ve animasyon (WebGL)", 6240)] }),
      new TableRow({ children: [cell("Flask + SocketIO", 3120), cell("WebSocket destekli backend sunucu", 6240)] }),
      new TableRow({ children: [cell("GRU (PyTorch)", 3120), cell("Sequence tabanl\u0131 i\u015Faret dili tan\u0131ma", 6240)] }),
      new TableRow({ children: [cell("Blender", 3120), cell("3D karakter modeli ve poz kaydetme", 6240)] }),
      new TableRow({ children: [cell("WebView (Flutter)", 3120), cell("Mobilde 3D avatar g\u00F6r\u00FCnt\u00FCleme", 6240)] }),
    ]
  }),
  emptyP(),

  subheading("3.11 \u00D6\u011Frenilen Dersler (Bahar D\u00F6nemi)"),
  bullet("Sequence tabanl\u0131 modeller (GRU/LSTM), object detection'a g\u00F6re dinamik hareketleri \u00E7ok daha iyi yakalar"),
  bullet("Data augmentation, s\u0131n\u0131rl\u0131 veri ile e\u011Fitimde kritik \u00F6neme sahiptir"),
  bullet("Mobil ve web aras\u0131ndaki farklar (kamera y\u00F6n\u00FC, EXIF, image size) \u00F6nceden planlanmal\u0131d\u0131r"),
  bullet("T\u00FCrk\u00E7e gibi sondan eklemeli dillerde NLP \u00E7\u00F6z\u00FCm\u00FC basit string matching ile ba\u015Flang\u0131\u00E7 d\u00FCzeyinde \u00E7al\u0131\u015Ft\u0131r\u0131labilir"),
  bullet("3D avatar animasyonunda smootherstep easing, linear interpolation'a g\u00F6re \u00E7ok daha do\u011Fal g\u00F6r\u00FCn\u00FCr"),
  emptyP(),
];

// ========== SONUÇ ==========
const sonuc = [
  separator(),
  heading("Sonu\u00E7 ve Gelecek \u00C7al\u0131\u015Fmalar"),
  p(normal("SignBridge projesi, g\u00FCz ve bahar d\u00F6nemleri boyunca \u00F6nemli bir geli\u015Fim g\u00F6stermi\u015Ftir. G\u00FCz d\u00F6neminde 10 s\u0131n\u0131fl\u0131 (0-9) rakam tan\u0131ma ve 20 kelimelik statik poz tespiti ile ba\u015Flayan proje, bahar d\u00F6neminde 100 kelimelik sequence tabanl\u0131 dinamik hareket tan\u0131ma, 3D avatar animasyon sistemi ve Flutter mobil uygulama ile kapsaml\u0131 bir \u00E7ift y\u00F6nl\u00FC \u00E7eviri platformuna d\u00F6n\u00FC\u015Fm\u00FC\u015Ft\u00FCr.")),
  emptyP(),
  p(bold("Elde Edilen Ba\u015Far\u0131lar:")),
  bullet("100 kelime GRU tabanl\u0131 tan\u0131ma: %80.3 validation accuracy (70392 augmented sequence)"),
  bullet("3D avatar ile 101 kelimelik T\u0130D animasyonu (Metin \u2192 T\u0130D y\u00F6n\u00FC)"),
  bullet("T\u00FCrk\u00E7e NLP: 115+ ek destekli suffix stripping ile c\u00FCmle \u00E7evirisi"),
  bullet("Flutter mobil uygulama: Kamera + Avatar modlar\u0131, offline NLP deste\u011Fi"),
  bullet("Web uygulamas\u0131: WebSocket tabanl\u0131 ger\u00E7ek zamanl\u0131 tan\u0131ma ve c\u00FCmle olu\u015Fturma"),
  bullet("\u00C7ift y\u00F6nl\u00FC ileti\u015Fim: T\u0130D\u2192Metin ve Metin\u2192T\u0130D ayn\u0131 platformda"),
  emptyP(),
  p(bold("Gelecek \u00C7al\u0131\u015Fmalar:")),
  bullet("Kelime say\u0131s\u0131n\u0131n 200-500 aral\u0131\u011F\u0131na \u00E7\u0131kar\u0131lmas\u0131"),
  bullet("Transformer tabanl\u0131 model mimarisi ile do\u011Fruluk art\u0131r\u0131m\u0131"),
  bullet("On-device inference (TFLite/ONNX) ile sunucu ba\u011F\u0131ms\u0131z \u00E7al\u0131\u015Fma"),
  bullet("iOS platformu deste\u011Fi"),
  bullet("C\u00FCmle d\u00FCzeyinde s\u00FCrekli \u00E7eviri (continuous recognition)"),
  bullet("Parmak heceleme (fingerspelling) deste\u011Fi"),
  bullet("Daha geli\u015Fmi\u015F NLP: morfolojik analiz ve c\u00FCmle yapma"),
  bullet("Avatar i\u00E7in el parmak animasyonu geli\u015Ftirme"),
  emptyP(),
  p(normal("Proje, T\u00FCrkiye\u2019deki i\u015Fitme engelli bireylerin g\u00FCnl\u00FCk ileti\u015Fim ihtiya\u00E7lar\u0131na teknolojik bir \u00E7\u00F6z\u00FCm sunmay\u0131 hedeflemektedir. Mevcut \u00E7al\u0131\u015Fma, 100 kelimelik bir kelime da\u011Farc\u0131\u011F\u0131 ile temel ileti\u015Fim senaryolar\u0131n\u0131 kar\u015F\u0131layabilecek d\u00FCzeyde olup, gelecek geli\u015Ftirmeler ile daha kapsaml\u0131 bir kullan\u0131m alan\u0131 hedeflenmektedir.")),
  emptyP(),
];

// ========== KAYNAKLAR ==========
const kaynaklar = [
  separator(),
  heading("Kaynaklar"),
  p(normal("[1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. IEEE CVPR.")),
  p(normal("[2] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation, 9(8), 1735-1780.")),
  p(normal("[3] Cho, K., van Merrienboer, B., Gulcehre, C., et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. arXiv:1406.1078.")),
  p(normal("[4] Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.")),
  p(normal("[5] Lugaresi, C., Tang, J., Nash, H., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172.")),
  p(normal("[6] Redmon, J., & Farhadi, A. (2018). YOLOv3: An Incremental Improvement. arXiv:1804.02767.")),
  p(normal("[7] Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8. GitHub: ultralytics/ultralytics.")),
  p(normal("[8] Cabrera, M. E., & Bodla, N. (2021). Sign Language Recognition with Temporal Convolutional Networks. IEEE Access.")),
  p(normal("[9] T\u00FCrkiye \u0130statistik Kurumu (2011). N\u00FCfus ve Konut Ara\u015Ft\u0131rmas\u0131 \u2013 Engelli Bireylere \u0130li\u015Fkin Sonu\u00E7lar.")),
  p(normal("[10] BM Engelli Haklar\u0131 S\u00F6zle\u015Fmesi (CRPD). (2006). Convention on the Rights of Persons with Disabilities.")),
  p(normal("[11] Donckers, N., Cabrera, M. E., & Escalera, S. (2020). Survey on Sign Language Recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence.")),
  p(normal("[12] Zhang, F., et al. (2020). MediaPipe Hands: On-device Real-time Hand Tracking. arXiv:2006.10214.")),
  emptyP(),
];

// ========== EKLER ==========
const ekler = [
  separator(),
  heading("Ekler"),

  // --- Güz Dönemi ---
  emptyP(),
  subheading("G\u00FCz D\u00F6nemi Ekleri"),

  p(bold("EK-1: Multimodal CNN Mimari Diyagram\u0131")),
  ...tryImage("guz_ek_1.png", "Multimodal CNN \u00E7ift kanall\u0131 mimari \u2013 RGB g\u00F6r\u00FCnt\u00FC + MediaPipe Landmark fusion", 400, 480),

  p(bold("EK-2: FastAPI Backend API Endpoint \u015Eemas\u0131")),
  ...tryImage("guz_ek_2.png", "FastAPI REST ve WebSocket endpoint mimarisi", 420, 380),

  p(bold("EK-3: YOLO E\u011Fitim Grafikleri (Loss, Precision, Recall, mAP)")),
  ...tryImage("guz_ek_3.png", "YOLOv11-nano e\u011Fitim s\u00FCreci \u2013 10 farkl\u0131 metrik grafi\u011Fi", 520, 280),

  p(bold("EK-4: YOLO Confusion Matrix (20 S\u0131n\u0131f)")),
  ...tryImage("guz_ek_4.png", "YOLO 20 kelime confusion matrix", 450, 380),

  p(bold("EK-5: YOLO Normalized Confusion Matrix")),
  ...tryImage("guz_ek_5.png", "YOLO normalize edilmi\u015F confusion matrix", 450, 380),

  p(bold("EK-6: Precision-Recall E\u011Frisi (mAP@0.5: 0.994)")),
  ...tryImage("guz_ek_6.png", "S\u0131n\u0131f bazl\u0131 precision-recall e\u011Frisi \u2013 mAP@0.5: 0.994", 480, 340),

  p(bold("EK-7: F1-Confidence E\u011Frisi")),
  ...tryImage("guz_ek_7.png", "S\u0131n\u0131f bazl\u0131 F1-confidence e\u011Frisi \u2013 optimal threshold 0.459", 480, 360),

  p(bold("EK-8: Veri Seti \u0130statistikleri ve Da\u011F\u0131l\u0131m Grafikleri")),
  ...tryImage("guz_ek_8.png", "Roboflow veri seti \u2013 s\u0131n\u0131f da\u011F\u0131l\u0131m\u0131, bounding box boyut ve konum analizi", 480, 380),

  p(bold("EK-9: YOLO Tespit \u00D6rnekleri (\u00C7oklu G\u00F6r\u00FCnt\u00FC)")),
  ...tryImage("guz_ek_9.png", "YOLO ger\u00E7ek zamanl\u0131 tespit \u00F6rnekleri \u2013 farkl\u0131 a\u00E7\u0131 ve ko\u015Fullar", 480, 380),

  p(bold("EK-10: YOLO Tespit Detay \u00D6rnekleri")),
  ...tryImage("guz_ek_10.png", "YOLO tek ki\u015Fi detayl\u0131 tespit \u00F6rnekleri \u2013 bounding box ve confidence", 480, 380),

  p(bold("EK-11: Web Aray\u00FCz\u00FC \u2013 Ana Sayfa (Kamera ve \u0130statistikler)")),
  ...tryImage("guz_ek_11.png", "Web aray\u00FCz\u00FC ana sayfa \u2013 kamera, tahmin, istatistik paneli", 500, 300),

  p(bold("EK-12: Web Aray\u00FCz\u00FC \u2013 Hastane \u0130leti\u015Fim Modu")),
  ...tryImage("guz_ek_12.png", "Hastane ileti\u015Fim modu \u2013 haz\u0131r mesaj butonlar\u0131 ve T\u0130D animasyonu", 500, 260),

  p(bold("EK-13: Web Aray\u00FCz\u00FC \u2013 Unity Avatar Sayfas\u0131")),
  ...tryImage("guz_ek_13.png", "Unity WebGL 3D avatar \u2013 kamera ve landmark g\u00F6r\u00FCnt\u00FCleme", 500, 280),

  p(bold("EK-14: Unity Editor \u2013 3D Avatar ve Animasyon Sistemi")),
  ...tryImage("guz_ek_14.png", "Unity Editor \u2013 humanoid avatar, kemik sistemi ve animasyon kontrol\u00FC", 520, 260),

  // Ek güz grafikleri
  p(bold("EK-15: YOLO E\u011Fitim Ge\u00E7mi\u015Fi (Detayl\u0131)")),
  ...tryImage("guz_egitim_grafikleri.png", "YOLO detayl\u0131 e\u011Fitim ge\u00E7mi\u015Fi grafi\u011Fi", 500, 350),

  p(bold("EK-16: LSTM 5 Kelime Confusion Matrix")),
  ...tryImage("guz_confusion_matrix.png", "LSTM 5 kelime confusion matrix \u2013 deneysel sonu\u00E7lar", 450, 350),

  // --- Bahar Dönemi ---
  emptyP(),
  subheading("Bahar D\u00F6nemi Ekleri"),

  p(bold("EK-17: Poz Edit\u00F6r\u00FC Aray\u00FCz\u00FC")),
  ...tryImage("poz_editoru.png", "Poz edit\u00F6r\u00FC \u2013 kemik rotasyon slider\u2019lar\u0131, keyframe sistemi, Kaydet/\u00D6nizle", 520, 330),

  p(bold("EK-18: Veri Toplama Aray\u00FCz\u00FC (Hareket Kaydedici)")),
  ...tryImage("veri_toplama.png", "MediaPipe tabanl\u0131 hareket kaydedici \u2013 kelime ad\u0131, frame say\u0131s\u0131, geri say\u0131m ayarlar\u0131", 520, 330),

  p(bold("EK-19: GRU E\u011Fitim Grafikleri (Loss ve Accuracy)")),
  ...tryImage("gru_training_history.png", "GRU model e\u011Fitim s\u00FCreci \u2013 Kay\u0131p (Loss) ve Do\u011Fruluk (Accuracy) e\u011Frileri", 500, 250),

  p(bold("EK-20: GRU Confusion Matrix (100 Kelime)")),
  ...tryImage("gru_confusion_matrix.png", "GRU 100 kelime confusion matrix \u2013 %80.3 validation do\u011Fruluk", 500, 420),

  p(bold("EK-21: Web Kamera \u2013 Ger\u00E7ek Zamanl\u0131 Tahmin")),
  ...tryImage("web_camera.png", "Bahar d\u00F6nemi web aray\u00FCz\u00FC \u2013 GRU tabanl\u0131 ger\u00E7ek zamanl\u0131 tahmin", 500, 300),

  p(bold("EK-22: Web Avatar \u2013 MERHABA Animasyonu")),
  ...tryImage("web_avatar_merhaba.png", "3D avatar MERHABA kelimesi animasyonu \u2013 Three.js render", 500, 300),

  p(bold("EK-23: Web Avatar \u2013 BABA Animasyonu")),
  ...tryImage("web_avatar_baba.png", "3D avatar BABA kelimesi animasyonu \u2013 smootherstep ge\u00E7i\u015F", 500, 300),

  p(bold("EK-24: Flutter Kamera Modu")),
  ...tryImage("mobile_camera", "Flutter mobil kamera \u2013 ger\u00E7ek zamanl\u0131 T\u0130D tan\u0131ma", 250, 450),

  p(bold("EK-25: Flutter Avatar \u2013 BUG\u00DCN")),
  ...tryImage("mobile_avatar_bugun", "Flutter mobil avatar \u2013 BUG\u00DCN kelimesi animasyonu", 250, 450),

  p(bold("EK-26: Flutter Avatar \u2013 EV")),
  ...tryImage("mobile_avatar_ev", "Flutter mobil avatar \u2013 EV kelimesi animasyonu", 250, 450),

  p(bold("EK-27: Flutter Kelime Listesi Paneli")),
  ...tryImage("mobile_kelime_listesi", "Flutter mobil kelime listesi \u2013 101 kelime, arama ve oynatma", 250, 450),

  p(bold("EK-28: Flutter Debug Konsol")),
  ...tryImage("mobile_debug", "Flutter debug ekran\u0131 \u2013 a\u011F istekleri ve tahmin sonu\u00E7lar\u0131", 250, 450),
];

const doc = new Document({
  compatibility: {
    doNotExpandShiftReturn: true, // Shift+Enter aralığını sabit tut
  },
  styles: {
    default: {
      document: {
        run: { font: "Calibri", size: 22 }, // 11pt Calibri - her PC'de mevcut
        paragraph: {
          spacing: { line: 276, lineRule: "auto", after: 80 }, // Sabit 1.15 satır aralığı
        }
      }
    }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({ children: [
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "SignBridge \u2013 T\u0130D Yapay Zek\u00E2 \u00C7eviri Uygulamas\u0131", size: 18, color: "999999" })] })
      ]})
    },
    footers: {
      default: new Footer({ children: [
        new Paragraph({ alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Sayfa ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })
      ]})
    },
    children: [...kisim1, ...kisim2, ...kisim3, ...sonuc, ...kaynaklar, ...ekler]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = "C:/Users/leven/OneDrive/Desktop/SignBridge_Rapor_Genisletilmis.docx";
  fs.writeFileSync(outPath, buffer);
  console.log("Rapor olusturuldu: " + outPath);
  console.log("Boyut: " + (buffer.length / 1024).toFixed(1) + " KB");
});
