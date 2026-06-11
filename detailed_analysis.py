"""Detaylı test verisi analizi — sınıf bazlı doğruluk ve confusion analizi"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from pathlib import Path
from collections import Counter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class SignTransformerPro(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, dropout=0.35):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Linear(input_size, d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.conv_block = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        el = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(el, num_layers=num_layers)
        self.pool_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_model//4), nn.Tanh(), nn.Linear(d_model//4, 1))
            for _ in range(4)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model*5), nn.Dropout(dropout),
            nn.Linear(d_model*5, d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*2, d_model), nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(d_model, num_classes))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.input_conv(x)
        x = x + self.conv_block(x.transpose(1, 2)).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.transformer(self.pos_encoder(x))
        seq = x[:, 1:]
        pooled = [F.softmax(h(seq), dim=1) * seq for h in self.pool_heads]
        pooled = [p.sum(dim=1) for p in pooled]
        return self.classifier(torch.cat(pooled + [seq.mean(dim=1)], dim=1))


def main():
    model_dir = Path('autsl_transformer/model')
    mean = np.load(model_dir / 'norm_mean.npy').flatten()
    std = np.maximum(np.load(model_dir / 'norm_std.npy').flatten(), 1e-8)

    with open(model_dir / 'label_map.json', 'r', encoding='utf-8') as f:
        lm = {int(k): v for k, v in json.load(f).items()}

    cp = torch.load(model_dir / 'autsl_ema_model.pt', map_location='cpu', weights_only=False)
    cfg = cp['config']
    model = SignTransformerPro(
        cfg['input_size'], cfg['d_model'], cfg['nhead'],
        cfg['num_layers'], cfg['num_classes'])
    model.load_state_dict(cp['model_state_dict'])
    model.eval()

    data = np.load('autsl_transformer/all_test_packed.npz')
    X, y = data['x'], data['y']

    correct = 0
    total = 0
    class_correct = Counter()
    class_total = Counter()
    confusions = Counter()
    all_confs = []

    for i in range(0, len(X), 128):
        bx = X[i:i+128]
        by = y[i:i+128]
        bn = (bx - mean) / std
        with torch.no_grad():
            out = model(torch.tensor(bn, dtype=torch.float32))
            probs = F.softmax(out, dim=1)
            conf, pred = probs.max(1)
        for j in range(len(by)):
            t = int(by[j])
            p = pred[j].item()
            c = conf[j].item()
            class_total[t] += 1
            all_confs.append((c, p == t))
            if p == t:
                correct += 1
                class_correct[t] += 1
            else:
                confusions[(t, p)] += 1
            total += 1

    print(f"{'='*60}")
    print(f"  GENEL DOGRULUK: {100*correct/total:.2f}% ({correct}/{total})")
    print(f"{'='*60}")

    # Confidence analysis
    confs = np.array([c for c, _ in all_confs])
    corr_mask = np.array([ok for _, ok in all_confs])
    print(f"\nConfidence Analizi:")
    print(f"  Ortalama conf (dogru): {confs[corr_mask].mean():.3f}")
    print(f"  Ortalama conf (yanlis): {confs[~corr_mask].mean():.3f}")
    for thr in [0.3, 0.5, 0.7, 0.9]:
        mask = confs >= thr
        if mask.sum() > 0:
            acc = corr_mask[mask].mean() * 100
            coverage = mask.mean() * 100
            print(f"  conf >= {thr}: dogruluk={acc:.1f}%, kapsam={coverage:.1f}%")

    # Worst classes
    accs = []
    for c in range(226):
        if class_total[c] > 0:
            acc = 100 * class_correct[c] / class_total[c]
            accs.append((c, acc, class_correct[c], class_total[c]))
    accs.sort(key=lambda x: x[1])

    print(f"\nEN KOTU 20 SINIF:")
    for c, a, cr, tot in accs[:20]:
        name = lm.get(c, "?")
        print(f"  [{c:3d}] {name:25s}: {a:5.1f}% ({cr}/{tot})")

    print(f"\nEN IYI 10 SINIF:")
    for c, a, cr, tot in accs[-10:]:
        name = lm.get(c, "?")
        print(f"  [{c:3d}] {name:25s}: {a:5.1f}% ({cr}/{tot})")

    # Accuracy distribution
    acc_vals = [a for _, a, _, _ in accs]
    print(f"\nSinif Dogruluk Dagilimi:")
    print(f"  0%: {sum(1 for a in acc_vals if a == 0)} sinif")
    print(f"  1-25%: {sum(1 for a in acc_vals if 0 < a <= 25)} sinif")
    print(f"  25-50%: {sum(1 for a in acc_vals if 25 < a <= 50)} sinif")
    print(f"  50-75%: {sum(1 for a in acc_vals if 50 < a <= 75)} sinif")
    print(f"  75-100%: {sum(1 for a in acc_vals if a > 75)} sinif")

    # Most confused pairs
    print(f"\nEN COK KARISTIRILAN CIFTLER:")
    for (tc, pc), cnt in confusions.most_common(15):
        tn = lm.get(tc, "?")
        pn = lm.get(pc, "?")
        print(f"  {tn:22s} -> {pn:22s}: {cnt}x")

    # Bidirectional confusion
    print(f"\nCIFT YONLU KARISTIRILMA:")
    seen = set()
    bi_conf = []
    for (tc, pc), cnt in confusions.items():
        if (pc, tc) in confusions and (min(tc,pc), max(tc,pc)) not in seen:
            seen.add((min(tc,pc), max(tc,pc)))
            bi_cnt = cnt + confusions[(pc, tc)]
            bi_conf.append((tc, pc, cnt, confusions[(pc, tc)], bi_cnt))
    bi_conf.sort(key=lambda x: -x[4])
    for tc, pc, c1, c2, total_c in bi_conf[:10]:
        tn = lm.get(tc, "?")
        pn = lm.get(pc, "?")
        print(f"  {tn:20s} <-> {pn:20s}: {c1}+{c2}={total_c}x")


if __name__ == '__main__':
    main()
