"""
=====================================================================================================
PART 2 of 2  --  SSPathNet on PlantVillage : ABLATION STUDY (+ optional cross-dataset test)
=====================================================================================================
Kaggle set-up : Add the PlantVillage dataset to the notebook, Accelerator = GPU T4, Internet not needed.
Run           : paste into a notebook cell (or `!python plantvillage_part2.py`) and run.
Independent   : this file does NOT depend on Part 1 in any way; every result is written to
                /kaggle/working/part2_results/  (results.json, summary.csv, per-run folders).
                It re-trains its own full-model reference (A0) under exactly the same conditions as the ablations,
                and uses the same deterministic train/val/test split as Part 1, so the numbers are comparable.

What this part runs (in priority order; a run is skipped if the remaining time budget cannot fit it):
  A0  full SSPathNet (reference)
  A1  stem: RGB only                      A2  stem: RGB + HSV            A3  stem: RGB + DCT
  A4  backbone: no deformable sampling    A5  backbone: no attention (conv/inception blocks only)
  A6  aggregation: none (last stage only) A7  aggregation: plain concatenation (instead of TCFA)
  A8  head: no center loss                A9  head: GELU instead of learnable adaptive activation
  + optional zero-shot cross-dataset test (PlantDoc etc.) with the trained A0 model, if CROSS_DATASET_PATH is set.

TIP: first set SMOKE_TEST = True (about 5 minutes) to verify everything works on your Kaggle session,
     then set it back to False for the real run.
=====================================================================================================
"""

# ============================ USER SETTINGS ============================
RUN_NAME       = "part2_results"
DATASET_PATH   = ""        # "" = auto-detect under /kaggle/input.  Or e.g. "/kaggle/input/plantvillage-dataset/color"
SMOKE_TEST     = False     # True = tiny quick test of the whole pipeline
SMOKE_PER_CLASS = 30

EPOCHS         = 30        # same as Part 1, so A0 is comparable to Part 1's SSPathNet_main
# Optional cross-dataset test: attach e.g. the PlantDoc dataset to the notebook and put its path here.
CROSS_DATASET_PATH = ""    # e.g. "/kaggle/input/plantdoc-dataset"
CROSS_CLASS_MAP    = {}    # optional manual overrides {"foreign folder name": "PlantVillage class name"}

IMG_SIZE       = 160       # multiple of 32
BATCH_SIZE     = 128       # lower to 96/64 if you ever see a CUDA out-of-memory error
LR             = 1e-3
WEIGHT_DECAY   = 0.05
WARMUP_EPOCHS  = 2
LABEL_SMOOTH   = 0.1
CENTER_WEIGHT  = 0.005
SEED           = 42
TIME_BUDGET_H  = 11.0      # safety margin below Kaggle's 12 h session limit
OUT_ROOT       = "/kaggle/working"
# ======================================================================
if SMOKE_TEST:
    EPOCHS, TIME_BUDGET_H, BATCH_SIZE = 1, 0.5, 32

# =====================================================================
# CORE (identical in both part files)                                  
# =====================================================================
import os, sys, time, json, math, random, csv, gc, copy, queue, threading, warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn import metrics as skm

warnings.filterwarnings("ignore")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
T_START = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_ROOT = Path(OUT_ROOT) / RUN_NAME
RUN_ROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else Path("/tmp")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
assert IMG_SIZE % 32 == 0, "IMG_SIZE must be a multiple of 32 (160 recommended)"


# ---------------------------------------------------------------------
# Logging / time budget
# ---------------------------------------------------------------------
def elapsed_h():
    return (time.time() - T_START) / 3600.0


def time_left_s():
    return TIME_BUDGET_H * 3600.0 - (time.time() - T_START)


def log(*a):
    msg = f"[{elapsed_h():5.2f}h] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(RUN_ROOT / "log.txt", "a") as f:
        f.write(msg + "\n")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------
def dir_has_images(d):
    try:
        for f in os.scandir(d):
            if f.is_file() and os.path.splitext(f.name)[1].lower() in IMG_EXT:
                return True
    except Exception:
        pass
    return False


def find_class_dirs(base, min_classes, max_depth=5):
    """Directories that directly contain >= min_classes sub-folders holding images."""
    base = Path(base)
    d0 = len(base.parts)
    cands = []
    for dp, dns, _ in os.walk(base, followlinks=True):
        if len(Path(dp).parts) - d0 >= max_depth:
            dns[:] = []
        if len(dns) >= min_classes:
            good = [d for d in dns if dir_has_images(Path(dp) / d)]
            if len(good) >= min_classes:
                cands.append((Path(dp), len(good)))
    cands.sort(key=lambda t: str(t[0]))
    return cands


def pick_dataset_root(user_path):
    base = user_path if user_path else "/kaggle/input"
    cands = find_class_dirs(base, 30)
    if not cands:
        cands = find_class_dirs(base, 10)
    if not cands:
        raise FileNotFoundError(
            f"No folder with class sub-folders of images found under '{base}'. "
            "Set DATASET_PATH at the top of the script to the folder whose sub-folders are the disease classes.")
    log("Dataset candidates found:")
    for p, n in cands:
        log(f"   {p}  ({n} class folders)")
    chosen = None
    for p, n in cands:                       # prefer the RGB 'color' version
        if p.name.lower() == "color":
            chosen = (p, n); break
    if chosen is None:
        for p, n in cands:
            if n == 38:
                chosen = (p, n); break
    if chosen is None:
        chosen = max(cands, key=lambda t: t[1])
    log(f"==> Using dataset root: {chosen[0]}  ({chosen[1]} classes)")
    if chosen[1] != 38:
        log(f"WARNING: expected 38 classes, found {chosen[1]}. Set DATASET_PATH if this is not the right folder.")
    return chosen[0]


def _load_img(args):
    p, S = args
    try:
        from PIL import Image
        with Image.open(p) as im:
            im = im.convert("RGB").resize((S, S), Image.BILINEAR)
            return np.asarray(im, dtype=np.uint8)
    except Exception:
        return None


def load_folder_images(root, class_names, S, max_per_class=None, desc="images"):
    """Read every image of the given class folders -> (uint8 N,S,S,3 ; int64 N ; kept_paths)."""
    paths, labels = [], []
    for ci, cn in enumerate(class_names):
        fs = sorted([f for f in os.listdir(Path(root) / cn) if os.path.splitext(f)[1].lower() in IMG_EXT])
        if max_per_class:
            fs = fs[:max_per_class]
        paths += [str(Path(root) / cn / f) for f in fs]
        labels += [ci] * len(fs)
    N = len(paths)
    X = np.empty((N, S, S, 3), dtype=np.uint8)
    ok = np.ones(N, dtype=bool)
    t0 = time.time()
    with Pool(processes=max(1, os.cpu_count() or 1)) as pool:
        for i, arr in enumerate(pool.imap(_load_img, [(p, S) for p in paths], chunksize=64)):
            if arr is None:
                ok[i] = False
            else:
                X[i] = arr
            if (i + 1) % 10000 == 0:
                log(f"   loaded {i + 1}/{N} {desc} ({time.time() - t0:.0f}s)")
    y = np.asarray(labels, dtype=np.int64)
    if not ok.all():
        log(f"WARNING: {(~ok).sum()} unreadable images skipped")
        X, y = X[ok], y[ok]
        paths = [p for p, k in zip(paths, ok) if k]
    return X, y, paths


DATA_X = None      # uint8 tensor  N,S,S,3  (CPU)
DATA_Y = None      # int64 tensor  N
CLASSES = []


def load_dataset():
    global DATA_X, DATA_Y, CLASSES
    root = pick_dataset_root(DATASET_PATH)
    CLASSES = sorted([d for d in os.listdir(root) if dir_has_images(root / d)])
    mpc = SMOKE_PER_CLASS if SMOKE_TEST else None
    sig = f"{root}|{IMG_SIZE}|{len(CLASSES)}|{mpc}"
    cx, cy, cs = CACHE_DIR / "pv_x.npy", CACHE_DIR / "pv_y.npy", CACHE_DIR / "pv_sig.txt"
    if cx.exists() and cy.exists() and cs.exists() and cs.read_text() == sig:
        log("Loading cached tensors ...")
        X, y = np.load(cx), np.load(cy)
    else:
        log(f"Decoding + resizing images to {IMG_SIZE}x{IMG_SIZE} (one-off, a few minutes) ...")
        X, y, _ = load_folder_images(root, CLASSES, IMG_SIZE, mpc)
        try:
            np.save(cx, X); np.save(cy, y); cs.write_text(sig)
        except Exception as e:
            log("cache write failed (continuing without cache):", e)
    DATA_X, DATA_Y = torch.from_numpy(X), torch.from_numpy(y)
    (RUN_ROOT / "classes.json").write_text(json.dumps(CLASSES, indent=1))
    cnt = np.bincount(y, minlength=len(CLASSES))
    log(f"Dataset: {len(y)} images, {len(CLASSES)} classes, per-class min/median/max = "
        f"{cnt.min()}/{int(np.median(cnt))}/{cnt.max()}, RAM {X.nbytes / 1e9:.2f} GB")


def fixed_split(val_frac=0.15, test_frac=0.15):
    """Stratified train/val/test split, deterministic (same in every part file)."""
    idx = np.arange(len(DATA_Y)); y = DATA_Y.numpy()
    tr, te = train_test_split(idx, test_size=test_frac, stratify=y, random_state=SEED)
    tr, va = train_test_split(tr, test_size=val_frac / (1 - test_frac), stratify=y[tr], random_state=SEED)
    return dict(train=tr, val=va, test=te)


def cv_splits(k):
    y = DATA_Y.numpy(); idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    for f, (trv, te) in enumerate(skf.split(idx, y)):
        tr, va = train_test_split(trv, test_size=0.10, stratify=y[trv], random_state=SEED + f)
        yield f, dict(train=tr, val=va, test=te)


# ---------------------------------------------------------------------
# Batch loader (CPU gather in a background thread) + GPU augmentation
# ---------------------------------------------------------------------
class BatchLoader:
    def __init__(self, idx, bs, shuffle, drop_last, seed=0):
        self.idx = torch.as_tensor(idx, dtype=torch.long)
        self.bs, self.shuffle, self.drop_last, self.seed = bs, shuffle, drop_last, seed
        self.epoch = 0

    def __len__(self):
        n = len(self.idx)
        return n // self.bs if self.drop_last else math.ceil(n / self.bs)

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch); self.epoch += 1
        order = self.idx[torch.randperm(len(self.idx), generator=g)] if self.shuffle else self.idx
        batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
        if self.drop_last and len(batches) and len(batches[-1]) < self.bs:
            batches = batches[:-1]
        q, stop = queue.Queue(maxsize=4), threading.Event()
        pin = DEV.type == "cuda"

        def worker():
            try:
                for b in batches:
                    if stop.is_set():
                        return
                    xb = DATA_X[b]
                    item = (xb.pin_memory() if pin else xb, DATA_Y[b])
                    while not stop.is_set():
                        try:
                            q.put(item, timeout=0.5); break
                        except queue.Full:
                            pass
                q.put(None)
            except Exception as e:  # pragma: no cover
                q.put(e)

        th = threading.Thread(target=worker, daemon=True); th.start()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()


def to_float(xb):
    """uint8 N,H,W,3 (cpu) -> float N,3,H,W in [0,1] on device."""
    return xb.to(DEV, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)


def gpu_augment(x):
    """Batch augmentation on the GPU: affine (rot/zoom/flip/shift), colour jitter, noise, cutout."""
    B, _, H, W = x.shape
    d = x.device
    r = lambda *s: torch.rand(*s, device=d)
    ang = (r(B) * 2 - 1) * math.radians(30)
    sc = 1.0 / (0.85 + 0.30 * r(B))
    fh = (r(B) < 0.5).float() * 2 - 1
    fv = (r(B) < 0.5).float() * 2 - 1
    tx, ty = (r(B) * 2 - 1) * 0.08, (r(B) * 2 - 1) * 0.08
    c, s = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([c * fh, -s, tx], 1), torch.stack([s * fh, c * fv, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    b = 1 + (r(B, 1, 1, 1) * 2 - 1) * 0.25
    x = x * b
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * (1 + (r(B, 1, 1, 1) * 2 - 1) * 0.25) + m
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * (1 + (r(B, 1, 1, 1) * 2 - 1) * 0.30) + gray
    x = x + 0.02 * torch.randn_like(x) * (r(B, 1, 1, 1) < 0.3).float()
    x = x.clamp_(0, 1)
    half = int(H * 0.15)
    ys = torch.arange(H, device=d).view(1, H, 1); xs = torch.arange(W, device=d).view(1, 1, W)
    cy = (r(B, 1, 1) * H).long(); cx = (r(B, 1, 1) * W).long()
    mask = ((ys - cy).abs() < half) & ((xs - cx).abs() < half) & (r(B, 1, 1) < 0.25)
    x = torch.where(mask.unsqueeze(1), torch.full_like(x, 0.5), x)
    return x


# =====================================================================
# MODEL: SSPathNet  (Spectro-Spatial Pathological Network)
# =====================================================================
class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__(); self.p = p

    def forward(self, x):
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        return x * x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep) / keep


class AdaptiveAct(nn.Module):
    """Learnable adaptive activation: f(x) = x * (a + (1-a) * sigmoid(beta*x)).
    a in (0,1) and beta>0 are learned per channel (a=0: Swish/SiLU family, a=1: identity)."""
    def __init__(self, ch):
        super().__init__()
        self.raw_beta = nn.Parameter(torch.full((ch,), 0.5413))   # softplus(0.5413) = 1
        self.raw_a = nn.Parameter(torch.full((ch,), -2.0))        # sigmoid(-2) ~ 0.12

    def forward(self, x):
        shp = [1, -1] + [1] * (x.dim() - 2)
        beta = F.softplus(self.raw_beta).view(shp)
        a = torch.sigmoid(self.raw_a).view(shp)
        return x * (a + (1 - a) * torch.sigmoid(beta * x))


class RGB2HSVFeat(nn.Module):
    """Continuous HSV: [s*cos(2*pi*h), s*sin(2*pi*h), s, v]  (no hue discontinuity)."""
    def forward(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        mx = x.max(1)[0]; mn = x.min(1)[0]; d = mx - mn; eps = 1e-6
        rc = (mx == r) & (d > eps)
        gc_ = (mx == g) & (d > eps) & ~rc
        bc = (d > eps) & ~rc & ~gc_
        h = torch.zeros_like(mx)
        h = torch.where(rc, torch.remainder((g - b) / (d + eps), 6.0), h)
        h = torch.where(gc_, (b - r) / (d + eps) + 2.0, h)
        h = torch.where(bc, (r - g) / (d + eps) + 4.0, h)
        h = h / 6.0
        s = d / (mx + eps)
        ang = 2 * math.pi * h
        return torch.stack([s * torch.cos(ang), s * torch.sin(ang), s, mx], 1)


class BlockDCT(nn.Module):
    """Fixed (non-learned) orthonormal 4x4 block-DCT on Y/Cb/Cr, stride 4 -> 48 channels at H/4."""
    def __init__(self, n=4):
        super().__init__()
        basis = torch.zeros(n * n, 1, n, n)
        for u in range(n):
            for v in range(n):
                au = math.sqrt(1.0 / n) if u == 0 else math.sqrt(2.0 / n)
                av = math.sqrt(1.0 / n) if v == 0 else math.sqrt(2.0 / n)
                for xx in range(n):
                    for yy in range(n):
                        basis[u * n + v, 0, xx, yy] = au * av * math.cos((2 * xx + 1) * u * math.pi / (2 * n)) \
                                                      * math.cos((2 * yy + 1) * v * math.pi / (2 * n))
        self.n = n
        self.register_buffer("w", basis.repeat(3, 1, 1, 1), persistent=False)          # (48,1,n,n)
        self.register_buffer("ycc", torch.tensor([[0.299, 0.587, 0.114],
                                                  [-0.1687, -0.3313, 0.5],
                                                  [0.5, -0.4187, -0.0813]]), persistent=False)
        self.out_ch = 3 * n * n

    def forward(self, x):
        ycc = torch.einsum("ij,bjhw->bihw", self.ycc, x)
        ycc = torch.cat([ycc[:, :1] - 0.5, ycc[:, 1:]], 1)
        z = F.conv2d(ycc, self.w, stride=self.n, groups=3)
        return torch.sign(z) * torch.log1p(4.0 * z.abs())


def conv_bn_act(ci, co, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(ci, co, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(co)]
    if act:
        layers.append(nn.GELU())
    return nn.Sequential(*layers)


class ChromaFreqStem(nn.Module):
    """Dual-stream Chromatic-Frequency stem: RGB | continuous-HSV | block-DCT -> gated fusion, output stride 4."""
    def __init__(self, out_ch=64, mode="full"):
        super().__init__()
        self.use_hsv = mode in ("rgb+hsv", "full")
        self.use_dct = mode in ("rgb+dct", "full")
        c = out_ch
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.rgb = nn.Sequential(conv_bn_act(3, c // 2, 3, 2), conv_bn_act(c // 2, c, 3, 2))
        n = 1
        if self.use_hsv:
            self.hsv_t = RGB2HSVFeat()
            self.hsv = nn.Sequential(conv_bn_act(4, c // 2, 3, 2), conv_bn_act(c // 2, c, 3, 2)); n += 1
        if self.use_dct:
            self.dct_t = BlockDCT(4)
            self.dct = nn.Sequential(nn.BatchNorm2d(self.dct_t.out_ch), conv_bn_act(self.dct_t.out_ch, c, 1),
                                     conv_bn_act(c, c, 3, 1, groups=c)); n += 1
        self.n = n
        if n > 1:
            self.gate = nn.Sequential(nn.Linear(n * c, c), nn.GELU(), nn.Linear(c, n * c))
        self.out = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):                                   # x: float32 RGB in [0,1]
        with torch.autocast(device_type="cuda", enabled=False):
            x = x.float()
            hsv = self.hsv_t(x) if self.use_hsv else None
            dct = self.dct_t(x) if self.use_dct else None
        feats = [self.rgb((x - self.mean) / self.std)]
        if self.use_hsv:
            feats.append(self.hsv(hsv))
        if self.use_dct:
            feats.append(self.dct(dct))
        if self.n > 1:
            pooled = torch.cat([f.mean((2, 3)) for f in feats], 1)
            g = self.gate(pooled).view(x.shape[0], self.n, -1).softmax(1)
            f = sum(g[:, i].unsqueeze(-1).unsqueeze(-1) * feats[i] for i in range(self.n))
        else:
            f = feats[0]
        return self.out(f)


class DeformDWConv(nn.Module):
    """Deformable depthwise KxK conv implemented with grid_sample (offsets predicted per offset-group).
    Offsets are zero-initialised -> starts as an ordinary depthwise conv; bounded to +-R pixels."""
    def __init__(self, ch, k=3, off_groups=4, deform=True, R=3.0):
        super().__init__()
        self.ch, self.k, self.K, self.deform, self.R = ch, k, k * k, deform, R
        self.G = off_groups if ch % off_groups == 0 else 1
        self.weight = nn.Parameter(torch.randn(ch, self.K) / math.sqrt(self.K))
        self.bias = nn.Parameter(torch.zeros(ch))
        if deform:
            self.off_dw = nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False)
            self.off_pw = nn.Conv2d(ch, 2 * self.K * self.G, 1)
            nn.init.zeros_(self.off_pw.weight); nn.init.zeros_(self.off_pw.bias)
        p = k // 2
        ys, xs = torch.meshgrid(torch.arange(-p, p + 1), torch.arange(-p, p + 1), indexing="ij")
        self.register_buffer("tap", torch.stack([xs.flatten(), ys.flatten()], 1).float(), persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        if not self.deform:
            return F.conv2d(x, self.weight.view(C, 1, self.k, self.k).to(x.dtype), self.bias.to(x.dtype),
                            padding=self.k // 2, groups=C)
        G, K = self.G, self.K
        off = self.off_pw(self.off_dw(x)).float()
        off = self.R * torch.tanh(off / self.R)
        off = off.view(B, G, K, 2, H, W)
        xs = torch.arange(W, device=x.device, dtype=torch.float32).view(1, 1, 1, W)
        ys = torch.arange(H, device=x.device, dtype=torch.float32).view(1, 1, H, 1)
        xg = x.float().reshape(B * G, C // G, H, W)
        out = 0
        for t in range(K):
            px = xs + self.tap[t, 0] + off[:, :, t, 0]
            py = ys + self.tap[t, 1] + off[:, :, t, 1]
            grid = torch.stack([(2 * px + 1) / W - 1, (2 * py + 1) / H - 1], -1).reshape(B * G, H, W, 2)
            smp = F.grid_sample(xg, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            out = out + smp.reshape(B, C, H, W) * self.weight[:, t].view(1, C, 1, 1)
        return out + self.bias.view(1, C, 1, 1)


class InceptionDeformMixer(nn.Module):
    """Inception-style multi-scale token mixer: [deformable-dw3x3 | dilated-dw3x3 | strip 1x7+7x1 | avg-pool]."""
    def __init__(self, dim, deform=True):
        super().__init__()
        q = dim // 4
        self.sp = [q, q, q, dim - 3 * q]
        self.a = DeformDWConv(q, 3, deform=deform)
        self.b = nn.Conv2d(q, q, 3, padding=2, dilation=2, groups=q)
        self.c = nn.Sequential(nn.Conv2d(q, q, (1, 7), padding=(0, 3), groups=q),
                               nn.Conv2d(q, q, (7, 1), padding=(3, 0), groups=q))
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.pw = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        xa, xb, xc, xd = torch.split(x, self.sp, 1)
        y = torch.cat([self.a(xa).to(x.dtype), self.b(xb), self.c(xc), self.pool(xd)], 1)
        return self.pw(self.act(self.bn(y)))


def window_partition(x, ws):                                # B,C,H,W -> (B*nW, ws*ws, C)
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // ws, ws, W // ws, ws).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(-1, ws * ws, C)


def window_reverse(x, ws, B, C, H, W):                      # (B*nW, ws*ws, C) -> B,C,H,W
    x = x.view(B, H // ws, W // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, C, H, W)


class WindowCrossAttention(nn.Module):
    """Lightweight windowed multi-head CROSS-attention: queries come from the local (depthwise-conv) context
    stream, keys/values from the raw token stream -> local-conv features attend to the tokens in their window."""
    def __init__(self, dim, heads, ws=5):
        super().__init__()
        self.h, self.ws = heads, ws
        self.local = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, 2 * dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(self.local(x))
        k, v = self.kv(x).chunk(2, 1)
        ws = min(self.ws, max(H, W))
        ph, pw = (ws - H % ws) % ws, (ws - W % ws) % ws
        if ph or pw:
            q, k, v = [F.pad(t, (0, pw, 0, ph)) for t in (q, k, v)]
        Hp, Wp = H + ph, W + pw

        def heads(t):
            t = window_partition(t, ws)
            return t.view(t.shape[0], ws * ws, self.h, C // self.h).transpose(1, 2)

        o = F.scaled_dot_product_attention(heads(q), heads(k), heads(v))
        o = o.transpose(1, 2).reshape(-1, ws * ws, C)
        o = window_reverse(o, ws, B, C, Hp, Wp)[:, :, :H, :W]
        return self.proj(o)


class DITBlock(nn.Module):
    """Deformable Inception-Transformer block. kind='conv' -> deformable-inception mixer, kind='attn' -> windowed cross-attn."""
    def __init__(self, dim, kind, heads, ws, mlp_ratio, dp, deform):
        super().__init__()
        self.n1 = nn.BatchNorm2d(dim)
        self.mix = InceptionDeformMixer(dim, deform) if kind == "conv" else WindowCrossAttention(dim, heads, ws)
        self.n2 = nn.BatchNorm2d(dim)
        self.mlp = nn.Sequential(nn.Conv2d(dim, dim * mlp_ratio, 1), nn.GELU(), nn.Conv2d(dim * mlp_ratio, dim, 1))
        self.gamma1 = nn.Parameter(0.1 * torch.ones(1, dim, 1, 1))
        self.gamma2 = nn.Parameter(0.1 * torch.ones(1, dim, 1, 1))
        self.dp = DropPath(dp)

    def forward(self, x):
        y = self.mix(self.n1(x))
        x = x + self.dp(self.gamma1.to(y.dtype) * y)
        y = self.mlp(self.n2(x))
        x = x + self.dp(self.gamma2.to(y.dtype) * y)
        return x


class ChannelSpatialGate(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(ch, ch // r), nn.GELU(), nn.Linear(ch // r, ch))
        self.sp = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, y):
        gc_ = torch.sigmoid(self.fc(y.mean((2, 3)))).unsqueeze(-1).unsqueeze(-1)
        gs = torch.sigmoid(self.sp(torch.cat([y.mean(1, keepdim=True), y.amax(1, keepdim=True)], 1)))
        return gc_ * gs


class TCFA(nn.Module):
    """Topological Cross-Layer Feature Aggregation: shallow->deep gated aggregation path.
    a_1 = f_1 ;  a_i = f_i + Gate(f_i) * Down_i(a_{i-1}).  Every shallow edge/texture stage reaches the classifier
    through learned channel-spatial gates instead of plain concatenation / residual addition."""
    def __init__(self, dims):
        super().__init__()
        self.down = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dims[i - 1], dims[i - 1], 3, 2, 1, groups=dims[i - 1], bias=False),
            nn.Conv2d(dims[i - 1], dims[i], 1, bias=False), nn.BatchNorm2d(dims[i])) for i in range(1, len(dims))])
        self.gates = nn.ModuleList([ChannelSpatialGate(dims[i]) for i in range(1, len(dims))])

    def forward(self, feats):
        a = feats[0]
        for i in range(1, len(feats)):
            a = feats[i] + self.gates[i - 1](feats[i]) * self.down[i - 1](a)
        return a


class CenterLoss(nn.Module):
    """Center loss (Wen et al. 2016): 0.5*||f - c_y||^2, centres updated by moving average (not by gradient)."""
    def __init__(self, num_classes, dim, alpha=0.5):
        super().__init__()
        self.alpha, self.K = alpha, num_classes
        self.register_buffer("centers", torch.zeros(num_classes, dim))

    def forward(self, emb, y):
        emb = emb.float()
        c = self.centers[y]
        loss = 0.5 * ((emb - c) ** 2).sum(1).mean()
        with torch.no_grad():
            delta = torch.zeros_like(self.centers).index_add_(0, y, c - emb.detach())
            cnt = torch.bincount(y, minlength=self.K).float().unsqueeze(1)
            self.centers -= self.alpha * delta / (1.0 + cnt)
        return loss


class CompactHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=1024, embed=256, act="adaptive", drop=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = AdaptiveAct(hidden) if act == "adaptive" else nn.GELU()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, embed)
        self.bn2 = nn.BatchNorm1d(embed)
        self.cls = nn.Linear(embed, num_classes)

    def forward(self, f):
        e = self.bn2(self.fc2(self.drop(self.act(self.bn1(self.fc1(f))))))
        return self.cls(e), e


class SSPathNet(nn.Module):
    def __init__(self, num_classes=38, dims=(64, 128, 256, 384), depths=(2, 3, 8, 2), head_dim=32, ws=5,
                 mlp_ratio=3, drop_path=0.1, stem="full", deform=True, use_attn=True, agg="tcfa",
                 head_act="adaptive", cam_stage=2):
        super().__init__()
        self.agg, self.cam_stage = agg, cam_stage
        self.stem = ChromaFreqStem(dims[0], stem)
        dpr = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.stages, self.downs, k = nn.ModuleList(), nn.ModuleList(), 0
        for i, (d, n) in enumerate(zip(dims, depths)):
            if i > 0:
                self.downs.append(nn.Sequential(nn.Conv2d(dims[i - 1], d, 3, 2, 1, bias=False), nn.BatchNorm2d(d)))
            blocks = []
            for j in range(n):
                kind = "attn" if (use_attn and j % 2 == 1) else "conv"
                blocks.append(DITBlock(d, kind, max(1, d // head_dim), ws, mlp_ratio, dpr[k], deform)); k += 1
            self.stages.append(nn.Sequential(*blocks))
        if agg == "tcfa":
            self.tcfa = TCFA(dims)
        feat_dim = sum(dims) if agg == "concat" else dims[-1]
        self.head = CompactHead(feat_dim, num_classes, act=head_act)
        self.embed_dim = 256
        self._cam_feat = None

    def forward(self, x, keep_cam=False):
        x = self.stem(x)
        feats = []
        for i, st in enumerate(self.stages):
            if i > 0:
                x = self.downs[i - 1](x)
            x = st(x)
            feats.append(x)
        if keep_cam:
            feats[self.cam_stage].retain_grad()
            self._cam_feat = feats[self.cam_stage]
        if self.agg == "tcfa":
            f = self.tcfa(feats).mean((2, 3))
        elif self.agg == "concat":
            f = torch.cat([t.mean((2, 3)) for t in feats], 1)
        else:
            f = feats[-1].mean((2, 3))
        return self.head(f)                                  # (logits, embedding)


# ---------------------------------------------------------------------
# Baselines (torchvision, trained FROM SCRATCH with the identical pipeline)
# ---------------------------------------------------------------------
class Baseline(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return self.net((x - self.mean) / self.std), None


def build_baseline(name, num_classes):
    import torchvision.models as tvm
    fn = dict(mobilenet_v2=tvm.mobilenet_v2, efficientnet_b0=tvm.efficientnet_b0, resnet18=tvm.resnet18,
              densenet121=tvm.densenet121, mobilenet_v3_large=tvm.mobilenet_v3_large)[name]
    return Baseline(fn(weights=None, num_classes=num_classes))


# =====================================================================
# ENGINE: training / evaluation / robustness / Grad-CAM
# =====================================================================
try:
    from torch.amp import GradScaler as _GS
    def make_scaler():
        return _GS("cuda", enabled=DEV.type == "cuda")
except Exception:  # older torch
    def make_scaler():
        return torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")


def autocast():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEV.type == "cuda")


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def make_optimizer(model, lr):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if (p.ndim <= 1 or "gamma" in n) else decay).append(p)
    return torch.optim.AdamW([dict(params=decay, weight_decay=WEIGHT_DECAY),
                              dict(params=no_decay, weight_decay=0.0)], lr=lr, betas=(0.9, 0.999))


@torch.no_grad()
def predict(model, X_uint8=None, idx=None, transform=None, bs=256):
    """Returns logits (N,K) on CPU.  Either idx (into DATA_X) or an explicit uint8 NHWC tensor X_uint8."""
    model.eval()
    src = DATA_X if X_uint8 is None else X_uint8
    n = len(idx) if idx is not None else len(src)
    idx_t = torch.as_tensor(idx, dtype=torch.long) if idx is not None else None
    outs = []
    for i in range(0, n, bs):
        xb = src[idx_t[i:i + bs]] if idx_t is not None else src[i:i + bs]
        x = to_float(xb)
        if transform is not None:
            x = transform(x)
        with autocast():
            outs.append(model(x)[0].float().cpu())
    return torch.cat(outs)


def compute_metrics(y_true, y_pred, K):
    p, r, f, _ = skm.precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wf = skm.precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2]
    return dict(accuracy=float(skm.accuracy_score(y_true, y_pred)), macro_precision=float(p), macro_recall=float(r),
                macro_f1=float(f), weighted_f1=float(wf), cohen_kappa=float(skm.cohen_kappa_score(y_true, y_pred)),
                mcc=float(skm.matthews_corrcoef(y_true, y_pred)))


# ---- robustness (synthetic domain shift on the test set) ----
def _gauss_blur(x, k=7, sigma=2.0):
    ax = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    C = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


def _hue_shift(x, deg=40.0):
    Y = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=x.device)
    a = math.radians(deg)
    R = torch.tensor([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]], device=x.device)
    T = torch.linalg.inv(Y) @ R @ Y
    return torch.einsum("ij,bjhw->bihw", T, x).clamp(0, 1)


def _occlude(x, n=3):
    B, _, H, W = x.shape; s = H // 5
    g = torch.Generator(device=x.device).manual_seed(123)
    x = x.clone()
    for _ in range(n):
        cy = torch.randint(0, H - s, (B,), device=x.device, generator=g)
        cx = torch.randint(0, W - s, (B,), device=x.device, generator=g)
        for b in range(B):
            x[b, :, cy[b]:cy[b] + s, cx[b]:cx[b] + s] = 0.5
    return x


def _noise(x, sigma=0.08):
    g = torch.Generator(device=x.device).manual_seed(321)
    return (x + sigma * torch.randn(x.shape, device=x.device, generator=g)).clamp(0, 1)


PERTURBATIONS = {
    "clean": lambda x: x,
    "dark": lambda x: x * 0.55,
    "bright": lambda x: (x * 1.6).clamp(0, 1),
    "low_contrast": lambda x: (x - x.mean((1, 2, 3), keepdim=True)) * 0.4 + x.mean((1, 2, 3), keepdim=True),
    "gauss_noise": _noise,
    "blur": _gauss_blur,
    "hue_shift": _hue_shift,
    "occlusion": _occlude,
}


def robustness_eval(model, test_idx):
    y = DATA_Y[torch.as_tensor(test_idx)].numpy()
    res = {}
    for k, fn in PERTURBATIONS.items():
        pr = predict(model, idx=test_idx, transform=fn).argmax(1).numpy()
        res[k] = float(skm.accuracy_score(y, pr))
    return res


# ---- efficiency ----
@torch.no_grad()
def efficiency(model):
    out = dict(params_M=count_params(model) / 1e6)
    model.eval()
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            model(torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=DEV))
        out["gflops_fwd"] = fc.get_total_flops() / 1e9          # note: grid_sample not counted
    except Exception:
        out["gflops_fwd"] = None
    if DEV.type == "cuda":
        x = torch.rand(128, 3, IMG_SIZE, IMG_SIZE, device=DEV)
        for _ in range(5):
            with autocast():
                model(x)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(15):
            with autocast():
                model(x)
        torch.cuda.synchronize()
        out["throughput_img_s"] = 128 * 15 / (time.time() - t)
        x1 = torch.rand(1, 3, IMG_SIZE, IMG_SIZE, device=DEV)
        for _ in range(10):
            with autocast():
                model(x1)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(50):
            with autocast():
                model(x1)
        torch.cuda.synchronize()
        out["latency_ms_bs1"] = (time.time() - t) / 50 * 1000
    return out


# ---- Grad-CAM (SSPathNet, stage-3 feature map, 10x10 at 160px) ----
def gradcam_figure(model, test_idx, y_pred, y_true, path, n_correct=8, n_wrong=4):
    if not hasattr(model, "cam_stage"):
        return
    rng = np.random.RandomState(SEED)
    correct = np.where(y_pred == y_true)[0]; wrong = np.where(y_pred != y_true)[0]
    sel = list(rng.choice(correct, min(n_correct, len(correct)), replace=False))
    sel += list(rng.choice(wrong, min(n_wrong, len(wrong)), replace=False)) if len(wrong) else []
    if len(sel) < n_correct + n_wrong and len(correct) > len(sel):
        rest = [i for i in rng.permutation(correct) if i not in sel]
        sel += rest[: n_correct + n_wrong - len(sel)]
    ids = np.asarray(test_idx)[sel]
    model.eval()
    x = to_float(DATA_X[torch.as_tensor(ids)])
    logits = model(x, keep_cam=True)[0]
    cls = logits.argmax(1)
    model.zero_grad(set_to_none=True)
    logits.gather(1, cls[:, None]).sum().backward()
    feat = model._cam_feat
    w = feat.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feat).sum(1, keepdim=True)).detach()
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    mn = cam.amin((2, 3), keepdim=True); mx = cam.amax((2, 3), keepdim=True)
    cam = ((cam - mn) / (mx - mn + 1e-8)).squeeze(1).cpu().numpy()
    imgs = DATA_X[torch.as_tensor(ids)].numpy()
    cols = 6; rows = int(math.ceil(len(ids) * 2 / cols / 2)) * 2
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3))
    axs = np.atleast_2d(axs)
    for a in axs.ravel():
        a.axis("off")
    for j in range(len(ids)):
        r0, c0 = (j // cols) * 2, j % cols
        t, p = int(y_true[sel[j]]), int(cls[j])
        axs[r0, c0].imshow(imgs[j])
        axs[r0, c0].set_title(f"true {t}" + ("" if t == p else f" | pred {p}"), fontsize=8,
                              color="k" if t == p else "r")
        axs[r0 + 1, c0].imshow(imgs[j]); axs[r0 + 1, c0].imshow(cam[j], cmap="jet", alpha=0.45)
    plt.suptitle("Grad-CAM (stage-3 features). Row pairs: image / heat-map overlay. Class ids in classes.json", fontsize=9)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


# ---- plots ----
def save_curves(hist, path):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(ep, [h["train_loss"] for h in hist]); ax[0].set_title("train loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(ep, [h["train_acc"] for h in hist], label="train"); ax[1].plot(ep, [h["val_acc"] for h in hist], label="val")
    ax[1].set_title("accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


def save_confusion(cm, path):
    fig = plt.figure(figsize=(8, 7))
    plt.imshow(np.log1p(cm), cmap="Blues"); plt.colorbar(label="log(1+count)")
    plt.xlabel("predicted class id"); plt.ylabel("true class id"); plt.title("Confusion matrix (test)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close(fig)


# ---- results store ----
RESULTS_FILE = RUN_ROOT / "results.json"
RESULTS = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}


def save_results():
    RESULTS_FILE.write_text(json.dumps(RESULTS, indent=1))
    rows = [(k, v) for k, v in RESULTS.items() if "test" in v]
    if rows:
        keys = ["run", "status", "params_M", "gflops_fwd", "throughput_img_s", "epochs_run", "train_min",
                "val_acc", "test_acc", "macro_f1", "macro_precision", "macro_recall", "kappa", "mcc"]
        with open(RUN_ROOT / "summary.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(keys)
            for k, v in rows:
                t = v["test"]
                w.writerow([k, v.get("status"), v.get("params_M"), v.get("gflops_fwd"), v.get("throughput_img_s"),
                            v.get("epochs_run"), v.get("train_min"), v.get("val_acc"), t["accuracy"], t["macro_f1"],
                            t["macro_precision"], t["macro_recall"], t["cohen_kappa"], t["mcc"]])


# ---- time estimate for one run ----
def estimate_epoch_time(build_fn, n_train):
    m = build_fn().to(DEV).train()
    opt = make_optimizer(m, 1e-4); scaler = make_scaler()
    x = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEV)
    y = torch.randint(0, len(CLASSES), (BATCH_SIZE,), device=DEV)
    ts = []
    for i in range(9):
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        with autocast():
            lg = m(gpu_augment(x))[0]
        loss = F.cross_entropy(lg.float(), y)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.time() - t)
    del m, opt, scaler; gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    step = float(np.median(ts[4:]))
    return step * (n_train // BATCH_SIZE) * 1.15


# =====================================================================
# One complete experiment
# =====================================================================
def run_experiment(name, build_fn, splits, epochs, use_center=True, gradcam=False, robust=False,
                   save_weights=False, must_run=False, post_hook=None):
    out = RUN_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    if name in RESULTS and "test" in RESULTS[name]:
        log(f"[{name}] already finished in a previous session -> skipping"); return RESULTS[name]

    n_train = len(splits["train"])
    est_epoch = estimate_epoch_time(build_fn, n_train)
    projected = est_epoch * epochs
    log(f"[{name}] est. {est_epoch:.0f}s/epoch -> projected {projected / 60:.0f} min for {epochs} epochs "
        f"(time left {time_left_s() / 60:.0f} min)")
    if not must_run and projected > time_left_s() - 240:
        log(f"[{name}] SKIPPED: not enough time left in this session's budget")
        RESULTS[name] = dict(status="skipped_time", projected_min=projected / 60); save_results(); return RESULTS[name]

    set_seed(SEED)
    model = build_fn().to(DEV)
    n_par = count_params(model)
    log(f"[{name}] parameters: {n_par / 1e6:.3f} M")
    opt = make_optimizer(model, LR); scaler = make_scaler()
    center = CenterLoss(len(CLASSES), 256).to(DEV) if (use_center and hasattr(model, "embed_dim")) else None
    loader = BatchLoader(splits["train"], BATCH_SIZE, True, True, seed=SEED)
    nb = len(loader); total = epochs * nb; warm = WARMUP_EPOCHS * nb
    yv = DATA_Y[torch.as_tensor(splits["val"])].numpy()
    hist, best_acc, best_state, best_ep, step, status = [], -1.0, None, -1, 0, "ok"
    t_run, last_ep_time = time.time(), est_epoch

    for ep in range(1, epochs + 1):
        if not must_run and time_left_s() < 1.5 * last_ep_time + 180:
            status = "truncated_time"; log(f"[{name}] time budget reached -> stopping early at epoch {ep - 1}"); break
        model.train(); t_ep = time.time()
        ls = torch.zeros((), device=DEV); cr = torch.zeros((), device=DEV); seen = 0
        for xb, yb in loader:
            x = gpu_augment(to_float(xb)); y = yb.to(DEV, non_blocking=True)
            lr = LR * (step + 1) / warm if step < warm else \
                1e-6 + 0.5 * (LR - 1e-6) * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            with autocast():
                logits, emb = model(x)
            loss = F.cross_entropy(logits.float(), y, label_smoothing=LABEL_SMOOTH)
            if center is not None and emb is not None:
                loss = loss + CENTER_WEIGHT * center(emb, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); step += 1
            ls += loss.detach() * len(y); cr += (logits.argmax(1) == y).sum(); seen += len(y)
        tl, ta = ls.item() / seen, cr.item() / seen
        if not math.isfinite(tl):
            status = "diverged"; log(f"[{name}] loss became NaN/Inf -> aborting run"); break
        vp = predict(model, idx=splits["val"]).argmax(1).numpy()
        va = float((vp == yv).mean())
        if va >= best_acc:
            best_acc, best_ep = va, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        last_ep_time = time.time() - t_ep
        hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_acc=va, lr=lr, sec=last_ep_time))
        log(f"[{name}] ep {ep:3d}/{epochs}  loss {tl:.4f}  train {ta:.4f}  val {va:.4f}  "
            f"best {best_acc:.4f}@{best_ep}  {last_ep_time:.0f}s")

    train_min = (time.time() - t_run) / 60
    if best_state is None:
        RESULTS[name] = dict(status=status, epochs_run=len(hist)); save_results(); return RESULTS[name]
    model.load_state_dict(best_state)
    yt = DATA_Y[torch.as_tensor(splits["test"])].numpy()
    pred = predict(model, idx=splits["test"]).argmax(1).numpy()
    met = compute_metrics(yt, pred, len(CLASSES))
    rec = dict(status=status, epochs_planned=epochs, epochs_run=len(hist), best_epoch=best_ep, train_min=train_min,
               val_acc=best_acc, test=met, n_train=n_train, n_val=len(splits["val"]), n_test=len(splits["test"]))
    rec.update(efficiency(model))
    if robust:
        rec["robustness"] = robustness_eval(model, splits["test"])
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys())); w.writeheader(); w.writerows(hist)
    cm = skm.confusion_matrix(yt, pred, labels=list(range(len(CLASSES))))
    np.save(out / "confusion_matrix.npy", cm)
    rep = skm.classification_report(yt, pred, labels=list(range(len(CLASSES))), target_names=CLASSES,
                                    output_dict=True, zero_division=0)
    (out / "classification_report.json").write_text(json.dumps(rep, indent=1))
    save_curves(hist, out / "curves.png"); save_confusion(cm, out / "confusion_matrix.png")
    if gradcam:
        try:
            gradcam_figure(model, splits["test"], pred, yt, out / "gradcam.png")
        except Exception as e:
            log(f"[{name}] Grad-CAM failed: {e}")
    if save_weights:
        torch.save(best_state, out / "best.pt")
    if post_hook is not None:
        try:
            post_hook(model, out, rec)
        except Exception as e:
            log(f"[{name}] post-hook failed: {e}")
    RESULTS[name] = rec; save_results()
    log(f"[{name}] DONE  test acc {met['accuracy']:.4f}  macro-F1 {met['macro_f1']:.4f}  "
        f"params {rec['params_M']:.2f}M  ({train_min:.0f} min, status {status})")
    del model, opt, best_state; gc.collect()
    if DEV.type == "cuda":
        torch.cuda.empty_cache()
    return rec


def print_table(names, title):
    log("=" * 96); log(title)
    log(f"{'run':<26}{'status':<16}{'paramsM':>8}{'test acc':>10}{'macroF1':>9}{'kappa':>8}{'min':>7}")
    for n in names:
        v = RESULTS.get(n)
        if not v or "test" not in v:
            log(f"{n:<26}{(v or {}).get('status', 'not run'):<16}"); continue
        t = v["test"]
        log(f"{n:<26}{v['status']:<16}{v['params_M']:>8.2f}{t['accuracy'] * 100:>9.2f}%{t['macro_f1'] * 100:>8.2f}%"
            f"{t['cohen_kappa']:>8.4f}{v['train_min']:>7.0f}")
    log("=" * 96)


def setup_env():
    log(f"torch {torch.__version__} | device {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == 'cuda' else ""))
    if DEV.type != "cuda":
        log("WARNING: no GPU detected. In Kaggle: Settings -> Accelerator -> GPU T4. (Running on CPU is far too slow.)")
    torch.backends.cudnn.benchmark = True
    set_seed(SEED)


def auto_batch_size(build_fn):
    """Probe one training step; halve BATCH_SIZE if the GPU runs out of memory."""
    global BATCH_SIZE
    if DEV.type != "cuda":
        return
    while True:
        try:
            torch.cuda.reset_peak_memory_stats()
            m = build_fn().to(DEV).train(); opt = make_optimizer(m, 1e-4); scaler = make_scaler()
            x = torch.rand(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=DEV)
            y = torch.randint(0, len(CLASSES), (BATCH_SIZE,), device=DEV)
            with autocast():
                lg = m(gpu_augment(x))[0]
            loss = F.cross_entropy(lg.float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            log(f"Batch-size probe: {BATCH_SIZE} fits (peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB "
                f"of {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
            del m, opt, scaler, x, y, lg, loss; gc.collect(); torch.cuda.empty_cache()
            return
        except torch.cuda.OutOfMemoryError:
            m = opt = scaler = x = y = lg = loss = None; gc.collect(); torch.cuda.empty_cache()
            BATCH_SIZE //= 2
            log(f"CUDA out of memory -> reducing BATCH_SIZE to {BATCH_SIZE}")
            if BATCH_SIZE < 16:
                raise


def safe_run(name, *args, **kwargs):
    """run_experiment that never takes down the remaining runs."""
    try:
        return run_experiment(name, *args, **kwargs)
    except Exception as e:
        import traceback
        log(f"[{name}] FAILED with {type(e).__name__}: {e}")
        log(traceback.format_exc())
        RESULTS[name] = dict(status=f"error: {type(e).__name__}: {str(e)[:200]}"); save_results()
        gc.collect()
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
        return RESULTS[name]


# =====================================================================
# Optional cross-dataset test (PlantDoc / any folder-per-class dataset)
# =====================================================================
import re

_STOP = {"leaf", "leaves", "plant", "disease", "diseases", "the", "of", "and"}
_SYN = {"maize": "corn", "capsicum": "pepper"}


def norm_tokens(s):
    toks = {_SYN.get(t, t) for t in re.findall(r"[a-z0-9]+", s.lower())} - _STOP
    return toks


def map_classes(foreign_names, pv_names, thr=0.4):
    """Best-effort token-overlap mapping  foreign folder name -> PlantVillage class name.
    Manual overrides in CROSS_CLASS_MAP win. Ambiguous / weak matches are dropped."""
    pv_tok = {n: norm_tokens(n) for n in pv_names}
    out = {}
    for fn in foreign_names:
        if fn in CROSS_CLASS_MAP and CROSS_CLASS_MAP[fn] in pv_names:
            out[fn] = CROSS_CLASS_MAP[fn]; continue
        ft = norm_tokens(fn)
        if not ft:
            continue
        if len(ft) == 1:                 # e.g. "Apple leaf" -> crop only -> assume the healthy class
            ft = ft | {"healthy"}
        scores = []
        for n, t in pv_tok.items():
            inter = len(ft & t)
            if inter == 0:
                continue
            scores.append((inter / len(ft | t), n))
        if not scores:
            continue
        scores.sort(reverse=True)
        if scores[0][0] >= thr and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            out[fn] = scores[0][1]
    return out


def cross_dataset_eval(model, out_dir):
    if not CROSS_DATASET_PATH:
        return None
    cands = find_class_dirs(CROSS_DATASET_PATH, 3, max_depth=4)
    if not cands:
        log(f"[cross] no class folders found under {CROSS_DATASET_PATH}"); return None
    Xs, ys, mapping_all = [], [], {}
    for root, n in cands:
        names = sorted([d for d in os.listdir(root) if dir_has_images(root / d)])
        mp = map_classes(names, CLASSES)
        log(f"[cross] {root}: {len(names)} folders, {len(mp)} mapped to PlantVillage classes")
        for k, v in mp.items():
            log(f"          {k!r:45s} -> {v}")
        if not mp:
            continue
        mapping_all[str(root)] = mp
        keep = list(mp.keys())
        X, y_loc, _ = load_folder_images(root, keep, IMG_SIZE, desc="cross-dataset images")
        Xs.append(X); ys.append(np.asarray([CLASSES.index(mp[keep[i]]) for i in y_loc]))
    if not Xs:
        log("[cross] no overlapping classes -> skipped (set CROSS_CLASS_MAP manually)"); return None
    X = torch.from_numpy(np.concatenate(Xs)); y = np.concatenate(ys)
    logits = predict(model, X_uint8=X)
    allowed = sorted(set(y.tolist()))
    p_restr = np.asarray(allowed)[logits[:, allowed].argmax(1).numpy()]
    p_full = logits.argmax(1).numpy()
    res = dict(n_images=int(len(y)), n_classes_matched=len(allowed),
               acc_restricted_to_matched_classes=float((p_restr == y).mean()),
               acc_all_38_classes=float((p_full == y).mean()),
               macro_f1_restricted=float(skm.f1_score(y, p_restr, average="macro", zero_division=0)),
               mapping=mapping_all)
    (Path(out_dir) / "cross_dataset.json").write_text(json.dumps(res, indent=1))
    log(f"[cross] n={res['n_images']}  acc(restricted)={res['acc_restricted_to_matched_classes']:.4f}  "
        f"acc(38-way)={res['acc_all_38_classes']:.4f}  (zero-shot: model never saw this dataset)")
    return res


# =====================================================================
# PART 2 MAIN
# =====================================================================
ABLATIONS = [   # (run name, SSPathNet kwargs, use_center_loss) -- listed in priority order
    ("A1_stem_rgb_only",   dict(stem="rgb"),          True),
    ("A4_no_deformable",   dict(deform=False),        True),
    ("A5_no_attention",    dict(use_attn=False),      True),
    ("A6_agg_none",        dict(agg="none"),          True),
    ("A8_no_center_loss",  dict(),                    False),
    ("A9_gelu_head",       dict(head_act="gelu"),     True),
    ("A2_stem_rgb_hsv",    dict(stem="rgb+hsv"),      True),
    ("A3_stem_rgb_dct",    dict(stem="rgb+dct"),      True),
    ("A7_agg_concat",      dict(agg="concat"),        True),
]


def main():
    setup_env()
    load_dataset()
    K = len(CLASSES)
    auto_batch_size(lambda: SSPathNet(K))
    splits = fixed_split()
    log(f"Split sizes -> train {len(splits['train'])}  val {len(splits['val'])}  test {len(splits['test'])}")
    abl = ABLATIONS[:2] if SMOKE_TEST else ABLATIONS

    hook = (lambda m, out, rec: rec.__setitem__("cross_dataset", cross_dataset_eval(m, out))) if CROSS_DATASET_PATH else None
    ref = "A0_full_SSPathNet"
    safe_run(ref, lambda: SSPathNet(K), splits, EPOCHS, use_center=True, robust=True, save_weights=True,
                   must_run=True, post_hook=hook)
    for name, kw, uc in abl:
        safe_run(name, (lambda kw=kw: SSPathNet(K, **kw)), splits, EPOCHS, use_center=uc, robust=True)

    names = [ref] + [n for n, _, _ in abl]
    print_table(names, "PART 2 - ablation study (fixed split, test set)")
    if ref in RESULTS and "test" in RESULTS[ref]:
        a0 = RESULTS[ref]["test"]
        log("Delta = ablated - full (percentage points). Negative = removing the component HURT, i.e. the component helps:")
        log(f"{'run':<26}{'d_acc':>9}{'d_macroF1':>11}{'d_params(M)':>13}")
        rows = []
        for n in names[1:]:
            v = RESULTS.get(n)
            if v and "test" in v:
                da = (v["test"]["accuracy"] - a0["accuracy"]) * 100
                df = (v["test"]["macro_f1"] - a0["macro_f1"]) * 100
                dp = v["params_M"] - RESULTS[ref]["params_M"]
                rows.append((n, da, df, dp)); log(f"{n:<26}{da:>+9.2f}{df:>+11.2f}{dp:>+13.2f}")
        RESULTS["ablation_deltas_vs_A0"] = {r[0]: dict(d_acc_pp=r[1], d_macro_f1_pp=r[2], d_params_M=r[3]) for r in rows}
        save_results()
    if ref in RESULTS and "robustness" in RESULTS[ref]:
        log("Robustness (test accuracy under synthetic domain shift):")
        cols = list(PERTURBATIONS.keys())
        log(f"{'model':<26}" + "".join(f"{c:>13}" for c in cols))
        for n in names:
            if n in RESULTS and "robustness" in RESULTS[n]:
                r = RESULTS[n]["robustness"]
                log(f"{n:<26}" + "".join(f"{r[c] * 100:>12.2f}%" for c in cols))
    log(f"PART 2 finished. Total time {elapsed_h():.2f} h.  Results in {RUN_ROOT}")


if __name__ == "__main__":
    main()
