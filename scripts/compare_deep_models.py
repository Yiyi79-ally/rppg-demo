#!/usr/bin/env python
r"""Compare the toolbox's pretrained deep rPPG models with POS on the same videos.

Save in D:\DIP\rPPG-Toolbox-main and run with the project's environment.

One-time setup (PyTorch for CPU, ~200 MB; Python 3.8 -> torch 2.4.1 is the last version):
  .\.venv\Scripts\python.exe -m pip install torch==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
  (if that mirror is slow: --index-url https://download.pytorch.org/whl/cpu)

Part 1 - UBFC-rPPG subjects 31-40 (reference = finger-PPG ground truth):
  .\.venv\Scripts\python.exe .\compare_deep_models.py
  .\.venv\Scripts\python.exe .\compare_deep_models.py --subjects 31 32 --models physnet factorizephys

Part 2 - your own recording from the phone page (reference = your taps):
  .\.venv\Scripts\python.exe .\compare_deep_models.py --video latest
        (newest heart_rate_*.webm + .csv pair in your Downloads folder)
  .\.venv\Scripts\python.exe .\compare_deep_models.py --video D:\path\heart_rate_x.webm --taps D:\path\heart_rate_x.csv

Models (all trained on PURE, so UBFC and your video are unseen data = a fair cross-dataset test):
  physnet, tscan, deepphys, efficientphys, physformer, rhythmformer, factorizephys, plus POS
  (same face crop) as the baseline. PhysMamba is skipped: it needs mamba-ssm, which is CUDA-only.

Preprocessing follows the toolbox's inference configs (configs/infer_configs/PURE_UBFC-rPPG_*):
first-frame Haar face box enlarged 1.5x, resized with INTER_AREA to 72x72 (128x128 for PhysFormer
and RhythmFormer), DiffNormalized / Standardized / Raw per model, the model's chunk length, and the
toolbox's HR estimate (detrend 100, 0.6-3.3 Hz band-pass, FFT peak). Reported for the whole video
(the toolbox's standard number) and for 10 s windows (closer to a live reading).
Face crops are cached in ..\results\deep_cache so a rerun only repeats the models.
Outputs: ..\results\deep_compare_<time>\summary.txt, per_video.csv, windows.csv
"""

import argparse
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import time
import types

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import cv2
import scipy.signal
import scipy.sparse

cv2.ocl.setUseOpenCL(False)
FS = 30
WINDOW_S = 10
CHECKED_BOXES = {"subject31": [309, 96, 161, 161]}   # inspected earlier (auto detection was off)

MODELS = {
    # name: weights, input size, chunk length, data types, label diff flag
    "physnet": ("PURE_PhysNet_DiffNormalized.pth", 72, 128, ["DiffNormalized"], True),
    "tscan": ("PURE_TSCAN.pth", 72, 180, ["DiffNormalized", "Standardized"], True),
    "deepphys": ("PURE_DeepPhys.pth", 72, 180, ["DiffNormalized", "Standardized"], True),
    "efficientphys": ("PURE_EfficientPhys.pth", 72, 180, ["Standardized"], True),
    "physformer": ("PURE_PhysFormer_DiffNormalized.pth", 128, 160, ["DiffNormalized"], True),
    "rhythmformer": ("PURE_RhythmFormer.pth", 128, 160, ["Standardized"], False),
    "factorizephys": ("PURE_FactorizePhys_FSAM_Res.pth", 72, 160, ["Raw"], False),
}


# ---------------------------------------------------------------- optional-package stubs
def install_stubs():
    """neurokit2 is only imported (not used) by FactorizePhys; timm only supplies two small
    helpers to RhythmFormer. Stub them when missing so only PyTorch has to be installed."""
    try:
        import neurokit2  # noqa: F401
    except ImportError:
        sys.modules["neurokit2"] = types.ModuleType("neurokit2")
    try:
        from timm.models.layers import trunc_normal_, DropPath  # noqa: F401
    except ImportError:
        import torch
        timm = types.ModuleType("timm"); models = types.ModuleType("timm.models"); layers = types.ModuleType("timm.models.layers")

        class DropPath(torch.nn.Module):   # identity at inference time
            def __init__(self, p=0.0):
                super().__init__()

            def forward(self, x):
                return x
        layers.trunc_normal_ = lambda t, mean=0.0, std=1.0, a=-2.0, b=2.0: torch.nn.init.trunc_normal_(t, mean, std, a, b)
        layers.DropPath = DropPath
        timm.models = models; models.layers = layers
        sys.modules.update({"timm": timm, "timm.models": models, "timm.models.layers": layers})


# ---------------------------------------------------------------- toolbox-equivalent preprocessing
def haar_box(frame_rgb):
    det = cv2.CascadeClassifier(str(HERE / "dataset" / "haarcascade_frontalface_default.xml"))
    zones = det.detectMultiScale(frame_rgb.astype(np.uint8))
    if len(zones) < 1:
        print("    no face found on the first frame; using the whole frame")
        return [0, 0, frame_rgb.shape[0], frame_rgb.shape[1]]
    return list(zones[int(np.argmax(zones[:, 2]))])


def enlarge(box, coef=1.5):
    x, y, w, h = [float(v) for v in box]
    return [int(max(0, x - (coef - 1) / 2 * w)), int(max(0, y - (coef - 1) / 2 * h)), int(coef * w), int(coef * h)]


def read_crops(video, sizes, fixed_box=None):
    """Read a video once; return {size: uint8 [T,s,s,3] RGB crops}, frame times (s), box."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("cannot open " + str(video))
    crops = {s: [] for s in sizes}
    times, box = [], None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if box is None:
            box = enlarge(fixed_box if fixed_box else haar_box(rgb))
        x, y, w, h = box
        face = rgb[max(y, 0):min(y + h, rgb.shape[0]), max(x, 0):min(x + w, rgb.shape[1])]
        for s in sizes:
            crops[s].append(cv2.resize(face, (s, s), interpolation=cv2.INTER_AREA))
        times.append(t)
    cap.release()
    return {s: np.asarray(v, dtype=np.uint8) for s, v in crops.items()}, np.asarray(times), box


def resample_frames(frames, times, fs=FS):
    """Variable-frame-rate recordings (phone/webcam) -> even fs grid by linear interpolation."""
    t = times - times[0]
    grid = np.arange(0, t[-1] + 1e-9, 1.0 / fs)
    j = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
    w = ((grid - t[j]) / np.maximum(t[j + 1] - t[j], 1e-9)).clip(0, 1)[:, None, None, None]
    return (frames[j] * (1 - w) + frames[j + 1] * w).astype(np.float32), grid


def diff_normalize(data):
    d = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)
    d = d / np.std(d)
    d = np.concatenate([d, np.zeros((1,) + d.shape[1:], np.float32)], 0)
    d[np.isnan(d)] = 0
    return d.astype(np.float32)


def standardize(data):
    d = (data - np.mean(data)) / np.std(data)
    d[np.isnan(d)] = 0
    return d.astype(np.float32)


# ---------------------------------------------------------------- HR (toolbox post_process)
def detrend(x, lam=100):
    n = len(x)
    I = np.identity(n)
    D2 = scipy.sparse.spdiags(np.array([np.ones(n), -2 * np.ones(n), np.ones(n)]), np.array([0, 1, 2]), n - 2, n).toarray()
    return np.dot(I - np.linalg.inv(I + lam ** 2 * np.dot(D2.T, D2)), x)


def hr_fft(sig, diff_flag, fs=FS):
    x = np.cumsum(sig) if diff_flag else np.asarray(sig, float)
    x = detrend(x, 100)
    b, a = scipy.signal.butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype="bandpass")
    x = scipy.signal.filtfilt(b, a, np.double(x))
    n = 1 << (len(x) - 1).bit_length()
    f, p = scipy.signal.periodogram(x[None, :], fs=fs, nfft=n, detrend=False)
    m = (f >= 0.6) & (f <= 3.3)
    return float(f[m][np.argmax(p[0][m])] * 60)


# ---------------------------------------------------------------- models
def load_model(name):
    import torch
    weights = HERE / "final_model_release" / MODELS[name][0]
    if name == "physnet":
        from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
        m = PhysNet_padding_Encoder_Decoder_MAX(frames=128)
    elif name == "tscan":
        from neural_methods.model.TS_CAN import TSCAN
        m = TSCAN(frame_depth=10, img_size=72)
    elif name == "deepphys":
        from neural_methods.model.DeepPhys import DeepPhys
        m = DeepPhys(img_size=72)
    elif name == "efficientphys":
        from neural_methods.model.EfficientPhys import EfficientPhys
        m = EfficientPhys(frame_depth=10, img_size=72)
    elif name == "physformer":
        from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp
        m = ViT_ST_ST_Compact3_TDC_gra_sharp(image_size=(160, 128, 128), patches=(4,) * 3, dim=96, ff_dim=144,
                                             num_heads=4, num_layers=12, dropout_rate=0.1, theta=0.7)
    elif name == "rhythmformer":
        from neural_methods.model.RhythmFormer import RhythmFormer
        m = RhythmFormer()
    elif name == "factorizephys":
        from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys
        md = {"FRAME_NUM": 160, "MD_TYPE": "NMF", "MD_FSAM": False, "MD_TRANSFORM": "T_KAB", "MD_S": 1, "MD_R": 1,
              "MD_STEPS": 4, "MD_INFERENCE": True, "MD_RESIDUAL": True}
        m = FactorizePhys(frames=160, md_config=md, in_channels=3, dropout=0.1, device=torch.device("cpu"))
    else:
        raise ValueError(name)
    try:
        state = torch.load(str(weights), map_location="cpu", weights_only=True)
    except TypeError:   # older torch without weights_only
        state = torch.load(str(weights), map_location="cpu")
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    missing, unexpected = m.load_state_dict(state, strict=(name != "factorizephys"))
    m.eval()
    return m


def run_model(name, model, frames):
    """frames: float32 [T,s,s,3] at 30 fps (0-255). Returns the predicted signal, one value per frame
    of the chunks that fit (T // chunk * chunk)."""
    import torch
    _, size, L, types_, _ = MODELS[name]
    data = np.concatenate([frames if t == "Raw" else diff_normalize(frames) if t == "DiffNormalized" else standardize(frames)
                           for t in types_], axis=-1)            # whole-video normalisation, as the toolbox
    out = []
    with torch.no_grad():
        for c in range(len(data) // L):
            clip = torch.from_numpy(np.ascontiguousarray(data[c * L:(c + 1) * L]))
            if name in ("physnet", "physformer", "factorizephys"):           # NCDHW
                x = clip.permute(3, 0, 1, 2).unsqueeze(0)
                if name == "physnet":
                    y = model(x)[0]
                elif name == "physformer":
                    y = model(x, 2.0)[0]
                else:
                    x = torch.cat([x, x[:, :, -1:]], 2)                      # one extra frame, as the trainer
                    y = model(x)[0]
                    y = (y - y.mean()) / y.std()
            elif name == "rhythmformer":                                      # NDCHW, batch 1
                y = model(clip.permute(0, 3, 1, 2).unsqueeze(0))
                y = (y - y.mean()) / y.std()
            else:                                                             # (N*D)CHW
                x = clip.permute(0, 3, 1, 2)
                if name == "efficientphys":
                    x = torch.cat([x, x[-1:]], 0)
                y = model(x)
            out.append(y.reshape(-1).numpy()[:L])
    return np.concatenate(out) if out else np.zeros(0)


def green_signal(frames):
    g = frames[..., 1].reshape(len(frames), -1).mean(axis=1)
    return 1 - g / g.mean()       # GREEN (Verkruysse 2008): more blood -> less green light


def pos_signal(frames):
    from unsupervised_methods.methods.POS_WANG import POS_WANG
    return np.asarray(POS_WANG(frames, FS)).reshape(-1)


# ---------------------------------------------------------------- references
def ubfc_ppg(subject_dir):
    rows = [r.split() for r in (subject_dir / "ground_truth.txt").read_text().splitlines() if r.strip()]
    return np.asarray([float(v) for v in rows[0]])


def tap_reference(csv_path):
    """Phone-page CSV: video start (page clock) + tap times -> function(t0, t1) giving the tapped
    BPM (60 / median interval) for a window measured from the start of the video."""
    start, taps, in_taps, page = None, [], False, []
    for line in Path(csv_path).read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("# video_start_s="):
            start = float(line.split("=", 1)[1])
        elif line.strip() == "tap_time_s":
            in_taps = True
        elif in_taps and line.strip():
            taps.append(float(line))
        elif not in_taps and line[:1].isdigit():   # the page's own live readings
            f = line.split(",")
            if len(f) >= 4 and f[1] and f[3] == "0":   # skip provisional readings
                page.append((float(f[0]), float(f[1])))
    if start is None:
        raise SystemExit("This CSV has no video start time. Record with the latest page (it saves the video and the CSV together).")
    taps = np.asarray(taps) - start
    page = [(t - start, b) for t, b in page]

    def ref(t0, t1):
        tt = taps[(taps >= t0) & (taps <= t1)]
        iv = np.diff(tt)
        iv = iv[(iv >= 0.25) & (iv <= 2.5)]
        return float(60 / np.median(iv)) if len(iv) >= 4 else None
    ref.page = page
    return ref


# ---------------------------------------------------------------- main
def evaluate(signals, ref_fn, n_frames):
    """signals: {method: (signal, diff_flag)}. Returns whole-video and per-window rows."""
    rows, wins = {}, []
    for m, (sig, diff) in signals.items():
        n = min(len(sig), n_frames)
        if n < FS * WINDOW_S:
            continue
        ref = ref_fn(0, n)
        rows[m] = (hr_fft(sig[:n], diff), ref)
        for w0 in range(0, n - FS * WINDOW_S + 1, FS * WINDOW_S):
            r = ref_fn(w0, w0 + FS * WINDOW_S)
            if r is not None:
                wins.append((m, w0 / FS, hr_fft(sig[w0:w0 + FS * WINDOW_S], diff), r))
    return rows, wins


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=HERE.parent / "data")
    p.add_argument("--subjects", nargs="+", type=int, default=list(range(31, 41)))
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--video", type=Path, help="your own recording (part 2)")
    p.add_argument("--taps", type=Path, help="the CSV exported together with --video")
    args = p.parse_args()
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        install_stubs()
    except ImportError:
        print("PyTorch is not installed, so only POS will run. Install it with:\n"
              "  .\\.venv\\Scripts\\python.exe -m pip install torch==2.4.1 -i https://pypi.tuna.tsinghua.edu.cn/simple\n")
        args.models = []
    out = HERE.parent / "results" / ("deep_compare_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    cache = HERE.parent / "results" / "deep_cache"
    out.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    print("Loading models:", ", ".join(args.models), flush=True)
    models = {}
    for m in args.models:
        try:
            models[m] = load_model(m)
        except Exception as exc:   # one broken model should not stop the others
            print("  {}: could not load ({})".format(m, exc), flush=True)

    if args.video and str(args.video).lower() == "latest":
        # newest heart_rate_*.webm/.mp4 in Downloads that has its CSV next to it
        dl = Path.home() / "Downloads"
        vids = sorted([v for ext in ("webm", "mp4") for v in dl.glob("heart_rate_*." + ext)
                       if (dl / (v.stem + ".csv")).is_file()], key=lambda v: v.stat().st_mtime)
        if not vids:
            p.error("no heart_rate_*.webm with a matching .csv in " + str(dl))
        args.video, args.taps = vids[-1], dl / (vids[-1].stem + ".csv")
        print("Using", args.video, "and", args.taps)
    if args.video:
        if not args.taps:
            p.error("--video needs --taps (the CSV saved with it)")
        if not args.video.is_file() or not args.taps.is_file():
            p.error("file not found: " + str(args.video if not args.video.is_file() else args.taps) +
                    "  (use the real paths, or --video latest for the newest recording in Downloads)")
        items = [(args.video.stem, args.video, tap_reference(args.taps), None)]
    else:
        items = []
        for s in args.subjects:
            d = args.data_root / "subject{}".format(s)
            if not (d / "vid.avi").is_file():
                print("subject{}: no vid.avi, skipped".format(s)); continue
            ppg = ubfc_ppg(d)
            items.append(("subject{}".format(s), d / "vid.avi",
                          (lambda ppg: lambda a, b: hr_fft(ppg[int(a):int(b)], False))(ppg), CHECKED_BOXES.get("subject{}".format(s))))

    per_video, windows = [], []
    for name, video, ref_fn, fixed in items:
        t0 = time.time()
        cfile = cache / (name + "_crops.npz")
        if cfile.is_file() and not args.video:
            z = np.load(str(cfile)); crops = {72: z["c72"], 128: z["c128"]}; times = z["t"]
        else:
            print("{}: reading video and cropping the face...".format(name), flush=True)
            crops, times, box = read_crops(video, (72, 128), fixed)
            np.savez(str(cfile), c72=crops[72], c128=crops[128], t=times)
        if args.video:   # phone/webcam recordings have uneven frame times
            f72, grid = resample_frames(crops[72].astype(np.float32), times)
            f128, _ = resample_frames(crops[128].astype(np.float32), times)
            ref = ref_fn
            ref_frames = lambda a, b: ref(a / FS, b / FS)
        else:
            f72, f128 = crops[72].astype(np.float32), crops[128].astype(np.float32)
            ref_frames = ref_fn
        n = len(f72)
        signals = {"POS": (pos_signal(f72), False), "GREEN": (green_signal(f72), False)}
        for m, model in models.items():
            ts = time.time()
            try:
                sig = run_model(m, model, f128 if MODELS[m][1] == 128 else f72)
                signals[m] = (sig, MODELS[m][4])
                print("  {:<14s} {:5.1f} s".format(m, time.time() - ts), flush=True)
            except Exception as exc:
                print("  {}: failed ({})".format(m, exc), flush=True)
        rows, wins = evaluate(signals, ref_frames, n)
        if args.video and getattr(ref_fn, "page", None):   # the web page's own live readings, same windows
            for w0 in range(0, n - FS * WINDOW_S + 1, FS * WINDOW_S):
                a, b = w0 / FS, w0 / FS + WINDOW_S
                r, vals = ref_fn(a, b), [v for t, v in ref_fn.page if a < t <= b]
                if r is not None and vals:
                    wins.append(("网页实时", a, float(np.mean(vals)), r))
        for m, (est, ref) in rows.items():
            per_video.append((name, m, est, ref))
        windows += [(name,) + w for w in wins]
        ref_txt = ", ".join("{} {:.1f}".format(m, e) for m, (e, r) in rows.items())
        print("{}: reference {:.1f} bpm | {}  ({:.0f} s)".format(name, list(rows.values())[0][1] or float("nan"), ref_txt, time.time() - t0), flush=True)

    # summary
    methods = ["POS", "GREEN"] + [m for m in models] + (["网页实时"] if args.video else [])
    lines = ["Deep rPPG models (trained on PURE) vs POS, same face crop. Reference: " +
             ("your taps" if args.video else "UBFC finger PPG"), ""]
    lines.append("{:<14s} {:>9s} {:>9s} {:>9s} {:>12s} {:>12s}".format("method", "MAE", "RMSE", "Pearson", "10s-win MAE", "10s within 5"))
    with (out / "per_video.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("video", "method", "estimated_bpm", "reference_bpm")] + per_video)
    with (out / "windows.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("video", "method", "window_start_s", "estimated_bpm", "reference_bpm")] + windows)
    for m in methods:
        pv = [(e, r) for (_, mm, e, r) in per_video if mm == m and r is not None]
        wv = [(e, r) for (_, mm, _, e, r) in windows if mm == m]
        if not pv and wv:   # the live page has window readings only
            we = np.array([abs(a - b) for a, b in wv])
            lines.append("{:<14s} {:>9s} {:>9s} {:>9s} {:12.2f} {:11.0f}%".format(m, "-", "-", "-", np.mean(we), 100 * np.mean(we <= 5)))
            continue
        if not pv:
            continue
        e = np.array([a - b for a, b in pv])
        corr = np.corrcoef([a for a, _ in pv], [b for _, b in pv])[0, 1] if len(pv) > 2 else float("nan")
        we = np.array([abs(a - b) for a, b in wv]) if wv else np.array([np.nan])
        lines.append("{:<14s} {:9.2f} {:9.2f} {:9.2f} {:12.2f} {:11.0f}%".format(
            m, np.mean(np.abs(e)), math.sqrt(np.mean(e ** 2)), corr, np.nanmean(we), 100 * np.mean(we <= 5)))
    lines += ["", "MAE/RMSE/Pearson: one HR per video (the toolbox's standard protocol).",
              "10s-win: one HR per 10 s window, closer to what a live reading sees.",
              "网页实时 (own video only): mean of the page's displayed (non-provisional) readings in each window."]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print("\nSaved to", out)


if __name__ == "__main__":
    main()
