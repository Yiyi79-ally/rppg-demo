# rPPG heart-rate demo

Contactless heart-rate measurement from an ordinary laptop or phone camera, built for our DIP course project.
Each heartbeat changes the colour of facial skin by a fraction of a percent; the camera picks it up and the
POS algorithm (Wang et al., 2017) turns it into a heart rate, updated every second.

**Try it:** open the live demo (English): `index.html` · Chinese version: `zh.html`
(on GitHub Pages: `https://yiyi79-ally.github.io/rppg-demo/`). Allow camera access, click **Start camera**,
keep still under even light. Everything runs in the browser; the video is never uploaded.
Not a medical device.

## Results (MAE = mean absolute error, beats per minute)

| Test | Result |
| --- | --- |
| UBFC-rPPG dataset, 9 held-out subjects, pulse-oximeter reference | 2.4 BPM |
| POS vs green-channel-only baseline (UBFC, 10 s windows) | 2.9 vs 9.3 BPM |
| Timestamp resampling vs none, 25% simulated frame loss | 3.1 vs 23.9 BPM |
| Real world, one person at rest, manual pulse reference | about 3 BPM (preliminary) |
| Real world, after exercise, green fill light: baseline vs sub-regions + tilt compensation | 9.4 vs 4.8 BPM (preliminary) |

## Pipeline

1. MediaPipe face landmarks outline the forehead and both cheeks
2. Average skin colour per region (glare pixels removed)
3. Resampling to an even 30 Hz grid using frame timestamps
4. POS algorithm, detrending and band-pass filter (0.6-3.3 Hz)
5. Spectrum per region (40-180 BPM), quality-weighted fusion, highest peak
6. Smoothing over 5 s and 10 s windows (median of the last 5 readings)

The signal processing is written from scratch in JavaScript (`src/rppg_core.js`).

## Files

| File | What it is |
| --- | --- |
| `index.html`, `zh.html` | The demo (English / Chinese), single self-contained files that also work offline |
| `src/rppg_core.js` | Signal processing: resampling, detrending, Butterworth band-pass, POS, spectrum, smoothing |
| `src/mp_face.js` | MediaPipe Face Landmarker wrapper and face regions |
| `src/face_tracker.js` | Fallback face tracker (pico.js) |
| `scripts/summarize_csv.py` | Error table from the CSV files the demo exports: `python summarize_csv.py <folder>` |
| `scripts/compare_deep_models.py` | Offline comparison with deep models; needs the rPPG-Toolbox environment |

## Datasets and credits

- UBFC-rPPG dataset: https://sites.google.com/view/ybenezeth/ubfcrppg (not redistributed here)
- Compressed UBFC videos: https://github.com/WangJieying/CompressedUBFC
- rPPG-Toolbox (offline evaluation): https://github.com/ubicomplab/rPPG-Toolbox
- MediaPipe Face Landmarker (Google, Apache-2.0), bundled inside the demo files
- pico.js face detector, Markus et al. 2013 (arXiv:1305.4537); the cascade is for non-commercial use with citation
- POS: W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, "Algorithmic Principles of Remote PPG", IEEE TBME, 2017
