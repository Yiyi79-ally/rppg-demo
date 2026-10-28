#!/usr/bin/env python
r"""Summarise the phone page's CSV exports: how close the live reading was to your taps.

Save in D:\DIP\rPPG-Toolbox-main and run (no extra packages needed):
  .\.venv\Scripts\python.exe .\summarize_csv.py                 (all heart_rate_*.csv in Downloads)
  .\.venv\Scripts\python.exe .\summarize_csv.py D:\DIP\results\my_recordings

Only non-provisional readings that have a tapped reference at the same second are scored.
Per file and per lighting + algorithm (off/POS, white/POS, green/GREEN, green/POS, green/REG): number of readings, mean absolute error of the
displayed value and of the raw 5 s value, share within 5 bpm, and how often it was off by > 15.
"""
import csv
import sys
from pathlib import Path


def load(path):
    rows = []
    with open(str(path), encoding="utf-8-sig") as f:
        header = None
        for r in csv.reader(f):
            if not r or r[0].startswith("#"):
                continue
            if r[0] == "time_s":
                header = r
                continue
            if r[0] == "tap_time_s":
                break
            if header and r[0][:1].isdigit():
                rows.append(dict(zip(header, r)))
    return rows


def score(rows):
    shown, raw = [], []
    for r in rows:
        if r.get("provisional") == "0" and r.get("tap_bpm"):
            ref = float(r["tap_bpm"])
            shown.append(abs(float(r["camera_bpm"]) - ref))
            raw.append(abs(float(r["raw_5s_bpm"]) - ref))
    if not shown:
        return None
    n = len(shown)
    return (n, sum(shown) / n, sum(raw) / n, 100.0 * sum(e <= 5 for e in shown) / n, sum(e > 15 for e in shown))


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads"
    files = sorted(folder.glob("heart_rate_*.csv"))
    if not files:
        sys.exit("no heart_rate_*.csv in " + str(folder))
    fmt = "{:<34s} {:<20s} {:>5s} {:>9s} {:>9s} {:>9s} {:>6s}"
    print(fmt.format("file", "light", "n", "MAE", "raw MAE", "<=5", ">15"))
    groups, seen = {}, set()
    for f in files:
        # older exports could repeat earlier sessions of the same page visit: skip rows already seen
        rows = [r for r in load(f) if (r["time_s"], r["camera_bpm"], r.get("tap_bpm")) not in seen]
        seen.update((r["time_s"], r["camera_bpm"], r.get("tap_bpm")) for r in rows)
        by = {}
        for r in rows:
            light = r.get("light", "?") or "?"
            # comparison build: every improvement variant logged as <variant>_bpm / <variant>_raw_5s
            variants = [k[:-4] for k in r if k.endswith("_bpm") and k != "camera_bpm" and k != "tap_bpm"
                        and r.get(k) and r.get(k[:-4] + "_raw_5s")]
            if "base_bpm" in r:
                for v in variants:
                    a = dict(r, camera_bpm=r[v + "_bpm"], raw_5s_bpm=r[v + "_raw_5s"])
                    by.setdefault(light + "/" + (r.get("method") or "?") + "/" + v, []).append(a)
                continue
            by.setdefault(light + "/" + (r.get("method") or "?"), []).append(r)
            # earlier build: every algorithm logged (pos_/green_/reg_ columns)
            done = False
            for v in variants:
                if v.upper() != (r.get("method") or ""):
                    a = dict(r, camera_bpm=r[v + "_bpm"], raw_5s_bpm=r[v + "_raw_5s"])
                    by.setdefault(light + "/" + v.upper(), []).append(a)
                    done = True
            # one build in between logged only the other algorithm (alt_*)
            if not done and r.get("alt_method") and r.get("alt_bpm") and r.get("alt_raw_5s_bpm"):
                a = dict(r, camera_bpm=r["alt_bpm"], raw_5s_bpm=r["alt_raw_5s_bpm"])
                by.setdefault(light + "/" + r["alt_method"], []).append(a)
        for light, rr in sorted(by.items()):
            s = score(rr)
            if s:
                groups.setdefault(light, []).extend(rr)
                print(fmt.format(f.name, light, str(s[0]), "%.1f" % s[1], "%.1f" % s[2], "%.0f%%" % s[3], str(s[4])))
    print("\nAll files together, by lighting / algorithm:")
    for light, rr in sorted(groups.items()):
        s = score(rr)
        print(fmt.format("", light, str(s[0]), "%.1f" % s[1], "%.1f" % s[2], "%.0f%%" % s[3], str(s[4])))
    print("\nMAE = displayed reading vs your taps; raw MAE = the raw 5 s value before the anti-jump step."
          "\nlight/POS/<variant>: comparison build (base, tilt, harm, motion, patch, combo -- all from the same frames)."
          "\ngreen/GREEN, green/POS and green/REG come from the same session (computed at once), so they compare fairly. REG = green minus what red/blue explain."
          "\nlight '?' = exported by an older page version that did not record the lighting.")


if __name__ == "__main__":
    main()
