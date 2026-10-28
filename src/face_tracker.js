// Face tracker for the browser: pico.js face detection (Markus et al. 2013,
// arXiv:1305.4537; MIT code, cascade for non-commercial use with citation)
// every few frames, plus per-frame skin-centroid following so the box keeps up
// with head motion between detections. Works offline: the cascade is embedded.
(function (root) {
  "use strict";

  // ---------------- pico.js (unchanged logic, from github.com/nenadmarkus/picojs)
  var pico = {};
  pico.unpack_cascade = function (bytes) {
    var dview = new DataView(new ArrayBuffer(4)), p = 8;
    dview.setUint8(0, bytes[p]); dview.setUint8(1, bytes[p + 1]); dview.setUint8(2, bytes[p + 2]); dview.setUint8(3, bytes[p + 3]);
    var tdepth = dview.getInt32(0, true); p += 4;
    dview.setUint8(0, bytes[p]); dview.setUint8(1, bytes[p + 1]); dview.setUint8(2, bytes[p + 2]); dview.setUint8(3, bytes[p + 3]);
    var ntrees = dview.getInt32(0, true); p += 4;
    var tcodes_ls = [], tpreds_ls = [], thresh_ls = [];
    for (var t = 0; t < ntrees; ++t) {
      Array.prototype.push.apply(tcodes_ls, [0, 0, 0, 0]);
      Array.prototype.push.apply(tcodes_ls, bytes.slice(p, p + 4 * Math.pow(2, tdepth) - 4));
      p = p + 4 * Math.pow(2, tdepth) - 4;
      for (var i = 0; i < Math.pow(2, tdepth); ++i) {
        dview.setUint8(0, bytes[p]); dview.setUint8(1, bytes[p + 1]); dview.setUint8(2, bytes[p + 2]); dview.setUint8(3, bytes[p + 3]);
        tpreds_ls.push(dview.getFloat32(0, true)); p += 4;
      }
      dview.setUint8(0, bytes[p]); dview.setUint8(1, bytes[p + 1]); dview.setUint8(2, bytes[p + 2]); dview.setUint8(3, bytes[p + 3]);
      thresh_ls.push(dview.getFloat32(0, true)); p += 4;
    }
    var tcodes = new Int8Array(tcodes_ls), tpreds = new Float32Array(tpreds_ls), thresh = new Float32Array(thresh_ls);
    return function (r, c, s, pixels, ldim) {
      r = 256 * r; c = 256 * c;
      var root = 0, o = 0.0, pow2tdepth = Math.pow(2, tdepth) >> 0;
      for (var i = 0; i < ntrees; ++i) {
        var idx = 1;
        for (var j = 0; j < tdepth; ++j)
          idx = 2 * idx + (pixels[((r + tcodes[root + 4 * idx + 0] * s) >> 8) * ldim + ((c + tcodes[root + 4 * idx + 1] * s) >> 8)] <= pixels[((r + tcodes[root + 4 * idx + 2] * s) >> 8) * ldim + ((c + tcodes[root + 4 * idx + 3] * s) >> 8)]);
        o = o + tpreds[pow2tdepth * i + idx - pow2tdepth];
        if (o <= thresh[i]) return -1;
        root += 4 * pow2tdepth;
      }
      return o - thresh[ntrees - 1];
    };
  };
  pico.run_cascade = function (image, classify_region, params) {
    var pixels = image.pixels, nrows = image.nrows, ncols = image.ncols, ldim = image.ldim;
    var shiftfactor = params.shiftfactor, minsize = params.minsize, maxsize = params.maxsize, scalefactor = params.scalefactor;
    var scale = minsize, detections = [];
    while (scale <= maxsize) {
      var step = Math.max(shiftfactor * scale, 1) >> 0, offset = (scale / 2 + 1) >> 0;
      for (var r = offset; r <= nrows - offset; r += step)
        for (var c = offset; c <= ncols - offset; c += step) {
          var q = classify_region(r, c, scale, pixels, ldim);
          if (q > 0.0) detections.push([r, c, scale, q]);
        }
      scale = scale * scalefactor;
    }
    return detections;
  };
  pico.cluster_detections = function (dets, iouthreshold) {
    dets = dets.sort(function (a, b) { return b[3] - a[3]; });
    var calculate_iou = function (det1, det2) {
      var r1 = det1[0], c1 = det1[1], s1 = det1[2], r2 = det2[0], c2 = det2[1], s2 = det2[2];
      var overr = Math.max(0, Math.min(r1 + s1 / 2, r2 + s2 / 2) - Math.max(r1 - s1 / 2, r2 - s2 / 2));
      var overc = Math.max(0, Math.min(c1 + s1 / 2, c2 + s2 / 2) - Math.max(c1 - s1 / 2, c2 - s2 / 2));
      return overr * overc / (s1 * s1 + s2 * s2 - overr * overc);
    };
    var assignments = new Array(dets.length).fill(0), clusters = [];
    for (var i = 0; i < dets.length; ++i) {
      if (assignments[i] === 0) {
        var r = 0.0, c = 0.0, s = 0.0, q = 0.0, n = 0;
        for (var j = i; j < dets.length; ++j)
          if (calculate_iou(dets[i], dets[j]) > iouthreshold) {
            assignments[j] = 1; r += dets[j][0]; c += dets[j][1]; s += dets[j][2]; q += dets[j][3]; n++;
          }
        clusters.push([r / n, c / n, s / n, q]);
      }
    }
    return clusters;
  };

  // ---------------- helpers
  function isSkin(r, g, b) {
    var y = 0.299 * r + 0.587 * g + 0.114 * b;
    var cr = (r - y) * 0.713 + 128, cb = (b - y) * 0.564 + 128;
    return y > 40 && cr >= 133 && cr <= 173 && cb >= 77 && cb <= 127;
  }
  function iou(a, b) {
    var ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    var iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    var inter = ix * iy, uni = a.w * a.h + b.w * b.h - inter;
    return uni > 0 ? inter / uni : 0;
  }
  function skinStats(img, x0, y0, x1, y1) {
    var d = img.data, W = img.width, n = 0, sx = 0, sy = 0, tot = 0;
    x0 = Math.max(0, x0 | 0); y0 = Math.max(0, y0 | 0); x1 = Math.min(W, x1 | 0); y1 = Math.min(img.height, y1 | 0);
    for (var y = y0; y < y1; y++) for (var x = x0; x < x1; x++) {
      var i = (y * W + x) * 4; tot++;
      if (isSkin(d[i], d[i + 1], d[i + 2])) { n++; sx += x; sy += y; }
    }
    return { n: n, frac: tot ? n / tot : 0, cx: n ? sx / n : 0, cy: n ? sy / n : 0 };
  }
  function base64ToBytes(b64) {
    var bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    var out = new Int8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i) << 24 >> 24;
    return out;
  }

  // ---------------- tracker (works in the pixel space of the small frame it is given)
  function FaceTracker(cascadeBase64, opts) {
    opts = opts || {};
    this.classify = pico.unpack_cascade(base64ToBytes(cascadeBase64));
    this.every = opts.detectEvery || 3;
    this.box = null;            // {x, y, w, h} in pixels of the given frame
    this.offset = null;         // box centre minus skin centroid, measured at detection
    this.frame = 0; this.sinceDet = 0; this.lowSkin = 0;
    this.lastScore = 0; this.source = "none"; this.thumb = null;
  }

  // 16x16 luminance thumbnail; a large mean change between frames = scene cut
  FaceTracker.prototype.cutCheck = function (img) {
    var W = img.width, H = img.height, d = img.data, th = new Float32Array(256), cnt = new Float32Array(256);
    for (var y = 0; y < H; y += 2) for (var x = 0; x < W; x += 2) {
      var i = (y * W + x) * 4, k = ((y * 16 / H) | 0) * 16 + ((x * 16 / W) | 0);
      th[k] += (2 * d[i] + 7 * d[i + 1] + d[i + 2]) / 10; cnt[k]++;
    }
    var diff = 0;
    for (var q = 0; q < 256; q++) th[q] /= Math.max(1, cnt[q]);
    if (this.thumb) { for (var r = 0; r < 256; r++) diff += Math.abs(th[r] - this.thumb[r]); diff /= 256; }
    this.thumb = th;
    return diff;
  };

  FaceTracker.prototype.detect = function (img) {
    var W = img.width, H = img.height, d = img.data, gray = new Uint8Array(W * H);
    for (var i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = (2 * d[j] + 7 * d[j + 1] + d[j + 2]) / 10;
    var m = Math.min(W, H);
    var dets = pico.run_cascade({ pixels: gray, nrows: H, ncols: W, ldim: W }, this.classify,
      { shiftfactor: 0.08, minsize: Math.max(16, 0.1 * m), maxsize: 1.3 * Math.max(W, H), scalefactor: 1.1 });
    dets = pico.cluster_detections(dets, 0.2);
    return dets.map(function (q) { return { x: q[1] - q[2] / 2, y: q[0] - q[2] / 2, w: q[2], h: q[2], score: q[3] }; });
  };

  FaceTracker.prototype.update = function (img) {
    var res = { jumped: false, lost: false, cut: false };
    this.frame++; this.sinceDet++;
    if (this.cutCheck(img) > 25) {   // shot change: the old position means nothing
      res.cut = true; this.box = null; this.offset = null; this.sinceDet = 0; this.lowSkin = 0;
    }
    var tryDetect = this.frame % (this.box ? this.every : 2) === 0;
    if (tryDetect) {
      var dets = this.detect(img), self = this, pick = null;
      if (this.box) {
        var bestIou = 0;
        dets.forEach(function (q) { var v = iou(q, self.box); if (v > 0.3 && q.score > 5 && v > bestIou) { bestIou = v; pick = q; } });
        if (!pick) {
          var strong = dets.filter(function (q) { return q.score > 40; }).sort(function (a, b) { return b.score - a.score; })[0];
          if (strong && iou(strong, this.box) < 0.1) { pick = strong; res.jumped = true; }
        }
      } else {
        pick = dets.filter(function (q) { return q.score > 15; }).sort(function (a, b) { return b.score - a.score; })[0] || null;
      }
      if (pick) {
        if (!this.box || res.jumped) this.box = { x: pick.x, y: pick.y, w: pick.w, h: pick.h };
        else {
          var a = 0.6;
          this.box = { x: a * this.box.x + (1 - a) * pick.x, y: a * this.box.y + (1 - a) * pick.y,
                       w: a * this.box.w + (1 - a) * pick.w, h: a * this.box.h + (1 - a) * pick.h };
        }
        var s0 = skinStats(img, this.box.x, this.box.y, this.box.x + this.box.w, this.box.y + this.box.h);
        this.offset = s0.n > 20 ? { x: this.box.x + this.box.w / 2 - s0.cx, y: this.box.y + this.box.h / 2 - s0.cy } : null;
        this.sinceDet = 0; this.lastScore = pick.score; this.source = "detect";
      }
    }
    if (!this.box) { res.lost = true; res.box = null; res.source = "none"; return res; }
    // Per-frame following: move the box with the skin centroid around it
    var b = this.box, ex = 0.25 * b.w, ey = 0.25 * b.h;
    var s = skinStats(img, b.x - ex, b.y - ey, b.x + b.w + ex, b.y + b.h + ey);
    var inBox = skinStats(img, b.x, b.y, b.x + b.w, b.y + b.h);
    if (this.offset && s.n > 0.15 * b.w * b.h) {
      var tx = s.cx + this.offset.x - b.w / 2, ty = s.cy + this.offset.y - b.h / 2;
      b.x += 0.5 * (tx - b.x); b.y += 0.5 * (ty - b.y);
      if (this.source !== "detect" || this.sinceDet > 0) this.source = this.sinceDet > 0 ? "track" : this.source;
    }
    this.lowSkin = inBox.frac < 0.12 ? this.lowSkin + 1 : 0;
    if (this.lowSkin > 15) { this.box = null; this.offset = null; res.lost = true; res.box = null; res.source = "none"; return res; }
    res.box = { x: b.x, y: b.y, w: b.w, h: b.h }; res.source = this.source; res.score = this.lastScore;
    return res;
  };

  // manual placement (viewer taps the face): follow it by skin colour from here
  FaceTracker.prototype.place = function (img, box) {
    this.box = { x: box.x, y: box.y, w: box.w, h: box.h };
    var s0 = skinStats(img, box.x, box.y, box.x + box.w, box.y + box.h);
    this.offset = s0.n > 20 ? { x: box.x + box.w / 2 - s0.cx, y: box.y + box.h / 2 - s0.cy } : { x: 0, y: 0 };
    this.source = "manual"; this.lowSkin = 0; this.sinceDet = 0;
  };

  // forehead + both cheeks inside a square face box (common rPPG ROI choice)
  FaceTracker.regions = function (b) {
    return [
      { x: b.x + 0.30 * b.w, y: b.y + 0.08 * b.h, w: 0.40 * b.w, h: 0.20 * b.h },
      { x: b.x + 0.12 * b.w, y: b.y + 0.48 * b.h, w: 0.24 * b.w, h: 0.24 * b.h },
      { x: b.x + 0.64 * b.w, y: b.y + 0.48 * b.h, w: 0.24 * b.w, h: 0.24 * b.h }
    ];
  };
  FaceTracker.isSkin = isSkin;
  FaceTracker.pico = pico;

  if (typeof module !== "undefined" && module.exports) module.exports = FaceTracker; else root.FaceTracker = FaceTracker;
})(typeof self !== "undefined" ? self : this);
