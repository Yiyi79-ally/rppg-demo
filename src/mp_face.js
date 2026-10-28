// MediaPipe Face Landmarker wrapper for the rPPG page: loads the model (from URLs or from
// inlined base64), tracks one face per video frame and returns skin regions as polygons that
// follow head rotation (forehead, both cheeks, face oval minus eyes / brows / mouth).
// MediaPipe tasks-vision and the face_landmarker model are Apache-2.0 (Google).
(function (root) {
  "use strict";
  // landmark indices of the canonical 468/478-point face mesh
  var IDX = {
    forehead: [103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 108, 69, 104],
    lcheek: [116, 117, 118, 101, 36, 205, 187, 147, 123],      // left side of the image
    rcheek: [345, 346, 347, 330, 266, 425, 411, 376, 352],
    // larger regions: whole forehead between hairline and brows, cheeks from under the eye
    // to the nasolabial fold (more skin pixels = less camera noise in the average)
    foreheadBig: [54, 103, 67, 109, 10, 338, 297, 332, 284, 300, 293, 334, 296, 336, 9, 107, 66, 105, 63, 70],
    lcheekBig: [116, 117, 118, 119, 120, 47, 126, 142, 203, 206, 216, 207, 187, 147, 123],
    rcheekBig: [345, 346, 347, 348, 349, 277, 355, 371, 423, 426, 436, 427, 411, 376, 352],
    oval: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
           148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    holes: [
      [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],                 // brow
      [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],        // brow
      [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],    // eye
      [362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382], // eye
      [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146] // lips
    ]
  };

  function b64ToBytes(b64) {
    var bin = atob(b64), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function blobUrl(bytesOrText, type) { return URL.createObjectURL(new Blob([bytesOrText], { type: type })); }

  // src: {bundleUrl, wasmJsUrl, wasmUrl, modelUrl} or {bundleB64, wasmJsB64, wasmB64, modelB64}
  function create(src, opts) {
    opts = opts || {};
    var bundleUrl = src.bundleUrl || blobUrl(b64ToBytes(src.bundleB64), "text/javascript");
    var fileset = {
      wasmLoaderPath: src.wasmJsUrl || blobUrl(b64ToBytes(src.wasmJsB64), "text/javascript"),
      wasmBinaryPath: src.wasmUrl || blobUrl(b64ToBytes(src.wasmB64), "application/wasm")
    };
    var modelP = src.modelB64 ? Promise.resolve(b64ToBytes(src.modelB64))
      : fetch(src.modelUrl).then(function (r) { return r.arrayBuffer(); }).then(function (b) { return new Uint8Array(b); });
    return import(bundleUrl).then(function (mp) {
      return modelP.then(function (model) {
        var make = function (delegate) {
          return mp.FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetBuffer: model, delegate: delegate },
            runningMode: "VIDEO", numFaces: 1,
            minFaceDetectionConfidence: opts.detect || 0.5, minFacePresenceConfidence: opts.presence || 0.5,
            minTrackingConfidence: opts.track || 0.5
          });
        };
        if (opts.delegate === "CPU") return make("CPU").then(function (lm) { return new MPFace(lm, "CPU", null, opts); });
        var cpu = function () { return make("CPU").then(function (lm) { return new MPFace(lm, "CPU", null); }); };
        return make("GPU").then(function (lm) {
          // quick benchmark on a blank frame: software WebGL is far slower than the CPU path
          var c = document.createElement("canvas"); c.width = 256; c.height = 256;
          c.getContext("2d").fillRect(0, 0, 256, 256);
          var ts = [];
          for (var i = 0; i < 6; i++) { var t0 = performance.now(); lm.detectForVideo(c, i + 1); ts.push(performance.now() - t0); }
          var m = ts.slice(2).sort(function (a, b) { return a - b; })[1];
          if (m > 40) { try { lm.close(); } catch (e) {} return cpu(); }
          var f = new MPFace(lm, "GPU", make, opts); f.lastTs = 6; return f;
        }, cpu);
      });
    });
  }

  // GPU is usually fastest, but some browsers fall back to software WebGL, which is far slower
  // than the CPU (XNNPACK) path: time the first frames and switch to CPU if GPU is slow.
  function MPFace(lm, delegate, make, opts) { this.smooth = (opts && opts.smooth) || 0; this.prev = null; this.lm = lm; this.delegate = delegate; this.make = make; this.lastTs = -1; this.times = []; this.switching = false; }
  // frame: video element or canvas; ts in ms (must increase). Returns null or
  // {pts: [[x,y] normalised 0..1], box: {x,y,w,h} normalised}
  MPFace.prototype.detect = function (frame, ts) {
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;
    var t0 = performance.now(), r = this.lm.detectForVideo(frame, ts), self = this;
    if (this.delegate === "GPU" && !this.switching && this.times.length < 8) {
      this.times.push(performance.now() - t0);
      if (this.times.length === 8) {   // skip 2 warm-up frames, median of the next 6
        var m = this.times.slice(2).sort(function (a, b) { return a - b; })[3];
        if (m > 40) {
          this.switching = true;
          this.make("CPU").then(function (lm) { var old = self.lm; self.lm = lm; self.delegate = "CPU"; self.lastTs = -1; try { old.close(); } catch (e) {} });
        }
      }
    }
    if (!r || !r.faceLandmarks || !r.faceLandmarks.length) { this.prev = null; return null; }
    var L = r.faceLandmarks[0], pts = new Array(L.length), x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    // light temporal smoothing of the landmarks (less flicker of region edges); a big move
    // (more than 30% of the face width since the last frame) is taken as is
    var a = this.smooth, P0 = this.prev;
    if (a && P0 && P0.length === L.length) {
      var w0 = Math.abs(P0[454][0] - P0[234][0]) || 1e-3;
      if (Math.hypot(L[1].x - P0[1][0], L[1].y - P0[1][1]) > 0.3 * w0) a = 0;
    } else a = 0;
    for (var i = 0; i < L.length; i++) {
      var x = a ? a * P0[i][0] + (1 - a) * L[i].x : L[i].x, y = a ? a * P0[i][1] + (1 - a) * L[i].y : L[i].y; pts[i] = [x, y];
      if (i < 468) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
    }
    this.prev = pts;
    return { pts: pts, box: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } };
  };

  // polygons in pixel coordinates of a W x H image
  function regions(face, W, H) {
    var P = function (ids) { return ids.map(function (k) { return [face.pts[k][0] * W, face.pts[k][1] * H]; }); };
    return { forehead: P(IDX.forehead), lcheek: P(IDX.lcheek), rcheek: P(IDX.rcheek), oval: P(IDX.oval),
             foreheadBig: P(IDX.foreheadBig), lcheekBig: P(IDX.lcheekBig), rcheekBig: P(IDX.rcheekBig),
             holes: IDX.holes.map(P) };
  }
  function inPoly(x, y, poly) {
    var c = false;
    for (var i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      var xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }
  // pixel indices (into RGBA data) inside any of `polys` and outside all `holes`
  function pixelsIn(polys, holes, W, H) {
    var out = [];
    polys.forEach(function (poly) {
      var bx0 = W, by0 = H, bx1 = 0, by1 = 0;
      poly.forEach(function (p) { bx0 = Math.min(bx0, p[0]); bx1 = Math.max(bx1, p[0]); by0 = Math.min(by0, p[1]); by1 = Math.max(by1, p[1]); });
      bx0 = Math.max(0, Math.floor(bx0)); by0 = Math.max(0, Math.floor(by0)); bx1 = Math.min(W - 1, Math.ceil(bx1)); by1 = Math.min(H - 1, Math.ceil(by1));
      for (var y = by0; y <= by1; y++) for (var x = bx0; x <= bx1; x++) {
        var cx = x + 0.5, cy = y + 0.5;
        if (!inPoly(cx, cy, poly)) continue;
        var hole = false;
        for (var h = 0; h < (holes || []).length && !hole; h++) if (inPoly(cx, cy, holes[h])) hole = true;
        if (!hole) out.push((y * W + x) * 4);
      }
    });
    return out;
  }

  MPFace.IDX = IDX; MPFace.create = create; MPFace.regions = regions; MPFace.pixelsIn = pixelsIn; MPFace.inPoly = inPoly;
  if (typeof module !== "undefined" && module.exports) module.exports = MPFace; else root.MPFace = MPFace;
})(typeof self !== "undefined" ? self : this);
