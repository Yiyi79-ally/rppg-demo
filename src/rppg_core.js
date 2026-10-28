// rPPG signal core: same maths as the Python pipeline (rPPG-Toolbox POS_WANG +
// run_pos_3sec.estimate_hr), plus Nature-2026-inspired confidence and smoothing.
(function (root) {
  "use strict";

  // ---- linear resampling onto an even grid (frame rate standardisation)
  function resample(times, values, fs) {
    var t0 = times[0], t1 = times[times.length - 1];
    var n = Math.floor((t1 - t0) * fs + 1e-9) + 1;
    var out = new Float64Array(n), j = 0;
    for (var i = 0; i < n; i++) {
      var t = t0 + i / fs;
      while (j < times.length - 2 && times[j + 1] < t) j++;
      var ta = times[j], tb = times[j + 1];
      var w = tb > ta ? (t - ta) / (tb - ta) : 0;
      if (w < 0) w = 0; if (w > 1) w = 1;
      out[i] = values[j] + w * (values[j + 1] - values[j]);
    }
    return out;
  }

  function mean(a, s, e) { var m = 0; for (var i = s; i < e; i++) m += a[i]; return m / (e - s); }
  function std(a, s, e) { // population std (numpy default)
    var m = mean(a, s, e), v = 0;
    for (var i = s; i < e; i++) v += (a[i] - m) * (a[i] - m);
    return Math.sqrt(v / (e - s));
  }

  // ---- smoothness-priors detrend: x - (I + lam^2 D'D)^-1 x, banded Cholesky
  function detrend(x, lam) {
    var n = x.length, l2 = lam * lam;
    if (n < 3) return Float64Array.from(x);
    // D'D for second differences is pentadiagonal; store A = I + l2 D'D
    var d0 = new Float64Array(n), d1 = new Float64Array(n), d2 = new Float64Array(n);
    for (var k = 0; k < n - 2; k++) { // row k of D = [1,-2,1] at k..k+2
      var c = [1, -2, 1];
      for (var p = 0; p < 3; p++) for (var q = p; q < 3; q++) {
        var v = l2 * c[p] * c[q], i = k + p, off = q - p;
        if (off === 0) d0[i] += v; else if (off === 1) d1[i] += v; else d2[i] += v;
      }
    }
    for (var i2 = 0; i2 < n; i2++) d0[i2] += 1;
    // Cholesky A = L L', L lower with bandwidth 2
    var L0 = new Float64Array(n), L1 = new Float64Array(n), L2 = new Float64Array(n);
    for (var r = 0; r < n; r++) {
      if (r >= 2) L2[r] = d2[r - 2] / L0[r - 2];
      if (r >= 1) L1[r] = (d1[r - 1] - (r >= 2 ? L2[r] * L1[r - 1] : 0)) / L0[r - 1];
      var s = d0[r] - L1[r] * L1[r] - L2[r] * L2[r];
      L0[r] = Math.sqrt(s);
    }
    var y = new Float64Array(n), z = new Float64Array(n);
    for (var a = 0; a < n; a++) {
      var t = x[a];
      if (a >= 1) t -= L1[a] * y[a - 1];
      if (a >= 2) t -= L2[a] * y[a - 2];
      y[a] = t / L0[a];
    }
    for (var b = n - 1; b >= 0; b--) {
      var u = y[b];
      if (b + 1 < n) u -= L1[b + 1] * z[b + 1];
      if (b + 2 < n) u -= L2[b + 2] * z[b + 2];
      z[b] = u / L0[b];
    }
    var out = new Float64Array(n);
    for (var o = 0; o < n; o++) out[o] = x[o] - z[o];
    return out;
  }

  // ---- order-1 Butterworth band-pass, identical to scipy.signal.butter(1, [lo, hi], 'bandpass')
  function butterBand(lowHz, highHz, fs) {
    var w1 = 4 * Math.tan(Math.PI * (lowHz / (fs / 2)) / 2);
    var w2 = 4 * Math.tan(Math.PI * (highHz / (fs / 2)) / 2);
    var bw = w2 - w1, w0sq = w1 * w2, k = 4;
    var a0 = k * k + bw * k + w0sq;
    return { b: [bw * k / a0, 0, -bw * k / a0], a: [1, (2 * w0sq - 2 * k * k) / a0, (k * k - bw * k + w0sq) / a0] };
  }

  function lfilter(b, a, x, z0, z1) { // direct form II transposed, a[0] = 1
    var y = new Float64Array(x.length);
    for (var i = 0; i < x.length; i++) {
      var yi = b[0] * x[i] + z0;
      z0 = b[1] * x[i] - a[1] * yi + z1;
      z1 = b[2] * x[i] - a[2] * yi;
      y[i] = yi;
    }
    return y;
  }

  // scipy.signal.filtfilt defaults: odd extension, padlen = 3*max(len(a),len(b)), lfilter_zi
  function filtfilt(f, x) {
    var b = f.b, a = f.a, n = x.length, pad = 9;
    if (n <= pad) throw new Error("signal too short for filtfilt");
    // lfilter_zi: solve [[1+a1, -1], [a2, 1]] zi = b[1:] - a[1:] b0
    var B1 = b[1] - a[1] * b[0], B2 = b[2] - a[2] * b[0];
    var det = (1 + a[1]) * 1 - (-1) * a[2];
    var zi0 = (B1 * 1 - (-1) * B2) / det, zi1 = ((1 + a[1]) * B2 - a[2] * B1) / det;
    var ext = new Float64Array(n + 2 * pad);
    for (var i = 0; i < pad; i++) ext[i] = 2 * x[0] - x[pad - i];
    for (var j = 0; j < n; j++) ext[pad + j] = x[j];
    for (var k = 0; k < pad; k++) ext[pad + n + k] = 2 * x[n - 1] - x[n - 2 - k];
    var y = lfilter(b, a, ext, zi0 * ext[0], zi1 * ext[0]);
    var r = new Float64Array(y.length);
    for (var m = 0; m < y.length; m++) r[m] = y[y.length - 1 - m];
    var y2 = lfilter(b, a, r, zi0 * r[0], zi1 * r[0]);
    var out = new Float64Array(n);
    for (var q = 0; q < n; q++) out[q] = y2[y2.length - 1 - (pad + q)];
    return out;
  }

  // ---- GREEN (Verkruysse et al. 2008): the green channel alone, inverted (more blood -> less
  // green light reflected). Used under a green fill light, where red and blue carry little light
  // and POS (which combines all three channels) has nothing to work with.
  function green(R, G, B) {
    var n = G.length, m = 0, out = new Float64Array(n), i;
    for (i = 0; i < n; i++) m += G[i];
    m /= n;
    for (i = 0; i < n; i++) out[i] = 1 - G[i] / m;
    return out;
  }

  // ---- POS (Wang et al. 2017), line-by-line port of rPPG-Toolbox POS_WANG
  function pos(R, G, B, fs, opts) {
    var N = R.length, l = Math.ceil(1.6 * fs), H = new Float64Array(N);
    var s0 = new Float64Array(l), s1 = new Float64Array(l), h = new Float64Array(l);
    for (var n = 0; n < N; n++) {
      var m = n - l;
      if (m < 0) continue;
      var mr = mean(R, m, n), mg = mean(G, m, n), mb = mean(B, m, n);
      for (var i = 0; i < l; i++) {
        var cr = mr > 1e-6 ? R[m + i] / mr : 0, cg = mg > 1e-6 ? G[m + i] / mg : 0, cb = mb > 1e-6 ? B[m + i] / mb : 0;   // a dark channel (e.g. under green light) adds nothing instead of NaN
        s0[i] = cg - cb; s1[i] = -2 * cr + cg + cb;
      }
      var alpha = std(s0, 0, l) / std(s1, 0, l), hm = 0;
      for (var j = 0; j < l; j++) { h[j] = s0[j] + alpha * s1[j]; hm += h[j]; }
      hm /= l;
      for (var k = 0; k < l; k++) H[m + k] += h[k] - hm;
    }
    var bvp = detrend(H, 100);
    return filtfilt(butterBand(0.75, 3, fs), bvp);
  }

  // ---- REG: green channel minus what red and blue can explain (for the green fill light)
  // Under a green light the pulse is mostly in G, while R and B (lit by the room) mainly carry
  // motion / breathing brightness changes. In each 1.6 s window (same overlap-add as POS) the
  // normalised channels are centred and G is regressed on R and B by least squares:
  //   h = g - k1*r - k2*b,  (k1, k2) = argmin |g - k1*r - k2*b|^2  (small ridge for stability)
  // A channel that is dark (mean ~ 0) is left out.
  function reg(R, G, B, fs) {
    var N = R.length, l = Math.ceil(1.6 * fs), H = new Float64Array(N);
    var r = new Float64Array(l), g = new Float64Array(l), b = new Float64Array(l);
    for (var n = 0; n < N; n++) {
      var m = n - l;
      if (m < 0) continue;
      var mr = mean(R, m, n), mg = mean(G, m, n), mb = mean(B, m, n), i;
      if (!(mg > 1e-6)) continue;
      var ur = mr > 0.5, ub = mb > 0.5;          // channel has some light (8-bit scale)
      for (i = 0; i < l; i++) { r[i] = ur ? R[m + i] / mr - 1 : 0; g[i] = G[m + i] / mg - 1; b[i] = ub ? B[m + i] / mb - 1 : 0; }
      var cr = 0, cb = 0, gr = 0, gb = 0, rr = 0, bb = 0, rb = 0;
      for (i = 0; i < l; i++) { cr += r[i]; cb += b[i]; } cr /= l; cb /= l;
      for (i = 0; i < l; i++) { r[i] -= cr; b[i] -= cb; rr += r[i] * r[i]; bb += b[i] * b[i]; rb += r[i] * b[i]; gr += g[i] * r[i]; gb += g[i] * b[i]; }
      var lam = 1e-3 * (rr + bb) + 1e-12, a = rr + lam, d = bb + lam, det = a * d - rb * rb;
      var k1 = det > 0 ? (gr * d - gb * rb) / det : 0, k2 = det > 0 ? (gb * a - gr * rb) / det : 0;
      var hm = 0, h = new Float64Array(l);
      for (i = 0; i < l; i++) { h[i] = g[i] - k1 * r[i] - k2 * b[i]; hm += h[i]; }
      hm /= l;
      for (i = 0; i < l; i++) H[m + i] += -(h[i] - hm);   // minus: more blood = darker = pulse up
    }
    var bvp = detrend(H, 100);
    return filtfilt(butterBand(0.75, 3, fs), bvp);
  }

  // ---- heart rate from a pulse wave: detrend 100, band-pass 0.6-3.3 Hz, power spectrum
  // on a fine bpm grid (equivalent to a zero-padded periodogram), plus
  // Nature-inspired confidence: 1 - normalised entropy of the 1-bpm power distribution.
  // power at bpm grid points (Goertzel recurrence: same values as the direct DFT sum, no trig in the loop)
  function powerAt(x, bpms, fs) {
    var n = x.length, out = new Float64Array(bpms.length);
    for (var k = 0; k < bpms.length; k++) {
      var c = 2 * Math.cos(2 * Math.PI * bpms[k] / 60 / fs), s1 = 0, s2 = 0;
      for (var i = 0; i < n; i++) { var s0 = x[i] + c * s1 - s2; s2 = s1; s1 = s0; }
      out[k] = s1 * s1 + s2 * s2 - c * s1 * s2;
    }
    return out;
  }
  // opts.tilt = k: spectrum weighted by (f / 1 Hz)^k before the peak is picked. Low-frequency
  //   interference (breathing, slow motion, light drift) has far more power than the pulse and
  //   pulls a short window's peak down; on UBFC almost every wrong reading was too low. k = 2
  //   equals taking the peak of the signal's derivative.
  // opts.harm: harmonic check -- a candidate's score adds half the power around its 2nd
  //   harmonic (a real pulse wave is not a pure sine); needs the wider 0.6-6.6 Hz band, so the
  //   input should be unfiltered (pos(..., {noFilter: true})). Only candidates with at least 30 %
  //   of the top power get the bonus, so a weak sub-harmonic cannot win through it.
  function heartRate(bvp, fs, opts) {
    opts = opts || {};
    var lo = opts.minBpm || 40, hi = opts.maxBpm || 180, step = opts.stepBpm || 0.25;
    var x = filtfilt(butterBand(0.6, opts.harm ? 6.6 : 3.3, fs), detrend(bvp, 100));
    var bins = Math.round((hi - lo) / step) + 1, bpm = new Float64Array(bins), k;
    for (k = 0; k < bins; k++) bpm[k] = lo + k * step;
    var pw = powerAt(x, bpm, fs), best = 0;
    if (opts.harm) {
      var mx = 0; for (k = 0; k < bins; k++) if (pw[k] > mx) mx = pw[k];
      var h2 = new Float64Array(bins), grid = [];
      for (var g = 2 * lo - 2; g <= 2 * hi + 2 + 1e-9; g += 0.5) grid.push(g);
      var P2 = powerAt(x, grid, fs), sc = new Float64Array(bins);
      for (k = 0; k < bins; k++) {
        var c0 = Math.round((2 * bpm[k] - 2 - grid[0]) / 0.5), m2 = 0;
        for (var q = c0; q <= c0 + 8 && q < P2.length; q++) if (q >= 0 && P2[q] > m2) m2 = P2[q];
        sc[k] = pw[k] + (pw[k] >= 0.3 * mx ? 0.5 * m2 : 0);
      }
      pw = sc;
    }
    if (opts.tilt) for (k = 0; k < bins; k++) pw[k] *= Math.pow(bpm[k] / 60, opts.tilt);
    for (k = 1; k < bins; k++) if (pw[k] > pw[best]) best = k;
    var q2 = quality(bpm, pw, step, best);
    return { bpm: bpm[best], confidence: q2.confidence, snrDb: q2.snrDb, spectrumBpm: bpm, spectrum: pw, filtered: x };
  }

  // ---- fusion of several regions' spectra (e.g. forehead, left cheek, right cheek).
  // Each region's power spectrum is normalised to sum 1 and weighted by its own signal quality
  // (linear SNR minus a small floor), so a region with glare, hair or shadow, or one that moves
  // with speech, counts less. Same bpm grid for all inputs (all from heartRate()).
  function fuse(hs) {
    hs = hs.filter(function (h) { return h && h.spectrum; });
    if (hs.length === 1) return hs[0];
    var n = hs[0].spectrum.length, S = new Float64Array(n), ws = [], wsum = 0, i, j;
    for (j = 0; j < hs.length; j++) {
      var w = Math.pow(10, hs[j].snrDb / 10) - 0.05;
      w = isFinite(w) ? Math.max(0, w) : 0; ws.push(w); wsum += w;
    }
    if (wsum <= 0) ws = hs.map(function () { return 1; });
    var top = 0;
    for (j = 0; j < hs.length; j++) {
      var P = hs[j].spectrum, tot = 0;
      for (i = 0; i < n; i++) tot += P[i];
      if (tot > 0) for (i = 0; i < n; i++) S[i] += ws[j] * P[i] / tot;
      if (ws[j] > ws[top]) top = j;
    }
    var bpm = hs[0].spectrumBpm, best = 0;
    for (i = 1; i < n; i++) if (S[i] > S[best]) best = i;
    var q = quality(bpm, S, bpm[1] - bpm[0], best);
    return { bpm: bpm[best], confidence: q.confidence, snrDb: q.snrDb, spectrumBpm: bpm, spectrum: S,
             filtered: hs[top].filtered, weights: ws };
  }
  function quality(bpm, pw, step, best) {
    var bins = pw.length, per = Math.round(1 / step), nb = Math.floor((bins - 1) / per), P = new Float64Array(nb), tot = 0;
    for (var b = 0; b < nb; b++) { for (var q = 0; q < per; q++) P[b] += pw[b * per + q]; tot += P[b]; }
    var Hn = 0;
    for (var c = 0; c < nb; c++) { var p = P[c] / tot; if (p > 0) Hn -= p * Math.log(p); }
    var f0 = bpm[best], sig = 0, rest = 0;
    for (var d = 0; d < bins; d++) {
      if (Math.abs(bpm[d] - f0) <= 6 || Math.abs(bpm[d] - 2 * f0) <= 6) sig += pw[d]; else rest += pw[d];
    }
    return { confidence: 1 - Hn / Math.log(nb), snrDb: 10 * Math.log10(sig / rest) };
  }

  // ---- 1-D Kalman filter for the displayed HR (random walk; noisier when confidence is low)
  function Kalman(q, r0) { this.q = q; this.r0 = r0; this.x = null; this.p = 0; }
  Kalman.prototype.update = function (z, conf) {
    var r = this.r0 / Math.max(conf, 0.05) / Math.max(conf, 0.05);
    if (this.x === null) { this.x = z; this.p = r; return this.x; }
    this.p += this.q;
    var k = this.p / (this.p + r);
    this.x += k * (z - this.x); this.p *= (1 - k);
    return this.x;
  };
  Kalman.prototype.reset = function () { this.x = null; this.p = 0; };


  // ---- stabiliser for the displayed reading (stops sudden jumps such as 70 -> 110 -> 70).
  // Tuned on odd-numbered UBFC subjects, checked on even ones (see stab/eval.js):
  //  1. long window: when the 10 s estimate agrees with the 5 s one (within 6 bpm), use the
  //     10 s spectrum (finer frequency resolution); otherwise stay on 5 s so changes show fast;
  //  2. ambiguous spectrum: if several peaks reach 60% of the highest, take the one nearest the
  //     median of the last 5 raw top peaks (the recent evidence, not what was displayed);
  //  3. display the median of the last 5 picked values, so a single odd reading never shows
  //     (a jump-confirmation rule was also tried; with the median it changed nothing);
  //  4. provisional ("预估") until the reading has settled: at least 5 readings since the start
  //     and the last 3 picked values within 10 bpm of each other. Once settled it stays settled
  //     until reset (new face, cut, face lost).
  function median(a) { var s = a.slice().sort(function (x, y) { return x - y; }), n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2; }
  function Stabilizer(o) {
    o = o || {};
    this.agree = o.agree || 6; this.amb = o.amb || 0.6; this.refN = o.refN || 5;
    this.medN = o.medN || 5; this.settleN = o.settleN || 3; this.settleBpm = o.settleBpm || 10; this.reset();
  }
  Stabilizer.prototype.reset = function () { this.top = []; this.picked = []; this.shown = []; this.settled = false; };
  Stabilizer.prototype.update = function (hShort, hLong) {
    var h = (hLong && Math.abs(hLong.bpm - hShort.bpm) <= this.agree) ? hLong : hShort;
    var f = h.bpm, P = h.spectrum, m = 0, k;
    this.top.push(h.bpm);
    if (this.top.length >= 2) {
      for (k = 0; k < P.length; k++) if (P[k] > m) m = P[k];
      var ref = median(this.top.slice(-this.refN)), best = null;
      for (k = 1; k < P.length - 1; k++) {
        if (P[k] >= P[k - 1] && P[k] > P[k + 1] && P[k] >= this.amb * m) {
          var b = h.spectrumBpm[k];
          if (best === null || Math.abs(b - ref) < Math.abs(best - ref)) best = b;
        }
      }
      if (best !== null) f = best;
    }
    this.picked.push(f);
    var out = median(this.picked.slice(-this.medN));
    if (!this.settled && this.picked.length >= this.medN) {
      var last = this.picked.slice(-this.settleN);
      this.settled = Math.max.apply(null, last) - Math.min.apply(null, last) <= this.settleBpm;
    }
    this.shown.push(out);
    if (this.top.length > 30) { this.top.shift(); this.picked.shift(); this.shown.shift(); }
    return { bpm: out, picked: f, usedLong: h === hLong, provisional: !this.settled };
  };

  // ---- remove what reference signals (e.g. head position and size) explain from a signal:
  // least squares h - sum k_i ref_i, references centred; small ridge for stability
  function regressOut(h, refs) {
    var n = h.length, m = refs.length, i, j, t;
    if (!m) return h;
    var R = refs.map(function (r) { var mu = 0; for (t = 0; t < n; t++) mu += r[t]; mu /= n; var c = new Float64Array(n); for (t = 0; t < n; t++) c[t] = r[t] - mu; return c; });
    var A = [], b = [], tr = 0;
    for (i = 0; i < m; i++) { A.push([]); var bi = 0; for (t = 0; t < n; t++) bi += R[i][t] * h[t]; b.push(bi);
      for (j = 0; j < m; j++) { var a = 0; for (t = 0; t < n; t++) a += R[i][t] * R[j][t]; A[i].push(a); } tr += A[i][i]; }
    if (!(tr > 0)) return h;
    for (i = 0; i < m; i++) A[i][i] += 1e-3 * tr / m;
    for (i = 0; i < m; i++) {           // Gaussian elimination
      var piv = A[i][i]; if (!(Math.abs(piv) > 1e-30)) return h;
      for (j = i + 1; j < m; j++) { var f = A[j][i] / piv; for (var c2 = i; c2 < m; c2++) A[j][c2] -= f * A[i][c2]; b[j] -= f * b[i]; }
    }
    var k = new Array(m);
    for (i = m - 1; i >= 0; i--) { var v = b[i]; for (j = i + 1; j < m; j++) v -= A[i][j] * k[j]; k[i] = v / A[i][i]; }
    var out = new Float64Array(n);
    for (t = 0; t < n; t++) { var y = h[t]; for (i = 0; i < m; i++) y -= k[i] * R[i][t]; out[t] = y; }
    return out;
  }

  // ---- re-pick a heart-rate result after weighting its spectrum (w: one factor per bin)
  function reweight(h, w) {
    var S = new Float64Array(h.spectrum.length), best = 0, i;
    for (i = 0; i < S.length; i++) S[i] = h.spectrum[i] * w[i];
    for (i = 1; i < S.length; i++) if (S[i] > S[best]) best = i;
    var bpm = h.spectrumBpm, q = quality(bpm, S, bpm[1] - bpm[0], best);
    return { bpm: bpm[best], confidence: q.confidence, snrDb: q.snrDb, spectrumBpm: bpm, spectrum: S, filtered: h.filtered, weights: h.weights };
  }

  var api = { resample: resample, detrend: detrend, butterBand: butterBand, filtfilt: filtfilt,
              pos: pos, green: green, reg: reg, regressOut: regressOut, reweight: reweight, heartRate: heartRate, Kalman: Kalman, Stabilizer: Stabilizer, median: median, fuse: fuse };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.RPPG = api;
})(typeof self !== "undefined" ? self : this);
