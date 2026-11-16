/* Tiện ích dùng chung cho các trang mô phỏng của khoá Học máy (docs/viz/ml-*.html).
 * window.MLV cung cấp: số ngẫu nhiên có hạt giống, giải hệ tuyến tính, khung đồ thị SVG
 * (trục, lưới, nhãn, đổi toạ độ), tô vùng quyết định lên canvas, biểu đồ đường nhỏ và SVM (SMO).
 * Dùng cùng viz.js và viz.css.
 */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  // Màu lớp: đọc được trên nền sáng lẫn tối
  var CLASS = ['#2f80f5', '#e8487f', '#0ea5a4', '#f08a24', '#7c3aed', '#16a34a'];

  function rng(seed) {
    var st = (seed >>> 0) || 1;
    return function () {
      st = (st + 0x6D2B79F5) >>> 0;
      var t = st;
      t = Math.imul(t ^ t >>> 15, t | 1);
      t ^= t + Math.imul(t ^ t >>> 7, t | 61);
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function randn(r) {
    var u = 1 - r(), v = r();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  // Khử Gauss có chọn trụ; trả null nếu suy biến
  function solve(A, b) {
    var n = b.length, i, j, k;
    var M = A.map(function (r, i) { return r.slice().concat([b[i]]); });
    for (i = 0; i < n; i++) {
      var p = i;
      for (j = i + 1; j < n; j++) if (Math.abs(M[j][i]) > Math.abs(M[p][i])) p = j;
      var t = M[i]; M[i] = M[p]; M[p] = t;
      if (Math.abs(M[i][i]) < 1e-14) return null;
      for (j = i + 1; j < n; j++) {
        var f = M[j][i] / M[i][i];
        for (k = i; k <= n; k++) M[j][k] -= f * M[i][k];
      }
    }
    var x = new Array(n);
    for (i = n - 1; i >= 0; i--) {
      var s = M[i][n];
      for (j = i + 1; j < n; j++) s -= M[i][j] * x[j];
      x[i] = s / M[i][i];
    }
    return x;
  }

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) if (attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function niceTicks(a, b, n) {
    var span = b - a, step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var v = Math.ceil(a / step) * step; v <= b + 1e-9 * span; v += step) out.push(Math.abs(v) < 1e-12 ? 0 : v);
    return out;
  }
  function fmtTick(v) {
    var s = Math.abs(v) >= 1e5 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0) : String(+v.toFixed(3));
    return s.replace('-', '−');
  }

  /* Khung đồ thị trên một <svg viewBox="0 0 W H">.
   * o: { W, H, x: [x0, x1], y: [y0, y1], m: {l, r, t, b}, xlabel, ylabel, logY, ticks: true, grid: true }
   * Trả về đối tượng có sx/sy (dữ liệu -> điểm ảnh), ix/iy (ngược lại), clear(), axes(), layer().
   */
  function Plot(svg, o) {
    var W = o.W || 600, H = o.H || 400, m = o.m || { l: 48, r: 14, t: 12, b: 40 };
    var id = 'clip' + Math.random().toString(36).slice(2, 8);
    var P = {
      svg: svg, W: W, H: H, m: m, x: o.x.slice(), y: o.y.slice(), logY: !!o.logY,
      pw: W - m.l - m.r, ph: H - m.t - m.b, clip: 'url(#' + id + ')',
      sx: function (x) { return m.l + (x - P.x[0]) / (P.x[1] - P.x[0]) * P.pw; },
      sy: function (y) {
        if (P.logY) { y = Math.log10(Math.max(y, 1e-300)); return H - m.b - (y - Math.log10(P.y[0])) / (Math.log10(P.y[1]) - Math.log10(P.y[0])) * P.ph; }
        return H - m.b - (y - P.y[0]) / (P.y[1] - P.y[0]) * P.ph;
      },
      ix: function (px) { return P.x[0] + (px - m.l) / P.pw * (P.x[1] - P.x[0]); },
      iy: function (py) { return P.y[0] + (H - m.b - py) / P.ph * (P.y[1] - P.y[0]); },
      el: function (tag, attrs, parent) { return el(tag, attrs, parent || svg); },
      clear: function () { while (svg.firstChild) svg.removeChild(svg.firstChild); },
      layer: function (clipped) { return el('g', clipped ? { 'clip-path': P.clip } : {}, svg); },
      inside: function (px, py) { return px >= m.l && px <= W - m.r && py >= m.t && py <= H - m.b; },
      // toạ độ dữ liệu của một sự kiện chuột
      at: function (e) {
        var pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
        var p = pt.matrixTransform(svg.getScreenCTM().inverse());
        return { px: p.x, py: p.y, x: P.ix(p.x), y: P.iy(p.y) };
      },
      axes: function (opt) {
        opt = opt || {};
        var defs = el('defs', {}, svg), cp = el('clipPath', { id: id }, defs);
        el('rect', { x: m.l, y: m.t, width: P.pw, height: P.ph }, cp);
        if (!opt.noFrame) el('rect', { x: m.l, y: m.t, width: P.pw, height: P.ph, class: 'pl-frame' }, svg);
        var xt = opt.xticks || niceTicks(P.x[0], P.x[1], o.nx || 6);
        xt.forEach(function (v) {
          var px = P.sx(v);
          if (o.grid !== false) el('line', { x1: px, x2: px, y1: m.t, y2: H - m.b, class: 'pl-grid' }, svg);
          el('text', { x: px, y: H - m.b + 15, 'text-anchor': 'middle', class: 'pl-tick' }, svg).textContent = opt.xfmt ? opt.xfmt(v) : fmtTick(v);
        });
        var yt;
        if (P.logY) {
          yt = [];
          var e0 = Math.ceil(Math.log10(P.y[0])), e1 = Math.floor(Math.log10(P.y[1])), es = Math.max(1, Math.ceil((e1 - e0) / 7));
          for (var e = e1; e >= e0; e -= es) yt.push(Math.pow(10, e));     // nhiều bậc thì chỉ ghi cách quãng
        } else yt = opt.yticks || niceTicks(P.y[0], P.y[1], o.ny || 5);
        yt.forEach(function (v) {
          var py = P.sy(v);
          if (o.grid !== false) el('line', { x1: m.l, x2: W - m.r, y1: py, y2: py, class: 'pl-grid' }, svg);
          el('text', { x: m.l - 6, y: py + 4, 'text-anchor': 'end', class: 'pl-tick' }, svg).textContent = fmtTick(v);
        });
        if (o.xlabel) el('text', { x: m.l + P.pw / 2, y: H - 6, 'text-anchor': 'middle', class: 'pl-lab' }, svg).textContent = o.xlabel;
        if (o.ylabel) {
          var cy = m.t + P.ph / 2;
          el('text', { x: 13, y: cy, 'text-anchor': 'middle', class: 'pl-lab', transform: 'rotate(-90 13 ' + cy + ')' }, svg).textContent = o.ylabel;
        }
      },
      // đường cong y = f(x) trên toàn trục ngang
      curve: function (f, attrs, parent) {
        var d = '', n = Math.max(60, Math.round(P.pw / 3));
        for (var i = 0; i <= n; i++) {
          var x = P.x[0] + (P.x[1] - P.x[0]) * i / n, y = f(x);
          if (!isFinite(y)) continue;
          var py = Math.max(-4000, Math.min(4000, P.sy(y)));
          d += (d ? 'L' : 'M') + P.sx(x).toFixed(1) + ' ' + py.toFixed(1);
        }
        var a = { d: d, fill: 'none' }; for (var k in attrs) a[k] = attrs[k];
        a['clip-path'] = P.clip;
        return el('path', a, parent || svg);
      },
      // đường gấp khúc qua các điểm [[x, y], ...]
      poly: function (pts, attrs, parent) {
        var d = '';
        pts.forEach(function (p) { if (isFinite(p[1])) d += (d ? 'L' : 'M') + P.sx(p[0]).toFixed(1) + ' ' + P.sy(p[1]).toFixed(1); });
        var a = { d: d, fill: 'none' }; for (var k in attrs) a[k] = attrs[k];
        if (!a['clip-path']) a['clip-path'] = P.clip;
        return el('path', a, parent || svg);
      }
    };
    return P;
  }

  /* Tô vùng lên một <canvas> nằm dưới svg (cùng kích thước khung vẽ).
   * colorAt(x, y) trả về [r, g, b, a] (0–255). step: cỡ ô (điểm ảnh của canvas).
   */
  function paint(canvas, P, colorAt, step) {
    step = step || 4;
    var ctx = canvas.getContext('2d'), W = canvas.width, H = canvas.height;
    var sxr = W / P.W, syr = H / P.H;
    ctx.clearRect(0, 0, W, H);
    var img = ctx.createImageData(W, H), data = img.data;
    var x0 = Math.floor(P.m.l * sxr), x1 = Math.ceil((P.W - P.m.r) * sxr), y0 = Math.floor(P.m.t * syr), y1 = Math.ceil((P.H - P.m.b) * syr);
    for (var py = y0; py < y1; py += step) {
      for (var px = x0; px < x1; px += step) {
        var c = colorAt(P.ix((px + step / 2) / sxr), P.iy((py + step / 2) / syr));
        for (var dy = 0; dy < step && py + dy < y1; dy++) for (var dx = 0; dx < step && px + dx < x1; dx++) {
          var k = ((py + dy) * W + px + dx) * 4;
          data[k] = c[0]; data[k + 1] = c[1]; data[k + 2] = c[2]; data[k + 3] = c[3];
        }
      }
    }
    ctx.putImageData(img, 0, 0);
  }
  function hexRgb(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }

  /* Biểu đồ đường nhỏ (vd. mất mát theo vòng lặp). series: [{ pts: [[x, y]], color, label, dash }] */
  function chart(svg, series, o) {
    o = o || {};
    var all = []; series.forEach(function (s) { s.pts.forEach(function (p) { if (isFinite(p[1])) all.push(p); }); });
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var vb = svg.viewBox.baseVal, W = vb.width || 600, H = vb.height || 200;
    if (!all.length) return null;
    var xs = all.map(function (p) { return p[0]; }), ys = all.map(function (p) { return p[1]; });
    var xr = o.x || [Math.min.apply(null, xs), Math.max.apply(null, xs)];
    if (xr[0] === xr[1]) xr = [xr[0] - 1, xr[1] + 1];
    var ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
    var yr = o.y || (o.logY ? [Math.pow(10, Math.floor(Math.log10(Math.max(ymin, 1e-12)))), Math.pow(10, Math.ceil(Math.log10(Math.max(ymax, 1e-11))))]
      : [Math.min(0, ymin), ymax === ymin ? ymax + 1 : ymax + (ymax - ymin) * 0.08]);
    var P = Plot(svg, { W: W, H: H, x: xr, y: yr, logY: o.logY, xlabel: o.xlabel, ylabel: o.ylabel, m: o.m || { l: 52, r: 12, t: 10, b: 36 }, nx: 6, ny: 4 });
    P.axes();
    series.forEach(function (s) {
      P.poly(s.pts, { stroke: s.color, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash || null });
      if (s.marker) s.pts.forEach(function (p) { if (isFinite(p[1])) el('circle', { cx: P.sx(p[0]), cy: P.sy(p[1]), r: 2.8, fill: s.color }, svg); });
    });
    if (o.vline !== undefined) el('line', { x1: P.sx(o.vline), x2: P.sx(o.vline), y1: P.m.t, y2: P.H - P.m.b, class: 'pl-vline' }, svg);
    return P;
  }


  /* SVM (bài toán đối ngẫu của chương 26–28) bằng SMO, chọn cặp vi phạm nhiều nhất như libsvm.
   * X: [[x1, x2, ...]], y: [+1/-1], o: { C, kernel(a, b), tol (1e-3), maxIter (50000) }.
   * Trả về { alpha, b, f(x), iters, converged, sv: chỉ số có alpha > 0 }; f(x) = sum alpha_n y_n K(x_n, x) + b.
   */
  function svm(X, y, o) {
    var n = X.length, C = o.C, K = o.kernel || kernels.linear(), tol = o.tol || 1e-3, maxIter = o.maxIter || 50000, TAU = 1e-12;
    var Km = [], i, j, t;
    for (i = 0; i < n; i++) { Km.push(new Float64Array(n)); for (j = 0; j <= i; j++) { Km[i][j] = K(X[i], X[j]); if (j < i) Km[j][i] = Km[i][j]; } }
    var a = new Float64Array(n), G = new Float64Array(n).fill(-1), it = 0, converged = false;
    function up(t) { return (y[t] > 0 && a[t] < C) || (y[t] < 0 && a[t] > 0); }
    function low(t) { return (y[t] < 0 && a[t] < C) || (y[t] > 0 && a[t] > 0); }
    while (it < maxIter) {
      var Gmax = -Infinity, Gmin = Infinity; i = -1; j = -1;
      for (t = 0; t < n; t++) {
        var v = -y[t] * G[t];
        if (up(t) && v >= Gmax) { Gmax = v; i = t; }
        if (low(t) && v <= Gmin) { Gmin = v; j = t; }
      }
      if (i < 0 || j < 0 || Gmax - Gmin < tol) { converged = true; break; }
      it++;
      var ai = a[i], aj = a[j], Qij = y[i] * y[j] * Km[i][j], q, d;
      if (y[i] !== y[j]) {
        q = Km[i][i] + Km[j][j] + 2 * Qij; if (q <= 0) q = TAU;
        d = (-G[i] - G[j]) / q; var diff = a[i] - a[j];
        a[i] += d; a[j] += d;
        if (diff > 0) { if (a[j] < 0) { a[j] = 0; a[i] = diff; } } else if (a[i] < 0) { a[i] = 0; a[j] = -diff; }
        if (diff > 0) { if (a[i] > C) { a[i] = C; a[j] = C - diff; } } else if (a[j] > C) { a[j] = C; a[i] = C + diff; }
      } else {
        q = Km[i][i] + Km[j][j] - 2 * Qij; if (q <= 0) q = TAU;
        d = (G[i] - G[j]) / q; var sum = a[i] + a[j];
        a[i] -= d; a[j] += d;
        if (sum > C) { if (a[i] > C) { a[i] = C; a[j] = sum - C; } } else if (a[j] < 0) { a[j] = 0; a[i] = sum; }
        if (sum > C) { if (a[j] > C) { a[j] = C; a[i] = sum - C; } } else if (a[i] < 0) { a[i] = 0; a[j] = sum; }
      }
      var dI = a[i] - ai, dJ = a[j] - aj;
      for (t = 0; t < n; t++) G[t] += y[t] * (y[i] * Km[i][t] * dI + y[j] * Km[j][t] * dJ);
    }
    // b như libsvm: trung bình trên các điểm 0 < alpha < C, nếu không có thì lấy giữa hai chặn
    var ub = Infinity, lb = -Infinity, nf = 0, sf = 0;
    for (t = 0; t < n; t++) {
      var yG = y[t] * G[t];
      if (a[t] >= C) { if (y[t] < 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else if (a[t] <= 0) { if (y[t] > 0) ub = Math.min(ub, yG); else lb = Math.max(lb, yG); }
      else { nf++; sf += yG; }
    }
    var rho = nf ? sf / nf : (ub + lb) / 2, sv = [];
    for (t = 0; t < n; t++) if (a[t] > 1e-8) sv.push(t);
    return {
      alpha: a, b: -rho, iters: it, converged: converged, sv: sv,
      f: function (x) { var s = -rho; for (var k = 0; k < sv.length; k++) s += a[sv[k]] * y[sv[k]] * K(X[sv[k]], x); return s; }
    };
  }
  // Hàm hạt nhân, cùng định nghĩa với sklearn (Bảng 28.2)
  var kernels = {
    linear: function () { return function (u, v) { var s = 0; for (var k = 0; k < u.length; k++) s += u[k] * v[k]; return s; }; },
    poly: function (g, r, d) { return function (u, v) { var s = 0; for (var k = 0; k < u.length; k++) s += u[k] * v[k]; return Math.pow(g * s + r, d); }; },
    rbf: function (g) { return function (u, v) { var s = 0; for (var k = 0; k < u.length; k++) { var e = u[k] - v[k]; s += e * e; } return Math.exp(-g * s); }; },
    sigmoid: function (g, r) { return function (u, v) { var s = 0; for (var k = 0; k < u.length; k++) s += u[k] * v[k]; return Math.tanh(g * s + r); }; }
  };


  /* Đường mức f(x, y) = level trên khung P (marching squares), trả về chuỗi path SVG.
   * F: mảng nx*ny giá trị f tại lưới đều trên P.x × P.y (hàng theo y tăng dần).
   */
  function contour(P, F, nx, ny, level) {
    var d = '', x0 = P.x[0], y0 = P.y[0], hx = (P.x[1] - x0) / (nx - 1), hy = (P.y[1] - y0) / (ny - 1);
    function pt(x, y) { return P.sx(x).toFixed(1) + ' ' + P.sy(y).toFixed(1); }
    for (var j = 0; j < ny - 1; j++) for (var i = 0; i < nx - 1; i++) {
      var a = F[j * nx + i] - level, b = F[j * nx + i + 1] - level, c = F[(j + 1) * nx + i + 1] - level, e = F[(j + 1) * nx + i] - level;
      var xa = x0 + i * hx, ya = y0 + j * hy, cr = [];
      if ((a > 0) !== (b > 0)) cr.push([xa + hx * a / (a - b), ya]);
      if ((b > 0) !== (c > 0)) cr.push([xa + hx, ya + hy * b / (b - c)]);
      if ((c > 0) !== (e > 0)) cr.push([xa + hx * e / (e - c), ya + hy]);
      if ((e > 0) !== (a > 0)) cr.push([xa, ya + hy * a / (a - e)]);
      if (cr.length >= 2) d += 'M' + pt(cr[0][0], cr[0][1]) + 'L' + pt(cr[1][0], cr[1][1]);
      if (cr.length === 4) d += 'M' + pt(cr[2][0], cr[2][1]) + 'L' + pt(cr[3][0], cr[3][1]);
    }
    return d;
  }

  window.MLV = { CLASS: CLASS, rng: rng, randn: randn, solve: solve, el: el, Plot: Plot, paint: paint, hexRgb: hexRgb, chart: chart, niceTicks: niceTicks, fmtTick: fmtTick, svm: svm, kernels: kernels, contour: contour };
})();
