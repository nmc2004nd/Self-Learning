/* Cây quyết định CART cho các trang mô phỏng (ml-tree.html, ml-ensemble.html).
 * window.MLTree.build(X, y, opt) trả về nút gốc; MLTree.predict(node, x) trả về giá trị ở lá.
 * opt: { task: 'cls' | 'reg', maxDepth, minLeaf, criterion: 'gini' | 'entropy', maxFeatures, rng, idx }
 *  - cls: y ∈ {0, 1}, lá lưu tỉ lệ lớp 1 (xác suất); reg: lá lưu trung bình y.
 *  - maxFeatures: số đặc trưng xét ở mỗi lần chia (random forest); mặc định là tất cả.
 *  - idx: danh sách chỉ số mẫu dùng để học (cho phép lặp lại, như mẫu bootstrap).
 */
(function () {
  'use strict';
  function imp(c1, n, crit) {
    if (n === 0) return 0;
    var p = c1 / n;
    if (crit === 'entropy') return (p > 0 ? -p * Math.log2(p) : 0) + (p < 1 ? -(1 - p) * Math.log2(1 - p) : 0);
    return 2 * p * (1 - p);
  }
  function leafValue(y, idx) { var s = 0; idx.forEach(function (i) { s += y[i]; }); return idx.length ? s / idx.length : 0; }
  function nodeImp(y, idx, o) {
    if (o.task === 'reg') { var m = leafValue(y, idx), s = 0; idx.forEach(function (i) { s += (y[i] - m) * (y[i] - m); }); return idx.length ? s / idx.length : 0; }
    var c = 0; idx.forEach(function (i) { c += y[i]; }); return imp(c, idx.length, o.criterion);
  }

  function bestSplit(X, y, idx, o) {
    var d = X[0].length, n = idx.length, feats = [], f, best = null;
    for (f = 0; f < d; f++) feats.push(f);
    if (o.maxFeatures && o.maxFeatures < d) {               // chọn ngẫu nhiên một tập con đặc trưng
      for (var k = d - 1; k > 0; k--) { var j = Math.floor(o.rng() * (k + 1)), t = feats[k]; feats[k] = feats[j]; feats[j] = t; }
      feats = feats.slice(0, o.maxFeatures);
    }
    var parent = nodeImp(y, idx, o);
    feats.forEach(function (f) {
      var s = idx.slice().sort(function (a, b) { return X[a][f] - X[b][f]; });
      var totS = 0, totQ = 0; s.forEach(function (i) { totS += y[i]; totQ += y[i] * y[i]; });
      var lS = 0, lQ = 0;
      for (var k = 0; k < n - 1; k++) {
        var i = s[k]; lS += y[i]; lQ += y[i] * y[i];
        var nl = k + 1, nr = n - nl;
        if (X[s[k]][f] === X[s[k + 1]][f] || nl < o.minLeaf || nr < o.minLeaf) continue;
        var child;
        if (o.task === 'reg') {
          var rS = totS - lS, rQ = totQ - lQ;
          child = (lQ - lS * lS / nl) + (rQ - rS * rS / nr);   // tổng bình phương sai lệch hai bên
          child /= n;
        } else child = (nl * imp(lS, nl, o.criterion) + nr * imp(totS - lS, nr, o.criterion)) / n;
        var gain = parent - child;
        if (gain > 1e-12 && (!best || gain > best.gain)) best = { f: f, t: (X[s[k]][f] + X[s[k + 1]][f]) / 2, gain: gain };
      }
    });
    return best;
  }

  function grow(X, y, idx, depth, o) {
    var node = { n: idx.length, v: leafValue(y, idx), depth: depth, imp: nodeImp(y, idx, o) };
    if (o.task === 'cls') { node.c1 = 0; idx.forEach(function (i) { node.c1 += y[i]; }); }
    if (depth >= o.maxDepth || idx.length < 2 * o.minLeaf || node.imp < 1e-12) return node;
    var b = bestSplit(X, y, idx, o);
    if (!b) return node;
    node.f = b.f; node.t = b.t; node.gain = b.gain;
    var L = [], R = [];
    idx.forEach(function (i) { (X[i][b.f] <= b.t ? L : R).push(i); });
    node.left = grow(X, y, L, depth + 1, o);
    node.right = grow(X, y, R, depth + 1, o);
    return node;
  }

  function build(X, y, opt) {
    var o = { task: 'cls', maxDepth: 6, minLeaf: 1, criterion: 'gini', maxFeatures: 0, rng: Math.random };
    for (var k in opt) o[k] = opt[k];
    var idx = o.idx || X.map(function (_, i) { return i; });
    return grow(X, y, idx, 0, o);
  }
  function predict(node, x) {
    while (node.left) node = x[node.f] <= node.t ? node.left : node.right;
    return node.v;
  }
  function leaves(node) { return node.left ? leaves(node.left) + leaves(node.right) : 1; }
  function depth(node) { return node.left ? 1 + Math.max(depth(node.left), depth(node.right)) : 0; }
  // các đoạn cắt của cây trong hộp [x0,x1]×[y0,y1] (chỉ cho dữ liệu 2 chiều)
  function cuts(node, box, out) {
    out = out || [];
    if (!node.left) return out;
    if (node.f === 0) {
      out.push({ x0: node.t, x1: node.t, y0: box[2], y1: box[3], depth: node.depth });
      cuts(node.left, [box[0], node.t, box[2], box[3]], out); cuts(node.right, [node.t, box[1], box[2], box[3]], out);
    } else {
      out.push({ x0: box[0], x1: box[1], y0: node.t, y1: node.t, depth: node.depth });
      cuts(node.left, [box[0], box[1], box[2], node.t], out); cuts(node.right, [box[0], box[1], node.t, box[3]], out);
    }
    return out;
  }
  window.MLTree = { build: build, predict: predict, leaves: leaves, depth: depth, cuts: cuts };
})();
