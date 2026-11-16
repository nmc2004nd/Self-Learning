// Điều hướng cho Hope for future (tự học AI): thanh trên cùng, bộ chuyển khoá học, mục lục thu gọn được,
// lối tắt trong chương, mục lục của trang và nút bài trước/bài sau. Đăng ký như một plugin của docsify.
(function () {
  var ICON = {
    menu: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 6h16M4 12h16M4 18h16"/></svg>',
    moon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
    sun: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
    github: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2a10 10 0 0 0-3.2 19.5c.5.1.7-.2.7-.5v-1.7c-2.8.6-3.4-1.3-3.4-1.3-.5-1.2-1.1-1.5-1.1-1.5-.9-.6.1-.6.1-.6 1 .1 1.5 1 1.5 1 .9 1.5 2.3 1.1 2.9.8.1-.6.3-1.1.6-1.3-2.2-.3-4.6-1.1-4.6-4.9 0-1.1.4-2 1-2.7-.1-.3-.4-1.3.1-2.6 0 0 .8-.3 2.8 1a9.600 9.600 0 0 1 5 0c1.900-1.300 2.800-1 2.800-1 .5 1.300.2 2.300.1 2.600.6.7 1 1.600 1 2.700 0 3.800-2.300 4.600-4.600 4.900.4.300.7.900.7 1.900V21c0 .3.2.6.7.5A10 10 0 0 0 12 2z"/></svg>',
    arrow: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
    logo: '<span class="brand-logo logo-cycle"><img class="is-on" src="img/brand/logo.png" alt=""></span>',
    users: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/></svg>',
    code: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m18 16 4-4-4-4M6 8l-4 4 4 4M14.5 4l-5 16"/></svg>',
    flag: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 22V4a1 1 0 0 1 .4-.8A6 6 0 0 1 8 2c3 0 5 2 7.3 2a6 6 0 0 0 3.7-1.2 1 1 0 0 1 1.6.8v10.8a1 1 0 0 1-.4.8A6 6 0 0 1 16.7 16c-2.3 0-4.3-2-7.3-2a6 6 0 0 0-5 2"/></svg>',
    layout: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12h18M3 6h18M3 18h12"/></svg>',
    note: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/><path d="M14 2v4a2 2 0 0 0 2 2h4M10 9H8M16 13H8M16 17H8"/></svg>',
    chat: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z"/><path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1"/></svg>',
    ext: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M15 3h6v6M10 14 21 3M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></svg>',
    link: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/></svg>',
    lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="18" height="11" x="3" y="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    cap: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 3 1 9l11 6 9-4.9V17h2V9z"/><path d="M5 13.2v4L12 21l7-3.8v-4L12 17z" opacity=".7"/></svg>',
    chevron: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6"/></svg>',
    down: '<svg class="chev" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>',
    home: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M11.3 3.3a1 1 0 0 1 1.4 0l8 7.5A1 1 0 0 1 20 12.5h-1V20a1 1 0 0 1-1 1h-4v-5.5a1 1 0 0 0-1-1h-2a1 1 0 0 0-1 1V21H6a1 1 0 0 1-1-1v-7.5H4a1 1 0 0 1-.7-1.7z"/></svg>',
    book: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 7v14"/><path d="M3 18a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h5a4 4 0 0 1 4 4 4 4 0 0 1 4-4h5a1 1 0 0 1 1 1v13a1 1 0 0 1-1 1h-6a3 3 0 0 0-3 3 3 3 0 0 0-3-3z"/></svg>',
    grid: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg>',
    map: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14.1 6.3 9.9 3.7a2 2 0 0 0-1.8 0L3.6 5.8A1 1 0 0 0 3 6.7v12.8a1 1 0 0 0 1.4.9l3.7-1.9a2 2 0 0 1 1.8 0l4.2 2.6a2 2 0 0 0 1.8 0l4.5-2.1a1 1 0 0 0 .6-.9V5.3a1 1 0 0 0-1.4-.9l-3.7 1.9a2 2 0 0 1-1.8 0z"/><path d="M15 5.8v15M9 3.2v15"/></svg>',
    search: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>',
    clock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
    bars: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" aria-hidden="true"><path d="M6 20v-6M12 20V9M18 20V4"/></svg>',
    list: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10" stroke-width="2"/><path d="M8 12.5l3 3 5-6"/></svg>',
    play: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5.5v13l11-6.5z" fill="currentColor"/></svg>',
    star: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2l2.9 6.6 7.1.6-5.4 4.7 1.6 7-6.2-3.7-6.2 3.7 1.6-7L2 9.2l7.1-.6z"/></svg>',
    trophy: '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 3h10v2h3a1 1 0 0 1 1 1v2a5 5 0 0 1-4.6 5A5 5 0 0 1 13 15.9V18h3a1 1 0 0 1 1 1v2H7v-2a1 1 0 0 1 1-1h3v-2.1A5 5 0 0 1 7.6 13 5 5 0 0 1 3 8V6a1 1 0 0 1 1-1h3zm0 4H5v1a3 3 0 0 0 2 2.8zm10 0v3.8A3 3 0 0 0 19 8V7z"/></svg>'
  };
  // Trang chung của cả site, không thuộc khoá nào.
  var GLOBAL_PAGES = ['/', '/roadmap_vi', '/mindmap_vi', '/mindmap_dl_vi'];
  var REPO = 'https://github.com/nmc2004nd/Self-Learning';
  var LAST_KEY = 'easyrl-last';
  // Các khoá học. Mục lục chung một tệp; mỗi nhóm thuộc về khoá của các trang bên trong nó,
  // và cột trái chỉ hiện các nhóm của khoá đang chọn.
  var COURSES = [
    { id: 'math', name: 'Toán nền tảng', tab: 'Toán', short: 'Toán', home: '/math/README_vi', icon: 'sigma', prefix: /^\/math\//, blurb: 'Đại số, xác suất, tối ưu lồi',
      desc: 'Nền toán cho học máy và học tăng cường',
      links: [['Machine Learning cơ bản', 'https://machinelearningcoban.com'], ['Linear Algebra (3Blue1Brown)', 'https://www.3blue1brown.com/topics/linear-algebra'], ['Convex Optimization (Boyd)', 'https://web.stanford.edu/~boyd/cvxbook/']] },
    { id: 'ml', name: 'Học máy', tab: 'Học máy', short: 'ML', home: '/ml/README_vi', icon: 'chart-spline', prefix: /^\/ml\//, blurb: 'Hồi quy, phân loại, phân cụm',
      desc: 'Từ hồi quy tuyến tính đến máy vector hỗ trợ',
      links: [['Machine Learning cơ bản', 'https://machinelearningcoban.com'], ['scikit-learn User Guide', 'https://scikit-learn.org/stable/user_guide.html'], ['NumPy Documentation', 'https://numpy.org/doc/stable/']] },
    { id: 'rl', name: 'Học tăng cường', tab: 'Tăng cường', short: 'RL', home: '/rl_vi', icon: 'target', start: '/chapter1/chapter1_vi', blurb: 'Từ MDP đến PPO, DDPG',
      desc: 'Tác tử học từ phần thưởng: MDP, DQN, PPO, DDPG',
      links: [['EasyRL (Datawhale)', 'https://github.com/datawhalechina/easy-rl'], ['Sách của Shiyu Zhao', 'https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning'], ['Gymnasium', 'https://gymnasium.farama.org'], ['OpenAI Spinning Up', 'https://spinningup.openai.com']] }
  ];
  var COURSE_KEY = 'hff-course';
  function courseById(id) {
    for (var i = 0; i < COURSES.length; i++) if (COURSES[i].id === id) return COURSES[i];
    return null;
  }
  function courseOf(path) {
    if (!path || GLOBAL_PAGES.indexOf(path) >= 0) return null;
    for (var i = 0; i < COURSES.length; i++) if (COURSES[i].prefix && COURSES[i].prefix.test(path)) return COURSES[i].id;
    return 'rl';
  }
  function lastKey(course) { return 'hff-last-' + course; }
  function store(key, val) { try { localStorage.setItem(key, val); } catch (e) { /* trình duyệt chặn lưu trữ */ } }
  function load(key) { try { return localStorage.getItem(key); } catch (e) { return null; } }
  // Mục lục gom từ hai nguồn: Sách Nấm EasyRL và sách của Shiyu Zhao.
  // Mỗi nhóm có màu và icon riêng; mỗi trang có icon theo nội dung.
  var NAV_GROUPS = {
    'Tổng quan': ['overview', 'house'],
    'Nhập môn': ['start', 'flag'],
    'Khởi đầu': ['start', 'flag'],
    'Tổng quan về học máy': ['base', 'blocks'],
    'Khởi động': ['policy', 'target'],
    'Mạng neuron nhân tạo': ['adv', 'brain-circuit'],
    'Kiến thức toán cơ bản': ['base', 'sigma'],
    'Tối ưu lồi': ['policy', 'mountain'],
    'Phụ lục': ['topic', 'book-open'],
    'Hệ thống gợi ý': ['value', 'sparkles'],
    'Giảm chiều dữ liệu': ['policy', 'layers'],
    'Support vector machine': ['adv', 'shield-check'],
    'Chủ đề bổ sung': ['math', 'sparkles'],
    'Bắt đầu': ['start', 'flag'],
    'Nền tảng': ['base', 'blocks'],
    'Dựa trên chính sách': ['policy', 'target'],
    'Dựa trên giá trị': ['value', 'chart-column'],
    'Nâng cao': ['adv', 'rocket'],
    'Chuyên đề': ['topic', 'sparkles'],
    'Nền tảng toán học': ['math', 'sigma']
  };
  // Nhãn nguồn chèn trước một nhóm, theo khoá học: '<khoá>:<nhóm>'.
  var NAV_PARTS = {
    'rl:base': ['Sách Nấm EasyRL', 'Wang, Yang, Jiang · Datawhale'],
    'rl:math': ['Sách của Shiyu Zhao', 'Mathematical Foundations of RL'],
    'ml:base': ['Machine Learning cơ bản', 'Vũ Hữu Tiệp · CC BY-SA 4.0'],
    'math:base': ['Machine Learning cơ bản', 'Vũ Hữu Tiệp · CC BY-SA 4.0'],
    'ml:math': ['Chủ đề bổ sung', 'Người biên soạn viết thêm']
  };
  var PAGE_ICON = {
    '/': 'house', '/ml/README_vi': 'chart-spline', '/math/README_vi': 'sigma', '/ml/ch07_vi': 'trending-up', '/ml/ch08_vi': 'chart-spline', '/ml/ch09_vi': 'network',
    '/ml/ch10_vi': 'chart-scatter', '/ml/ch12_vi': 'mountain', '/ml/ch13_vi': 'sliders-horizontal', '/ml/ch14_vi': 'gauge',
    '/ml/ch15_vi': 'layers', '/ml/ch16_vi': 'brain',
    '/ml/ch00_vi': 'book-open', '/ml/ch05_vi': 'blocks', '/ml/ch06_vi': 'sliders-horizontal', '/ml/ch11_vi': 'dices', '/ml/appendix_b_vi': 'eye',
    '/ml/ch17_vi': 'gift', '/ml/ch18_vi': 'network', '/ml/ch19_vi': 'layout-grid', '/ml/ch20_vi': 'square-function', '/ml/ch21_vi': 'chart-scatter', '/ml/ch22_vi': 'target',
    '/ml/ch26_vi': 'shield-check', '/ml/ch27_vi': 'gauge', '/ml/ch28_vi': 'sparkles', '/ml/ch29_vi': 'layout-grid', '/ml/eval_vi': 'chart-column', '/ml/tree_vi': 'network', '/ml/ensemble_vi': 'layers', '/ml/cluster_vi': 'chart-scatter', '/ml/prep_vi': 'funnel',
    '/math/ch01_vi': 'layout-grid', '/math/ch02_vi': 'square-function', '/math/ch03_vi': 'dices', '/math/ch04_vi': 'sliders-horizontal',
    '/math/ch23_vi': 'layers', '/math/ch24_vi': 'target', '/math/ch25_vi': 'repeat', '/math/appendix_a_vi': 'library', '/math/appendix_a_code_vi': 'terminal', '/rl_vi': 'target', '/guide_vi': 'compass', '/explorer_vi': 'map', '/notation_vi': 'whole-word',
    '/chapter1/chapter1_vi': 'sprout', '/chapter2/chapter2_vi': 'network', '/chapter3/chapter3_vi': 'table-2',
    '/chapter4/chapter4_vi': 'trending-up', '/chapter5/chapter5_vi': 'shield-check',
    '/chapter6/chapter6_vi': 'brain', '/chapter7/chapter7_vi': 'brain-circuit', '/chapter8/chapter8_vi': 'sliders-horizontal',
    '/chapter9/chapter9_vi': 'drama', '/chapter10/chapter10_vi': 'gift', '/chapter11/chapter11_vi': 'footprints',
    '/chapter12/chapter12_vi': 'gauge', '/chapter13/chapter13_vi': 'gamepad-2', '/chapter14/ls-imagine_vi': 'pickaxe',
    '/chapter15/chapter15_vi': 'eye', '/chapter16/chapter16_vi': 'globe',
    '/mathrl/README_vi': 'book-open', '/mathrl/ch01_vi': 'layout-grid', '/mathrl/ch02_vi': 'square-function',
    '/mathrl/ch03_vi': 'crown', '/mathrl/ch04_vi': 'repeat', '/mathrl/ch05_vi': 'dices', '/mathrl/ch06_vi': 'chart-scatter',
    '/mathrl/ch07_vi': 'timer', '/mathrl/ch08_vi': 'chart-spline', '/mathrl/ch09_vi': 'mountain',
    '/mathrl/ch10_vi': 'drama', '/mathrl/appendix_vi': 'library'
  };
  var SUB_ICON = [
    [/_notes_vi$/, 'notebook-pen'], [/_questions&keywords_vi$/, 'circle-help'], [/_review_vi$/, 'layers'],
    [/_viz_vi$/, 'monitor-play'], [/_code_vi$/, 'terminal'], [/\/project\d+_vi$/, 'code-xml'],
    [/_lib_vi$/, 'library'], [/_app_vi$/, 'rocket']
  ];
  function navIcon(name, cls) {
    var paths = (window.NAV_ICONS || {})[name];
    if (!paths) return '';
    return '<svg class="' + cls + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + paths + '</svg>';
  }
  function pageIcon(path) {
    if (PAGE_ICON[path]) return PAGE_ICON[path];
    for (var i = 0; i < SUB_ICON.length; i++) if (SUB_ICON[i][0].test(path)) return SUB_ICON[i][1];
    return null;
  }
  var DONE_KEY = 'easyrl-done';
  var REWARD_KEY = 'easyrl-reward';
  var manuallyOpen = {};
  var pagerObserver = null;
  var revealObserver = null;

  // ---------- hiệu ứng vui ----------
  function calm() { return matchMedia('(prefers-reduced-motion: reduce)').matches; }

  function readDone() {
    try { return JSON.parse(localStorage.getItem(DONE_KEY) || '[]'); } catch (e) { return []; }
  }

  function toast(msg) {
    var el = document.querySelector('.fun-toast');
    if (!el) {
      el = document.createElement('div');
      el.className = 'fun-toast';
      el.setAttribute('role', 'status');
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(el._t);
    el._t = setTimeout(function () { el.classList.remove('show'); }, 2600);
  }

  // Pháo giấy bắn ra từ điểm (x, y) trên màn hình.
  function confetti(x, y, n) {
    if (calm() || !document.body.animate) return;
    var colors = ['#3d6bf5', '#22b8f0', '#7c3aed', '#f5b43a', '#e8487f', '#0ea5a4'];
    for (var i = 0; i < (n || 30); i++) {
      var c = document.createElement('i');
      c.className = 'confetti';
      c.style.left = x + 'px';
      c.style.top = y + 'px';
      c.style.background = colors[i % colors.length];
      document.body.appendChild(c);
      var ang = -Math.PI / 2 + (Math.random() - .5) * Math.PI * 1.1;
      var v = 120 + Math.random() * 160;
      var dx = Math.cos(ang) * v;
      var dy = Math.sin(ang) * v;
      var rot = (Math.random() - .5) * 720;
      c.animate([
        { transform: 'translate(0,0) rotate(0)', opacity: 1 },
        { transform: 'translate(' + dx + 'px,' + dy + 'px) rotate(' + rot / 2 + 'deg)', opacity: 1, offset: .45 },
        { transform: 'translate(' + dx * 1.3 + 'px,' + (dy + 260) + 'px) rotate(' + rot + 'deg)', opacity: 0 }
      ], { duration: 1300 + Math.random() * 500, easing: 'cubic-bezier(.2,.7,.4,1)' }).onfinish = (function (el) {
        return function () { el.remove(); };
      })(c);
    }
  }

  // Số liệu trong phần đầu trang chủ đếm từ 0 lên.
  function countUp(section) {
    if (calm()) return;
    section.querySelectorAll('[data-count]:not([data-counted])').forEach(function (el, k) {
      el.setAttribute('data-counted', '');
      var to = parseInt(el.dataset.count, 10);
      var t0 = performance.now() + k * 120;
      (function tick(now) {
        var f = Math.max(0, Math.min(1, (now - t0) / 1100));
        el.textContent = Math.round(to * (1 - Math.pow(1 - f, 3)));
        if (f < 1 && el.isConnected) requestAnimationFrame(tick);
      })(t0);
    });
  }

  // Robot ở trang chủ: bấm để thưởng, mỗi 10 điểm có pháo giấy.
  function bindBot(section) {
    var bot = section.querySelector('#art-bot');
    if (!bot || bot.dataset.ready) return;
    bot.dataset.ready = '1';
    var fx = section.querySelector('#art-fx');
    function reward() {
      var r = 0;
      try { r = parseInt(localStorage.getItem(REWARD_KEY) || '0', 10) || 0; } catch (e) { /* bỏ qua */ }
      r += 1;
      try { localStorage.setItem(REWARD_KEY, String(r)); } catch (e) { /* bỏ qua */ }
      bot.classList.remove('jump');
      void bot.getBoundingClientRect();
      bot.classList.add('jump');
      var big = r % 10 === 0;
      var t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('class', 'art-fx-text' + (big ? ' big' : ''));
      t.setAttribute('x', String(232 + (Math.random() - .5) * 60));
      t.setAttribute('y', '30');
      t.setAttribute('text-anchor', 'middle');
      t.textContent = big ? 'G = ' + r + ' 🏆' : '+1';
      fx.appendChild(t);
      setTimeout(function () { t.remove(); }, 1200);
      bot.setAttribute('aria-label', 'Thưởng cho tác tử một điểm. Tổng phần thưởng: ' + r);
      if (big) {
        var box = bot.getBoundingClientRect();
        confetti(box.left + box.width / 2, box.top + box.height / 3, 40);
        toast('Tác tử đã nhận ' + r + ' điểm thưởng. Chính sách của nó đang tốt dần lên!');
      }
    }
    bot.addEventListener('click', reward);
    bot.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); reward(); }
    });
    bot.addEventListener('animationend', function () { bot.classList.remove('jump'); });
  }

  // Các khối trang chủ hiện dần khi cuộn tới.
  function reveal(section) {
    if (revealObserver) revealObserver.disconnect();
    if (!('IntersectionObserver' in window)) return;
    revealObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { en.target.classList.add('in'); revealObserver.unobserve(en.target); }
      });
    }, { rootMargin: '0px 0px -40px 0px' });
    section.querySelectorAll('.home-h2, .home-h2-row, .home-tools > a, .home-row, .home-banner, .home-agent, .hub-dash > *, .hub-layers > li').forEach(function (el, i) {
      if (el.classList.contains('in')) return;
      if (!el.classList.contains('reveal')) {
        el.classList.add('reveal');
        var grid = /\b(home-tools|hub-dash|hub-layers)\b/.test(el.parentElement.className);
        el.style.transitionDelay = (grid ? (Array.prototype.indexOf.call(el.parentElement.children, el) % 5) * 80 : 0) + 'ms';
      }
      revealObserver.observe(el);
    });
  }

  // Cuộn tới cuối bài lần đầu: đánh dấu đã đọc và ăn mừng một chút.
  function watchFinish(pager) {
    if (pagerObserver) { pagerObserver.disconnect(); pagerObserver = null; }
    if (!pager || !('IntersectionObserver' in window)) return;
    var path = currentPath;
    pagerObserver = new IntersectionObserver(function (entries) {
      if (!entries[0].isIntersecting || window.scrollY < 300 || path !== currentPath) return;
      pagerObserver.disconnect();
      var done = readDone();
      if (done.indexOf(path) >= 0) return;
      done.push(path);
      try { localStorage.setItem(DONE_KEY, JSON.stringify(done)); } catch (e) { return; }
      enhanceSidebar();
      renderRail(courseOf(path), sideLink(path));
      renderPanel(courseOf(path), sideLink(path));
      updateGlobalNav();
      var box = pager.getBoundingClientRect();
      confetti(box.left + box.width / 2, box.top + 10, 34);
      toast('Xong bài! +1 phần thưởng 🎉 (' + done.length + ' bài đã đọc)');
    }, { threshold: .6 });
    pagerObserver.observe(pager);
  }
  var currentPath = '/';
  var tocObserver = null;

  function isDark() {
    var t = document.documentElement.dataset.theme;
    return t ? t === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  }

  // Mô phỏng nhúng cùng origin và đọc data-theme, nên truyền luôn lựa chọn của người đọc xuống.
  function syncFrameTheme(frame) {
    try {
      var root = frame.contentDocument && frame.contentDocument.documentElement;
      if (!root) return;
      if (document.documentElement.dataset.theme) root.dataset.theme = document.documentElement.dataset.theme;
      else delete root.dataset.theme;
    } catch (e) { /* khung chưa sẵn sàng: sự kiện load sẽ gọi lại */ }
  }

  function buildBar() {
    if (document.querySelector('.site-bar')) return;
    var bar = document.createElement('div');
    bar.className = 'site-bar';
    var courseLinks = COURSES.map(function (c, k) {
      return '<a href="#' + c.home + '"><i class="t' + (k + 1) + '">' + navIcon(c.icon, '') + '</i><span>' + c.name +
        '<small>' + c.blurb + '</small></span></a>';
    }).join('');
    bar.innerHTML =
      '<button class="bar-btn" id="bar-menu" type="button" aria-label="Ẩn hoặc hiện mục lục">' + ICON.menu + '</button>' +
      '<a class="bar-brand" href="#/" aria-label="Hope for Future, về trang chủ"><span class="logo-cycle"><img class="is-on" src="img/brand/logo.png" alt=""></span><b>Hope for Future</b></a>' +
      '<nav class="bar-nav" aria-label="Điều hướng chính">' +
        '<a data-path="/" href="#/">' + ICON.home + 'Trang chủ</a>' +
        '<a data-path="/roadmap_vi" href="#/roadmap_vi">' + ICON.book + '<span>Lộ trình<span class="lbl-long"> học</span></span></a>' +
        '<div class="bar-menu"><button type="button" aria-expanded="false" aria-haspopup="true">' + ICON.grid + 'Khoá học' + ICON.down + '</button>' +
          '<div class="bar-pop">' + courseLinks + '</div></div>' +
        '<div class="bar-menu"><button type="button" aria-expanded="false" aria-haspopup="true">' + ICON.code + 'Tài nguyên' + ICON.down + '</button>' +
          '<div class="bar-pop">' +
            '<a href="#/mindmap_vi"><i class="t6">' + ICON.map + '</i><span>Bản đồ kiến thức<small>Mọi khoá học trên một sơ đồ</small></span></a>' +
            '<a href="#/ml/mindmap_vi"><i class="t4">' + ICON.map + '</i><span>Sơ đồ tư duy học máy<small>Khái niệm, công thức, mô hình</small></span></a>' +
            '<a href="#/mindmap_dl_vi"><i class="t5">' + ICON.map + '</i><span>Sơ đồ tư duy học sâu<small>Kiến trúc mạng, TensorFlow</small></span></a>' +
            '<a href="#/explorer_vi"><i class="t3">' + navIcon('target', '') + '</i><span>Bản đồ thuật toán RL<small>48 thuật toán và khái niệm</small></span></a>' +
            '<a href="#/notation_vi"><i class="t2">' + navIcon('whole-word', '') + '</i><span>Quy ước ký hiệu<small>Ký hiệu dùng chung</small></span></a>' +
            '<a href="#/guide_vi"><i class="t5">' + navIcon('compass', '') + '</i><span>Hướng dẫn đọc sách RL<small>EasyRL và sách của Zhao</small></span></a>' +
          '</div></div>' +
        '<a class="opt" href="' + REPO + '/issues" target="_blank" rel="noopener">' + ICON.users + 'Cộng đồng</a>' +
      '</nav>' +
      '<div class="bar-search search-host" id="bar-search"><kbd class="bar-kbd">Ctrl K</kbd></div>' +
      '<div class="bar-spacer"></div>' +
      '<button class="bar-btn" id="bar-find" type="button" aria-label="Tìm kiếm">' + ICON.search + '</button>' +
      '<button class="bar-btn" id="bar-theme" type="button"></button>' +
      '<span class="bar-sep" aria-hidden="true"></span>' +
      '<a class="bar-btn" href="' + REPO + '" target="_blank" rel="noopener" aria-label="Mã nguồn bản dịch trên GitHub">' + ICON.github + '</a>' +
      '<a class="bar-cta" id="bar-cta" href="#/chapter1/chapter1_vi"></a>' +
      '<div class="progress" id="bar-progress"></div>';
    document.body.appendChild(bar);

    // Các menu thả xuống trên thanh trên của trang chủ.
    var menus = Array.prototype.slice.call(bar.querySelectorAll('.bar-menu'));
    function setMenu(menu, open) {
      menu.classList.toggle('open', open);
      menu.querySelector('button').setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    menus.forEach(function (menu) {
      menu.querySelector('button').addEventListener('click', function (e) {
        e.stopPropagation();
        var open = !menu.classList.contains('open');
        menus.forEach(function (m) { setMenu(m, false); });
        setMenu(menu, open);
      });
      menu.querySelector('.bar-pop').addEventListener('click', function () { setMenu(menu, false); });
    });
    document.addEventListener('click', function (e) { menus.forEach(function (m) { if (!m.contains(e.target)) setMenu(m, false); }); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { menus.forEach(function (m) { setMenu(m, false); }); setNav(false); } });

    // Nút kính lúp (trang chủ): đưa con trỏ vào ô tìm kiếm lớn.
    document.getElementById('bar-find').addEventListener('click', function () {
      var input = searchInput();
      if (!input) return;
      window.scrollTo({ top: 0, behavior: calm() ? 'auto' : 'smooth' });
      input.focus();
    });

    // Nút ba gạch: mở cột trái dạng ngăn kéo khi màn hình hẹp (thay cho cơ chế đóng/mở của docsify).
    document.getElementById('bar-menu').addEventListener('click', function (e) {
      e.stopPropagation();
      setNav(!document.body.classList.contains('nav-open'));
    });
    var backdrop = document.createElement('div');
    backdrop.className = 'nav-backdrop';
    backdrop.addEventListener('click', function () { setNav(false); });
    document.body.appendChild(backdrop);

    var themeBtn = document.getElementById('bar-theme');
    function paintTheme() {
      var dark = isDark();
      themeBtn.innerHTML = dark ? ICON.sun : ICON.moon;
      themeBtn.setAttribute('aria-label', dark ? 'Chuyển sang nền sáng' : 'Chuyển sang nền tối');
    }
    function applyTheme() {
      var next = isDark() ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('easyrl-theme', next); } catch (e) { /* trình duyệt chặn lưu trữ */ }
      paintTheme();
      document.querySelectorAll('iframe.viz-frame').forEach(syncFrameTheme);
    }
    themeBtn.addEventListener('click', function () {
      if (!document.startViewTransition || calm()) { applyTheme(); return; }
      var r = themeBtn.getBoundingClientRect();
      var x = r.left + r.width / 2;
      var y = r.top + r.height / 2;
      var end = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
      document.startViewTransition(applyTheme).ready.then(function () {
        document.documentElement.animate(
          { clipPath: ['circle(0px at ' + x + 'px ' + y + 'px)', 'circle(' + end + 'px at ' + x + 'px ' + y + 'px)'] },
          { duration: 600, easing: 'cubic-bezier(.4, 0, .2, 1)', pseudoElement: '::view-transition-new(root)' }
        );
      }).catch(function () { /* trình duyệt bỏ qua hiệu ứng */ });
    });
    paintTheme();

    // Ctrl+K hoặc "/" để tìm kiếm, Esc để thoát.
    document.addEventListener('keydown', function (e) {
      var input = searchInput();
      if (!input) return;
      var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName) || e.target.isContentEditable;
      if ((e.key === 'k' && (e.ctrlKey || e.metaKey)) || (e.key === '/' && !typing)) {
        e.preventDefault();
        input.focus();
        input.select();
      } else if (e.key === 'Escape' && e.target === input) {
        input.blur();
      }
    });

    var brand = document.querySelector('.sidebar .app-name-link');
    if (brand) {
      brand.innerHTML = ICON.logo + '<span class="brand-text"><b>Hope for</b><b>Future</b></span>';
      brand.setAttribute('aria-label', 'Hope for Future, về trang chủ');
    }
    buildGlobalNav();
    buildPanel();
    var side = document.querySelector('.sidebar');
    if (side && !side.querySelector('.side-foot')) {
      var foot = document.createElement('div');
      foot.className = 'side-foot';
      side.appendChild(foot);
    }

    document.addEventListener('pointermove', function (e) {
      var el = e.target.closest && e.target.closest('.spot');
      if (!el) return;
      var b = el.getBoundingClientRect();
      el.style.setProperty('--mx', (e.clientX - b.left) + 'px');
      el.style.setProperty('--my', (e.clientY - b.top) + 'px');
    }, { passive: true });

    var ticking = false;
    window.addEventListener('scroll', function () {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(function () {
        var el = document.documentElement;
        var max = el.scrollHeight - el.clientHeight;
        document.getElementById('bar-progress').style.transform = 'scaleX(' + (max > 0 ? el.scrollTop / max : 0) + ')';
        document.body.classList.toggle('is-scrolled', el.scrollTop > 8);
        ticking = false;
      });
    }, { passive: true });
  }

  function setNav(open) {
    document.body.classList.toggle('nav-open', open);
    var btn = document.getElementById('bar-menu');
    if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  }

  // Vòng tiến độ nhỏ cạnh tên khoá ở cột trái.
  function miniRing(pct) {
    var C = 2 * Math.PI * 8;
    if (pct >= 100) return '<span class="mini-ring is-done">' + ICON.check + '</span>';
    return '<span class="mini-ring"><svg viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="8"/><circle class="fg" cx="10" cy="10" r="8" stroke-dasharray="' + C + '" stroke-dashoffset="' + (C * (1 - pct / 100)) + '"/></svg></span>';
  }

  // Cột trái chung (giống bản thiết kế Learning Hub): trang chủ, lộ trình, các khoá, tài nguyên, cộng đồng.
  function buildGlobalNav() {
    var side = document.querySelector('.sidebar');
    if (!side || side.querySelector('.side-global')) return;
    var nav = document.createElement('nav');
    nav.className = 'side-global';
    nav.setAttribute('aria-label', 'Điều hướng chung');
    function item(path, icon, label, extra) {
      return '<a data-path="' + path + '" href="#' + path + '">' + icon + '<span>' + label + '</span>' + (extra || '') + '</a>';
    }
    var courses = [
      { id: 'math' }, { id: 'ml' }, { soon: 'Học sâu' }, { id: 'rl' }, { soon: 'LLM và ứng dụng' }
    ].map(function (c) {
      if (c.soon) return '<span class="is-soon">' + ICON.lock + '<span>' + c.soon + '</span><small>Sắp có</small></span>';
      var co = courseById(c.id);
      return '<a data-course-link="' + co.id + '" href="#' + co.home + '"><i class="ring-slot"></i><span>' + co.name + '</span></a>';
    }).join('');
    nav.innerHTML =
      '<div class="sg-main">' + item('/', ICON.home, 'Trang chủ') + item('/roadmap_vi', ICON.book, 'Lộ trình học') + item('/mindmap_vi', ICON.map, 'Bản đồ kiến thức') + '</div>' +
      '<p class="sg-title">Khoá học</p><div class="sg-courses">' + courses + '</div>' +
      '<p class="sg-title">Tài nguyên</p>' +
      item('/ml/mindmap_vi', ICON.map, 'Sơ đồ tư duy học máy') + item('/mindmap_dl_vi', ICON.map, 'Sơ đồ tư duy học sâu') + item('/explorer_vi', navIcon('target', ''), 'Bản đồ thuật toán RL') + item('/notation_vi', ICON.note, 'Quy ước ký hiệu') + item('/guide_vi', navIcon('compass', ''), 'Hướng dẫn đọc sách RL') +
      '<p class="sg-title">Cộng đồng</p>' +
      '<a href="' + REPO + '/issues" target="_blank" rel="noopener">' + ICON.chat + '<span>Thảo luận, góp ý</span></a>' +
      '<a href="' + REPO + '" target="_blank" rel="noopener">' + ICON.github + '<span>Mã nguồn trên GitHub</span></a>';
    var brand = side.querySelector(':scope > h1');
    if (brand) brand.after(nav); else side.insertBefore(nav, side.firstChild);
    nav.addEventListener('click', function (e) { if (e.target.closest('a')) setNav(false); });
  }

  function updateGlobalNav() {
    var nav = document.querySelector('.side-global');
    var here = courseOf(currentPath);
    document.querySelectorAll('.side-global a[data-path], .bar-nav a[data-path]').forEach(function (a) {
      var on = a.dataset.path === currentPath;
      a.classList.toggle('is-here', on);
      if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
    if (!nav) return;
    nav.querySelectorAll('a[data-course-link]').forEach(function (a) {
      var id = a.dataset.courseLink;
      var p = courseProgress(id);
      a.querySelector('.ring-slot').innerHTML = miniRing(p.pct);
      a.title = courseById(id).name + ': đã đọc ' + p.done + '/' + p.total + ' chương';
      a.classList.toggle('is-here', here === id);
    });
  }

  // Cột giữa của trang bài học: thẻ khoá học, danh sách chương (mục lục docsify), bài tiếp theo, mục lục nhanh.
  var panelMq = matchMedia('(min-width: 1024px)');
  function buildPanel() {
    if (document.querySelector('.course-panel')) return;
    var panel = document.createElement('aside');
    panel.className = 'course-panel';
    panel.setAttribute('aria-label', 'Khoá học đang đọc');
    panel.innerHTML =
      '<section class="cp-card"><div class="cp-head"><i class="cp-ico"></i><div><b class="cp-name"></b><small class="cp-desc"></small></div></div>' +
      '<div class="cp-progress"><span class="cp-bar"><i></i></span><b class="cp-pct"></b></div><div class="cp-nav"></div></section>' +
      '<a class="cp-next" hidden></a><div class="cp-toc"></div>';
    var nav = document.querySelector('.sidebar-nav');
    if (nav) panel.querySelector('.cp-nav').appendChild(nav);
    document.body.appendChild(panel);
    placePanel();
    if (panelMq.addEventListener) panelMq.addEventListener('change', function () { placePanel(); placeToc(); });
  }
  // Màn hình hẹp: cột giữa nằm trong ngăn kéo bên trái, dưới phần điều hướng chung.
  function placePanel() {
    var panel = document.querySelector('.course-panel');
    var side = document.querySelector('.sidebar');
    if (!panel || !side) return;
    if (panelMq.matches) { if (panel.parentElement !== document.body) document.body.appendChild(panel); }
    else if (panel.parentElement !== side) side.insertBefore(panel, side.querySelector('.side-foot'));
  }

  function renderPanel(courseId, link) {
    var panel = document.querySelector('.course-panel');
    if (!panel) return;
    var course = courseById(courseId);
    panel.classList.toggle('has-course', !!course);
    if (!course) return;
    var p = courseProgress(courseId);
    panel.querySelector('.cp-ico').innerHTML = navIcon(course.icon, '');
    panel.querySelector('.cp-name').textContent = course.name;
    panel.querySelector('.cp-desc').textContent = course.desc;
    panel.querySelector('.cp-pct').textContent = p.pct + '%';
    var bar = panel.querySelector('.cp-bar i');
    requestAnimationFrame(function () { bar.style.width = p.pct + '%'; });
    var next = nextLesson(p, link);
    var nx = panel.querySelector('.cp-next');
    nx.hidden = !next;
    if (next) {
      nx.href = next.getAttribute('href');
      nx.innerHTML = '<img src="img/brand/tile-' + ({ math: 1, ml: 3, rl: 2 }[courseId] || 1) + '.jpg" alt=""><span><b>' + ICON.star + 'Tiếp tục học</b><small></small></span>' + ICON.chevron;
      nx.querySelector('small').textContent = 'Bài tiếp theo: ' + next.textContent.trim();
    }
  }

  // Chương nên đọc tiếp: chương chưa xong đầu tiên sau chương đang mở, nếu không có thì chương chưa xong đầu tiên.
  function nextLesson(p, link) {
    var chapter = link && chapterItemOf(link);
    var here = chapter ? chapter.querySelector(':scope > a') : link;
    var at = here ? p.list.indexOf(here) : -1;
    var done = readDone();
    for (var i = at + 1; i < p.list.length; i++) if (done.indexOf(hrefPath(p.list[i])) < 0) return p.list[i];
    return p.firstOpen;
  }

  function setCourseView(id, remember) {
    if (!courseById(id)) id = 'rl';
    document.body.dataset.course = id;
    if (remember) store(COURSE_KEY, id);
  }

  // Plugin search của docsify đặt ô tìm kiếm trong cột trái; chuyển nó lên thanh trên (giữ nguyên sự kiện).
  function mountSearch() {
    var slot = document.getElementById('bar-search');
    var search = document.querySelector('.sidebar .search');
    if (!slot || !search) return;
    slot.insertBefore(search, slot.firstChild);
    var input = search.querySelector('input');
    input.setAttribute('aria-label', 'Tìm kiếm trong sách');
    input.addEventListener('input', function () { search.parentElement.classList.toggle('has-text', !!input.value); });
    var panel = search.querySelector('.results-panel');
    if (panel) new MutationObserver(function () { fixSearchResults(panel, input); }).observe(panel, { childList: true });
  }

  function searchInput() { return document.querySelector('.search-host .search input'); }

  // Trang chủ có ô tìm kiếm lớn giữa phần đầu trang: chuyển chính ô của plugin vào đó (giữ nguyên sự kiện).
  // Docsify thay nội dung trang khi chuyển trang, nên trước đó phải đưa ô về thanh trên (xem beforeEach).
  function placeSearch() {
    var search = document.querySelector('.search-host .search');
    var target = document.getElementById('hub-search') || document.getElementById('bar-search');
    if (!search || !target || search.parentElement === target) return;
    target.insertBefore(search, target.firstChild);
    var go = target.querySelector('.hub-search-go');
    if (go && !go.dataset.bound) {
      go.dataset.bound = '1';
      go.addEventListener('click', function () {
        var input = searchInput();
        if (!input) return;
        input.focus();
        input.dispatchEvent(new Event('input'));
      });
    }
  }
  function rescueSearch() {
    var search = document.querySelector('#hub-search .search');
    var bar = document.getElementById('bar-search');
    if (search && bar) bar.insertBefore(search, bar.firstChild);
  }

  // Plugin search 4.13 bỏ dấu tiếng Việt khi hiển thị kết quả ("đieu khien sai phan thoi gian").
  // Lấy lại tiêu đề và đoạn trích gốc từ chỉ mục mà plugin lưu trong localStorage.
  var searchIndex = null;
  function loadSearchIndex() {
    if (searchIndex) return searchIndex;
    try {
      var raw = JSON.parse(localStorage.getItem('docsify.search.index') || 'null');
      if (!raw) return null;
      searchIndex = {};
      Object.keys(raw).forEach(function (path) {
        Object.keys(raw[path]).forEach(function (slug) { searchIndex[slug] = raw[path][slug]; });
      });
    } catch (e) { return null; }
    return searchIndex;
  }

  // Bỏ dấu giống plugin để so khớp, kèm bảng ánh xạ về vị trí trong chuỗi gốc.
  function fold(s) {
    var t = '';
    var map = [];
    for (var i = 0; i < s.length; i++) {
      var d = s[i].normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
      for (var k = 0; k < d.length; k++) { t += d[k]; map.push(i); }
    }
    map.push(s.length);
    return { t: t, map: map };
  }

  function decodeHtml(s) {
    var el = document.createElement('textarea');
    el.innerHTML = s;
    return el.value;
  }

  // Dựng đoạn văn có tô sáng từ khóa trên chuỗi gốc (giữ dấu).
  function highlight(text, terms, maxLen) {
    var f = fold(text);
    var hits = [];
    terms.forEach(function (term) {
      var from = 0;
      var at;
      while (term && (at = f.t.indexOf(term, from)) >= 0) {
        hits.push([f.map[at], f.map[at + term.length]]);
        from = at + term.length;
      }
    });
    hits.sort(function (a, b) { return a[0] - b[0]; });
    var start = 0;
    var end = text.length;
    if (maxLen && text.length > maxLen) {
      start = hits.length ? Math.max(0, hits[0][0] - 30) : 0;
      end = Math.min(text.length, start + maxLen);
    }
    var frag = document.createDocumentFragment();
    if (start > 0) frag.appendChild(document.createTextNode('…'));
    var pos = start;
    hits.forEach(function (h) {
      if (h[0] < pos || h[1] > end) return;
      frag.appendChild(document.createTextNode(text.slice(pos, h[0])));
      var em = document.createElement('em');
      em.className = 'search-keyword';
      em.textContent = text.slice(h[0], h[1]);
      frag.appendChild(em);
      pos = h[1];
    });
    frag.appendChild(document.createTextNode(text.slice(pos, end)));
    if (end < text.length) frag.appendChild(document.createTextNode('…'));
    return frag;
  }

  // Tiêu đề trong chỉ mục tìm kiếm là markdown thô, có thể chứa $...$.
  // Phần chữ vẫn tô sáng từ khóa, phần công thức thì vẽ bằng KaTeX.
  function titleWithMath(title, terms) {
    var parts = title.split(/\$([^$]+)\$/);
    if (parts.length === 1 || !window.katex) return highlight(title, terms, 0);
    var frag = document.createDocumentFragment();
    parts.forEach(function (part, i) {
      if (i % 2 === 0) { if (part) frag.appendChild(highlight(part, terms, 0)); return; }
      var span = document.createElement('span');
      // \tag chỉ dùng được trong công thức hiển thị riêng dòng; ở đây vẽ nội dòng nên bỏ đi.
      var tex = part.replace(/\\tag\*?\{[^}]*\}/g, '');
      try { span.innerHTML = katex.renderToString(tex, { throwOnError: false }); }
      catch (e) { span.textContent = part; }
      frag.appendChild(span);
    });
    return frag;
  }

  // Đoạn trích nội dung có công thức: cắt cửa sổ quanh từ khóa đầu tiên nhưng không cắt
  // ngang một công thức $...$, rồi vẽ bằng titleWithMath.
  function snippetWithMath(text, terms, maxLen) {
    text = text.replace(/\$\$/g, '$');
    if (text.indexOf('$') < 0 || !window.katex) return highlight(text, terms, maxLen);
    var f = fold(text);
    var first = -1;
    terms.forEach(function (term) {
      var at = term ? f.t.indexOf(term) : -1;
      if (at >= 0 && (first < 0 || f.map[at] < first)) first = f.map[at];
    });
    var start = first > 0 ? Math.max(0, first - 30) : 0;
    var end = Math.min(text.length, start + maxLen);
    var re = /\$[^$]+\$/g, m;
    while ((m = re.exec(text))) {
      var a = m.index, b = a + m[0].length;
      if (start > a && start < b) start = a;
      if (end > a && end < b) end = b - a <= maxLen ? b : a;
    }
    var frag = document.createDocumentFragment();
    if (start > 0) frag.appendChild(document.createTextNode('…'));
    frag.appendChild(titleWithMath(text.slice(start, end), terms));
    if (end < text.length) frag.appendChild(document.createTextNode('…'));
    return frag;
  }

  function fixSearchResults(panel, input) {
    var posts = panel.querySelectorAll('.matching-post:not([data-fixed])');
    if (!posts.length) return;
    var index = loadSearchIndex();
    var query = input.value.trim();
    var terms = [query].concat(query.split(/[\s\-，\\/]+/)).map(function (w) { return fold(w).t; })
      .filter(function (w, i, all) { return w && all.indexOf(w) === i; });
    posts.forEach(function (post) {
      post.setAttribute('data-fixed', '');
      var a = post.querySelector('a');
      var href = a ? a.getAttribute('href') : '';
      // Trang chủ chỉ là trang điều hướng, kết quả của nó toàn mã HTML.
      if (/^#\/(rl_vi)?(\?|$)/.test(href)) { post.remove(); return; }
      var page = document.querySelector('.sidebar-nav a[href="' + href.split('?')[0].replace(/"/g, '') + '"]');
      if (page && a && !a.querySelector('.search-where')) {
        var where = document.createElement('small');
        where.className = 'search-where';
        var ch = chapterItemOf(page);
        var head = ch && ch.querySelector(':scope > a');
        var course = courseById(courseOf(hrefPath(page)));
        where.textContent = (course ? course.short + ' · ' : '') +
          (head && head !== page ? shortChapter(head.textContent) + ' · ' + page.textContent : page.textContent);
        a.insertBefore(where, a.firstChild);
      }
      var entry = index && index[href];
      if (!entry) return;
      var h2 = post.querySelector('h2');
      var p = post.querySelector('p');
      if (h2 && entry.title) { h2.textContent = ''; h2.appendChild(titleWithMath(decodeHtml(entry.title), terms)); }
      if (p && entry.body) { p.textContent = ''; p.appendChild(snippetWithMath(entry.body.replace(/\s+/g, ' ').trim(), terms, 150)); }
    });
    if (!panel.querySelector('.matching-post')) {
      panel.innerHTML = '<p class="empty">Không tìm thấy kết quả</p>';
    }
  }

  function resetSearch() {
    var input = searchInput();
    if (!input || !input.value) return;
    input.value = '';
    input.dispatchEvent(new Event('input'));
    input.blur();
  }

  function hrefPath(a) {
    return decodeURIComponent(a.getAttribute('href').replace(/^#/, '').split('?')[0]);
  }

  // Gắn màu nhóm, nhãn nguồn và icon trước mỗi mục. Chạy lại được nhiều lần.
  function decorateNav(nav) {
    var top = nav.querySelector(':scope > ul');
    if (!top) return;
    Array.prototype.slice.call(top.children).forEach(function (li) {
      if (li.classList.contains('nav-part') || li.dataset.g) return;
      var first = li.querySelector('a[href^="#/"]');
      var c = first ? courseOf(hrefPath(first)) : null;
      li.dataset.course = c || 'all';
      var label = li.querySelector(':scope > p') || li.firstChild;
      var text = label ? label.textContent.trim() : '';
      var g = NAV_GROUPS[text];
      if (!g) return;
      li.dataset.g = g[0];
      li.classList.add('nav-group');
      var head = document.createElement('div');
      head.className = 'nav-group-head';
      head.innerHTML = navIcon(g[1], 'nav-group-ico') + '<span></span>';
      head.lastChild.textContent = text;
      li.replaceChild(head, label);
      var part = NAV_PARTS[li.dataset.course + ':' + g[0]];
      if (part) {
        var p = document.createElement('li');
        p.className = 'nav-part';
        p.dataset.part = g[0];
        p.dataset.course = li.dataset.course;
        p.innerHTML = '<b></b><small></small>';
        p.firstChild.textContent = part[0];
        p.lastChild.textContent = part[1];
        top.insertBefore(p, li);
      }
    });
    nav.querySelectorAll('a[href^="#/"]').forEach(function (a) {
      if (a.querySelector('.nav-ico')) return;
      var name = pageIcon(hrefPath(a));
      if (name) a.insertAdjacentHTML('afterbegin', navIcon(name, 'nav-ico'));
    });
  }

  // Docsify dựng lại cột trái mỗi lần chuyển trang, nên trạng thái mở được áp lại ở đây.
  function enhanceSidebar() {
    var nav = document.querySelector('.sidebar-nav');
    if (!nav) return;
    nav.querySelectorAll('li').forEach(function (li) {
      var sub = li.querySelector(':scope > ul');
      var link = li.querySelector(':scope > a');
      if (!sub || !link) return;
      li.classList.add('has-sub');
      if (!li.querySelector(':scope > .sub-toggle')) {
        var btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'sub-toggle';
        btn.innerHTML = ICON.chevron;
        btn.setAttribute('aria-label', 'Mở hoặc thu gọn: ' + link.textContent);
        link.after(btn);
      }
      var key = hrefPath(link);
      var containsCurrent = Array.prototype.some.call(li.querySelectorAll('a[href^="#/"]'), function (a) {
        return hrefPath(a) === currentPath;
      });
      var open = containsCurrent || manuallyOpen[key] === true;
      li.classList.toggle('is-open', open);
      li.querySelector(':scope > .sub-toggle').setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    decorateNav(nav);
    var done = readDone();
    nav.querySelectorAll('a[href^="#/"]').forEach(function (a) {
      a.classList.toggle('is-done', done.indexOf(hrefPath(a)) >= 0);
    });
    nav.querySelectorAll('a').forEach(function (a) {
      var here = a.getAttribute('href').indexOf('#/') === 0 && hrefPath(a) === currentPath;
      a.classList.toggle('is-current', here);
      if (here) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
  }

  function chapterItemOf(link) {
    var li = link.closest('li');
    if (li.classList.contains('has-sub')) return li;
    var parent = li.parentElement.closest('li.has-sub');
    return parent || null;
  }

  function shortChapter(text) {
    var m = text.match(/^(Chương \d+)/);
    return m ? m[1] : text;
  }

  // Lối tắt: Bài đọc · Ghi chú · Bài tập · Ôn tập · Mô phỏng của chương đang đọc.
  function renderChapterTools(section, link) {
    var chapter = link && chapterItemOf(link);
    if (!chapter) return;
    var head = chapter.querySelector(':scope > a');
    var items = [{ a: head, label: 'Bài đọc' }];
    chapter.querySelectorAll(':scope > ul > li > a').forEach(function (a) { items.push({ a: a, label: a.textContent.replace(/^(Dự án \d+).*/, '$1') }); });
    if (items.length < 2) return;
    var box = document.createElement('nav');
    box.className = 'ch-tools';
    box.setAttribute('aria-label', 'Tài liệu của chương');
    items.forEach(function (it) {
      var a = document.createElement('a');
      a.href = it.a.getAttribute('href');
      a.textContent = it.label;
      if (it.a === link) { a.className = 'is-here'; a.setAttribute('aria-current', 'page'); }
      box.appendChild(a);
    });
    section.insertBefore(box, section.firstChild);
  }

  // Nhãn mục lục cho một tiêu đề. Công thức KaTeX chứa thêm bản MathML ẩn (có cả mã LaTeX),
  // nên textContent sẽ ra "v∗v^{\ast}v∗". Giữ phần HTML hiển thị của KaTeX, bỏ phần MathML.
  function setHeadingLabel(link, h) {
    if (!h.querySelector('.katex')) { link.textContent = h.textContent; return; }
    var clone = (h.querySelector('a.anchor') || h).cloneNode(true);
    clone.querySelectorAll('.katex-mathml').forEach(function (m) { m.remove(); });
    link.innerHTML = clone.innerHTML;
    link.setAttribute('aria-label', clone.textContent.replace(/\s+/g, ' ').trim());
  }

  function renderPageToc(section) {
    if (tocObserver) { tocObserver.disconnect(); tocObserver = null; }
    document.querySelectorAll('.page-toc').forEach(function (t) { t.remove(); });
    var heads = Array.prototype.filter.call(section.querySelectorAll('h2, h3'), function (h) {
      return h.querySelector('a.anchor');
    });
    document.body.classList.toggle('has-toc', heads.length >= 3);
    if (heads.length < 3) return;

    var toc = document.createElement('details');
    toc.className = 'page-toc';
    var html = '<summary>Mục lục nhanh</summary><ul>';
    heads.forEach(function (h) {
      html += '<li class="lv' + h.tagName[1] + '"><a></a></li>';
    });
    toc.innerHTML = html + '</ul>';
    var links = toc.querySelectorAll('a');
    heads.forEach(function (h, i) {
      links[i].href = h.querySelector('a.anchor').getAttribute('href');
      setHeadingLabel(links[i], h);
    });
    var wide = panelMq;
    toc.open = wide.matches;
    toc.querySelector('summary').addEventListener('click', function (e) { if (wide.matches) e.preventDefault(); });
    toc.addEventListener('click', function (e) { if (!wide.matches && e.target.tagName === 'A') toc.open = false; });
    section.insertBefore(toc, section.firstChild);

    if (!('IntersectionObserver' in window)) return;
    var visible = {};
    tocObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { visible[heads.indexOf(en.target)] = en.isIntersecting; });
      var first = -1;
      for (var i = 0; i < heads.length; i++) { if (visible[i]) { first = i; break; } }
      if (first < 0) return;
      links.forEach(function (a, i) {
        a.classList.toggle('is-current', i === first);
        if (i === first && wide.matches) a.scrollIntoView({ block: 'nearest' });
      });
    }, { rootMargin: '-60px 0px -65% 0px' });
    heads.forEach(function (h) { tocObserver.observe(h); });
  }

  // Thứ tự đọc chỉ gồm các trang của sách; ôn tập và mô phỏng là trang riêng nên không nằm trong đó.
  function readingOrder() {
    return Array.prototype.slice.call(document.querySelectorAll('.sidebar-nav a[href^="#/"]'))
      .filter(function (a) { return a.getAttribute('href') !== '#/' && courseOf(hrefPath(a)) === courseOf(currentPath); });
  }
  function pagerLabel(a) {
    var ch = chapterItemOf(a);
    var head = ch && ch.querySelector(':scope > a');
    return head && head !== a ? shortChapter(head.textContent) + ' · ' + a.textContent : a.textContent;
  }
  function renderPager(section, link) {
    var all = readingOrder();
    var i = link ? all.indexOf(link) : -1;
    if (i < 0) return;
    var label = pagerLabel;
    function item(a, cls, word) {
      if (!a) return null;
      var el = document.createElement('a');
      el.className = cls;
      el.href = a.getAttribute('href');
      var s = document.createElement('small');
      s.textContent = word;
      el.appendChild(s);
      el.appendChild(document.createTextNode(label(a)));
      return el;
    }
    var pager = document.createElement('nav');
    pager.className = 'pager';
    pager.setAttribute('aria-label', 'Bài trước và bài sau');
    var prev = item(all[i - 1], 'prev', 'Bài trước');
    var next = item(all[i + 1], 'next', 'Bài sau');
    if (prev) pager.appendChild(prev);
    if (next) pager.appendChild(next);
    pager.querySelectorAll('a').forEach(function (a) { a.classList.add('spot'); });
    section.appendChild(pager);
    watchFinish(pager);
  }

  // Đưa mục đang đọc vào vùng nhìn thấy của cột trái.
  function revealInSidebar(link) {
    var side = document.querySelector(panelMq.matches ? '.course-panel' : '.sidebar');
    if (!link || !side) return;
    var box = side.getBoundingClientRect();
    var pos = link.getBoundingClientRect();
    if (pos.top < box.top + 80 || pos.bottom > box.bottom - 40) side.scrollTop += pos.top - box.top - box.height / 3;
  }

  function sideLink(path) {
    return path ? document.querySelector('.sidebar-nav a[href="#' + path.replace(/"/g, '') + '"]') : null;
  }

  // Tiến độ một khoá tính theo chương (mục cấp một trong mỗi nhóm của cột trái) đã đọc xong.
  function courseProgress(id) {
    var done = readDone();
    var list = Array.prototype.filter.call(document.querySelectorAll('.sidebar-nav > ul > li > ul > li > a[href^="#/"]'), function (a) {
      return courseOf(hrefPath(a)) === id;
    });
    var n = 0;
    var firstOpen = null;
    list.forEach(function (a) {
      if (done.indexOf(hrefPath(a)) >= 0) n++;
      else if (!firstOpen) firstOpen = a;
    });
    return { done: n, total: list.length, pct: list.length ? Math.round(100 * n / list.length) : 0, list: list, firstOpen: firstOpen };
  }

  // Bảng điều khiển ở trang chủ: trạng thái lộ trình và thanh tiến độ của từng khoá.
  function renderDashboard(section) {
    var roadmap = section.querySelector('.roadmap');
    section.querySelectorAll('[data-course] .pbar').forEach(function (bar) {
      var box = bar.closest('[data-course]');
      var p = courseProgress(box.dataset.course);
      var txt = box.querySelector('.ptext');
      if (txt) txt.textContent = p.done + ' / ' + p.total + ' chương';
      var st = box.querySelector('.rm-status');
      if (st) {
        st.className = 'rm-status' + (p.total && p.done === p.total ? ' is-done' : p.done ? ' is-now' : '');
        st.textContent = p.total && p.done === p.total ? 'Hoàn thành' : p.done ? 'Đang học' : 'Chưa bắt đầu';
      }
      box.classList.toggle('is-done', !!p.total && p.done === p.total);
      box.classList.toggle('is-now', p.done > 0 && p.done < p.total);
      requestAnimationFrame(function () { bar.firstChild.style.width = p.pct + '%'; });
    });
    if (!roadmap) return;
    var lastCourse = courseOf(load(LAST_KEY));
    var current = null;
    roadmap.querySelectorAll('li[data-course]').forEach(function (li) {
      var p = courseProgress(li.dataset.course);
      var st = li.querySelector('.rm-state');
      var finished = p.total > 0 && p.done === p.total;
      li.classList.toggle('is-done', finished);
      li.classList.remove('is-current');
      if (finished) { st.className = 'rm-state is-done'; st.textContent = 'Hoàn thành'; }
      else if (p.done > 0) { st.className = 'rm-state'; st.textContent = p.done + '/' + p.total + ' bài'; }
      else { st.className = 'rm-state'; st.innerHTML = ICON.chevron; }
      if (!finished && (li.dataset.course === lastCourse || (!current && !lastCourse))) current = current || li;
    });
    if (!current) current = roadmap.querySelector('li[data-course]:not(.is-done)');
    if (current) current.classList.add('is-current');
    section.querySelectorAll('.fc[data-course]').forEach(function (card) {
      var p = courseProgress(card.dataset.course);
      var bar = card.querySelector('.fc-bar');
      bar.title = 'Đã đọc ' + p.done + '/' + p.total + ' chương';
      requestAnimationFrame(function () { bar.firstChild.style.width = Math.max(p.pct, 0) + '%'; });
    });
  }

  // ---------- đầu trang bài học: đường dẫn và thẻ tiêu đề ----------
  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html) e.innerHTML = html;
    return e;
  }

  function renderLessonHead(section, link, courseId) {
    var head = document.querySelector('.lesson-head');
    if (!head) {
      head = el('div', 'lesson-head');
      section.parentNode.insertBefore(head, section);
    }
    head.innerHTML = '';
    var course = courseById(courseId);
    var group = link && link.closest('.nav-group');
    var groupName = group ? group.querySelector('.nav-group-head span').textContent : '';
    var chapter = link && chapterItemOf(link);
    var chHead = chapter && chapter.querySelector(':scope > a');

    var crumbs = el('nav', 'crumbs');
    crumbs.setAttribute('aria-label', 'Vị trí trang');
    var sep = ICON.chevron.replace('<svg ', '<svg class="sep" ');
    var home = el('a', '', ICON.home);
    home.href = '#/';
    home.setAttribute('aria-label', 'Trang chủ');
    crumbs.appendChild(home);
    function add(text, href) {
      crumbs.insertAdjacentHTML('beforeend', sep);
      var x = el(href ? 'a' : 'span');
      x.textContent = text;
      if (href) x.href = href;
      crumbs.appendChild(x);
    }
    if (course && course.home !== currentPath) add(course.name, '#' + course.home);
    if (groupName && groupName !== 'Tổng quan') add(groupName);
    if (chHead && chHead !== link) add(shortChapter(chHead.textContent), chHead.getAttribute('href'));

    var h1 = section.querySelector(':scope > h1');
    if (h1) { section._h1 = h1; section._h1For = currentPath; }
    else if (section._h1For === currentPath) h1 = section._h1;
    add(link ? link.textContent.trim() : (h1 ? h1.textContent.trim() : ''));
    head.appendChild(crumbs);
    if (!h1) return;

    var words = 0;
    section.querySelectorAll('p, li, td').forEach(function (p) { words += (p.textContent.match(/\S+/g) || []).length; });
    var mins = Math.max(1, Math.round(words / 220));
    var parts = section.querySelectorAll(':scope > h2').length;
    var isDone = readDone().indexOf(currentPath) >= 0;
    var chipIcon = course ? navIcon(course.icon, '') : ICON.book;
    var hero = el('header', 'lesson-hero',
      '<div class="lesson-hero-text"><span class="lesson-chip">' + chipIcon + '<span></span></span>' +
      '<div class="lesson-meta"><span>' + ICON.clock + mins + ' phút đọc</span>' +
      (course ? '<span>' + ICON.bars + course.name + '</span>' : '') +
      (parts ? '<span>' + ICON.list + parts + ' mục</span>' : '') +
      (isDone ? '<span class="is-done">' + ICON.check + 'Đã đọc xong</span>' : '') + '</div></div>');
    hero.querySelector('.lesson-chip span').textContent = (chHead && chHead !== link) ? shortChapter(chHead.textContent) : (groupName && groupName !== 'Tổng quan' ? groupName : (course ? course.name : 'Hope for Future'));
    hero.querySelector('.lesson-meta').before(h1);
    hero.prepend(sceneLayer());
    hero.appendChild(el('img', 'hero-mascot'));
    hero.lastChild.src = 'img/brand/whale-hero.png';
    hero.lastChild.alt = '';
    head.appendChild(hero);
    head.classList.remove('page-enter');
    void head.offsetWidth;
    head.classList.add('page-enter');
  }

  // ---------- cột phải: tiến độ, mục tiêu, liên kết hữu ích, lời nhắn ----------
  function renderRail(courseId, link) {
    var rail = document.querySelector('.lesson-rail');
    if (!rail) {
      rail = el('aside', 'lesson-rail');
      rail.setAttribute('aria-label', 'Tiến độ và liên kết');
      document.body.appendChild(rail);
    }
    rail.innerHTML = '';
    var course = courseById(courseId);
    if (course) {
      var p = courseProgress(courseId);
      var C = 2 * Math.PI * 52;
      var nextA = nextLesson(p, link);
      var card = el('section', 'rail-card rail-progress',
        '<h3>Tiến độ học tập</h3>' +
        '<div class="ring"><svg viewBox="0 0 120 120" aria-hidden="true"><defs><linearGradient id="ring-grad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#60a5fa"/><stop offset="1" stop-color="#1e40af"/></linearGradient></defs>' +
        '<circle class="ring-bg" cx="60" cy="60" r="52"/><circle class="ring-fg" cx="60" cy="60" r="52" stroke-dasharray="' + C + '" stroke-dashoffset="' + C + '"/></svg>' +
        '<b>' + p.pct + '<small>%</small></b></div>' +
        '<p>Hoàn thành <b>' + p.done + ' / ' + p.total + '</b> chương</p>');
      if (nextA) {
        var go = el('a', 'rail-btn', ICON.play + '<span>Tiếp tục học</span>');
        go.href = nextA.getAttribute('href');
        go.title = nextA.textContent.trim();
        card.appendChild(go);
      }
      rail.appendChild(card);
      var fg = card.querySelector('.ring-fg');
      requestAnimationFrame(function () { requestAnimationFrame(function () { fg.style.strokeDashoffset = C * (1 - p.pct / 100); }); });

      var goal = el('section', 'rail-card rail-goal',
        '<h3>' + ICON.trophy + 'Mục tiêu của bạn</h3>' +
        '<div class="goal-row"><i>' + ICON.flag + '</i><span>Hoàn thành lộ trình <b></b></span></div>' +
        '<div class="goal-bar"><i></i></div><small>' + p.done + ' / ' + p.total + ' chương</small>');
      goal.querySelector('b').textContent = course.name;
      rail.appendChild(goal);
      var gb = goal.querySelector('.goal-bar i');
      requestAnimationFrame(function () { gb.style.width = Math.max(p.pct, 2) + '%'; });

      var links = el('section', 'rail-card rail-links', '<h3>' + ICON.link + 'Liên kết hữu ích</h3><ul></ul>');
      course.links.forEach(function (l) {
        var li = el('li', '', '<a target="_blank" rel="noopener">' + ICON.link + '<span></span>' + ICON.ext + '</a>');
        li.querySelector('a').href = l[1];
        li.querySelector('span').textContent = l[0];
        links.querySelector('ul').appendChild(li);
      });
      rail.appendChild(links);
    }
    rail.appendChild(el('figure', 'rail-card rail-quote',
      '<p>Học hôm nay,<br>kiến tạo tương lai.</p><small>– Hope for Future</small>' +
        '<img class="rq-whale" src="img/brand/whale-love.png" alt="" data-no-zoom>'));
  }

  // Mục lục của trang: nằm ở cột giữa khi màn hình đủ rộng, ngược lại nằm đầu bài.
  function placeToc() {
    var toc = document.querySelector('.page-toc');
    var section = document.querySelector('.markdown-section');
    var slot = document.querySelector('.course-panel .cp-toc');
    if (!toc || !section) return;
    if (panelMq.matches && slot) {
      slot.appendChild(toc);
      toc.open = true;
    } else {
      var tools = section.querySelector(':scope > .ch-tools');
      if (tools) tools.after(toc); else section.insertBefore(toc, section.firstChild);
      toc.open = false;
    }
  }

  // Trang chủ và trang giới thiệu khoá: nút trên thanh trên dẫn tới bài đang đọc dở (của khoá đó,
  // hoặc của bất kỳ khoá nào nếu đang ở trang chủ), lọc lộ trình theo nhóm chương.
  function renderHome(section) {
    var course = courseById(courseOf(currentPath));
    var cta = document.getElementById('bar-cta');
    var last = load(course ? lastKey(course.id) : LAST_KEY);
    var lastLink = sideLink(last);
    var fallback = course ? (course.start || course.home) : '/ml/README_vi';
    cta.href = '#' + (lastLink ? last : fallback);
    cta.innerHTML = (lastLink ? 'Đọc tiếp' : 'Bắt đầu học') + ICON.arrow;
    if (lastLink) cta.title = lastLink.textContent;
    else cta.removeAttribute('title');

    // Thẻ khoá ở trang chủ: thêm lối "Đọc tiếp" nếu đã đọc dở khoá đó.
    section.querySelectorAll('.hub-resume[data-resume]').forEach(function (a) {
      var path = load(lastKey(a.dataset.resume));
      var link = sideLink(path);
      a.hidden = !link;
      if (!link) return;
      a.href = '#' + path;
      a.textContent = 'Đọc tiếp: ' + link.textContent.trim();
    });

    var done = readDone();
    section.querySelectorAll('.home-row').forEach(function (row) {
      var a = row.querySelector('.home-row-title');
      row.classList.toggle('is-done', !!a && done.indexOf(a.getAttribute('href').slice(1)) >= 0);
    });
    renderDashboard(section);
    bindBot(section);
    reveal(section);
    countUp(section);
    section.querySelectorAll('.home-tools > a, .home-banner, .home-agent, .hub-card').forEach(function (el) { el.classList.add('spot'); });
    if (window.EasyAgent) EasyAgent.mount(section.querySelector('.home-agent'));

    var filter = section.querySelector('.home-filter');
    if (!filter || filter.dataset.ready) return;
    filter.dataset.ready = '1';
    filter.addEventListener('click', function (e) {
      var btn = e.target.closest('button');
      if (!btn) return;
      var g = btn.dataset.g;
      filter.querySelectorAll('button').forEach(function (b) { b.setAttribute('aria-pressed', b === btn ? 'true' : 'false'); });
      section.querySelectorAll('.home-row').forEach(function (row) { row.hidden = g !== 'all' && row.dataset.g !== g; });
      // Ẩn tiêu đề nguồn (EasyRL / Zhao) khi không còn chương nào của nguồn đó hiện ra.
      section.querySelectorAll('.home-src').forEach(function (head) {
        var any = false;
        for (var el = head.nextElementSibling; el && !el.classList.contains('home-src'); el = el.nextElementSibling) if (!el.hidden) any = true;
        head.hidden = !any;
      });
    });
  }

  // Plugin KaTeX của docsify bỏ qua công thức nằm trong khối HTML (chú thích hình <div align=center>).
  // Vẽ nốt các $...$ còn lại trong chữ thường; chỉ áp dụng cho khoá Toán và Học máy.
  function renderStrayMath(section) {
    if (!window.katex) return;
    var walker = document.createTreeWalker(section, NodeFilter.SHOW_TEXT);
    var nodes = [], n;
    while ((n = walker.nextNode())) {
      if (n.parentElement.closest('pre, code, .katex, script, style')) continue;
      if (/\$[^$]+\$/.test(n.textContent)) nodes.push(n);
    }
    nodes.forEach(function (node) {
      var parts = node.textContent.split(/\$([^$]+)\$/);
      var frag = document.createDocumentFragment();
      parts.forEach(function (part, i) {
        if (i % 2 === 0) { if (part) frag.appendChild(document.createTextNode(part)); return; }
        var span = document.createElement('span');
        try { span.innerHTML = katex.renderToString(part, { throwOnError: false }); }
        catch (e) { span.textContent = '$' + part + '$'; }
        frag.appendChild(span);
      });
      node.parentNode.replaceChild(frag, node);
    });
  }

  function renderPage() {
    var section = document.querySelector('.markdown-section');
    if (!section) return;
    var link = Array.prototype.find.call(document.querySelectorAll('.sidebar-nav a[href^="#/"]'), function (a) {
      return hrefPath(a) === currentPath;
    }) || null;
    // Chờ cột trái có dữ liệu ở lần tải đầu; observer sẽ gọi lại.
    if (section.querySelector('.ch-tools, .pager')) return;
    enhanceSidebar();
    var course = courseOf(currentPath);
    if (course) setCourseView(course, true);
    if (course === 'ml' || course === 'math') renderStrayMath(section);
    var landing = !!section.querySelector('.home-hero, .hub-hero, .rm-page');
    document.body.classList.toggle('is-home', landing);
    document.body.classList.toggle('is-hub', !!section.querySelector('.hub-hero'));
    document.body.classList.remove('is-lesson');
    document.body.classList.toggle('is-wide', !!section.querySelector('.rm-page'));
    updateGlobalNav();
    if (landing) {
      document.body.classList.remove('has-toc', 'is-app');
      renderHome(section);
      revealInSidebar(link);
      return;
    }
    // Trang ứng dụng (bản đồ kiến thức): khung chiếm trọn vùng nội dung, không mục lục hay nút bài sau.
    var app = section.querySelector('iframe.app-frame');
    document.body.classList.toggle('is-app', !!app);
    if (app) {
      document.body.classList.remove('has-toc');
      var staleHead = document.querySelector('.lesson-head');
      if (staleHead) staleHead.innerHTML = '';
      revealInSidebar(link);
      if (!app.dataset.bound) {
        app.dataset.bound = '1';
        app.addEventListener('load', function () { syncFrameTheme(app); });
      }
      syncFrameTheme(app);
      return;
    }
    store(LAST_KEY, currentPath);
    if (course) store(lastKey(course), currentPath);
    document.body.classList.add('is-lesson');
    renderLessonHead(section, link, course);
    renderPageToc(section);
    renderChapterTools(section, link);
    renderRail(course, link);
    renderPanel(course, link);
    placeToc();
    renderPager(section, link);
    revealInSidebar(link);
    section.querySelectorAll('iframe.viz-frame').forEach(function (f) {
      f.addEventListener('load', function () { syncFrameTheme(f); });
      syncFrameTheme(f);
    });
  }

  // Nền đầu trang đổi lần lượt trời -> biển -> đáy biển -> đêm (ảnh pano-1..4).
  var SCENES = 4, sceneStep = 0, sceneTimer = null;
  function sceneLayer() {
    var box = el('div', 'scene-cycle');
    box.setAttribute('aria-hidden', 'true');
    for (var k = 0; k < SCENES; k++) {
      var d = el('div', k === sceneStep ? 'is-on' : '');
      d.style.backgroundImage = 'url("img/brand/pano-' + (k + 1) + '.jpg")';
      box.appendChild(d);
    }
    return box;
  }
  function applyScene() {
    document.querySelectorAll('.scene-cycle').forEach(function (box) {
      Array.prototype.forEach.call(box.children, function (d, k) { d.classList.toggle('is-on', k === sceneStep); });
      box.parentElement.setAttribute('data-scene', sceneStep + 1);
    });
  }
  function mountScenes() {
    var rm = document.querySelector('.rm-hero');
    if (rm && !rm.querySelector('.scene-cycle')) rm.prepend(sceneLayer());
    applyScene();
    if (sceneTimer || (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
    sceneTimer = setInterval(function () {
      if (document.hidden) return;
      sceneStep = (sceneStep + 1) % SCENES;
      applyScene();
    }, 7000);
  }

  // Logo đổi lần lượt sang các tư thế cá voi linh vật rồi quay về logo.
  // Ảnh phụ chỉ tải sau lần đổi đầu tiên; đứng yên nếu người dùng giảm chuyển động.
  var LOGO_FRAMES = ['logo', 'cap', 'read', 'love', 'sleep', 'bubble', 'hero'];
  var logoStep = 0, logoTimer = null;
  function logoFill(box) {
    if (box.children.length > 1) return;
    LOGO_FRAMES.slice(1).forEach(function (n) {
      var img = document.createElement('img');
      img.src = 'img/brand/mark-' + n + '.png';
      img.alt = '';
      box.appendChild(img);
    });
  }
  function logoTick() {
    if (document.hidden) return;
    logoStep = (logoStep + 1) % LOGO_FRAMES.length;
    document.querySelectorAll('.logo-cycle').forEach(function (box) {
      logoFill(box);
      Array.prototype.forEach.call(box.children, function (img, k) { img.classList.toggle('is-on', k === logoStep); });
    });
  }
  function startLogoCycle() {
    if (logoTimer || (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
    logoTimer = setInterval(logoTick, 3200);
    document.addEventListener('mouseover', function (e) {
      var a = e.target.closest && e.target.closest('.bar-brand, .app-name-link');
      if (!a || a.contains(e.relatedTarget)) return;
      clearInterval(logoTimer);
      logoTick();
      logoTimer = setInterval(logoTick, 3200);
    });
  }

  window.$docsify = window.$docsify || {};
  window.$docsify.plugins = (window.$docsify.plugins || []).concat(function (hook, vm) {
    hook.mounted(buildBar);
    hook.ready(startLogoCycle);
    hook.beforeEach(function (content) { rescueSearch(); return content; });
    hook.doneEach(function () {
      if (!document.querySelector('.search-host .search')) mountSearch();
      resetSearch();
      var nextPath = decodeURIComponent(vm.route.path || '/');
      var sec = document.querySelector('.markdown-section');
      if (sec && nextPath !== currentPath) {
        sec.classList.remove('page-enter');
        void sec.offsetWidth;
        sec.classList.add('page-enter');
      }
      currentPath = nextPath;
      setNav(false);
      enhanceSidebar();
      renderPage();
      mountScenes();
      placeSearch();
      window.dispatchEvent(new Event('scroll'));
    });
    hook.ready(function () {
      var nav = document.querySelector('.sidebar-nav');
      if (nav) {
        nav.addEventListener('click', function (e) {
          var btn = e.target.closest('.sub-toggle');
          if (!btn) return;
          e.preventDefault();
          e.stopPropagation();
          var li = btn.parentElement;
          var open = !li.classList.contains('is-open');
          manuallyOpen[hrefPath(li.querySelector(':scope > a'))] = open;
          li.classList.toggle('is-open', open);
          btn.setAttribute('aria-expanded', open ? 'true' : 'false');
        });
        new MutationObserver(function () {
          enhanceSidebar();
          renderPage();
        }).observe(nav, { childList: true });
      }
    });
  });
})();
