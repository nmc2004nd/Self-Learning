"""Chuyển sơ đồ MindNode của Daniel Martinez thành JSON cho trang docs/mindmap/ml.html,
kèm ảnh công thức nén sang WebP.
python convert_mindnode.py "<path>/Data Science.mindnode" docs/mindmap ml "Machine Learning"
python convert_mindnode.py "<path>/Deep Learning.mindnode" docs/mindmap dl "Deep Learning"
"""
import sys, os, re, json, html, plistlib
from PIL import Image

src, out = sys.argv[1], sys.argv[2]
key = sys.argv[3] if len(sys.argv) > 3 else 'ml'
root_name = sys.argv[4] if len(sys.argv) > 4 else 'Machine Learning'
os.makedirs(f'{out}/img', exist_ok=True)

# Nút trùng tên một bài trên site -> thêm liên kết "đọc bài".
LINKS = {
    'Principal Component Analysis (PCA)': 'ml/ch21_vi',
    'Singular Value Decomposition (SVD)': 'ml/ch20_vi',
    'Linear Discriminant Analysis (LDA)': 'ml/ch22_vi',
    'Linear Discriminant Analysis': 'ml/ch22_vi',
    'Linear Regression': 'ml/ch07_vi',
    'Logistic Regression': 'ml/ch14_vi',
    'k-nearest Neighbour (kNN)': 'ml/ch09_vi',
    'k-Means': 'ml/ch10_vi',
    'Naive Bayes': 'ml/ch11_vi',
    'Overfitting': 'ml/ch08_vi',
    'Underfitting': 'ml/ch08_vi',
    'Gradient Descent': 'ml/ch12_vi',
    'Stochastic Gradient Descent (SGD)': 'ml/ch12_vi',
    'Mini-batch Stochastic Gradient Descent (SGD)': 'ml/ch12_vi',
    'Momentum': 'ml/ch12_vi',
    'Backpropagation': 'ml/ch16_vi',
    'Softmax': 'ml/ch15_vi',
    'Support vector machines': 'ml/ch26_vi',
    'Hinge Loss': 'ml/ch27_vi',
    'Feature Engineering': 'ml/ch06_vi',
    'Reinforcement Learning': 'rl_vi',
    'Reinforcement learning': 'rl_vi',
    'Maximum Likelihood Estimation (MLE)': 'math/ch04_vi',
    'Probability': 'math/ch03_vi',
    'Linear Algebra': 'math/ch01_vi',
    'Eigenvectors and Eigenvalues': 'math/ch01_vi',
    'Derivatives Chain Rule': 'math/ch02_vi',
    'Jacobian Matrix': 'math/ch02_vi',
    'Gradient': 'math/ch02_vi',
    'Performance Analysis': 'ml/eval_vi',
    'Confusion Matrix': 'ml/eval_vi',
    'Accuracy': 'ml/eval_vi',
    'f1 score': 'ml/eval_vi',
    'Precision': 'ml/eval_vi',
    'Recall': 'ml/eval_vi',
    'ROC Curve - Receiver Operating Characteristics': 'ml/eval_vi',
    'Bias-Variance Tradeoff': 'ml/eval_vi',
    'Mean Squared Error (MSE)': 'ml/eval_vi',
    'Cross-validation': 'ml/ch08_vi',
    'Cross Validation': 'ml/ch08_vi',
    'Validation Dataset': 'ml/ch08_vi',
    'Decision Tree': 'ml/tree_vi',
    'Decision tree learning': 'ml/tree_vi',
    'Classification and Regression Tree (CART)': 'ml/tree_vi',
    'Entropy': 'ml/tree_vi',
    'Random Forest': 'ml/ensemble_vi',
    'Bagging': 'ml/ensemble_vi',
    'Gradient Boosting Machines (GBM)': 'ml/ensemble_vi',
    'Gradient Boosted Regression Trees (GBRT)': 'ml/ensemble_vi',
    'Expectation Maximization': 'ml/cluster_vi',
    'DBSCAN': 'ml/cluster_vi',
    'Hierarchical Clustering': 'ml/cluster_vi',
    'Silhouette Width': 'ml/cluster_vi',
    'How many clusters do we select?': 'ml/cluster_vi',
    'Data Processing': 'ml/prep_vi',
    'Feature Selection': 'ml/prep_vi',
    'Feature Encoding': 'ml/prep_vi',
    'Label Encoding': 'ml/prep_vi',
    'One Hot Encoding': 'ml/prep_vi',
    'Feature Normalisation or Scaling': 'ml/prep_vi',
    'Rescaling': 'ml/prep_vi',
    'Standardization': 'ml/prep_vi',
    'Scaling to unit length': 'ml/prep_vi',
    'Feature Imputation': 'ml/prep_vi',
    'Mean-substitution': 'ml/prep_vi',
    'Feature Cleaning': 'ml/prep_vi',
    'Missing values': 'ml/prep_vi',
    'Outliers': 'ml/prep_vi',
    'Data Exploration': 'ml/prep_vi',
    'Data Types': 'ml/prep_vi',
    'Wrapper Methods': 'ml/prep_vi',
    'Filter Methods': 'ml/prep_vi',
    'Embedded Methods': 'ml/prep_vi',
    'Recursive Feature Ellimination': 'ml/prep_vi',
    'Forward Selection': 'ml/prep_vi',
    'Backward Elimination': 'ml/prep_vi',
    'Chi-Square': 'ml/prep_vi',
    'Discretization': 'ml/prep_vi',
}
# Bản dịch tiếng Việt cho các đoạn giải thích dài (khoá là câu gốc tiếng Anh).
VI = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vi_notes.json'), encoding='utf-8'))
ORDER = ['Machine Learning Concepts', 'Machine Learning Process', 'Machine Learning Data Processing',
         'Machine Learning Mathematics', 'Machine Learning Models']

def text(n):
    t = n.get('title', {}).get('text', '')
    t = re.sub(r'</p>\s*<p[^>]*>|<br\s*/?>', '\n', t)
    t = html.unescape(re.sub(r'<[^>]+>', '', t))
    t = t.replace(' ', '\n').replace('\xa0', ' ')
    return '\n'.join(re.sub(r'[ \t]+', ' ', l).strip() for l in t.split('\n')).strip()

used = set()
def image(a):
    name = a['fileName']; stem = name.rsplit('.', 1)[0][:8].lower()
    w, h = [float(v) for v in re.findall(r'[\d.]+', a['size'])]
    im = Image.open(f'{src}/resources/{name}').convert('RGB')
    tw = min(im.width, round(w * 2))
    if tw < im.width: im = im.resize((tw, round(im.height * tw / im.width)), Image.LANCZOS)
    if stem not in used:
        im.save(f'{out}/img/{stem}.webp', quality=82, method=6); used.add(stem)
    return [stem, round(w), round(h)]

def node(n, depth=1):
    d = {'t': text(n)}
    if re.fullmatch(r'https?://\S+', d['t']): d['u'] = d['t']
    if d['t'] in LINKS: d['l'] = LINKS[d['t']]
    if VI.get(d['t'], d['t']) != d['t']: d['v'] = VI[d['t']]
    if 'attachment' in n and depth >= 2: d['i'] = image(n['attachment'])  # bỏ logo ở nút lớn
    kids = [node(c, depth + 1) for c in n.get('subnodes', [])]
    if kids: d['c'] = kids
    return d

doc = plistlib.load(open(f'{src}/contents.xml', 'rb'))
maps = [node(m['mainNode']) for m in doc['canvas']['mindMaps']]
maps.sort(key=lambda m: ORDER.index(m['t']) if m['t'] in ORDER else 99)
for m in maps: m['t'] = m['t'].replace('Machine Learning ', '')
root = {'t': root_name, 'c': maps}
json.dump(root, open(f'{out}/{key}-mindmap.json', 'w'), ensure_ascii=False, separators=(',', ':'))
cnt = lambda n: 1 + sum(cnt(c) for c in n.get('c', []))
print(cnt(root), 'nodes,', len(used), 'images,', sum(1 for _ in re.finditer('"l":', json.dumps(root))), 'site links')
