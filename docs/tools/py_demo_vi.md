# Code demo Python

> Ghi chú của người biên soạn: trang này chạy lại các script trong thư mục [`1-math_ml_basic/src/python`](https://github.com/harleyszhang/dl_note/tree/main/1-math_ml_basic/src/python) của *dl_note* (**Harley Zhang**, giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)): `python_basic.py`, `multi_cal_tasks.py`, `multi_io_tasks.py`, `conv_layer.py`, `gradio_demo.py`, `chatgpt_api_demo.py`. Đây cũng là code trong các ảnh chụp của bốn sơ đồ tư duy Python. Code được chia nhỏ theo chủ đề, chú thích dịch sang tiếng Việt, và sửa những chỗ chạy sai; mỗi chỗ sửa có ô **Biên tập**. Hai script còn lại của thư mục được dùng ở nơi khác: `gd_double_variable.py` ở [trang Code nền tảng toán](dlnote/math_code_vi), `thop_demo.py` và `classifynet_torch_to_onnx.py` ở phần triển khai mô hình của khoá Ghi chú học sâu.

## Cách dùng trang này

Các khối chạy lần lượt từ trên xuống, dùng chung biến, trên Python 3.11. Mục 1 đến 3 chỉ cần thư viện chuẩn; mục 4 cần `numpy`, mục 5 cần `torch`, mục 6 cần `gradio`. Kết quả in ra dưới mỗi khối là kết quả chạy thật trên một máy 16 luồng CPU; số đo thời gian sẽ khác trên máy bạn.

## 1. Generator

<small class="viz-ref">Sơ đồ: kiểu dữ liệu và tính năng nâng cao › Generator</small>

Generator sinh từng phần tử khi được hỏi, nên biểu diễn được dãy vô hạn (list thì không), và không cần giữ cả dãy trong bộ nhớ.

<!-- run -->
```python
import sys

def fibonacci():
    """Dãy Fibonacci vô hạn: mỗi lần yield trả về một số, trạng thái (prev, curr) được giữ lại."""
    prev, curr = 0, 1
    while True:
        yield curr
        prev, curr = curr, prev + curr

for i, fib in enumerate(fibonacci()):
    print(f"fib({i}) = {fib}", end="; ")
    if i >= 10:
        break
print()

squares_list = [i * i for i in range(1_000_000)]
squares_gen = (i * i for i in range(1_000_000))
print(f"list 1 triệu phần tử: {sys.getsizeof(squares_list):>9,} byte (chưa tính chính các số)")
print(f"generator tương ứng : {sys.getsizeof(squares_gen):>9,} byte")
print("tổng qua generator  :", sum(squares_gen))
```

<!-- output -->
```output
fib(0) = 1; fib(1) = 1; fib(2) = 2; fib(3) = 3; fib(4) = 5; fib(5) = 8; fib(6) = 13; fib(7) = 21; fib(8) = 34; fib(9) = 55; fib(10) = 89; 
list 1 triệu phần tử: 8,448,728 byte (chưa tính chính các số)
generator tương ứng :       208 byte
tổng qua generator  : 333332833333500000
```
<!-- /output -->

Ứng dụng điển hình là đọc tệp lớn từng dòng. Bản gốc chỉ để mã giả trong chuỗi chú thích; ở đây chạy với một tệp tạm:

<!-- run -->
```python
def read_large_file(file_path):
    with open(file_path, encoding="utf-8") as f:
        for line in f:          # đối tượng tệp vốn là iterator: mỗi lần đọc một dòng
            yield line.rstrip("\n")

with open("large_file.txt", "w", encoding="utf-8") as f:
    f.writelines(f"dòng {i}\n" for i in range(100_000))

n_lines = sum(1 for _ in read_large_file("large_file.txt"))
first = next(read_large_file("large_file.txt"))
print(n_lines, "dòng; dòng đầu:", repr(first))
```

<!-- output -->
```output
100000 dòng; dòng đầu: 'dòng 0'
```
<!-- /output -->

> **Biên tập:** Bản gốc đọc bằng vòng `while True: data = f.readline()`. Duyệt thẳng `for line in f` làm đúng việc đó và gọn hơn, vì đối tượng tệp đã là iterator.

## 2. Decorator

<small class="viz-ref">Sơ đồ: lập trình hàm › Decorator</small>

<!-- run -->
```python
import time
import functools

def timer(func):
    """Decorator: in thời gian chạy của hàm."""
    @functools.wraps(func)              # giữ tên, docstring của hàm gốc
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f"{func.__name__} chạy mất {time.perf_counter() - start:.6f} giây")
        return result
    return wrapper

@timer
def fib_list(n):
    """n số Fibonacci đầu tiên."""
    result, prev, curr = [], 0, 1
    for _ in range(n):
        result.append(curr)
        prev, curr = curr, prev + curr
    return result

r = fib_list(300)
print("số thứ 300 có", len(str(r[-1])), "chữ số;  fib_list.__name__ =", fib_list.__name__,
      "; docstring:", fib_list.__doc__)
```

<!-- output -->
```output
fib_list chạy mất 0.000039 giây
số thứ 300 có 63 chữ số;  fib_list.__name__ = fib_list ; docstring: n số Fibonacci đầu tiên.
```
<!-- /output -->

> **Biên tập:** Decorator `timer` của bản gốc thiếu `functools.wraps`, nên hàm sau khi bọc có tên `wrapper` và mất docstring (decorator `cache` ngay sau thì có). Bản gốc còn có decorator `runTime` gần như trùng `timer` (và viết sai chính tả `wapper`), ở đây bỏ. Đo thời gian nên dùng `time.perf_counter()` thay vì `time.time()`.

Decorator cache: lưu kết quả theo tham số, biến Fibonacci đệ quy từ hàm mũ thành tuyến tính.

<!-- run -->
```python
def cache(func):
    cached_results = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cached_results:
            cached_results[key] = func(*args, **kwargs)
        return cached_results[key]
    return wrapper

calls = 0
def fib_plain(n):
    global calls
    calls += 1
    return n if n < 2 else fib_plain(n - 1) + fib_plain(n - 2)

@cache
def fib_cached(n):
    global calls
    calls += 1
    return n if n < 2 else fib_cached(n - 1) + fib_cached(n - 2)

for f in (fib_plain, fib_cached):
    calls = 0
    print(f"{f.__name__}(25) = {f(25)}, số lần gọi hàm thật sự: {calls}")

@functools.lru_cache(maxsize=None)       # thư viện chuẩn đã có sẵn, nên dùng cái này
def fib_lru(n):
    return n if n < 2 else fib_lru(n - 1) + fib_lru(n - 2)
print("fib_lru(200) =", fib_lru(200), "|", fib_lru.cache_info())
```

<!-- output -->
```output
fib_plain(25) = 75025, số lần gọi hàm thật sự: 242785
fib_cached(25) = 75025, số lần gọi hàm thật sự: 26
fib_lru(200) = 280571172992510140037611932413038677189525 | CacheInfo(hits=198, misses=201, maxsize=None, currsize=201)
```
<!-- /output -->

Decorator truy cập được tham số của hàm gốc:

<!-- run -->
```python
def show_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("tham số:", args, kwargs)
        return func(*args, **kwargs)
    return wrapper

@show_args
def example_function(x, y):
    return x + y

print("kết quả:", example_function(2, y=3))
```

<!-- output -->
```output
tham số: (2,) {'y': 3}
kết quả: 5
```
<!-- /output -->

> **Biên tập:** Bản gốc in `args[0], args[1]`, nên báo `IndexError` khi hàm được gọi bằng tham số tên (`example_function(2, y=3)`); in cả `args` và `kwargs` thì luôn đúng.

## 3. Hướng đối tượng

<small class="viz-ref">Sơ đồ: hướng đối tượng cơ bản và nâng cao</small>

### 3.1 Kế thừa, đa hình, lớp trừu tượng

<!-- run -->
```python
import math
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self): ...
    @abstractmethod
    def perimeter(self): ...

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width, self.height = width, height
    def area(self):
        return self.width * self.height
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return math.pi * self.radius ** 2
    def perimeter(self):
        return 2 * math.pi * self.radius

def describe(shape: Shape):
    # đa hình: cùng một lời gọi, mỗi lớp tự cài đặt
    print(f"{type(shape).__name__:<9} diện tích {shape.area():7.2f}, chu vi {shape.perimeter():6.2f}")

for s in [Rectangle(5, 10), Circle(5)]:
    describe(s)

try:
    Shape()
except TypeError as e:
    print("Không tạo được lớp trừu tượng:", e)
```

<!-- output -->
```output
Rectangle diện tích   50.00, chu vi  30.00
Circle    diện tích   78.54, chu vi  31.42
Không tạo được lớp trừu tượng: Can't instantiate abstract class Shape with abstract methods area, perimeter
```
<!-- /output -->

> **Biên tập:** Bản gốc gán `self.__name__ = 'Rectangle'` trong `__init__` để in tên lớp; tên lớp đã có sẵn ở `type(obj).__name__`. Lớp `Shape` của bản gốc dùng `pass` nên vẫn tạo được thể hiện; dùng `ABC` và `@abstractmethod` (như sơ đồ OOP cơ bản giới thiệu) thì lớp con quên cài đặt sẽ báo lỗi ngay. Kết quả mẫu trong bản gốc thiếu dòng chu vi của hình tròn.

### 3.2 `@property`

<!-- run -->
```python
class Person:
    def __init__(self, age):
        self.age = age                     # đi qua setter, nên cũng được kiểm tra

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if not 0 <= value <= 150:
            raise ValueError(f"tuổi không hợp lệ: {value}")
        self._age = value

    @property
    def age_group(self):                   # thuộc tính tính toán, chỉ đọc
        if self._age < 18:
            return "dưới 18"
        return "18-64" if self._age < 65 else "từ 65"

p = Person(30)
print(p.age, p.age_group)
p.age = 70
print(p.age, p.age_group)
for bad in (-10, 200):
    try:
        p.age = bad
    except ValueError as e:
        print("ValueError:", e)
try:
    p.age_group = "x"
except AttributeError as e:
    print("AttributeError:", e)
```

<!-- output -->
```output
30 18-64
70 từ 65
ValueError: tuổi không hợp lệ: -10
ValueError: tuổi không hợp lệ: 200
AttributeError: property 'age_group' of 'Person' object has no setter
```
<!-- /output -->

> **Biên tập:** Bản gốc gán `self._age = age` thẳng trong `__init__` nên `Person(-5)` vẫn tạo được, và lưu sẵn `_age_group` ở hai nơi phải nhớ cập nhật cùng nhau. Gán qua `self.age` và tính `age_group` khi đọc thì không còn hai vấn đề đó.

### 3.3 Đa kế thừa và MRO

<!-- run -->
```python
class A:
    def method1(self): print("A.method1")

class B:
    def method1(self): print("B.method1")
    def method2(self): print("B.method2")

class C(A, B):
    def method3(self):
        super().method1()      # tìm theo MRO: C → A → B, gặp A.method1 trước
        super().method2()      # A không có method2, tìm tiếp tới B.method2
        B.method1(self)        # gọi thẳng lớp cha cụ thể
        print("C.method3")

C().method3()
print("MRO:", [k.__name__ for k in C.__mro__])

# Kế thừa kim cương: super() hợp tác làm Base.__init__ chỉ chạy một lần
class Base:
    def __init__(self): print("Base.__init__"); super().__init__()
class Left(Base):
    def __init__(self): print("Left.__init__"); super().__init__()
class Right(Base):
    def __init__(self): print("Right.__init__"); super().__init__()
class Diamond(Left, Right):
    def __init__(self): print("Diamond.__init__"); super().__init__()
Diamond()
print("MRO:", [k.__name__ for k in Diamond.__mro__])
```

<!-- output -->
```output
A.method1
B.method2
B.method1
C.method3
MRO: ['C', 'A', 'B', 'object']
Diamond.__init__
Left.__init__
Right.__init__
Base.__init__
MRO: ['Diamond', 'Left', 'Right', 'Base', 'object']
```
<!-- /output -->

### 3.4 Tuỳ biến lớp: kiểu chứa

<!-- run -->
```python
class IntList:
    def __init__(self, data):
        self._data = list(data)
    def __len__(self):
        return len(self._data)
    def __getitem__(self, index):
        if isinstance(index, slice):
            return IntList(self._data[index])
        return self._data[index]
    def __setitem__(self, index, value):
        if isinstance(index, slice):
            self._data[index] = value._data if isinstance(value, IntList) else list(value)
        elif isinstance(value, int):
            self._data[index] = value
        else:
            raise TypeError("IntList chỉ chứa int")
    def __delitem__(self, index):
        del self._data[index]
    def __contains__(self, item):
        return item in self._data
    def __iter__(self):
        return iter(self._data)
    def __repr__(self):
        return f"IntList({self._data})"

xs = IntList([1, 2, 3, 4, 5])
print(len(xs), xs[0], xs[1:4])
xs[0] = 10; xs[1:4] = IntList([20, 30]); print(xs)
del xs[1]; print(xs, 30 in xs, list(xs))
try:
    xs[0] = "a"
except TypeError as e:
    print("TypeError:", e)

print("list có __iter__:", hasattr([], "__iter__"), "| list có __next__:", hasattr([], "__next__"),
      "| iter(list) có __next__:", hasattr(iter([]), "__next__"))
```

<!-- output -->
```output
5 1 IntList([2, 3, 4])
IntList([10, 20, 30, 5])
IntList([10, 30, 5]) True [10, 30, 5]
TypeError: IntList chỉ chứa int
list có __iter__: True | list có __next__: False | iter(list) có __next__: True
```
<!-- /output -->

> **Biên tập:** `__setitem__` của bản gốc lặng lẽ bỏ qua giá trị sai kiểu (`xs[0] = "a"` không làm gì, không báo lỗi). Dòng cuối kiểm tra ô Biên tập của sơ đồ OOP nâng cao: `list` chỉ có `__iter__`, còn `__next__` nằm ở iterator mà `iter()` trả về.

### 3.5 Metaclass và Singleton

<!-- run -->
```python
import datetime

class Meta(type):
    def __init__(cls, name, bases, attrs):
        cls.created_at = datetime.datetime.now()     # chạy lúc *tạo lớp*, không phải lúc tạo thể hiện
        super().__init__(name, bases, attrs)

class MyClass(metaclass=Meta):
    @property
    def name(self):
        return "MyClass"

print(MyClass().name, "được tạo lúc", MyClass.created_at.strftime("%H:%M:%S"), "| type(MyClass) =", type(MyClass).__name__)

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):              # chạy khi gọi MyService(...)
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class MyService(metaclass=Singleton):
    def __init__(self, x):
        self.x = x

a, b = MyService(1), MyService(2)
print(a.x, b.x, a is b)        # lần gọi thứ hai trả về đúng đối tượng cũ, x vẫn là 1
```

<!-- output -->
```output
MyClass được tạo lúc 09:29:08 | type(MyClass) = Meta
1 1 True
```
<!-- /output -->

### 3.6 staticmethod, classmethod, dataclass, total_ordering

<!-- run -->
```python
from dataclasses import dataclass, field
from functools import total_ordering

class StringUtils:
    @staticmethod
    def reverse(s):                        # không cần self hay cls
        return s[::-1]
    @classmethod
    def from_words(cls, *words):           # phương thức nhà máy: tạo thể hiện theo cách khác
        return cls(" ".join(words))
    def __init__(self, text):
        self.text = text
    def reversed_text(self):
        return self.reverse(self.text)

print(StringUtils.reverse("Hello, world!"), "|", StringUtils.from_words("Hello,", "world!").reversed_text())

@dataclass(order=True)
class Student:
    age: int
    name: str = field(compare=False)       # so sánh theo tuổi, bỏ qua tên

print(Student(20, "Lan"), Student(20, "Lan") == Student(20, "Minh"), Student(20, "Lan") < Student(30, "Hoa"))

@total_ordering
class Version:
    def __init__(self, s):
        self.parts = tuple(int(x) for x in s.split("."))
    def __eq__(self, other):
        return self.parts == other.parts
    def __lt__(self, other):               # chỉ cần __eq__ và __lt__, total_ordering sinh <=, >, >=
        return self.parts < other.parts

print(Version("2.10") > Version("2.9"), Version("1.0") <= Version("1.0"), "| so chuỗi:", "2.10" > "2.9")
```

<!-- output -->
```output
!dlrow ,olleH | !dlrow ,olleH
Student(age=20, name='Lan') True True
True True | so chuỗi: False
```
<!-- /output -->

> **Biên tập:** Bản gốc định nghĩa lớp dataclass tên `Person`, đè mất lớp `Person` ở mục `@property`, và dùng `@total_ordering` cho một lớp chỉ so theo tuổi, việc mà `@dataclass(order=True)` làm sẵn. Ví dụ `Version` ở trên cho thấy chỗ `total_ordering` thực sự có ích: so sánh tuỳ ý mà chỉ phải viết hai phương thức. Hàm `get_dict_depth` cuối tệp gốc không liên quan tới hướng đối tượng nên không chép lại.

## 4. Đa luồng, đa tiến trình

Script `multi_cal_tasks.py` so sánh nhân ma trận 200×200 bằng ba vòng lặp Python khi chạy một luồng, nhiều luồng và nhiều tiến trình. Bản gốc có hai lỗi khiến phép so sánh sai:

- Bản đa tiến trình truyền một **hàm lồng** (`multiply` định nghĩa trong hàm) cho `Pool.apply_async`. Hàm lồng không pickle được, nên mọi tác vụ lỗi; lỗi bị nuốt vì không ai gọi `.get()`. Thời gian đo được chỉ là thời gian tạo pool.
- Kể cả khi chạy được, mỗi tiến trình con ghi vào **bản sao** của ma trận `C`, nên `C` ở tiến trình chính vẫn toàn số 0.

Ta tái hiện lỗi rồi sửa. Hàm làm việc đặt trong một module riêng (`matmul_worker.py`) để tiến trình con import được; trong một script bình thường, chỉ cần định nghĩa hàm ở cấp module.

<!-- run -->
```python
import os, time, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

N_WORKERS = os.cpu_count()
rng = np.random.default_rng(0)
N = 200                                # như bản gốc
A, B = rng.random((N, N)), rng.random((N, N))

def matmul_rows(A, B, i_start, i_end, C):
    for i in range(i_start, i_end):
        for j in range(B.shape[1]):
            s = 0.0
            for k in range(A.shape[1]):
                s += A[i, k] * B[k, j]
            C[i, j] = s

def chunks(n, parts):
    step = -(-n // parts)
    return [(i, min(i + step, n)) for i in range(0, n, step)]

t = time.perf_counter(); C1 = np.zeros((N, N)); matmul_rows(A, B, 0, N, C1); t1 = time.perf_counter() - t

t = time.perf_counter(); C2 = np.zeros((N, N))
with ThreadPoolExecutor(N_WORKERS) as ex:
    for f in [ex.submit(matmul_rows, A, B, i0, i1, C2) for i0, i1 in chunks(N, N_WORKERS)]:
        f.result()
t2 = time.perf_counter() - t

# Cách của bản gốc: hàm lồng + apply_async không lấy kết quả
def buggy_multiprocess(A, B, n_proc):
    C = np.zeros((A.shape[0], B.shape[1]))
    def multiply(i_start, i_end):
        matmul_rows(A, B, i_start, i_end, C)
    with mp.Pool(n_proc) as p:
        results = [p.apply_async(multiply, (i0, i1)) for i0, i1 in chunks(A.shape[0], n_proc)]
        p.close(); p.join()
    return C, results

t = time.perf_counter(); C3, results = buggy_multiprocess(A, B, N_WORKERS); t3 = time.perf_counter() - t
try:
    results[0].get()
except Exception as e:
    err = f"{type(e).__name__}: {e}"

print(f"1 luồng            : {t1:6.3f} s")
print(f"{N_WORKERS} luồng           : {t2:6.3f} s   (GIL: chỉ một luồng chạy bytecode Python tại một thời điểm)")
print(f"{N_WORKERS} tiến trình (gốc): {t3:6.3f} s   nhưng C toàn số 0: {not C3.any()}")
print("lỗi bị nuốt:", err[:90])
```

<!-- output -->
```output
1 luồng            :  2.527 s
16 luồng           :  5.138 s   (GIL: chỉ một luồng chạy bytecode Python tại một thời điểm)
16 tiến trình (gốc):  0.108 s   nhưng C toàn số 0: True
lỗi bị nuốt: AttributeError: Can't pickle local object 'buggy_multiprocess.<locals>.multiply'
```
<!-- /output -->

Bản sửa: hàm làm việc ở cấp module, **trả về** các hàng đã tính thay vì ghi vào biến dùng chung, tiến trình chính ghép lại.

<!-- run -->
```python
from pathlib import Path
import importlib, sys

Path("matmul_worker.py").write_text('''
import numpy as np

def rows(args):
    """Tính các hàng i_start..i_end-1 của A @ B bằng ba vòng lặp Python và trả về chúng."""
    A, B, i_start, i_end = args
    out = np.zeros((i_end - i_start, B.shape[1]))
    for i in range(i_start, i_end):
        for j in range(B.shape[1]):
            s = 0.0
            for k in range(A.shape[1]):
                s += A[i, k] * B[k, j]
            out[i - i_start, j] = s
    return i_start, out
''')
sys.path.insert(0, os.getcwd())
matmul_worker = importlib.import_module("matmul_worker")

t = time.perf_counter()
C4 = np.zeros((N, N))
with mp.Pool(N_WORKERS) as p:
    for i0, block in p.map(matmul_worker.rows, [(A, B, i0, i1) for i0, i1 in chunks(N, N_WORKERS)]):
        C4[i0:i0 + len(block)] = block
t4 = time.perf_counter() - t

import timeit
C5 = A @ B
t5 = min(timeit.repeat(lambda: A @ B, number=100, repeat=5)) / 100   # NumPy quá nhanh: đo trung bình 100 lần
print(f"{N_WORKERS} tiến trình (sửa): {t4:6.3f} s  (nhanh hơn 1 luồng {t1 / t4:.1f} lần), đúng: {np.allclose(C4, A @ B)}")
print(f"NumPy A @ B        : {t5 * 1e6:6.1f} µs (nhanh hơn 1 luồng Python {t1 / t5:,.0f} lần)")
print("ba vòng lặp 1 luồng đúng:", np.allclose(C1, C5), "| đa luồng đúng:", np.allclose(C2, C5))
```

<!-- output -->
```output
16 tiến trình (sửa):  0.603 s  (nhanh hơn 1 luồng 4.2 lần), đúng: True
NumPy A @ B        :  510.1 µs (nhanh hơn 1 luồng Python 4,953 lần)
ba vòng lặp 1 luồng đúng: True | đa luồng đúng: True
```
<!-- /output -->

Kết quả cho thấy ba điều. Đa luồng không nhanh hơn mà còn chậm hơn một luồng, vì GIL chỉ cho một luồng chạy bytecode Python tại một thời điểm và các luồng còn tranh nhau khoá. Đa tiến trình (bản sửa) tăng tốc thật, khoảng 4 lần, nhưng còn xa 16 lần vì phải trả chi phí tạo tiến trình và chép ma trận sang từng tiến trình. Bài học lớn nhất ở dòng NumPy: phép nhân ma trận nên giao cho thư viện (BLAS), nhanh hơn khoảng 5000 lần.

Với tác vụ chờ I/O (tải tệp, đọc mạng), đa luồng lại có ích vì luồng nhả GIL khi chờ. Script `multi_io_tasks.py` của bản gốc tải 10 ảnh từ picsum.photos; để kết quả không phụ thuộc mạng, ở đây thay việc tải bằng `time.sleep(0.2)` (cũng nhả GIL như khi chờ mạng):

<!-- run -->
```python
def fake_download(url):
    time.sleep(0.2)                       # giả lập chờ mạng 200 ms
    return len(url)

URLS = [f"https://picsum.photos/id/{i}/200/300" for i in range(1, 11)]

t = time.perf_counter(); [fake_download(u) for u in URLS]; s1 = time.perf_counter() - t
t = time.perf_counter()
with ThreadPoolExecutor(10) as ex:
    list(ex.map(fake_download, URLS))
s2 = time.perf_counter() - t
print(f"tuần tự : {s1:.2f} s\n10 luồng: {s2:.2f} s")
```

<!-- output -->
```output
tuần tự : 2.00 s
10 luồng: 0.21 s
```
<!-- /output -->

> **Biên tập:** Trong `multi_io_tasks.py`, hàm `thread_pool_download` đọc `work_queue.empty()` rồi mới `get()`; cách viết này vẫn chạy vì có `get(block=False)` bắt `queue.Empty`, nhưng `concurrent.futures.ThreadPoolExecutor` như trên làm cùng việc trong ba dòng. Để chọn giữa luồng và tiến trình: chờ I/O thì dùng luồng (hoặc `asyncio`); tính toán Python thuần thì dùng tiến trình; tính toán đã nằm trong NumPy/PyTorch thì thư viện tự dùng nhiều lõi, thường không cần làm gì thêm.

## 5. Tích chập: vòng lặp NumPy và PyTorch

`conv_layer.py` cài phép tích chập 2D bằng vòng lặp NumPy rồi so thời gian với `torch.nn.Conv2d`. Bản gốc không chạy được: giữa tệp có một đoạn sáu vòng lặp dùng các biến chưa định nghĩa (`batch_size`, `output`, `oh`...), báo `NameError` ngay. Ngoài ra hai phiên bản dùng trọng số ngẫu nhiên khác nhau, nên không kiểm tra được chúng cho cùng kết quả. Bản sửa dưới bỏ đoạn hỏng, chép trọng số từ PyTorch sang NumPy, và giảm kích thước (batch 4, ảnh 64×64) để vòng lặp chạy trong vài giây.

<!-- run -->
```python
import torch

class Conv2DNumpy:
    """Tích chập stride 1, padding giữ nguyên kích thước (kernel lẻ)."""
    def __init__(self, weight, bias):
        self.weight, self.bias = weight, bias          # (C_out, C_in, k, k), (C_out,)
        self.k = weight.shape[-1]

    def forward(self, x):
        n, c_in, h, w = x.shape
        p = self.k // 2
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        out = np.zeros((n, self.weight.shape[0], h, w))
        for b in range(n):
            for oc in range(self.weight.shape[0]):
                for r in range(h):
                    for c in range(w):
                        out[b, oc, r, c] = np.sum(xp[b, :, r:r + self.k, c:c + self.k] * self.weight[oc]) + self.bias[oc]
        return out

torch.manual_seed(0)
conv = torch.nn.Conv2d(3, 16, 3, padding=1)
x = torch.randn(4, 3, 64, 64)
conv_np = Conv2DNumpy(conv.weight.detach().numpy(), conv.bias.detach().numpy())

t = time.perf_counter(); y_np = conv_np.forward(x.numpy()); t_np = time.perf_counter() - t
with torch.no_grad():
    conv(x)                                            # chạy làm nóng một lần
    t = time.perf_counter(); y_t = conv(x); t_t = time.perf_counter() - t
print("kích thước đầu ra:", y_np.shape, tuple(y_t.shape))
print("sai khác lớn nhất:", float(np.abs(y_np - y_t.numpy()).max()))
print(f"NumPy vòng lặp: {t_np:.3f} s | PyTorch: {t_t * 1e3:.2f} ms | nhanh hơn {t_np / t_t:,.0f} lần")
```

<!-- output -->
```output
kích thước đầu ra: (4, 16, 64, 64) (4, 16, 64, 64)
sai khác lớn nhất: 7.152557373046875e-07
NumPy vòng lặp: 1.327 s | PyTorch: 0.47 ms | nhanh hơn 2,817 lần
```
<!-- /output -->

Hai cách cho cùng kết quả (sai khác cỡ $10^{-6}$ do làm tròn float32); khác biệt hàng nghìn lần về tốc độ đến từ việc PyTorch đưa tích chập về phép nhân ma trận lớn và chạy trên nhiều lõi bằng code C++ tối ưu.

## 6. Gradio và API mô hình ngôn ngữ

Hai script cuối cần môi trường tương tác (trình duyệt, khoá API) nên trang chỉ kiểm tra code dựng được, không chạy giao diện hay gọi API.

`gradio_demo.py` dựng một khung chat trả lời ngẫu nhiên "Yes"/"No". Bản gốc dùng định dạng lịch sử cũ (danh sách cặp `[user, bot]`), đã bị bỏ trong Gradio 5–6; bản dưới dùng định dạng tin nhắn `{"role", "content"}` như API chat hiện nay. Chạy `demo.launch()` trên máy mình để mở giao diện.

<!-- run -->
```python
import random
import gradio as gr

def user(message, history):
    return "", history + [{"role": "user", "content": message}]

def bot(history):
    time.sleep(0.5)
    return history + [{"role": "assistant", "content": random.choice(["Yes", "No"])}]

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.Button("Xoá")
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: [], None, chatbot, queue=False)

# demo.launch()   # mở http://127.0.0.1:7860 trên máy bạn
random.seed(1)
_, h = user("Hôm nay trời có mưa không?", [])
print("gradio", gr.__version__, "| dựng giao diện xong; thử hai hàm xử lý:", bot(h))
```

<!-- output -->
```output
gradio 6.30.0 | dựng giao diện xong; thử hai hàm xử lý: [{'role': 'user', 'content': 'Hôm nay trời có mưa không?'}, {'role': 'assistant', 'content': 'Yes'}]
```
<!-- /output -->

`chatgpt_api_demo.py` gọi `openai.Completion.create(engine="text-davinci-003", ...)`. Cả hàm này (bị bỏ từ thư viện `openai` 1.0, tháng 11/2023) lẫn mô hình `text-davinci-003` (ngừng phục vụ tháng 1/2024) đều không còn dùng được. Cách gọi hiện nay, với thư viện `openai` 1.x/2.x:

```python
from openai import OpenAI

client = OpenAI()                     # đọc khoá từ biến môi trường OPENAI_API_KEY, đừng viết khoá vào code
response = client.responses.create(
    model="gpt-4o-mini",              # thay bằng mô hình có trong tài khoản của bạn
    input="Hướng dẫn từng bước cách tạo khoá API trên trang OpenAI.",
    max_output_tokens=1024,
    temperature=0.5,
)
print(response.output_text)
```

> **Biên tập:** Bản gốc gán khoá API thẳng trong code (`openai.api_key = "YOUR_API_KEY"`). Khoá viết trong code rất dễ lọt lên GitHub; nên để trong biến môi trường hoặc tệp `.env` không đưa vào git. Ngoài API `responses`, API `client.chat.completions.create(model=..., messages=[...])` vẫn được hỗ trợ và là định dạng mà nhiều nhà cung cấp khác (và các máy chủ mô hình mở như vLLM) cũng dùng.

## Tóm tắt

- Generator và `functools.lru_cache` thay được phần lớn code tự viết trong `python_basic.py`; decorator tự viết nên có `functools.wraps`.
- Python không có biến riêng, không có nạp chồng hàm, và `list` không có `__next__`: ba chỗ sơ đồ gốc viết sai, kiểm tra được bằng vài dòng code.
- Bản đa tiến trình của `multi_cal_tasks.py` không làm gì cả (lỗi pickle bị nuốt, kết quả toàn 0); sửa xong nó nhanh hơn một luồng khoảng 4 lần với 16 tiến trình, còn NumPy nhanh hơn khoảng 5000 lần. Đa luồng làm tác vụ tính toán chậm đi gấp đôi; nó chỉ có ích khi chờ I/O (10 luồng: 2 s còn 0,2 s).
- `conv_layer.py` không chạy được ở bản gốc; bản sửa cho cùng kết quả với `nn.Conv2d` nhưng chậm hơn hàng nghìn lần.
- API `openai.Completion` và mô hình `text-davinci-003` đã bị bỏ; định dạng lịch sử dạng cặp của Gradio cũng vậy.
