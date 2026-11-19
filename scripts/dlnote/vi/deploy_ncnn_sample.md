# Triển khai mô hình: phân tích mã nguồn ncnn, chạy ví dụ mẫu

[ncnn](https://github.com/Tencent/ncnn) là framework suy luận mạng neuron hiệu năng cao của Tencent, tối ưu cho CPU di động (ARM), viết bằng C++ không phụ thuộc thư viện ngoài, có cả backend Vulkan cho GPU. Bài này đi qua các kiến thức nền cần để biên dịch ncnn, rồi phân tích ví dụ phân loại ảnh `shufflenetv2.cpp`. Trang [Code](dlnote/deploy_code_vi) đi kèm (mục 5) làm lại ví dụ đó bằng gói Python `ncnn`: chuyển ShuffleNet v2 sang ncnn bằng pnnx, đọc tệp `.param`, chạy bằng `Net`/`Extractor` và so với PyTorch.

<!-- theory: qa/ch17_vi#17.10.1 -->

## 1 Kiến thức nhanh về các thư viện phụ thuộc

### aarch64

`aarch64`, còn gọi là ARM64, là kiến trúc tập lệnh 64 bit dựa trên `ARMv8-A`. Đây là thế hệ mới của kiến trúc ARM, nhằm cho hiệu năng và hiệu suất năng lượng tốt hơn. So với kiến trúc `ARM` 32 bit trước đó, aarch64 có không gian địa chỉ lớn hơn, nhiều thanh ghi hơn và hiệu năng số thực tốt hơn.

Trong terminal Linux, gõ lệnh sau để xem kiến trúc `cpu`:

```bash
uname -m  # máy chủ Intel của tác giả in x86_64, máy Mac M1 Pro in arm64
```

### OpenMP

`OpenMP` (Open Multi-Processing) là API lập trình song song trên bộ nhớ dùng chung, dùng để viết chương trình đa luồng. Với `OpenMP`, lập trình viên **chèn chỉ thị** vào chương trình để chỉ ra chỗ có thể song song. Các chỉ thị bắt đầu bằng `#pragma`, cho trình biên dịch biết cách song song hoá đoạn code.

```cpp
#include <stdio.h>
#include <omp.h>

int main() {
    int i;
    #pragma omp parallel for
    for(i = 0; i < 10; i++) {
        printf("Thread %d executing iteration %d\n", omp_get_thread_num(), i);
    }
    return 0;
}
```

ncnn dùng OpenMP để chia việc của mỗi tầng cho nhiều luồng; số luồng đặt bằng `net.opt.num_threads`.

### AVX512

`AVX` (Advanced Vector Extensions, phần mở rộng vector nâng cao) là tập lệnh xử lý dữ liệu nhiều chiều, ví dụ vector tối đa 8 số thực 64 bit hoặc tối đa 16 số thực 32 bit (với AVX-512).

`AVX512` là lệnh `SIMD` (một lệnh, nhiều dữ liệu). Trên kiến trúc `x86`, lệnh SIMD đầu tiên là `SSE` 128 bit, rồi tới AVX/AVX2 256 bit, và nay là AVX-512 512 bit.

### submodule

Git submodule (mô-đun con) cho phép đặt một kho Git làm thư mục con của một kho Git khác: clone một kho khác vào dự án của mình mà các commit vẫn độc lập. ncnn dùng submodule cho glslang (biên dịch shader Vulkan) và pybind11, nên khi clone phải chạy thêm `git submodule update --init`.

### apt upgrade

- `apt update`: chỉ cập nhật danh sách gói, kiểm tra xem các gói đã cài có bản mới không, không cài gì.
- `apt upgrade`: nâng cấp các gói đã cài.

## 2 Kiến thức nhanh về phần cứng

### 2.1 Bộ nhớ

Hai đặc tính quan trọng của `RAM` (bộ nhớ truy cập ngẫu nhiên) là băng thông (`bandwidth`) và độ trễ (`latency`).

### 2.2 CPU

Bộ xử lý trung tâm (central processing unit, `CPU`) là lõi của mọi máy tính, gồm các thành phần chính:

- **lõi xử lý** (processor cores): thực thi mã máy;
- **bus**: nối các thành phần (topo của bus khác nhau rõ rệt giữa các dòng, các thế hệ và các hãng);
- **bộ nhớ đệm** (cache): thường có ba cấp (L1/L2/L3), băng thông đọc cao hơn và độ trễ thấp hơn bộ nhớ chính.

CPU hiện đại đều có đơn vị xử lý vector, cung cấp lệnh `SIMD` để xử lý nhiều dữ liệu trong một lệnh, phục vụ đại số tuyến tính và tích chập hiệu năng cao. Các lệnh `SIMD` này có tên khác nhau: trên ARM là NEON, trên x86 là AVX2.

> **Biên tập:** Bản gốc ghi "AVX2156", do chép cả số chú thích 156 trong sách *Dive into Deep Learning* bản tiếng Trung, nơi đoạn này và hình dưới được lấy ra (mục 12.4 Phần cứng).

Kiến trúc lõi của một CPU Intel Skylake bốn lõi dòng tiêu dùng điển hình như hình dưới.

![Kiến trúc CPU Intel Skylake bốn lõi](../images/ncnn/cpu_architecture.png)

Hình 2.1: CPU Intel Skylake bốn lõi dòng tiêu dùng (từ *Dive into Deep Learning*, hình 12.4.3).

## 3 Suy luận mô hình bằng ncnn

### 3.1 Phân tích ví dụ suy luận ShuffleNet v2

Lấy mạng phân loại ShuffleNet v2 làm ví dụ để xem cách suy luận bằng `ncnn`. Mã nguồn ở tệp `ncnn/examples/shufflenetv2.cpp`, gồm hai hàm chính: `detect_shufflenetv2()` chạy mạng phân loại ảnh, và `print_topk()` in ra N lớp có điểm cao nhất. Luồng code tóm tắt như sau:

1. Trong `detect_shufflenetv2`, lớp `ncnn::Net` được dùng để nạp và chạy mô hình:
   - nạp tệp tham số cấu trúc (`.param`) và tệp trọng số (`.bin`);
   - chuyển ảnh đầu vào từ `cv::Mat` sang `ncnn::Mat`, đồng thời resize và chuẩn hoá;
   - tạo đối tượng `ncnn::Extractor`, đặt đầu vào và đầu ra;
   - chạy suy luận, được kết quả phân loại;
   - áp `softmax` lên kết quả;
   - chuyển kết quả thành `vector<float>`, lưu vào `cls_scores`.

2. Gọi `print_topk` để in `topk` lớp có điểm cao nhất trong `cls_scores`:
   - tạo vector `std::vector<std::pair<float, int>> vec`, mỗi phần tử gồm điểm của lớp và chỉ số lớp;
   - duyệt `cls_scores`, ghép mỗi điểm với chỉ số của nó thành một cặp `<float, int>` rồi đưa vào `vec`;
   - dùng `std::partial_sort()` sắp xếp một phần `vec` theo điểm giảm dần;
   - duyệt `vec` đã sắp, in chỉ số và điểm của `topk` phần tử đầu.

3. Cuối cùng, hàm `main` đọc ảnh bằng `cv::imread`, rồi gọi `detect_shufflenetv2` và `print_topk` để chạy mạng và in xác suất của các lớp.

Code của `print_topk` kèm chú thích:

```cpp
// đầu vào: vector điểm cls_scores và số lớp cần in topk
static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // 1. tạo vector vec, phần tử kiểu <float, int>, lưu điểm và chỉ số lớp
    int size = cls_scores.size();
    std::vector<std::pair<float, int> > vec;
    vec.resize(size);

    // 2. duyệt điểm các lớp, ghép với chỉ số thành cặp <float, int> và lưu vào vec
    for (int i = 0; i < size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    // 3. sắp xếp một phần vec bằng std::partial_sort(), theo điểm giảm dần
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int> >());

    // 4. in chỉ số và điểm của topk phần tử đầu
    for (int i = 0; i < topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }

    return 0;
}
```

Lưu ý `print_topk` chỉ cho điểm cao nhất và chỉ số lớp; còn phải **đổi chỉ số lớp thành tên lớp**. Thường cần định nghĩa trước một vector `class_names` chứa tên mọi lớp, khớp một-một với chỉ số. `class_names` phải theo đúng nhãn lúc huấn luyện, nếu không tên lớp sẽ bị lệch.

Cuối cùng, chạy thử ví dụ: mô hình là ShuffleNet v2 huấn luyện trên ImageNet, chương trình `shufflenetv2` đã biên dịch chạy trên ảnh thử. Ảnh đầu vào và kết quả:

![Ảnh thử](../images/ncnn/dog.png)

```bash
/ncnn/build/examples# ./shufflenetv2 demo.jpeg
270 = 0.455700
279 = 0.303561
174 = 0.057936
```

Bản gốc tra [bảng tên 1000 lớp ImageNet 2012](https://zhuanlan.zhihu.com/p/315368462) và kết luận lớp `270` là chó.

> **Biên tập:** Trong thứ tự lớp chuẩn của ImageNet-1k (thứ tự của torchvision và hầu hết mô hình), lớp 270 là *white wolf* (sói trắng), 279 là *Arctic fox* (cáo Bắc Cực), 174 là *Norwegian elkhound* (một giống chó); các giống chó nằm ở chỉ số 151–268. Nếu mô hình dùng thứ tự lớp chuẩn này, nó dự đoán con chó lông vàng nhạt trong ảnh là sói trắng (46%) hoặc cáo Bắc Cực (30%), và chỉ cho 6% một giống chó: một dự đoán sai (có thể hiểu được), chứ không phải "lớp 270 là chó". Trang Code làm lại ví dụ với ảnh mèo của scikit-image: ShuffleNet v2 chạy bằng ncnn cho `285 = 0.992` (*Egyptian cat*).

### 3.2 Phân tích quá trình suy luận

Luồng code **suy luận** tổng thể:

1. Tạo đối tượng `Net`, rồi dùng hai giao diện `load_param` và `load_model` để nạp tệp tham số cấu trúc và tệp trọng số:

```cpp
// lược bớt so với code chính thức cho dễ đọc
ncnn::Net shufflenetv2;
shufflenetv2.load_param("shufflenet_v2_x0.5.param");
shufflenetv2.load_model("shufflenet_v2_x0.5.bin");
```

2. Sau khi tạo `Net`, gọi `create_extractor` để tạo `Extractor`. Đối tượng `Extractor` lo việc đưa dữ liệu ảnh vào và chạy suy luận, thực chất cũng là lớp bọc các giao diện liên quan của `Net`.

```cpp
ncnn::Extractor ex = shufflenetv2.create_extractor();
ex.input("data", in);
ncnn::Mat out;
ex.extract("fc", out);  // lấy kết quả của mạng vào ma trận out
```

`Extractor` chỉ tính những tầng cần để ra blob được yêu cầu, và giữ lại các blob trung gian đã tính, nên có thể `extract` nhiều đầu ra từ cùng một extractor mà không tính lại. Mỗi ảnh nên dùng một extractor mới (tạo extractor rất rẻ).

3. Hậu xử lý kết quả: áp softmax lên đầu ra để được xác suất, rồi chuyển thành `vector<float>`.

```cpp
// áp softmax lên ma trận đầu ra
// manually call softmax on the fc output
// convert result into probability
// skip if your model already has softmax operation
{
    ncnn::Layer* softmax = ncnn::create_layer("Softmax");

    ncnn::ParamDict pd;
    softmax->load_param(pd);

    softmax->forward_inplace(out, shufflenetv2.opt);

    delete softmax;
}

// chuyển kết quả softmax thành vector<float>, lưu vào cls_scores
out = out.reshape(out.w * out.h * out.c);

cls_scores.resize(out.w);
for (int j = 0; j < out.w; j++)
{
    cls_scores[j] = out[j];
}
```

Phải gọi tầng softmax bằng tay vì tầng cuối trong tệp cấu trúc của mô hình ShuffleNet v2 chính thức là tầng `fc`, không có tầng `softmax`. Ba dòng cuối của tệp `shufflenet_v2_x0.5.param` là:

```text
ReLU             conv5/relu  1 1 conv5_conv5/bn/scale conv5_conv5/relu
Pooling          pool        1 1 conv5_conv5/relu pool 0=1 1=7 2=1 3=0 4=0
Convolution      fc          1 1 pool fc 0=1000 1=1 2=1 3=1 4=0 5=1 6=1024000
```

> **Biên tập:** Bản gốc là ảnh chụp màn hình ba dòng này; đã gõ lại. Tầng `fc` ở đây được cài bằng tích chập 1×1 (`Convolution` với `0=1000` kênh ra, `1=1` kernel 1×1, `6=1024000 = 1024 × 1000` trọng số) vì mô hình gốc chuyển từ Caffe. Tệp sinh bằng pnnx trên trang Code dùng tầng `InnerProduct` cho tầng cuối, và cũng không có Softmax. Ý nghĩa các trường của một dòng xem [bài lớp Net của ncnn](dlnote/deploy_ncnn_net_vi).

Bản gốc viết tiếp: `ncnn::Mat` mặc định dùng định dạng NCHW (kênh trước: Number-Channel-Height-Width); với bài toán phân loại thường gặp, đầu ra của mạng ncnn thường là tensor kích thước [1, 1, num_classes], chiều thứ ba là số lớp, nên trong code trên `out.w` là số lớp, còn `out.h` và `out.c` đều bằng 1.

> **Biên tập:** `ncnn::Mat` không có chiều batch (N): nó lưu tối đa 4 chiều `w, h, d, c` của *một* mẫu, trong đó `w` là chiều thay đổi nhanh nhất trong bộ nhớ, nên với ảnh là bố cục CHW (mỗi kênh là một mặt phẳng `h × w`, các kênh được căn lề theo 16 byte). Đầu ra phân loại là `Mat` một chiều (`dims = 1`) với `w = 1000`, `h = c = 1`; trang Code in đúng các giá trị này. Gói Python `ncnn` cho phép đổi qua lại `ncnn.Mat` và mảng NumPy (`np.array(mat)`).

### 3.3 Tóm tắt quá trình suy luận

1. Quá trình suy luận gồm các bước:

    1. **Chuẩn bị dữ liệu đầu vào**: có thể là ảnh, văn bản hay dạng khác. Trong ncnn, dữ liệu vào được chuyển thành tensor nhiều chiều (`ncnn::Mat`, một mẫu mỗi lần, không có chiều batch), mô tả hình dạng và kích thước của dữ liệu.
    2. **Nạp tệp tham số cấu trúc và tệp trọng số**: qua hai giao diện `load_param` và `load_model` của `Net`.
    3. **Tính lan truyền xuôi**: từ tầng vào, tính lần lượt từng tầng. Mỗi tầng nhận đầu ra của tầng trước làm đầu vào và thực hiện toán tử của nó, như tích chập, gộp, kết nối đầy đủ; tham số và trọng số của tầng được dùng để tính đầu ra của nó. Cuối cùng đầu ra được đưa tới tầng ra của mô hình.
    4. **Phân tích dữ liệu đầu ra**: đầu ra thường được chuyển sang dạng mà ứng dụng bên ngoài dùng được. Ví dụ trong phân loại ảnh, đầu ra có thể là một **vector xác suất** biểu diễn phân phối xác suất ảnh thuộc từng lớp. Trong ncnn, đầu ra có thể chuyển thành tensor nhiều chiều hay dạng khác.

> **Biên tập:** Bản gốc viết giao diện nạp trọng số là `load_bin` (ở cả mục 3.2 và 3.3) và nói "tham số và trọng số của tầng cũng được dùng để cập nhật đầu ra". Giao diện đúng là `load_model` (như chính đoạn code ở mục 3.2), và khi suy luận trọng số chỉ được đọc, không có gì được "cập nhật". Bước 1 bản gốc nói chiều đầu tiên của tensor là số lượng dữ liệu; như ô Biên tập ở mục 3.2, ncnn không có chiều batch.

2. Các bước ncnn nạp và phân tích tệp cấu trúc, tệp trọng số khá phức tạp, tóm tắt như sau:

    1. Đọc tệp tham số (văn bản hoặc nhị phân) và tệp trọng số, lưu thành mảng byte.
    2. Phân tích phần đầu của tệp tham số: số "ma thuật" (magic number) `7767517` cho biết định dạng, rồi số tầng và số blob.
    3. Phân tích thông tin từng tầng: **tên, loại, các blob vào và ra** của tầng, lưu vào `blobs`. Lớp `Blob` gồm tên, **chỉ số tầng phụ thuộc** (`producer`: tầng sinh ra blob, `consumer`: tầng dùng blob) và **hình dạng** của blob.
    4. Phân tích tham số và trọng số của từng tầng, lưu thành ma trận hay vector (`ncnn::Mat`).

Mã nguồn của các bước này được phân tích ở [bài lớp Net của ncnn](dlnote/deploy_ncnn_net_vi).

## 4 Bổ sung: chuyển mô hình PyTorch sang ncnn và chạy bằng Python

Mục này do người biên soạn viết thêm.

Ngày nay cách khuyên dùng để có mô hình ncnn từ PyTorch là **pnnx** (PyTorch Neural Network eXchange) của chính tác giả ncnn: nó truy vết mô hình bằng TorchScript và sinh thẳng tệp ncnn, tránh được các lỗi toán tử của chuỗi cũ PyTorch → ONNX → `onnx2ncnn`. Cả `pnnx` lẫn ncnn đều có gói Python cài bằng `pip`:

```python
import torch, torchvision, pnnx, ncnn, numpy as np

model = torchvision.models.shufflenet_v2_x1_0(weights="IMAGENET1K_V1").eval()
pnnx.export(model, "shufflenet.pt", (torch.rand(1, 3, 224, 224),), fp16=False)
# sinh shufflenet.ncnn.param, shufflenet.ncnn.bin (và các tệp pnnx trung gian)

net = ncnn.Net()
net.opt.num_threads = 4
net.load_param("shufflenet.ncnn.param")
net.load_model("shufflenet.ncnn.bin")

mat = ncnn.Mat.from_pixels_resize(img_rgb, ncnn.Mat.PixelType.PIXEL_RGB, w, h, 224, 224)
mat.substract_mean_normalize([0.485 * 255, 0.456 * 255, 0.406 * 255],
                             [1 / (0.229 * 255), 1 / (0.224 * 255), 1 / (0.225 * 255)])
ex = net.create_extractor()
ex.input("in0", mat)
ret, out = ex.extract("out0")
scores = np.array(out)            # 1000 logit; softmax tự làm nếu cần xác suất
```

Kết quả trên trang Code:

- Với `fp16=False`, đầu ra của ncnn lệch PyTorch tối đa $1,4\times10^{-5}$ trên logit có độ lớn tới 17,5. Mặc định pnnx dùng `fp16=True`, **lưu trọng số ở fp16** trong tệp `.bin` (4,35 MiB thay vì 8,66 MiB), khi đó sai khác là $0,018$; lớp dự đoán không đổi.
- Hàm `substract_mean_normalize` (tên giữ nguyên lỗi chính tả "substract" của ncnn) tính `(x - mean) * norm`, nên tham số thứ hai là **nghịch đảo** của độ lệch chuẩn, và tính trên thang 0–255 của điểm ảnh.
- Tên đầu vào, đầu ra do pnnx đặt là `in0`, `out0`; với mô hình chuyển từ Caffe là tên blob gốc như `data`, `fc`.
- Độ trễ batch 1 của ShuffleNet v2 trên CPU laptop x86, 4 luồng, qua nhiều lần chạy: PyTorch eager 7–16 ms, ONNX Runtime 2,4–4,7 ms, ncnn 2,5–6,8 ms. Thế mạnh của ncnn là trên ARM (NEON, tối ưu cho big.LITTLE), kích thước thư viện nhỏ và không phụ thuộc gì, hợp với ứng dụng di động.

## Tài liệu tham khảo

1. [Hướng dẫn dùng Git submodule (phần 1)](https://juejin.cn/post/6844903812524670984)
2. [ncnn](https://github.com/Tencent/ncnn), ví dụ [examples/shufflenetv2.cpp](https://github.com/Tencent/ncnn/blob/master/examples/shufflenetv2.cpp)
3. [pnnx](https://github.com/pnnx/pnnx)
4. Zhang et al. *Dive into Deep Learning* (bản tiếng Trung), mục 12.4 Phần cứng.
