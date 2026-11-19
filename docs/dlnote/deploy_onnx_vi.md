# Triển khai mô hình: phân tích và sử dụng mô hình ONNX

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/6-model_deploy/ONNX模型分析与使用.md), [kho mã](https://github.com/harleyszhang/dl_note), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0); phần dịch có chỉnh sửa). Số mục giữ như bản gốc. Chỗ bản gốc sai, thiếu hoặc đã lỗi thời được sửa ngay trong bài và ghi ở ô **Biên tập**. Các mục có chữ **Bổ sung** do người biên soạn viết thêm.

Trang [Code](dlnote/deploy_code_vi) đi kèm (mục 3) xuất ResNet-18 sang ONNX bằng cả bộ xuất mới (`dynamo=True`) lẫn bộ xuất cũ, mở `ModelProto` ra xem từng thành phần, chạy bằng ONNX Runtime, và chạy lại mọi đoạn code của mục 5 (kể cả ví dụ `Pad`, đã sửa).

<div class="qa-theory">

**Lý thuyết:** [Ôn tập · 17.5.1 Các phương pháp tối ưu, tăng tốc mô hình](qa/ch17_vi?id=_1751-các-phương-pháp-tối-ưu-tăng-tốc-mô-hình)

</div>

> Phần lớn nội dung bài là tóm tắt và dịch tài liệu chính thức của ONNX, một số ý tham khảo các blog chất lượng trên mạng.

## 1 Tổng quan về ONNX

Phần lớn thuật toán học sâu thực hiện việc học thông qua đồ thị luồng dữ liệu tính toán. Một số framework (như CNTK, Caffe2, Theano và TensorFlow 1) dùng **đồ thị tĩnh**, số khác (như PyTorch và Chainer) dùng **đồ thị động**. Nhưng các framework đều cung cấp giao diện để nhà phát triển dễ dàng dựng đồ thị tính toán và môi trường chạy (runtime) xử lý đồ thị một cách tối ưu. Các đồ thị này đóng vai trò biểu diễn trung gian (intermediate representation, IR), nắm bắt ý định của mã nguồn, giúp tối ưu và chuyển đổi để chạy trên thiết bị cụ thể (CPU, GPU, FPGA…).

**Bản chất ONNX chỉ là một bộ chuẩn mở cho mô hình học máy. Tệp mô hình chỉ lưu cấu trúc liên kết (topology) của mạng và trọng số (thực ra mô hình mà mọi framework lưu ra cuối cùng đều tương tự), nên tách khỏi framework thì không thể suy luận (`inference`) trực tiếp trên mô hình.** Cần một runtime như ONNX Runtime, TensorRT hay OpenVINO để chạy nó.

### 1.1 Vì sao dùng một IR chung

Các framework học sâu hiện nay cung cấp chức năng tương tự nhau, nhưng API, đồ thị tính toán và runtime lại độc lập với nhau, gây nhiều khó khăn cho nhà phát triển AI khi triển khai các mô hình khác nhau trên các nền tảng khác nhau. ONNX nhằm cung cấp một định dạng biểu diễn trung gian dùng chung giữa các framework, phục vụ chuyển đổi và triển khai mô hình. Đồ thị tính toán của ONNX là chung, và định dạng là mã nguồn mở.

## 2 Đặc tả ONNX

> Open Neural Network Exchange Intermediate Representation (ONNX IR) Specification.

Tệp định nghĩa cấu trúc `.proto` và `.proto3` của `ONNX` nằm trong thư mục [onnx](https://github.com/onnx/onnx/tree/main/onnx) của kho mã, theo giao thức `Protobuf` của Google. ONNX là đặc tả mở gồm các thành phần:

- định nghĩa mô hình đồ thị tính toán mở rộng được;
- định nghĩa các kiểu dữ liệu chuẩn;
- định nghĩa các toán tử có sẵn.

ONNX IR phiên bản 6 trở về trước chỉ dùng cho suy luận; từ `IR7` (ONNX 1.7), ONNX hỗ trợ cả huấn luyện. Các đối tượng chính của `onnx.proto`:

- ModelProto
- GraphProto
- NodeProto
- AttributeProto
- ValueInfoProto
- TensorProto

Quan hệ giữa chúng: nạp (`load`) một mô hình ONNX được một `ModelProto`, chứa thông tin phiên bản, thông tin công cụ sinh ra mô hình, và một `GraphProto` rất quan trọng. `GraphProto` chứa bốn mảng `repeated` chính: `node` (kiểu NodeProto), `input` (kiểu ValueInfoProto), `output` (kiểu ValueInfoProto) và `initializer` (kiểu TensorProto). `node` chứa mọi nút tính toán của mô hình, `input` chứa các đầu vào, `output` chứa các đầu ra, `initializer` chứa mọi trọng số. Cấu trúc liên kết giữa các nút được xác định qua quan hệ trỏ giữa hai mảng chuỗi `input` và `output` của từng nút, nên từ các thông tin trên có thể dựng nhanh đồ thị của mô hình. Cuối cùng, mỗi nút còn chứa một mảng `AttributeProto` mô tả thuộc tính của nút, ví dụ `Conv` có các thuộc tính `group`, `pads`, `strides`…; thuộc tính, đầu vào và đầu ra của từng loại nút xem tài liệu [Operators.md](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

Bản gốc lưu ý thêm: mảng `input` của `GraphProto` không chỉ chứa nút đầu vào ảnh như ta thường hiểu, mà còn chứa mọi trọng số của mô hình. Ví dụ trọng số `W` của tầng `Conv` được lưu trong `initializer`, đồng thời có một đầu vào cùng tên trong `input`; logic phía sau là coi trọng số cũng là đầu vào của mô hình, được khởi tạo (điền giá trị) bằng tensor trong `initializer`.

> **Biên tập:** Điều này chỉ đúng với các mô hình theo IR phiên bản 3 trở về trước. Từ IR 4 (ONNX 1.3), initializer không bắt buộc phải xuất hiện trong `input`; trọng số chỉ có trong `initializer` là hằng số, còn trọng số có cả trong `input` thì người dùng có thể ghi đè lúc chạy. Bộ xuất của PyTorch mặc định không đưa trọng số vào `input`: trang Code cho thấy cả hai bộ xuất đều sinh graph có đúng một `input` (ảnh) và 42–44 initializer; chỉ khi đặt `keep_initializers_as_inputs=True` thì `input` mới có 43 phần tử.

### 2.1 Model

Mục đích chính của cấu trúc model là gắn siêu dữ liệu (`metadata`) với đồ thị (`graph`), nơi chứa mọi thành phần thực thi được. Siêu dữ liệu được dùng khi đọc tệp mô hình, cho biên runtime đủ thông tin để quyết định có chạy được mô hình hay không, ghi log, báo lỗi… Siêu dữ liệu cũng có ích cho các công cụ như IDE và kho mô hình, để cho người dùng biết mục đích và đặc điểm của mô hình.

Mỗi model có các thành phần:

| Tên | Kiểu | Mô tả |
| --- | --- | --- |
| ir_version | int64 | Phiên bản ONNX IR mà mô hình dùng. |
| opset_import | OperatorSetId | Tập các định danh bộ toán tử mà mô hình dùng. Runtime phải hỗ trợ mọi toán tử trong các bộ đó, nếu không phải từ chối mô hình. |
| producer_name | string | Tên công cụ sinh ra mô hình. |
| producer_version | string | Phiên bản của công cụ đó. |
| domain | string | Tên miền ngược (reverse-DNS) chỉ không gian tên của mô hình, ví dụ 'org.onnx'. |
| model_version | int64 | Phiên bản của chính mô hình, là một số nguyên. |
| doc_string | string | Tài liệu cho người đọc; cho phép Markdown. |
| graph | Graph | Đồ thị có tham số, được thực thi khi chạy mô hình. |
| metadata_props | map<string,string> | Các cặp siêu dữ liệu có tên; khoá không được trùng. |
| training_info | TrainingInfoProto[] | Phần mở rộng tuỳ chọn, chứa thông tin cho huấn luyện. |

Với ResNet-18, bộ xuất mới của PyTorch 2.14 sinh mô hình `ir_version` 10, `opset` 20; bộ xuất cũ với `opset_version=17` sinh `ir_version` 8 (trang Code).

### 2.2 Bộ toán tử (Operator Sets)

Mỗi mô hình phải ghi rõ các bộ toán tử mà chức năng của nó phụ thuộc vào. Một bộ toán tử định nghĩa các toán tử có sẵn, phiên bản và trạng thái của chúng. Mỗi mô hình nhập các bộ toán tử theo miền (domain) của chúng. Mọi mô hình đều ngầm nhập bộ toán tử mặc định của ONNX.

Thuộc tính của đối tượng bộ toán tử (`Operators Sets`):

| Tên | Kiểu | Mô tả |
| --- | --- | --- |
| magic | string | Giá trị 'ONNXOPSET'. |
| ir_version | int32 | Phiên bản ONNX IR tương ứng với các toán tử. |
| ir_version_prerelease | string | Phần prerelease trong số phiên bản SemVer của IR. |
| ir_build_metadata | string | Siêu dữ liệu bản dựng của phiên bản bộ toán tử này. |
| domain | string | Miền của bộ toán tử; phải duy nhất trong mọi bộ. |
| opset_version | int64 | Phiên bản của bộ toán tử. |
| doc_string | string | Tài liệu cho người đọc; cho phép Markdown. |
| operator | Operator[] | Các toán tử trong bộ. |

### 2.3 Toán tử ONNX (Operator)

Mỗi toán tử dùng trong đồ thị (`graph`) phải được khai báo rõ bởi một bộ toán tử mà mô hình (`model`) nhập vào.

Thuộc tính của đối tượng toán tử (`Operator`):

| Tên | Kiểu | Mô tả |
| --- | --- | --- |
| op_type | string | Tên toán tử, dùng trong các nút của đồ thị. Phải duy nhất trong miền của bộ toán tử. |
| since_version | int64 | Phiên bản bộ toán tử khi toán tử này được thêm vào. |
| status | OperatorStatus | 'EXPERIMENTAL' hoặc 'STABLE'. |
| doc_string | string | Tài liệu cho người đọc; cho phép Markdown. |

### 2.4 Đồ thị ONNX (Graph)

Một đồ thị tuần tự hoá gồm các trường siêu dữ liệu (`metadata`), danh sách tham số của mô hình (`a list of model parameters`) và danh sách nút tính toán (`a list of computation nodes`). Mỗi đồ thị luồng dữ liệu được dựng thành danh sách nút đã sắp xếp topo, không được có chu trình. Mỗi nút là một lần gọi một toán tử, có không hoặc nhiều đầu vào và một hoặc nhiều đầu ra.

Thuộc tính của đối tượng đồ thị (Graph):

| Tên | Kiểu | Mô tả |
| --- | --- | --- |
| name | string | Tên đồ thị của mô hình. |
| node | Node[] | Danh sách nút, xếp theo thứ tự topo, tạo thành đồ thị tính toán (thứ tự bộ phận) dựa trên quan hệ phụ thuộc đầu vào/đầu ra. |
| initializer | Tensor[] | Danh sách tensor có tên. Khi một `initializer` trùng tên với một đầu vào của `graph`, nó là giá trị mặc định của đầu vào đó; nếu không, nó là một hằng số. |
| doc_string | string | Tài liệu cho người đọc. |
| input | ValueInfo[] | Các đầu vào của `graph`; giá trị khởi tạo mặc định (nếu có) nằm trong `initializer`. |
| output | ValueInfo[] | Các đầu ra của `graph`. |
| value_info | ValueInfo[] | Thông tin kiểu và hình dạng của các giá trị trung gian (ngoài đầu vào, đầu ra). |

### 2.5 ValueInfo

Thuộc tính của đối tượng `ValueInfo`:

| Tên | Kiểu | Mô tả |
| --- | --- | --- |
| name | string | Tên của giá trị hay tham số. |
| type | Type | Kiểu của giá trị, **gồm cả thông tin hình dạng**. |
| doc_string | string | Tài liệu cho người đọc; cho phép Markdown. |

Mô hình xuất từ PyTorch thường để `value_info` trống; `onnx.shape_inference.infer_shapes` điền nó bằng cách suy luận hình dạng qua từng nút (trang Code: 48 giá trị trung gian của ResNet-18).

### 2.6 Các kiểu dữ liệu chuẩn

Bản gốc nói chuẩn ONNX có hai biến thể (ONNX và ONNX-ML), khác nhau chủ yếu ở các kiểu dữ liệu và toán tử hỗ trợ. Dưới đây là các kiểu dữ liệu mà đồ thị, nút và initializer hỗ trợ. Số nguyên, chuỗi và kiểu bool phải được dùng làm phần tử của tensor.

#### 2.6.1 Kiểu phần tử của tensor

| Nhóm | Kiểu | Mô tả |
| --- | --- | --- |
| Số thực | float16, float32, float64, bfloat16; float8 (e4m3fn, e4m3fnuz, e5m2, e5m2fnuz), float4e2m1 | Số thực theo IEEE 754-2008 (bfloat16 và float8, float4 theo đặc tả riêng). |
| Số nguyên có dấu | int4, int8, int16, int32, int64 | Số nguyên có dấu 4 tới 64 bit. |
| Số nguyên không dấu | uint4, uint8, uint16, uint32, uint64 | Số nguyên không dấu 4 tới 64 bit. |
| Số phức | complex64, complex128 | Số phức có phần thực, phần ảo 32 hoặc 64 bit. |
| Khác | string | Dữ liệu văn bản, mã hoá UTF-8. |
| Khác | bool | Giá trị logic, chỉ có `true` và `false`. |

> **Biên tập:** Bảng gốc (khoảng 2020) chỉ có uint8, uint16 cho số nguyên không dấu và thiếu bfloat16. ONNX đã có uint32, uint64 và bfloat16 từ lâu; float8 được thêm ở ONNX 1.14 (IR 9), int4/uint4 ở ONNX 1.16 (IR 10), float4e2m1 ở ONNX 1.18 (IR 11).

#### 2.6.2 Kiểu đầu vào, đầu ra

Các kiểu sau dùng để khai báo đầu vào, đầu ra của đồ thị và của các nút:

| Biến thể | Kiểu | Mô tả |
| --- | --- | --- |
| ONNX | dense tensors | Tensor đặc, tổng quát hoá của vector và ma trận. |
| ONNX | sequence | Dãy có thứ tự các phần tử đặc. |
| ONNX | map | Bảng kết hợp, xác định bởi kiểu khoá và kiểu giá trị. |
| ONNX | optional | Giá trị có thể có hoặc không (từ IR 8). |
| ONNX | sparse tensors | Tensor thưa (`SparseTensorProto`, từ IR 6). |

> **Biên tập:** Bản gốc viết "ONNX hiện chưa định nghĩa kiểu tensor thưa". `SparseTensorProto` đã có từ IR 6 (ONNX 1.6, 2019), dù ít runtime hỗ trợ đầy đủ.

## 3 Quản lý phiên bản của ONNX

*Mục này trong bản gốc để trống; nội dung do người biên soạn viết.*

ONNX có ba loại số phiên bản độc lập:

- **Phiên bản IR** (`ir_version`): phiên bản của định dạng tệp (cấu trúc protobuf). Tăng khi đặc tả thêm trường hay kiểu mới, ví dụ IR 7 thêm thông tin huấn luyện, IR 9 thêm float8, IR 10 thêm int4. Runtime chỉ đọc được mô hình có IR không mới hơn bản nó hỗ trợ.
- **Phiên bản bộ toán tử** (`opset_version`): mỗi miền (`ai.onnx` mặc định, `ai.onnx.ml`, hay miền riêng như `com.microsoft`) có số phiên bản riêng, tăng mỗi khi có toán tử được thêm hoặc đổi ngữ nghĩa. Mỗi toán tử có `since_version`: ví dụ `Pad` đổi từ thuộc tính `pads` (Pad-2) sang đầu vào `pads` dạng tensor int64 (Pad-11), rồi thêm đầu vào `axes` (Pad-18). Mô hình khai báo opset nào thì runtime dùng định nghĩa toán tử của đúng opset đó.
- **Phiên bản mô hình** (`model_version`): do người dùng tự đặt cho mô hình của mình.

Khi xuất từ PyTorch, chọn opset bằng tham số `opset_version`; opset càng mới thì càng nhiều toán tử được hỗ trợ trực tiếp, nhưng runtime đích (nhất là NPU nhúng) có thể chưa hỗ trợ. Có thể đổi opset của mô hình có sẵn bằng `onnx.version_converter.convert_version(model, target)`.

## 4 Tổng quan các toán tử chính

*Mục này trong bản gốc để trống; nội dung do người biên soạn viết.*

Bộ toán tử `ai.onnx` (opset 23 năm 2025) có khoảng 200 toán tử. Các nhóm hay gặp trong mô hình thị giác và ngôn ngữ:

| Nhóm | Toán tử |
| --- | --- |
| Tích chập, gộp | `Conv`, `ConvTranspose`, `MaxPool`, `AveragePool`, `GlobalAveragePool` |
| Đại số tuyến tính | `Gemm` (tầng FC), `MatMul`, `Einsum` |
| Kích hoạt | `Relu`, `Sigmoid`, `Tanh`, `LeakyRelu`, `HardSwish`, `Gelu` (từ opset 20), `Softmax` |
| Chuẩn hoá | `BatchNormalization`, `LayerNormalization` (opset 17), `InstanceNormalization`, `GroupNormalization`, `RMSNormalization` (opset 23) |
| Từng phần tử | `Add`, `Mul`, `Sub`, `Div`, `Pow`, `Sqrt`, `Exp`, `Where`, `Clip` |
| Đổi hình dạng | `Reshape`, `Flatten`, `Transpose`, `Concat`, `Split`, `Slice`, `Gather`, `Squeeze`, `Unsqueeze`, `Expand`, `Resize` |
| Rút gọn | `ReduceMean`, `ReduceSum`, `ReduceMax`, `ArgMax`, `TopK` |
| Lượng tử hoá | `QuantizeLinear`, `DequantizeLinear` (định dạng QDQ), `QLinearConv`, `QLinearMatMul` |
| Điều khiển luồng | `If`, `Loop`, `Scan` |
| Attention | `Attention` (opset 23) |

ResNet-18 chỉ cần 7 loại: `Conv`, `Relu`, `MaxPool`, `Add`, `GlobalAveragePool` (bộ xuất mới dùng `ReduceMean`), `Flatten` (`Reshape`) và `Gemm`. Lỗi chuyển đổi thường gặp nhất là mô hình dùng toán tử mà runtime đích chưa hỗ trợ, hoặc toán tử có hình dạng động (`Reshape` theo kích thước đầu vào, `Resize` với tỉ lệ tính lúc chạy) mà NPU không chạy được; khi đó phải viết lại phần đó của mô hình hoặc đặt hình dạng cố định.

## 5 Dùng Python API

### 5.1 Nạp mô hình

1. **Nạp một mô hình ONNX**

```python
import onnx
# onnx_model là một ModelProto trong bộ nhớ
onnx_model = onnx.load('path/to/the/model.onnx')  # nạp mô hình onnx
```

2. **Nạp mô hình có dữ liệu ngoài (external data)**

- (Cách nạp mặc định) Nếu dữ liệu ngoài (`external data`) nằm cùng thư mục với tệp mô hình, chỉ cần `onnx.load()` như trên.
- Nếu dữ liệu ngoài nằm ở thư mục khác, sau `onnx.load()` cần gọi thêm `load_external_data_for_model()` để chỉ đường dẫn dữ liệu ngoài.

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

onnx_model = onnx.load('path/to/the/model.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, 'data/directory/path/')
# onnx_model đã nạp dữ liệu ngoài từ thư mục chỉ định
```

3. **Chuyển mô hình ONNX sang dạng dữ liệu ngoài**

```python
from onnx.external_data_helper import convert_model_to_external_data

# onnx_model là một ModelProto trong bộ nhớ
onnx_model = ...
convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location='filename', size_threshold=1024, convert_attribute=False)
# dữ liệu thô của onnx_model đã chuyển thành dữ liệu ngoài; phải gọi save sau đó
```

> Dữ liệu ngoài cần thiết khi mô hình lớn hơn 2 GB, giới hạn kích thước của một thông điệp protobuf; các mô hình ngôn ngữ lớn đều phải lưu theo cách này. Trang Code tách ResNet-18 thành tệp `.onnx` 15 KiB (chỉ cấu trúc) và tệp `.data` 44,6 MiB (trọng số), rồi nạp lại và chạy được.

### 5.2 Lưu mô hình

1. **Lưu mô hình ONNX**

```python
import onnx

# onnx_model là một ModelProto trong bộ nhớ
onnx_model = ...

# lưu mô hình ONNX
onnx.save(onnx_model, 'path/to/the/model.onnx')
```

2. **Chuyển và lưu mô hình ONNX dưới dạng dữ liệu ngoài**

```python
import onnx

# onnx_model là một ModelProto trong bộ nhớ
onnx_model = ...
onnx.save_model(onnx_model, 'path/to/save/the/model.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='filename', size_threshold=1024, convert_attribute=False)
# dữ liệu thô đã chuyển thành dữ liệu ngoài và lưu vào thư mục chỉ định
```

### 5.3 Chuyển đổi giữa TensorProto và mảng NumPy

```python
import numpy
import onnx
from onnx import numpy_helper

# tạo một mảng NumPy
numpy_array = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
print('Original Numpy array:\n{}\n'.format(numpy_array))

# chuyển mảng NumPy thành TensorProto
tensor = numpy_helper.from_array(numpy_array)
print('TensorProto:\n{}'.format(tensor))

# chuyển TensorProto về mảng NumPy
new_array = numpy_helper.to_array(tensor)
print('After round trip, Numpy array:\n{}\n'.format(new_array))

# lưu TensorProto
with open('tensor.pb', 'wb') as f:
    f.write(tensor.SerializeToString())

# nạp TensorProto
new_tensor = onnx.TensorProto()
with open('tensor.pb', 'rb') as f:
    new_tensor.ParseFromString(f.read())
print('After saving and loading, new TensorProto:\n{}'.format(new_tensor))
```

Lưu ý `dtype=float` của NumPy là `float64`, nên tensor có kiểu `DOUBLE`, 48 byte cho 6 phần tử (trang Code); mô hình thường dùng `float32`.

### 5.4 Tạo mô hình ONNX

Có thể tạo mô hình ONNX bằng các hàm của mô-đun `helper`, chủ yếu là `helper.make_graph`. Trước khi tạo `graph`, cần tạo các `NodeProto` (nút) tương ứng, đặt thuộc tính theo tài liệu, chỉ định đầu vào và đầu ra của nút; nếu nút có trọng số thì còn phải tạo `ValueInfoProto` và `TensorProto` tương ứng, lần lượt đặt vào `input` và `initializer` của `graph`. Không được thiếu bước nào.

```python
import onnx
from onnx import helper
from onnx import TensorProto

# Đầu vào (ValueInfoProto). Pad-11 trở đi: pads là tensor int64 có 2 x rank phần tử,
# constant_value là vô hướng cùng kiểu với dữ liệu.
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
pads = helper.make_tensor_value_info('pads', TensorProto.INT64, [4])
value = helper.make_tensor_value_info('value', TensorProto.FLOAT, [])

# Đầu ra (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

# Nút (NodeProto)
node_def = helper.make_node(
    'Pad',                  # loại toán tử
    ['X', 'pads', 'value'], # đầu vào
    ['Y'],                  # đầu ra
    mode='constant',        # thuộc tính
)

# Đồ thị (GraphProto)
graph_def = helper.make_graph(
    [node_def],        # các nút
    'pad-model',       # tên
    [X, pads, value],  # đầu vào
    [Y],               # đầu ra
)

# Mô hình (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example',
                              opset_imports=[helper.make_opsetid("", 18)])

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def, full_check=True)
print('The model is checked!')
```

> **Biên tập:** Ví dụ của bản gốc (lấy từ tài liệu ONNX cũ) khai báo `pads` là tensor `FLOAT` hình dạng `[1, 4]` và `value` có kiểu phần tử `AttributeProto.FLOAT` (hằng số của *thuộc tính*, chỉ tình cờ có cùng giá trị 1 với `TensorProto.FLOAT`) với hình dạng `[1]`. Với Pad-11 trở đi, `pads` phải là tensor `int64` một chiều, còn `constant_value` là vô hướng cùng kiểu với `X`. `check_model` mặc định không bắt được lỗi này, nhưng với `full_check=True` (chạy cả suy luận kiểu) thì báo `pads ... has unsupported type: tensor(float)`, và ONNX Runtime từ chối nạp mô hình. Bản trên đã sửa và chạy được: đệm ma trận 3×2 thêm một cột giá trị 9 mỗi bên, cho kết quả 3×4 (trang Code).

### 5.5 Kiểm tra mô hình

Sau khi nạp hay tạo mô hình `ONNX`, nên kiểm tra mô hình bằng `onnx.checker.check_model()`.

```python
import onnx

# nạp mô hình ONNX
model_path = 'path/to/the/model.onnx'
onnx_model = onnx.load(model_path)

print('The model is:\n{}'.format(onnx_model))

# kiểm tra mô hình
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')
```

> **Biên tập:** Bản gốc ghi tên hàm là `onnx.check.check_model()`; tên đúng là `onnx.checker.check_model()`. Với mô hình lớn hơn 2 GB, truyền đường dẫn tệp thay cho `ModelProto`.

### 5.6 Hàm tiện ích

Hàm `extract_model()` tách một mô hình con khỏi mô hình ONNX, xác định bởi tên các tensor đầu vào và đầu ra. Hàm này giúp gỡ lỗi: so đầu ra từng phần của mô hình gốc và mô hình ONNX đã chuyển (sai khác nhỏ hơn một ngưỡng), không cần sửa tay mô hình ONNX.

```python
import onnx

input_path = 'path/to/the/original/model.onnx'
output_path = 'path/to/save/the/extracted/model.onnx'
input_names = ['input_0', 'input_1', 'input_2']
output_names = ['output_0', 'output_1']

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
```

Trang Code tách ResNet-18 từ đầu vào tới đầu ra của `layer1` (13 nút), chạy bằng ONNX Runtime và so với đầu ra `layer1` lấy bằng forward hook trong PyTorch: sai khác $3,7\times10^{-6}$.

### 5.7 Công cụ

Hàm `update_inputs_outputs_dims()` cập nhật số chiều của đầu vào, đầu ra mô hình thành các giá trị chỉ định; có thể dùng `dim_param` (tên chiều) để đặt kích thước động.

```python
import onnx
from onnx.tools import update_model_dims

model = onnx.load('path/to/the/model.onnx')
# 'seq', 'batch' và -1 đều là chiều động (dim_param)
variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {'input_name': ['seq', 'batch', 3, -1]}, {'output_name': ['seq', 'batch', 1, -1]})
# kiểm tra lại mô hình sau khi đổi kích thước
onnx.checker.check_model(variable_length_model)
```

> **Biên tập:** Hàm này chỉ đổi khai báo hình dạng của đầu vào, đầu ra; nếu bên trong mô hình có nút phụ thuộc kích thước cố định (như `Reshape` với hình dạng hằng số), mô hình vẫn không chạy được với kích thước mới. Cách chắc chắn hơn là xuất lại từ PyTorch với chiều động: `dynamic_axes` ở bộ xuất cũ, `dynamic_shapes` ở bộ xuất mới. Trang Code cho thấy mô hình kích thước cố định từ chối batch 4 (`Got: 4 Expected: 1`), còn mô hình xuất với `dynamic_axes` chạy được.

## 6 Bổ sung: xuất mô hình PyTorch và chạy bằng ONNX Runtime

Mục này do người biên soạn viết thêm (bản gốc có tệp `classifynet_torch_to_onnx.py` ở phần 1 của kho mã, dùng API năm 2021).

```python
import torch, torchvision, onnxruntime as ort

model = torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()   # luôn eval() trước khi xuất
x = torch.randn(1, 3, 224, 224)

# Bộ xuất mới (mặc định từ PyTorch 2.9): dựa trên torch.export
onnx_program = torch.onnx.export(model, (x,), dynamo=True,
                                 dynamic_shapes={"x": {0: torch.export.Dim("batch")}})
onnx_program.save("resnet18.onnx")

# Bộ xuất cũ (TorchScript), vẫn dùng được nhưng đã lỗi thời
torch.onnx.export(model, (x,), "resnet18_legacy.onnx", dynamo=False, opset_version=17,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

sess = ort.InferenceSession("resnet18_legacy.onnx", providers=["CPUExecutionProvider"])
out = sess.run(None, {"input": x.numpy()})[0]
```

Kết quả trên CPU chạy trang Code: đầu ra ONNX Runtime lệch PyTorch tối đa khoảng $4\times10^{-6}$, và độ trễ batch 1 của ResNet-18 khoảng 12–16 ms so với 15–21 ms của PyTorch eager (tuỳ lần chạy). Vài điểm cần nhớ:

- **Bộ xuất mới** truy vết bằng `torch.export` (bắt toàn bộ đồ thị ở mức Python, hỗ trợ điều khiển luồng qua `torch.cond`), sinh opset mới hơn (20 trở lên) và tự tối ưu đồ thị; tên đầu vào lấy theo tên tham số của `forward` (`x`). Bộ xuất cũ dựa trên TorchScript, chỉ ghi lại đúng nhánh đã chạy với đầu vào mẫu.
- **Luôn kiểm tra kết quả** bằng cách chạy cùng một đầu vào qua PyTorch và runtime đích, so sai khác; với từng phần mô hình dùng `extract_model` như mục 5.6.
- **Công cụ hữu ích**: [Netron](https://netron.app) để xem đồ thị, `onnxsim` (onnx-simplifier) hay `onnxscript.optimizer` để gập hằng số và rút gọn đồ thị, `onnxruntime.quantization` để lượng tử hoá INT8 (định dạng QDQ).
- Mã tham khảo của bản gốc dùng `torch.autograd.Variable` (đã lỗi thời từ PyTorch 0.4) và `opset_version=9`; ngày nay nên dùng opset 17 trở lên.

## Tài liệu tham khảo

1. [ONNX: định dạng biểu diễn trung gian giữa các framework](https://zhuanlan.zhihu.com/p/41255090) (ONNX--跨框架的模型中间表达框架)
2. [Chuyện chuyển đổi và triển khai mô hình học sâu (có phân tích chi tiết định dạng ONNX)](https://bindog.github.io/blog/2020/03/13/deep-learning-model-convert-and-depoly/)
3. [ONNX tutorials](https://github.com/onnx/tutorials)
4. [ONNX IR specification](https://github.com/onnx/onnx/blob/main/docs/IR.md) và [Versioning](https://github.com/onnx/onnx/blob/main/docs/Versioning.md)
5. [PyTorch: torch.onnx](https://docs.pytorch.org/docs/stable/onnx.html)
6. [ONNX Runtime](https://onnxruntime.ai/docs/)
