# PyTorch nâng cao: thư viện c10, Stream và Event

> Ghi chú của người biên soạn: trang này dịch từ ghi chú *pytorch-c10模块详解* trong *dl_note* của **Harley Zhang** ([bản gốc](https://github.com/harleyszhang/dl_note/blob/main/2-deep_learning_basic/pytorch_code/pytorch-c10模块详解.md), giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)), đọc mã nguồn PyTorch 2.6. Mã C++ giữ nguyên, chú thích dịch sang tiếng Việt. Bốn ảnh chụp danh sách phương thức của bản gốc được thay bằng danh sách lấy từ bản PyTorch đã cài. Chỗ sai ghi ở ô **Biên tập**; mục 5 do người biên soạn viết thêm.

<div class="gpu-pending">Khối cuối của mục 5 cần GPU CUDA nên chưa chạy; sẽ chạy trên máy có GPU bằng lệnh <code>python3 scripts/build_code_pages.py --gpu docs/tools/pt_c10_vi.md</code>.</div>

## 1 Tổng quan về c10

c10 là **thư viện lõi tầng thấp nhất** của PyTorch: định nghĩa các khái niệm nền (tensor, thiết bị, stream, kiểu dữ liệu, khoá điều phối) mà mọi phần khác dùng chung, nhưng không chứa kernel tính toán nào; các kernel (tích chập, nhân ma trận...) nằm ở ATen, xây trên c10. Các mô-đun chính:

+ **`c10/core`**: kiểu cơ bản và khái niệm lõi như **Device**, **Stream**, **DataPtr**, xử lý lỗi (họ macro `C10_THROW_ERROR`, `TORCH_CHECK`)...
    - `TensorImpl`: cài đặt tầng thấp của tensor, lưu con trỏ dữ liệu, kích thước, bước nhảy, thiết bị... (bài [TensorImpl](tools/pt_tensorimpl_vi));
    - `ScalarType`: định nghĩa các kiểu **vô hướng** (`kFloat`, `kInt`...);
    - **Device**: thông tin thiết bị (CPU, CUDA...), quản lý thống nhất giữa các thiết bị;
    - **Stream**: trừu tượng của stream bất đồng bộ (như CUDA stream), cho phép các thao tác chạy song song trên thiết bị;
    - các thành phần khác: cấp phát bộ nhớ (`Allocator`), khoá điều phối (`DispatchKey`).
+ `c10/cuda/`: quản lý thiết bị CUDA, stream, event, cấp phát bộ nhớ:
    - `CUDAStream`: bọc `cudaStream_t`, quản lý các thao tác GPU bất đồng bộ;
    - `CUDAGuard`: đặt thiết bị và stream hiện tại trong một phạm vi (guard);
    - `CUDAFunctions`: bọc API runtime CUDA (đồng bộ thiết bị, chép bộ nhớ...);
    - `CUDACachingAllocator`: bộ cấp phát bộ nhớ CUDA có bộ nhớ đệm (memory pool), tránh gọi `cudaMalloc`/`cudaFree` mỗi lần.
+ **`c10/util`**: template và công cụ phụ trợ như **ArrayRef**, `intrusive_ptr`, công cụ đặc tính kiểu (type traits), giúp đơn giản hoá quản lý dữ liệu khắp framework.
+ **`c10/macros`**: macro kiểm tra lỗi, gợi ý rẽ nhánh (`C10_LIKELY`/`C10_UNLIKELY`), xuất thông tin gỡ lỗi, thích ứng trình biên dịch và nền tảng; dùng nhiều ở đường mã nóng.
+ **Các mô-đun khác**: tuỳ phiên bản, c10 còn có mã thích ứng backend (`hip`, `xpu`...), hệ thống log, công cụ cho phân tán hay thực thi bất đồng bộ.

> **Biên tập:** Bản gốc viết "c10 thiên về trừu tượng cấp cao, còn ATen gần phần cứng hơn". Cách nói này dễ gây hiểu nhầm: c10 là tầng **thấp nhất** (ATen phụ thuộc c10, không có chiều ngược lại), chỉ là nó cung cấp cấu trúc dữ liệu và trừu tượng thiết bị thay vì kernel. `c10::optional` mà bản gốc liệt kê đã được thay bằng `std::optional` từ khi PyTorch chuyển sang C++17.

## 2 Lớp Stream

### 2.1 Lớp trừu tượng Stream

`Stream` biểu diễn một hàng đợi thao tác (luồng thực thi) trên một thiết bị, thường là GPU (CUDA). Lớp `Stream` của c10 được định nghĩa trong `c10/core/Stream.h`. Một `Stream` chỉ gồm hai trường: thiết bị (`device_`) và mã số (`id_`); việc thật sự (đợi, đồng bộ) do backend của thiết bị đó làm.

```cpp
using StreamId = int64_t;

class C10_API Stream final {
 private:
  Device device_;
  StreamId id_;

 public:
  enum Unsafe { UNSAFE };
  enum Default { DEFAULT };

  /// Tạo Stream "không an toàn" từ Device và StreamId. Nói chung chỉ cài đặt stream
  /// của từng backend mới tự tạo Stream kiểu này; người dùng khác nên lấy stream qua API.
  /// Backend không cần đảm bảo gì về StreamId khác 0, tuỳ ý cấp phát.
  explicit Stream(Unsafe, Device device, StreamId id)
      : device_(device), id_(id) {}

  /// Tạo stream mặc định của một Device. Stream mặc định KHÁC stream hiện tại:
  /// stream mặc định cố định, không bao giờ đổi; stream hiện tại có thể đổi bằng StreamGuard.
  explicit Stream(Default, Device device) : device_(device), id_(0) {}

  bool operator==(const Stream& other) const noexcept {
    return this->device_ == other.device_ && this->id_ == other.id_;
  }
  bool operator!=(const Stream& other) const noexcept {
    return !(*this == other);
  }

  Device device() const noexcept { return device_; }
  DeviceType device_type() const noexcept { return device_.type(); }
  DeviceIndex device_index() const noexcept { return device_.index(); }
  StreamId id() const noexcept { return id_; }

  // Đưa một lệnh chờ vào hàng đợi của stream. Lệnh này không làm gì nếu event
  // chưa được đánh dấu ghi; nếu có, stream dừng xử lý cho tới khi event được ghi xong.
  template <typename T>
  void wait(const T& event) const {
    event.block(*this);
  }

  // Mọi việc bất đồng bộ đã xếp vào stream này đã chạy xong trên thiết bị chưa.
  bool query() const;

  // Chặn luồng gọi (CPU) cho tới khi mọi việc đã xếp vào stream chạy xong trên thiết bị.
  void synchronize() const;

  // Đóng gói Stream thành một số uint64_t để dễ chuyển qua lại với Python
  // (khỏi phải dựng một lớp hai trường). Cách đóng gói là chi tiết cài đặt, đừng dựa vào.
  uint64_t hash() const noexcept {
    uint64_t bits = static_cast<uint64_t>(device_type()) << 56 |
        static_cast<uint64_t>(device_index()) << 48 |
        // bỏ phần mở rộng dấu của địa chỉ 64 bit, vì id có thể dùng để chứa con trỏ
        (static_cast<uint64_t>(id()) & ((1ull << 48) - 1));
    return bits;
  }

  struct StreamData3 pack3() const {
    return {id(), device_index(), device_type()};
  }

  static Stream unpack3(StreamId stream_id, DeviceIndex device_index, DeviceType device_type) {
    TORCH_CHECK(isValidDeviceType(device_type));
    return Stream(UNSAFE, Device(device_type, device_index), stream_id);
  }

  // Cố ý KHÔNG có setter: đổi thiết bị của một stream để làm gì?
  // Cứ tạo đúng ngay từ đầu.
};
```

Các phương thức công khai của `Stream`: `device`, `device_type`, `device_index`, `id`, `wait`, `query`, `synchronize`, `hash`, `pack3`, `unpack3`. Từ PyTorch 2.x, lớp này cũng có mặt ở Python dưới tên `torch.Stream` (và `torch.Event`), độc lập với thiết bị:

<!-- run -->
```python
import torch
pub = lambda cls: [m for m in dir(cls) if not m.startswith("_")]
print("torch.Stream     :", pub(torch.Stream))
print("torch.cuda.Stream:", pub(torch.cuda.Stream))
print("torch.cuda.Event :", pub(torch.cuda.Event))
print("CUDA khả dụng:", torch.cuda.is_available(), "| accelerator hiện tại:", torch.accelerator.current_accelerator())
```

<!-- output -->
```output
torch.Stream     : ['device', 'device_index', 'device_type', 'is_capturing', 'native_handle', 'query', 'record_event', 'stream_id', 'synchronize', 'wait_event', 'wait_stream']
torch.cuda.Stream: ['query', 'record_event', 'synchronize', 'wait_event', 'wait_stream']
torch.cuda.Event : ['elapsed_time', 'from_ipc_handle', 'ipc_handle', 'query', 'record', 'synchronize', 'wait']
CUDA khả dụng: False | accelerator hiện tại: None
```
<!-- /output -->

### 2.2 Lớp CUDAStream

Hàm của CUDA chia hai loại: đồng bộ và bất đồng bộ. **Hàm đồng bộ chặn luồng CPU (host); hàm bất đồng bộ trả quyền điều khiển về host ngay để chạy tiếp.** Hàm bất đồng bộ và stream là hai nền tảng của song song ở mức grid.

`CUDAStream` là lớp cốt lõi để PyTorch quản lý thao tác CUDA bất đồng bộ, điều khiển song song và đồng bộ các tác vụ GPU. Một CUDA stream là một dãy thao tác CUDA bất đồng bộ, chạy trên thiết bị **theo đúng thứ tự** host đã gọi; các thao tác ở **hai stream khác nhau** thì không có ràng buộc thứ tự, có thể chạy chồng lên nhau. Ta cũng có thể truy vấn trạng thái các thao tác trong stream.

Lớp `CUDAStream` nằm trong `c10/cuda/CUDAStream.h` và `CUDAStream.cpp`; nó chỉ bọc một `Stream` của c10:

```cpp
class C10_CUDA_API CUDAStream {
 public:
  enum Unchecked { UNCHECKED };

  /// Tạo CUDAStream từ một Stream. Có kiểm tra: báo lỗi nếu Stream đó không phải CUDA stream.
  explicit CUDAStream(Stream stream) : stream_(stream) {
    TORCH_CHECK(stream_.device_type() == DeviceType::CUDA);
  }
  // ... (lược bớt)
 private:
  Stream stream_;   // đối tượng Stream trừu tượng chung, định nghĩa ở c10/core/Stream.h
};
```

Tạo và lấy `CUDAStream`:

- **Stream mặc định**: mỗi thiết bị có một stream mặc định, lấy bằng `getDefaultCUDAStream`. Nếu không chỉ định stream, mọi tính toán của PyTorch chạy trên stream này.

```cpp
/// Lấy stream CUDA mặc định của thiết bị truyền vào, hoặc của thiết bị hiện tại nếu không truyền.
/// Đây là nơi phần lớn tính toán diễn ra khi không dùng stream tường minh.
C10_API CUDAStream getDefaultCUDAStream(DeviceIndex device_index = -1);
```

- **Stream riêng**: lấy bằng `getStreamFromPool` từ một bể (pool) stream, tránh chi phí tạo và huỷ stream liên tục. Các stream riêng có thể chạy song song với nhau, nhưng phải tự đồng bộ.

```cpp
/// Lấy một stream mới từ bể stream CUDA. Có thể coi như "tạo" stream mới, nhưng thực ra
/// không tạo gì: các stream được cấp phát sẵn trong bể và trả về lần lượt (round-robin).
/// isHighPriority = true để lấy từ bể ưu tiên cao; device để chọn thiết bị (mặc định là hiện tại).
C10_API CUDAStream getStreamFromPool(const bool isHighPriority = false, DeviceIndex device = -1);
// không có độ ưu tiên mặc định, để tránh nhập nhằng giữa hai overload
C10_API CUDAStream getStreamFromPool(const int priority, DeviceIndex device = -1);
```

Ví dụ một phương thức, `synchronize`:

```cpp
// Mục đích: đảm bảo mọi thao tác CUDA của stream này đã xong.
void synchronize() const {
    // chuyển thiết bị hiện tại sang thiết bị của stream trước khi gọi API CUDA
    DeviceGuard guard{stream_.device()};
    // chặn cho tới khi mọi thao tác trong stream chạy xong
    c10::cuda::stream_synchronize(stream());
}
```

> **Biên tập:** Bản gốc viết "stream mặc định là stream đồng bộ, thao tác chạy tuần tự". Chính xác hơn: stream mặc định của PyTorch là "legacy default stream" của CUDA, vẫn **bất đồng bộ với CPU**, nhưng **đồng bộ ngầm với các stream khác** (nó chờ mọi stream blocking khác xong, và chúng chờ nó). Ngoài ra bản gốc có hai mục cùng đánh số 2.2; mục Python dưới đây đánh số 2.3.

### 2.3 Lớp Stream ở Python

Lớp `Stream` và `Event` ở Python nằm trong [`torch/cuda/streams.py`](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/cuda/streams.py), lần lượt kế thừa `torch._C._CudaStreamBase` và `torch._C._CudaEventBase` (danh sách phương thức in ở mục 2.1):

+ `Stream`: `wait_event`, `wait_stream`, `record_event`, `query`, `synchronize`...
+ `Event`: `record`, `wait`, `query`, `elapsed_time`, `synchronize`...

## 3 Lớp Event

### 3.1 Event là gì

Event là thiết kế đi cùng stream: chèn một "mốc" vào CUDA stream, **để ghi thời điểm của thao tác GPU hoặc đồng bộ giữa các stream**. Các phương thức chính của `Event`:

- `record(stream=None)`: ghi event vào stream chỉ định (mặc định stream hiện tại);
- `synchronize()`: chặn luồng CPU cho tới khi mọi việc trước event đã xong;
- `wait(stream=None)`: bắt mọi việc **sau này** đưa vào stream chỉ định phải chờ event; không chỉ định thì dùng `torch.cuda.current_stream()`;
- `elapsed_time(end_event)`: thời gian giữa hai event (mili giây; event phải tạo với `enable_timing=True`);
- `query()`: kiểm tra không chặn xem event đã xong chưa.

### 3.2 Vòng đời của một event

- **Tạo** (`cudaEventCreate`): gọi `cudaEventCreate()` hoặc `cudaEventCreateWithFlags()` để tạo đối tượng event. (PyTorch tạo lười: chỉ tạo event CUDA thật ở lần `record` đầu tiên.)
- **Ghi** (`cudaEventRecord`): đặt event vào stream chỉ định. Nếu ghi vào stream mặc định (legacy), do stream mặc định đồng bộ ngầm với các stream khác, event chỉ hoàn tất khi các việc đã đưa vào trước đó ở mọi stream blocking cũng xong.
- **Truy vấn** (`cudaEventQuery`): hỏi không chặn xem event đã xong chưa. Nếu chưa, hàm trả về ngay `cudaErrorNotReady`; chỉ khi mọi thao tác đứng trước event trong stream đã chạy xong mới trả về `cudaSuccess`.
- **Chờ / đồng bộ** (`cudaEventSynchronize`): chặn luồng CPU tới khi event hoàn tất.
- **Huỷ** (`cudaEventDestroy`): giải phóng event.

> **Biên tập:** Bản gốc viết "nếu dùng stream mặc định thì event có hiệu lực tuần tự trên mọi stream (xem phần ngữ nghĩa stream mặc định bên dưới)", nhưng không có phần đó; câu trên viết lại theo ngữ nghĩa legacy default stream đã nêu ở mục 2.2.

Lớp `Event` trong `c10/core/Event.h` chỉ là vỏ; cài đặt nằm ở `c10/core/impl/InlineEvent.h`, một template nhận "backend" của thiết bị làm tham số (với CUDA là `CUDAGuardImpl`). Mã kèm chú thích:

```cpp
// InlineEvent: template bọc các thao tác trên event; hành vi phụ thuộc backend (tham số T).
// Không có hàm tạo mặc định: phải truyền kiểu thiết bị và (tuỳ chọn) EventFlag.
template <typename T>
struct InlineEvent final {
  InlineEvent() = delete;

  InlineEvent(const DeviceType _device_type,
              const EventFlag _flag = EventFlag::PYTORCH_DEFAULT)
      : backend_{_device_type},      // khởi tạo backend theo kiểu thiết bị
        device_type_{_device_type},  // kiểu thiết bị của event
        flag_{_flag} {}              // cờ của event

  // cấm sao chép, tránh vô tình chép event (hai đối tượng cùng giữ một event CUDA)
  InlineEvent(const InlineEvent&) = delete;
  InlineEvent& operator=(const InlineEvent&) = delete;

  // cho phép di chuyển: chuyển quyền sở hữu event
  InlineEvent(InlineEvent&& other) noexcept
      : event_(other.event_),
        backend_(std::move(other.backend_)),
        device_type_(other.device_type_),
        device_index_(other.device_index_),
        flag_(other.flag_),
        was_marked_for_recording_(other.was_marked_for_recording_) {
    other.event_ = nullptr;          // để đối tượng cũ không huỷ event lần nữa
  }

  InlineEvent& operator=(InlineEvent&& other) noexcept {
    swap(other);
    return *this;
  }

  void swap(InlineEvent& other) noexcept {
    std::swap(event_, other.event_);
    std::swap(backend_, other.backend_);
    std::swap(device_type_, other.device_type_);
    std::swap(device_index_, other.device_index_);
    std::swap(flag_, other.flag_);
    std::swap(was_marked_for_recording_, other.was_marked_for_recording_);
  }

  // huỷ: nếu đã có event thật thì nhờ backend giải phóng
  ~InlineEvent() noexcept {
    if (event_)
      backend_.destroyEvent(event_, device_index_);
  }

  DeviceType device_type() const noexcept { return device_type_; }
  DeviceIndex device_index() const noexcept { return device_index_; }
  EventFlag flag() const noexcept { return flag_; }
  bool was_marked_for_recording() const noexcept { return was_marked_for_recording_; }

  // ghi một lần: chỉ ghi nếu chưa từng ghi
  void recordOnce(const Stream& stream) {
    if (!was_marked_for_recording_)
      record(stream);
  }

  // ghi event vào stream
  void record(const Stream& stream) {
    TORCH_CHECK(stream.device_type() == device_type_,
        "Event device type ", DeviceTypeName(device_type_),
        " does not match recording stream's device type ",
        DeviceTypeName(stream.device_type()), ".");
    // backend tạo event (nếu chưa có) rồi ghi vào stream
    backend_.record(&event_, stream, device_index_, flag_);
    was_marked_for_recording_ = true;
    device_index_ = stream.device_index();
  }

  // bắt stream chờ event: các việc đưa vào stream sau lời gọi này chỉ chạy khi event xong.
  // Không chặn CPU.
  void block(const Stream& stream) const {
    if (!was_marked_for_recording_)
      return;
    TORCH_CHECK(stream.device_type() == device_type_,
        "Event device type ", DeviceTypeName(device_type_),
        " does not match blocking stream's device type ",
        DeviceTypeName(stream.device_type()), ".");
    backend_.block(event_, stream);
  }

  // event chưa từng được ghi thì coi như đã xong; ngược lại hỏi backend
  bool query() const {
    if (!was_marked_for_recording_)
      return true;
    return backend_.queryEvent(event_);
  }

  void* eventId() const { return event_; }

  // thời gian giữa hai event (thường tính bằng mili giây)
  double elapsedTime(const InlineEvent& other) const {
    TORCH_CHECK(other.was_marked_for_recording(), "other was not marked for recording.");
    TORCH_CHECK(was_marked_for_recording(), "self was not marked for recording.");
    TORCH_CHECK(other.device_type() == device_type_,
        "Event device type ", DeviceTypeName(device_type_),
        " does not match other's device type ",
        DeviceTypeName(other.device_type()), ".");
    return backend_.elapsedTime(event_, other.event_, device_index_);
  }

  // chặn CPU tới khi event xong
  void synchronize() const {
    if (!was_marked_for_recording_)
      return;
    backend_.synchronizeEvent(event_);
  }

 private:
  void* event_ = nullptr;          // con trỏ tới event của backend (vd. cudaEvent_t), tạo lười
  T backend_;                      // backend của thiết bị (tham số template)
  DeviceType device_type_;         // kiểu thiết bị (CPU, CUDA...)
  DeviceIndex device_index_ = -1;  // chỉ số thiết bị, -1 là chưa đặt
  EventFlag flag_ = EventFlag::PYTORCH_DEFAULT;
  bool was_marked_for_recording_ = false;   // đã ghi lần nào chưa
};
```

> **Biên tập:** Chú thích của bản gốc cho `block` viết "chặn chờ stream cho tới khi mọi thao tác trước event xong", dễ hiểu thành chặn CPU. `block` (tương ứng `Stream::wait` và `cudaStreamWaitEvent`) chỉ đưa một lệnh chờ vào **stream**; CPU chạy tiếp ngay.

## 4 Công cụ quản lý thiết bị: InlineDeviceGuard

`c10/core/impl/InlineDeviceGuard.h` định nghĩa hai lớp `InlineDeviceGuard` và `InlineOptionalDeviceGuard`, dùng cơ chế **RAII** để đặt và khôi phục thiết bị tính toán (CPU/GPU) một cách an toàn:

1. `InlineDeviceGuard`:
    - **Tác dụng**: khi tạo thì chuyển thiết bị hiện tại sang thiết bị chỉ định, khi huỷ thì khôi phục thiết bị ban đầu. Lớp nhận tham số template `T` (cài đặt guard cụ thể, như `CUDAGuardImpl` hay `VirtualGuardImpl`) để làm việc với backend, hỗ trợ điều phối tĩnh hoặc động.
    - **Phương thức chính**: `set_device()`, `reset_device()`, `current_device()`, `original_device()`.
2. `InlineOptionalDeviceGuard`:
    - **Tác dụng**: như `InlineDeviceGuard` nhưng bọc guard trong `std::optional`, cho trường hợp có thể chưa cần đổi thiết bị (ví dụ thiết bị chưa biết lúc tạo guard).
    - **Phương thức**: tương tự.

Ở Python, tương đương là `with torch.cuda.device(1): ...` (đổi thiết bị hiện tại trong khối) và `with torch.cuda.stream(s): ...` (đổi stream hiện tại).

## 5 Bổ sung: stream và event trong thực tế

Mục này do người biên soạn viết thêm.

Vì các phép tính CUDA là bất đồng bộ, đo thời gian một phép tính GPU bằng `time.time()` mà không đồng bộ sẽ chỉ đo thời gian **xếp hàng** kernel. Cách đúng là dùng event (hoặc gọi `torch.cuda.synchronize()` trước khi đọc giờ). Stream riêng cho phép chồng việc chép dữ liệu CPU→GPU với tính toán, miễn là bộ nhớ CPU được khoá trang (`pin_memory()`) và dùng `non_blocking=True`.

Khối dưới cần GPU CUDA: đo một phép nhân ma trận bằng ba cách, và so thời gian chép dữ liệu tuần tự với chép trên một stream riêng song song với tính toán.

<!-- run gpu -->
```python
import time, torch
assert torch.cuda.is_available(), "khối này cần GPU CUDA"
dev = torch.device("cuda")
A = torch.randn(8192, 8192, device=dev); B = torch.randn(8192, 8192, device=dev)
torch.cuda.synchronize()

t = time.perf_counter(); C = A @ B; t_naive = time.perf_counter() - t           # chỉ đo thời gian xếp hàng
torch.cuda.synchronize()
t = time.perf_counter(); C = A @ B; torch.cuda.synchronize(); t_sync = time.perf_counter() - t
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record(); C = A @ B; end.record(); end.synchronize()
print(f"time.time không đồng bộ: {t_naive * 1e3:7.2f} ms (sai)")
print(f"time.time + synchronize: {t_sync * 1e3:7.2f} ms")
print(f"cuda.Event              : {start.elapsed_time(end):7.2f} ms")

host = torch.randn(64, 1024, 1024).pin_memory()             # 256 MB, bộ nhớ khoá trang
copy_stream = torch.cuda.Stream()
def run(overlap):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if overlap:
        with torch.cuda.stream(copy_stream):
            dst = host.to(dev, non_blocking=True)             # chép trên stream riêng
        for _ in range(10): A @ B                              # tính trên stream mặc định cùng lúc
        torch.cuda.current_stream().wait_stream(copy_stream)   # đồng bộ trước khi dùng dst
    else:
        dst = host.to(dev)
        for _ in range(10): A @ B
    torch.cuda.synchronize(); return time.perf_counter() - t0
run(False); run(True)
print(f"chép rồi tính : {run(False) * 1e3:7.1f} ms")
print(f"chép song song: {run(True) * 1e3:7.1f} ms")
```

<!-- output -->
<div class="gpu-pending">Chưa có kết quả: khối này huấn luyện mô hình nên sẽ được chạy trên máy có GPU. Bạn có thể tự chạy code trên Colab hoặc máy của mình.</div>
<!-- /output -->

## Tài liệu tham khảo

- [cuda stream and event](https://huangzhiyuan.github.io/2020/03/24/cuda-stream-and-event/index.html)
- [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) (tài liệu PyTorch: stream, đồng bộ, bộ cấp phát)
- [CUDA C++ Programming Guide: Streams](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
