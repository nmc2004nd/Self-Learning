# Triển khai mô hình: ghi chú cơ bản về TensorRT

Bản gốc viết năm 2021 với TensorRT 7.2. Phần dịch giữ nguyên nội dung và ghi các thay đổi ở ô **Biên tập**; mục 4 và 5 do người biên soạn viết thêm cho TensorRT 10. Trang [Code](dlnote/deploy_code_vi) đi kèm có khối dựng engine TensorRT từ tệp ONNX ở fp32 và fp16 (mục 4 của trang). Khối này cần GPU NVIDIA nên chưa có kết quả.

<!-- theory: qa/ch17_vi#17.5.2 ; qa/ch17_vi#17.5.3 ; qa/ch17_vi#17.5.4 -->

## 1 Tổng quan

`TensorRT` là engine tăng tốc suy luận học sâu hiệu năng cao do NVIDIA phát triển, xây trên `CUDA` và `cuDNN`, giúp triển khai mô hình học sâu trên `GPU` với độ trễ thấp và thông lượng cao. Nó viết bằng `C++`, có API cho `C++` và `Python`, hỗ trợ mô hình từ TensorFlow, PyTorch, Caffe, MXNet…; riêng `MXNet` và `PyTorch` phải chuyển sang định dạng trung gian `ONNX` trước. Tới ngày 21/4/2021, phiên bản mới nhất của `TensorRT` là `v7.2.3.4`.

> **Biên tập:** Phiên bản hiện tại là TensorRT 10 (dòng 10.x từ 2024). Trình phân tích Caffe và UFF (TensorFlow 1) đã bị bỏ từ TensorRT 10; đường vào chính là ONNX (qua `OnnxParser`), ngoài ra có Torch-TensorRT (biên dịch thẳng mô hình PyTorch, kể cả qua `torch.compile(backend="tensorrt")`) và TensorRT-LLM cho mô hình ngôn ngữ lớn.

Cách hiểu chung về **độ trễ và thông lượng** trong học sâu:

- **Độ trễ** (`Latency`): người và máy khi ra quyết định hay hành động đều cần thời gian phản ứng. Độ trễ là **thời gian từ khi gửi yêu cầu tới khi nhận được phản hồi**. Với phần lớn hệ thống phần mềm phục vụ con người (không chỉ hệ thống AI), độ trễ được đo bằng **mili giây**.
- **Thông lượng** (`Throughput`): với một mạng học sâu đã tạo hay đã triển khai ở quy mô cho trước, có thể trả về bao nhiêu kết quả suy luận. Hiểu đơn giản là **số mẫu đầu vào tối đa mạng xử lý được trong một đơn vị thời gian (như một giây)**.

## 2 Quy trình làm việc của TensorRT

Trước khi nói về nguyên lý tối ưu của `TensorRT`, cần hiểu quy trình làm việc của nó. Đầu tiên đưa vào một tệp mô hình `FP32` đã huấn luyện, phân tích nó trong `TensorRT` qua `parser`. Phân tích xong, TensorRT tối ưu đồ thị tính toán (nguyên lý ở chương sau) để tạo `engine`. `engine` đã tối ưu có thể tuần tự hoá (serialize) vào bộ nhớ (`buffer`) hay tệp (`file`); khi đọc lại cần giải tuần tự hoá (deserialize) để có `engine` dùng được. Khi chạy, tạo một `context` (ngữ cảnh thực thi), chủ yếu để cấp phát trước tài nguyên; `engine` cộng `context` là suy luận (`Inference`) được.

![Quy trình làm việc của TensorRT](../images/TensorRT/tensorrt_workflow.jpg)

Hình 2.1: Bước 1: tối ưu mô hình đã huấn luyện thành engine (các plan đã tối ưu). Bước 2: khi suy luận, giải tuần tự hoá engine và triển khai ra trung tâm dữ liệu, xe tự lái hay hệ nhúng.

Một engine (tệp "plan") **gắn với đúng GPU và phiên bản TensorRT đã dựng nó**: dựng trên RTX 4090 thì không chạy được trên Jetson, và nâng TensorRT thì phải dựng lại. Vì vậy thường dựng engine ngay trên máy đích, hoặc cho từng loại GPU.

## 3 Nguyên lý tối ưu của TensorRT

Các tối ưu chính của `TensorRT`:

1. **Hợp nhất toán tử (gộp tầng)**: hàm chạy trên `GPU` gọi là `Kernel`. `TensorRT` gọi các `Kernel`, và gọi `Kernel` thường xuyên tốn chi phí hiệu năng, chủ yếu ở: chi phí lập lịch đồ thị luồng dữ liệu, chi phí khởi chạy kernel GPU, và chi phí truyền dữ liệu giữa các kernel. Phần lớn mạng có chuỗi tầng tích chập `conv`, `bias` và kích hoạt `relu` liên tiếp; ba tầng này cần gọi ba API cuDNN, nhưng thực ra có thể hợp nhất thành một cấu trúc `CBR`. Đây là **hợp nhất dọc**. Ngoài ra, mạng hiện nay vừa sâu vừa rộng, có thể chạy song song nhiều tích chập cùng kích thước; các tích chập này cũng gộp lại tính một lần được (**hợp nhất ngang**). Ví dụ trong `GoogLeNet`, các tầng cùng cấu trúc nhưng trọng số khác nhau được gộp thành một tầng rộng hơn. Xem [bài hợp nhất Conv và BN](dlnote/deploy_fuse_vi).
2. **Loại bỏ tầng `concat`.** Với `concat` theo chiều `channel`, `TensorRT` cho đầu ra của các tầng trước ghi thẳng vào đúng vị trí trong vùng nhớ kết quả, không cần sao chép, nhờ đó bỏ được tầng `concat` và giảm số lần truy cập bộ nhớ.
3. **Tự chọn kernel** (kernel auto-tuning). `TensorRT` có sẵn nhiều cài đặt `GPU` cho mỗi tầng, và khi dựng engine sẽ chạy thử để chọn cài đặt nhanh nhất theo `batch size` và kích thước bài toán. Các lựa chọn gồm: dùng loại lõi `CUDA` nào (CUDA core hay Tensor Core), chia việc thế nào, mỗi `block` có bao nhiêu luồng, mỗi `grid` có bao nhiêu `block`… (bản gốc ghi "chưa tìm được tài liệu để hiểu").
4. **`FP32 → FP16, INT8, INT4`**: lượng tử hoá xuống độ chính xác thấp, mô hình nhỏ hơn, bộ nhớ chiếm và độ trễ thấp hơn.
5. Với các phần cứng khác nhau, như card `P4`, `V100` hay GPU nhúng, `TensorRT` đều tối ưu riêng, cho ra `engine` tối ưu cho đúng phần cứng đó.

> **Biên tập:** Hai mục còn thiếu trong danh sách gốc: **tối ưu bộ nhớ động cho tensor** (dynamic tensor memory: cấp phát bộ nhớ cho mỗi tensor trung gian chỉ trong thời gian nó còn dùng, các tensor dùng chung vùng nhớ) và **thực thi đa luồng (multi-stream)** để chạy nhiều đầu vào song song. Đây là 6 nhóm tối ưu mà tài liệu của NVIDIA liệt kê. Ở mục 4, INT4 trong TensorRT 10 chủ yếu dùng cho lượng tử hoá trọng số (weight-only) của mô hình ngôn ngữ lớn; thêm FP8 (từ GPU Hopper) và FP4 (Blackwell). Từ TensorRT 10, đường lượng tử hoá được khuyên dùng là mô hình có sẵn nút `QuantizeLinear/DequantizeLinear` (QDQ, từ QAT hoặc công cụ ModelOpt) thay cho hiệu chỉnh INT8 ngầm (implicit calibration) như ở TensorRT 7.

## 4 Bổ sung: dựng và chạy engine với TensorRT 10

Mục này do người biên soạn viết thêm.

Cách nhanh nhất là công cụ dòng lệnh `trtexec` đi kèm TensorRT:

```bash
# dựng engine fp16 từ ONNX và đo độ trễ
trtexec --onnx=resnet18.onnx --saveEngine=resnet18_fp16.plan --fp16
# chạy lại engine đã lưu
trtexec --loadEngine=resnet18_fp16.plan
# với đầu vào batch động: khai báo khoảng hình dạng (optimization profile)
trtexec --onnx=resnet18_dyn.onnx --minShapes=input:1x3x224x224 --optShapes=input:8x3x224x224 --maxShapes=input:32x3x224x224 --fp16
```

Dùng API Python (đây là khối của trang Code):

```python
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, fp16=False):
    builder = trt.Builder(logger)
    network = builder.create_network(0)                    # TRT 10: mạng luôn explicit batch
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        assert parser.parse(f.read()), parser.get_error(0)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    return builder.build_serialized_network(network, config)   # bước 1: "plan" đã tuần tự hoá

plan = build_engine("resnet18.onnx", fp16=True)
engine = trt.Runtime(logger).deserialize_cuda_engine(plan)    # bước 2: giải tuần tự hoá
context = engine.create_execution_context()
context.set_tensor_address("input", x_gpu.data_ptr())          # bộ nhớ GPU, ví dụ tensor CUDA của PyTorch
context.set_tensor_address("output", out_gpu.data_ptr())
context.execute_async_v3(stream_handle)
```

So với TensorRT 7 của bản gốc, API thay đổi đáng kể: bỏ chế độ implicit batch (cờ `EXPLICIT_BATCH` thành mặc định), bỏ `build_engine` (dùng `build_serialized_network`), bỏ `execute_v2` với danh sách binding (dùng `set_tensor_address` theo tên tensor và `execute_async_v3`), và `max_workspace_size` thay bằng `set_memory_pool_limit`. Code viết cho TensorRT 7, 8 trên mạng thường phải sửa những chỗ này.

<div class="gpu-pending">Chưa có kết quả đo: khối TensorRT trên trang Code cần GPU NVIDIA, sẽ được chạy và điền số khi chuyển sang máy có GPU.</div>

## 5 Bổ sung: TensorRT trong hệ sinh thái hiện nay

Mục này do người biên soạn viết thêm.

- **Torch-TensorRT** biên dịch mô hình PyTorch trực tiếp: `torch.compile(model, backend="tensorrt")` hoặc `torch_tensorrt.compile(...)`; các phần đồ thị TensorRT không hỗ trợ sẽ chạy bằng PyTorch.
- **ONNX Runtime** có `TensorrtExecutionProvider`: dùng TensorRT cho phần đồ thị hỗ trợ được, CUDA cho phần còn lại, không phải tự dựng engine.
- **TensorRT-LLM** chuyên cho mô hình ngôn ngữ lớn (KV cache phân trang, batching liên tục, FP8/FP4, giải mã suy đoán), dùng với Triton Inference Server hoặc trực tiếp.
- **Triton Inference Server** phục vụ mô hình (TensorRT, ONNX, PyTorch…) qua HTTP/gRPC, có dynamic batching, chạy nhiều phiên bản mô hình song song.
- **Jetson** (Orin, Thor) dùng cùng TensorRT với GPU máy chủ; NVIDIA DLA (bộ tăng tốc học sâu trên Jetson) cũng được TensorRT điều khiển.

## Tài liệu tham khảo

1. [Hợp nhất kernel: "vũ khí tăng tốc" của học sâu trên GPU](https://www.msra.cn/zh-cn/news/features/kernel-fusion-20170925) (MSRA)
2. [Thực chiến engine suy luận hiệu năng cao TensorRT](https://zhuanlan.zhihu.com/p/35657027)
3. *NVIDIA TensorRT và ghi chép thực chiến* (bản PPT).
4. [TIRIAS Research: NVIDIA PLASTER Deep Learning Framework](https://www.tiriasresearch.com/wp-content/uploads/2018/05/TIRIAS-Research-NVIDIA-PLASTER-Deep-Learning-Framework.pdf)
5. [NVIDIA TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/) và [Migrating from TensorRT 8.x to 10.x](https://docs.nvidia.com/deeplearning/tensorrt/latest/api/migration-guide.html)
6. [Torch-TensorRT](https://pytorch.org/TensorRT/)
