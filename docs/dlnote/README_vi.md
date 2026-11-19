# Ghi chú học sâu

Khoá này là bộ ghi chú thực chiến về học sâu: nền toán và các thành phần của mạng, các backbone kinh điển và CNN hiệu quả, mẹo huấn luyện, và nhất là **nén, tăng tốc và triển khai mô hình**, mảng mà các khoá [Học máy](ml/README_vi) và [Học sâu](dl/README_vi) chưa có. Mỗi bài lý thuyết đi kèm trang **Code** chạy lại các công thức và thí nghiệm của bài trên CPU.

## Nguồn và cách biên soạn

Nội dung dịch từ kho ghi chú [harleyszhang/dl_note](https://github.com/harleyszhang/dl_note) (深度学习系统笔记) của **Harley Zhang**, giấy phép [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). Phần dịch và phần viết thêm trên site cũng theo Apache-2.0.

- **Dịch đầy đủ, có biên tập.** Số mục giữ như bản gốc để dễ đối chiếu. Đây là ghi chú cá nhân viết trong nhiều năm (2021–2025), nên còn lỗi công thức, lỗi đánh số và vài chỗ đã lỗi thời. Những chỗ này được sửa ngay trong bài, mỗi chỗ có một ô **Biên tập** ghi bản gốc viết gì và vì sao sửa.
- **Cập nhật.** Bài nào đã cũ có thêm một mục **Bổ sung** ở cuối, do người biên soạn viết, tóm tắt những thay đổi từ khi bài gốc ra đời (ví dụ lượng tử hóa cho mô hình ngôn ngữ lớn).
- **Trang Code** do người biên soạn viết: mọi con số nêu trong bài và trong trang Code đều lấy từ kết quả chạy thật.
- Thuật ngữ thống nhất với các khoá khác trên site; thuật ngữ tiếng Anh được ghi kèm ở lần xuất hiện đầu tiên.

## Nội dung

Thứ tự các phần theo bản gốc:

| Phần | Nội dung | Trạng thái |
| --- | --- | --- |
| 1. Nền tảng toán và học máy | [Xác suất và lý thuyết thông tin](dlnote/math_prob_vi), [nguyên lý cơ bản của học máy](dlnote/math_ml_vi), [toán của gradient descent](dlnote/math_grad_vi) ([Code](dlnote/math_code_vi)) | Xong |
| 2. Nền tảng học sâu | [Lớp tích chập](dlnote/dl_conv_vi) (cả tích chập nhóm, depthwise), [BN](dlnote/dl_bn_vi), [hàm kích hoạt](dlnote/dl_act_vi), [hàm mất mát](dlnote/dl_loss_vi), [khởi tạo](dlnote/dl_init_vi), [lan truyền ngược](dlnote/dl_bp_vi), [thuật toán tối ưu](dlnote/dl_optim_vi), [tổng hợp 40 câu hỏi](dlnote/dl_summary_vi) ([Code](dlnote/dl_basic_code_vi)) | Xong |
| 3. Backbone kinh điển | [Tổng kết](dlnote/bb_summary_vi) (thêm các backbone sau 2019), [ResNet](dlnote/bb_resnet_vi), [ResNet v2](dlnote/bb_resnetv2_vi), [DenseNet](dlnote/bb_densenet_vi); CNN hiệu quả: [MobileNet v1](dlnote/bb_mobilenet_vi), [ShuffleNet v2](dlnote/bb_shufflenet_vi), [RepVGG](dlnote/bb_repvgg_vi), [CSPNet](dlnote/bb_cspnet_vi), [VoVNet](dlnote/bb_vovnet_vi) ([Code](dlnote/backbone_code_vi)) | Xong |
| 4. Mẹo huấn luyện | [Siêu tham số](dlnote/alch_hyper_vi), [chuẩn hoá dữ liệu](dlnote/alch_norm_vi), [mẫu mất cân bằng](dlnote/alch_imbalance_vi), [tăng cường dữ liệu](dlnote/alch_augment_vi), [regularization](dlnote/alch_reg_vi) và [trực quan hoá](dlnote/alch_vis_vi) (hai bài cuối bản gốc mới có dàn ý, phần lớn do người biên soạn viết) ([Code](dlnote/alchemy_code_vi)) | Xong |
| 5. Nén mô hình | [Tổng quan](dlnote/compress_overview_vi), [mạng nhẹ](dlnote/compress_light_vi), [thuật toán cắt tỉa](dlnote/compress_prune_vi), [cắt tỉa bằng PyTorch](dlnote/compress_prune_torch_vi) ([Code](dlnote/compress_prune_torch_code_vi)), [lượng tử hóa](dlnote/compress_quant_vi) ([Code](dlnote/compress_quant_code_vi)), [chưng cất tri thức](dlnote/compress_kd_vi) ([Code](dlnote/compress_kd_code_vi)) | Xong (một khối GPU chờ chạy) |
| 6. Triển khai mô hình | [Tổng quan](dlnote/deploy_overview_vi), [độ phức tạp của CNN](dlnote/deploy_flops_vi), [hợp nhất Conv và BN](dlnote/deploy_fuse_vi), [ONNX](dlnote/deploy_onnx_vi), [TensorRT](dlnote/deploy_trt_vi), ncnn ([chạy ví dụ mẫu](dlnote/deploy_ncnn_sample_vi), [lớp Net](dlnote/deploy_ncnn_net_vi)), [chip AI](dlnote/deploy_chips_vi) ([Code](dlnote/deploy_code_vi)) | Xong (một khối GPU chờ chạy) |

Phần ghi chú về công cụ trong kho gốc (SSH, Python, C++, PyTorch, HuggingFace) nằm ở khoá [Công cụ lập trình](tools/README_vi).

## Liên quan trên site

- [Học sâu](dl/README_vi): lý thuyết nền theo *Dive into Deep Learning*; phần 2 và 3 ở đây dẫn sang các chương tương ứng.
- [Ôn tập phỏng vấn](qa/README_vi), nhất là [chương 17](qa/ch17_vi) về nén và triển khai mô hình.
