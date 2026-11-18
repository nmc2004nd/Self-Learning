# Ôn tập phỏng vấn học sâu

Khu này gom các câu hỏi thường gặp khi phỏng vấn và thi viết cho vị trí kỹ sư AI: từ nền toán, học máy, học sâu đến thị giác máy tính, xử lý ngôn ngữ tự nhiên, nén và triển khai mô hình. Mỗi câu hỏi có lời giải ngắn gọn. Mỗi phần có ô **Lý thuyết** dẫn về bài học tương ứng trong các khoá [Toán](math/README_vi), [Học máy](ml/README_vi), [Học sâu](dl/README_vi) và [Học tăng cường](rl_vi), để bạn đọc kỹ khi câu trả lời ngắn chưa đủ.

## Nguồn và cách biên soạn

Nội dung dịch từ *Deep Learning 500 câu hỏi* (深度学习500问, [scutan90/DeepLearning-500-questions](https://github.com/scutan90/DeepLearning-500-questions)), do Tan Jiyong (谈继勇) chủ biên cùng cộng đồng đóng góp, giấy phép [GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html). Phần dịch trên trang này cũng theo giấy phép GPL-3.0.

- **Dịch đầy đủ** mọi câu hỏi và lời giải. Số chương, mục và câu hỏi giữ như bản gốc, nên bạn dễ đối chiếu.
- **Có biên tập**: bản gốc do nhiều người viết nên còn lỗi công thức, lỗi kiến thức và lỗi chính tả. Những chỗ này được sửa ngay trong bài, và mỗi chỗ sửa đáng kể có một ô **Biên tập** ghi lại bản gốc viết gì, sửa thành gì. Các con số trong ví dụ được tính lại bằng NumPy.
- Thuật ngữ thống nhất với các khoá khác trên site. Thuật ngữ tiếng Anh được ghi kèm ở lần xuất hiện đầu tiên.

## Cách dùng

| Cách học | Làm thế nào |
| --- | --- |
| **Đọc lướt** | Đọc câu hỏi và lời giải như một cuốn sổ tay. |
| **Tự kiểm tra** | Bấm nút *Tự kiểm tra: ẩn đáp án* ở đầu mỗi chương. Đáp án sẽ bị ẩn; tự trả lời trong đầu (hoặc ra giấy), rồi bấm vào câu hỏi để so. Lựa chọn này được nhớ cho các lần sau. |
| **Lấp chỗ hổng** | Câu nào trả lời chưa chắc, bấm link trong ô **Lý thuyết** để đọc lại bài gốc. |

## Các chương

| Chương | Nội dung | Lý thuyết liên quan trên site |
| --- | --- | --- |
| 1 | [Nền tảng toán học](qa/ch01_vi) | [Toán](math/README_vi) chương 1–3 |
| 2 | [Nền tảng học máy](qa/ch02_vi) | [Học máy](ml/README_vi) |
| 3 | [Nền tảng học sâu](qa/ch03_vi) | [Học máy](ml/README_vi) chương 12–16, [Học sâu](dl/README_vi) chương 3–5 |
| 4 | [Các mạng kinh điển](qa/ch04_vi) | [Học sâu](dl/README_vi) chương 7–8 |
| 5 | [Mạng neuron tích chập (CNN)](qa/ch05_vi) | [Học sâu](dl/ch07_vi) chương 7 |
| 6 | [Mạng neuron hồi tiếp (RNN)](qa/ch06_vi) | [Học sâu](dl/README_vi) chương 9–10 |
| 7 | [Mạng đối sinh (GAN)](qa/ch07_vi) | [d2l chương 20](https://d2l.ai/chapter_generative-adversarial-networks/index.html) (tiếng Anh) |
| 8 | [Phát hiện đối tượng](qa/ch08_vi) | [d2l chương 14](https://d2l.ai/chapter_computer-vision/index.html) (tiếng Anh) |
| 9 | [Phân vùng ảnh](qa/ch09_vi) | [d2l chương 14](https://d2l.ai/chapter_computer-vision/index.html) (tiếng Anh) |
| 10 | [Học tăng cường](qa/ch10_vi) | [Học tăng cường](rl_vi) |
| 11 | [Học chuyển giao](qa/ch11_vi) | [Học sâu](dl/ch04_s7_vi) mục 4.7, [d2l 14.2](https://d2l.ai/chapter_computer-vision/fine-tuning.html) (tiếng Anh) |
| 12 | [Xây dựng và huấn luyện mạng](qa/ch12_vi) | [Học sâu](dl/ch06_vi) chương 6 |
| 13 | [Thuật toán tối ưu](qa/ch13_vi) | [Học sâu](dl/ch12_vi) chương 12 |
| 14 | [Tinh chỉnh siêu tham số](qa/ch14_vi) | [Học sâu](dl/ch12_s11_vi) mục 12.11 |
| 15 | [Tính toán dị thể, GPU và chọn framework](qa/ch15_vi) | [Học sâu](dl/ch06_s7_vi) mục 6.7 |
| 16 | [Xử lý ngôn ngữ tự nhiên (NLP)](qa/ch16_vi) | [Ôn tập](qa/ch06_vi) chương 6 (RNN); [d2l](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) chương 15–16 (tiếng Anh) |
| 17 | [Nén, tăng tốc mô hình và triển khai trên thiết bị di động](qa/ch17_vi) | [Ôn tập](qa/ch05_vi) chương 5 (CNN), [chương 15](qa/ch15_vi) |
| 18 | [Chọn kiến trúc backend và kịch bản ứng dụng](qa/ch18_vi) | [Ôn tập](qa/ch15_vi) chương 15 (phân tán), [chương 16](qa/ch16_vi) (NLP); [Học máy](ml/eval_vi) đánh giá mô hình |
| 19 | Đăng ký bằng sáng chế phần mềm và bảo hộ quyền *(không dịch: chương nói về thủ tục pháp lý ở Trung Quốc, ít liên quan tới ôn tập kỹ thuật)* | |

Một số chủ đề (GAN, phát hiện đối tượng, phân vùng ảnh, học chuyển giao, NLP, nén mô hình) chưa có bài lý thuyết riêng trên site; ô **Lý thuyết** của các chương này sẽ dẫn sang mục d2l tương ứng khi khoá Học sâu dịch tới.

**Cập nhật kiến thức (2026).** Bản gốc viết khoảng 2018–2020, nên một số chương thiếu các hướng quan trọng xuất hiện về sau. Phần thiếu được bổ sung thành một mục "Bổ sung" ở cuối chương, do người biên tập viết và ghi rõ như vậy; phần dịch giữ nguyên, chỗ lỗi thời có thêm ô **Biên tập**:

| Chương | Mục bổ sung |
| --- | --- |
| 2 | [2.22](qa/ch02_vi) bagging/boosting, tăng cường gradient, XGBoost/LightGBM/CatBoost |
| 3 | [3.15](qa/ch03_vi) GELU/SwiGLU, LayerNorm/RMSNorm, pre-norm, AdamW, khởi động và lịch cosin |
| 4 | [4.8](qa/ch04_vi) EfficientNet, ViT, Swin, ConvNeXt, chọn mạng nền |
| 6 | [6.15](qa/ch06_vi) Transformer thay RNN, mô hình không gian trạng thái (Mamba) |
| 7 | [7.6](qa/ch07_vi) mô hình khuếch tán, sinh ảnh từ văn bản |
| 8 | [8.9](qa/ch08_vi) FCOS/CenterNet, DETR, YOLOv4–v12, phát hiện từ vựng mở |
| 9 | [9.14](qa/ch09_vi) SegFormer, Mask2Former, SAM, phân vùng từ vựng mở |
| 10 | [10.11](qa/ch10_vi) PPO/SAC, MuZero/Dreamer, học tăng cường ngoại tuyến, RLHF/DPO/GRPO |
| 11 | [11.4](qa/ch11_vi) mô hình nền tảng, LoRA, CLIP, học trong ngữ cảnh |
| 12 | [12.7](qa/ch12_vi) TensorFlow 2/Keras 3, PyTorch 2, JAX |
| 13 | [13.19](qa/ch13_vi) bộ tối ưu sau Adam, cắt gradient, độ chính xác hỗn hợp, tiết kiệm bộ nhớ |
| 14 | [14.7](qa/ch14_vi) Hyperband, Optuna, luật tỉ lệ và µP |
