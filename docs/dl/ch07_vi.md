# Chương 7 Mạng Neuron Tích chập

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_convolutional-neural-networks/index.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Dữ liệu ảnh được biểu diễn dưới dạng một lưới điểm ảnh (pixel) hai chiều, dù ảnh là
đơn sắc hay có màu. Theo đó, mỗi điểm ảnh tương ứng với một
hoặc nhiều giá trị số. Cho đến nay ta đã bỏ qua cấu trúc phong phú
này và coi ảnh như các vector số bằng cách *trải phẳng* (flatten) chúng, bất kể quan hệ không gian giữa các điểm ảnh. Cách làm
rất không thỏa đáng này là cần thiết để có thể đưa
các vector một chiều thu được qua một MLP kết nối đầy đủ.

Vì các mạng này bất biến với thứ tự của các đặc trưng, ta
sẽ thu được kết quả tương tự dù giữ một thứ tự
tương ứng với cấu trúc không gian của các điểm ảnh hay hoán vị
các cột của ma trận thiết kế trước khi khớp các tham số của MLP.
Lý tưởng nhất, ta nên tận dụng kiến thức tiên nghiệm rằng các điểm ảnh lân cận
thường có liên quan với nhau, để xây dựng các mô hình hiệu quả
cho việc học từ dữ liệu ảnh.

Chương này giới thiệu *mạng neuron tích chập* (convolutional neural network – CNN)
([LeCun *et al.*, 1995](dl/ch07_vi?id=ref-lecun-jackel-bottou-ea-1995)), một họ mạng neuron mạnh mẽ
được thiết kế cho chính mục đích này.
Các kiến trúc dựa trên CNN
nay đã có mặt khắp nơi trong lĩnh vực thị giác máy tính.
Chẳng hạn, trên bộ dữ liệu Imagnet
([Deng *et al.*, 2009](dl/ch07_vi?id=ref-deng-dong-socher-ea-2009)), chỉ khi dùng mạng neuron
tích chập, gọi tắt là Convnet, mới đạt được cải thiện
đáng kể về hiệu năng ([Krizhevsky *et al.*, 2012](dl/ch07_vi?id=ref-krizhevsky-sutskever-hinton-2012)).

Các CNN hiện đại, như cách gọi thông thường, có thiết kế bắt nguồn từ
cảm hứng sinh học, lý thuyết nhóm, và không ít
thử nghiệm mày mò. Bên cạnh hiệu quả về mẫu (sample efficiency) khi
đạt được các mô hình chính xác, CNN thường hiệu quả về mặt tính toán,
vừa vì chúng cần ít tham số hơn các kiến trúc
kết nối đầy đủ, vừa vì tích chập dễ song song hóa trên các nhân
GPU ([Chetlur *et al.*, 2014](dl/ch07_vi?id=ref-chetlur-woolley-vandermersch-ea-2014)). Do đó, người làm thực tế thường
áp dụng CNN bất cứ khi nào có thể, và chúng ngày càng trở thành
đối thủ đáng gờm ngay cả với các tác vụ có cấu trúc chuỗi
một chiều, như âm thanh ([Abdel-Hamid *et al.*, 2014](dl/ch07_vi?id=ref-abdel-hamid-mohamed-jiang-ea-2014)), văn bản
([Kalchbrenner *et al.*, 2014](dl/ch07_vi?id=ref-kalchbrenner-grefenstette-blunsom-2014)), và phân tích chuỗi thời gian
([LeCun *et al.*, 1995](dl/ch07_vi?id=ref-lecun-bengio-ea-1995)), vốn là nơi mạng neuron hồi tiếp
thường được dùng. Một số biến thể khéo léo của CNN cũng đã
đưa chúng vào dữ liệu có cấu trúc đồ thị ([Kipf và Welling, 2016](dl/ch07_vi?id=ref-kipf-welling-2016)) và
vào các hệ thống gợi ý.

Trước hết, ta sẽ đi sâu hơn vào động lực của mạng neuron
tích chập. Tiếp theo là phần đi qua các phép toán cơ bản
tạo nên xương sống của mọi mạng tích chập.
Các phép toán này gồm bản thân các tầng tích chập,
những chi tiết tỉ mỉ như đệm (padding) và sải bước (stride),
các tầng gộp (pooling) dùng để tổng hợp thông tin
trên các vùng không gian kề nhau,
việc dùng nhiều kênh ở mỗi tầng,
và một thảo luận cẩn thận về cấu trúc của các kiến trúc hiện đại.
Ta sẽ kết thúc chương bằng một ví dụ hoạt động hoàn chỉnh của LeNet,
mạng tích chập đầu tiên được triển khai thành công,
từ rất lâu trước khi học sâu hiện đại trỗi dậy.
Trong chương tiếp theo, ta sẽ đi sâu vào cài đặt đầy đủ
của một số kiến trúc CNN phổ biến và tương đối mới,
mà thiết kế của chúng đại diện cho hầu hết các kỹ thuật
thường được người làm thực tế hiện nay sử dụng.

## Các mục trong chương

- [7.1 Từ Tầng Kết nối Đầy đủ đến Tích chập](dl/ch07_s1_vi)
- [7.2 Tích chập cho Ảnh](dl/ch07_s2_vi)
- [7.3 Đệm và Sải bước](dl/ch07_s3_vi)
- [7.4 Nhiều Kênh Đầu vào và Nhiều Kênh Đầu ra](dl/ch07_s4_vi)
- [7.5 Gộp](dl/ch07_s5_vi)
- [7.6 Mạng Neuron Tích chập (LeNet)](dl/ch07_s6_vi)

## Tài liệu tham khảo

- <a id="ref-lecun-jackel-bottou-ea-1995"></a>LeCun, Yann, Jackel, LD, Bottou, Leon, ... (1995). *Comparison of learning algorithms for handwritten digit recognition*. International Conference on Artificial Neural Networks.
- <a id="ref-deng-dong-socher-ea-2009"></a>Deng, Jia, Dong, Wei, Socher, Richard, ... (2009). *Imagenet: A large-scale hierarchical image database*. 2009 IEEE Conference on Computer Vision and Pattern Recognition.
- <a id="ref-krizhevsky-sutskever-hinton-2012"></a>Krizhevsky, Alex, Sutskever, Ilya, Hinton, Geoffrey E (2012). *ImageNet classification with deep convolutional neural networks*. Advances in Neural Information Processing Systems.
- <a id="ref-chetlur-woolley-vandermersch-ea-2014"></a>Chetlur, Sharan, Woolley, Cliff, Vandermersch, Philippe, ... (2014). *cuDNN: Efficient primitives for deep learning*. ArXiv:1410.0759.
- <a id="ref-abdel-hamid-mohamed-jiang-ea-2014"></a>Abdel-Hamid, Ossama, Mohamed, Abdel-Rahman, Jiang, Hui, ... (2014). *Convolutional neural networks for speech recognition*. IEEE/ACM Transactions on Audio, Speech, and Language Processing.
- <a id="ref-kalchbrenner-grefenstette-blunsom-2014"></a>Kalchbrenner, Nal, Grefenstette, Edward, Blunsom, Phil (2014). *A convolutional neural network for modelling sentences*. ArXiv:1404.2188.
- <a id="ref-lecun-bengio-ea-1995"></a>LeCun, Yann, Bengio, Yoshua, et al. (1995). *Convolutional networks for images, speech, and time series*. The Handbook of Brain Theory and Neural Networks.
- <a id="ref-kipf-welling-2016"></a>Kipf, Thomas N, Welling, Max (2016). *Semi-supervised classification with graph convolutional networks*. ArXiv:1609.02907.
