# Mạng Neuron Tích chập
:label:`chap_cnn`

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
:cite:`LeCun.Jackel.Bottou.ea.1995`, một họ mạng neuron mạnh mẽ
được thiết kế cho chính mục đích này.
Các kiến trúc dựa trên CNN
nay đã có mặt khắp nơi trong lĩnh vực thị giác máy tính.
Chẳng hạn, trên bộ dữ liệu Imagnet
:cite:`Deng.Dong.Socher.ea.2009`, chỉ khi dùng mạng neuron
tích chập, gọi tắt là Convnet, mới đạt được cải thiện
đáng kể về hiệu năng :cite:`Krizhevsky.Sutskever.Hinton.2012`.

Các CNN hiện đại, như cách gọi thông thường, có thiết kế bắt nguồn từ
cảm hứng sinh học, lý thuyết nhóm, và không ít
thử nghiệm mày mò. Bên cạnh hiệu quả về mẫu (sample efficiency) khi
đạt được các mô hình chính xác, CNN thường hiệu quả về mặt tính toán,
vừa vì chúng cần ít tham số hơn các kiến trúc
kết nối đầy đủ, vừa vì tích chập dễ song song hóa trên các nhân
GPU :cite:`Chetlur.Woolley.Vandermersch.ea.2014`. Do đó, người làm thực tế thường
áp dụng CNN bất cứ khi nào có thể, và chúng ngày càng trở thành
đối thủ đáng gờm ngay cả với các tác vụ có cấu trúc chuỗi
một chiều, như âm thanh :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, văn bản
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014`, và phân tích chuỗi thời gian
:cite:`LeCun.Bengio.ea.1995`, vốn là nơi mạng neuron hồi tiếp
thường được dùng. Một số biến thể khéo léo của CNN cũng đã
đưa chúng vào dữ liệu có cấu trúc đồ thị :cite:`Kipf.Welling.2016` và
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

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
