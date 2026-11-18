# 7.1 Từ Tầng Kết nối Đầy đủ đến Tích chập

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_convolutional-neural-networks/why-conv.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Cho đến nay, các mô hình mà ta đã thảo luận vẫn là những lựa chọn thích hợp khi ta làm việc với dữ liệu dạng bảng. Dữ liệu dạng bảng nghĩa là dữ liệu gồm các hàng tương ứng với các mẫu và các cột tương ứng với các đặc trưng. Với dữ liệu dạng bảng, ta có thể dự đoán rằng các khuôn mẫu ta tìm kiếm có thể liên quan đến tương tác giữa các đặc trưng, nhưng ta không giả định *trước* (a priori) bất kỳ cấu trúc nào về cách các đặc trưng tương tác với nhau.

Đôi khi, ta thực sự thiếu kiến thức để có thể định hướng việc xây dựng những kiến trúc tinh vi hơn. Trong những trường hợp này, một MLP có thể là thứ tốt nhất ta làm được. Tuy nhiên, với dữ liệu cảm nhận (perceptual data) nhiều chiều, các mạng không có cấu trúc như vậy có thể trở nên cồng kềnh.

Chẳng hạn, hãy quay lại ví dụ xuyên suốt của chúng ta về việc phân biệt mèo với chó. Giả sử ta thu thập dữ liệu rất kỹ lưỡng, có được một tập dữ liệu đã gán nhãn gồm các ảnh một megapixel. Điều này nghĩa là mỗi đầu vào của mạng có một triệu chiều. Ngay cả khi giảm mạnh xuống còn một nghìn chiều ẩn, ta vẫn cần một tầng kết nối đầy đủ với $10^6 \times 10^3 = 10^9$ tham số. Trừ khi ta có rất nhiều GPU, có tài tối ưu hóa phân tán, và có lòng kiên nhẫn phi thường, việc học các tham số của mạng này có thể là bất khả thi.

Một độc giả cẩn thận có thể phản bác lập luận này với lý do rằng độ phân giải một megapixel có thể là không cần thiết. Tuy nhiên, dù ta có thể xoay xở với một trăm nghìn điểm ảnh, tầng ẩn kích thước 1000 của ta đã đánh giá quá thấp số nút ẩn cần thiết để học được các biểu diễn tốt của ảnh, nên một hệ thống thực tế vẫn sẽ cần hàng tỷ tham số. Hơn nữa, học một bộ phân loại bằng cách khớp nhiều tham số như vậy có thể đòi hỏi thu thập một tập dữ liệu khổng lồ. Thế nhưng ngày nay cả con người và máy tính đều có thể phân biệt mèo với chó khá tốt, có vẻ mâu thuẫn với những trực giác trên. Đó là vì ảnh thể hiện một cấu trúc phong phú mà cả con người lẫn các mô hình học máy đều có thể khai thác. Mạng neuron tích chập (CNN) là một cách sáng tạo mà học máy đã áp dụng để khai thác một phần cấu trúc đã biết trong ảnh tự nhiên.

## 7.1.1 Tính Bất biến

Hãy tưởng tượng rằng ta muốn phát hiện một vật thể trong ảnh. Có vẻ hợp lý khi bất kỳ phương pháp nào ta dùng để nhận dạng vật thể cũng không nên quá bận tâm đến vị trí chính xác của vật thể trong ảnh. Lý tưởng nhất, hệ thống của ta nên khai thác kiến thức này. Lợn thường không bay và máy bay thường không bơi. Dù vậy, ta vẫn nên nhận ra một con lợn nếu nó xuất hiện ở phía trên cùng của ảnh. Ta có thể lấy cảm hứng ở đây từ trò chơi trẻ em "Waldo ở đâu" (bản thân trò chơi này đã truyền cảm hứng cho nhiều phiên bản ngoài đời thực, như phiên bản được minh họa trong [Hình 7.1.1](dl/ch07_s1_vi?id=img-waldo)). Trò chơi gồm một số cảnh hỗn loạn đầy ắp các hoạt động. Waldo xuất hiện ở đâu đó trong mỗi cảnh, thường ẩn nấp ở một vị trí khó ngờ. Mục tiêu của người đọc là tìm ra anh ta. Dù Waldo có bộ trang phục đặc trưng, việc này có thể khó một cách bất ngờ do có quá nhiều thứ gây xao nhãng. Tuy nhiên, *Waldo trông như thế nào* không phụ thuộc vào *Waldo ở đâu*. Ta có thể quét qua ảnh bằng một bộ phát hiện Waldo, gán cho mỗi mảng ảnh một điểm số cho biết khả năng mảng đó chứa Waldo. Trên thực tế, nhiều thuật toán phát hiện và phân đoạn vật thể dựa trên cách tiếp cận này ([Long *et al.*, 2015](dl/ch07_s1_vi?id=ref-long-shelhamer-darrell-2015)). CNN hệ thống hóa ý tưởng về *tính bất biến không gian* (spatial invariance) này, khai thác nó để học các biểu diễn hữu ích với ít tham số hơn.

<div align=center id="img-waldo">
<img width="400" src="../img/dl/waldo-football.jpg" alt="Hình 7.1.1"/>
</div>
<div align=center>Hình 7.1.1: Bạn có tìm được Waldo không (ảnh của William Murphy (Infomatique))?</div>

Giờ ta có thể làm cho những trực giác này cụ thể hơn bằng cách liệt kê một vài yêu cầu mong muốn để định hướng việc thiết kế một kiến trúc mạng neuron phù hợp cho thị giác máy tính:

1. Ở những tầng đầu tiên, mạng của ta nên phản hồi giống nhau với cùng một mảng ảnh, bất kể nó xuất hiện ở đâu trong ảnh. Nguyên tắc này được gọi là *bất biến tịnh tiến* (translation invariance) (hay *đẳng biến tịnh tiến* – translation equivariance).
1. Những tầng đầu tiên của mạng nên tập trung vào các vùng cục bộ, không quan tâm đến nội dung của ảnh ở các vùng xa. Đây là nguyên tắc *tính cục bộ* (locality). Cuối cùng, các biểu diễn cục bộ này có thể được tổng hợp lại để đưa ra dự đoán ở cấp độ toàn ảnh.
1. Khi đi sâu hơn, các tầng sâu hơn nên có khả năng nắm bắt các đặc trưng có phạm vi rộng hơn của ảnh, theo cách tương tự như thị giác cấp cao trong tự nhiên.

Hãy xem những điều này được chuyển thành toán học như thế nào.

## 7.1.2 Ràng buộc MLP

Để bắt đầu, ta có thể xét một MLP với đầu vào là các ảnh hai chiều $\mathbf{X}$ và các biểu diễn ẩn trực tiếp $\mathbf{H}$ của chúng cũng được biểu diễn tương tự dưới dạng ma trận (chúng là các tensor hai chiều trong code), trong đó cả $\mathbf{X}$ và $\mathbf{H}$ có cùng kích thước. Hãy ngẫm kỹ điều này. Giờ ta hình dung rằng không chỉ đầu vào mà cả các biểu diễn ẩn cũng có cấu trúc không gian.

Gọi $[\mathbf{X}]_{i, j}$ và $[\mathbf{H}]_{i, j}$ lần lượt là điểm ảnh tại vị trí $(i,j)$ trong ảnh đầu vào và trong biểu diễn ẩn. Do đó, để mỗi nút ẩn nhận đầu vào từ mỗi điểm ảnh đầu vào, ta sẽ chuyển từ việc dùng các ma trận trọng số (như ta đã làm trước đây với MLP) sang biểu diễn các tham số dưới dạng tensor trọng số bậc bốn $\mathsf{W}$. Giả sử $\mathbf{U}$ chứa các hệ số điều chỉnh (bias), ta có thể biểu diễn tầng kết nối đầy đủ một cách hình thức như sau

$$
\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}
$$

Việc chuyển từ $\mathsf{W}$ sang $\mathsf{V}$ hiện tại chỉ mang tính hình thức vì có một tương ứng một-một giữa các hệ số trong cả hai tensor bậc bốn. Ta chỉ đơn giản đánh lại chỉ số $(k, l)$ sao cho $k = i+a$ và $l = j+b$. Nói cách khác, ta đặt $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$. Các chỉ số $a$ và $b$ chạy qua cả độ lệch dương và âm, bao phủ toàn bộ ảnh. Với bất kỳ vị trí ($i$, $j$) nào trong biểu diễn ẩn $[\mathbf{H}]_{i, j}$, ta tính giá trị của nó bằng cách lấy tổng các điểm ảnh trong $x$, có tâm quanh $(i, j)$ và được đánh trọng số bởi $[\mathsf{V}]_{i, j, a, b}$. Trước khi tiếp tục, hãy xét tổng số tham số cần cho *một* tầng duy nhất trong cách tham số hóa này: một ảnh $1000 \times 1000$ (1 megapixel) được ánh xạ thành một biểu diễn ẩn $1000 \times 1000$. Điều này đòi hỏi $10^{12}$ tham số, vượt xa khả năng xử lý của máy tính hiện nay.

### 7.1.2.1 Bất biến Tịnh tiến

Giờ hãy áp dụng nguyên tắc đầu tiên đã nêu ở trên: bất biến tịnh tiến ([Zhang *et al.*, 1988](dl/ch07_s1_vi?id=ref-zhang-ea-1988)). Điều này hàm ý rằng một sự dịch chuyển ở đầu vào $\mathbf{X}$ chỉ đơn giản dẫn đến một sự dịch chuyển ở biểu diễn ẩn $\mathbf{H}$. Điều này chỉ có thể xảy ra nếu $\mathsf{V}$ và $\mathbf{U}$ thực ra không phụ thuộc vào $(i, j)$. Do đó, ta có $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ và $\mathbf{U}$ là một hằng số, chẳng hạn $u$. Kết quả là ta có thể đơn giản hóa định nghĩa của $\mathbf{H}$:

$$
[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.
$$

Đây chính là một phép *tích chập* (convolution)! Về thực chất, ta đang đánh trọng số các điểm ảnh tại $(i+a, j+b)$ ở lân cận vị trí $(i, j)$ với các hệ số $[\mathbf{V}]_{a, b}$ để thu được giá trị $[\mathbf{H}]_{i, j}$. Lưu ý rằng $[\mathbf{V}]_{a, b}$ cần ít hệ số hơn nhiều so với $[\mathsf{V}]_{i, j, a, b}$ vì nó không còn phụ thuộc vào vị trí trong ảnh. Do đó, số tham số cần thiết không còn là $10^{12}$ mà là con số hợp lý hơn nhiều $4 \times 10^6$: ta vẫn còn sự phụ thuộc vào $a, b \in (-1000, 1000)$. Tóm lại, ta đã tiến được một bước đáng kể. Mạng neuron trễ thời gian (time-delay neural network – TDNN) là một trong những ví dụ đầu tiên khai thác ý tưởng này ([Waibel *et al.*, 1989](dl/ch07_s1_vi?id=ref-waibel-hanazawa-hinton-ea-1989)).

### 7.1.2.2 Tính Cục bộ

Giờ hãy áp dụng nguyên tắc thứ hai: tính cục bộ. Như đã lập luận ở trên, ta tin rằng không cần nhìn quá xa vị trí $(i, j)$ để thu thập thông tin liên quan nhằm đánh giá điều gì đang diễn ra tại $[\mathbf{H}]_{i, j}$. Điều này nghĩa là bên ngoài một phạm vi $|a|> \Delta$ hoặc $|b| > \Delta$, ta nên đặt $[\mathbf{V}]_{a, b} = 0$. Một cách tương đương, ta có thể viết lại $[\mathbf{H}]_{i, j}$ như sau

$$
[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}. \tag{7.1.1}
$$

Điều này giảm số tham số từ $4 \times 10^6$ xuống $4 \Delta^2$, trong đó $\Delta$ thường nhỏ hơn $10$. Như vậy, ta đã giảm số tham số thêm bốn bậc độ lớn nữa. Lưu ý rằng [(7.1.1)](dl/ch07_s1_vi), nói gọn lại, chính là thứ được gọi là một *tầng tích chập* (convolutional layer). *Mạng neuron tích chập* (CNN) là một họ mạng neuron đặc biệt có chứa các tầng tích chập. Trong cộng đồng nghiên cứu học sâu, $\mathbf{V}$ được gọi là *kernel tích chập* (convolution kernel), *bộ lọc* (filter), hoặc đơn giản là *trọng số* của tầng, tức các tham số học được.

Trong khi trước đây ta có thể cần hàng tỷ tham số chỉ để biểu diễn một tầng duy nhất trong một mạng xử lý ảnh, giờ đây ta thường chỉ cần vài trăm tham số mà không làm thay đổi số chiều của đầu vào hay của các biểu diễn ẩn. Cái giá phải trả cho sự cắt giảm mạnh mẽ về số tham số này là các đặc trưng của ta giờ đây bất biến tịnh tiến và tầng của ta chỉ có thể kết hợp thông tin cục bộ khi xác định giá trị của mỗi kích hoạt ẩn. Mọi quá trình học đều phụ thuộc vào việc áp đặt thiên kiến quy nạp (inductive bias). Khi thiên kiến đó khớp với thực tế, ta có được các mô hình hiệu quả về mẫu, tổng quát hóa tốt với dữ liệu chưa thấy. Nhưng tất nhiên, nếu những thiên kiến đó không khớp với thực tế, chẳng hạn nếu hóa ra ảnh không bất biến tịnh tiến, các mô hình của ta có thể gặp khó khăn ngay cả khi khớp dữ liệu huấn luyện.

Sự cắt giảm mạnh mẽ về số tham số này đưa ta đến yêu cầu mong muốn cuối cùng, cụ thể là các tầng sâu hơn nên biểu diễn những khía cạnh lớn hơn và phức tạp hơn của ảnh. Điều này có thể đạt được bằng cách xen kẽ lặp đi lặp lại các hàm phi tuyến và các tầng tích chập.

## 7.1.3 Tích chập

Hãy điểm qua lý do vì sao [(7.1.1)](dl/ch07_s1_vi) được gọi là một phép tích chập. Trong toán học, *tích chập* giữa hai hàm số ([W.~Rudin, ](dl/ch07_s1_vi?id=ref-rudin-1973)), chẳng hạn $f, g: \mathbb{R}^d \to \mathbb{R}$, được định nghĩa là

$$
(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.
$$

Tức là, ta đo phần chồng lấn giữa $f$ và $g$ khi một hàm bị "lật" và dịch đi một đoạn $\mathbf{x}$. Bất cứ khi nào ta có các đối tượng rời rạc, tích phân trở thành tổng. Chẳng hạn, với các vector thuộc tập các vector vô hạn chiều có tổng bình phương hữu hạn, với chỉ số chạy trên $\mathbb{Z}$, ta có định nghĩa sau:

$$
(f * g)(i) = \sum_a f(a) g(i-a).
$$

Với các tensor hai chiều, ta có một tổng tương ứng với các chỉ số lần lượt là $(a, b)$ cho $f$ và $(i-a, j-b)$ cho $g$:

$$
(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b). \tag{7.1.2}
$$

Công thức này trông giống [(7.1.1)](dl/ch07_s1_vi), với một khác biệt lớn. Thay vì dùng $(i+a, j+b)$, ta dùng hiệu. Tuy nhiên, lưu ý rằng sự khác biệt này phần lớn chỉ mang tính hình thức vì ta luôn có thể làm cho ký hiệu giữa [(7.1.1)](dl/ch07_s1_vi) và [(7.1.2)](dl/ch07_s1_vi) khớp nhau. Định nghĩa ban đầu của ta trong [(7.1.1)](dl/ch07_s1_vi) mô tả đúng hơn một phép *tương quan chéo* (cross-correlation). Ta sẽ quay lại vấn đề này trong mục tiếp theo.

## 7.1.4 Kênh

Quay lại bộ phát hiện Waldo, hãy xem nó trông như thế nào. Tầng tích chập chọn các cửa sổ có kích thước cho trước và đánh trọng số các cường độ theo bộ lọc $\mathsf{V}$, như minh họa trong [Hình 7.1.2](dl/ch07_s1_vi?id=fig-waldo-mask). Ta có thể đặt mục tiêu học một mô hình sao cho ở bất cứ đâu "độ Waldo" cao nhất, ta sẽ thấy một đỉnh trong các biểu diễn của tầng ẩn.

<div align=center id="fig-waldo-mask">
<img width="400" src="../img/dl/waldo-mask.jpg" alt="Hình 7.1.2"/>
</div>
<div align=center>Hình 7.1.2: Phát hiện Waldo (ảnh của William Murphy (Infomatique)).</div>

Chỉ có một vấn đề với cách tiếp cận này. Cho đến nay, ta đã vô tư bỏ qua việc ảnh gồm ba kênh: đỏ, lục và lam. Tóm lại, ảnh không phải là các đối tượng hai chiều mà là các tensor bậc ba, được đặc trưng bởi chiều cao, chiều rộng và kênh, chẳng hạn với kích thước $1024 \times 1024 \times 3$ điểm ảnh. Trong khi hai trục đầu liên quan đến quan hệ không gian, trục thứ ba có thể được xem như gán một biểu diễn nhiều chiều cho mỗi vị trí điểm ảnh. Do đó ta đánh chỉ số $\mathsf{X}$ là $[\mathsf{X}]_{i, j, k}$. Bộ lọc tích chập cũng phải thích ứng tương ứng. Thay vì $[\mathbf{V}]_{a,b}$, giờ ta có $[\mathsf{V}]_{a,b,c}$.

Hơn nữa, cũng như đầu vào là một tensor bậc ba, hóa ra việc biểu diễn các biểu diễn ẩn dưới dạng tensor bậc ba $\mathsf{H}$ tương tự là một ý tưởng hay. Nói cách khác, thay vì chỉ có một biểu diễn ẩn duy nhất tương ứng với mỗi vị trí không gian, ta muốn có cả một vector biểu diễn ẩn tương ứng với mỗi vị trí không gian. Ta có thể hình dung các biểu diễn ẩn gồm nhiều lưới hai chiều xếp chồng lên nhau. Giống như ở đầu vào, chúng đôi khi được gọi là các *kênh* (channel). Đôi khi chúng còn được gọi là các *ánh xạ đặc trưng* (feature map), vì mỗi kênh cung cấp một tập đặc trưng đã học được, có tổ chức theo không gian, cho tầng tiếp theo. Một cách trực quan, bạn có thể tưởng tượng rằng ở các tầng thấp gần đầu vào, một số kênh có thể chuyên nhận dạng các cạnh trong khi các kênh khác có thể nhận dạng các kết cấu.

Để hỗ trợ nhiều kênh ở cả đầu vào ($\mathsf{X}$) và các biểu diễn ẩn ($\mathsf{H}$), ta có thể thêm một tọa độ thứ tư vào $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$. Gộp mọi thứ lại, ta có:

$$
[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c}, \tag{7.1.3}
$$

trong đó $d$ đánh chỉ số các kênh đầu ra trong các biểu diễn ẩn $\mathsf{H}$. Tầng tích chập tiếp theo sẽ nhận một tensor bậc ba, $\mathsf{H}$, làm đầu vào. Vì tính tổng quát của nó, ta lấy [(7.1.3)](dl/ch07_s1_vi) làm định nghĩa của một tầng tích chập nhiều kênh, trong đó $\mathsf{V}$ là kernel hay bộ lọc của tầng.

Vẫn còn nhiều phép toán mà ta cần giải quyết. Chẳng hạn, ta cần tìm ra cách kết hợp tất cả các biểu diễn ẩn thành một đầu ra duy nhất, ví dụ như liệu có Waldo ở *bất cứ đâu* trong ảnh hay không. Ta cũng cần quyết định cách tính toán hiệu quả, cách kết hợp nhiều tầng, các hàm kích hoạt thích hợp, và cách đưa ra các lựa chọn thiết kế hợp lý để tạo ra các mạng hiệu quả trong thực tế. Ta sẽ chuyển sang những vấn đề này trong phần còn lại của chương.

## 7.1.5 Tóm tắt và Thảo luận

Trong mục này, ta đã suy ra cấu trúc của mạng neuron tích chập từ các nguyên lý cơ bản. Dù không rõ đây có phải con đường dẫn đến việc phát minh ra CNN hay không, thật thỏa mãn khi biết rằng chúng là lựa chọn *đúng đắn* khi áp dụng những nguyên lý hợp lý về cách các thuật toán xử lý ảnh và thị giác máy tính nên hoạt động, ít nhất là ở các cấp thấp. Cụ thể, tính bất biến tịnh tiến trong ảnh hàm ý rằng mọi mảng của một ảnh sẽ được xử lý theo cùng một cách. Tính cục bộ nghĩa là chỉ một vùng lân cận nhỏ các điểm ảnh được dùng để tính các biểu diễn ẩn tương ứng. Một số tài liệu sớm nhất nhắc đến CNN là dưới dạng Neocognitron ([Fukushima, 1982](dl/ch07_s1_vi?id=ref-fukushima-1982)).

Nguyên lý thứ hai mà ta gặp trong quá trình suy luận là làm thế nào để giảm số tham số trong một lớp hàm mà không giới hạn khả năng biểu diễn của nó, ít nhất là khi những giả định nhất định về mô hình được thỏa mãn. Ta đã thấy độ phức tạp giảm mạnh nhờ ràng buộc này, biến các bài toán bất khả thi về mặt tính toán và thống kê thành các mô hình xử lý được.

Việc thêm kênh cho phép ta đưa trở lại một phần độ phức tạp đã mất do các ràng buộc về tính cục bộ và bất biến tịnh tiến áp đặt lên kernel tích chập. Lưu ý rằng việc thêm các kênh khác ngoài đỏ, lục và lam là khá tự nhiên. Nhiều ảnh vệ tinh, đặc biệt là cho nông nghiệp và khí tượng học, có hàng chục đến hàng trăm kênh, tạo thành các ảnh siêu phổ (hyperspectral image). Chúng ghi nhận dữ liệu ở nhiều bước sóng khác nhau. Sau đây ta sẽ thấy cách dùng tích chập hiệu quả để thay đổi số chiều của ảnh mà chúng xử lý, cách chuyển từ biểu diễn dựa trên vị trí sang biểu diễn dựa trên kênh, và cách xử lý hiệu quả số lượng lớn các lớp.

## 7.1.6 Bài tập

1. Giả sử kích thước của kernel tích chập là $\Delta = 0$. Chứng minh rằng trong trường hợp này kernel tích chập cài đặt một MLP độc lập cho mỗi tập kênh. Điều này dẫn đến các kiến trúc Network in Network ([Lin *et al.*, 2013](dl/ch07_s1_vi?id=ref-lin-chen-yan-2013)).
1. Dữ liệu âm thanh thường được biểu diễn dưới dạng một chuỗi một chiều.
    1. Khi nào bạn có thể muốn áp đặt tính cục bộ và bất biến tịnh tiến cho âm thanh?
    1. Suy ra các phép tích chập cho âm thanh.
    1. Bạn có thể xử lý âm thanh bằng cùng các công cụ như thị giác máy tính không? Gợi ý: dùng phổ đồ (spectrogram).
1. Vì sao rốt cuộc bất biến tịnh tiến có thể không phải là một ý tưởng hay? Hãy cho một ví dụ.
1. Bạn có nghĩ rằng các tầng tích chập cũng có thể áp dụng cho dữ liệu văn bản không? Bạn có thể gặp những vấn đề gì với ngôn ngữ?
1. Điều gì xảy ra với tích chập khi một vật thể nằm ở biên của ảnh?
1. Chứng minh rằng tích chập có tính đối xứng, tức là $f * g = g * f$.

[Thảo luận (tiếng Anh)](https://discuss.d2l.ai/t/64)

## Tài liệu tham khảo

- <a id="ref-long-shelhamer-darrell-2015"></a>Long, Jonathan, Shelhamer, Evan, Darrell, Trevor (2015). *Fully convolutional networks for semantic segmentation*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
- <a id="ref-zhang-ea-1988"></a>Zhang, Wei, Tanida, Jun, Itoh, Kazuyoshi, ... (1988). *Shift-invariant pattern recognition neural network and its optical architecture*. Proceedings of Annual Conference of the Japan Society of Applied Physics.
- <a id="ref-waibel-hanazawa-hinton-ea-1989"></a>Waibel, Alex, Hanazawa, Toshiyuki, Hinton, Geoffrey, ... (1989). *Phoneme recognition using time-delay neural networks*. IEEE Transactions on Acoustics, Speech, and Signal Processing.
- <a id="ref-rudin-1973"></a>W.~Rudin (). *Functional Analysis*. McGraw-Hill.
- <a id="ref-fukushima-1982"></a>Fukushima, Kunihiko (1982). *Neocognitron: A self-organizing neural network model for a mechanism of visual pattern recognition*. Competition and Cooperation in Neural Nets.
- <a id="ref-lin-chen-yan-2013"></a>Lin, Min, Chen, Qiang, Yan, Shuicheng (2013). *Network in network*. ArXiv:1312.4400.
