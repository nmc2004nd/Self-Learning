# Tổng quát hóa
:label:`sec_generalization_basics`

Hãy xét hai sinh viên đại học đang chăm chỉ ôn thi cuối kỳ. Thông thường, việc ôn tập gồm luyện tập và kiểm tra khả năng bằng cách làm đề thi của những năm trước. Dù vậy, làm tốt các đề thi cũ không đảm bảo rằng họ sẽ xuất sắc vào lúc quan trọng. Chẳng hạn, hãy tưởng tượng một sinh viên, Ellie Phi thường, chỉ ôn tập bằng cách học thuộc đáp án của các câu hỏi trong đề thi những năm trước. Ngay cả khi Ellie được trời phú một trí nhớ phi thường, và do đó có thể nhớ hoàn hảo đáp án của bất kỳ câu hỏi nào *đã từng gặp*, cô vẫn có thể đứng hình khi đối mặt với một câu hỏi mới (*chưa từng gặp*). Để so sánh, hãy tưởng tượng một sinh viên khác, Irene Quy nạp, có khả năng ghi nhớ kém tương đương, nhưng lại có năng khiếu nhận ra khuôn mẫu. Lưu ý rằng nếu đề thi thực sự gồm những câu hỏi lặp lại từ năm trước, Ellie sẽ dễ dàng vượt trội Irene. Ngay cả khi các khuôn mẫu mà Irene suy ra cho dự đoán chính xác 90%, chúng cũng không bao giờ cạnh tranh nổi với khả năng nhớ 100% của Ellie. Tuy nhiên, ngay cả khi đề thi gồm toàn câu hỏi mới, Irene vẫn có thể giữ được mức trung bình 90%.

Là những nhà khoa học học máy, mục tiêu của ta là khám phá các *khuôn mẫu*. Nhưng làm sao ta chắc chắn rằng mình đã thực sự khám phá ra một khuôn mẫu *tổng quát* chứ không chỉ đơn giản là học thuộc dữ liệu? Hầu hết thời gian, dự đoán của ta chỉ hữu ích nếu mô hình khám phá ra một khuôn mẫu như vậy. Ta không muốn dự đoán giá cổ phiếu hôm qua, mà là ngày mai. Ta không cần nhận ra những bệnh đã được chẩn đoán ở những bệnh nhân đã gặp, mà là những bệnh chưa được chẩn đoán ở những bệnh nhân chưa từng gặp. Vấn đề này, làm sao khám phá ra những khuôn mẫu có thể *tổng quát hóa* (generalize), là vấn đề cơ bản của học máy, và có thể nói là của toàn bộ thống kê học. Ta có thể coi vấn đề này chỉ là một lát cắt của một câu hỏi lớn hơn nhiều bao trùm toàn bộ khoa học: khi nào ta mới có cơ sở để nhảy từ những quan sát cụ thể đến những phát biểu tổng quát hơn?

Trong đời thực, ta phải khớp mô hình bằng một tập dữ liệu hữu hạn. Quy mô điển hình của dữ liệu đó khác nhau rất nhiều giữa các lĩnh vực. Với nhiều bài toán y khoa quan trọng, ta chỉ có thể tiếp cận vài nghìn điểm dữ liệu. Khi nghiên cứu các bệnh hiếm, có được vài trăm điểm đã là may mắn. Ngược lại, những tập dữ liệu công khai lớn nhất gồm các bức ảnh đã gán nhãn, chẳng hạn ImageNet :cite:`Deng.Dong.Socher.ea.2009`, chứa hàng triệu ảnh. Và một số bộ sưu tập ảnh chưa gán nhãn như tập dữ liệu Flickr YFC100M còn có thể lớn hơn nữa, chứa hơn 100 triệu ảnh :cite:`thomee2016yfcc100m`. Tuy nhiên, ngay cả ở quy mô cực lớn này, số điểm dữ liệu sẵn có vẫn nhỏ không đáng kể so với không gian mọi bức ảnh có thể có ở độ phân giải một megapixel. Bất cứ khi nào làm việc với các mẫu hữu hạn, ta phải luôn nhớ đến rủi ro là ta có thể khớp được dữ liệu huấn luyện, chỉ để rồi phát hiện ra rằng mình đã không khám phá được một khuôn mẫu có thể tổng quát hóa.

Hiện tượng khớp sát với dữ liệu huấn luyện hơn là với phân phối nền tảng được gọi là *quá khớp* (overfitting), và các kỹ thuật chống quá khớp thường được gọi là các phương pháp *điều chuẩn* (regularization). Dù không thể thay thế cho một phần giới thiệu bài bản về lý thuyết học thống kê (xem :citet:`Vapnik98,boucheron2005theory`), chúng tôi sẽ cung cấp cho bạn vừa đủ trực giác để bắt đầu. Ta sẽ quay lại vấn đề tổng quát hóa trong nhiều chương của cuốn sách, tìm hiểu cả những gì đã biết về các nguyên lý nền tảng của tổng quát hóa trong nhiều mô hình khác nhau, lẫn các kỹ thuật kinh nghiệm đã được phát hiện (bằng thực nghiệm) là giúp cải thiện tổng quát hóa trên những tác vụ có ý nghĩa thực tế.

## Lỗi Huấn luyện và Lỗi Tổng quát hóa

Trong bối cảnh học có giám sát tiêu chuẩn, ta giả định rằng dữ liệu huấn luyện và dữ liệu kiểm tra được lấy *độc lập* từ các phân phối *giống hệt nhau*. Điều này thường được gọi là *giả định IID* (independent and identically distributed). Dù giả định này là mạnh, cần lưu ý rằng nếu không có bất kỳ giả định nào như vậy, ta sẽ bó tay. Vì sao ta nên tin rằng dữ liệu huấn luyện được lấy mẫu từ phân phối $P(X,Y)$ lại cho ta biết cách đưa ra dự đoán trên dữ liệu kiểm tra được sinh ra bởi một *phân phối khác* $Q(X,Y)$? Hóa ra để thực hiện những bước nhảy như vậy, cần có những giả định mạnh về cách $P$ và $Q$ liên hệ với nhau. Sau này ta sẽ thảo luận một số giả định cho phép phân phối dịch chuyển, nhưng trước tiên ta cần hiểu trường hợp IID, khi $P(\cdot) = Q(\cdot)$.

Để bắt đầu, ta cần phân biệt giữa *lỗi huấn luyện* (training error) $R_\textrm{emp}$, là một *thống kê* được tính trên tập dữ liệu huấn luyện, và *lỗi tổng quát hóa* (generalization error) $R$, là một *kỳ vọng* lấy theo phân phối nền tảng. Bạn có thể coi lỗi tổng quát hóa là thứ bạn sẽ thấy nếu áp dụng mô hình lên một dòng vô hạn các mẫu dữ liệu bổ sung được lấy từ cùng phân phối dữ liệu nền tảng. Một cách hình thức, lỗi huấn luyện được biểu diễn dưới dạng *tổng* (với cùng ký hiệu như :numref:`sec_linear_regression`):

$$R_\textrm{emp}[\mathbf{X}, \mathbf{y}, f] = \frac{1}{n} \sum_{i=1}^n l(\mathbf{x}^{(i)}, y^{(i)}, f(\mathbf{x}^{(i)})),$$

trong khi lỗi tổng quát hóa được biểu diễn dưới dạng tích phân:

$$R[p, f] = E_{(\mathbf{x}, y) \sim P} [l(\mathbf{x}, y, f(\mathbf{x}))] =
\int \int l(\mathbf{x}, y, f(\mathbf{x})) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

Vấn đề là ta không bao giờ tính được chính xác lỗi tổng quát hóa $R$. Không ai cho ta biết dạng chính xác của hàm mật độ $p(\mathbf{x}, y)$. Hơn nữa, ta không thể lấy mẫu một dòng vô hạn các điểm dữ liệu. Do đó, trong thực tế, ta phải *ước lượng* lỗi tổng quát hóa bằng cách áp dụng mô hình lên một tập kiểm tra độc lập, gồm các mẫu $\mathbf{X}'$ và nhãn $\mathbf{y}'$ được chọn ngẫu nhiên và giữ riêng khỏi tập huấn luyện. Việc này gồm áp dụng cùng công thức đã dùng để tính lỗi huấn luyện thực nghiệm nhưng trên tập kiểm tra $\mathbf{X}', \mathbf{y}'$.

Điều cốt yếu là khi đánh giá bộ phân loại trên tập kiểm tra, ta đang làm việc với một bộ phân loại *cố định* (nó không phụ thuộc vào mẫu của tập kiểm tra), và do đó việc ước lượng lỗi của nó đơn giản chỉ là bài toán ước lượng giá trị trung bình. Tuy nhiên, không thể nói điều tương tự với tập huấn luyện. Lưu ý rằng mô hình mà ta thu được phụ thuộc tường minh vào việc chọn tập huấn luyện, và do đó lỗi huấn luyện nói chung sẽ là một ước lượng chệch của lỗi thực sự trên tổng thể. Câu hỏi trọng tâm của tổng quát hóa khi đó là khi nào ta nên mong đợi lỗi huấn luyện gần với lỗi trên tổng thể (và do đó gần với lỗi tổng quát hóa).

### Độ phức tạp của Mô hình

Theo lý thuyết cổ điển, khi có mô hình đơn giản và dữ liệu dồi dào, lỗi huấn luyện và lỗi tổng quát hóa thường gần nhau. Tuy nhiên, khi làm việc với mô hình phức tạp hơn và/hoặc ít mẫu hơn, ta dự đoán lỗi huấn luyện sẽ giảm nhưng khoảng cách tổng quát hóa sẽ tăng. Điều này không có gì đáng ngạc nhiên. Hãy tưởng tượng một lớp mô hình có khả năng biểu diễn mạnh đến mức với bất kỳ tập dữ liệu $n$ mẫu nào, ta đều tìm được một bộ tham số khớp hoàn hảo với những nhãn tùy ý, kể cả khi được gán ngẫu nhiên. Trong trường hợp này, ngay cả khi khớp hoàn hảo dữ liệu huấn luyện, làm sao ta có thể kết luận được gì về lỗi tổng quát hóa? Biết đâu lỗi tổng quát hóa của ta cũng chẳng tốt hơn đoán bừa.

Nói chung, nếu không có ràng buộc nào lên lớp mô hình, ta không thể kết luận, chỉ dựa trên việc khớp dữ liệu huấn luyện, rằng mô hình đã khám phá ra một khuôn mẫu có thể tổng quát hóa :cite:`vapnik1994measuring`. Mặt khác, nếu lớp mô hình của ta không có khả năng khớp những nhãn tùy ý, thì hẳn nó đã khám phá ra một khuôn mẫu. Các ý tưởng lý thuyết học về độ phức tạp mô hình đã lấy một phần cảm hứng từ ý tưởng của Karl Popper, một triết gia khoa học có ảnh hưởng lớn, người đã hình thức hóa tiêu chí khả bác bỏ (falsifiability). Theo Popper, một lý thuyết có thể giải thích mọi quan sát thì hoàn toàn không phải là một lý thuyết khoa học! Suy cho cùng, nó đã cho ta biết điều gì về thế giới nếu nó không loại trừ bất kỳ khả năng nào? Tóm lại, điều ta muốn là một giả thuyết *không thể* giải thích mọi quan sát mà ta có thể hình dung ra, nhưng lại tình cờ tương thích với những quan sát mà ta *thực sự* có được.

Còn chính xác thì điều gì tạo nên một khái niệm phù hợp về độ phức tạp mô hình lại là một vấn đề phức tạp. Thường thì, các mô hình có nhiều tham số hơn có khả năng khớp nhiều nhãn được gán tùy ý hơn. Tuy nhiên, điều này không nhất thiết đúng. Chẳng hạn, các phương pháp kernel hoạt động trong những không gian có vô số tham số, nhưng độ phức tạp của chúng được kiểm soát bằng những cách khác :cite:`Scholkopf.Smola.2002`. Một khái niệm độ phức tạp thường tỏ ra hữu ích là khoảng giá trị mà các tham số có thể nhận. Ở đây, một mô hình có tham số được phép nhận giá trị tùy ý sẽ phức tạp hơn. Ta sẽ quay lại ý tưởng này trong mục tiếp theo, khi giới thiệu *suy giảm trọng số* (weight decay), kỹ thuật điều chuẩn thực tế đầu tiên của bạn. Đáng chú ý là có thể khó so sánh độ phức tạp giữa các thành viên của những lớp mô hình khác nhau đáng kể (chẳng hạn cây quyết định và mạng neuron).

Đến đây, ta phải nhấn mạnh một điểm quan trọng khác mà ta sẽ quay lại khi giới thiệu mạng neuron sâu. Khi một mô hình có khả năng khớp những nhãn tùy ý, lỗi huấn luyện thấp không nhất thiết hàm ý lỗi tổng quát hóa thấp. *Tuy nhiên, nó cũng không nhất thiết hàm ý lỗi tổng quát hóa cao!* Tất cả những gì ta có thể nói chắc chắn là chỉ riêng lỗi huấn luyện thấp thì không đủ để khẳng định lỗi tổng quát hóa thấp. Hóa ra mạng neuron sâu chính là những mô hình như vậy: dù chúng tổng quát hóa tốt trong thực tế, chúng quá mạnh để ta có thể kết luận nhiều điều chỉ dựa trên lỗi huấn luyện. Trong những trường hợp này, ta phải dựa nhiều hơn vào dữ liệu được giữ riêng để xác nhận khả năng tổng quát hóa sau khi đã huấn luyện. Lỗi trên dữ liệu giữ riêng, tức tập xác thực (validation set), được gọi là *lỗi xác thực* (validation error).

## Dưới khớp hay Quá khớp?

Khi so sánh lỗi huấn luyện và lỗi xác thực, ta cần lưu tâm đến hai tình huống thường gặp. Thứ nhất, ta cần để ý những trường hợp mà lỗi huấn luyện và lỗi xác thực đều đáng kể nhưng khoảng cách giữa chúng nhỏ. Nếu mô hình không thể giảm lỗi huấn luyện, điều đó có thể nghĩa là mô hình quá đơn giản (tức là không đủ khả năng biểu diễn) để nắm bắt khuôn mẫu mà ta đang cố mô hình hóa. Hơn nữa, vì *khoảng cách tổng quát hóa* (generalization gap) ($R_\textrm{emp} - R$) giữa lỗi huấn luyện và lỗi tổng quát hóa nhỏ, ta có lý do để tin rằng có thể dùng một mô hình phức tạp hơn. Hiện tượng này được gọi là *dưới khớp* (underfitting).

Mặt khác, như đã thảo luận ở trên, ta cần để ý những trường hợp mà lỗi huấn luyện thấp hơn đáng kể so với lỗi xác thực, cho thấy *quá khớp* nghiêm trọng. Lưu ý rằng quá khớp không phải lúc nào cũng là điều xấu. Đặc biệt trong học sâu, những mô hình dự đoán tốt nhất thường hoạt động tốt hơn nhiều trên dữ liệu huấn luyện so với dữ liệu giữ riêng. Sau cùng, ta thường quan tâm đến việc giảm lỗi tổng quát hóa, và chỉ quan tâm đến khoảng cách trong chừng mực nó trở thành trở ngại cho mục tiêu đó. Lưu ý rằng nếu lỗi huấn luyện bằng không, khoảng cách tổng quát hóa chính xác bằng lỗi tổng quát hóa và ta chỉ có thể tiến bộ bằng cách giảm khoảng cách đó.

### Khớp Đường cong Đa thức
:label:`subsec_polynomial-curve-fitting`

Để minh họa một số trực giác cổ điển về quá khớp và độ phức tạp mô hình, hãy xét bài toán sau: cho dữ liệu huấn luyện gồm một đặc trưng duy nhất $x$ và một nhãn giá trị thực $y$ tương ứng, ta cố gắng tìm đa thức bậc $d$

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

để ước lượng nhãn $y$. Đây chỉ là một bài toán hồi quy tuyến tính trong đó các đặc trưng là các lũy thừa của $x$, trọng số của mô hình là $w_i$, và hệ số điều chỉnh là $w_0$ vì $x^0 = 1$ với mọi $x$. Vì đây chỉ là một bài toán hồi quy tuyến tính, ta có thể dùng bình phương sai số làm hàm mất mát.

Một hàm đa thức bậc cao phức tạp hơn một hàm đa thức bậc thấp, vì đa thức bậc cao có nhiều tham số hơn và phạm vi lựa chọn hàm mô hình rộng hơn. Với tập dữ liệu huấn luyện cố định, các hàm đa thức bậc cao luôn đạt lỗi huấn luyện thấp hơn (tệ nhất là bằng) so với các đa thức bậc thấp hơn. Thực ra, bất cứ khi nào mỗi mẫu dữ liệu có một giá trị $x$ riêng biệt, một hàm đa thức có bậc bằng số mẫu dữ liệu có thể khớp hoàn hảo tập huấn luyện. Ta so sánh mối quan hệ giữa bậc đa thức (độ phức tạp mô hình) với cả dưới khớp lẫn quá khớp trong :numref:`fig_capacity_vs_error`.

![Ảnh hưởng của độ phức tạp mô hình lên dưới khớp và quá khớp.](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`

### Kích thước Tập dữ liệu

Như cận ở trên đã cho thấy, một yếu tố lớn khác cần lưu tâm là kích thước tập dữ liệu. Cố định mô hình, tập dữ liệu huấn luyện càng ít mẫu thì càng có khả năng (và càng nghiêm trọng) gặp phải quá khớp. Khi tăng lượng dữ liệu huấn luyện, lỗi tổng quát hóa thường giảm. Hơn nữa, nói chung, có thêm dữ liệu không bao giờ gây hại. Với một tác vụ và phân phối dữ liệu cố định, độ phức tạp mô hình không nên tăng nhanh hơn lượng dữ liệu. Khi có nhiều dữ liệu hơn, ta có thể thử khớp một mô hình phức tạp hơn. Khi thiếu dữ liệu, các mô hình đơn giản hơn có thể khó bị đánh bại hơn. Với nhiều tác vụ, học sâu chỉ vượt trội các mô hình tuyến tính khi có sẵn nhiều nghìn mẫu huấn luyện. Một phần, thành công hiện nay của học sâu là nhờ đáng kể vào sự dồi dào của các tập dữ liệu khổng lồ đến từ các công ty Internet, lưu trữ giá rẻ, các thiết bị kết nối và sự số hóa rộng khắp của nền kinh tế.

## Lựa chọn Mô hình
:label:`subsec_generalization-model-selection`

Thông thường, ta chỉ chọn mô hình cuối cùng sau khi đánh giá nhiều mô hình khác nhau theo nhiều cách (kiến trúc khác nhau, mục tiêu huấn luyện, đặc trưng được chọn, tiền xử lý dữ liệu, tốc độ học, v.v.). Việc chọn giữa nhiều mô hình được gọi một cách thích hợp là *lựa chọn mô hình* (model selection).

Về nguyên tắc, ta không nên đụng đến tập kiểm tra cho đến khi đã chọn xong mọi siêu tham số. Nếu dùng dữ liệu kiểm tra trong quá trình lựa chọn mô hình, có nguy cơ là ta sẽ quá khớp dữ liệu kiểm tra. Khi đó ta sẽ gặp rắc rối nghiêm trọng. Nếu quá khớp dữ liệu huấn luyện, ta luôn có việc đánh giá trên dữ liệu kiểm tra để giữ mình trung thực. Nhưng nếu quá khớp dữ liệu kiểm tra, làm sao ta biết được? Xem :citet:`ong2005learning` để có ví dụ về việc điều này có thể dẫn đến những kết quả vô lý như thế nào, ngay cả với những mô hình mà độ phức tạp có thể được kiểm soát chặt chẽ.

Do đó, ta không bao giờ nên dựa vào dữ liệu kiểm tra để lựa chọn mô hình. Thế nhưng ta cũng không thể chỉ dựa vào dữ liệu huấn luyện để lựa chọn mô hình, vì ta không thể ước lượng lỗi tổng quát hóa trên chính dữ liệu mà ta dùng để huấn luyện mô hình.

Trong các ứng dụng thực tế, bức tranh còn mờ mịt hơn. Dù lý tưởng nhất là ta chỉ đụng đến dữ liệu kiểm tra một lần, để đánh giá mô hình tốt nhất hoặc để so sánh một số ít mô hình với nhau, dữ liệu kiểm tra trong thực tế hiếm khi bị loại bỏ chỉ sau một lần sử dụng. Ta hiếm khi đủ khả năng có một tập kiểm tra mới cho mỗi vòng thí nghiệm. Thực ra, việc tái sử dụng dữ liệu đánh giá chuẩn trong hàng thập kỷ có thể có tác động đáng kể đến sự phát triển của các thuật toán, chẳng hạn cho [phân loại ảnh](https://paperswithcode.com/sota/image-classification-on-imagenet) và [nhận dạng ký tự quang học](https://paperswithcode.com/sota/image-classification-on-mnist).

Cách làm phổ biến để giải quyết vấn đề *huấn luyện trên tập kiểm tra* là chia dữ liệu thành ba phần, thêm một *tập xác thực* (validation set) bên cạnh tập huấn luyện và tập kiểm tra. Kết quả là một tình trạng lẫn lộn, nơi ranh giới giữa dữ liệu xác thực và dữ liệu kiểm tra mơ hồ một cách đáng lo ngại. Trừ khi được nói rõ khác đi, trong các thí nghiệm của cuốn sách này, ta thực sự đang làm việc với thứ mà đúng ra phải gọi là dữ liệu huấn luyện và dữ liệu xác thực, không có tập kiểm tra thực sự. Do đó, độ chính xác được báo cáo trong mỗi thí nghiệm của cuốn sách thực ra là độ chính xác trên tập xác thực, không phải độ chính xác trên một tập kiểm tra thực sự.

### Kiểm định Chéo

Khi dữ liệu huấn luyện khan hiếm, ta thậm chí có thể không đủ khả năng giữ riêng đủ dữ liệu để tạo thành một tập xác thực đúng nghĩa. Một giải pháp phổ biến cho vấn đề này là dùng *kiểm định chéo $K$ phần* ($K$-fold cross-validation). Ở đây, dữ liệu huấn luyện ban đầu được chia thành $K$ tập con không chồng lấn. Sau đó việc huấn luyện và xác thực mô hình được thực hiện $K$ lần, mỗi lần huấn luyện trên $K-1$ tập con và xác thực trên một tập con khác (tập không được dùng để huấn luyện trong vòng đó). Cuối cùng, lỗi huấn luyện và lỗi xác thực được ước lượng bằng cách lấy trung bình kết quả từ $K$ thí nghiệm.

## Tóm tắt

Mục này đã tìm hiểu một số nền tảng của tổng quát hóa trong học máy. Một số ý tưởng trong đó trở nên phức tạp và phản trực giác khi ta đến với các mô hình sâu hơn; ở đó, mô hình có khả năng quá khớp dữ liệu rất tệ, và các khái niệm liên quan về độ phức tạp có thể vừa ngầm định vừa phản trực giác (chẳng hạn những kiến trúc lớn hơn với nhiều tham số hơn lại tổng quát hóa tốt hơn). Chúng tôi để lại cho bạn một vài quy tắc kinh nghiệm:

1. Dùng tập xác thực (hoặc *kiểm định chéo $K$ phần*) để lựa chọn mô hình;
1. Mô hình phức tạp hơn thường cần nhiều dữ liệu hơn;
1. Các khái niệm liên quan về độ phức tạp gồm cả số lượng tham số lẫn khoảng giá trị mà chúng được phép nhận;
1. Giữ mọi thứ khác như nhau, nhiều dữ liệu hơn hầu như luôn dẫn đến tổng quát hóa tốt hơn;
1. Toàn bộ thảo luận về tổng quát hóa này đều dựa trên giả định IID. Nếu nới lỏng giả định này, cho phép phân phối dịch chuyển giữa giai đoạn huấn luyện và kiểm tra, thì ta không thể nói gì về tổng quát hóa nếu không có thêm một giả định (có lẽ nhẹ hơn) nào khác.

## Bài tập

1. Khi nào bạn có thể giải chính xác bài toán hồi quy đa thức?
1. Hãy đưa ra ít nhất năm ví dụ trong đó các biến ngẫu nhiên phụ thuộc khiến việc coi bài toán là dữ liệu IID trở nên không nên.
1. Bạn có bao giờ có thể mong đợi thấy lỗi huấn luyện bằng không không? Trong những hoàn cảnh nào bạn sẽ thấy lỗi tổng quát hóa bằng không?
1. Vì sao kiểm định chéo $K$ phần rất tốn kém về mặt tính toán?
1. Vì sao ước lượng lỗi bằng kiểm định chéo $K$ phần bị chệch?
1. Chiều VC được định nghĩa là số điểm lớn nhất có thể được phân loại với các nhãn tùy ý $\{\pm 1\}$ bởi một hàm thuộc một lớp hàm. Vì sao đây có thể không phải là một ý tưởng hay để đo độ phức tạp của lớp hàm? Gợi ý: xét độ lớn của các hàm.
1. Quản lý của bạn đưa cho bạn một tập dữ liệu khó mà thuật toán hiện tại của bạn hoạt động không tốt lắm. Bạn sẽ biện minh với ông ấy thế nào rằng bạn cần thêm dữ liệu? Gợi ý: bạn không thể tăng dữ liệu nhưng có thể giảm nó.

[Discussions](https://discuss.d2l.ai/t/97)
