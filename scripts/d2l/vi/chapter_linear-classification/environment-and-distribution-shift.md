# Môi trường và Dịch chuyển Phân phối
:label:`sec_environment-and-distribution-shift`

Trong các mục trước, ta đã đi qua một số ứng dụng thực hành của học máy, khớp mô hình với nhiều tập dữ liệu khác nhau. Thế nhưng, ta chưa bao giờ dừng lại để suy ngẫm dữ liệu ban đầu đến từ đâu, hay cuối cùng ta định làm gì với đầu ra của mô hình. Quá thường xuyên, những người phát triển học máy có trong tay dữ liệu vội vã phát triển mô hình mà không dừng lại cân nhắc những vấn đề cơ bản này.

Nhiều lần triển khai học máy thất bại có thể truy về sự thiếu sót này. Đôi khi mô hình có vẻ hoạt động tuyệt vời khi đo bằng độ chính xác trên tập kiểm tra nhưng lại thất bại thảm hại khi triển khai, lúc phân phối dữ liệu đột ngột dịch chuyển. Nguy hiểm hơn, đôi khi chính việc triển khai một mô hình lại là chất xúc tác làm nhiễu loạn phân phối dữ liệu. Chẳng hạn, giả sử ta đã huấn luyện một mô hình dự đoán ai sẽ trả nợ và ai sẽ vỡ nợ khi vay, và phát hiện ra rằng lựa chọn giày dép của người nộp đơn có liên quan đến rủi ro vỡ nợ (giày Oxford cho thấy trả nợ, giày thể thao cho thấy vỡ nợ). Sau đó, ta có thể có xu hướng cho vay với bất kỳ ai đi giày Oxford và từ chối mọi người đi giày thể thao.

Trong trường hợp này, bước nhảy thiếu cân nhắc từ nhận dạng khuôn mẫu sang ra quyết định, cùng với việc không xem xét một cách có phê phán đến môi trường, có thể gây ra hậu quả tai hại. Trước hết, ngay khi ta bắt đầu ra quyết định dựa trên giày dép, khách hàng sẽ nhận ra và thay đổi hành vi. Chẳng bao lâu, mọi người nộp đơn đều sẽ đi giày Oxford, mà khả năng trả nợ của họ chẳng hề được cải thiện theo. Hãy dành một phút để suy ngẫm điều này vì những vấn đề tương tự xuất hiện khắp nơi trong nhiều ứng dụng của học máy: khi đưa các quyết định dựa trên mô hình vào môi trường, ta có thể làm hỏng chính mô hình.

Dù không thể trình bày đầy đủ những chủ đề này trong một mục, ở đây chúng tôi muốn nêu ra một số mối lo thường gặp, và khơi gợi tư duy phản biện cần thiết để phát hiện sớm những tình huống như vậy, giảm thiểu thiệt hại và sử dụng học máy một cách có trách nhiệm. Một số giải pháp thì đơn giản (yêu cầu dữ liệu "đúng"), một số thì khó về mặt kỹ thuật (cài đặt một hệ thống học tăng cường), và số khác đòi hỏi ta bước ra hoàn toàn khỏi phạm vi dự đoán thống kê và vật lộn với những câu hỏi triết học khó khăn liên quan đến việc ứng dụng thuật toán một cách có đạo đức.

## Các Loại Dịch chuyển Phân phối

Để bắt đầu, ta giữ nguyên bối cảnh dự đoán thụ động, xem xét những cách khác nhau mà phân phối dữ liệu có thể dịch chuyển và có thể làm gì để cứu vãn hiệu năng mô hình. Trong một thiết lập kinh điển, ta giả định rằng dữ liệu huấn luyện được lấy mẫu từ một phân phối $p_S(\mathbf{x},y)$ nào đó nhưng dữ liệu kiểm tra sẽ gồm các mẫu chưa gán nhãn lấy từ một phân phối khác $p_T(\mathbf{x},y)$. Ngay lập tức, ta phải đối mặt với một thực tế phũ phàng. Nếu không có giả định nào về cách $p_S$ và $p_T$ liên hệ với nhau, việc học một bộ phân loại bền vững là không thể.

Hãy xét một bài toán phân loại nhị phân, trong đó ta muốn phân biệt chó và mèo. Nếu phân phối có thể dịch chuyển theo những cách tùy ý, thì thiết lập của ta cho phép một trường hợp bệnh lý trong đó phân phối của đầu vào giữ nguyên: $p_S(\mathbf{x}) = p_T(\mathbf{x})$, nhưng các nhãn đều bị đảo ngược: $p_S(y \mid \mathbf{x}) = 1 - p_T(y \mid \mathbf{x})$. Nói cách khác, nếu Thượng đế bỗng quyết định rằng trong tương lai mọi "mèo" giờ là chó và những gì ta trước đây gọi là "chó" giờ là mèo, mà không có bất kỳ thay đổi nào trong phân phối đầu vào $p(\mathbf{x})$, thì ta không thể nào phân biệt tình huống này với một tình huống mà phân phối hoàn toàn không thay đổi.

May mắn thay, dưới một số giả định hạn chế về cách dữ liệu có thể thay đổi trong tương lai, các thuật toán có cơ sở có thể phát hiện sự dịch chuyển và đôi khi thậm chí thích nghi ngay lập tức, cải thiện độ chính xác của bộ phân loại ban đầu.

### Dịch chuyển Hiệp biến

Trong các loại dịch chuyển phân phối, dịch chuyển hiệp biến có lẽ được nghiên cứu rộng rãi nhất. Ở đây, ta giả định rằng dù phân phối đầu vào có thể thay đổi theo thời gian, hàm gán nhãn, tức phân phối có điều kiện $P(y \mid \mathbf{x})$, không thay đổi. Các nhà thống kê gọi đây là *dịch chuyển hiệp biến* (covariate shift) vì vấn đề nảy sinh do sự dịch chuyển trong phân phối của các hiệp biến (đặc trưng). Dù đôi khi ta có thể lập luận về dịch chuyển phân phối mà không cần viện đến quan hệ nhân quả, ta lưu ý rằng dịch chuyển hiệp biến là giả định tự nhiên để dùng trong những bối cảnh mà ta tin rằng $\mathbf{x}$ gây ra $y$.

Hãy xét thử thách phân biệt mèo và chó. Dữ liệu huấn luyện của ta có thể gồm những ảnh thuộc loại trong :numref:`fig_cat-dog-train`.

![Dữ liệu huấn luyện để phân biệt mèo và chó (minh họa: Lafeez Hossain / 500px / Getty Images; ilkermetinkursova / iStock / Getty Images Plus; GlobalP / iStock / Getty Images Plus; Musthafa Aboobakuru / 500px / Getty Images).](../img/cat-dog-train.png)
:label:`fig_cat-dog-train`

Khi kiểm tra, ta được yêu cầu phân loại các ảnh trong :numref:`fig_cat-dog-test`.

![Dữ liệu kiểm tra để phân biệt mèo và chó (minh họa: SIBAS_minich / iStock / Getty Images Plus; Ghrzuzudu / iStock / Getty Images Plus; id-work / DigitalVision Vectors / Getty Images; Yime / iStock / Getty Images Plus).](../img/cat-dog-test.png)
:label:`fig_cat-dog-test`

Tập huấn luyện gồm ảnh chụp, trong khi tập kiểm tra chỉ chứa tranh hoạt hình. Huấn luyện trên một tập dữ liệu có đặc điểm khác đáng kể so với tập kiểm tra có thể gây rắc rối nếu không có một kế hoạch mạch lạc về cách thích nghi với miền mới.

### Dịch chuyển Nhãn

*Dịch chuyển nhãn* (label shift) mô tả bài toán ngược lại. Ở đây, ta giả định rằng phân phối biên của nhãn $P(y)$ có thể thay đổi nhưng phân phối có điều kiện theo lớp $P(\mathbf{x} \mid y)$ giữ cố định giữa các miền. Dịch chuyển nhãn là một giả định hợp lý khi ta tin rằng $y$ gây ra $\mathbf{x}$. Chẳng hạn, ta có thể muốn dự đoán chẩn đoán dựa trên triệu chứng (hoặc các biểu hiện khác), ngay cả khi tỉ lệ mắc tương đối của các chẩn đoán đang thay đổi theo thời gian. Dịch chuyển nhãn là giả định phù hợp ở đây vì bệnh gây ra triệu chứng. Trong một số trường hợp suy biến, giả định dịch chuyển nhãn và dịch chuyển hiệp biến có thể đồng thời đúng. Chẳng hạn, khi nhãn là tất định, giả định dịch chuyển hiệp biến sẽ được thỏa mãn, ngay cả khi $y$ gây ra $\mathbf{x}$. Thú vị là, trong những trường hợp này, thường có lợi khi làm việc với các phương pháp xuất phát từ giả định dịch chuyển nhãn. Đó là vì các phương pháp này có xu hướng thao tác trên những đối tượng trông giống nhãn (thường ít chiều), thay vì những đối tượng trông giống đầu vào, vốn thường nhiều chiều trong học sâu.

### Dịch chuyển Khái niệm

Ta cũng có thể gặp một bài toán liên quan là *dịch chuyển khái niệm* (concept shift), nảy sinh khi chính định nghĩa của các nhãn có thể thay đổi. Điều này nghe có vẻ kỳ lạ: *mèo* thì vẫn là *mèo* chứ? Tuy nhiên, những phạm trù khác có thể thay đổi cách dùng theo thời gian. Tiêu chí chẩn đoán bệnh tâm thần, những gì được coi là hợp thời trang, và chức danh công việc đều chịu mức độ dịch chuyển khái niệm đáng kể. Hóa ra nếu đi khắp nước Mỹ, dịch chuyển nguồn dữ liệu theo địa lý, ta sẽ thấy sự dịch chuyển khái niệm đáng kể về phân phối của các tên gọi cho *nước ngọt có ga* như trong :numref:`fig_popvssoda`.

![Dịch chuyển khái niệm đối với tên gọi nước ngọt có ga ở Hoa Kỳ (CC-BY: Alan McConchie, PopVsSoda.com).](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

Nếu xây dựng một hệ thống dịch máy, phân phối $P(y \mid \mathbf{x})$ có thể khác nhau tùy theo vị trí của ta. Vấn đề này có thể khó phát hiện. Ta có thể hy vọng khai thác được kiến thức rằng sự dịch chuyển chỉ diễn ra dần dần theo nghĩa thời gian hoặc địa lý.

## Các Ví dụ về Dịch chuyển Phân phối

Trước khi đi sâu vào hình thức hóa và thuật toán, ta có thể thảo luận một số tình huống cụ thể mà dịch chuyển hiệp biến hay dịch chuyển khái niệm có thể không hiển nhiên.

### Chẩn đoán Y khoa

Hãy tưởng tượng bạn muốn thiết kế một thuật toán phát hiện ung thư. Bạn thu thập dữ liệu từ người khỏe mạnh và người bệnh rồi huấn luyện thuật toán. Nó hoạt động tốt, cho bạn độ chính xác cao và bạn kết luận rằng mình đã sẵn sàng cho một sự nghiệp thành công trong chẩn đoán y khoa. *Đừng vội.*

Các phân phối sinh ra dữ liệu huấn luyện và những phân phối bạn sẽ gặp ngoài thực tế có thể khác nhau đáng kể. Điều này đã xảy ra với một công ty khởi nghiệp kém may mắn mà một số tác giả chúng tôi từng làm việc cùng nhiều năm trước. Họ đang phát triển một xét nghiệm máu cho một căn bệnh chủ yếu ảnh hưởng đến đàn ông lớn tuổi và hy vọng nghiên cứu nó bằng các mẫu máu đã thu thập từ bệnh nhân. Tuy nhiên, lấy mẫu máu từ đàn ông khỏe mạnh khó hơn đáng kể so với từ những bệnh nhân đã có trong hệ thống. Để bù lại, công ty khởi nghiệp kêu gọi sinh viên trong một khuôn viên đại học hiến máu để làm nhóm đối chứng khỏe mạnh khi phát triển xét nghiệm. Sau đó họ hỏi liệu chúng tôi có thể giúp họ xây dựng một bộ phân loại phát hiện bệnh hay không.

Như chúng tôi đã giải thích với họ, quả thực sẽ dễ dàng phân biệt nhóm khỏe mạnh và nhóm bệnh với độ chính xác gần như hoàn hảo. Tuy nhiên, đó là vì các đối tượng xét nghiệm khác nhau về tuổi tác, nồng độ hormone, hoạt động thể chất, chế độ ăn, mức tiêu thụ rượu và nhiều yếu tố khác không liên quan đến bệnh. Điều này khó có thể xảy ra với bệnh nhân thật. Do quy trình lấy mẫu của họ, ta có thể dự đoán sẽ gặp dịch chuyển hiệp biến cực độ. Hơn nữa, trường hợp này khó có thể được khắc phục bằng các phương pháp thông thường. Tóm lại, họ đã lãng phí một khoản tiền đáng kể.

### Xe Tự lái

Giả sử một công ty muốn tận dụng học máy để phát triển xe tự lái. Một thành phần then chốt ở đây là bộ phát hiện lề đường. Vì dữ liệu thật đã chú thích rất đắt, họ có ý tưởng (thông minh nhưng đáng ngờ) là dùng dữ liệu tổng hợp từ một bộ máy kết xuất đồ họa trò chơi làm dữ liệu huấn luyện bổ sung. Cách này hoạt động rất tốt trên "dữ liệu kiểm tra" lấy từ bộ máy kết xuất. Than ôi, trong một chiếc xe thật thì đó là một thảm họa. Hóa ra lề đường đã được kết xuất với một kết cấu rất đơn giản. Quan trọng hơn, *toàn bộ* lề đường đều được kết xuất với *cùng một* kết cấu và bộ phát hiện lề đường đã học được "đặc trưng" này rất nhanh.

Một điều tương tự đã xảy ra với Quân đội Mỹ khi họ lần đầu thử phát hiện xe tăng trong rừng. Họ chụp ảnh từ trên không khu rừng khi không có xe tăng, rồi lái xe tăng vào rừng và chụp một bộ ảnh khác. Bộ phân loại có vẻ hoạt động *hoàn hảo*. Đáng tiếc là nó chỉ học được cách phân biệt cây có bóng với cây không có bóng: bộ ảnh đầu được chụp vào sáng sớm, bộ thứ hai vào buổi trưa.

### Phân phối Không dừng

Một tình huống tinh vi hơn nhiều nảy sinh khi phân phối thay đổi chậm (còn được gọi là *phân phối không dừng* – nonstationary distribution) và mô hình không được cập nhật đầy đủ. Dưới đây là một số trường hợp điển hình.

* Ta huấn luyện một mô hình quảng cáo điện toán rồi không cập nhật nó thường xuyên (chẳng hạn, ta quên đưa vào việc một thiết bị mới ít người biết tên là iPad vừa ra mắt).
* Ta xây dựng một bộ lọc thư rác. Nó hoạt động tốt trong việc phát hiện mọi thư rác mà ta đã thấy cho đến nay. Nhưng rồi những kẻ gửi thư rác trở nên khôn hơn và soạn ra những thông điệp mới trông không giống bất cứ thứ gì ta từng thấy.
* Ta xây dựng một hệ thống gợi ý sản phẩm. Nó hoạt động suốt mùa đông nhưng rồi vẫn tiếp tục gợi ý mũ ông già Noel rất lâu sau Giáng sinh.

### Thêm Vài Giai thoại

* Ta xây dựng một bộ phát hiện khuôn mặt. Nó hoạt động tốt trên mọi bộ đánh giá chuẩn. Đáng tiếc là nó thất bại trên dữ liệu kiểm tra: những mẫu gây lỗi là ảnh chụp cận cảnh trong đó khuôn mặt chiếm toàn bộ bức ảnh (không có dữ liệu như vậy trong tập huấn luyện).
* Ta xây dựng một công cụ tìm kiếm web cho thị trường Mỹ và muốn triển khai nó ở Anh.
* Ta huấn luyện một bộ phân loại ảnh bằng cách tập hợp một tập dữ liệu lớn trong đó mỗi lớp trong một tập lớp lớn được đại diện như nhau, chẳng hạn 1000 loại, mỗi loại được đại diện bởi 1000 ảnh. Sau đó ta triển khai hệ thống ra thế giới thực, nơi phân phối nhãn thực tế của các bức ảnh chắc chắn là không đều.

## Hiệu chỉnh Dịch chuyển Phân phối

Như đã thảo luận, có nhiều trường hợp mà phân phối huấn luyện và kiểm tra $P(\mathbf{x}, y)$ khác nhau. Trong một số trường hợp, ta gặp may và mô hình vẫn hoạt động dù có dịch chuyển hiệp biến, nhãn hay khái niệm. Trong những trường hợp khác, ta có thể làm tốt hơn bằng cách dùng các chiến lược có cơ sở để đối phó với sự dịch chuyển. Phần còn lại của mục này mang tính kỹ thuật hơn đáng kể. Độc giả thiếu kiên nhẫn có thể chuyển sang mục tiếp theo vì nội dung này không phải là kiến thức tiên quyết cho các khái niệm về sau.

### Rủi ro Thực nghiệm và Rủi ro
:label:`subsec_empirical-risk-and-risk`

Trước tiên hãy suy ngẫm chính xác điều gì đang diễn ra trong quá trình huấn luyện mô hình: ta duyệt qua các đặc trưng và nhãn tương ứng của dữ liệu huấn luyện $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ và cập nhật các tham số của mô hình $f$ sau mỗi minibatch. Để đơn giản, ta không xét điều chuẩn, nên về cơ bản ta cực tiểu hóa mất mát trên tập huấn luyện:

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n l(f(\mathbf{x}_i), y_i),$$
:eqlabel:`eq_empirical-risk-min`

trong đó $l$ là hàm mất mát đo dự đoán $f(\mathbf{x}_i)$ "tệ đến mức nào" so với nhãn tương ứng $y_i$. Các nhà thống kê gọi số hạng trong :eqref:`eq_empirical-risk-min` là *rủi ro thực nghiệm* (empirical risk). *Rủi ro thực nghiệm* là mất mát trung bình trên dữ liệu huấn luyện, dùng để xấp xỉ *rủi ro* (risk), tức kỳ vọng của mất mát trên toàn bộ tổng thể dữ liệu được lấy từ phân phối thực của chúng $p(\mathbf{x},y)$:

$$E_{p(\mathbf{x}, y)} [l(f(\mathbf{x}), y)] = \int\int l(f(\mathbf{x}), y) p(\mathbf{x}, y) \;d\mathbf{x}dy.$$
:eqlabel:`eq_true-risk`

Tuy nhiên, trong thực tế ta thường không thể có được toàn bộ tổng thể dữ liệu. Do đó, *cực tiểu hóa rủi ro thực nghiệm* (empirical risk minimization), tức cực tiểu hóa rủi ro thực nghiệm trong :eqref:`eq_empirical-risk-min`, là một chiến lược thực tế cho học máy, với hy vọng cực tiểu hóa xấp xỉ được rủi ro.

### Hiệu chỉnh Dịch chuyển Hiệp biến
:label:`subsec_covariate-shift-correction`

Giả sử ta muốn ước lượng một quan hệ phụ thuộc $P(y \mid \mathbf{x})$ nào đó mà ta có dữ liệu đã gán nhãn $(\mathbf{x}_i, y_i)$. Đáng tiếc là các quan sát $\mathbf{x}_i$ được lấy từ một *phân phối nguồn* (source distribution) $q(\mathbf{x})$ nào đó thay vì *phân phối đích* (target distribution) $p(\mathbf{x})$. May mắn thay, giả định về quan hệ phụ thuộc nghĩa là phân phối có điều kiện không thay đổi: $p(y \mid \mathbf{x}) = q(y \mid \mathbf{x})$. Nếu phân phối nguồn $q(\mathbf{x})$ là "sai", ta có thể hiệu chỉnh điều đó bằng cách dùng đồng nhất thức đơn giản sau trong rủi ro:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(y \mid \mathbf{x})p(\mathbf{x}) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(y \mid \mathbf{x})q(\mathbf{x})\frac{p(\mathbf{x})}{q(\mathbf{x})} \;d\mathbf{x}dy.
\end{aligned}
$$

Nói cách khác, ta cần đánh lại trọng số cho mỗi mẫu dữ liệu bằng tỉ số giữa xác suất nó được lấy từ phân phối đúng và xác suất nó được lấy từ phân phối sai:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(\mathbf{x}_i)}{q(\mathbf{x}_i)}.$$

Thay trọng số $\beta_i$ cho mỗi mẫu dữ liệu $(\mathbf{x}_i, y_i)$, ta có thể huấn luyện mô hình bằng *cực tiểu hóa rủi ro thực nghiệm có trọng số* (weighted empirical risk minimization):

$$\mathop{\mathrm{minimize}}_f \frac{1}{n} \sum_{i=1}^n \beta_i l(f(\mathbf{x}_i), y_i).$$
:eqlabel:`eq_weighted-empirical-risk-min`

Than ôi, ta không biết tỉ số đó, nên trước khi có thể làm bất cứ điều gì hữu ích, ta cần ước lượng nó. Có nhiều phương pháp, gồm một số cách tiếp cận lý thuyết toán tử tinh vi cố gắng hiệu chỉnh trực tiếp toán tử kỳ vọng bằng nguyên lý chuẩn nhỏ nhất hoặc entropy cực đại. Lưu ý rằng với bất kỳ cách tiếp cận nào như vậy, ta cần các mẫu lấy từ cả hai phân phối: phân phối "thực" $p$, chẳng hạn bằng cách truy cập dữ liệu kiểm tra, và phân phối dùng để sinh tập huấn luyện $q$ (cái sau hiển nhiên là có sẵn). Tuy nhiên, lưu ý rằng ta chỉ cần các đặc trưng $\mathbf{x} \sim p(\mathbf{x})$; ta không cần truy cập nhãn $y \sim p(y)$.

Trong trường hợp này, tồn tại một cách tiếp cận rất hiệu quả cho kết quả gần như tốt bằng cách gốc: đó là hồi quy logistic, một trường hợp đặc biệt của hồi quy softmax (xem :numref:`sec_softmax`) cho phân loại nhị phân. Đây là tất cả những gì cần thiết để tính các tỉ số xác suất ước lượng. Ta học một bộ phân loại để phân biệt dữ liệu lấy từ $p(\mathbf{x})$ với dữ liệu lấy từ $q(\mathbf{x})$. Nếu không thể phân biệt hai phân phối, điều đó nghĩa là các mẫu tương ứng có khả năng đến từ phân phối này hay phân phối kia là như nhau. Mặt khác, bất kỳ mẫu nào có thể được phân biệt tốt sẽ cần được tăng hoặc giảm trọng số đáng kể tương ứng.

Để đơn giản, giả sử ta có số mẫu bằng nhau từ cả hai phân phối $p(\mathbf{x})$ và $q(\mathbf{x})$. Giờ ký hiệu $z$ là nhãn bằng $1$ cho dữ liệu lấy từ $p$ và $-1$ cho dữ liệu lấy từ $q$. Khi đó xác suất trong một tập dữ liệu trộn được cho bởi

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \textrm{ và do đó } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Do đó, nếu ta dùng cách tiếp cận hồi quy logistic, trong đó $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-h(\mathbf{x}))}$ ($h$ là một hàm có tham số), ta suy ra

$$
\beta_i = \frac{1/(1 + \exp(-h(\mathbf{x}_i)))}{\exp(-h(\mathbf{x}_i))/(1 + \exp(-h(\mathbf{x}_i)))} = \exp(h(\mathbf{x}_i)).
$$

Kết quả là ta cần giải hai bài toán: thứ nhất, phân biệt dữ liệu lấy từ hai phân phối, và sau đó là một bài toán cực tiểu hóa rủi ro thực nghiệm có trọng số trong :eqref:`eq_weighted-empirical-risk-min`, trong đó ta đánh trọng số các số hạng bằng $\beta_i$.

Giờ ta đã sẵn sàng mô tả một thuật toán hiệu chỉnh. Giả sử ta có một tập huấn luyện $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ và một tập kiểm tra chưa gán nhãn $\{\mathbf{u}_1, \ldots, \mathbf{u}_m\}$. Với dịch chuyển hiệp biến, ta giả định rằng $\mathbf{x}_i$ với mọi $1 \leq i \leq n$ được lấy từ một phân phối nguồn nào đó và $\mathbf{u}_i$ với mọi $1 \leq i \leq m$ được lấy từ phân phối đích. Đây là một thuật toán nguyên mẫu để hiệu chỉnh dịch chuyển hiệp biến:

1. Tạo một tập huấn luyện phân loại nhị phân: $\{(\mathbf{x}_1, -1), \ldots, (\mathbf{x}_n, -1), (\mathbf{u}_1, 1), \ldots, (\mathbf{u}_m, 1)\}$.
1. Huấn luyện một bộ phân loại nhị phân bằng hồi quy logistic để có được hàm $h$.
1. Đánh trọng số dữ liệu huấn luyện bằng $\beta_i = \exp(h(\mathbf{x}_i))$ hoặc tốt hơn là $\beta_i = \min(\exp(h(\mathbf{x}_i)), c)$ với một hằng số $c$ nào đó.
1. Dùng các trọng số $\beta_i$ để huấn luyện trên $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ trong :eqref:`eq_weighted-empirical-risk-min`.

Lưu ý rằng thuật toán trên dựa vào một giả định then chốt. Để phương án này hoạt động, ta cần mỗi mẫu dữ liệu trong phân phối đích (chẳng hạn lúc kiểm tra) có xác suất xuất hiện khác không ở thời điểm huấn luyện. Nếu ta tìm thấy một điểm mà $p(\mathbf{x}) > 0$ nhưng $q(\mathbf{x}) = 0$, thì trọng số quan trọng (importance weight) tương ứng sẽ là vô cùng.

### Hiệu chỉnh Dịch chuyển Nhãn

Giả sử ta đang làm một tác vụ phân loại với $k$ loại. Dùng cùng ký hiệu như trong :numref:`subsec_covariate-shift-correction`, $q$ và $p$ lần lượt là phân phối nguồn (chẳng hạn lúc huấn luyện) và phân phối đích (chẳng hạn lúc kiểm tra). Giả sử phân phối nhãn dịch chuyển theo thời gian: $q(y) \neq p(y)$, nhưng phân phối có điều kiện theo lớp giữ nguyên: $q(\mathbf{x} \mid y)=p(\mathbf{x} \mid y)$. Nếu phân phối nguồn $q(y)$ là "sai", ta có thể hiệu chỉnh nó theo đồng nhất thức sau trong rủi ro được định nghĩa ở :eqref:`eq_true-risk`:

$$
\begin{aligned}
\int\int l(f(\mathbf{x}), y) p(\mathbf{x} \mid y)p(y) \;d\mathbf{x}dy =
\int\int l(f(\mathbf{x}), y) q(\mathbf{x} \mid y)q(y)\frac{p(y)}{q(y)} \;d\mathbf{x}dy.
\end{aligned}
$$

Ở đây, các trọng số quan trọng sẽ tương ứng với tỉ số hợp lý của nhãn:

$$\beta_i \stackrel{\textrm{def}}{=} \frac{p(y_i)}{q(y_i)}.$$

Một điều hay của dịch chuyển nhãn là nếu ta có một mô hình khá tốt trên phân phối nguồn, ta có thể có được các ước lượng nhất quán của những trọng số này mà không bao giờ phải làm việc với số chiều của không gian đầu vào. Trong học sâu, đầu vào thường là những đối tượng nhiều chiều như ảnh, trong khi nhãn thường là những đối tượng đơn giản hơn như các loại.

Để ước lượng phân phối nhãn đích, trước tiên ta lấy bộ phân loại có sẵn khá tốt (thường được huấn luyện trên dữ liệu huấn luyện) và tính ma trận "nhầm lẫn" của nó bằng tập xác thực (cũng thuộc phân phối huấn luyện). *Ma trận nhầm lẫn* (confusion matrix) $\mathbf{C}$ đơn giản là một ma trận $k \times k$, trong đó mỗi cột tương ứng với loại nhãn (giá trị thực) và mỗi hàng tương ứng với loại mà mô hình dự đoán. Giá trị của mỗi ô $c_{ij}$ là tỉ lệ trên tổng số dự đoán trên tập xác thực mà nhãn thực là $j$ còn mô hình dự đoán là $i$.

Giờ, ta không thể tính trực tiếp ma trận nhầm lẫn trên dữ liệu đích vì ta không được thấy nhãn của các mẫu ngoài thực tế, trừ khi đầu tư vào một quy trình chú thích thời gian thực phức tạp. Tuy nhiên, điều ta có thể làm là lấy trung bình mọi dự đoán của mô hình lúc kiểm tra, thu được đầu ra trung bình của mô hình $\mu(\hat{\mathbf{y}}) \in \mathbb{R}^k$, trong đó phần tử thứ $i$, $\mu(\hat{y}_i)$, là tỉ lệ trên tổng số dự đoán trên tập kiểm tra mà mô hình dự đoán là $i$.

Hóa ra dưới một số điều kiện nhẹ (nếu ngay từ đầu bộ phân loại khá chính xác, nếu dữ liệu đích chỉ chứa những loại mà ta đã thấy trước đó, và nếu giả định dịch chuyển nhãn đúng ngay từ đầu, giả định mạnh nhất ở đây), ta có thể ước lượng phân phối nhãn của tập kiểm tra bằng cách giải một hệ tuyến tính đơn giản

$$\mathbf{C} p(\mathbf{y}) = \mu(\hat{\mathbf{y}}),$$

vì như một ước lượng, $\sum_{j=1}^k c_{ij} p(y_j) = \mu(\hat{y}_i)$ đúng với mọi $1 \leq i \leq k$, trong đó $p(y_j)$ là phần tử thứ $j$ của vector phân phối nhãn $k$ chiều $p(\mathbf{y})$. Nếu ngay từ đầu bộ phân loại đủ chính xác, thì ma trận nhầm lẫn $\mathbf{C}$ sẽ khả nghịch, và ta có nghiệm $p(\mathbf{y}) = \mathbf{C}^{-1} \mu(\hat{\mathbf{y}})$.

Vì ta quan sát được nhãn trên dữ liệu nguồn, việc ước lượng phân phối $q(y)$ là dễ dàng. Sau đó, với bất kỳ mẫu huấn luyện $i$ nào có nhãn $y_i$, ta có thể lấy tỉ số ước lượng $p(y_i)/q(y_i)$ để tính trọng số $\beta_i$, và thay nó vào bài toán cực tiểu hóa rủi ro thực nghiệm có trọng số trong :eqref:`eq_weighted-empirical-risk-min`.

### Hiệu chỉnh Dịch chuyển Khái niệm

Dịch chuyển khái niệm khó khắc phục một cách có cơ sở hơn nhiều. Chẳng hạn, trong một tình huống mà bài toán đột ngột thay đổi từ phân biệt mèo với chó sang phân biệt động vật màu trắng với màu đen, sẽ không hợp lý khi giả định rằng ta có thể làm tốt hơn nhiều so với việc chỉ thu thập nhãn mới và huấn luyện lại từ đầu. May mắn thay, trong thực tế, những sự dịch chuyển cực đoan như vậy rất hiếm. Thay vào đó, điều thường xảy ra là tác vụ cứ thay đổi chậm rãi. Để cụ thể hơn, đây là một số ví dụ:

* Trong quảng cáo điện toán, các sản phẩm mới được ra mắt, các sản phẩm cũ trở nên kém phổ biến hơn. Điều này nghĩa là phân phối của các quảng cáo và mức độ phổ biến của chúng thay đổi dần dần và mọi bộ dự đoán tỉ lệ nhấp chuột cũng cần thay đổi dần theo.
* Ống kính camera giao thông xuống cấp dần do hao mòn từ môi trường, ảnh hưởng dần dần đến chất lượng ảnh.
* Nội dung tin tức thay đổi dần dần (tức là phần lớn tin tức giữ nguyên nhưng các câu chuyện mới xuất hiện).

Trong những trường hợp như vậy, ta có thể dùng cùng cách tiếp cận đã dùng để huấn luyện mạng để làm cho chúng thích nghi với sự thay đổi của dữ liệu. Nói cách khác, ta dùng trọng số mạng hiện có và chỉ đơn giản thực hiện vài bước cập nhật với dữ liệu mới thay vì huấn luyện lại từ đầu.

## Phân loại các Bài toán Học

Với kiến thức về cách xử lý sự thay đổi của phân phối, giờ ta có thể xem xét một số khía cạnh khác của việc phát biểu bài toán học máy.

### Học theo Lô

Trong *học theo lô* (batch learning), ta có quyền truy cập các đặc trưng và nhãn huấn luyện $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$, mà ta dùng để huấn luyện một mô hình $f(\mathbf{x})$. Sau đó, ta triển khai mô hình này để chấm điểm dữ liệu mới $(\mathbf{x}, y)$ lấy từ cùng phân phối. Đây là giả định mặc định cho mọi bài toán mà ta thảo luận ở đây. Chẳng hạn, ta có thể huấn luyện một bộ phát hiện mèo dựa trên thật nhiều ảnh mèo và chó. Khi đã huấn luyện xong, ta đưa nó vào một hệ thống thị giác máy tính cho cửa ra vào thông minh chỉ cho mèo đi vào. Hệ thống này sau đó được lắp đặt tại nhà khách hàng và không bao giờ được cập nhật nữa (trừ những trường hợp đặc biệt).

### Học Trực tuyến

Giờ hãy tưởng tượng dữ liệu $(\mathbf{x}_i, y_i)$ đến từng mẫu một. Cụ thể hơn, giả sử trước tiên ta quan sát $\mathbf{x}_i$, rồi ta cần đưa ra một ước lượng $f(\mathbf{x}_i)$. Chỉ sau khi đã làm điều này, ta mới quan sát được $y_i$ và do đó nhận phần thưởng hoặc chịu mất mát tùy theo quyết định của mình. Nhiều bài toán thực tế thuộc loại này. Chẳng hạn, ta cần dự đoán giá cổ phiếu ngày mai, điều cho phép ta giao dịch dựa trên ước lượng đó và cuối ngày ta mới biết ước lượng của mình có mang lại lợi nhuận hay không. Nói cách khác, trong *học trực tuyến* (online learning), ta có chu trình sau, trong đó ta liên tục cải thiện mô hình khi có quan sát mới:

$$\begin{aligned}&\textrm{mô hình } f_t \longrightarrow \textrm{dữ liệu }  \mathbf{x}_t \longrightarrow \textrm{ước lượng } f_t(\mathbf{x}_t) \longrightarrow\\ \textrm{qua}&\textrm{n sát } y_t \longrightarrow \textrm{mất mát } l(y_t, f_t(\mathbf{x}_t)) \longrightarrow \textrm{mô hình } f_{t+1}\end{aligned}$$

### Bandit

*Bandit* là một trường hợp đặc biệt của bài toán trên. Trong khi ở hầu hết các bài toán học, ta có một hàm $f$ được tham số hóa liên tục mà ta muốn học các tham số của nó (chẳng hạn một mạng sâu), trong bài toán *bandit*, ta chỉ có một số hữu hạn cần gạt có thể kéo, tức là một số hữu hạn hành động có thể thực hiện. Không có gì ngạc nhiên khi với bài toán đơn giản hơn này, ta có thể có được những đảm bảo lý thuyết mạnh hơn về tính tối ưu. Chúng tôi liệt kê nó chủ yếu vì bài toán này thường được coi (một cách gây nhầm lẫn) như thể nó là một bối cảnh học riêng biệt.

### Điều khiển

Trong nhiều trường hợp, môi trường ghi nhớ những gì ta đã làm. Không nhất thiết theo cách thù địch, nhưng nó sẽ ghi nhớ và phản hồi sẽ phụ thuộc vào những gì đã xảy ra trước đó. Chẳng hạn, bộ điều khiển nồi đun cà phê sẽ quan sát được nhiệt độ khác nhau tùy thuộc vào việc trước đó nó có đang đun nóng nồi hay không. Các thuật toán điều khiển PID (tỉ lệ–tích phân–vi phân) là lựa chọn phổ biến ở đó. Tương tự, hành vi của người dùng trên một trang tin tức sẽ phụ thuộc vào những gì ta đã cho họ xem trước đó (chẳng hạn, họ sẽ chỉ đọc hầu hết tin tức một lần). Nhiều thuật toán như vậy xây dựng một mô hình của môi trường mà chúng hoạt động trong đó để các quyết định của chúng trông bớt ngẫu nhiên hơn. Gần đây, lý thuyết điều khiển (chẳng hạn các biến thể của PID) cũng đã được dùng để tự động tinh chỉnh siêu tham số nhằm đạt chất lượng tách biệt và tái tạo tốt hơn, cũng như cải thiện sự đa dạng của văn bản được sinh ra và chất lượng tái tạo của ảnh được sinh ra :cite:`Shao.Yao.Sun.ea.2020`.

### Học Tăng cường

Trong trường hợp tổng quát hơn của một môi trường có bộ nhớ, ta có thể gặp những tình huống mà môi trường cố gắng hợp tác với ta (các trò chơi hợp tác, đặc biệt là các trò chơi có tổng khác không), hoặc những tình huống khác mà môi trường sẽ cố gắng giành chiến thắng. Cờ vua, cờ vây, cờ thỏ cáo hay StarCraft là một số trường hợp trong *học tăng cường*. Tương tự, ta có thể muốn xây dựng một bộ điều khiển tốt cho xe tự lái. Những chiếc xe khác có khả năng phản ứng với phong cách lái của xe tự lái theo những cách không tầm thường, chẳng hạn cố gắng tránh nó, cố gắng gây tai nạn, hoặc cố gắng hợp tác với nó.

### Xem xét Môi trường

Một khác biệt then chốt giữa các tình huống khác nhau ở trên là một chiến lược có thể đã hoạt động suốt trong trường hợp môi trường dừng, có thể không hoạt động suốt trong một môi trường có thể thích nghi. Chẳng hạn, một cơ hội kinh doanh chênh lệch giá mà một nhà giao dịch phát hiện ra có khả năng biến mất ngay khi nó được khai thác. Tốc độ và cách thức môi trường thay đổi quyết định phần lớn loại thuật toán mà ta có thể áp dụng. Chẳng hạn, nếu biết rằng mọi thứ chỉ có thể thay đổi chậm, ta có thể buộc mọi ước lượng cũng chỉ thay đổi chậm. Nếu biết rằng môi trường có thể thay đổi tức thời, nhưng rất hiếm khi, ta có thể tính đến điều đó. Những loại kiến thức này là then chốt đối với nhà khoa học dữ liệu đầy tham vọng khi đối phó với dịch chuyển khái niệm, tức là khi bài toán đang được giải có thể thay đổi theo thời gian.

## Công bằng, Trách nhiệm Giải trình và Minh bạch trong Học máy

Cuối cùng, điều quan trọng cần nhớ là khi triển khai các hệ thống học máy, bạn không chỉ đơn thuần tối ưu một mô hình dự đoán: bạn thường đang cung cấp một công cụ sẽ được dùng để tự động hóa (một phần hoặc toàn bộ) các quyết định. Những hệ thống kỹ thuật này có thể ảnh hưởng đến cuộc sống của những cá nhân chịu tác động của các quyết định đó. Bước nhảy từ xem xét dự đoán sang ra quyết định không chỉ đặt ra những câu hỏi kỹ thuật mới mà còn cả một loạt câu hỏi đạo đức cần được cân nhắc cẩn thận. Nếu triển khai một hệ thống chẩn đoán y khoa, ta cần biết nó có thể hoạt động cho những nhóm dân cư nào và có thể không hoạt động cho những nhóm nào. Bỏ qua những rủi ro có thể lường trước đối với phúc lợi của một nhóm dân cư con có thể khiến ta cung cấp dịch vụ chăm sóc kém chất lượng hơn. Hơn nữa, một khi xem xét các hệ thống ra quyết định, ta phải lùi lại và xem xét lại cách ta đánh giá công nghệ của mình. Trong số những hệ quả khác của sự thay đổi phạm vi này, ta sẽ thấy rằng *độ chính xác* hiếm khi là thước đo đúng. Chẳng hạn, khi chuyển dự đoán thành hành động, ta thường sẽ muốn tính đến độ nhạy chi phí tiềm năng của việc mắc lỗi theo những cách khác nhau. Nếu một cách phân loại sai một bức ảnh có thể bị coi là một sự xúc phạm về chủng tộc, trong khi phân loại sai sang một loại khác lại vô hại, thì ta có thể muốn điều chỉnh ngưỡng cho phù hợp, tính đến các giá trị xã hội khi thiết kế quy trình ra quyết định. Ta cũng muốn cẩn thận về việc các hệ thống dự đoán có thể dẫn đến các vòng phản hồi như thế nào. Chẳng hạn, hãy xét các hệ thống dự đoán tội phạm, phân bổ cảnh sát tuần tra đến những khu vực được dự báo có nhiều tội phạm. Dễ thấy một khuôn mẫu đáng lo ngại có thể hình thành như thế nào:

 1. Các khu dân cư có nhiều tội phạm hơn được tuần tra nhiều hơn.
 1. Do đó, nhiều tội phạm hơn được phát hiện ở những khu này, đi vào dữ liệu huấn luyện dùng cho những lần lặp sau.
 1. Tiếp xúc với nhiều mẫu dương tính hơn, mô hình dự đoán thêm nhiều tội phạm hơn nữa ở những khu này.
 1. Ở lần lặp tiếp theo, mô hình đã cập nhật nhắm vào cùng khu dân cư đó còn nặng nề hơn, dẫn đến phát hiện thêm nhiều tội phạm hơn nữa, v.v.

Thường thì, những cơ chế khác nhau khiến dự đoán của mô hình gắn kết với dữ liệu huấn luyện của nó không được tính đến trong quá trình mô hình hóa. Điều này có thể dẫn đến thứ mà các nhà nghiên cứu gọi là *vòng phản hồi mất kiểm soát* (runaway feedback loop). Ngoài ra, ta muốn cẩn thận về việc ngay từ đầu ta có đang giải quyết đúng vấn đề hay không. Các thuật toán dự đoán giờ đây đóng vai trò lớn bất thường trong việc điều phối sự lan truyền thông tin. Liệu tin tức mà một cá nhân gặp có nên được quyết định bởi tập các trang Facebook mà họ đã *Thích*? Đây chỉ là một vài trong số rất nhiều tình huống tiến thoái lưỡng nan về đạo đức mà bạn có thể gặp trong sự nghiệp học máy.

## Tóm tắt

Trong nhiều trường hợp, tập huấn luyện và tập kiểm tra không đến từ cùng một phân phối. Điều này được gọi là dịch chuyển phân phối. Rủi ro là kỳ vọng của mất mát trên toàn bộ tổng thể dữ liệu được lấy từ phân phối thực của chúng. Tuy nhiên, toàn bộ tổng thể này thường không có sẵn. Rủi ro thực nghiệm là mất mát trung bình trên dữ liệu huấn luyện để xấp xỉ rủi ro. Trong thực tế, ta thực hiện cực tiểu hóa rủi ro thực nghiệm.

Dưới các giả định tương ứng, dịch chuyển hiệp biến và dịch chuyển nhãn có thể được phát hiện và hiệu chỉnh lúc kiểm tra. Không tính đến độ chệch này có thể trở thành vấn đề lúc kiểm tra. Trong một số trường hợp, môi trường có thể ghi nhớ các hành động tự động và phản hồi theo những cách đáng ngạc nhiên. Ta phải tính đến khả năng này khi xây dựng mô hình và tiếp tục giám sát các hệ thống đang vận hành, sẵn sàng cho khả năng mô hình và môi trường sẽ đan xen vào nhau theo những cách không lường trước.

## Bài tập

1. Điều gì có thể xảy ra khi ta thay đổi hành vi của một công cụ tìm kiếm? Người dùng có thể làm gì? Còn các nhà quảng cáo thì sao?
1. Cài đặt một bộ phát hiện dịch chuyển hiệp biến. Gợi ý: xây dựng một bộ phân loại.
1. Cài đặt một bộ hiệu chỉnh dịch chuyển hiệp biến.
1. Ngoài dịch chuyển phân phối, còn điều gì khác có thể ảnh hưởng đến việc rủi ro thực nghiệm xấp xỉ rủi ro tốt đến đâu?

[Discussions](https://discuss.d2l.ai/t/105)
