# Hướng dẫn Xây dựng Mô hình
:label:`chap_computation`

Bên cạnh các tập dữ liệu khổng lồ và phần cứng mạnh mẽ, các công cụ phần mềm tuyệt vời đã đóng vai trò không thể thiếu trong sự tiến bộ nhanh chóng của học sâu. Bắt đầu từ thư viện tiên phong Theano ra mắt năm 2007, các công cụ mã nguồn mở linh hoạt đã cho phép các nhà nghiên cứu nhanh chóng tạo nguyên mẫu mô hình, tránh công việc lặp lại khi tái sử dụng các thành phần tiêu chuẩn, trong khi vẫn giữ được khả năng chỉnh sửa ở mức thấp. Theo thời gian, các thư viện học sâu đã tiến hóa để cung cấp những tầng trừu tượng ngày càng thô hơn. Giống như các nhà thiết kế bán dẫn đã đi từ việc đặc tả từng transistor đến mạch logic rồi đến viết code, các nhà nghiên cứu mạng neuron đã chuyển từ việc suy nghĩ về hành vi của từng neuron nhân tạo riêng lẻ sang hình dung mạng theo từng tầng hoàn chỉnh, và giờ đây thường thiết kế kiến trúc với những *khối* (block) còn thô hơn nhiều.

Cho đến nay, chúng tôi đã giới thiệu một số khái niệm học máy cơ bản, dần tiến tới các mô hình học sâu hoạt động hoàn chỉnh. Trong chương trước, ta đã tự cài đặt từ đầu từng thành phần của một MLP và thậm chí còn chỉ ra cách tận dụng các API cấp cao để triển khai cùng những mô hình đó một cách dễ dàng. Để đưa bạn đi xa như vậy và nhanh như vậy, chúng tôi đã *gọi đến* các thư viện, nhưng bỏ qua những chi tiết nâng cao hơn về *cách chúng hoạt động*. Trong chương này, chúng tôi sẽ vén bức màn, đi sâu hơn vào các thành phần then chốt của tính toán trong học sâu, cụ thể là xây dựng mô hình, truy cập và khởi tạo tham số, thiết kế các tầng và khối tùy chỉnh, đọc và ghi mô hình ra ổ đĩa, và tận dụng GPU để tăng tốc đáng kể. Những hiểu biết này sẽ đưa bạn từ *người dùng cuối* thành *người dùng thành thạo*, cho bạn các công cụ cần thiết để tận hưởng lợi ích của một thư viện học sâu trưởng thành trong khi vẫn giữ được sự linh hoạt để cài đặt những mô hình phức tạp hơn, kể cả những mô hình do chính bạn phát minh! Dù chương này không giới thiệu mô hình hay tập dữ liệu mới nào, các chương mô hình hóa nâng cao tiếp theo đều dựa nhiều vào những kỹ thuật này.

```toc
:maxdepth: 2

model-construction
parameters
init-param
lazy-init
custom-layer
read-write
use-gpu
```
