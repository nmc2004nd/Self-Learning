# Tài liệu

Dù ta không thể giới thiệu mọi hàm và lớp của PyTorch (và thông tin có thể nhanh chóng trở nên lỗi thời), [tài liệu API](https://pytorch.org/docs/stable/index.html) cùng các [hướng dẫn](https://pytorch.org/tutorials/beginner/basics/intro.html) và ví dụ bổ sung cung cấp những tài liệu như vậy. Mục này đưa ra một số hướng dẫn về cách khám phá API của PyTorch.

```{.python .input}
import torch
```

## Các Hàm và Lớp trong một Mô-đun

Để biết những hàm và lớp nào có thể được gọi trong một mô-đun, ta gọi hàm `dir`. Chẳng hạn, ta có thể (**truy vấn mọi thuộc tính trong mô-đun sinh số ngẫu nhiên**):

```{.python .input  n=1}
print(dir(torch.distributions))
```

Nói chung, ta có thể bỏ qua những hàm bắt đầu và kết thúc bằng `__` (các đối tượng đặc biệt trong Python) hoặc những hàm bắt đầu bằng một dấu `_` (thường là các hàm nội bộ). Dựa trên tên các hàm hoặc thuộc tính còn lại, ta có thể đoán rằng mô-đun này cung cấp nhiều phương thức sinh số ngẫu nhiên, gồm lấy mẫu từ phân phối đều (`uniform`), phân phối chuẩn (`normal`) và phân phối đa thức (`multinomial`).

## Các Hàm và Lớp Cụ thể

Để có hướng dẫn cụ thể về cách dùng một hàm hoặc lớp cho trước, ta có thể gọi hàm `help`. Ví dụ, hãy [**tìm hiểu hướng dẫn sử dụng hàm `ones` của tensor**].

```{.python .input}
help(torch.ones)
```

Từ tài liệu, ta có thể thấy hàm `ones` tạo một tensor mới với kích thước chỉ định và đặt mọi phần tử bằng giá trị 1. Bất cứ khi nào có thể, bạn nên (**chạy một phép thử nhanh**) để xác nhận cách hiểu của mình:

```{.python .input}
torch.ones(4)
```

Trong Jupyter notebook, ta có thể dùng `?` để hiển thị tài liệu trong một cửa sổ khác. Chẳng hạn, `list?` sẽ tạo ra nội dung gần như giống hệt `help(list)`, hiển thị nó trong một cửa sổ trình duyệt mới. Ngoài ra, nếu dùng hai dấu chấm hỏi, chẳng hạn `list??`, mã Python cài đặt hàm đó cũng sẽ được hiển thị.

Tài liệu chính thức cung cấp rất nhiều mô tả và ví dụ vượt ra ngoài cuốn sách này. Chúng tôi nhấn mạnh những trường hợp sử dụng quan trọng giúp bạn nhanh chóng bắt đầu với các bài toán thực tế, thay vì bao quát đầy đủ. Chúng tôi cũng khuyến khích bạn nghiên cứu mã nguồn của các thư viện để thấy những ví dụ cài đặt chất lượng cao của code dùng trong sản phẩm thực tế. Làm như vậy, bạn sẽ trở thành một kỹ sư giỏi hơn, bên cạnh việc trở thành một nhà khoa học giỏi hơn.

[Discussions](https://discuss.d2l.ai/t/39)
