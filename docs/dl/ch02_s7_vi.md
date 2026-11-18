# 2.7 Tài liệu

> Ghi chú của người biên soạn: trang này dịch nguyên văn từ sách *Dive into Deep Learning* của **Aston Zhang, Zachary C. Lipton, Mu Li và Alexander J. Smola** ([bản gốc](https://d2l.ai/chapter_preliminaries/lookup-api.html), [mã nguồn](https://github.com/d2l-ai/d2l-en), giấy phép [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); code theo giấy phép MIT-0). Chỉ giữ code PyTorch. Số mục, hình và công thức giữ như bản HTML d2l.ai; thuật ngữ đối chiếu với bản dịch [Đắm mình vào Học Sâu](https://github.com/aivivn/d2l-vn).

Dù ta không thể giới thiệu mọi hàm và lớp của PyTorch (và thông tin có thể nhanh chóng trở nên lỗi thời), [tài liệu API](https://pytorch.org/docs/stable/index.html) cùng các [hướng dẫn](https://pytorch.org/tutorials/beginner/basics/intro.html) và ví dụ bổ sung cung cấp những tài liệu như vậy. Mục này đưa ra một số hướng dẫn về cách khám phá API của PyTorch.

<!-- run -->
```python
import torch
```

## 2.7.1 Các Hàm và Lớp trong một Mô-đun

Để biết những hàm và lớp nào có thể được gọi trong một mô-đun, ta gọi hàm `dir`. Chẳng hạn, ta có thể truy vấn mọi thuộc tính trong mô-đun sinh số ngẫu nhiên:

<!-- run -->
```python
print(dir(torch.distributions))
```

<!-- output -->
```output
['AbsTransform', 'AffineTransform', 'Bernoulli', 'Beta', 'Binomial', 'CatTransform', 'Categorical', 'Cauchy', 'Chi2', 'ComposeTransform', 'ContinuousBernoulli', 'CorrCholeskyTransform', 'CumulativeDistributionTransform', 'Dirichlet', 'Distribution', 'ExpTransform', 'Exponential', 'ExponentialFamily', 'FisherSnedecor', 'Gamma', 'GeneralizedPareto', 'Geometric', 'Gumbel', 'HalfCauchy', 'HalfNormal', 'Independent', 'IndependentTransform', 'InverseGamma', 'Kumaraswamy', 'LKJCholesky', 'Laplace', 'LogNormal', 'LogisticNormal', 'LowRankMultivariateNormal', 'LowerCholeskyTransform', 'MixtureSameFamily', 'Multinomial', 'MultivariateNormal', 'NegativeBinomial', 'Normal', 'OneHotCategorical', 'OneHotCategoricalStraightThrough', 'Pareto', 'Poisson', 'PositiveDefiniteTransform', 'PowerTransform', 'RelaxedBernoulli', 'RelaxedOneHotCategorical', 'ReshapeTransform', 'SigmoidTransform', 'SoftmaxTransform', 'SoftplusTransform', 'StackTransform', 'StickBreakingTransform', 'StudentT', 'TanhTransform', 'Transform', 'TransformedDistribution', 'Uniform', 'VonMises', 'Weibull', 'Wishart', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'bernoulli', 'beta', 'biject_to', 'binomial', 'categorical', 'cauchy', 'chi2', 'constraint_registry', 'constraints', 'continuous_bernoulli', 'dirichlet', 'distribution', 'exp_family', 'exponential', 'fishersnedecor', 'gamma', 'generalized_pareto', 'geometric', 'gumbel', 'half_cauchy', 'half_normal', 'identity_transform', 'independent', 'inverse_gamma', 'kl', 'kl_divergence', 'kumaraswamy', 'laplace', 'lkj_cholesky', 'log_normal', 'logistic_normal', 'lowrank_multivariate_normal', 'mixture_same_family', 'multinomial', 'multivariate_normal', 'negative_binomial', 'normal', 'one_hot_categorical', 'pareto', 'poisson', 'register_kl', 'relaxed_bernoulli', 'relaxed_categorical', 'studentT', 'transform_to', 'transformed_distribution', 'transforms', 'uniform', 'utils', 'von_mises', 'weibull', 'wishart']
```
<!-- /output -->

Nói chung, ta có thể bỏ qua những hàm bắt đầu và kết thúc bằng `__` (các đối tượng đặc biệt trong Python) hoặc những hàm bắt đầu bằng một dấu `_` (thường là các hàm nội bộ). Dựa trên tên các hàm hoặc thuộc tính còn lại, ta có thể đoán rằng mô-đun này cung cấp nhiều phương thức sinh số ngẫu nhiên, gồm lấy mẫu từ phân phối đều (`uniform`), phân phối chuẩn (`normal`) và phân phối đa thức (`multinomial`).

## 2.7.2 Các Hàm và Lớp Cụ thể

Để có hướng dẫn cụ thể về cách dùng một hàm hoặc lớp cho trước, ta có thể gọi hàm `help`. Ví dụ, hãy tìm hiểu hướng dẫn sử dụng hàm `ones` của tensor.

<!-- run -->
```python
help(torch.ones)
```

<!-- output -->
```output
Help on built-in function ones in module torch:

ones(...)
    ones(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor
    
    Returns a tensor filled with the scalar value `1`, with the shape defined
    by the variable argument :attr:`size`.
    
    Args:
        size (int...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.
    
    Keyword arguments:
        out (Tensor, optional): the output tensor.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        device (:class:`torch.device`, optional): the desired device of returned tensor.
            Default: if ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
            for CPU tensor types and the current CUDA device for CUDA tensor types.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
    
    Example::
    
        >>> torch.ones(2, 3)
        tensor([[ 1.,  1.,  1.],
                [ 1.,  1.,  1.]])
    
        >>> torch.ones(5)
        tensor([ 1.,  1.,  1.,  1.,  1.])
```
<!-- /output -->

Từ tài liệu, ta có thể thấy hàm `ones` tạo một tensor mới với kích thước chỉ định và đặt mọi phần tử bằng giá trị 1. Bất cứ khi nào có thể, bạn nên chạy một phép thử nhanh để xác nhận cách hiểu của mình:

<!-- run -->
```python
torch.ones(4)
```

<!-- output -->
```output
tensor([1., 1., 1., 1.])
```
<!-- /output -->

Trong Jupyter notebook, ta có thể dùng `?` để hiển thị tài liệu trong một cửa sổ khác. Chẳng hạn, `list?` sẽ tạo ra nội dung gần như giống hệt `help(list)`, hiển thị nó trong một cửa sổ trình duyệt mới. Ngoài ra, nếu dùng hai dấu chấm hỏi, chẳng hạn `list??`, mã Python cài đặt hàm đó cũng sẽ được hiển thị.

Tài liệu chính thức cung cấp rất nhiều mô tả và ví dụ vượt ra ngoài cuốn sách này. Chúng tôi nhấn mạnh những trường hợp sử dụng quan trọng giúp bạn nhanh chóng bắt đầu với các bài toán thực tế, thay vì bao quát đầy đủ. Chúng tôi cũng khuyến khích bạn nghiên cứu mã nguồn của các thư viện để thấy những ví dụ cài đặt chất lượng cao của code dùng trong sản phẩm thực tế. Làm như vậy, bạn sẽ trở thành một kỹ sư giỏi hơn, bên cạnh việc trở thành một nhà khoa học giỏi hơn.

[Thảo luận (tiếng Anh)](https://discuss.d2l.ai/t/39)
