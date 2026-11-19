# Mẹo huấn luyện: tăng cường dữ liệu

Trang [Code](dlnote/alchemy_code_vi) đi kèm (mục 4) chạy lại các biến đổi hình học bằng OpenCV, dùng `torchvision.transforms.v2` và Albumentations để biến đổi cùng lúc ảnh và hộp giới hạn, và minh hoạ MixUp, CutMix, RandAugment.

<!-- theory: qa/ch03_vi#3.13 ; qa/ch08_vi#8.5.1 -->

## 1 Tổng quan về tăng cường dữ liệu

Tăng cường dữ liệu (data augmentation, còn gọi là mở rộng dữ liệu) nhằm mở rộng dữ liệu và nâng khả năng tổng quát hoá của mô hình. Tăng cường hiệu quả không chỉ tăng số lượng mẫu huấn luyện mà còn tăng độ đa dạng của chúng: một mặt tránh quá khớp, mặt khác cải thiện hiệu năng mô hình.

Các cách tăng cường thường dùng: **lật ngang/dọc, cắt ngẫu nhiên (random crop), co giãn (scaling) và xoay (rotating)**. Co giãn và xoay giúp mạng neuron tích chập bền vững (`robust`) hơn với kích thước và hướng của vật thể.

Thêm vào đó, đổi màu ngẫu nhiên (`color jittering`) trên ảnh gốc hoặc ảnh (vùng ảnh) đã biến đổi cũng là một cách tăng cường phổ biến, tức thay đổi bốn mặt của màu: **độ sáng, độ tương phản, độ bão hoà và sắc độ**. Có thể nhiễu nhẹ phân phối màu trong không gian `RGB`, hoặc trong không gian `HSV` thay đổi ngẫu nhiên độ bão hoà và độ sáng (kênh `S` và `V`) hay xê dịch nhẹ sắc độ (thay đổi kênh `H` trong phạm vi nhỏ).

> Không gian màu HSV biểu diễn ảnh màu bằng ba thành phần:
> Hue (sắc độ, màu sắc);
> Saturation (độ bão hoà, độ thuần của màu);
> Value (độ sáng).

Trong quy trình học máy (`pipeline`), tăng cường dữ liệu làm trước khi đưa vào mô hình, theo một trong hai cách:

- **Tăng cường ngoại tuyến** (offline augmentation): biến đổi trước và lưu ra ổ đĩa; hợp với tập dữ liệu nhỏ.
- **Tăng cường trực tuyến** (online augmentation): biến đổi ngẫu nhiên ngay khi nạp từng batch; hợp với tập dữ liệu lớn.

> **Biên tập:** Ngày nay tăng cường trực tuyến là mặc định ở mọi quy mô: nó không tốn ổ đĩa, và mỗi epoch mô hình thấy một phiên bản khác của từng ảnh, điều tăng cường ngoại tuyến (một số ít bản sao cố định) không làm được. Tăng cường ngoại tuyến còn hữu ích khi phép biến đổi rất đắt (sinh ảnh bằng mô hình sinh, render 3D) hoặc khi cần kiểm tra bằng mắt từng ảnh sinh ra.

## 2 Tăng cường ảnh bằng OpenCV: biến đổi hình học

Các hàm biến đổi hình học của OpenCV:

1. **Co giãn**: đổi kích thước ảnh.

   `cv2.resize()`: tham số thường dùng là kích thước đích, hệ số co giãn và cách nội suy.

2. **Tịnh tiến**: dời vật thể sang vị trí khác.

   `cv2.warpAffine()`: tham số thứ nhất là ảnh gốc, thứ hai là ma trận dịch chuyển, thứ ba là kích thước ảnh đầu ra (rộng, cao). Ví dụ muốn dịch theo hướng $(x, y)$ một khoảng $(t_x, t_y)$ thì ma trận dịch chuyển là:

$$
\begin{bmatrix}
1 & 0 & t_x\\
0 & 1 & t_y
\end{bmatrix}
$$

3. **Xoay**: xoay ảnh một góc $\theta$.

   Trước hết dùng `cv2.getRotationMatrix2D` tạo ma trận xoay $M$, rồi dùng `cv2.warpAffine()` để biến đổi ảnh. Tham số của `getRotationMatrix2D` lần lượt là tâm xoay, góc xoay (độ, ngược chiều kim đồng hồ) và hệ số co giãn sau khi xoay.

4. **Biến đổi affine**: các đường thẳng song song trong ảnh gốc vẫn song song trong ảnh kết quả.

   Để tìm ma trận biến đổi, cần lấy ba điểm trong ảnh vào cùng vị trí tương ứng của chúng trong ảnh ra. `cv2.getAffineTransform` dựng một ma trận $2\times 3$ từ ba cặp điểm đó, rồi ma trận được truyền cho `cv2.warpAffine`.

> **Biên tập:** Bản gốc gọi biến đổi affine (仿射变换) bằng chữ đồng âm 放射变换 ("phóng xạ") và ghi chú "còn gọi là biến đổi phẳng / biến đổi xuyên tâm" (径向变换, radial); hàm trong code cũng tên là `radial_trans_aug`. Biến đổi xuyên tâm là một loại khác (như méo ống kính), không liên quan; đã sửa thành "affine".

5. **Biến đổi phối cảnh** (perspective transformation): sau biến đổi, đường thẳng vẫn là đường thẳng (nhưng các đường song song có thể không còn song song).

   > Nguyên lý (theo Baidu Baike): biến đổi phối cảnh dùng điều kiện tâm phối cảnh, điểm ảnh và điểm đích thẳng hàng; theo định luật xoay phối cảnh, mặt chiếu được xoay quanh trục phối cảnh một góc, làm thay đổi chùm tia chiếu ban đầu nhưng vẫn giữ nguyên hình chiếu hình học trên mặt chiếu.

   Biến đổi phối cảnh cần một **ma trận biến đổi** $3\times 3$. Để tìm ma trận này, cần 4 điểm trên ảnh vào và vị trí tương ứng của chúng trên ảnh ra, trong đó không có 3 điểm nào thẳng hàng. Dùng `cv2.getPerspectiveTransform` để tìm ma trận, rồi `cv2.warpPerspective` để áp dụng nó.

**Ví dụ code** biến đổi hình học:

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class GeometricTransAug:
    def __init__(self, image_path):
        self.img_np = np.array(Image.open(image_path).convert("RGB"))
        self.rows, self.cols, self.ch = self.img_np.shape

    def resize_aug(self, img):
        # đặt thẳng hệ số co giãn: gấp 2 lần kích thước gốc
        return cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    def translate_aug(self, img):
        # ma trận dịch chuyển: dời mọi điểm ảnh đi (100, 50)
        M = np.float32([[1, 0, 100], [0, 1, 50]])
        return cv2.warpAffine(img, M, (self.cols, self.rows))

    def rotation_aug(self, img):
        # xoay 90 độ quanh tâm ảnh, không co giãn
        M = cv2.getRotationMatrix2D((self.cols / 2, self.rows / 2), 90, 1)
        return cv2.warpAffine(img, M, (self.cols, self.rows))

    def affine_aug(self, img):
        # affine cần ba điểm trên ảnh gốc và vị trí của chúng trên ảnh ra
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
        M = cv2.getAffineTransform(pts1, pts2)   # ma trận 2x3
        return cv2.warpAffine(img, M, dsize=(self.cols, self.rows))

    def perspective_aug(self, img):
        # phối cảnh cần ma trận 3x3 từ bốn cặp điểm
        pts1 = np.float32([[56, 65], [368, 52], [28, 287], [389, 290]])
        pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(img, M, dsize=(300, 300))  # dsize: kích thước ảnh ra

    def show(self):
        imgs = [self.resize_aug(self.img_np), self.translate_aug(self.img_np), self.rotation_aug(self.img_np),
                self.affine_aug(self.img_np), self.perspective_aug(self.img_np)]
        titles = ["resize", "tịnh tiến", "xoay 90°", "affine", "phối cảnh"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        for ax, im, t in zip(axes.flat, imgs, titles):
            ax.imshow(im); ax.set_title(t); ax.axis('off')
        axes.flat[-1].axis('off')

GeometricTransAug('cat.jpg').show()
```

Kết quả các phép **biến đổi hình học**:

<div align=center><img src="../img/code/alchemy_code/19_1.png" alt="Biến đổi hình học bằng OpenCV"/></div>

Hình 2.1: Ảnh gốc và năm phép biến đổi hình học bằng OpenCV, kèm kích thước ảnh ra (trang Code, ảnh `chelsea` của scikit-image).

> **Biên tập:** Code đã được viết lại gọn hơn: bản gốc gọi hàm vẽ ngay trong `__init__`, đặt tên phép affine là `radial_trans_aug`, và vẽ 5 ảnh lên lưới 2×3 nên ô thứ sáu còn trục trống. Điểm thứ ba, thứ tư của phép phối cảnh được đổi từ $y=387, 390$ về $y=287, 290$ cho vừa ảnh mẫu cao 300 điểm ảnh. Lưu ý ảnh "resize ×2" trông giống ảnh gốc vì `imshow` co ảnh vừa ô; kích thước thật là 902×600.

## 3 Tăng cường ảnh trong PyTorch

Trong `PyTorch`, mô-đun `torchvision.transforms` cung cấp các phép tăng cường, áp dụng được cho cả `PIL Image` lẫn `torch.Tensor`. Giới thiệu chi tiết xem tài liệu chính thức: [Transforming and augmenting images](https://pytorch.org/vision/stable/transforms.html). Bản gốc tóm tắt 22 phép biến đổi của `transforms.py`, chia thành bốn nhóm:

1. **Cắt** (Crop)

   - cắt ở giữa: `transforms.CenterCrop`
   - cắt ngẫu nhiên: `transforms.RandomCrop`
   - cắt ngẫu nhiên với tỉ lệ diện tích và tỉ lệ khung hình ngẫu nhiên, rồi resize: `transforms.RandomResizedCrop`
   - cắt bốn góc và giữa: `transforms.FiveCrop`
   - cắt bốn góc và giữa, cộng bản lật của chúng: `transforms.TenCrop`

2. **Lật và xoay** (Flip and Rotation)

   - lật ngang với xác suất p: `transforms.RandomHorizontalFlip(p=0.5)`
   - lật dọc với xác suất p: `transforms.RandomVerticalFlip(p=0.5)`
   - xoay ngẫu nhiên: `transforms.RandomRotation`

3. **Biến đổi ảnh**

   - resize: `transforms.Resize`
   - chuyển sang tensor, chia cho 255: `transforms.ToTensor()`
   - chuẩn hoá theo kênh: `transforms.Normalize()`
   - đệm viền: `transforms.Pad`
   - đổi độ sáng, độ tương phản, độ bão hoà, sắc độ: `transforms.ColorJitter`
   - chuyển sang ảnh xám: `transforms.Grayscale`
   - biến đổi tuyến tính: `transforms.LinearTransformation()`
   - biến đổi affine ngẫu nhiên: `transforms.RandomAffine`
   - chuyển sang ảnh xám với xác suất `p`: `transforms.RandomGrayscale`
   - chuyển dữ liệu sang `PILImage`: `transforms.ToPILImage`
   - áp một hàm lambda do người dùng định nghĩa: `transforms.Lambda`

4. **Các phép kết hợp transforms**, giúp tăng cường linh hoạt hơn

   - `transforms.RandomChoice(transforms)`: chọn ngẫu nhiên một phép trong danh sách để áp dụng;
   - `transforms.RandomApply(transforms, p=0.5)`: áp dụng danh sách phép biến đổi với xác suất p;
   - `transforms.RandomOrder`: áp dụng các phép biến đổi theo thứ tự ngẫu nhiên.

> **Biên tập:** Danh sách trên là của API cũ `torchvision.transforms` (khoảng 2021). Từ torchvision 0.16 (2023), API được khuyên dùng là `torchvision.transforms.v2`: cùng tên lớp, nhưng biến đổi đồng thời ảnh, hộp giới hạn, mặt nạ phân đoạn, điểm mốc và video (bọc trong `tv_tensors`), nhanh hơn trên tensor, và có thêm các phép mới như `RandAugment`, `TrivialAugmentWide`, `AutoAugment`, `AugMix`, `MixUp`, `CutMix`, `RandomErasing`, `ElasticTransform`, `JPEG`. Bản 0.29 có 63 lớp biến đổi. Bản gốc cũng xếp `ToTensor` là "min-max normalization"; xem lại [bài chuẩn hoá dữ liệu](dlnote/alch_norm_vi), mục 2.3.

Với `transforms.v2`, ảnh và hộp giới hạn được biến đổi cùng lúc, nên tăng cường cho bài toán phát hiện đối tượng không cần tự tính lại toạ độ hộp:

```python
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

img = tv_tensors.Image(img_uint8_chw)                                   # ảnh (3, H, W)
boxes = tv_tensors.BoundingBoxes([[120, 20, 340, 280]], format="XYXY", canvas_size=(H, W))
aug = v2.Compose([
    v2.RandomResizedCrop(224, scale=(0.4, 1.0), antialias=True),
    v2.RandomHorizontalFlip(),
    v2.RandomRotation(20),
    v2.ColorJitter(0.4, 0.4, 0.4, 0.05),
    v2.ClampBoundingBoxes(),
    v2.SanitizeBoundingBoxes(labels_getter=None),   # bỏ hộp bị cắt mất hoặc quá nhỏ
])
img2, boxes2 = aug(img, boxes)
```

<div align=center><img src="../img/code/alchemy_code/20_1.png" alt="transforms.v2 biến đổi cả ảnh và hộp giới hạn"/></div>

Hình 3.1: Bốn lần áp cùng một pipeline `transforms.v2`; hộp giới hạn (đỏ) đi theo ảnh qua các phép cắt, lật, xoay (trang Code).

Lấy `resize` làm ví dụ, xem hiệu quả đổi kích thước ảnh đầu vào:

```python
orig_img = Image.open('cat.jpg')
print(orig_img.size)
# size: kích thước đầu ra mong muốn
resized_imgs = [v2.Resize(size=size)(orig_img) for size in (30, 50, 100, orig_img.size[::-1])]
```

<div align=center><img src="../img/code/alchemy_code/21_1.png" alt="Resize với các kích thước khác nhau"/></div>

Hình 3.2: `Resize` với `size` là 30, 50, 100 và kích thước gốc (trang Code).

> **Biên tập:** Khi `size` là một số nguyên, `Resize` đưa **cạnh ngắn** về giá trị đó và giữ tỉ lệ khung hình (ảnh 451×300 thành 45×30, 75×50, 150×100); muốn ảnh vuông phải truyền `(h, w)`. `size` dạng bộ là (cao, rộng), ngược với `PIL.Image.size` là (rộng, cao), nên code của bản gốc truyền `orig_img.size` cho ô cuối sẽ cho ảnh 300×451 bị kéo dẹt khi ảnh không vuông; đã sửa thành `orig_img.size[::-1]`. Code của bản gốc còn có `torch.manual_seed(0)` và chú thích về phép biến đổi ngẫu nhiên, nhưng `Resize` không ngẫu nhiên nên đã bỏ.

## 4 Tăng cường ảnh bằng imgaug và Albumentations

imgaug là thư viện tăng cường ảnh cho thí nghiệm học máy. Nó hỗ trợ nhiều kỹ thuật tăng cường, cho phép kết hợp chúng dễ dàng, chạy theo thứ tự ngẫu nhiên hoặc trên nhiều nhân CPU, có giao diện ngẫu nhiên đơn giản mà mạnh, và tăng cường được không chỉ ảnh mà cả điểm mốc (keypoint/landmark), hộp giới hạn, bản đồ nhiệt (heatmap) và bản đồ phân đoạn.

Các nhóm chức năng của imgaug:

- Basics: cơ bản
- Keypoints: điểm mốc
- Bounding Boxes: hộp giới hạn
- Heatmaps: bản đồ nhiệt
- Segmentation Maps and Masks: bản đồ phân đoạn và mặt nạ
- Stochastic Parameters: tham số ngẫu nhiên
- Blending/Overlaying images: trộn, chồng ảnh
- Augmenters: tổng quan các bộ tăng cường

Cách dùng từng phần xem [tài liệu imgaug](https://imgaug.readthedocs.io/en/latest/).

> **Biên tập:** imgaug không còn được bảo trì (bản cuối 0.4.0 năm 2020) và gặp lỗi với các bản NumPy mới. Thư viện thay thế phổ biến là [Albumentations](https://albumentations.ai/), có cùng phạm vi (ảnh, hộp, điểm mốc, mặt nạ) và nhanh hơn; hoặc dùng thẳng `torchvision.transforms.v2` như mục 3. Ảnh ví dụ của imgaug trong bản gốc được thay bằng kết quả của Albumentations dưới đây.

```python
import albumentations as A

aug = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.4, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.Affine(rotate=(-20, 20), p=0.7),
    A.ColorJitter(0.4, 0.4, 0.4, 0.05, p=0.8),
    A.OneOf([A.GaussianBlur(blur_limit=(3, 7)), A.GaussNoise(std_range=(0.05, 0.15))], p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.3))

out = aug(image=img_hwc_uint8, bboxes=[[120, 20, 340, 280]], labels=['mèo'])
img2, boxes2 = out['image'], out['bboxes']
```

<div align=center><img src="../img/code/alchemy_code/22_1.png" alt="Albumentations"/></div>

Hình 4.1: Bốn lần áp pipeline Albumentations ở trên, gồm cả làm mờ hoặc thêm nhiễu Gauss với xác suất 0,5 (trang Code, Albumentations 2.0).

## 5 Bổ sung: tăng cường dữ liệu từ 2017 tới nay

Mục này do người biên soạn viết thêm.

**Trộn ảnh.** **MixUp** (Zhang và cộng sự, 2018) lấy tổ hợp lồi của hai ảnh và hai nhãn one-hot với hệ số $\lambda \sim \mathrm{Beta}(\alpha, \alpha)$: $\tilde x = \lambda x_i + (1-\lambda) x_j$, $\tilde y = \lambda y_i + (1-\lambda) y_j$. **CutMix** (Yun và cộng sự, 2019) dán một hình chữ nhật cắt từ ảnh này vào ảnh kia, nhãn trộn theo tỉ lệ diện tích. Cả hai cho nhãn mềm, nên hàm mất mát phải nhận nhãn dạng phân phối (`F.cross_entropy` của PyTorch nhận được trực tiếp).

<div align=center><img src="../img/code/alchemy_code/23_1.png" alt="MixUp và CutMix"/></div>

Hình 5.1: MixUp (trên) và CutMix (dưới) trên một batch FashionMNIST bằng `v2.MixUp`, `v2.CutMix`; tiêu đề ghi hai lớp có trọng số lớn nhất trong nhãn mềm (trang Code).

**Chính sách tăng cường tự động.** **AutoAugment** (Cubuk và cộng sự, 2019) dùng học tăng cường để tìm chính sách (cặp phép biến đổi, xác suất, cường độ) cho từng tập dữ liệu, rất tốn kém. **RandAugment** (2020) thay bằng hai siêu tham số: số phép $N$ chọn ngẫu nhiên đều mỗi ảnh và cường độ chung $M$. **TrivialAugment** (2021) còn đơn giản hơn: mỗi ảnh một phép, cường độ ngẫu nhiên, không có siêu tham số, mà kết quả ngang hoặc hơn.

<div align=center><img src="../img/code/alchemy_code/24_1.png" alt="RandAugment, TrivialAugment, AutoAugment, RandomErasing"/></div>

Hình 5.2: Mỗi hàng là bốn lần áp một chính sách lên cùng một ảnh: RandAugment(2, 9), TrivialAugmentWide, AutoAugment (chính sách ImageNet) và RandomErasing (trang Code).

**Xoá ngẫu nhiên.** **Cutout** (2017) và **Random Erasing** (2020) che một hình chữ nhật ngẫu nhiên của ảnh, buộc mô hình không dựa vào một vùng duy nhất.

**Công thức huấn luyện hiện nay.** Các mô hình phân loại ảnh mạnh (DeiT, ConvNeXt, công thức torchvision cho ResNet-50) dùng chung một bộ: RandomResizedCrop + lật ngang, RandAugment hoặc TrivialAugment, MixUp + CutMix (chọn ngẫu nhiên một trong hai cho mỗi batch), Random Erasing, cộng label smoothing. Với phát hiện đối tượng, **Mosaic** (ghép 4 ảnh, YOLOv4) và **copy-paste** (dán đối tượng từ ảnh này sang ảnh khác theo mặt nạ) là các phép mạnh nhất; YOLO thường tắt Mosaic ở vài epoch cuối để mô hình quen với ảnh thật. **Tăng cường lúc kiểm tra** (test-time augmentation, TTA), như dự đoán trên ảnh gốc và ảnh lật rồi lấy trung bình, cho thêm chút độ chính xác với chi phí suy luận gấp nhiều lần.

**Dữ liệu tổng hợp.** Từ khoảng 2023, mô hình khuếch tán (diffusion) được dùng để sinh thêm ảnh huấn luyện cho lớp ít mẫu hoặc để thay nền, đổi điều kiện chụp; hiệu quả phụ thuộc nhiều vào việc ảnh sinh có giống phân phối thật không.

## Tài liệu tham khảo

1. Ngụy Tú Tham. *Giải tích mạng neuron tích chập* (解析卷积神经网络), chương 5, 6.
2. *OpenCV-Python Tutorial* (bản tiếng Trung).
3. [torchvision: Transforming and augmenting images](https://pytorch.org/vision/stable/transforms.html)
4. Buslaev et al. [Albumentations: Fast and Flexible Image Augmentations](https://www.mdpi.com/2078-2489/11/2/125). Information, 2020.
5. Zhang et al. [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412). ICLR 2018.
6. Yun et al. [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899). ICCV 2019.
7. Cubuk et al. [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719). NeurIPS 2020.
8. Müller, Hutter. [TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation](https://arxiv.org/abs/2103.10158). ICCV 2021.
9. Zhong et al. [Random Erasing Data Augmentation](https://arxiv.org/abs/1708.04896). AAAI 2020.
10. Ghiasi et al. [Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177). CVPR 2021.
