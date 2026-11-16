# Chương 26 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Lề lớn nhất và vector hỗ trợ

<small class="viz-ref">Mục 26.1.2, 26.2, 26.3.5 · công thức (26.3), (26.14) · Hình 26.2, 26.4</small>

SVM được giải ngay trong trình duyệt bằng thuật toán SMO (cùng cách chọn biến như libsvm, thư viện bên dưới `sklearn.svm.SVC`).

Gợi ý:

1. **Ví dụ của sách:** 20 điểm của mục 26.4 cho đúng ba vector hỗ trợ như Hình 26.4. Bấm xoá một điểm không có vòng tròn: đường phân chia đứng yên. Xoá một điểm có vòng tròn: đường xoay đi và lề rộng ra.
2. **Bật "Hiện một nghiệm của PLA"** rồi bấm "Nghiệm PLA khác" vài lần: mỗi nghiệm của perceptron ([chương 13](ml/ch13_vi)) đều phân loại đúng, nhưng luôn có một điểm nằm sát nó hơn lề của SVM.
3. **Thêm một điểm vào giữa hai đường nét đứt**, đúng phía lớp của nó: điểm mới trở thành vector hỗ trợ và lề hẹp lại. Thêm nó ở sai phía: bài toán lề cứng vô nghiệm.

<iframe class="viz-frame" src="viz/ml-svm.html?embed" title="Mô phỏng: lề lớn nhất" loading="lazy"></iframe>
