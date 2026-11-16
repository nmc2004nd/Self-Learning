# Chương 19 · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm, không có trong sách gốc.

## Phân tích ma trận

<small class="viz-ref">Mục 19.2 · công thức (19.4)–(19.7)</small>

Gợi ý:

1. **Bấm "Phát":** các vector ẩn của sản phẩm (vuông) và người dùng (tròn) dịch chuyển dần; hàm mất mát giảm.
2. **Kéo $\lambda$ lên khoảng 1:** mọi vector co về gốc, mô hình chỉ còn các độ lệch. Đây là điều đã xảy ra với tham số của sách trên MovieLens ([trang Code](ml/ch19_code_vi)).
3. **Kéo $\lambda$ xuống rất nhỏ và chạy lâu:** mô hình khớp gần như hoàn toàn 20 ô đã biết, còn dự đoán cho các ô trống trở nên cực đoan (quá khớp).

<iframe class="viz-frame" src="viz/ml-mf.html?embed" title="Mô phỏng: Phân tích ma trận" loading="lazy"></iframe>
