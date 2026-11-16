# Model evaluation · Mô phỏng

> Ghi chú của người biên soạn: mô phỏng này do người biên soạn làm thêm.

## Ngưỡng, ma trận nhầm lẫn, ROC và precision–recall

<small class="viz-ref">Mục E.2–E.7 · hình E.2, E.5–E.8</small>

Một mô hình chấm điểm cho 1000 mẫu. Điểm số của mỗi lớp có phân phối chuẩn: lớp âm vẽ phía trên trục, lớp dương phía dưới. Mọi mẫu bên phải ngưỡng bị gọi là dương. Kéo ngưỡng trực tiếp trên biểu đồ, hoặc dùng thanh trượt, rồi thử bốn việc:

1. Ở chế độ **hai lớp cân bằng**, kéo ngưỡng qua lại. Precision và recall đổi chỗ cho nhau, chấm đen chạy dọc đường ROC và đường PR, còn AUC thì đứng yên.
2. Bấm **Bệnh hiếm 2%, ngưỡng cao**. Accuracy gần 0.98 nhưng chỉ ngang mốc "luôn đoán âm" (vạch đen trên thanh accuracy), trong khi recall rất thấp.
3. Bấm **Bệnh hiếm, hạ ngưỡng**. Recall tăng vọt, nhưng số mẫu báo nhầm (FP) lớn hơn nhiều lần số mẫu bắt đúng (TP). Đường ROC vẫn đẹp, còn đường PR thì tụt sâu. Bật **Cùng thang cho hai lớp** để thấy lớp dương nhỏ bé thế nào.
4. Kéo **độ tách hai lớp** về 0: mô hình đoán mò, ROC bám đường chéo và AUC về 0.5.

<iframe class="viz-frame" src="viz/ml-eval.html?embed" title="Mô phỏng: Ngưỡng và chỉ số đánh giá" loading="lazy"></iframe>
