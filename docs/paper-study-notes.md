# Notes khi đọc `2503.16997v1.pdf`

File này dùng để ghi lại các điểm cần nhớ, các câu hỏi đã được giải thích, và các nhận xét quan trọng khi đọc paper.

## Note 1 - Vì sao Section 3.3 chỉ xét self-confidence của U-Net?

Trong Section 3.3, bài báo định nghĩa pseudo-label ensemble:

```text
P_hat_w^EN = alpha * P_hat_w^UT + (1 - alpha) * P_hat_w^MS
```

Trong đó:

| Ký hiệu | Ý nghĩa |
|---|---|
| `P_hat_w^UT` | Dự đoán của teacher U-Net trên ảnh không nhãn weak augmentation `Uw` |
| `P_hat_w^MS` | Dự đoán của teacher MedSAM trên `Uw` |
| `alpha` | Trọng số quyết định tin U-Net nhiều hay MedSAM nhiều |

Điểm dễ thắc mắc: paper tính **self-confidence** bằng cách so sánh U-Net student với U-Net teacher, nhưng không tính tương tự giữa MedSAM student và MedSAM teacher.

Lý do chính: `alpha` được thiết kế để đo **nên tin U-Net bao nhiêu**, không phải để đo độ tin cậy đối xứng của cả hai mô hình.

Nếu:

```text
alpha lớn
```

thì ensemble tin U-Net nhiều hơn.

Nếu:

```text
alpha nhỏ
```

thì ensemble tin MedSAM nhiều hơn.

Vì vậy tác giả cần đánh giá xem U-Net ở thời điểm hiện tại có đáng tin hay không. Họ dùng hai tiêu chí:

```text
Phi_self = độ giống nhau giữa U-Net student và U-Net teacher
Phi_mut  = độ giống nhau giữa U-Net teacher và MedSAM teacher
alpha = Phi_self * Phi_mut
```

Diễn giải trực quan:

- `Phi_self` trả lời câu hỏi: **U-Net có tự ổn định với chính nó không?**
- `Phi_mut` trả lời câu hỏi: **U-Net có đồng thuận với MedSAM không?**
- `alpha` trả lời câu hỏi: **Nên cho U-Net bao nhiêu quyền trong pseudo-label cuối?**

MedSAM vẫn có student và teacher trong framework, nhưng trong thiết kế này MedSAM được dùng như một nguồn tham chiếu mạnh hơn vì nó là foundation model đã có prior knowledge từ pretraining. Do đó, bài báo không dùng self-confidence của MedSAM để tính `alpha`.

Một thiết kế đối xứng hơn có thể tính thêm:

```text
Phi_self^MS = độ giống nhau giữa MedSAM student và MedSAM teacher
```

rồi dùng cả độ tin cậy của U-Net và MedSAM để trộn pseudo-label. Tuy nhiên, đó không phải thiết kế của SynFoC trong paper này. SynFoC chọn hướng thực dụng hơn: dùng MedSAM làm điểm tựa, còn `alpha` chủ yếu kiểm soát mức độ tin vào U-Net.

