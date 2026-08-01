# Thống kê log trong `model_right`

Nguồn dữ liệu: 5 file `log.txt` trong `model_right/*/train/*/log.txt`.

Lệnh kiểm tra nhanh:

```bash
python3 code/summarize_logs.py --model-dir model_right --format markdown
```

## Cách đọc số liệu

- `UNet` là nhánh conventional model trong log, được đánh dấu bởi block `test unet model`.
- `SAM/student` là nhánh foundation model/student trong log, được đánh dấu bởi block `test sam model` và các dòng `stu_val_*`.
- `Best avg Dice` lấy từ `val_best_avg_dice`, tức checkpoint có Dice trung bình tốt nhất, phù hợp với cách code lưu `*_avg_dice_best_model.pth`.
- `Final avg Dice` là Dice trung bình ở lần evaluation cuối cùng. Chỉ số này không nên dùng làm kết quả chính nếu báo cáo theo checkpoint tốt nhất.
- Các log này đều dùng `seed=1337`, vì vậy đây là kết quả một lần chạy, chưa có mean/std nhiều seed.

## Tổng quan cấu hình

| Dataset | Save name | lb_num | Domain num | Max iterations | Last iteration | Last epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BUSI | `BUSI_dm1_lb64_paper` | 64 | 2 | 30000 | 30000 | 60 |
| BUSI | `BUSI_dm1_lb129_paper` | 129 | 2 | 30000 | 30000 | 60 |
| MNMS | `MNMS_dm1_lb5_paper` | 5 | 4 | 60000 | 60000 | 120 |
| fundus | `fundus_dm1_lb20_paper` | 20 | 4 | 30000 | 30000 | 60 |
| prostate | `prostate_dm1_lb20_paper` | 20 | 6 | 60000 | 60000 | 120 |

## Kết quả chi tiết

| Dataset | lb_num | Branch | Best avg Dice | Best iter | Dice thành phần tại best avg | Final avg Dice |
| --- | ---: | --- | ---: | ---: | --- | ---: |
| BUSI | 64 | UNet | 0.578330 | 11000 | base=0.578330 | 0.514604 |
| BUSI | 64 | SAM/student | 0.671036 | 27500 | base=0.671036 | 0.576435 |
| BUSI | 129 | UNet | 0.606139 | 23500 | base=0.606139 | 0.490227 |
| BUSI | 129 | SAM/student | 0.732751 | 6000 | base=0.732751 | 0.584510 |
| MNMS | 5 | UNet | 0.788516 | 33000 | lv=0.852252, myo=0.765161, rv=0.748134 | 0.751585 |
| MNMS | 5 | SAM/student | 0.794637 | 34500 | lv=0.857438, myo=0.775278, rv=0.751194 | 0.767342 |
| fundus | 20 | UNet | 0.877648 | 14500 | cup=0.826829, disc=0.928468 | 0.856847 |
| fundus | 20 | SAM/student | 0.882851 | 15000 | cup=0.836663, disc=0.929039 | 0.853123 |
| prostate | 20 | UNet | 0.877299 | 60000 | base=0.877299 | 0.877299 |
| prostate | 20 | SAM/student | 0.877399 | 55000 | base=0.877399 | 0.873505 |

## Chênh lệch giữa SAM/student và UNet

| Dataset | lb_num | SAM best - UNet best | Nhận xét nhanh |
| --- | ---: | ---: | --- |
| BUSI | 64 | +0.092706 | SAM/student vượt UNet rõ ràng. |
| BUSI | 129 | +0.126612 | SAM/student vượt UNet rõ ràng nhất trong các log. |
| MNMS | 5 | +0.006121 | SAM/student nhỉnh hơn nhẹ. |
| fundus | 20 | +0.005203 | SAM/student nhỉnh hơn nhẹ ở best checkpoint. |
| prostate | 20 | +0.000100 | Hai nhánh gần như hòa. |

Macro-average trên 5 cấu hình:

| Branch | Mean best avg Dice |
| --- | ---: |
| UNet | 0.745586 |
| SAM/student | 0.791735 |

## Phân tích

Kết quả `model_right` đang dùng đúng label budget theo paper: prostate/fundus dùng 20 labels, MNMS dùng 5 labels, BUSI có hai thiết lập 64 labels và 129 labels. BUSI không phải chạy lặp lại cùng một cấu hình, mà là hai thiết lập số nhãn riêng.

SAM/student là nhánh có kết quả tốt hơn UNet ở tất cả 5 cấu hình nếu dùng `Best avg Dice`. Lợi thế lớn nhất nằm ở BUSI, đặc biệt `lb129` tăng 0.126612 Dice so với UNet. Với MNMS, fundus và prostate, chênh lệch nhỏ hơn nhiều; prostate gần như không có khác biệt.

Không nên lấy `Final avg Dice` làm kết quả chính. Ở nhiều log, kết quả cuối kém checkpoint tốt nhất khá rõ: BUSI `lb129` của SAM/student giảm từ 0.732751 xuống 0.584510, và UNet giảm từ 0.606139 xuống 0.490227. Điều này cho thấy quá trình training có dao động/thoái lui sau best checkpoint, nên report cần nói rõ là lấy best checkpoint theo validation/test log.

Fundus và MNMS có log thành phần theo lớp, nên có thể viết kết quả chi tiết hơn: fundus cải thiện chủ yếu ở cup Dice, còn disc Dice gần như không đổi; MNMS cải thiện nhẹ trên cả lv, myo và rv. BUSI/prostate chỉ có `base_dice`, nên không có phân tích theo lớp.

Hạn chế lớn nhất của thống kê hiện tại là mỗi cấu hình chỉ có một seed (`seed=1337`). Nếu muốn kết luận chặt chẽ hơn, cần chạy thêm nhiều seed và báo cáo mean/std. Ngoài ra, `model_right` chỉ bao gồm cấu hình theo paper; không nên trộn với các log `model`/`model_wrong` có `lb40_full` vì khác label budget.
