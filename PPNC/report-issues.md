# Báo cáo LaTeX PPNC: Các phần cần chỉnh sửa

Phạm vi: đã rà soát báo cáo LaTeX trong `PPNC/` đối chiếu với `2503.16997v1.pdf`, mã nguồn cục bộ hiện tại và các log huấn luyện hiện có.

## Mức ưu tiên cao

### 1. Các cột miền đang được so sánh với ý nghĩa khác nhau

Vị trí: `PPNC/sections/04-experiment.tex:21-38`, `PPNC/sections/04-experiment.tex:47-64`, `PPNC/sections/04-experiment.tex:73-81`, `PPNC/sections/04-experiment.tex:115-130`

Vấn đề: Trong các bảng của bài báo gốc, các cột như RUNMC, BMC, HCRUDB, UCL, BIDMC, HK là các thiết lập miền nguồn có nhãn. Bài báo nêu rằng mỗi cột báo cáo hiệu năng trung bình trên tất cả các miền kiểm thử khi dữ liệu có nhãn đến từ miền nguồn đó. Trong các log cục bộ, `domain1`, `domain2`, v.v. là kết quả trên từng miền kiểm thử từ một lần chạy với `--lb_domain 1`.

Vì sao chưa hợp lý: Báo cáo đặt các giá trị baseline từ bài báo gốc và các giá trị SynFoC cục bộ vào cùng một bảng như thể mỗi cột có cùng ý nghĩa. Thực tế chúng không cùng ý nghĩa.

Đề xuất chỉnh sửa: Tách bảng thành hai phần: baseline/trung bình theo miền nguồn của bài báo và kết quả miền kiểm thử cục bộ với `lb_domain=1`. Nếu cần so sánh nghiêm ngặt, cần chạy lại từng thiết lập miền nguồn và báo cáo theo cùng quy trình tổng hợp như bài báo.

### 2. Chế độ số lượng nhãn bị trộn lẫn giữa bài báo và log cục bộ

Vị trí: `PPNC/sections/04-experiment.tex:15`, `PPNC/sections/04-experiment.tex:17-38`, `PPNC/sections/04-experiment.tex:43-64`, `PPNC/sections/04-experiment.tex:69-98`, `PPNC/sections/04-experiment.tex:115-130`

Vấn đề: Bài báo dùng 20 mẫu có nhãn cho Prostate và Fundus, 5 mẫu cho M&Ms, và 64/129 mẫu cho BUSI. Các bảng hiện tại trong báo cáo lại dùng log cục bộ nằm trong `model/*/train/*_lb40_full`, tất cả đều bắt đầu với `--lb_num 40`.

Vì sao chưa hợp lý: Việc in đậm SynFoC là phương pháp tốt nhất so với baseline của bài báo là gây hiểu nhầm khi SynFoC được đánh giá với số lượng nhãn khác.

Đề xuất chỉnh sửa: Hoặc dùng các log gần với thiết lập bài báo hơn trong `model_right/` nếu có, hoặc giữ kết quả `lb40_full` nhưng bỏ các khẳng định trực tiếp rằng đây là phương pháp tốt nhất so với baseline 20 nhãn/5 nhãn/64 nhãn.

### 3. Công thức SMC được mô tả không khớp với triển khai cục bộ hiện tại

Vị trí: `PPNC/sections/03-method.tex:42-61`

Vấn đề: Báo cáo mô tả SMC theo bài báo gốc ở mức từng mẫu, trong đó `alpha = Phi_self * Phi_mut`. Triển khai cục bộ hiện tại import `blend_region_wise_probabilities` từ `code/utils/region_smc.py`, hàm này tính `alpha_map` theo không gian từ độ tin cậy cục bộ, độ tin cậy tự thân và độ tin cậy tương hỗ.

Vì sao chưa hợp lý: Nếu các kết quả trong báo cáo đến từ mã hiện tại, báo cáo đang không mô tả đúng phương pháp đã tạo ra các số liệu. Nếu mục tiêu là tái lập chính xác bài báo, mã hiện tại là một biến thể cục bộ và không nên được mô tả như SynFoC không thay đổi.

Đề xuất chỉnh sửa: Cần xác định rõ định danh của báo cáo. Nếu là tái lập chính xác, hãy khôi phục/báo cáo công thức trong bài báo. Nếu là biến thể cục bộ, hãy đổi tên phần này thành sửa đổi SMC theo vùng và giải thích điểm khác biệt so với SynFoC.

### 4. Trích dẫn SynFoC bị thiếu hoặc đang trỏ tới sai bài báo

Vị trí: `PPNC/sections/00-abstract.tex:2`, `PPNC/sections/01-introduction.tex:6-8`, `PPNC/sections/03-method.tex:3`, `PPNC/references.bib`

Vấn đề: Báo cáo nhiều lần trích dẫn `ma2024constructing` cho các khẳng định riêng của SynFoC. Trong `PPNC/references.bib`, `ma2024constructing` là bài SymGD CVPR 2024, không phải `2503.16997v1.pdf`.

Vì sao chưa hợp lý: Các khẳng định cốt lõi về SynFoC, SMC, CDCR và kết quả gốc cần trích dẫn chính bài báo SynFoC.

Đề xuất chỉnh sửa: Thêm một mục BibTeX cho `arXiv:2503.16997v1` và dùng mục này cho tất cả phát biểu riêng về SynFoC. Chỉ giữ `ma2024constructing` cho phần nền tảng MiDSS/SymGD.

### 5. Khẳng định rằng triển khai bám theo mã gốc là quá mạnh

Vị trí: `PPNC/sections/00-abstract.tex:2`, `PPNC/sections/01-introduction.tex:6`, `PPNC/sections/04-experiment.tex:12`

Vấn đề: Báo cáo nói rằng triển khai bám theo mã nguồn gốc được công bố, nhưng mã cục bộ hiện tại có module SMC theo vùng và các log được báo cáo là các lần chạy `lb40_full` cục bộ.

Vì sao chưa hợp lý: Cách diễn đạt này tạo cảm giác đây là tái lập chính xác, trong khi bằng chứng cho thấy hoặc đã có sửa đổi triển khai, hoặc ít nhất đã dùng thiết lập khác với bài báo.

Đề xuất chỉnh sửa: Thay bằng phát biểu hẹp hơn, ví dụ "dựa trên triển khai SynFoC được công bố, với các lần chạy cục bộ sử dụng ..." và liệt kê rõ các điểm sai khác.

### 6. Kết luận "log tái lập cao hơn" không hợp lệ

Vị trí: `PPNC/sections/04-experiment.tex:115-130`

Vấn đề: Bảng `original_vs_reproduced` nói rằng log tái lập cao hơn cho Prostate, Fundus và M&Ms.

Vì sao chưa hợp lý: Các thiết lập khác nhau về số lượng nhãn, ý nghĩa cột miền và có thể cả triển khai SMC. Do đó, so sánh cao/thấp không có ý nghĩa trong các điều kiện khác nhau này.

Đề xuất chỉnh sửa: Thay các nhận xét "cao hơn" bằng "không thể so sánh trực tiếp" và chỉ báo cáo chênh lệch mô tả, trừ khi dùng cùng một protocol.

## Mức ưu tiên trung bình

### 7. Định dạng in đậm "tốt nhất" gây hiểu nhầm trong các bảng trộn thiết lập

Vị trí: `PPNC/sections/04-experiment.tex:37-38`, `PPNC/sections/04-experiment.tex:63-64`

Vấn đề: Báo cáo in đậm các giá trị SynFoC như kết quả tốt nhất giữa các phương pháp, nhưng baseline và các hàng SynFoC không đến từ cùng một protocol thí nghiệm.

Đề xuất chỉnh sửa: Bỏ in đậm so sánh chéo phương pháp trong các bảng trộn thiết lập. Chỉ dùng in đậm trong bảng mà mọi hàng cùng chia sẻ một protocol.

### 8. Các hình định tính có thể không phải đầu ra tái lập

Vị trí: `PPNC/sections/04-experiment.tex:103-112`

Vấn đề: Các file hình chứa nhiều phương pháp baseline và có vẻ khớp với các so sánh trực quan kiểu bài báo gốc. Báo cáo nói rằng các hình hỗ trợ xu hướng định lượng tái lập và nhắc đến "mặt nạ tái lập".

Vì sao cần thận trọng: Nếu các hình được sao chép/chỉnh sửa từ bài báo thay vì được sinh từ checkpoint cục bộ, báo cáo không nên trình bày chúng như bằng chứng tái lập cục bộ.

Đề xuất chỉnh sửa: Ghi rõ "điều chỉnh từ bài báo gốc" nếu đúng, hoặc thay bằng hình được tạo từ checkpoint cục bộ.

### 9. Báo cáo diễn giải quá mức bằng chứng về SMC và CDCR trong phần tái lập cục bộ

Vị trí: `PPNC/sections/00-abstract.tex:2`, `PPNC/sections/01-introduction.tex:8`, `PPNC/sections/05-discussion.tex:4`

Vấn đề: Báo cáo nói rằng SMC và CDCR cải thiện độ ổn định và là các cơ chế hữu ích. Bài báo chứng minh điều này bằng ablation, nhưng báo cáo hiện tại không có bảng ablation cục bộ cho SMC-only, CDCR-only, CR, alpha hằng số, CPS, tuyến tính, v.v.

Đề xuất chỉnh sửa: Gán các khẳng định này cho ablation của bài báo SynFoC gốc, hoặc bổ sung ablation cục bộ trước khi xem chúng là kết quả đã tái lập.

### 10. Cách trình bày chọn mô hình cuối khác với khung suy luận của bài báo

Vị trí: `PPNC/sections/04-experiment.tex:17`, `PPNC/sections/04-experiment.tex:69`, `PPNC/sections/05-discussion.tex:2`

Vấn đề: Báo cáo nhấn mạnh nhánh nào có DSC trung bình cục bộ tốt nhất, ví dụ U-Net cho Prostate và M&Ms. Bài báo gốc vẫn đặt SynFoC trong khung cải thiện cả hai nhánh và nói rằng MedSAM được giữ lại trong giai đoạn kiểm thử.

Đề xuất chỉnh sửa: Nếu báo cáo cả hai nhánh, hãy nói rõ đây là đánh giá chẩn đoán theo từng nhánh. Tránh hàm ý rằng chính sách suy luận cuối cùng của SynFoC là chọn nhánh thắng theo từng dataset, trừ khi đó là một protocol mới được định nghĩa rõ.

### 11. Nguồn gốc của Jaccard, 95HD và ASD cần được nêu rõ

Vị trí: `PPNC/sections/04-experiment.tex:9`, tất cả các bảng kết quả

Vấn đề: Phần mô tả nói rằng các hàng được lấy từ checkpoint có DSC trung bình tốt nhất. Log rõ ràng theo dõi DSC trung bình tốt nhất, nhưng báo cáo cần nói rõ Jaccard, 95HD và ASD được lấy từ cùng checkpoint đó, từ một lần kiểm thử lại riêng, hay từ các giá trị tốt nhất độc lập theo từng metric.

Đề xuất chỉnh sửa: Thêm một câu giải thích cách trích xuất metric và tránh trộn các giá trị tốt nhất được chọn độc lập.

### 12. M&Ms và BUSI có cơ sở so sánh yếu hơn Prostate/Fundus

Vị trí: `PPNC/sections/04-experiment.tex:69-98`

Vấn đề: Các bảng M&Ms và BUSI chỉ gồm các hàng SynFoC U-Net và SynFoC MedSAM cục bộ. Chúng không so sánh với baseline gốc trong cùng một bảng.

Đề xuất chỉnh sửa: Hoặc giữ chúng như kết quả cục bộ theo từng nhánh, hoặc bổ sung baseline có thể so sánh được dưới cùng protocol.

## Mức ưu tiên thấp

### 13. "Version-2 logs" là cách gọi nội bộ

Vị trí: `PPNC/sections/05-discussion.tex:2`

Vấn đề: "version-2 logs" không có ý nghĩa rõ ràng với độc giả bên ngoài.

Đề xuất chỉnh sửa: Thay bằng "các log cục bộ hiện có" hoặc nêu rõ nhóm lần chạy cụ thể.

### 14. Kết luận chứa ngôn ngữ quy trình/meta

Vị trí: `PPNC/sections/06-conclusion.tex:2`

Vấn đề: Kết luận nói rằng bản thảo đã được tổ chức lại theo bố cục IEEE và "sẵn sàng nộp hơn". Đây là ghi chú chỉnh sửa dự án, không phải kết luận khoa học.

Đề xuất chỉnh sửa: Kết luận bằng các phát hiện về khả năng tái lập, các sai khác so với bài báo và các giới hạn.

### 15. Tóm tắt cần nêu rõ sai khác trong tái lập

Vị trí: `PPNC/sections/00-abstract.tex:2`

Vấn đề: Tóm tắt nói rằng báo cáo đánh giá triển khai được tái lập nhưng không nhắc rằng các kết quả hiện có dùng log cục bộ 40 nhãn, miền có nhãn 1 và có thể không khớp protocol bài báo gốc.

Đề xuất chỉnh sửa: Thêm một câu ngắn về giới hạn hoặc tránh các khẳng định nhạy với protocol trong phần tóm tắt.

## Hướng viết lại được khuyến nghị

Dùng cấu trúc sau nếu báo cáo cần phản ánh trung thực bằng chứng hiện tại:

1. "Chúng tôi nghiên cứu SynFoC dựa trên bài báo gốc và triển khai cục bộ."
2. "Các lần chạy cục bộ hiện tại không phải tái lập nghiêm ngặt theo protocol của bài báo vì dùng `lb_domain=1`, `lb_num=40` và mã SMC theo vùng cục bộ."
3. "Bảng 1-2 chỉ tóm tắt baseline của bài báo như bối cảnh."
4. "Bảng 3-4 báo cáo kết quả cục bộ theo từng nhánh từ các log đã hoàn tất."
5. "Không đưa ra khẳng định trực tiếp về state-of-the-art trừ khi dùng cùng số lượng nhãn, cùng protocol miền nguồn và cùng triển khai phương pháp."
