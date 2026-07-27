# Phân tích source code: các điểm có thể làm giảm hiệu năng và hướng tối ưu

Tài liệu này tổng hợp kết quả rà soát source code trong `code/` của repo, tập trung vào hai nghĩa của "hiệu năng":

- **Hiệu năng mô hình**: Dice/JC/HD/ASD, chất lượng pseudo-label, khả năng tổng quát qua domain shift.
- **Hiệu năng mã nguồn/vận hành**: tốc độ train/eval, tính ổn định, khả năng reproduce, khả năng mở rộng dự án.

Kết luận ngắn: **có thể cải thiện dự án đáng kể bằng cách tối ưu mã nguồn**. Những việc nên ưu tiên không phải refactor lớn, mà là sửa một số điểm trong loss, evaluation, EMA teacher, split labeled data, scheduler và SAM/LoRA loading.

## Phạm vi đã đọc

Đã rà soát các file Python chính trong:

- `code/train.py`, `code/test.py`
- `code/dataloaders/`
- `code/utils/`
- `code/networks/`
- `code/segment_anything/`
- `code/sam_lora_image_encoder.py`
- `code/summarize_logs.py` và test liên quan

Không xem các file `.pyc`, dữ liệu mẫu trong `data_format/`, và các bản snapshot train trong `model/*/train/*/train.py` như source chính, vì chúng là output/thực nghiệm được sinh ra từ script.

## Tổng kết ưu tiên

| Ưu tiên | Khu vực | Vấn đề | Ảnh hưởng | Hướng xử lý |
| --- | --- | --- | --- | --- |
| P0 | Loss semi-supervised | Mask Dice unsupervised xử lý sai class nền | Pseudo-label low-confidence vẫn tác động vào background, có thể kéo Dice xuống | Sửa `DiceLossWithMask._one_hot_mask_encoder()` để confidence mask được áp dụng cho mọi class |
| P0 | Loss semi-supervised | CE unsupervised chia trung bình trên toàn ảnh | Loss thay đổi theo mask ratio, threshold cao làm tín hiệu unsup gần biến mất | Chuẩn hóa theo số pixel hợp lệ: `sum(masked_loss) / mask.sum().clamp_min(1)` |
| P0 | Evaluation | `test.py` load sai đường dẫn checkpoint so với `train.py` | Dễ đánh giá nhầm hoặc fail checkpoint | Chuẩn hóa `experiment_dir`, hỗ trợ `train/<save_name>` hoặc tự động tìm checkpoint |
| P0 | Metric | HD/ASD không xử lý đầy đủ mask rỗng | Có thể crash hoặc phạt điểm sai cho slice không có object | Xử lý 4 case: both empty, pred empty, gt empty, both non-empty |
| P1 | Training | EMA teacher đang ở `train()` khi sinh pseudo-label | BN/running stats của teacher bị nhiễm batch unlabeled, teacher kém ổn định | Đặt EMA teacher `eval()` trong bước pseudo-label |
| P1 | LR schedule | Warmup hiện tại gần như không warmup thật | SAM có thể học quá chậm hoặc quá gắt tùy flag | Thêm scheduler theo iteration cho SAM và U-Net |
| P1 | Data split | Labeled set lấy `first N` sau sort file | Labeled data có thể lệch patient/slice/domain | Shuffle/stratified split bằng seed và lưu split file |
| P1 | Metric Dice | Smoothing Dice `+1` làm inflate object nhỏ/false positive | Metric có thể lạc quan giả, nhất là BUSI/cup nhỏ | Dùng epsilon nhỏ và policy both-empty rõ ràng |
| P2 | SAM/LoRA | SAM train không prompt, `multimask_output` bị bỏ qua | Mất lợi thế promptable segmentation của SAM | Thử prompt từ label/pseudo-label hoặc đặt tên rõ "semantic mode" |
| P2 | Runtime | CPU/NumPy/Scipy trong training loop | Train chậm, GPU bị idle | Đưa confidence/resize/metric sang torch GPU nếu có thể |
| P2 | Data transform | Resize image cho U-Net dùng nearest-neighbor | Ảnh đầu vào bị vỡ hạt/aliasing, có thể giảm Dice U-Net | Dùng bilinear/area cho image, nearest chỉ cho label |
| P3 | Maintainability | Một số import/debug dependency, API checkpoint, DSBN | Khó deploy/mở rộng | Dọn import, checkpoint dict đầy đủ, assert domain batch |

## Phát hiện chi tiết

### 1. Mask Dice unsupervised đang sai với confidence mask

Trong [code/utils/losses.py](../code/utils/losses.py), `DiceLossWithMask._one_hot_mask_encoder()` hiện tại:

```python
temp_prob = input_tensor * i == i * torch.ones_like(input_tensor)
```

Khi `i = 0`, điều kiện này luôn đúng, vì `input_tensor * 0 == 0`. Nghĩa là class background luôn được tính Dice trên toàn bộ ảnh, kể cả vùng low-confidence đã bị threshold loại bỏ.

Nó được dùng trong unsupervised loss tại [code/train.py](../code/train.py#L621) và [code/train.py](../code/train.py#L623). Nếu confidence mask có ý nghĩa là "chỉ học từ pixel đủ tin cậy", cách hiện tại phá vỡ ý nghĩa đó cho class nền.

Khuyến nghị:

- Nếu `mask` là confidence mask dạng shape `B,1,H,W`, hãy expand/repeat mask cho tất cả class.
- Nếu cần mask riêng theo class, nên tạo mask từ `target_onehot * confidence_mask`, không so sánh `input_tensor * i`.
- Thêm unit test nhỏ cho `DiceLossWithMask`: với mask toàn 0 thì Dice unsup phải gần 0 contribution cho mọi class.

Mức độ tác động: **cao**. Đây là điểm có khả năng ảnh hưởng trực tiếp đến Dice vì nằm ngay trong tín hiệu học từ unlabeled data.

### 2. CE unsupervised bị scale theo mask ratio

Trong [code/train.py](../code/train.py#L621) và [code/train.py](../code/train.py#L623):

```python
(ce_loss(logits, pseudo_label) * mask.squeeze(1)).mean()
```

`mean()` chia theo toàn bộ pixel, không chia theo số pixel được chọn. Khi threshold cao, mask nhỏ, loss bị nhỏ đi rất mạnh. Khi mask ratio dao động theo iteration/domain, scale của unsupervised objective cũng dao động theo, làm training khó ổn định.

Khuyến nghị:

```python
pixel_loss = ce_loss(logits, pseudo_label)
valid = mask.squeeze(1)
masked_ce = (pixel_loss * valid).sum() / valid.sum().clamp_min(1.0)
```

Có thể log riêng `mask_ratio` và `masked_ce` để xem tín hiệu unsup có thực sự đang học hay không.

Mức độ tác động: **cao**.

### 3. EMA teacher đang để `train()` khi tạo pseudo-label

Trong [code/train.py](../code/train.py#L471)-[code/train.py](../code/train.py#L474), cả student và EMA model đều được đặt `train()` mỗi epoch. Sau đó pseudo-label được tạo trong `torch.no_grad()` tại [code/train.py](../code/train.py#L527). `no_grad()` không ngăn BatchNorm cập nhật running mean/var nếu model ở train mode.

Với U-Net có BatchNorm trong [code/networks/unet_parts.py](../code/networks/unet_parts.py#L15), EMA teacher có thể bị "nhiễm" thống kê batch unlabeled, làm teacher kém ổn định.

Khuyến nghị:

- Student: `SAM_model.train()`, `unet_model.train()`.
- Teacher EMA khi sinh pseudo-label: `ema_SAM_model.eval()`, `ema_unet_model.eval()`.
- Nếu vẫn cần train mode cho một module đặc biệt, tách rõ module đó và test ablation.

Mức độ tác động: **trung bình-cao**.

### 4. Warmup/scheduler chưa thực sự hoạt động

Trong [code/train.py](../code/train.py#L440)-[code/train.py](../code/train.py#L448):

- Nếu `--warmup`, LR của SAM được gán bằng `base_lr / warmup_period`, nhưng không có bước nào tăng dần lên `base_lr`.
- U-Net luôn dùng `base_lr`.
- `LambdaLR` được import tại [code/train.py](../code/train.py#L33) nhưng không dùng.
- Nhánh không `--AdamW` dùng `model.parameters()` tại [code/train.py](../code/train.py#L447), trong khi `model` không tồn tại trong scope này; chạy không có `--AdamW` sẽ lỗi.

Khuyến nghị:

- Sửa bug `model.parameters()` thành `SAM_model.parameters()`.
- Thêm scheduler theo iteration: warmup từ LR nhỏ lên target LR, sau đó cosine/poly decay.
- Tách LR cho SAM/LoRA và U-Net. LoRA/decoder của SAM thường nên có LR nhỏ hơn U-Net, hoặc ít nhất được log riêng.
- Log `sam_lr` và `unet_lr` mỗi epoch/iteration.

Mức độ tác động: **cao với cấu hình không AdamW**, **trung bình-cao với chất lượng train**.

### 5. Evaluation có nguy cơ load sai checkpoint

`train.py` tạo output tại [code/train.py](../code/train.py#L769):

```python
../model/<dataset>/train/<save_name>/
```

Nhưng `test.py` load tại [code/test.py](../code/test.py#L238) và [code/test.py](../code/test.py#L241):

```python
../model/<dataset>/<save_name>/...
```

Do đó, một run train bằng `--save_name prostate_dm1_lb40_full` sẽ cần test với `--save_name train/prostate_dm1_lb40_full`, không phải `--save_name prostate_dm1_lb40_full`. Đây là gotcha dễ đánh giá nhầm.

Thêm nữa, `test.py` default `--model unet` tại [code/test.py](../code/test.py#L26), trong khi `train.py` default `--model MedSAM` tại [code/train.py](../code/train.py#L47). Nếu người dùng truyền `--model MedSAM` vào `test.py`, code không vào nhánh `SAM` hoặc `unet` tại [code/test.py](../code/test.py#L237)-[code/test.py](../code/test.py#L242).

Khuyến nghị:

- Chuẩn hóa argument `--experiment_dir` hoặc `--checkpoint_dir`.
- Cho phép `--model SAM` và `--model MedSAM` cùng map vào nhánh LoRA SAM.
- Khi eval, assert checkpoint tồn tại và in đường dẫn checkpoint rõ ràng.
- Không tạo/chặn thư mục checkpoint trong eval như [code/test.py](../code/test.py#L295)-[code/test.py](../code/test.py#L298).

Mức độ tác động: **cao**, vì metric có thể không phản ánh run vừa train.

### 6. Metric HD/ASD xử lý mask rỗng chưa đúng

Trong [code/test.py](../code/test.py#L112)-[code/test.py](../code/test.py#L123) và bản copy trong [code/train.py](../code/train.py#L207)-[code/train.py](../code/train.py#L218), code chỉ check prediction rỗng:

- Pred rỗng: gán HD/ASD = 100.
- Pred không rỗng: gọi `medpy.binary.hd95/asd`.

Nhưng cần xử lý cả ground truth rỗng. Nếu pred và gt cùng rỗng, phạt 100 là không hợp lý. Nếu pred có object nhưng gt rỗng, MedPy có thể lỗi vì một input rỗng.

Khuyến nghị:

- `pred_empty and gt_empty`: Dice policy rõ ràng, HD/ASD = 0.
- `pred_empty xor gt_empty`: penalty cố định, ví dụ 100.
- Cả hai có object: gọi `hd95/asd`.

Mức độ tác động: **cao với dataset có slice/background rỗng**, đặc biệt BUSI/MNMS.

### 7. Dice metric có smoothing làm inflate kết quả object nhỏ

Trong [code/utils/metrics.py](../code/utils/metrics.py#L139)-[code/utils/metrics.py](../code/utils/metrics.py#L143), Dice dùng:

```python
(2 * intersection + 1.0) / (1.001 + segmentation_pixels + gt_label_pixels)
```

Với ground truth rỗng và prediction 1 pixel, Dice xấp xỉ `1 / 2.001`, tức gần 0.5. Điều này làm false positive nhỏ trong object nhỏ/rỗng nhìn tốt hơn thực tế.

Khuyến nghị:

- Dùng epsilon nhỏ, ví dụ `1e-6`, chỉ để tránh chia 0.
- Explicit policy:
  - both empty -> 1.0 nếu coi là đúng background;
  - one empty -> 0.0;
  - both non-empty -> công thức Dice chuẩn.
- Báo cáo rõ policy trong paper/report để so sánh công bằng.

Mức độ tác động: **trung bình-cao**.

### 8. Trung bình metric theo batch thay vì theo ảnh

Trong evaluation, `dice_calcu` trả trung bình theo batch, sau đó cộng/chia theo `len(cur_dataloader)` tại [code/test.py](../code/test.py#L136)-[code/test.py](../code/test.py#L137). Nếu `test_bs > 1`, batch cuối nhỏ vẫn có trọng số bằng batch đầy.

Default `test_bs=1` nên hiện tại ít bị ảnh hưởng, nhưng code để người dùng đổi `--test_bs`.

Khuyến nghị:

- Tích lũy tổng metric theo số ảnh.
- Hoặc khóa `test_bs=1` khi tính metric paper.
- Log `num_cases`/`num_slices` mỗi domain.

Mức độ tác động: **trung bình**.

### 9. Labeled split lấy first-N sau khi sort file

Trong [code/train.py](../code/train.py#L415)-[code/train.py](../code/train.py#L422), labeled index là:

```python
lb_idxs = list(range(lb_num))
unlabeled_idxs = list(range(lb_num, data_num))
```

Trong dataloader, file được `imagelist.sort()` trước khi chọn index. Nếu tên file sắp theo patient/case/slice, labeled set sẽ không đại diện cho phân phối thật.

Khuyến nghị:

- Tạo split bằng RNG seed: shuffle index rồi lấy `lb_num`.
- Lưu split vào file JSON/TXT để reproduce.
- Với prostate/MNMS, nên split theo patient/case thay vì slice nếu có metadata.
- Với BUSI, nên stratified theo benign/malignant và kích thước lesion.

Mức độ tác động: **cao với hiệu năng mô hình và độ tin cậy thực nghiệm**.

### 10. Data transform có thể làm xấu ảnh đầu vào U-Net

Trong [code/dataloaders/custom_transforms.py](../code/dataloaders/custom_transforms.py#L766)-[code/dataloaders/custom_transforms.py](../code/dataloaders/custom_transforms.py#L778), `zoom(..., order=0)` được dùng cho cả `unet_size_img` và `unet_size_strong_aug`. `order=0` là nearest-neighbor, phù hợp cho label nhưng không phù hợp cho image liên tục.

Ảnh y tế bị resize nearest-neighbor có thể bị blocky/aliasing, làm U-Net học biên và texture kém hơn.

Khuyến nghị:

- Image: dùng bilinear (`order=1`) hoặc area/antialias.
- Label: giữ nearest-neighbor (`order=0`).
- Viết test shape/value range để đảm bảo không làm đổi label class.

Mức độ tác động: **trung bình**.

### 11. So sánh string bằng `is`

Trong [code/dataloaders/dataloader.py](../code/dataloaders/dataloader.py#L100), [code/dataloaders/dataloader.py](../code/dataloaders/dataloader.py#L231), [code/dataloaders/dataloader.py](../code/dataloaders/dataloader.py#L343) và nhiều vị trí khác, code dùng:

```python
if _img.mode is 'RGB':
```

`is` là identity comparison, không phải value comparison. Trong Python, string interning có thể làm nó "có vẻ đúng" trong một số trường hợp, nhưng không đảm bảo.

Khuyến nghị:

- Đổi tất cả thành `==`.
- Thêm lint rule hoặc test nhỏ cho dataloader mode conversion.

Mức độ tác động: **trung bình**.

### 12. SAM/MedSAM preprocessing cần được kiểm chứng bằng ablation

Khi tạo SAM tại [code/train.py](../code/train.py#L354)-[code/train.py](../code/train.py#L358), repo truyền:

```python
pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1]
```

Trong khi `Normalize_tf(dataRange=[0,1])` đưa image về `[0,1]`. Nếu checkpoint là SAM gốc, pretrained normalization thường dựa trên pixel `[0,255]` và ImageNet mean/std trong [code/segment_anything/modeling/sam.py](../code/segment_anything/modeling/sam.py#L28)-[code/segment_anything/modeling/sam.py](../code/segment_anything/modeling/sam.py#L48). Nếu checkpoint MedSAM của dự án đã được train với `[0,1]` thì cách này có thể đúng.

Khuyến nghị:

- Chạy ablation:
  - MedSAM checkpoint + current `[0,1]`.
  - SAM checkpoint + SAM normalization gốc.
  - MedSAM checkpoint + z-score/domain-wise normalization.
- Ghi rõ preprocessing theo checkpoint trong config/log.

Mức độ tác động: **cần kiểm chứng**, nhưng có thể lớn nếu normalization mismatch.

### 13. SAM đang train theo no-prompt semantic segmentation

Trong [code/segment_anything/modeling/sam.py](../code/segment_anything/modeling/sam.py#L64)-[code/segment_anything/modeling/sam.py](../code/segment_anything/modeling/sam.py#L66), `forward_train()` gọi prompt encoder với `points=None, boxes=None, masks=None`. Nghĩa là SAM/MedSAM được dùng như một semantic segmentation model không prompt.

Đây có thể là chủ đích của paper/repo, nhưng nó làm mất một lợi thế lớn của SAM: promptable segmentation.

Hướng phát triển:

- Sinh box prompt từ labeled mask cho supervised branch.
- Sinh box/point prompt từ pseudo-label có confidence cao cho unlabeled branch.
- Thêm random perturbation vào box prompt để tăng robust.
- Nếu tiếp tục no-prompt, nên đổi tên mode rõ ràng thành semantic mode và dọn API `multimask_output`.

Mức độ tác động: **trung bình-cao theo hướng research**, không nhất thiết là bug.

### 14. Resize relative position khi load checkpoint hard-code theo ViT-B

Trong [code/segment_anything/build_sam.py](../code/segment_anything/build_sam.py#L153), `global_rel_pos_keys` được chọn bằng substring `'2'`, `'5'`, `'8'`, `'11'`. Cách này đúng với ViT-B default indexes `[2, 5, 8, 11]`, nhưng sai với ViT-L/H và có nguy cơ match nhầm key.

Khuyến nghị:

- Truyền `encoder_global_attn_indexes` vào `load_from()`.
- Parse key dạng `image_encoder.blocks.{idx}.attn.rel_pos_h/w`.
- Resize bất cứ key relative-position nào có shape mismatch, thay vì hard-code theo substring.

Mức độ tác động: **cao nếu đổi `--vit_name` hoặc `--img_size`**.

### 15. Một số overhead trong SAM có thể giảm tốc độ train

Các điểm runtime:

- `PromptEncoder.get_dense_pe()` tạo positional encoding mới mỗi forward tại [code/segment_anything/modeling/prompt_encoder.py](../code/segment_anything/modeling/prompt_encoder.py#L62), được gọi trong [code/segment_anything/modeling/sam.py](../code/segment_anything/modeling/sam.py#L69).
- `get_rel_pos()` tạo `torch.arange` không chỉ định device tại [code/segment_anything/modeling/image_encoder.py](../code/segment_anything/modeling/image_encoder.py#L319)-[code/segment_anything/modeling/image_encoder.py](../code/segment_anything/modeling/image_encoder.py#L320).
- `unet_stu_output_ulb_x_w` tại [code/train.py](../code/train.py#L537) chỉ dùng để tính confidence, nhưng không bọc `torch.no_grad()`.
- Confidence/Dice trong training loop chuyển qua CPU/NumPy tại [code/train.py](../code/train.py#L542)-[code/train.py](../code/train.py#L546).

Khuyến nghị:

- Cache dense positional encoding theo device/dtype.
- Tạo `torch.arange(..., device=rel_pos.device)`.
- Bọc các forward/metric chỉ dùng cho confidence bằng `torch.no_grad()`.
- Đưa Dice confidence sang torch GPU, tránh `np.asarray(...cpu())` trong mỗi iteration.

Mức độ tác động: **trung bình về tốc độ**, có thể giúp train nhanh và ổn định hơn với batch lớn.

### 16. Checkpointing chưa đủ để resume/reproduce

Trong [code/train.py](../code/train.py#L729) và [code/train.py](../code/train.py#L753), repo chỉ lưu best student model weights. Không lưu optimizer, scheduler, scaler AMP, EMA teacher, iteration, RNG state. Ngoài ra `--save_model` mặc định tắt tại [code/train.py](../code/train.py#L78), nên người dùng có thể train xong mà không có checkpoint.

Khuyến nghị:

- Lưu checkpoint dict đầy đủ:
  - `model_state_dict`
  - `ema_model_state_dict`
  - `optimizer_state_dict`
  - `scaler_state_dict`
  - `iter_num`, `epoch`, `best_metrics`
  - RNG states
- Thêm `--resume`.
- Mặc định nên save checkpoint best, hoặc cảnh báo rõ nếu `--save_model` tắt.

Mức độ tác động: **trung bình về vận hành**, **cao nếu cần reproduce nghiên cứu**.

### 17. Những điểm maintainability nên dọn

- `icecream` import nhiều nơi nhưng không dùng: [code/segment_anything/modeling/sam.py](../code/segment_anything/modeling/sam.py#L10), [code/segment_anything/build_sam.py](../code/segment_anything/build_sam.py#L9), [code/sam_lora_image_encoder.py](../code/sam_lora_image_encoder.py#L14). Nên bỏ để giảm dependency.
- LoRA load checkpoint dùng `torch.load(filename)` không `map_location` tại [code/sam_lora_image_encoder.py](../code/sam_lora_image_encoder.py#L152), rồi gán `Parameter(saved_tensor)` tại [code/sam_lora_image_encoder.py](../code/sam_lora_image_encoder.py#L157). Nên load theo device của model và dùng `load_state_dict`/copy data có check missing keys.
- `MaskDecoder.forward()` bỏ qua `multimask_output` tại [code/segment_anything/modeling/mask_decoder.py](../code/segment_anything/modeling/mask_decoder.py#L102)-[code/segment_anything/modeling/mask_decoder.py](../code/segment_anything/modeling/mask_decoder.py#L111). Nếu dùng semantic class logits thì nên document/rename, nếu dùng SAM predictor thì nên khôi phục behavior.
- DSBN trong [code/networks/dsbn.py](../code/networks/dsbn.py#L26) chỉ lấy `domain_label[0]` cho cả batch. Nếu batch có nhiều domain, BN sai. Nên assert batch cùng domain hoặc split theo domain.
- WRN pooling trong [code/networks/wrn.py](../code/networks/wrn.py#L95) giả định feature map vuông. Nên dùng `F.adaptive_avg_pool2d(out, 1)`.

## Lộ trình tối ưu để phát triển dự án

### Giai đoạn 1: Sửa bug/metric để kết quả đáng tin

1. Sửa `DiceLossWithMask` confidence mask.
2. Sửa masked CE normalization.
3. Sửa `test.py` checkpoint path/model alias.
4. Sửa HD/ASD empty-mask handling.
5. Sửa Dice smoothing policy.
6. Sửa `model.parameters()` -> `SAM_model.parameters()` trong nhánh SGD.

Kỳ vọng: metric ổn định hơn, tránh đánh giá nhầm, unsupervised signal đúng hơn.

### Giai đoạn 2: Ổn định training

1. Đặt EMA teacher `eval()` khi tạo pseudo-label.
2. Thêm LR scheduler thật cho SAM và U-Net.
3. Log LR, mask ratio, supervised/unsupervised loss normalized.
4. Lưu split labeled bằng seed.
5. Lưu checkpoint dict đầy đủ và thêm resume.

Kỳ vọng: dễ reproduce, dễ debug, giảm dao động giữa seed/domain.

### Giai đoạn 3: Tối ưu tốc độ

1. Chuyển confidence Dice sang torch GPU.
2. Bọc các forward chỉ dùng để tính confidence bằng `no_grad()`.
3. Cache dense PE và relative-position indices trong SAM.
4. Tăng `num_workers`/`persistent_workers` nếu IO là bottleneck.
5. Bỏ import/dependency không dùng.

Kỳ vọng: train nhanh hơn, GPU ít idle hơn.

### Giai đoạn 4: Phát triển research/model

1. Thử prompt từ box/point lấy từ label và pseudo-label.
2. Thử uncertainty calibration thay vì threshold cố định 0.95.
3. Domain-aware sampling cho unlabeled data, tránh domain lớn lấn át.
4. Thêm test-time augmentation hoặc connected-component post-processing theo dataset.
5. Đánh giá EMA teacher như candidate checkpoint riêng.

Kỳ vọng: cải thiện khả năng tổng quát trong MiDSS/domain shift, đặc biệt khi labeled domain cách xa unlabeled domains.

## Các thí nghiệm nên chạy để xác nhận

Nên chạy ablation nhỏ trước khi đổi nhiều thứ cùng lúc:

| Thí nghiệm | Thay đổi | Metric cần xem |
| --- | --- | --- |
| Baseline | Code hiện tại, cùng seed | Dice từng domain, avg Dice, mask ratio |
| Loss-mask fix | Sửa Dice mask + masked CE | Dice, unsup loss scale, mask ratio |
| EMA eval | Teacher eval khi pseudo-label | Dice, self/mutual confidence |
| Scheduler | Warmup + cosine/poly decay | Loss curve, best Dice iter |
| Random split | Shuffle/stratified labeled split | Mean/std qua nhiều seed |
| SAM prompt | Box/point prompt supervised/unlabeled | SAM Dice, ensemble pseudo-label Dice |

Khi báo cáo, nên ghi rõ:

- Dataset, `lb_domain`, `lb_num`, seed.
- Checkpoint SAM/MedSAM và preprocessing.
- `save_name`, commit hash, command train/test.
- Best avg Dice và Dice từng class/domain.
- Có dùng EMA checkpoint hay student checkpoint.

## Kết luận

Dự án này có ý tưởng tốt: kết hợp conventional U-Net với foundation model SAM/MedSAM để hỗ trợ semi-supervised segmentation dưới domain shift. Tuy nhiên, hiện có một số điểm cài đặt có khả năng làm giảm kết quả hoặc làm metric không đáng tin, đặc biệt là **mask trong unsupervised Dice**, **normalization của masked CE**, **evaluation checkpoint path**, **empty-mask metric**, **EMA teacher mode**, và **split labeled data**.

Nếu chỉ có thời gian sửa ít, nên ưu tiên Giai đoạn 1 và Giai đoạn 2. Đây là nhóm thay đổi ít mang tính kiến trúc nhất, nhưng có khả năng tăng độ tin cậy và hiệu năng thực nghiệm rõ nhất.
