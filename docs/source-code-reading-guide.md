# Hướng dẫn đọc mã nguồn SynFoC từ đầu đến cuối

Tài liệu này là lộ trình đọc source code cho repo SynFoC. Mục tiêu là giúp bạn hiểu project theo đúng luồng chạy thực tế: dữ liệu đi vào như thế nào, hai mô hình U-Net và SAM/MedSAM được tạo ra ra sao, pseudo-label được sinh và dùng để train thế nào, cuối cùng metric và checkpoint được đánh giá ở đâu.

Nên đọc tài liệu này cùng với:

- [`README.md`](../README.md): mô tả ngắn về paper, dataset, train/test.
- [`docs/run-datasets.md`](run-datasets.md): cách chạy từng dataset.
- [`docs/source-code-performance-audit.md`](source-code-performance-audit.md): các điểm code có thể làm giảm hiệu năng.
- [`2503.16997v1.pdf`](../2503.16997v1.pdf): paper gốc để đối chiếu ý tưởng SynFoC/MiDSS.

## 1. Bức tranh tổng thể

Repo này xoay quanh bài toán **Mixed Domain Semi-Supervised Medical Image Segmentation**:

- Dữ liệu có nhãn chỉ đến từ một domain nguồn.
- Dữ liệu chưa gán nhãn đến từ nhiều domain trộn lẫn.
- Project train đồng thời hai mô hình:
  - **U-Net**: mô hình segmentation truyền thống, dễ overfit domain có nhãn nhưng học nhanh.
  - **SAM/MedSAM + LoRA**: foundation model có đặc trưng mạnh hơn, được fine-tune nhẹ bằng LoRA và mask decoder.

Luồng chính:

```text
Dataset folders
  -> dataloaders/dataloader.py
  -> dataloaders/custom_transforms.py
  -> train.py
       -> tạo U-Net student + U-Net EMA teacher
       -> tạo SAM/MedSAM student + SAM/MedSAM EMA teacher
       -> EMA teacher sinh pseudo-label trên unlabeled images
       -> trộn pseudo-label theo self-confidence và mutual-confidence
       -> train hai student bằng supervised + unsupervised + consistency losses
       -> đánh giá định kỳ bằng test() trong train.py
       -> lưu best checkpoints nếu bật --save_model
  -> test.py
       -> load checkpoint
       -> đánh giá từng domain
  -> summarize_logs.py
       -> tổng hợp log.txt thành json/csv/markdown
```

## 2. Thứ tự đọc khuyến nghị

Không nên bắt đầu bằng thư mục `segment_anything/`, vì đó là phần nặng nhất và có nhiều code được kế thừa từ SAM. Cách đọc dễ hiểu hơn là đi từ script entry point, sau đó lần ngược xuống dataset, loss, model.

Thứ tự nên đọc:

1. [`README.md`](../README.md)
2. [`docs/run-datasets.md`](run-datasets.md)
3. [`code/train.py`](../code/train.py)
4. [`code/dataloaders/dataloader.py`](../code/dataloaders/dataloader.py)
5. [`code/dataloaders/custom_transforms.py`](../code/dataloaders/custom_transforms.py)
6. [`code/utils/losses.py`](../code/utils/losses.py)
7. [`code/utils/metrics.py`](../code/utils/metrics.py)
8. [`code/networks/unet_model.py`](../code/networks/unet_model.py) và [`code/networks/unet_parts.py`](../code/networks/unet_parts.py)
9. [`code/sam_lora_image_encoder.py`](../code/sam_lora_image_encoder.py)
10. [`code/segment_anything/build_sam.py`](../code/segment_anything/build_sam.py)
11. [`code/segment_anything/modeling/sam.py`](../code/segment_anything/modeling/sam.py)
12. [`code/test.py`](../code/test.py)
13. [`code/summarize_logs.py`](../code/summarize_logs.py)

Các file trong `model/<dataset>/train/<experiment>/train.py` là bản snapshot được copy từ lúc chạy thí nghiệm. Chúng hữu ích để tái hiện experiment cũ, nhưng không nên xem là source chính khi tìm hiểu project.

## 3. Đọc `train.py`: entry point quan trọng nhất

File [`code/train.py`](../code/train.py) là nơi nên dành nhiều thời gian nhất.

### 3.1 Đọc phần CLI arguments

Bắt đầu từ nhóm `parser.add_argument(...)` ở đầu file. Các tham số quan trọng:

| Tham số | Ý nghĩa |
| --- | --- |
| `--dataset` | Chọn dataset: `fundus`, `prostate`, `MNMS`, `BUSI` |
| `--save_name` | Tên experiment |
| `--model` | Chọn checkpoint nền: `SAM` hoặc `MedSAM` |
| `--lb_domain` | Domain có nhãn |
| `--lb_num` | Số mẫu có nhãn lấy từ domain đó |
| `--threshold` | Ngưỡng confidence để giữ pseudo-label |
| `--rank` | Rank của LoRA |
| `--AdamW` | Dùng AdamW cho SAM/MedSAM |
| `--warmup` | Bật warmup learning rate cho SAM/MedSAM |
| `--save_model` | Lưu best checkpoint |

Sau khi đọc phần này, bạn nên tự trả lời được: khi chạy một lệnh train, cấu hình nào quyết định dataset, số domain, số label, checkpoint SAM/MedSAM và output folder?

### 3.2 Đọc block `if __name__ == "__main__"`

Tiếp theo đọc từ cuối file lên. Đây là nơi `train.py` map từng dataset sang cấu hình cụ thể:

| Dataset | `train_data_path` | `num_channels` | `patch_size` | `num_classes` | `part` |
| --- | --- | ---: | ---: | ---: | --- |
| `fundus` | `../data/Fundus` | 3 | 256 | 2 | `cup`, `disc` |
| `prostate` | `../data/ProstateSlice` | 1 | 384 | 1 | `base` |
| `MNMS` | `../data/mnms` | 1 | 288 | 3 | `lv`, `myo`, `rv` |
| `BUSI` | `../data/Dataset_BUSI_with_GT` | 1 | 256 | 1 | `base` |

Ở đoạn này cũng có các chi tiết quan trọng:

- `snapshot_path` được tạo theo dạng `../model/<dataset>/train/<save_name>/`.
- `args.max_iterations` bị ghi đè theo dataset.
- `args.ckpt` được chọn từ `SAM` hoặc `MedSAM`.
- Seed và CUDA deterministic được set nếu `--deterministic`.
- Script hiện tại được copy vào experiment folder để lưu lại code lúc chạy.

Sau đoạn này, bạn sẽ hiểu vì sao cùng một `--save_name` nhưng train và test có thể cần đường dẫn khác nhau.

### 3.3 Đọc các helper trước vòng train

Các hàm nên đọc nhanh:

- `get_current_consistency_weight(...)`: tính hệ số ramp-up cho loss consistency.
- `update_Unet_ema_variables(...)`: cập nhật EMA teacher của U-Net.
- `update_SAM_ema_variables(...)`: cập nhật EMA cho LoRA layers, prompt encoder và mask decoder của SAM.
- `cycle(...)`: tạo iterator vô hạn cho DataLoader.
- `to_2d(...)` và `to_3d(...)`: đổi label index thành nhiều kênh foreground để tính Dice cho Fundus/MNMS.
- `obtain_cutmix_box(...)`: tạo vùng CutMix.
- `statistics`: class nhỏ để ghi trung bình các chỉ số training.

### 3.4 Đọc `train(args, snapshot_path)`

Đây là lõi của project. Nên đọc theo 5 lớp thay vì đọc tuần tự một mạch.

Lớp 1: tạo model.

- `create_model('SAM')` gọi `sam_model_registry[...]`, sau đó bọc bằng `LoRA_Sam`.
- `create_model('unet')` tạo U-Net với `n_classes = num_classes + 1`, tức có cả background.
- Mỗi mô hình có một bản student và một bản EMA teacher.

Lớp 2: tạo transform và dataset.

- `weak`: scale/crop, rotate, horizontal flip, elastic transform.
- `strong`: brightness, contrast, gaussian blur.
- `normal_toTensor`: normalize về `[0, 1]`, tạo thêm `low_res_label`, `unet_size_img`, `unet_size_label`.
- `lb_dataset`: chỉ lấy domain có nhãn `lb_domain`.
- `ulb_dataset`: lấy dữ liệu chưa gán nhãn từ tất cả domain.
- `test_dataset`: tạo DataLoader riêng cho từng domain để đánh giá.

Lớp 3: chuẩn hóa mask theo dataset.

Trong vòng lặp train, label gốc được đổi thành class index:

- Fundus: mask cup/disc được map thành 2 lớp foreground.
- Prostate: foreground được lấy bằng `mask.eq(0)`.
- MNMS: giữ class index `1`, `2`, `3`.
- BUSI: foreground được lấy bằng `mask.eq(255)`.

Đây là đoạn rất quan trọng vì nếu mapping mask sai, mọi loss và metric phía sau đều sai.

Lớp 4: sinh pseudo-label.

EMA teacher tạo prediction trên ảnh weak unlabeled:

- `ema_SAM_model(ulb_x_w, ...)` sinh logits low-resolution của SAM.
- `ema_unet_model(ulb_unet_size_x_w)` sinh logits ở kích thước U-Net.
- Lấy `softmax`, `max probability`, `argmax label`.
- Tính `self_conf`: độ ổn định giữa U-Net student và U-Net teacher.
- Tính `mutual_conf`: độ đồng thuận giữa U-Net teacher và SAM teacher.
- Tạo `ratio = self_conf * mutual_conf`.
- Trộn xác suất SAM và U-Net:

```python
unet_size_prob_ulb_x_w = (1 - ratio) * unet_size_sam_prob_ulb_x_w + ratio * unet_prob_ulb_x_w
```

Ý nghĩa: nếu U-Net tự ổn định và đồng thuận tốt với SAM, U-Net có trọng số cao hơn; nếu không, SAM chi phối nhiều hơn.

Lớp 5: tính loss và cập nhật.

Các loss chính:

- `sam_sup_loss`: CE + Dice trên labeled data cho SAM.
- `unet_sup_loss`: CE + Dice trên labeled data cho U-Net.
- `sam_unsup_loss`: CE + Dice trên pseudo-label cho SAM.
- `unet_unsup_loss`: CE + Dice trên pseudo-label cho U-Net.
- `cons_loss`: entropy-style loss ở vùng hai mô hình đồng thuận.
- `discons_loss`: MSE ở vùng hai mô hình bất đồng.

Loss tổng:

```python
loss = sam_sup_loss + unet_sup_loss + consistency_weight * (
    sam_unsup_loss + unet_unsup_loss + cons_loss + discons_loss
)
```

Sau đó code:

- backward bằng AMP nếu `--amp`;
- step optimizer của SAM và U-Net;
- cập nhật EMA teacher;
- log loss, confidence, mask ratio, pseudo-label Dice;
- mỗi `num_eval_iter` iteration thì đánh giá U-Net và SAM.

## 4. Đọc dataloader: dữ liệu thật sự vào model như thế nào

File [`code/dataloaders/dataloader.py`](../code/dataloaders/dataloader.py) định nghĩa các class dataset.

Các class đang dùng trong `train.py` và `test.py`:

- `FundusSegmentation`
- `ProstateSegmentation`
- `MNMSSegmentation`
- `BUSISegmentation`

Các class khác như `ACDCSegmentation`, `MSCMRSegSegmentation` có trong file nhưng không được chọn bởi CLI hiện tại.

Khi đọc mỗi dataset class, hãy tìm 4 điểm:

1. `domain_name`: domain id map sang folder nào.
2. Cách tạo `imagelist`: dùng `glob`, `sort`, folder `train/test`.
3. Cách chọn `selected_idxs`: dùng để tách labeled và unlabeled.
4. `__getitem__`: ảnh/mask được resize, convert channel, transform và trả về sample ra sao.

Sample cuối cùng thường có các key:

| Key | Ý nghĩa |
| --- | --- |
| `image` | ảnh ở kích thước SAM/MedSAM input |
| `label` | mask ở kích thước gốc/đã resize theo dataset |
| `low_res_label` | mask ở kích thước low-res cho SAM decoder |
| `unet_size_img` | ảnh resize về `patch_size` cho U-Net |
| `unet_size_label` | mask resize về `patch_size` cho U-Net |
| `strong_aug` | ảnh unlabeled sau strong augmentation |
| `unet_size_strong_aug` | bản strong augmentation resize cho U-Net |
| `img_name` | tên ảnh |
| `dc` | domain code |

Điểm cần chú ý: trong train, labeled split hiện lấy `first N` sau khi sort filename. Nếu bạn đang nghiên cứu độ tin cậy thực nghiệm, hãy đọc kỹ đoạn `selected_idxs` trong từng dataset class.

## 5. Đọc transform: augmentation và tensor shape

File [`code/dataloaders/custom_transforms.py`](../code/dataloaders/custom_transforms.py) chứa augmentation và bước đổi PIL/NumPy sang Tensor.

Các transform chính trong train:

- `RandomScaleCrop`: phóng to ngẫu nhiên rồi crop lại.
- `RandomScaleRotate`: xoay ảnh/mask.
- `RandomHorizontalFlip`: lật ngang.
- `elastic_transform`: biến dạng đàn hồi.
- `Brightness`, `Contrast`, `GaussianBlur`: strong augmentation cho unlabeled data.
- `Normalize_tf`: normalize pixel về `[0, 1]` trong cấu hình hiện tại.
- `ToTensor`: tạo tensor chính và các bản resize cho SAM/U-Net.

Nên đọc kỹ `ToTensor`, vì đây là nơi tạo nhiều view của cùng một sample:

```text
image              -> dùng cho SAM/MedSAM
low_res_label      -> dùng cho SAM low_res_logits
unet_size_img      -> dùng cho U-Net
unet_size_label    -> dùng cho U-Net loss/metric
strong_aug         -> dùng cho SAM trên unlabeled strong image
unet_size_strong_aug -> dùng cho U-Net trên unlabeled strong image
```

Khi debug lỗi shape, file này thường là nơi nên kiểm tra đầu tiên.

## 6. Đọc loss và metric

### 6.1 Loss

File [`code/utils/losses.py`](../code/utils/losses.py) chứa nhiều loss, nhưng trong `train.py` phần quan trọng nhất là:

- `DiceLossWithMask`
- `CrossEntropyLoss(reduction='none')` từ PyTorch

`DiceLossWithMask` hỗ trợ:

- supervised Dice khi `mask=None`;
- unsupervised Dice khi có confidence mask;
- `softmax=True` cho segmentation nhiều class.

Khi đọc, hãy chú ý shape:

```text
inputs: B, C, H, W
target: B, 1, H, W hoặc B, H, W
mask:   B, 1, H, W
```

### 6.2 Metric

File [`code/utils/metrics.py`](../code/utils/metrics.py) chứa Dice helper:

- `dice_coeff`: binary foreground, dùng cho Prostate/BUSI.
- `dice_coeff_2label`: hai foreground layer, dùng cho Fundus.
- `dice_coeff_3label`: ba foreground class, dùng cho MNMS.

HD95, ASD, DC, JC được tính trực tiếp trong `train.py::test(...)` và `test.py::test(...)` bằng `medpy.metric.binary`.

Khi đọc metric, luôn kiểm tra policy với mask rỗng. Đây là điểm ảnh hưởng lớn tới kết quả trên dataset có slice không chứa object.

## 7. Đọc U-Net

U-Net nằm trong:

- [`code/networks/unet_model.py`](../code/networks/unet_model.py)
- [`code/networks/unet_parts.py`](../code/networks/unet_parts.py)

Cấu trúc khá chuẩn:

```text
DoubleConv
  -> Down x4
  -> Up x4
  -> OutConv
```

Điều quan trọng nhất khi đọc U-Net trong repo này không phải kiến trúc, mà là interface:

```python
model = UNet(n_channels=num_channels, n_classes=num_classes + 1)
output = model(unet_size_img)
```

Output luôn có channel background + foreground classes. Vì vậy:

- Prostate/BUSI: `num_classes=1`, output có 2 channel.
- Fundus: `num_classes=2`, output có 3 channel.
- MNMS: `num_classes=3`, output có 4 channel.

## 8. Đọc SAM/MedSAM và LoRA

### 8.1 Registry và build model

Bắt đầu từ [`code/segment_anything/build_sam.py`](../code/segment_anything/build_sam.py).

Các hàm:

- `build_sam_vit_b`
- `build_sam_vit_l`
- `build_sam_vit_h`
- `_build_sam`
- `load_from`

Trong project hiện tại, CLI default dùng `vit_b`. `_build_sam` tạo:

- `ImageEncoderViT`
- `PromptEncoder`
- `MaskDecoder`
- `Sam`

Sau đó load checkpoint SAM/MedSAM. Nếu checkpoint không khớp shape, `load_from` resize positional embedding.

### 8.2 SAM forward

Đọc [`code/segment_anything/modeling/sam.py`](../code/segment_anything/modeling/sam.py).

Trong training, code gọi:

```python
model(data, multimask_output, args.img_size)
```

Vì input là tensor, `Sam.forward(...)` đi vào `forward_train(...)`, không phải `forward_test(...)`.

`forward_train(...)` làm các bước:

```text
preprocess image
  -> image_encoder
  -> prompt_encoder(points=None, boxes=None, masks=None)
  -> mask_decoder
  -> postprocess_masks
  -> trả về masks, iou_predictions, low_res_logits
```

Điểm rất quan trọng: trong train path, SAM/MedSAM đang chạy **không có prompt**. Nó được dùng như một semantic segmentation model thay vì promptable interactive segmentation model.

### 8.3 LoRA wrapper

Đọc [`code/sam_lora_image_encoder.py`](../code/sam_lora_image_encoder.py).

Các phần chính:

- `_LoRA_qkv`: thay đổi projection `qkv` trong attention block.
- `LoRA_Sam`: freeze image encoder gốc, chèn LoRA vào các block selected.
- `save_lora_parameters`: chỉ lưu LoRA + prompt encoder + mask decoder.
- `load_lora_parameters`: load lại các thành phần đó.

Khi đọc LoRA, hãy nhớ:

- image encoder gốc bị freeze;
- LoRA chỉ thêm cập nhật low-rank vào Q và V;
- prompt encoder và mask decoder vẫn được lưu/load trong checkpoint LoRA;
- `forward(...)` chỉ gọi tiếp sang `self.sam(...)`.

## 9. Đọc evaluation bằng `test.py`

File [`code/test.py`](../code/test.py) gần giống phần `test(...)` trong `train.py`, nhưng dùng để chạy riêng sau khi đã có checkpoint.

Luồng đọc:

1. CLI args: `--dataset`, `--save_name`, `--model`, `--gpu`, `--rank`.
2. `main(...)`: tạo model SAM trước để lấy `img_embedding_size`, tạo transform và test dataloader.
3. Nếu `--model unet`: tạo U-Net và load `unet_avg_dice_best_model.pth`.
4. Nếu `--model SAM`: tạo LoRA-SAM và load `SAM_avg_dice_best_model.pth`.
5. `test(...)`: chạy từng domain, tính Dice/DC/JC/HD/ASD.

Gotcha cần nhớ: `train.py` lưu vào:

```text
../model/<dataset>/train/<save_name>/
```

Trong khi `test.py` load từ:

```text
../model/<dataset>/<save_name>/
```

Vì vậy nếu train bằng:

```bash
python train.py --dataset prostate --save_name prostate_dm1_lb40_full --save_model
```

thì test thường cần:

```bash
python test.py --dataset prostate --save_name train/prostate_dm1_lb40_full --model unet
```

## 10. Đọc log summarizer

File [`code/summarize_logs.py`](../code/summarize_logs.py) độc lập với training. Nó đọc các file `log.txt` trong `model/`, parse command, namespace args và metric cuối cùng.

Nên đọc theo thứ tự:

1. Regex constants ở đầu file.
2. `parse_namespace`, `parse_command`, `parse_metric_fields`.
3. `summarize_log`: state machine nhỏ để biết đang đọc metric của U-Net hay SAM.
4. `write_csv`, `write_json`, `write_markdown`.
5. `main`.

File này có unit test riêng ở [`code/tests/test_summarize_logs.py`](../code/tests/test_summarize_logs.py), phù hợp để học cách repo hiện viết test.

## 11. Cách lần theo một batch dữ liệu

Nếu muốn hiểu sâu nhất, hãy tự trace một batch unlabeled:

```text
dataloader.py::__getitem__
  -> sample = {image, label, img_name, dc}
  -> weak transform
  -> strong transform
  -> Normalize_tf
  -> ToTensor
  -> train.py nhận ulb_sample
  -> ema_SAM_model sinh sam_pseudo_label
  -> ema_unet_model sinh unet_pseudo_label
  -> tính self_conf và mutual_conf
  -> trộn probability thành ensemble pseudo-label
  -> tạo confidence mask bằng threshold
  -> CutMix vùng labeled vào unlabeled strong image
  -> student SAM/U-Net học từ supervised + pseudo-label
```

Đây là tuyến đọc quan trọng nhất để hiểu contribution của SynFoC trong source code.

## 12. Checklist tự kiểm tra sau khi đọc

Sau khi đọc hết lộ trình trên, bạn nên tự trả lời được các câu hỏi này:

- Dataset nào được chọn bởi `--dataset`, và domain id map sang folder nào?
- Vì sao output channel của model là `num_classes + 1`?
- SAM/MedSAM dùng ảnh ở size nào, U-Net dùng ảnh ở size nào?
- `low_res_label` dùng cho phần nào của SAM?
- EMA teacher khác student ở đâu?
- `self_conf`, `mutual_conf`, `ratio` được tính từ prediction nào?
- Khi nào pixel pseudo-label bị loại bởi confidence threshold?
- CutMix đang trộn ảnh/mask labeled vào unlabeled ở kích thước nào?
- Best checkpoint được lưu theo điều kiện gì?
- `test.py` load checkpoint từ đường dẫn nào?
- Dice/DC/JC/HD/ASD được tính theo từng domain hay toàn bộ dataset?
- File nào nên sửa nếu muốn thêm dataset mới?
- File nào nên sửa nếu muốn thay đổi loss?
- File nào nên sửa nếu muốn thay đổi LoRA rank/layer hoặc cách load SAM?

## 13. Bản đồ sửa đổi theo mục tiêu

| Muốn làm gì | Đọc/sửa file nào trước |
| --- | --- |
| Chạy dataset mới | `dataloaders/dataloader.py`, block dataset config trong `train.py` và `test.py` |
| Đổi augmentation | `dataloaders/custom_transforms.py`, phần `weak`/`strong` trong `train.py` |
| Đổi pseudo-label strategy | vòng lặp chính trong `train.py` |
| Đổi confidence threshold | `train.py`, đoạn tạo `unet_size_mask` và `low_res_mask` |
| Đổi loss supervised/unsupervised | `train.py`, `utils/losses.py` |
| Đổi metric | `train.py::test`, `test.py::test`, `utils/metrics.py` |
| Đổi U-Net | `networks/unet_model.py`, `networks/unet_parts.py` |
| Đổi SAM/MedSAM checkpoint | `train.py`, `test.py`, `segment_anything/build_sam.py` |
| Đổi LoRA | `sam_lora_image_encoder.py` |
| Tổng hợp kết quả experiment | `summarize_logs.py` |

## 14. Lộ trình đọc nhanh trong 90 phút

Nếu chỉ có ít thời gian, đọc theo lịch này:

| Thời gian | Việc cần đọc |
| --- | --- |
| 0-10 phút | `README.md`, `docs/run-datasets.md` |
| 10-25 phút | cuối `train.py`: dataset config, checkpoint, output path |
| 25-45 phút | `train.py::train`: tạo model, tạo dataloader, vòng lặp pseudo-label/loss |
| 45-60 phút | `dataloader.py` và `custom_transforms.py::ToTensor` |
| 60-70 phút | `losses.py::DiceLossWithMask`, `metrics.py::dice_coeff*` |
| 70-80 phút | `sam_lora_image_encoder.py`, `segment_anything/modeling/sam.py::forward_train` |
| 80-90 phút | `test.py` và đường dẫn checkpoint |

## 15. Lộ trình đọc sâu trong 1-2 ngày

Ngày 1:

- Đọc paper để hiểu MiDSS và ý tưởng mutual aid.
- Đọc `train.py` từ dưới lên, ghi lại toàn bộ biến global theo dataset.
- Trace một batch labeled và một batch unlabeled.
- Vẽ lại graph loss: supervised, unsupervised, consistency, disagreement.

Ngày 2:

- Đọc từng dataset class và đối chiếu với `data_format/`.
- Đọc toàn bộ `custom_transforms.py`, đặc biệt các bước resize.
- Đọc SAM build/forward/LoRA.
- Chạy một lệnh train nhỏ hoặc syntax check nếu môi trường sẵn sàng.
- Đọc `docs/source-code-performance-audit.md` để biết các điểm cần cẩn trọng khi đánh giá kết quả.

## 16. Lệnh hỗ trợ khi đọc code

Chạy từ root repo:

```bash
rg -n "def train|def test|create_model|pseudo|cons_loss|discons_loss|save_lora|load_lora" code
rg -n "class .*Segmentation|selected_idxs|domain_name|__getitem__" code/dataloaders
rg -n "DiceLossWithMask|dice_coeff|hd95|asd|CrossEntropyLoss" code
```

Chạy từ thư mục `code/` để kiểm tra syntax:

```bash
python -m compileall train.py test.py dataloaders networks utils
```

Chạy unit test hiện có cho log summarizer:

```bash
python -m unittest tests/test_summarize_logs.py
```

## 17. Gợi ý cách ghi chú khi đọc

Nên tạo một bảng nhỏ cho từng experiment:

| Mục | Giá trị cần ghi |
| --- | --- |
| Dataset | `fundus/prostate/MNMS/BUSI` |
| Labeled domain | `lb_domain` |
| Số label | `lb_num` hoặc `lb_ratio` |
| SAM checkpoint | `SAM` hay `MedSAM` |
| `img_size` | input cho SAM |
| `patch_size` | input cho U-Net |
| `threshold` | ngưỡng pseudo-label |
| `rank` | LoRA rank |
| Best U-Net Dice | từ `log.txt` |
| Best SAM Dice | từ `log.txt` |

Cách ghi này giúp nối được ba lớp thông tin: command chạy, cấu hình code, và kết quả metric.

## 18. Kết luận

Để hiểu repo này, đừng đọc theo thứ tự folder. Hãy đọc theo luồng thực thi:

```text
train.py config
  -> dataset class
  -> transform
  -> model creation
  -> pseudo-label generation
  -> loss
  -> EMA update
  -> evaluation
  -> checkpoint/log summarization
```

Sau khi nắm được luồng này, phần `segment_anything/` sẽ dễ đọc hơn nhiều: bạn không cần hiểu toàn bộ SAM ngay lập tức, chỉ cần hiểu interface mà SynFoC đang dùng trong training path.
