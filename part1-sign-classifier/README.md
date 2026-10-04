# Phần 1 — Bộ phân loại ký hiệu lẻ (3315 từ)

Model phân loại **một ký hiệu lẻ** từ chuỗi keypoint MediaPipe (201 chiều,
60 frame) thành một trong **3315 từ** của bộ từ vựng VSL. Đây là nền tảng
dữ liệu + model dùng chung cho Part 2 (avatar) và Part 3 (chấm điểm).

```
Webcam ──► MediaPipe Holistic ──► 201-dim/keypoint ──► buffer 60 frame
                                                        │
                                              normalize (scaler.npz)
                                                        │
                                              BiLSTM 3315 lớp ──► top-3 từ
                                                        │
                                        smoothing (majority vote 5 frame)
                                                        │
                              từ ổn định ≥8 frame ──► WordBuffer (tích lũy)
                                                        │
                                đủ 3 từ (auto) / phím 's' ──► LLM API
                                                        │
                                     "Anh thích cà phê." (fallback: ghép cơ bản)
```

## Cấu trúc thư mục

```
part1-sign-classifier/
├── model.py                 # VSLModel — BiLSTM 3 lớp (hidden 384, bidir)
├── dataset.py               # VSLDataset + create_dataloaders (+ augment)
├── train.py                 # Training loop (150 epoch, early stopping)
├── real_time_predict.py     # Nhận diện realtime + ghép câu
├── sentence_builder.py      # WordBuffer + SentenceComposer (LLM API/fallback)
├── .env.example             # Cấu hình LLM (copy thành .env, điền key)
├── tests/                   # Test sentence_builder (pytest)
├── best_model.pth           # Trọng số đã huấn luyện (~44 MB)
├── scaler.npz               # mean/std chuẩn hóa (201 chiều)
├── label_map.json           # 3315 từ → chỉ số lớp
├── train/ val/ test/        # ~145k mẫu NPZ (mỗi mẫu: sequence (60,201))
├── checkpoint/              # Checkpoint dở dang (last.pth)
└── logs/                    # training_curves.png
```

## Chạy

```bash
cd "$HOME/Documents"

# nhận diện realtime + ghép câu — phím: 's' ghép · 'c' xóa · 'u' xóa từ cuối · 'q' thoát
./run_part1.sh

# huấn luyện lại từ đầu
"$HOME/Documents/venv/bin/python" part1-sign-classifier/train.py
```

## Ghép câu thành tiếng Việt (LLM API)

Từ đã nhận diện được tích lũy vào buffer (chỉ khi có bàn tay, độ tin cậy ≥ 0.2,
ổn định ≥ 8 frame, bỏ gloss `Anh (nước Anh)` → `Anh`). Khi buffer **đủ 3 từ sẽ
tự ghép câu**, hoặc nhấn **`s`** để ghép bất cứ lúc nào.

| Phím | Chức năng |
|------|-----------|
| `s` | Ghép câu (thread nền, không freeze camera) |
| `c` | Xóa buffer + câu |
| `u` | Xóa từ cuối |
| `q` | Thoát |

**Cấu hình LLM** (tùy chọn, endpoint kiểu OpenAI — OpenAI/Groq/DeepSeek/OpenRouter):

```bash
cp part1-sign-classifier/.env.example part1-sign-classifier/.env
# điền VSL_LLM_API_KEY (file .env nằm trong .gitignore, không commit key)
```

| Env | Mặc định | Ý nghĩa |
|-----|----------|---------|
| `VSL_LLM_ENABLED` | `1` | `0` = tắt LLM, luôn ghép cơ bản |
| `VSL_LLM_BASE_URL` | OpenAI chat completions | URL tương thích OpenAI |
| `VSL_LLM_API_KEY` | (trống) | Key của bạn; chưa điền → fallback |
| `VSL_LLM_MODEL` | `gpt-4o-mini` | Tên model theo provider |

Không có key / API lỗi / offline → **fallback ghép cơ bản**
(`"Anh thích cà phê."`), hiển thị nhãn `[co ban]` trên khung hình;
có key thành công hiện `[LLM]`.

## Chi tiết model

| Hạng mục  | Giá trị                                              |
| --------- | ---------------------------------------------------- |
| Kiến trúc | BiLSTM 3 lớp, hidden 384, bidirectional, dropout 0.4 |
| Input     | `(60, 201)` — pose 75 + tay trái 63 + tay phải 63    |
| Output    | 3315 lớp (khớp `vsl_data/vocabulary.json`)           |
| Loss      | CrossEntropy + label_smoothing 0.2                   |
| Optimizer | AdamW + ReduceLROnPlateau (patience 20)              |
| Kết quả   | Val accuracy **1.0000** (xem `training_output.log`)  |

## Dùng từ code khác

```python
from shared.vslshared.classifier import create_classifier

clf = create_classifier()          # ưu tiên model, fallback template
pred = clf.predict(sequence)       # sequence: np.ndarray (60, 201)
```

> Hai backend cùng interface (`ModelBackend` = BiLSTM, `TemplateBackend` =
> nearest-template) — xem `shared/vslshared/classifier.py`.
> Schema 201-dim là **nguồn duy nhất** tại `shared/vslshared/keypoints.py`.
