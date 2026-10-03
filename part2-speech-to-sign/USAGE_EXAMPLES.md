# Part 2 — Usage Examples

## 1. Dùng API qua `curl` — text → VSL → avatar

```bash
cd "$HOME/Documents"
./venv/bin/python -m uvicorn --app-dir part2-speech-to-sign backend.app:app --port 8001 &
sleep 2

curl -s http://localhost:8001/api/v1/translate-text \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hôm qua tôi mua xe đạp","language":"vi"}'
```

```json
{
  "original_text": "Hôm qua tôi mua xe đạp",
  "vsl_structure": {
    "topic": "tôi",
    "comment": "mua xe đạp hôm qua",
    "words": ["tôi", "mua", "xe đạp", "hôm qua"]
  },
  ...
}
```

## 2. Audio → text → avatar (Python)

```python
import base64, sys
sys.path.insert(0, ".venv/lib/python3.12/site-packages")  # sau khi pip install -e shared
from vslshared.template_reference import load_reference_set
R = load_reference_set()
print("Thư viện hoạt hình:", len(R), "ký hiệu")
```

## 3. Chạy pipeline trong code (không GUI)

```python
import sys; sys.path[:0] = ["part2-speech-to-sign", "part2-speech-to-sign/backend"]
from frontend.pipeline import process_text
res = process_text("Tôi không thích")           # dict với vsl_structure + sequence
for step in res["animation_sequence"]:
    print(step["word"], step["duration_ms"], step["parameters"]["frame_count"])
```

> Output mong đợi: 3 ký hiệu (`tôi`, `thích`, `không`) — phủ định được đặt
> sau động từ.

## 4. Tinh chỉnh với LLM local (Ollama)

```bash
ollama pull qwen2.5:3b
cd part2-speech-to-sign && VSL_LLM_ENABLED=1 VSL_OLLAMA_MODEL=qwen2.5:3b \
  ../../venv/bin/python -m frontend.main
```

Nếu Ollama không chạy/thiếu model → log cảnh báo và fallback về grammar quy tắc
(app vẫn chạy bình thường).

## 5. Tổ hợp phím trong giao diện

| phím | tác dụng |
|------|----------|
| `⏺ Ghi` / `⏸ Dừng` | thu/nhận audio |
| `Enter` | dịch text nhập tay |
| `Space` | play/pause avatar |
| `+` / `-` | tăng/giảm tốc độ phát |
| `R` | replay clip |
| `→` | clip kế tiếp |