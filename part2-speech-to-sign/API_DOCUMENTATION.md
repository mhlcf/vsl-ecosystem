# Part 2 — API Documentation

Base URL: `http://localhost:8001`. Format JSON. Lỗi trả về
`{"status":"error","detail":"..."}` với HTTP status phù hợp.

Khởi động server:

```bash
cd "$HOME/Documents/part2-speech-to-sign"
"$HOME/Documents/venv/bin/python" -m backend.app        # mặc định port 8001
```

OpenAPI (swagger) tự sinh tại `http://localhost:8001/docs`.

---

## `GET /health`

Trạng thái server + thông tin môi trường.

```json
{"status": "ok", "model": "…whisper size…", "llm_enabled": false}
```

## `POST /api/v1/process-speech`

Nhận audio base64, transcribe bằng Whisper, dịch sang cấu trúc VSL và sinh
animation sequence.

**Request**

| field | type | mô tả |
|-------|------|-------|
| `audio_file` | string | base64 (WAV/MP3/OGG/…) |
| `language` | string | `vi` (mặc định) |
| `duration_ms` | int | 100–60000, mặc định 3000 |

**Response** (`ProcessSpeechResponse`)

- `original_text` — chuỗi đã nhận dạng
- `vsl_structure.{topic, comment, words[]}` — cấu trúc topic-comment VSL
- `animation_sequence[]` — mỗi bước có `word`, `animation_id`, `duration_ms`,
  `parameters.{template_key, frames, fps, frame_count}` (frames kèm khi ở chế độ
  nội tuyến; `frame_count>0` → frontend tự sinh frame keypoint)
- `controls`, `confidence`, `language`

**Ví dụ** — ghi tiếng Việt: *"Tôi không thích cà phê"*

```bash
curl -s http://localhost:8001/api/v1/process-speech \
  -H 'Content-Type: application/json' \
  -d '{"audio_file":"BASE64_WAV","duration_ms":3000}'
```

```json
{
  "original_text": "Tôi không thích cà phê",
  "vsl_structure": {"topic": "tôi", "comment": "không thích cà phê", "words": ["tôi", "thích", "không", "cà phê"]},
  "animation_sequence": [
    {"word": "tôi", "animation_id": "anim_xxxxx", "duration_ms": 800, "parameters": {"template_key": "reference://tôi", "fps": 20, "frame_count": 16}},
    {"word": "thích", "animation_id": "anim_xxxxy", "duration_ms": 800, "parameters": {"template_key": "reference://thích", "fps": 20, "frame_count": 16}}
  ],
  "controls": {"playback_speed": 1.0},
  "confidence": 0.91,
  "language": "vi"
}
```

> Khi không có model Whisper (lỗi `STTUnavailableError`), server trả
> `{"status":"error","detail":"...STT model not available..."}` — dùng
> `translate-text` để tiếp tục.

## `POST /api/v1/translate-text`

Không cần audio — nhận text nhập tay, chạy grammar translator + avatar
controller, trả về đúng cấu trúc của `process-speech`.

**Request**: `{"text": "Tôi ăn cơm", "language": "vi"}`

## `GET /api/v1/signs/{q}`

Tìm ký hiệu theo từ khóa trong `vocabulary.json` (fuzzy, không dấu).

**Response**: `{"query":"cà","matches":[{"sign_name":"Cà phê","animation_id":"anim_xxx"}]}`

---

## Lỗi phổ biến

| HTTP | meaning |
|------|---------|
| 400 | thiếu/base64 sai `audio_file` hoặc `duration_ms` ngoài 100–60000 |
| 422 | payload không khớp schema |
| 500 | lỗi pipeline (grammar/avatar); `detail` ghi rõ điểm lỗi |

## Độ an toàn

- Không gửi audio/text lên cloud (Whisper & LLM đều local).
- `GET /signs/{q}` URL-encode từ khóa (VD: `%20` cho dấu cách).