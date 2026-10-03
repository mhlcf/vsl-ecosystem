# Part 2 — Architecture

## Tổng quan (component diagram)

```
┌──────────────────────────────────────────────────────────────┐
│  frontend/ (PyQt6)                                           │
│                                                              │
│  SpeechCapture ──(wav)──► recorder.MicrophoneRecorder        │
│        │                     │                               │
│        │<─ text w/ STT ──────┘   (hoặc gõ text trực tiếp)     │
│        │                                                     │
│  pipeline.process_audio / process_text                        │
│   │  │                                                       │
│   │  ├────────────────────────┐                               │
│   │  ▼                        ▼                              │
│  backend (FastAPI, in-process cũng được)                     │
│  speech_recognition   grammar_translator   avatar_controller  │
│   │ (faster-whisper)      │ rules + LLM opt  │ resolve_sign   │
│   │                       ▼                  ▼                │
│   └─────────────► VSLStructure ◄─────── vocabulary.json       │
│                                                     │         │
│                                   reference://sign ─┤         │
│                                                     ▼         │
│  avatar/ ──► KeypointAvatar (matplotlib 3D)  ◄── frames (30,201)│
│     │    renderer.py: split_frame/transform                  │
│     ▼                                                       │
│  AvatarViewport (view.py): play / pause / speed / replay    │
└──────────────────────────────────────────────────────────────┘
```

## Data flow — một lượt dịch

1. `MicrophoneRecorder` thu 3 giây @16kHz → `write_wav` (numpy → WAV).
2. `faster-whisper` (`STTResult` = text + confidence), nạp model **lazy**
   khi có GPU/CPU tương ứng.
3. `translate_to_vsl(text)`:
   - tách câu thành cụm, phát hiện trợ từ / thì / phủ định;
   - bỏ trợ từ (`là`, `và`, `của`…), đưa thời gian vào đầu, phủ định đặt
     **sau** động từ (đúng trật tự VSL);
   - nếu `VSL_LLM_ENABLED` và có Ollama → refine lần 2, luôn so khớp về từ
     vựng; bất kỳ lỗi LLM nào → fallback kết quả quy tắc.
4. `avatar_controller.build_animation_sequence(vsl_structure)`:
   - `resolve_sign(word)` → mục trong vocabulary (kèm `animation_id`);
   - chia `duration_ms` thành từng `AnimationStep`; mỗi bước tham chiếu
     `reference://<sign>` → được render bằng frames keypoint `(30,201)`
     (nội tuyến khi cần, hoặc `frame_count` + tốc độ để frontend sinh lại).
5. `AvatarViewport` (matplotlib) vẽ khung xương cánh tay/mắt, phát theo
   `fps`; điều khiển: play/pause, `±` tốc độ, `R` replay, lịch sử.

## Backend run modes

| mode | cách chạy | ý nghĩa |
|------|-----------|---------|
| Standalone GUI | `python -m frontend.main` | pipeline chạy in-process (mặc định khi dev/test) |
| API server | `python -m backend.app` | REST `:8001`; frontend gọi qua HTTP nếu muốn tách rời |

`avatar/renderer.py` độc lập với GUI (thuần numpy/matplotlib) — tách bản vẽ
khỏi widget để dễ test và reuse ở Phần 3.

## Phụ thuộc dữ liệu

```
vsl_data/reference/reference_templates.npz   (3311 sign → (30,201) float32)
vsl_data/vocabulary.json                     (3315 entries, có animation_id)
```

Hai file trên là **giao ước chung với Phần 3** — game dạy ký hiệu nào thì avatar
Phần 2 anim phải phát được ký hiệu đó (được đảm bảo bằng
`shared/tests/test_integration.py`).