# Part 3 — Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ frontend/ (PyQt6, offscreen-safe)                           │
│  VSLApp (QStackedWidget) ▲                                    │
│  Dashboard ─ Level Select ─ Gameplay ─ Results ─ Profile    │
│      ▲             │   ▲            ▲                        │
│      │             ▼   │            │                        │
│  state.GameSession  CameraWorker (thread)                    │
│      │             MediaPipe overlay (draw_overlay)          │
│      ▼                                                    │
│ backend/ (FastAPI, cũng in-process khi chạy GUI)             │
│  level_manager.level_for ──► vocabulary.json                 │
│  sign_classifier ──► template_reference (DTW) — reference:// │
│  scoring_engine.classify_live / score_sequence               │
│     ├─ hand_shape      ├─ position                            │
│     ├─ movement        └─ palm_orientation                   │
│  feedback_generator ──► câu gợi ý theo tiêu chí thấp nhất    │
│  user_manager ──► vsl_data/vsl.db (stars, points, unlock)│
└─────────────────────────────────────────────────────────────┘
```

## Data flow — một lượt chơi

1. **start-session** → chọn level `N`: `level_is_unlocked(N, user)` kiểm tra
   toàn bộ `1..N-1` đã hoàn thành; trả về `word` mà người dùng phải ký.
2. **Gameplay**: `CameraWorker` đọc khung hình → lớp `draw_overlay` vẽ khung
   xương MediaPipe lên mặt nạ video; tích lũy **60 khung** gần nhất.
3. **classify_live(seq)** (tham chiếu `reference://word`) → 4 tiêu chí mỗi cao
   nhất khi cửa sổ so khớp tốt với template mẫu.
4. **results**: feedback mô tả lỗi thấp nhất + `stars_for_accuracy(acc)`.
5. **complete-level** → `UserManager` ghi sao/điểm; bảng user-level cập nhật,
   `next_level_unlocked` = đủ điều kiện tất cả level trước.

## Chấm điểm 4 tiêu chí

| tiêu chí | ý nghĩa | nhạy → |
|----------|---------|--------|
| `hand_shape` | biên dạng khớp ngón tay | khớp chuẩn hóa góc tay |
| `position` | vị trí không gian tay | khớp tọa độ tiêu chuẩn |
| `movement` | quỹ đạo/chuyển động | khớp temporal với template |
| `palm_orientation` | hướng lòng bàn tay | khớp pháp tuyến lòng bàn tay |

## Persistence

`vsl_data/vsl.db` (SQLite) — schema:

```
users(user_id PK, username UNIQUE, total_stars, total_points)
user_levels(user_id, level_id, stars, points, attempts, completed, PK(user_id, level_id))
```

Test tự set `VSL_DB_PATH` trỏ DB tạm nên không làm nhiễu dữ liệu thật.

## Giao ước với Part 2 (cùng data layer)

- **vocabulary.json** → bộ ký hiệu giảng dạy (lọc theo `animation_id`, dưỡng
  dưỡng khớp reference `shared/tests/test_integration.py`).
- **reference_templates.npz** → mẫu chuẩn cho scoring AND animation.