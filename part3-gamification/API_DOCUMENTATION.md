# Part 3 — API Documentation

Base URL: `http://localhost:8002`. Format JSON.

```bash
cd "$HOME/Documents/part3-gamification"
"$HOME/Documents/venv/bin/python" -m backend.app        # mặc định port 8002
```

Swagger: `http://localhost:8002/docs`.

---

## `GET /health`

```json
{"status": "ok", "classifier": "template", "levels": 30}
```

## `POST /api/v1/game/start-session`

Tạo phiên chơi cho một level (mở khóa theo tiến độ người dùng).

**Request**: `{"user_id": 1, "level_id": 3}`

**Response** (`StartSessionResponse`)

| field | mô tả |
|-------|-------|
| `session_id` | id phiên (gửi lại khi chấm điểm) |
| `level`, `word` | id level và khóa ký hiệu chuẩn hóa (vd `xin_chào`) |
| `word_name` | nhãn tiếng Việt (vd `Xin chào`) |
| `video_reference` | vị trí clip mẫu (hiện placeholder) |
| `instructions` | hướng dẫn ngắn |

## `POST /api/v1/game/submit-pose`

Chấm điểm khung hình/keypoints người dùng theo 4 tiêu chí.

**Request**

| field | type | mô tả |
|-------|------|-------|
| `session_id` | string | từ `start-session` |
| `frame_data` | any | **chuỗi base64 ảnh** (RGB) *hoặc* `{"sequence": [[...201 số...] × T]}` — chuỗi keypoint đã chuẩn hóa `(T,201)` |

**Ví dụ** — gửi chuỗi keypoint:

```json
{"session_id": "abc123", "frame_data": {"sequence": [[0.0]*201, [0.1]*201]}}
```

Sai định dạng → HTTP 400 `{"detail":"bad frame_data: ..."}`.

**Response** (`SubmitPoseResponse`)

```json
{
  "status": "processing",
  "confidence": 0.72,
  "accuracy_breakdown": {
    "hand_shape": 0.81, "position": 0.55,
    "movement": 0.66, "palm_orientation": 0.88
  },
  "feedback": "Tăng độ lệch 4: vị trí của lòng bàn tay chưa khớp mẫu.",
  "is_correct": false
}
```

## `POST /api/v1/game/complete-level`

Đóng phiên: chốt sao + điểm, mở khóa level kế tiếp.

**Request**: `{"session_id":"...", "attempts":3, "average_accuracy":0.68}`

**Response**

| field | mô tả |
|-------|-------|
| `stars_earned` | 0–3 (`stars_for_accuracy`) |
| `level_completed` | có vượt (≥ threshold 1 sao) |
| `next_level_unlocked` | mở level tiếp theo chưa |
| `total_points` | lũy kế của user |

## `GET /api/v1/users/{user_id}/stats`

```json
{
  "user_id": 1, "username": "learner",
  "total_levels_completed": 8, "total_stars": 17, "total_points": 940,
  "progress": [{"level_id":1,"stars":3,"points":120}, "..."]
}
```

---

## Điểm & sao

- `stars_for_accuracy(acc)` → 0/1/2/3 theo ngưỡng cấu hình
  (`STAR_1/2/3_THRESHOLD`).
- Level `N` mở khóa khi hoàn thành **tất cả** level `< N` (điều kiện đúng
  "cần qua trước"), không chỉ level trước liền kề.

## Lỗi

| HTTP | meaning |
|------|---------|
| 404 | level/session/user không tồn tại |
| 403 | level chưa mở khóa |
| 422 | payload sai schema |
| 500 | lỗi scoring |