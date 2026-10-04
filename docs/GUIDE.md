# Hướng dẫn sử dụng

Toàn bộ đã cài sẵn trong `~/Documents` (Python venv, dữ liệu, model). Không cần
lệnh cài đặt gì thêm — chỉ cần chạy.

```bash
cd ~/Documents
```

---

## 1. Chạy ứng dụng — chế độ đồ họa (GUI)

### Phần 3 — Học ký hiệu VSL qua webcam

```bash
./run_part3.sh
```

**Từng màn hình:**

| Màn hình | Thao tác |
|----------|----------|
| **Dashboard** | Nhập biệt danh → **Vào học**. Thấy tổng quan tiến độ. |
| **Level Select** | 30 cấp chia 3 độ khó. Cấp **sáng** = đã mở, **mờ** = còn khóa (phải qua cấp trước). Bấm một cấp để bắt đầu chơi. |
| **Gameplay** | Từ cần ký hiện trên màn hình + hướng dẫn. Bấm **Bắt đầu** để bật camera, thực hiện cử chỉ trước camera ~2 giây trong vòng 60 khung hình. |
| **Results** | Điểm theo 4 tiêu chí (Hình dạng tay, Vị trí, Chuyển động, Hướng lòng bàn tay) + gợi ý sửa + số sao (1–3). **Next Level** để qua cấp, **Replay** để luyện lại. |
| **Profile** | Hồ sơ: số cấp, sao, điểm, tiến trình từng cấp; **Xuất CSV**. |

**Mẹo:** đứng cách camera 0.5–1.5 m, ánh sáng đều, nền ít chi tiết. MediaPipe
cần nhìn thấy rõ bàn tay.

### Phần 2 — Nói tiếng Việt → Xem avatar làm cử chỉ

```bash
./run_part2.sh
```

1. Bấm **● Bắt đầu nói**, nói câu tiếng Việt, bấm **■ Dừng** → app chuyển thành
   **cấu trúc VSL** (topic / comment / chuỗi ký hiệu) và phát **avatar 3D** làm
   từng cử chỉ.
2. Không có mic / đang offline? **Gõ câu** vào ô text rồi Enter — avatar vẫn chạy.
3. Lần đầu có thể phải tải model nhận dạng giọng nói (~0.5 GB, 1 lần duy nhất).

**Phím tắt:** `Space` = play/pause avatar, `+`/`-` = tăng/giảm tốc độ, `R` = replay.

> Nếu chưa có model Whisper, gõ text vẫn dùng được bình thường (xem gợi ý ở
> `part2-speech-to-sign/TROUBLESHOOTING.md`).

### Phần 1 — Nhận diện ký hiệu realtime + ghép câu

```bash
./run_part1.sh
```

Nhận ký hiệu lẻ (3315 từ) theo thời gian thực. Từ ổn định được tích lũy —
**đủ 3 từ tự ghép thành câu tiếng Việt**, hoặc nhấn `s` để ghép ngay.

| Phím | Chức năng |
|------|-----------|
| `s` | Ghép câu (qua LLM API nếu cấu hình, không thì ghép cơ bản) |
| `c` | Xóa buffer + câu |
| `u` | Xóa từ cuối |
| `q` | Thoát |

> Muốn câu tự nhiên hơn? Tạo `part1-sign-classifier/.env` (copy từ
> `.env.example`) và điền `VSL_LLM_API_KEY` — hỗ trợ mọi endpoint kiểu
> OpenAI. Chưa có key vẫn chạy tốt với ghép cơ bản.

---

## 2. Chạy dạng API server (cho tích hợp / lập trình)

Hai server FastAPI, có Swagger tự sinh để thao tác thử.

```bash
# Phần 2 — Speech→Sign (port 8001)
cd ~/Documents/part2-speech-to-sign && ../../venv/bin/python -m backend.app

# Phần 3 — Game/Scoring (port 8002)
cd ~/Documents/part3-gamification  && ../../venv/bin/python -m backend.app
```

| Server | Swagger | Endpoint chính |
|--------|---------|----------------|
| Part 2 | `http://localhost:8001/docs` | `POST /api/v1/process-speech`, `POST /api/v1/translate-text`, `GET /api/v1/signs/{q}` |
| Part 3 | `http://localhost:8002/docs` | `POST /api/v1/game/start-session`, `POST /api/v1/game/submit-pose`, `POST /api/v1/game/complete-level`, `GET /api/v1/users/{id}/stats` |

Kiểm tra nhanh cả hệ thống (bật 2 server + gọi API thật):

```bash
cd ~/Documents && ./smoke.sh
```

Ví dụ gọi nhanh bằng `curl` — xem `part2-speech-to-sign/USAGE_EXAMPLES.md` và
`part3-gamification/USAGE_EXAMPLES.md`.

---

## 3. Chạy kiểm thử

```bash
cd ~/Documents && ./run_tests.sh        # 104 tests (cả 3 phần + shared)
```

Chạy từng phần:

```bash
./venv/bin/python -m pytest part3-gamification -q
./venv/bin/python -m pytest part2-speech-to-sign -q
./venv/bin/python -m pytest shared/tests -q
```

---

## 4. Gỡ lỗi nhanh

| Lỗi | Cách xử lý |
|-----|-----------|
| `No module named 'vslshared'` | venv đã được `pip install -e shared`; chạy bằng `./venv/bin/python`, không dùng python hệ thống |
| Camera không bật | cấp quyền webcam, đóng app khác đang dùng cam |
| Whisper không nhận dạng | tải model lần đầu cần mạng; sau đó dùng offline; hoặc gõ text |
| Cấp bị khóa | phải hoàn thành level trước đó với ≥ 60% độ chính xác |
| Ảnh/UI xấu (font tiếng Việt) | cài `fonts-noto-core` |
| Chi tiết khác | `part2-speech-to-sign/TROUBLESHOOTING.md` · `part3-gamification/TROUBLESHOOTING.md` |

---

## Tài liệu liên quan

- [`../README.md`](../README.md) — tổng quan & cấu trúc dự án
- `part1-sign-classifier/` — README, model, dataset, script train/inference
- `part2-speech-to-sign/` — README, API_DOCUMENTATION, ARCHITECTURE, USAGE_EXAMPLES, TROUBLESHOOTING
- `part3-gamification/` — README, API_DOCUMENTATION, ARCHITECTURE, USAGE_EXAMPLES, TROUBLESHOOTING
- Ảnh minh họa: [`screenshots/`](screenshots/)