# Part 3 — Usage Examples

## 1. Flow hoàn chỉnh qua API

```bash
cd "$HOME/Documents"
./venv/bin/python -m uvicorn --app-dir part3-gamification backend.app:app --port 8002 &
# hoặc: (cd part3-gamification && ../../venv/bin/python -m backend.app)
sleep 2
H=http://localhost:8002

# tạo/mở session level 1
S=$(curl -s $H/api/v1/game/start-session -H 'Content-Type: application/json' \
     -d '{"user_id":1,"level_id":1}')
SID=$(echo "$S" | python3 -c 'import sys,json;print(json.load(sys.stdin)["session_id"])')

# chấm điểm bằng chuỗi keypoint chuẩn hóa (T×201)
curl -s $H/api/v1/game/submit-pose -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"frame_data\":{\"sequence\":[[0.0]*201]}}"

# chấm bằng ảnh base64
curl -s $H/api/v1/game/submit-pose -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"frame_data\":\"$(base64 -w0 frame.png)\"}"

# chốt cấp
curl -s $H/api/v1/game/complete-level -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"$SID\",\"attempts\":3,\"average_accuracy\":0.72}"
```

## 2. Scoring trực tiếp từ keypoints (không GUI)

```python
import sys; sys.path[:0] = ["part3-gamification", "part3-gamification/backend"]
from scoring_engine import classify_live
from level_manager import level_for, stars_for_accuracy

lv = level_for(1)
print("Ký hiệu level 1:", lv.word_name)

fake = [[[0.1, 0.2, 0.3] for _ in range(21)] for _ in range(60)]  # pose giả
res = classify_live(fake, lv.sign_name)
print("acc:", res.accuracy, "| breakdown:", res.accuracy_breakdown)
print("sao:", stars_for_accuracy(res.accuracy))
```

## 3. Unlock logic

```python
from user_manager import UserManager
m = UserManager()
m.register("test_user")
m.complete_level(user_id=1, level_id=1, stars=2)   # hồi lưu
print(m.is_unlocked(2, m.get_or_create("test_user").user_id))  # True
```

## 4. Mở khóa trong GUI

- Dashboard → nhấn level (chỉ level mở khóa mới sáng).
- Gameplay đạt ≥ `STAR_1_THRESHOLD` → Results hiện nút **Next Level**.
- Nút **Replay** cho phép cải thiện sao của level đã mở.

## 5. Feedbacks mẫu

| tình huống | feedback |
|------------|----------|
| `hand_shape` thấp | "Các ngón tay của bạn chưa khép đúng độ cong mẫu." |
| `position` thấp | "Đưa bàn tay gần với vị trí chuẩn hơn." |
| `movement` thấp | "Thực hiện chậm và đủ biên độ chuyển động." |
| `palm_orientation` thấp | "Xoay lòng bàn tay vào đúng hướng mẫu." |

## 6. Chạy thử với webcam

```bash
./run_part3.sh       # khởi động GUI; cấp quyền camera nếu hệ thống hỏi
```

Nên thử nhiều giờ ánh sáng/background đồng nhất; đứng cách camera 0.5–1.5 m
để MediaPipe bắt trọn hai bàn tay.