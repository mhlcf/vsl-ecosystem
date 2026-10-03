# Part 3 — Troubleshooting

| Vấn đề | Nguyên nhân | Xử lý |
|--------|-------------|-------|
| Camera không mở / đen | quyền thiết bị, thiếu `opencv-python` | `pip install opencv-python`; kiểm tra `v4l2-ctl --list-devices`; đóng app đang độc chiếm cam |
| MediaPipe không detect tay | ánh sáng yếu / khoảng cách / nhiều đối tượng | đứng cách 0.5–1.5m, nền đồng nhất, chiếu đủ sáng |
| Điểm luôn thấp dù làm đúng | góc so sánh khắt khe với template chuẩn hóa | thử tăng ngưỡng (`STAR_1_THRESHOLD`); dùng `movement` đủ biên độ |
| `learners.db is locked` | hai tiến trình cùng truy cập DB | đóng server API khi chạy GUI (hoặc đổi port) |
| Khởi động test lỗi `No module named 'backend'` | test chạy dưới `pytest` path khác; part3 conftest không tự load | chạy từ root: `./run_tests.sh` (root + part conftest logs sys.modules alias `p3backend`/`p3frontend`) |
| UI phóng to/model cửa sổ lỗi | thiếu sự kiện Qt trong môi trường headless | GUI cần `QT_QPA_PLATFORM` mặc định; test đã dùng `offscreen` |
| `classify_live` chậm | DTW trên 60 khung x 201 | chỉ chạy khi webcam bật; có thể giảm `kịch bản`; hiệu năng ~real-time trong thử nghiệm |
| Tất cả level bị khóa | user mới chưa hoàn thành level 1 | hoàn thành level 1 → `next_level_unlocked` |
| Sao hiển thị sai threshold | cấu hình `STAR_*_THRESHOLD` | kiểm tra `backend/config.py` |

## Gỡ lỗi nhanh

```bash
cd "$HOME/Documents"
./venv/bin/python -m pytest part3-gamification -q        # chỉ Phần 3
./venv/bin/python -m pytest shared/tests/test_integration.py -q   # ràng buộc Phần 2↔3
./run_tests.sh                                            # toàn bộ 89 tests
```