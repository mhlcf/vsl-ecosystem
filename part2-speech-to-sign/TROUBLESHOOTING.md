# Part 2 — Troubleshooting

| Vấn đề | Nguyên nhân | Xử lý |
|--------|-------------|-------|
| `RuntimeError: No GPU...` | torch không thấy GPU | `python -c "import torch; print(torch.cuda.is_available())"`; cài torch cu124 nếu `False` |
| Lỗi kết nối micro | thiếu `sounddevice`/portaudio | `pip install sounddevice`; kiểm tra `aplay -L`, thử mic khác trong hệ thống |
| Nút Ghi chạy nhưng không có text | model Whisper chưa tải / mạng chậm | lần đầu tải `.cache/huggingface` (cần mạng); chờ đủ, sau đó offline OK |
| API trả `STT model not available` | **tính năng theo thiết kế** khi vắng model | dùng `/api/v1/translate-text` hoặc gõ text trên GUI |
| Avatar không xuất hiện trong viewport | thiếu matplotlib backend | `pip install matplotlib`; nếu chạy qua SSH: cần `QT_QPA_PLATFORM=offscreen` và bỏ phần hiển thị |
| Khởi động app giải mã lỗi `QGradient`/font tiếng Việt | thiếu font | cài `fonts-noto-core` / `fonts-dejavu-core` |
| `No module named 'vslshared'` | chưa install shared | `pip install -e shared` (đã có sẵn trong venv) |
| Test chậm ở `test_integration.py` | nạp reference set (3311 mẫu) | bình thường ~30s; chỉ chạy khi cần: `-k "not integration"` |
| `library 'libcusparse.so.12' is not found` khi dừng nói | ctranslate2 không thấy lib CUDA 12 của pip (máy chỉ có driver, không có CUDA toolkit) | đã fix trong `run_part2.sh` (tự export `LD_LIBRARY_PATH` tới `venv/.../nvidia/*/lib`); nếu chạy tay, export tương tự hoặc `WHISPER_DEVICE=cpu` |

## Gỡ lỗi nhanh

```bash
cd "$HOME/Documents"
./venv/bin/python -m pytest part2-speech-to-sign -q -x     # chạy riêng Phần 2
./venv/bin/python -m pytest part2-speech-to-sign/frontend/tests -q
./run_tests.sh                                             # toàn bộ 104 tests
```

Để xem log backend: chạy app với `-v` / bật logging:

```bash
VSL_LOG=debug ../../venv/bin/python -m backend.app
```