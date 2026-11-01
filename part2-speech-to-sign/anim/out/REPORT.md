# Hand Talk 8.0.7 — reverse engineering: avatar 3D + animation

Ngày: 2026-11-01 · Mục tiêu: bóc avatar Hugo/Maya và dựng lại animation ngôn ngữ ký hiệu.

## Nguồn

| | |
|---|---|
| APK | `https://d.apkpure.net/b/APK/br.com.handtalk?version=latest` → `hand-talk-8.0.7.apk` |
| SHA-256 | `603e94f31b14efbc7e0c6d7a568bddccce439650099554cb26ff36cd0c08a728` |
| Package | `br.com.handtalk` (75.033.508 byte, 1247 file) |
| Ghi chú | APK gốc từ Play (có `stamp-cert-sha256`) |

## Kiến trúc app

- **React Native** (`assets/index.android.bundle`, Hermes bytecode) cho UI.
- **WebView + three.js** cho avatar: `assets/index.html`, `assets/js/core.min.js` (832 KB, chỉ minify, không mã hoá).
- **Không có** Unity / IL2CPP / Flutter / native lib nào trong APK.

## Avatar (tĩnh)

9 file glTF 2.0, xuất từ Blender, **tự chứa** (buffer + texture nhúng base64):

| File | Nội dung |
|---|---|
| `HUGO.gltf` | thân Hugo, 9 mesh, 248 joint, 60 morph target (khuôn mặt) |
| `HUGO_HUGO_{carnaval,kimono,medico,natal,saojoao}.gltf` | trang phục |
| `MAYA.gltf`, `MAYA_MAYA_carnaval.gltf` | Maya + trang phục |

Rig: 119 bone điều khiển (`DEF-*`), tên kiểu Rigify. Trong glTF tên có dấu chấm (`DEF-thigh.L.01`), trong bảng mã của app thì không (`DEF-thighL01`).

## Animation — nguồn và cách dựng lại

**Không có clip nào trong APK, cũng không có trên CDN.** Animation được server trả về dạng chuỗi nén khi dịch:

```
POST https://translation-v3.handtalk.me/api/plugin
headers: key: 2b28e79b5d5ee2c3d6b1a531c2d1d316   (hardcode trong core.min.js)
form:    mode=default, token=<public token của widget web>, q=<câu>, subtype=text, lang=bzs
→ 200, JSON: [phrase][word][hta-string]
```

`hta-string` là DSL nội bộ:

| Ký tự | Ý nghĩa |
|---|---|
| `?` | phân cách các block (mỗi block = 1 track) |
| `*` | phân cách field trong block: `code*times*axis1*axis2*axis3` |
| `#` | phân cách giá trị |
| `times` | số frame × 1/24 s + offset |
| `axis1..3` | góc Euler XYZ, đơn vị = độ/2.7 → radian → quaternion |
| block bắt đầu `U..Z` | morph target (giá trị %/100) thay vì quaternion |

Bảng ghép mã (lấy nguyên từ `core.min.js`): `tf` (179 mã: 119 bone + 60 morph), `dr[lang].animations` (bảng chữ cái offline cho `bzs`/`ase`/`bsl`), `nf` (ký tự → mã).

**Decoder dùng chính code của app**, không viết lại công thức: `build/decode.js` cắt nguyên đoạn `tf`/`lv`/`dv`/`pv`/`gv`/`xv`/`Ev` từ `core.min.js` rồi chạy trong `node:vm`.

## Kết quả

| File | Nội dung |
|---|---|
| `out/HUGO_animated.glb` | Hugo + 55 animation (9,1 MB) |
| `out/MAYA_animated.glb` | Maya + 55 animation (13,4 MB) |
| `out/raw/*.gltf` | 8 avatar/trang phục gốc, chưa có animation |
| `out/viewer.html` | viewer three.js (chọn clip, xoay/zoom) |
| `out/shot_*.png` | ảnh render kiểm tra |
| `out/api/*.json` | response API thô |

55 clip mỗi avatar: 47 ký hiệu offline (a–z, 0–9, dấu câu) + 8 câu (`ola`, `bom dia`, `obrigado`, `eu te amo`, `tudo bem`, `tchau`, `desculpa`, `por favor`). Mỗi clip 120 channel = 119 bone quaternion + 1 channel morph (60 target). `bzs_repouso2` là idle 25 s.

## Kiểm chứng

- `@gltf-transform/cli validate` → **No errors found** (chỉ warning `BUFFER_VIEW_TARGET_MISSING` có sẵn từ file gốc của Blender).
- `build/check.js`: magic/length GLB, accessor nằm trong bufferView, alignment 4, channel `node|path` không trùng, `times` tăng dần, **quaternion norm = 1.000**, `weights` đúng `frames × 60`.
- `build/probe-motion.js` (three.js thật trong Brave headless): bone tay/ngón/đầu dịch chuyển khi phát clip; morph đổi từ 0 → 0,6 ở `bom dia`.
- `build/probe-pose.js`: fingertip ở cùng thời điểm khác nhau giữa các ký hiệu (a vs o: 0,065 m; a vs câu: 0,17 m) → không phải clip rỗng/trùng.
- `ola` và `tchau` trả về **cùng dữ liệu** — đúng ngữ nghĩa Libras (cùng một dấu vẫy tay).

## Mở rộng

```bash
# thêm câu: thêm vào queries trong build/fetch.py rồi
python3 build/fetch.py && node build/build.js

# ASL / BSL: đổi lang trong fetch.py (ase / bsl) và collectClips('ase'),
# bảng chữ cái offline của 2 ngôn ngữ này đã có sẵn trong core.min.js
```

## Giới hạn

- Chỉ dựng Libras (`bzs`); `ase`/`bsl` cần thêm 1 lần chạy.
- 24 fps, nội suy LINEAR — khớp engine gốc nhưng không thêm easing.
- `token` dùng là token công khai của widget handtalk.me; có thể bị thu hồi.
- Asset thuộc bản quyền Hand Talk — dùng cho nghiên cứu/học tập, không phát hành lại.
