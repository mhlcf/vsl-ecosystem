# Hand Talk animation port (Part 2 — Speech → Sign avatar)

Port của pipeline animation từ app Hand Talk (`br.com.handtalk` 8.0.7) để tự build
avatar GLB có bone + morph animations, dùng cho ngôn ngữ ký hiệu mục tiêu.

- `core.min.js` — core của Hand Talk (minify, không obfuscate), source của decoder
- `bones.json` — 119 bone code → Rigify name map
- `decode.js` — decoder `hta-string` (vm + slicing từ core, giống `build/decode.js` gốc)
- `drdict.js` — trích bảng động từ offline `dr[lang].animations`
- `build.js` — merge segments → viết GLB (`node build.js [lang]`, mặc định `bzs`)
- `out/` — raw gltfs + API responses + generated `*_animated.glb`

## Dùng

```bash
cd part2-speech-to-sign/anim
node build.js bzs   # Libras (47 offline + 8 API)
node build.js ase    # ASL (43 offline + 8 API-derived)
```

## API

- `POST https://translation-v3.handtalk.me/api/plugin` (key/token trong `fetch.py`) → `hta-string`
- DSL: `?` block sep · `*` field sep (`code*times*axis1*axis2*axis3`) · `#` value sep
- Block `U..Z` = morph target; bone track = `.quaternion` (vec4 = wxyz)
- Output GLB: glTF 2.0 binary, animation = channel(sampler → node.rotation | node.weights)

## Giấy phép

Asset/động từ thuộc Hand Talk — dùng cho nghiên cứu/học tập, không phát hành lại.
