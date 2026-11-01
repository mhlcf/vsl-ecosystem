import json, os, sys, time, urllib.request, urllib.error

K = "2b28e79b5d5ee2c3d6b1a531c2d1d316"
T = "894ad581fb6306c7363cb9c32a5dd9dd"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "api")
os.makedirs(OUT, exist_ok=True)


def fetch(q, lang="bzs"):
    parts = [("mode", "default"), ("token", T), ("q", q), ("subtype", "text"), ("lang", lang)]
    body = b""
    for k, v in parts:
        body += b"--X\r\nContent-Disposition: form-data; name=\"" + k.encode() + b"\"\r\n\r\n" + v.encode() + b"\r\n"
    body += b"--X--\r\n"
    req = urllib.request.Request(
        "https://translation-v3.handtalk.me/api/plugin",
        data=body,
        headers={
            "key": K,
            "Content-Type": "multipart/form-data; boundary=X",
            "Origin": "https://www.handtalk.me",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


lang = sys.argv[1] if len(sys.argv) > 1 else "bzs"
queries = sys.argv[2:] or (
    [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(i) for i in range(10)]
    + ["ola", "bom dia", "obrigado", "eu te amo", "tudo bem", "tchau", "desculpa", "por favor"]
)

manifest = []
for q in queries:
    slug = q.replace(" ", "_")
    try:
        raw = fetch(q, lang=lang)
        d = json.loads(raw)
        path = os.path.join(OUT, slug + ".json")
        json.dump(d, open(path, "w"))
        shapes = [len(x) for x in d]
        manifest.append({"q": q, "slug": slug, "bytes": len(raw), "phrases": len(d), "words": shapes})
        print(f"{q!r:14} {len(raw):7}B phrases={len(d)} words={shapes}")
    except Exception as e:
        print(f"{q!r:14} FAIL {e}")
    time.sleep(0.25)

json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
print("saved", len(manifest))
