// Build self-contained .glb avatars with bone + morph animations.
// Animations: offline alphabet from core.min.js `dr`, plus API-decoded words/phrases.
const fs = require('fs');
const path = require('path');
const { decode, tf, rf } = require('./decode.js');
const dr = require('./drdict.js');

const ROOT = __dirname;
const RAW = path.join(ROOT, 'out/raw');
const API = path.join(ROOT, 'out/api');
const OUT = path.join(ROOT, 'out');

const MORPH_NAMES = Object.entries(tf).filter(([c]) => 'UVWXYZ'.includes(c[0])).map(([, n]) => n);

// ---------- clip collection ----------
function segmentsOfApiResponse(d) {
  // d = [phrase][word][hta]
  const out = [];
  for (const phrase of d) for (const word of phrase) for (const hta of word) if (hta) out.push(hta);
  return out;
}

function collectClips(lang) {
  const clips = [];
  const dict = dr[lang] && dr[lang].animations ? dr[lang].animations : {};
  for (const k of Object.keys(dict)) rf[k] = dict[k];
  for (const k of Object.keys(dict)) clips.push({ name: `${lang}_${k}`, segments: [dict[k]] });

  for (const f of fs.readdirSync(API).sort()) {
    if (!f.endsWith('.json') || f === 'manifest.json') continue;
    let d;
    try { d = JSON.parse(fs.readFileSync(path.join(API, f), 'utf8')); } catch { continue; }
    const segs = segmentsOfApiResponse(d);
    if (!segs.length) continue;
    const slug = f.replace(/\.json$/, '');
    clips.push({ name: `${lang}_api_${slug}`, segments: segs });
  }
  return clips;
}

// ---------- decode + merge to channel data ----------
function mergeSegments(segments, morphMap, headNodeName) {
  const bone = new Map(); // nodeName -> {times:[],values:[]}
  const morph = new Map(); // idx -> {times:[],values:[]}
  let offset = 0;

  for (const hta of segments) {
    const clip = decode(hta, { morphMap, prefix: headNodeName });
    let end = 0;
    for (const tr of clip.tracks) {
      const t = tr.times.map((x) => x + offset);
      for (const x of tr.times) end = Math.max(end, x);
      const m = /^(.*)\.morphTargetInfluences\[(\d+)\]$/.exec(tr.name);
      if (m) {
        const idx = Number(m[2]);
        if (!morph.has(idx)) morph.set(idx, { times: [], values: [] });
        const rec = morph.get(idx);
        for (let i = 0; i < t.length; i++) { rec.times.push(t[i]); rec.values.push(tr.values[i]); }
      } else if (tr.name.endsWith('.quaternion')) {
        const node = tr.name.slice(0, -'.quaternion'.length);
        if (!bone.has(node)) bone.set(node, { times: [], values: [] });
        const rec = bone.get(node);
        for (let i = 0; i < t.length; i++) rec.times.push(t[i]);
        for (const v of tr.values) rec.values.push(v);
      }
    }
    offset += Math.max(end, 0.1);
  }
  return { bone, morph, duration: offset };
}

// ---------- glTF/GLB writer ----------
function align4(n) { return (n + 3) & ~3; }

class Builder {
  constructor(gltf) {
    this.gltf = gltf;
    const uri = gltf.buffers[0].uri;
    if (!uri || !uri.startsWith('data:')) throw new Error('expected embedded buffer');
    this.bin = Buffer.from(uri.split(',')[1], 'base64');
    if (this.bin.length !== gltf.buffers[0].byteLength) throw new Error('buffer length mismatch');
    this.gltf.animations = [];
  }
  addView(buf) {
    const off = align4(this.bin.length);
    this.bin = Buffer.concat([this.bin, Buffer.alloc(off - this.bin.length), buf, Buffer.alloc(align4(buf.length) - buf.length)]);
    this.gltf.bufferViews.push({ buffer: 0, byteOffset: off, byteLength: buf.length });
    return this.gltf.bufferViews.length - 1;
  }
  addAccessor(buf, type, components, count, withMinMax) {
    const n = { SCALAR: 1, VEC3: 3, VEC4: 4 }[type];
    if (buf.length !== count * n * 4) throw new Error('accessor size mismatch');
    const view = this.addView(buf);
    const acc = { bufferView: view, componentType: 5126, count, type };
    if (withMinMax) {
      const min = new Array(n).fill(Infinity), max = new Array(n).fill(-Infinity);
      for (let i = 0; i < count; i++) for (let c = 0; c < n; c++) {
        const v = buf.readFloatLE((i * n + c) * 4);
        if (v < min[c]) min[c] = v;
        if (v > max[c]) max[c] = v;
      }
      acc.min = min; acc.max = max;
    }
    this.gltf.accessors.push(acc);
    return this.gltf.accessors.length - 1;
  }
  addAnimation(name, merged, nodeIndexByBone, headNode, morphCount) {
    const channels = [], samplers = [];
    let end = 0;
    for (const [nodeName, rec] of merged.bone) {
      const node = nodeIndexByBone.get(nodeName);
      if (node === undefined) continue;
      const times = Float32Array.from(rec.times);
      const vals = Float32Array.from(rec.values);
      end = Math.max(end, rec.times[rec.times.length - 1] || 0);
      samplers.push({
        input: this.addAccessor(Buffer.from(times.buffer), 'SCALAR', 1, times.length, true),
        output: this.addAccessor(Buffer.from(vals.buffer), 'VEC4', 4, times.length, false),
        interpolation: 'LINEAR',
      });
      channels.push({ sampler: samplers.length - 1, target: { node, path: 'rotation' } });
    }
    if (merged.morph.size && headNode !== undefined && morphCount) {
      const union = [...new Set([...merged.morph.values()].flatMap((r) => r.times))].sort((a, b) => a - b);
      const frames = union.length;
      if (frames) {
        const out = new Float32Array(frames * morphCount);
        for (const [idx, rec] of merged.morph) {
          if (idx >= morphCount) continue;
          let p = 0, cur = 0;
          for (let f = 0; f < frames; f++) {
            while (p < rec.times.length && rec.times[p] <= union[f] + 1e-6) { cur = rec.values[p]; p++; }
            out[f * morphCount + idx] = cur;
          }
        }
        const tbuf = Float32Array.from(union);
        end = Math.max(end, union[frames - 1]);
        samplers.push({
          input: this.addAccessor(Buffer.from(tbuf.buffer), 'SCALAR', 1, frames, true),
          output: this.addAccessor(Buffer.from(out.buffer), 'SCALAR', 1, frames * morphCount, false),
          interpolation: 'LINEAR',
        });
        channels.push({ sampler: samplers.length - 1, target: { node: headNode, path: 'weights' } });
      }
    }
    if (!channels.length) return 0;
    this.gltf.animations.push({ name, channels, samplers });
    return end;
  }
  write(file) {
    this.gltf.buffers[0].byteLength = this.bin.length;
    this.gltf.buffers[0].uri = undefined;
    this.gltf.buffers[0].uri = 'data:application/octet-stream;base64,' + this.bin.toString('base64');
    const json = Buffer.from(JSON.stringify(this.gltf));
    const jsonPad = Buffer.concat([json, Buffer.alloc(align4(json.length) - json.length, 0x20)]);
    const binPad = Buffer.concat([this.bin, Buffer.alloc(align4(this.bin.length) - this.bin.length)]);
    const total = 12 + 8 + jsonPad.length + 8 + binPad.length;
    const head = Buffer.alloc(12);
    head.write('glTF', 0);
    head.writeUInt32LE(2, 4);
    head.writeUInt32LE(total, 8);
    const jh = Buffer.alloc(8); jh.writeUInt32LE(jsonPad.length, 0); jh.writeUInt32LE(0x4e4f534a, 4);
    const bh = Buffer.alloc(8); bh.writeUInt32LE(binPad.length, 0); bh.writeUInt32LE(0x004e4942, 4);
    fs.writeFileSync(file, Buffer.concat([head, jh, jsonPad, bh, binPad]));
    return total;
  }
}

function buildAvatar(avatar, lang) {
  const gltf = JSON.parse(fs.readFileSync(path.join(RAW, `${avatar}.gltf`), 'utf8'));
  const nodeIndexByBone = new Map();
  gltf.nodes.forEach((n, i) => {
    if (!n.name) return;
    const key = n.name.replace(/\./g, '');
    if (!nodeIndexByBone.has(key)) nodeIndexByBone.set(key, i);
    if (!nodeIndexByBone.has(n.name)) nodeIndexByBone.set(n.name, i);
  });
  // head mesh node: the one whose mesh declares the most morph targets
  let headNode, morphCount = 0;
  gltf.nodes.forEach((n, i) => {
    if (n.mesh === undefined) return;
    for (const p of gltf.meshes[n.mesh].primitives) if (p.targets && p.targets.length > morphCount) { morphCount = p.targets.length; headNode = i; }
  });
  const targetNames = morphCount ? gltf.meshes[gltf.nodes[headNode].mesh].extras?.targetNames : null;
  if (targetNames) {
    const missing = MORPH_NAMES.filter((n) => !targetNames.includes(n));
    if (missing.length) console.warn(`  ! ${avatar}: ${missing.length} morph names not in targetNames (e.g. ${missing[0]})`);
  }
  const morphMap = {};
  MORPH_NAMES.forEach((n) => { const i = targetNames ? targetNames.indexOf(n) : -1; if (i >= 0) morphMap[n] = i; });

  const headNodeName = headNode !== undefined ? gltf.nodes[headNode].name : '';
  const b = new Builder(gltf);
  let built = 0, skipped = 0;
  for (const clip of collectClips(lang)) {
    const merged = mergeSegments(clip.segments, morphMap, headNodeName);
    const end = b.addAnimation(clip.name, merged, nodeIndexByBone, headNode, morphCount);
    if (end) built++; else skipped++;
  }
  const file = path.join(OUT, `${avatar}_animated.glb`);
  const size = b.write(file);
  console.log(`  ${avatar}: ${built} clips (${skipped} empty), ${gltf.animations.length} in file, ${(size / 1e6).toFixed(2)} MB -> ${path.relative(ROOT, file)}`);
  return { avatar, clips: built, size };
}

const results = ['HUGO', 'MAYA'].map((a) => buildAvatar(a, process.argv[2] || 'bzs'));
json_write: {
  fs.writeFileSync(path.join(OUT, 'build.json'), JSON.stringify(results, null, 1));
}
console.log('done', results);
