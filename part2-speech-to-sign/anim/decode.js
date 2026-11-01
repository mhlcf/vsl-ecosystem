// Decode Hand Talk "hta" animation strings using the ORIGINAL decoder from core.min.js.
// We slice tf/lv/rf/hv/fv/dv/pv/mv/gl/vv/gv/yv/xv/Ev verbatim, then run them with tiny stubs.
const fs = require('fs');
const vm = require('vm');

const CORE = __dirname + '/core.min.js';
const core = fs.readFileSync(CORE, 'utf8');

function sliceFrom(startMarker, endMarker) {
  const s = core.indexOf(startMarker);
  if (s < 0) throw new Error('start not found: ' + startMarker);
  const e = core.indexOf(endMarker, s);
  if (e < 0) throw new Error('end not found: ' + endMarker);
  return core.slice(s, e);
}

// tf = bone/morph code table; ends right before lv=tf
const tfSrc = sliceFrom('tf={BA:"DEF-hips"', 'lv=tf');
// decoder block: lv=tf ... Ev(...) ->  ends at the clip object return
const decStart = core.indexOf('lv=tf');
const evStart = core.indexOf('Ev=(e,t,n,i,r', decStart);
if (evStart < 0) throw new Error('Ev not found');
// brace-match the Ev arrow body
let i = core.indexOf('{', evStart);
let depth = 0, end = -1;
for (let k = i; k < core.length; k++) {
  const c = core[k];
  if (c === '{') depth++;
  else if (c === '}') { depth--; if (depth === 0) { end = k + 1; break; } }
}
if (end < 0) throw new Error('Ev body not closed');
const decSrc = core.slice(decStart, end);

const stubs = `
const dt = 2.7;
const T1 = class { constructor(){} log(){} };
const Xo = (e) => e;
const dr = { bzs: { workspaceId: "HT-BZS", oralLanguageId: "por", animations: __rf } };
const Fe = new Proxy(function(){}, { get: (t, k) => (typeof k === 'string' ? (() => 'ERR:' + k) : undefined) });
`;

const src = stubs + '\n' + tfSrc + '\n' + decSrc + '\n;({Ev, tf});';
const ctx = {};
const __rf = {};
const sandbox = { module: { exports: {} }, console, __rf };
vm.createContext(sandbox);
const exported = vm.runInContext(src, sandbox, { filename: 'ht-decoder.js' });

// bones: name -> name (identity) so track names stay readable
const boneMap = {};
for (const code in exported.tf) {
  const v = exported.tf[code];
  if (!/^[UVWXYZ]/.test(code)) boneMap[v] = v;
}
const boneCount = Object.keys(boneMap).length;

function decode(hta, { morphMap = {}, prefix = '', offset = 0 } = {}) {
  if (boneCount !== 119) throw new Error('bone map must be 119, got ' + boneCount);
  return exported.Ev(hta, boneMap, morphMap, prefix, offset);
}

module.exports = { decode, tf: exported.tf, boneMap, boneCount, rf: __rf };

if (require.main === module) {
  const [hta] = process.argv.slice(2);
  const clip = decode(hta || 'BC*2#3#4#5*1#2#3#4*1#2#3#4*1#2#3#4?');
  console.log(JSON.stringify({ name: clip.name, duration: clip.duration, tracks: clip.tracks.length, first: clip.tracks[0] }, null, 1).slice(0, 900));
}
