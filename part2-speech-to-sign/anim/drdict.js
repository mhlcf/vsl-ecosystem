// Extract the language dictionary `dr` (offline alphabet animations) out of core.min.js
const fs=require('fs'),vm=require('vm');
const core=fs.readFileSync(__dirname + '/core.min.js','utf8');
const s=core.indexOf('dr={bzs:{');
let i=core.indexOf('{',s),depth=0,end=-1;
for(let k=i;k<core.length;k++){const c=core[k];if(c==='{')depth++;else if(c==='}'){depth--;if(depth===0){end=k+1;break;}}}
const src=core.slice(s,end+1)+';dr;';
const dr=vm.runInNewContext(src,{});
module.exports=dr;
if(require.main===module){
  for(const lang of Object.keys(dr)){const a=dr[lang].animations||{};console.log(lang,'workspace='+dr[lang].workspaceId,'oral='+dr[lang].oralLanguageId,'keys='+Object.keys(a).length,'sample='+Object.keys(a).slice(0,6).join(','));}
  const a=dr.bzs.animations;
  for(const k of ['a','z','0','9'])console.log('bzs',k,'len',(a[k]||'').length);
}
