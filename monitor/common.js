"use strict";
const $=id=>document.getElementById(id);
const names={math:"数学",instruction:"指令遵循"};
const colors={math:"#08796b",instruction:"#ad5917"};
const stateNames={training:"训练中",complete:"训练完成",failed:"训练失败",stopped:"已停止",loading:"加载模型",pending:"未开始"};
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):"—";
const pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+"%":"—";
const shortPath=p=>p?p.replace(/^\/ssd\/lzx\/mopd\//,""):"—";
const when=t=>Number.isFinite(t)?new Date(t*1000).toLocaleString([], {month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"}):"—";
function node(tag,text="",cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e}
function pvalue(p){return p<0.001?"p<0.001":"p="+p.toFixed(3)}
// Change of one checkpoint against a reference row, from the paired test the backend computed.
function delta(row,ref,cmp,domain){
 const a=row?.domains?.[domain]?.accuracy,b=ref?.domains?.[domain]?.accuracy,t=cmp?.[domain];
 if(!Number.isFinite(a)||!Number.isFinite(b))return null;
 const pt=(a-b)*100;
 return {pt,text:(pt>=0?"+":"")+pt.toFixed(1)+" pt",detail:t?`+${t.gained}/−${t.lost} · ${pvalue(t.p)}`:"",significant:!!t&&t.p<0.05};
}
