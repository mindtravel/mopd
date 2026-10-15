let runs=[];
function score(row,domain,base){
 const a=row?.domains?.[domain]?.accuracy;if(!Number.isFinite(a))return "—";
 const b=base?.domains?.[domain]?.accuracy;
 return pct(a)+(Number.isFinite(b)?` (${a>=b?"+":""}${((a-b)*100).toFixed(1)})`:"")+(row.step?` · 第 ${row.step} 步`:"");
}
function render(){
 const body=$("runs");body.replaceChildren();
 const shown=runs.filter(r=>$("minor").checked||!r.minor);
 for(const r of shown){
  const tr=node("tr");if(r.minor)tr.className="minor";
  const link=node("a",r.key);link.href="/run/"+encodeURIComponent(r.key);
  const name=node("td");name.append(link);
  const ins=r.domains.instruction;const base=r.base;
  tr.append(name,node("td",stateNames[r.state]||r.state,"state-"+r.state),node("td",`${r.step||0}/${r.steps||"?"}`),
   node("td",r.init?shortPath(r.init).replace(/^runs\//,""):"base"),
   node("td",ins?dataLabel(ins).replace(/^instruction\//,""):"—"),
   node("td",score(r.latest,"instruction",base)),node("td",score(r.best,"instruction",base)),
   node("td",score(r.latest,"math",base)),node("td",r.evaluated||"未评测"),node("td",when(r.updated)));
  body.append(tr);
 }
 if(!shown.length){const tr=node("tr"),td=node("td","没有符合条件的实验。");td.colSpan=10;tr.append(td);body.append(tr)}
 $("connection").className="";
 $("connection").textContent=`${shown.length} 个实验（共 ${runs.length} 个）· 页面更新 ${new Date().toLocaleTimeString()}`;
}
async function refresh(){
 try{const response=await fetch("/api/index");if(!response.ok)throw Error(response.status);runs=(await response.json()).runs;render()}
 catch(e){$("connection").className="error";$("connection").textContent="刷新失败，保留最近数据。请检查 SSH 隧道："+e.message}
}
$("minor").onchange=render;
setInterval(()=>{if($("auto").checked)refresh()},30000);refresh();
