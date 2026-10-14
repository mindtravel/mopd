"use strict";
const $=id=>document.getElementById(id);
const names={math:"数学",instruction:"指令遵循"};
const colors={math:"#08796b",instruction:"#ad5917"};
let data=null,loading=false,pendingRefresh=false,offset=0,totalCases=0,lastCaseKey="";
const fmt=(v,n=2)=>Number.isFinite(v)?v.toFixed(n):"—";
const pct=v=>Number.isFinite(v)?(v*100).toFixed(1)+"%":"—";
const byteSize=v=>Number.isFinite(v)?(v>=1048576?(v/1048576).toFixed(1)+" MiB":v+" B"):"大小未知";
function node(tag,text="",cls){const e=document.createElement(tag);e.textContent=text;if(cls)e.className=cls;return e}
function svgNode(tag,attrs,text){const e=document.createElementNS("http://www.w3.org/2000/svg",tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e}
function chart(title,series,note){
 const box=node("article","","plot");box.append(node("h3",title));
 series=series.map(s=>({...s,points:s.points.filter(p=>Number.isFinite(p[1]))}));
 const points=series.flatMap(s=>s.points);
 if(!points.length){box.append(node("div","等待记录完整数据","empty"));return box}
 const ys=points.map(p=>p[1]);let low=Math.min(...ys),high=Math.max(...ys);
 if(low===high){low-=Math.max(.05,Math.abs(low)*.1);high+=Math.max(.05,Math.abs(high)*.1)}
 if(title.includes("（%）")){low=0;high=Math.max(1,...ys)}
 const xmax=Math.max(1,...points.map(p=>p[0])),x=v=>58+v/xmax*466,y=v=>167-(v-low)/(high-low)*137;
 const svg=svgNode("svg",{viewBox:"0 0 550 210",role:"img","aria-label":title});
 for(let i=0;i<4;i++){const v=low+(high-low)*i/3;svg.append(svgNode("path",{d:`M58 ${y(v)}H530`,stroke:"#e0e7eb"}),svgNode("text",{x:2,y:y(v)+4,"font-size":11,fill:"#536873"},fmt(v)))}
 for(const s of series){if(!s.points.length)continue;
  svg.append(svgNode("path",{d:s.points.map((p,i)=>(i?"L":"M")+x(p[0])+" "+y(p[1])).join(" "),fill:"none",stroke:s.color,"stroke-width":2}));
  for(const p of s.points)svg.append(svgNode("circle",{cx:x(p[0]),cy:y(p[1]),r:3,fill:s.color}));
 }
 svg.append(svgNode("text",{x:210,y:199,"font-size":12,fill:"#536873"},"训练步 0 → "+xmax));box.append(svg);
 const legend=node("div","","legend");for(const s of series){const label=node("span",s.label);label.style.setProperty("--series",s.color);legend.append(label)}box.append(legend,node("p",note,"muted"));return box;
}
function fieldSeries(ms,key){
 return ["math","instruction"].map(domain=>({label:names[domain],color:colors[domain],points:ms.map(m=>[m.step,m.domains?.[domain]?.[key]])}));
}
function legacyMetrics(ms){
 if(data.run!=="math")return ms;
 return ms.map(m=>({...m,domains:{math:{loss:m.loss,tokens:m.tokens,truncated:m.truncated,count:data.config.accumulation||4}}}));
}
function renderBenchmark(){
 const reports=data.benchmarks||[];const table=$("benchmark-table");table.replaceChildren();
 for(const report of reports){const summary=report.summary&&Object.keys(report.summary).length?report.summary:(report.progress||{}),domains=summary.domains||{};
  const progress=summary.total?`${summary.done??summary.scored_count??0}/${summary.total}`:"—";
  const scores=Object.entries(domains).map(([domain,metrics])=>`${names[domain]||domain} ${Number.isFinite(metrics.accuracy)?pct(metrics.accuracy):"评分中"}`).join("；")||"等待评分";
  const tr=node("tr");tr.append(node("td",report.model_version_id||"未登记"),node("td",`${report.protocol_id||"协议未登记"} · ${report.backend||"后端未登记"}`),node("td",progress),node("td",scores));table.append(tr)}
 $("benchmark-summary").textContent=reports.length?"仅显示 generation manifest 明确归属于当前模型版本的评测。":"此模型版本还没有可追溯的 benchmark 结果。旧评测缺少 model_version_id，保留在历史目录中，不并入当前版本。";
 $("benchmark-cases").replaceChildren();
}
function renderAbility(){
 const items=data.evaluations||[];const done=items.filter(x=>x.complete);
 const base=done.find(x=>x.step===0),latest=done.filter(x=>x.step>0).at(-1);
 let assessment="尚无完整评测结果，暂不能判断能力提升。";
 if(base&&!latest)assessment="基线评测已完成，等待训练检查点的同题评测。";
 if(base&&latest){
  const changes=[];
  for(const domain of ["math","instruction"]){const b=base.result.domains?.[domain],a=latest.result.domains?.[domain];
   if(b&&a&&Number.isFinite(b.accuracy)&&Number.isFinite(a.accuracy)){const delta=(a.accuracy-b.accuracy)*100;changes.push(names[domain]+" "+(delta>=0?"+":"")+delta.toFixed(1)+" 个百分点")}}
  assessment="第 "+latest.step+" 步相对原始学生："+changes.join("；")+"。小样本变化仍需更多题验证。";
 }
 $("assessment").textContent=assessment;
 const st=data.evaluation_status||{};
 const stateNames={evaluating:"评测进行中",waiting_checkpoint:"等待检查点",complete:"全部检查点评测完成",failed:"评测失败，请查看评测日志"};
 $("eval-status").textContent=st.state?(stateNames[st.state]||st.state)+(Number.isFinite(st.step)?" · "+(st.step===0?"原始学生":"第 "+st.step+" 步"):""):"等待评测任务";
 $("evaluation").replaceChildren();
 for(const item of items){const r=item.result||{},domains=r.domains||{},tr=node("tr");
  tr.append(node("td",item.step===0?"原始学生":"第 "+item.step+" 步"),
   node("td",r.total?`${r.done}/${r.total}${item.complete?" · 完成":" · 进行中"}`:"准备中"),
   node("td",Number.isFinite(domains.math?.accuracy)?pct(domains.math.accuracy)+(item.complete?"":"（进行中）"):"—"),
   node("td",Number.isFinite(domains.instruction?.accuracy)?pct(domains.instruction.accuracy)+(item.complete?"":"（进行中）"):"—"),
   node("td",Number.isFinite(domains.instruction?.strict_instruction_accuracy)?pct(domains.instruction.strict_instruction_accuracy)+(item.complete?"":"（进行中）"):"—"));$("evaluation").append(tr)}
 if(!items.length){const tr=node("tr");const td=node("td","基线评测尚未写入结果，训练继续运行。");td.colSpan=5;tr.append(td);$("evaluation").append(tr)}
 $("eval-note").textContent="评测结果属于当前选择的模型版本；未完成的批次只显示进度，不作为能力变化结论。";
 $("ability-plots").replaceChildren(chart("固定留出题通过率（%）",["math","instruction"].map(domain=>({
  label:names[domain],color:colors[domain],points:done.map(e=>[e.step,Number.isFinite(e.result.domains?.[domain]?.accuracy)?e.result.domains[domain].accuracy*100:NaN])})),
  "只绘制完整评测结果；原始学生为第 0 步。"));
}
function renderTraining(ms){
 $("training-plots").replaceChildren(
  chart("策略梯度 loss",fieldSeries(ms,"loss"),"分别按领域求均值，不能按越低越好解释。"),
  chart("采样 reverse KL（估计）",fieldSeries(ms,"sampled_log_ratio"),"学生实际生成 token 上的 log p(学生) − log p(教师)；是采样估计，不是完整词表 KL。"),
  chart("各领域的梯度贡献",fieldSeries(ms,"grad_contribution_l2"),"裁剪前各领域贡献的 L2 范数，检查两个领域是否参与更新。"),
  chart("达到输出上限的比例（%）",["math","instruction"].map(domain=>({label:names[domain],color:colors[domain],
   points:ms.map(m=>{const s=m.domains?.[domain];return[m.step,s?.count?100*s.truncated/s.count:NaN]})})),
   "输出上限为 "+data.config.max_new_tokens+" token；达到上限不等于回答错误。"));
 if(data.validation?.length)$("training-plots").append(chart("历史固定轨迹 reverse KL",[{label:"数学",color:colors.math,points:data.validation.map(r=>[r.step,r.reverse_kl])}],"历史单领域实验的短固定轨迹验证，不属于当前双领域评测。"));
}
function renderVersions(){
 const box=$("version-list");box.replaceChildren();
 for(const v of (data.versions||[])){
  const d=node("details","");if(v.key===data.run)d.open=true;
  const student=v.components?.student;
  d.append(node("summary",v.version_id+" · "+v.state+" · "+(v.step||0)+"/"+(v.steps||"?")+" steps"),
   node("p","运行目录："+v.path,"muted"),
   node("p","学生底座："+(student?.base_model||v.student||"-")+" · 更新方式："+(student?.update_type||(v.student_adapter?"lora":"lora"))+" · 前向精度："+(student?.forward_precision||"未登记")),
   node("p","量化配置："+JSON.stringify(student?.quantization??null),"muted"),
   node("p","学生生成后端："+(v.rollout_backend||"未记录"),"muted"),
   node("p","检查点："+((v.checkpoints||[]).join("、")||"尚未发现"),"muted"));
  for(const [name,c] of Object.entries(v.domains||{}))
   d.append(node("p",names[name]||name,"label"),node("p","教师："+(c.teacher||"-")+" · 训练数据："+(c.train_file||"-")+" · 配比："+(c.weight??"未记录"),"muted"));
  const inputs=v.training?.inputs?.domains||{};
  for(const [name,files] of Object.entries(inputs)){
   const records=[["训练",files.train],...(files.holdouts||[]).map((file,index)=>["留出 "+(index+1),file])];
   for(const [kind,file] of records)if(file)
    d.append(node("p",(names[name]||name)+" · "+kind+"："+file.path+" · "+byteSize(file.size_bytes)+" · SHA-256 "+file.sha256,"muted fingerprint"));
  }
  if(!Object.keys(inputs).length)d.append(node("p","此版本未记录训练数据内容指纹。","muted"));
  box.append(d);
 }
}
function renderResources(ms){
 const roles={};const cfg=data.config;
 if(cfg.student_device)roles[cfg.student_device.replace("cuda:","")]="学生更新";
 if(cfg.domains)for(const[d,s]of Object.entries(cfg.domains))roles[s.device.replace("cuda:","")]=names[d]+"教师";
 else if(cfg.teacher_device)roles[cfg.teacher_device.replace("cuda:","")]="数学教师";
 for(const d of cfg.rollout_devices||[])roles[d.replace("cuda:","")]="学生生成";
 $("gpus").replaceChildren();
 for(const g of data.gpu.devices||[]){const tr=node("tr");
  tr.append(node("td",String(g.index)),node("td",data.gpu_roles?.[String(g.index)]||roles[g.index]||"非本实验分配"),
   node("td",fmt(g.used_mib/1024,1)+" / "+fmt(g.total_mib/1024,1)+" GiB"),node("td",g.utilization+"%"));$("gpus").append(tr)}
 if(data.gpu.error){const tr=node("tr"),td=node("td",data.gpu.error);td.colSpan=4;tr.append(td);$("gpus").append(tr)}
 const phase=[["generation_seconds","生成","#08796b"],["teacher_seconds","教师评分","#ad5917"],["student_seconds","学生前后向","#396ea4"]];
 const devices=cfg.rollout_devices||[];
 $("resource-plots").replaceChildren(
  chart("每步耗时（秒）",phase.map(([key,label,color])=>({label,color,points:ms.map(m=>[m.step,m[key]])})),"生成包含等待最长回答的时间。"),
  chart("实际生成吞吐（token/s）",[{label:"有效输出",color:colors.math,points:ms.map(m=>[m.step,m.generated_tokens_per_second??(m.generation_seconds?m.tokens/m.generation_seconds:NaN)])}],"使用自然结束的真实回答；不是等长压力测试结果。"),
  chart("各生成卡耗时（秒）",devices.map((device,i)=>({label:device.replace("cuda:","GPU "),color:i?colors.instruction:colors.math,
    points:ms.map(m=>[m.step,m.rollout?.workers?.find(w=>w.device===device)?.seconds])})),"同一步两卡耗时差越大，长尾等待越明显。"),
  chart("生成峰值 allocated 显存（GiB）",devices.map((device,i)=>({label:device.replace("cuda:","GPU "),color:i?colors.instruction:colors.math,
    points:ms.map(m=>[m.step,m.rollout?.workers?.find(w=>w.device===device)?.peak_gib])})),"本任务生成阶段的实际分配峰值，不包含全部缓存保留量。"));
}
async function renderCases(){
 const run=$("run").value;const query=new URLSearchParams({run,step:$("step").value,domain:$("domain").value,truncated:$("cut").checked?"1":"0",offset:String(offset),limit:"20"});
 const response=await fetch("/api/cases?"+query);if(!response.ok)throw Error("回答读取失败 "+response.status);
 const result=await response.json();if(run!==$("run").value)return;
 const opened=new Set([...$("cases").querySelectorAll("details[open]")].map(e=>e.dataset.key));
 $("cases").replaceChildren();totalCases=result.total;
 for(const r of result.rows){const d=node("details");d.dataset.key=[r.step,r.domain,r.id].join(":");d.open=opened.has(d.dataset.key);
  d.append(node("summary",`第 ${r.step} 步 · ${names[r.domain]||r.domain} · ${r.tokens} token · ${r.truncated?"达到长度上限":"自然结束"}`),
   node("p",`样本 ${r.id} · 采样 log 概率比 ${fmt(r.sampled_log_ratio)}`,"muted"),
   node("h3","提示"),node("pre",r.prompt||r.problem||""),node("h3","学生回答"),node("pre",r.response||""));
  $("cases").append(d)}
 if(!result.rows.length)$("cases").append(node("p","暂无符合条件的回答。","muted"));
 $("case-count").textContent=result.total?`${offset+1}–${Math.min(offset+20,result.total)} / ${result.total} 条`:"0 条";
 $("previous").disabled=offset===0;$("next").disabled=offset+20>=result.total;
}
async function refresh(){
 if(loading){pendingRefresh=true;return}loading=true;
 try{
  const run=$("run").value||"dual",response=await fetch("/api?run="+encodeURIComponent(run));if(!response.ok)throw Error(response.status);
  const fresh=await response.json();
  const options=fresh.runs||[];const selectedRun=options.some(item=>item.key===run)?run:(options.some(item=>item.key===fresh.run)?fresh.run:options[0]?.key||run);
  const select=$("run");select.replaceChildren(...options.map(item=>new Option(item.label,item.key)));
  if(selectedRun)select.value=selectedRun;if(selectedRun!==run){loading=false;refresh();return}
  if(run!==select.value)return;data=fresh;
  const ms=legacyMetrics(data.metrics),last=ms.at(-1)||{},rs=data.response_summary,cfg=data.config;
  const batch=cfg.domains?Object.values(cfg.domains).reduce((n,d)=>n+d.count,0):cfg.accumulation;
  const version=data.versions.find(item=>item.key===run),student=version?.components?.student||{};
  const domains=Object.entries(cfg.domains||{}).map(([name,spec])=>`${names[name]||name} ${(Number(spec.weight||0)*100).toFixed(0)}%`).join(" + ")||"单领域";
  $("description").textContent=`${student.backend||"HF/LoRA"} · ${domains} · 更新 ${student.update_type||"lora"} · 每步 ${batch} 条 · 输出上限 ${cfg.max_new_tokens} token`;
  const phase={generating:"学生生成",scoring:"教师评分 / 学生更新",loading:"加载模型"};
  const state={training:"训练中",complete:"训练完成",failed:"训练失败",loading:"加载模型"};
  $("connection").className=data.status.state==="failed"?"error":"";
  $("connection").textContent=(state[data.status.state]||data.status.state||"等待状态")+" · "+(phase[data.status.phase]||"")+" · 已完成 "+(last.step||0)+"/"+cfg.steps+" 步 · 页面更新 "+new Date().toLocaleTimeString();
  const totalCut=Object.values(rs.domains).reduce((n,d)=>n+d.truncated,0);
  $("stats").replaceChildren();
  for(const[label,value,note]of [["完成更新",`${last.step||0} / ${cfg.steps}`,"以完成 optimizer 更新为准"],["已记录回答",String(rs.count),"包含正在评分的批次"],["最长回答",String(rs.max_tokens)+" token","实际记录的输出长度"],["累计达到上限",rs.count?pct(totalCut/rs.count):"—",totalCut+" 条 / "+rs.count+" 条"]]){
   const a=node("article","","stat");a.append(node("div",label,"label"),node("strong",value),node("small",note));$("stats").append(a)}
  renderAbility();renderBenchmark();renderTraining(ms);renderResources(ms);renderVersions();
  const selected=$("step").value;$("step").replaceChildren(new Option("全部","all"));for(const n of rs.steps)$("step").add(new Option("第 "+n+" 步",n));
  $("step").value=selected;if(!$("step").value)$("step").value="all";
  const key=[run,rs.count,$("step").value,$("domain").value,$("cut").checked,offset].join("|");
  if(key!==lastCaseKey){await renderCases();lastCaseKey=key}
 }catch(e){$("connection").className="error";$("connection").textContent="刷新失败，保留最近数据。请检查 SSH 隧道："+e.message}
 finally{loading=false;if(pendingRefresh){pendingRefresh=false;refresh()}}
}
$("refresh").onclick=refresh;
$("run").onchange=()=>{offset=0;lastCaseKey="";$("step").value="all";refresh()};
for(const id of ["step","domain","cut"])$(id).onchange=()=>{offset=0;lastCaseKey="";refresh()};
$("previous").onclick=()=>{offset=Math.max(0,offset-20);lastCaseKey="";refresh()};
$("next").onclick=()=>{if(offset+20<totalCases)offset+=20;lastCaseKey="";refresh()};
setInterval(()=>{if($("auto").checked)refresh()},10000);refresh();
