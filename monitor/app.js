const RUN=decodeURIComponent(location.pathname.replace(/^\/run\//,"").replace(/\/$/,""));
let data=null,loading=false,pendingRefresh=false,offset=0,totalCases=0,lastCaseKey="";
const byteSize=v=>Number.isFinite(v)?(v>=1048576?(v/1048576).toFixed(1)+" MiB":v+" B"):"大小未知";
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
 return Object.keys(names).map(domain=>({label:names[domain],color:colors[domain],points:ms.map(m=>[m.step,m.domains?.[domain]?.[key]])}));
}
function deltaCell(row,ref,cmp,domain){
 const td=node("td");const d=delta(row,ref,cmp,domain);
 if(!d){td.textContent="—";return td}
 td.append(node(d.significant?"strong":"span",d.text));if(d.detail)td.append(node("small"," "+d.detail));return td;
}
function mainTable(){return (data.evaluation_tables||[])[0]}
function renderAbility(){
 const box=$("eval-tables");box.replaceChildren();const tables=data.evaluation_tables||[];
 for(const table of tables){
  const base=table.rows.find(r=>r.kind==="base"),init=table.rows.find(r=>r.kind==="init");
  box.append(node("h3",table.protocol+(table.base_from&&table.base_from!==RUN?"（原始学生结果取自 "+table.base_from+"，同底座同协议）":"")));
  const wrap=node("div","","table-wrap"),t=node("table"),head=node("tr");
  const cols=["版本","进度","IFEval 整题","相对 base","IFEval 严格指令","数学","相对 base"];if(init)cols.push("IFEval 相对初始化","数学 相对初始化");cols.push("截断 指令/数学");
  for(const c of cols)head.append(node("th",c));const thead=node("thead");thead.append(head);const tbody=node("tbody");
  for(const r of table.rows){const tr=node("tr");if(r.kind!=="checkpoint")tr.className="reference";
   const ins=r.domains.instruction,math=r.domains.math;
   tr.append(node("td",r.label),node("td",r.complete?"完成":(r.total?`${r.done??0}/${r.total} 进行中`:"排队")),
    node("td",r.complete?pct(ins?.accuracy):"—"),r.kind==="checkpoint"?deltaCell(r,base,r.vs_base,"instruction"):node("td",""),
    node("td",r.complete?pct(ins?.strict_instruction_accuracy):"—"),
    node("td",r.complete?pct(math?.accuracy):"—"),r.kind==="checkpoint"?deltaCell(r,base,r.vs_base,"math"):node("td",""));
   if(init)tr.append(...(r.kind==="checkpoint"?[deltaCell(r,init,r.vs_init,"instruction"),deltaCell(r,init,r.vs_init,"math")]:[node("td",""),node("td","")]));
   tr.append(node("td",r.complete?`${ins?.truncated??"—"} / ${math?.truncated??"—"}`:"—"));tbody.append(tr)}
  t.append(thead,tbody);wrap.append(t);box.append(wrap);
 }
 const table=mainTable();
 if(!table){$("assessment").textContent=data.status.state==="training"?"训练进行中，尚无检查点评测结果。":"此版本没有检查点评测结果。";$("ability-plots").replaceChildren();return}
 const base=table.rows.find(r=>r.kind==="base"),done=table.rows.filter(r=>r.kind==="checkpoint"&&r.complete);
 const pending=table.rows.filter(r=>r.kind==="checkpoint"&&!r.complete).length;
 if(!done.length)$("assessment").textContent="检查点评测进行中，尚无完整结果。";
 else{
  const last=done.at(-1),best=done.reduce((a,b)=>(b.domains.instruction?.accuracy??-1)>(a.domains.instruction?.accuracy??-1)?b:a);
  const part=domain=>{const d=delta(last,base,last.vs_base,domain);return `${names[domain]} ${pct(last.domains[domain]?.accuracy)}`+(d?`（相对 base ${d.text}，${d.detail.split(" · ")[1]||""}）`:"")};
  $("assessment").textContent=`最新完成评测：${last.label}，${part("instruction")}；${part("math")}。`+
   (best!==last?`IFEval 最高在${best.label}：${pct(best.domains.instruction?.accuracy)}。`:"")+(pending?`还有 ${pending} 个检查点在评测。`:"");
 }
 const refs=table.rows.filter(r=>r.complete&&r.kind!=="init");
 const series=Object.keys(names).map(domain=>({label:names[domain],color:colors[domain],
  points:refs.map(r=>[r.step,Number.isFinite(r.domains[domain]?.accuracy)?r.domains[domain].accuracy*100:NaN])}));
 $("ability-plots").replaceChildren(chart(table.protocol+" 正确率（%）",series,"第 0 步为原始学生；只绘制完整评测。"));
}
function renderTraining(ms){
 $("training-plots").replaceChildren(
  chart("策略梯度 loss",fieldSeries(ms,"loss"),"分别按领域求均值，不能按越低越好解释。"),
  chart("采样 reverse KL（估计）",fieldSeries(ms,"sampled_log_ratio"),"学生实际生成 token 上的 log p(学生) − log p(教师)；是采样估计，不是完整词表 KL。"),
  chart("各领域的梯度贡献",fieldSeries(ms,"grad_contribution_l2"),"裁剪前各领域贡献的 L2 范数，检查各领域是否参与更新。"),
  chart("达到输出上限的比例（%）",Object.keys(names).map(domain=>({label:names[domain],color:colors[domain],
   points:ms.map(m=>{const s=m.domains?.[domain];return[m.step,s?.count?100*s.truncated/s.count:NaN]})})),
   "输出上限为 "+data.config.max_new_tokens+" token；达到上限不等于回答错误。"));
}
function renderResources(ms){
 const training=data.status.state==="training"||data.status.state==="loading";
 $("gpu-wrap").hidden=!training;
 $("gpu-note").textContent=training?"GPU 表显示整卡占用，包含其它任务及缓存；曲线中的生成峰值是本任务的 allocated 显存，两者口径不同。":"训练已结束，不再显示 GPU 归属；以下为训练期间记录的耗时与显存。";
 $("gpus").replaceChildren();
 if(training){for(const g of data.gpu.devices||[]){const tr=node("tr");
  tr.append(node("td",String(g.index)),node("td",data.gpu_roles?.[String(g.index)]||"非本实验分配"),
   node("td",fmt(g.used_mib/1024,1)+" / "+fmt(g.total_mib/1024,1)+" GiB"),node("td",g.utilization+"%"));$("gpus").append(tr)}}
 const phase=[["generation_seconds","生成","#08796b"],["teacher_seconds","教师评分","#ad5917"],["student_seconds","学生前后向","#396ea4"]];
 const devices=data.config.rollout_devices||[];
 $("resource-plots").replaceChildren(
  chart("每步耗时（秒）",phase.map(([key,label,color])=>({label,color,points:ms.map(m=>[m.step,m[key]])})),"生成包含等待最长回答的时间。"),
  chart("实际生成吞吐（token/s）",[{label:"有效输出",color:colors.math,points:ms.map(m=>[m.step,m.generated_tokens_per_second])}],"使用自然结束的真实回答；不是等长压力测试结果。"),
  chart("各生成卡耗时（秒）",devices.map((device,i)=>({label:device.replace("cuda:","GPU "),color:i?colors.instruction:colors.math,
    points:ms.map(m=>{const s=(m.rollout?.workers||[]).filter(w=>w.device===device).map(w=>w.seconds);return[m.step,s.length?Math.max(...s):NaN]})})),
   "每张卡上最慢一批的耗时；两卡差越大，长尾等待越明显。"));
}
function renderVersion(){
 const box=$("version-info");box.replaceChildren();const v=(data.versions||[]).find(item=>item.key===RUN);
 if(!v){box.append(node("p","此运行没有 model-version.json。","muted"));return}
 const student=v.components?.student||{},cfg=data.config;
 box.append(node("p","运行目录："+v.path,"muted"),
  node("p","学生底座："+(student.base_model||v.student||"-")+" · 后端："+(student.backend||"hf_lora")+" · 更新方式："+(student.update_type||"lora")+" · 前向精度："+(student.forward_precision||"未登记")+" · 输入格式："+(student.prompt_format||cfg.student_prompt_format||"未登记")),
  node("p","初始化："+(cfg.student_adapter||"从 base 新建 LoRA")+" · 学习率 "+cfg.lr+" · seed "+cfg.seed+" · 每 "+cfg.save_every+" 步保存","muted"),
  node("p","量化配置："+JSON.stringify(student.quantization??null),"muted"),
  node("p","学生生成后端："+(v.rollout_backend||"未记录")+" · 检查点："+((v.checkpoints||[]).join("、")||"尚未发现"),"muted"));
 for(const [name,c] of Object.entries(cfg.domains||{}))
  box.append(node("p",(names[name]||name)+" · 每步 "+c.count+" 条 · 权重 "+c.weight,"label"),
   node("p","教师："+c.teacher+" · 训练数据："+c.train_file,"muted"),node("p","留出集："+(c.holdout_files||[]).map(shortPath).join("、"),"muted"));
 const inputs=v.training?.inputs?.domains||{};
 for(const [name,files] of Object.entries(inputs)){
  const records=[["训练",files.train],...(files.holdouts||[]).map((file,index)=>["留出 "+(index+1),file])];
  for(const [kind,file] of records)if(file)
   box.append(node("p",(names[name]||name)+" · "+kind+"："+file.path+" · "+byteSize(file.size_bytes)+" · SHA-256 "+file.sha256,"muted fingerprint"));
 }
 if(!Object.keys(inputs).length)box.append(node("p","此版本未记录训练数据内容指纹。","muted"));
}
async function renderCases(){
 const query=new URLSearchParams({run:RUN,step:$("step").value,domain:$("domain").value,truncated:$("cut").checked?"1":"0",offset:String(offset),limit:"20"});
 const response=await fetch("/api/cases?"+query);if(!response.ok)throw Error("回答读取失败 "+response.status);
 const result=await response.json();
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
function renderHeader(ms){
 const cfg=data.config,st=data.status,last=ms.at(-1)||{},rs=data.response_summary;
 document.title="MOPD · "+RUN;$("title").textContent=RUN;
 const domains=Object.entries(cfg.domains||{}).map(([name,spec])=>`${names[name]||name} ${spec.count} 条/步（${shortPath(spec.train_file).replace(/^data\//,"")}）`).join(" + ");
 $("description").textContent=`${cfg.student_adapter?"续训自 "+shortPath(cfg.student_adapter).replace(/^runs\//,""):"从 base 训练"} · ${domains} · lr ${cfg.lr} · 输出上限 ${cfg.max_new_tokens} token`;
 const phase={generating:"学生生成",scoring:"教师评分 / 学生更新"};
 $("connection").className=st.state==="failed"?"error":"";
 $("connection").textContent=(stateNames[st.state]||st.state||"等待状态")+(st.phase?" · "+(phase[st.phase]||st.phase):"")+" · 已完成 "+(last.step||0)+"/"+cfg.steps+" 步 · 页面更新 "+new Date().toLocaleTimeString();
 const table=mainTable(),done=(table?.rows||[]).filter(r=>r.kind==="checkpoint"&&r.complete),latest=done.at(-1),base=table?.rows.find(r=>r.kind==="base");
 const tile=(domain)=>{if(!latest)return["—","尚无完整评测"];const d=delta(latest,base,latest.vs_base,domain);return[pct(latest.domains[domain]?.accuracy),latest.label+(d?" · 相对 base "+d.text:"")]};
 const totalCut=Object.values(rs.domains).reduce((n,d)=>n+d.truncated,0);
 $("stats").replaceChildren();
 for(const[label,[value,note]]of [["完成更新",[`${last.step||0} / ${cfg.steps}`,"以完成 optimizer 更新为准"]],["IFEval 整题",tile("instruction")],["数学",tile("math")],["训练回答达到上限",[rs.count?pct(totalCut/rs.count):"—",totalCut+" / "+rs.count+" 条"]]]){
  const a=node("article","","stat");a.append(node("div",label,"label"),node("strong",value),node("small",note));$("stats").append(a)}
}
async function refresh(){
 if(loading){pendingRefresh=true;return}loading=true;
 try{
  const response=await fetch("/api?run="+encodeURIComponent(RUN));
  if(response.status===400)throw Error("没有名为 "+RUN+" 的实验");if(!response.ok)throw Error(response.status);
  data=await response.json();const ms=data.metrics,rs=data.response_summary;
  renderHeader(ms);renderAbility();renderTraining(ms);renderResources(ms);renderVersion();
  const selected=$("step").value;$("step").replaceChildren(new Option("全部","all"));for(const n of rs.steps)$("step").add(new Option("第 "+n+" 步",n));
  $("step").value=selected;if(!$("step").value)$("step").value="all";
  const key=[rs.count,$("step").value,$("domain").value,$("cut").checked,offset].join("|");
  if(key!==lastCaseKey){await renderCases();lastCaseKey=key}
 }catch(e){$("connection").className="error";$("connection").textContent="刷新失败，保留最近数据："+e.message}
 finally{loading=false;if(pendingRefresh){pendingRefresh=false;refresh()}}
}
$("refresh").onclick=refresh;
for(const id of ["step","domain","cut"])$(id).onchange=()=>{offset=0;lastCaseKey="";refresh()};
$("previous").onclick=()=>{offset=Math.max(0,offset-20);lastCaseKey="";refresh()};
$("next").onclick=()=>{if(offset+20<totalCases)offset+=20;lastCaseKey="";refresh()};
setInterval(()=>{if($("auto").checked)refresh()},10000);refresh();
