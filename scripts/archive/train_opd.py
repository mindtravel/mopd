"""Single frozen teacher, student LoRA, fresh on-policy samples every update."""
import argparse,json,time,hashlib
from pathlib import Path
import torch
from mopd.backends.hf import load_pair
from mopd.rollout.sampling import generate
from mopd.teacher.scoring import score
from mopd.training.losses import sampled_reverse_kl
from mopd.data.prompts import TRAIN,HELDOUT
from peft import get_peft_model_state_dict,PeftModel

def save_json(path,data): path.write_text(json.dumps(data,ensure_ascii=False,indent=2))
def digest(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        h.update(n.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    c=json.loads(Path(args.config).read_text());out=Path(c['output']);out.mkdir(parents=True,exist_ok=False)
    save_json(out/'config.json',c);torch.manual_seed(c['seed']);tok,student,teacher=load_pair(c)
    trainable={n:p for n,p in student.named_parameters() if p.requires_grad}
    assert trainable and all('lora_' in n for n in trainable)
    teacher_hash=digest(teacher)
    initial={n:p.detach().cpu().clone() for n,p in trainable.items()}
    optimizer=torch.optim.AdamW(trainable.values(),lr=c['lr'])
    # Fixed held-out trajectories measure before/after distribution alignment, not task accuracy.
    held=[]
    for i,prompt in enumerate(HELDOUT):
        ids,mask,text=generate(student,tok,prompt,c['student_device'],c['max_new_tokens'],c['seed']+100+i)
        tlogits=score(teacher,ids,c['teacher_device'],c['student_device'])
        with torch.no_grad():
            logits=student(ids,use_cache=False).logits
            selected=mask[:,1:]
            tp=tlogits[:,:-1][selected].float().log_softmax(-1)
            sp=logits[:,:-1][selected].float().log_softmax(-1)
            kl=(sp.exp()*(sp-tp)).sum(-1).mean().item()
        held.append((ids,mask,tp.cpu(),{'prompt':prompt,'text':text,'kl_before':kl}))
        del tlogits,logits,sp,tp
    with (out/'metrics.jsonl').open('w',buffering=1) as metrics,(out/'rollouts.jsonl').open('w',buffering=1) as rollouts:
        for step in range(c['steps']):
            start=time.time();prompt=TRAIN[step%len(TRAIN)]
            ids,mask,text=generate(student,tok,prompt,c['student_device'],c['max_new_tokens'],c['seed']+step)
            tlogits=score(teacher,ids,c['teacher_device'],c['student_device'])
            student.train();optimizer.zero_grad(set_to_none=True)
            logits=student(ids,use_cache=False).logits
            loss,stats=sampled_reverse_kl(logits,tlogits,ids,mask)
            assert torch.isfinite(loss);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(trainable.values(),1.0)
            assert torch.isfinite(grad)
            optimizer.step()
            row={'step':step+1,'student_version':step,'loss':loss.item(),'grad_norm':float(grad),'seconds':time.time()-start,**stats}
            metrics.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
            rollouts.write(json.dumps({'step':step+1,'prompt':prompt,'response':text,'input_ids':ids.cpu().tolist(),'response_mask':mask.cpu().tolist()},ensure_ascii=False)+'\n')
            del ids,mask,logits,tlogits,loss
    changed=sum((initial[n]-p.detach().cpu()).float().square().sum().item() for n,p in trainable.items())**0.5
    assert changed>0 and all(p.grad is None for p in teacher.parameters())
    assert teacher_hash==digest(teacher),'Teacher weights changed'
    student.eval();evals=[]
    with torch.no_grad():
        for ids,mask,tp,item in held:
            logits=student(ids,use_cache=False).logits
            sp=logits[:,:-1][mask[:,1:]].float().log_softmax(-1)
            item['kl_after']=(sp.exp()*(sp-tp.to(sp.device))).sum(-1).mean().item();evals.append(item)
    adapter=out/'adapter';student.save_pretrained(adapter,safe_serialization=True);tok.save_pretrained(out/'tokenizer')
    # Reload adapter onto the already allocated base, avoiding a second base allocation.
    probe=held[0][0]
    with torch.no_grad(): expected=student(probe,use_cache=False).logits.float().cpu()
    base=student.unload();reloaded=PeftModel.from_pretrained(base,adapter).eval()
    with torch.no_grad(): actual=reloaded(probe,use_cache=False).logits.float().cpu()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    save_json(out/'summary.json',{'status':'PASS','steps':c['steps'],'trainable_parameters':sum(p.numel() for p in trainable.values()),'adapter_delta_l2':changed,'teacher_sha256_before_after':teacher_hash,'teacher_unchanged':True,'adapter_reload_exact':True,'heldout':evals,'scope':'4B synchronous sampled-token OPD infrastructure validation, not a task-quality benchmark'})
if __name__=='__main__':main()
