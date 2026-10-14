"""Audit paired benchmark changes after the final Numina checkpoint evaluation."""
import json,math
from pathlib import Path
root=Path("/ssd/lzx/mopd/runs/20261013-math-evaluation")
def read(name):
 rows=[json.loads(s) for s in (root/name/"cases.jsonl").read_text().splitlines()]
 assert len(rows)==128 and len({r["id"] for r in rows})==128
 return {r["id"]:r for r in rows}
before=read("student_before");teacher=read("teacher");after=read("numina_student_after")
assert before.keys()==teacher.keys()==after.keys()
for k in before:
 assert before[k]["problem"]==teacher[k]["problem"]==after[k]["problem"]
improved=[k for k in before if not before[k]["correct"] and after[k]["correct"]]
regressed=[k for k in before if before[k]["correct"] and not after[k]["correct"]]
n=len(improved)+len(regressed)
p=min(1.,2*sum(math.comb(n,i) for i in range(min(len(improved),len(regressed))+1))/2**n) if n else 1.
report={"n":128,"improved":len(improved),"regressed":len(regressed),
"paired_exact_p":p,"accuracy_delta_pp":100*(len(improved)-len(regressed))/128,
"models":{name:{"correct":sum(bool(x["correct"]) for x in rows.values()),
"token_limit":sum(bool(x["hit_token_limit"]) for x in rows.values()),
"missing_box":sum(x.get("prediction") is None for x in rows.values()),
"grade_errors":sum(x.get("grade_error") is not None for x in rows.values())}
for name,rows in [("before",before),("teacher",teacher),("after",after)]},
"caveat":"One fixed 128-problem subset and one training seed; accuracy includes boxed-format compliance."}
(root/"numina_paired_audit.json").write_text(json.dumps(report,indent=2))
(root/"numina_changed_cases.json").write_text(json.dumps(
[{"id":k,"change":"improved" if k in improved else "regressed",
"before":before[k],"after":after[k],"teacher":teacher[k]} for k in improved+regressed],ensure_ascii=False,indent=2))
print(json.dumps(report,indent=2))
