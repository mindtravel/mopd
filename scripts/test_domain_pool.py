import unittest
from mopd.data.domains import teacher_content
from mopd.data.pool import build_plan,validate_holdout
class Tests(unittest.TestCase):
 def test_templates(self):
  r={"domain":"instruction","prompt":"Only output OK"}
  self.assertEqual(teacher_content(r),"Only output OK")
  self.assertIn("boxed",teacher_content({"domain":"math","prompt":"1+1"}))
 def test_holdout_overlap(self):
  with self.assertRaises(ValueError):validate_holdout([{"prompt":"hello  world"}],[{"prompt":"hello world"}])
  validate_holdout([{"prompt":"hello"}],[{"prompt":"world"}])
 def test_plan(self):
  rows=[{"id":str(i),"domain":"instruction","prompt":str(i),"split":"train"} for i in range(12)]
  a=build_plan(rows,seed=3,steps=3,batch_size=4)
  self.assertEqual(a,build_plan(rows,seed=3,steps=3,batch_size=4))
  self.assertEqual(len({r["id"] for b in a for r in b}),12)
  with self.assertRaises(ValueError):build_plan(rows,seed=3,steps=4,batch_size=4)
 def test_reject_eval_and_duplicate(self):
  r={"id":"a","domain":"instruction","prompt":"hello","split":"eval"}
  with self.assertRaises(ValueError):build_plan([r],seed=1,steps=1,batch_size=1)
  r["split"]="train"
  with self.assertRaises(ValueError):build_plan([r,dict(r,id="b")],seed=1,steps=1,batch_size=1)
if __name__=="__main__":unittest.main()
