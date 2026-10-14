import unittest
import torch
from mopd.data.multidomain import balanced_plan, domain_scale
class Tests(unittest.TestCase):
 def test_balanced_without_repeats(self):
  pools={d:[{"id":str(i),"domain":d,"prompt":d+str(i),"split":"train"} for i in range(8)] for d in ("math","instruction")}
  plan=balanced_plan(pools,{"math":2,"instruction":2},seed=4,steps=4)
  self.assertEqual(plan,balanced_plan(pools,{"math":2,"instruction":2},seed=4,steps=4))
  for step in plan:
   self.assertEqual([r["domain"] for r in step],["math","math","instruction","instruction"])
  self.assertEqual(len({(r["domain"],r["id"]) for s in plan for r in s}),16)
  with self.assertRaises(ValueError):balanced_plan(pools,{"math":2,"instruction":2},seed=4,steps=5)
 def test_weighting_is_domain_mean(self):
  x=torch.tensor(2.,requires_grad=True)
  weights={"math":0.5,"instruction":0.5};counts={"math":1,"instruction":3}
  loss=x*2*domain_scale("math",weights,counts)
  for k in (1,2,3):loss=loss+x*k*domain_scale("instruction",weights,counts)
  loss.backward()
  self.assertAlmostEqual(x.grad.item(),2.)
 def test_invalid_weights(self):
  with self.assertRaises(ValueError):domain_scale("math",{"math":0.8,"instruction":0.8},{"math":2,"instruction":2})
if __name__=="__main__":unittest.main()
