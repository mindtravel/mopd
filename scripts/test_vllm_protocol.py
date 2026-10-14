"""Regression checks for token identity and policy-version boundaries."""
import unittest
from copy import deepcopy
from mopd.rollout.vllm_protocol import validate_reply
class ProtocolTests(unittest.TestCase):
 def setUp(self):
  self.inputs=[{"index":4,"prefix":[10,11],"row":{"id":"a"}},{"index":8,"prefix":[12],"row":{"id":"b"}}]
  self.reply={"version":7,"adapter_sha256":"abc","output":[{"index":8,"answer":[21,99]},{"index":4,"answer":[20,99]}]}
 def check(self,r):return validate_reply(r,self.inputs,7,"abc",4,99)
 def test_reorders_without_reencoding(self):
  rows=self.check(self.reply)
  self.assertEqual([r["index"] for r in rows],[4,8])
  self.assertEqual(rows[0]["answer"],[20,99]);self.assertEqual(rows[0]["prefix"],[10,11])
 def test_rejects_stale_policy_and_wrong_file(self):
  for key,value in [("version",6),("adapter_sha256","old")]:
   r=deepcopy(self.reply);r[key]=value
   with self.assertRaises(ValueError):self.check(r)
 def test_rejects_duplicate_or_missing(self):
  for outputs in [[self.reply["output"][0]]*2,self.reply["output"][:1]]:
   r=deepcopy(self.reply);r["output"]=outputs
   with self.assertRaises(ValueError):self.check(r)
 def test_rejects_empty_long_and_post_eos(self):
  for answer in [[],[1]*5,[99,20],[True]]:
   r=deepcopy(self.reply);r["output"][0]["answer"]=answer
   with self.assertRaises(ValueError):self.check(r)
if __name__=="__main__":unittest.main()
