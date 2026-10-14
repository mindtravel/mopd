import unittest
import torch
from mopd.training.losses import sampled_reverse_kl

class LossChecks(unittest.TestCase):
    def test_response_only_and_teacher_detached(self):
        logits = torch.tensor([[[1.,0.],[0.,1.],[1.,0.]]],requires_grad=True)
        teacher = torch.tensor([[[0.,1.],[1.,0.],[0.,1.]]],requires_grad=True)
        ids = torch.tensor([[0,1,0]])
        mask = torch.tensor([[False,True,False]])
        loss, stats = sampled_reverse_kl(logits,teacher,ids,mask)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertEqual(logits.grad[0,1:].abs().sum().item(),0)
        self.assertGreater(logits.grad[0,0].abs().sum().item(),0)
        self.assertEqual(stats['tokens'],1)
    def test_identical_policy_zero_gradient(self):
        logits=torch.randn(1,4,5,requires_grad=True)
        loss,_=sampled_reverse_kl(logits,logits.detach().clone(),torch.tensor([[1,2,3,4]]),torch.ones(1,4,dtype=torch.bool))
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(),0)
    def test_empty_mask_rejected(self):
        with self.assertRaises(ValueError):
            sampled_reverse_kl(torch.zeros(1,2,3),torch.zeros(1,2,3),torch.zeros(1,2,dtype=torch.long),torch.zeros(1,2,dtype=torch.bool))
if __name__=='__main__': unittest.main()
