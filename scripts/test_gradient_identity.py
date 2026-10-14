"""Enumerate all actions: sampled PG expectation equals exact reverse-KL gradient."""
import unittest
import torch
from mopd.training.losses import sampled_reverse_kl

class ReverseKLGradient(unittest.TestCase):
    def test_expected_sample_gradient_matches_full_kl(self):
        theta=torch.tensor([.2,-.3,.7],requires_grad=True)
        teacher=torch.tensor([-.4,.8,.1])
        p=theta.softmax(-1)
        direct=(p*(theta.log_softmax(-1)-teacher.log_softmax(-1))).sum()
        expected=torch.autograd.grad(direct,theta)[0]
        observed=torch.zeros_like(theta)
        for action in range(3):
            local=theta.detach().clone().requires_grad_(True)
            sl=local.view(1,1,3).expand(1,2,3)
            tl=teacher.view(1,1,3).expand(1,2,3)
            loss,_=sampled_reverse_kl(sl,tl,torch.tensor([[0,action]]),torch.tensor([[False,True]]),advantage_clip=100)
            observed+=p[action].detach()*torch.autograd.grad(loss,local)[0]
        torch.testing.assert_close(observed,expected)
if __name__=='__main__':unittest.main()
