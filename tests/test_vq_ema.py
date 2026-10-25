import unittest
import torch
from diffusion_demo.core.models.vae import VectorQuantizer

class TestVQEMA(unittest.TestCase):
    def test_update_is_explicit_and_eval_read_only(self):
        torch.manual_seed(1)
        vq=VectorQuantizer(8, 3, codebook_decay=.9, dead_code_threshold=.1)
        z=torch.randn(2,3,4,4,requires_grad=True)
        vq.initialize_from_data(z)
        before=vq.embeddings.weight.detach().clone()
        q, loss, ids=vq(z)
        (q.square().mean()+loss).backward()
        self.assertTrue(torch.equal(before,vq.embeddings.weight))
        self.assertTrue(z.grad.isfinite().all())
        self.assertIsNone(vq.embeddings.weight.grad)
        vq.update_codebook()
        self.assertFalse(torch.equal(before,vq.embeddings.weight))
        self.assertEqual(vq.ema_updates.item(),1)
        state={k:v.clone() for k,v in vq.state_dict().items()}
        vq.eval();vq(z);vq.update_codebook()
        self.assertTrue(all(torch.equal(v,state[k]) for k,v in vq.state_dict().items()))
        restored=VectorQuantizer(8,3,codebook_decay=.9,dead_code_threshold=.1)
        restored.load_state_dict(state);restored.eval()
        self.assertTrue(torch.equal(vq(z)[2],restored(z)[2]))
    def test_dead_revival(self):
        vq=VectorQuantizer(8,3,codebook_decay=.9,dead_code_threshold=.1)
        vq.ema_count.zero_();vq.ema_sum.zero_();vq.ema_updates.fill_(99)
        vq(torch.randn(2,3,4,4));vq.update_codebook()
        self.assertTrue((vq.ema_count>0).all())
        self.assertTrue(vq.embeddings.weight.isfinite().all())
