"""Offline perceptual weights and a batch-independent discriminator for VQ fine tuning."""
from pathlib import Path
import torch
from torch import nn


def load_perceptual(weights, device='cpu'):
    """LPIPS v0.1 SqueezeNet, [-1,1] RGB; no downloads or trainable metric weights."""
    import lpips
    model = lpips.LPIPS(net='squeeze', pretrained=False, pnet_rand=True,
                        eval_mode=True, verbose=False)
    model.load_state_dict(torch.load(Path(weights), map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


class PatchDiscriminator(nn.Module):
    """48px -> 6px patch logits; no running buffers to mutate in the generator phase."""
    def __init__(self, width=64):
        super().__init__()
        layers = []
        for i, (cin, cout) in enumerate([(3,width),(width,width*2),(width*2,width*4)]):
            layers += [nn.Conv2d(cin,cout,4,2,1)]
            if i: layers += [nn.GroupNorm(8,cout)]
            layers += [nn.LeakyReLU(.2)]
        self.net = nn.Sequential(*layers,nn.Conv2d(width*4,1,3,1,1))

    def forward(self, x):
        return self.net(x)


def assert_same_latent_mapping(a, b):
    """Decoder changes may reuse a prior only if encoder AND quantizer are identical."""
    for key, value in a.state_dict().items():
        if key.startswith(('encoder.', 'vq.')) and not torch.equal(value.cpu(), b.state_dict()[key].cpu()):
            raise ValueError(f'Latent mapping changed: {key}')
    if a.get_latent_shape() != b.get_latent_shape():
        raise ValueError('Latent shapes differ')
