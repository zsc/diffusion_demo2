import torch
from vae_pca_probe import pca_from_moments,project_restore

def test_full_rank_identity_and_posterior_noise_covariance():
    torch.manual_seed(1);a=torch.randn(100,4,dtype=torch.float64);noise=torch.tensor([1.,2.,3.,4.],dtype=torch.float64)*100
    p=pca_from_moments(100,a.sum(0),a.T@a,noise)
    z=torch.randn(3,4,2,2)
    for mode in p:
        torch.testing.assert_close(project_restore(z,p[mode]['center'],p[mode]['basis'],4),z,atol=1e-6,rtol=1e-6)
    cov=a.T@a/100-torch.outer(a.mean(0),a.mean(0))+torch.diag(noise/100)
    b=p['sample']['basis'].double();torch.testing.assert_close(b@torch.diag(p['sample']['eigenvalues'])@b.T,cov,atol=1e-6,rtol=1e-6)
