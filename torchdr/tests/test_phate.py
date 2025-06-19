from torchdr.tests.utils import toy_dataset
from torchdr.diffusion_embedding.phate import PHATE
from torchdr.affinity.knn_normalized import NegPotentialAffinity
from torchdr.utils import pykeops
import torch
import pytest

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
USE_KEOPS = [False]
if pykeops:
    USE_KEOPS.append(True)

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("keops", USE_KEOPS)
def test_potential_dist(device, keops):
    data, _ = toy_dataset(return_tensor=True, dtype=torch.float32)
    data.requires_grad = True
    neg_affinity = NegPotentialAffinity(keops=keops, device=device)(data)
    assert neg_affinity.shape == (data.shape[0], data.shape[0])
    assert neg_affinity.min() < 0
    neg_affinity.sum().backward()
    assert data.grad is not None

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("keops", USE_KEOPS)
def test_phate(device, keops):
    torch.autograd.set_detect_anomaly(True)
    data, _ = toy_dataset(return_tensor=True, dtype=torch.float32)
    data = data.to(device)
    phate = PHATE(keops=keops, device=device)
    embedding = phate.fit_transform(data)
    assert embedding.shape == (data.shape[0], 2)



if __name__ == "__main__":
    pytest.main([__file__])
