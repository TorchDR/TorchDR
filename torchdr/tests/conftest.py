# conftest.py
import pytest
import torch
import numpy as np
import random


@pytest.fixture(autouse=True, scope="session")
def set_deterministic_seed():
    seed = 42  # or any fixed integer you prefer
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
