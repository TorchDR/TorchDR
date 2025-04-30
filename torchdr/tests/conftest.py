# conftest.py
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True, scope="session")
def set_deterministic_seed():
    seed = 42  # or any fixed integer you prefer
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
