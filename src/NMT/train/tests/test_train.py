import pytest
from NMT.train import train

@pytest.mark.skip()
def test_best_checkpoint():
    train._best_checkpoint()
