import torch
from architecture.layers import FeedForward

def test_feedforward_preserves_shape():
    batch_size, seq_len, d_model, d_ff = 2, 4, 8, 16
    layer = FeedForward(d_model=d_model, d_ff=d_ff)
    x = torch.randn(batch_size, seq_len, d_model)

    y = layer(x)

    assert y.shape == x.shape

def test_feedforward_requires_correct_last_dim():
    layer = FeedForward(d_model=8, d_ff=16)
    x = torch.randn(2, 4, 7)  # wrong last dimension

    try:
        layer(x)
    except RuntimeError as e:
        # torch usually raises RuntimeError for shape mismatches
        assert "mat1 and mat2 shapes" in str(e)
    else:
        assert False, "Expected RuntimeError due to shape mismatch"
