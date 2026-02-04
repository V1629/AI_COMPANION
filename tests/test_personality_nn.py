import numpy as np
from ai_companion.personality_nn import SimpleMLP

def test_forward_output_range():
    model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
    x = np.random.rand(2, 16)
    out = model.forward(x)
    assert out.shape == (2, 4)
    assert (out >= 0).all() and (out <= 1).all()

def test_train_step_reduces_loss():
    model = SimpleMLP(input_size=16, hidden_size=8, output_size=4)
    x = np.random.rand(8, 16)
    y = np.random.rand(8, 4)
    before = np.mean((model.forward(x) - y) ** 2)
    model.train_step(x, y, lr=0.01)
    after = np.mean((model.forward(x) - y) ** 2)
    assert after <= before + 1e-6