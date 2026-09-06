"""
Neural Networks From Scratch: Forward and Backward

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - numerical_gradient
def numerical_gradient(f, x, eps=1e-5):
    x = np.array(x, dtype=float, copy=True)
    grad = np.zeros_like(x, dtype=float)

    for idx in np.ndindex(x.shape):
        original = x[idx]

        x[idx] = original + eps
        f_plus = f(x)

        x[idx] = original - eps
        f_minus = f(x)

        x[idx] = original

        grad[idx] = (f_plus - f_minus) / (2.0 * eps)

    return grad

# Step 2 - gradient_check
import numpy as np

def gradient_check(analytic_grad, numeric_grad, tol=1e-5):

    analytic_grad = np.asarray(analytic_grad, dtype=float)  # change it to numpy array you can also use np.array()
    numeric_grad = np.asarray(numeric_grad, dtype=float)

    diff = np.abs(analytic_grad - numeric_grad)

    scale = np.maximum(
        np.maximum(np.abs(analytic_grad), 
        np.abs(numeric_grad)),
        tol
    )

    relative_error = diff / scale

    return float(np.max(relative_error))

# Step 3 - make_dense
def make_dense(in_dim, out_dim, weight_init_fn):
    """Create a fully connected layer."""

    # params
    W, b = weight_init_fn(in_dim, out_dim)

    params = {
        'W': W,
        'b': b
    }

    # forward pass
    def forward(x):
        y = x @ params['W'] + params['b']
        cache = x
        return y, cache

    # backward pass
    def backward(dout, cache):
        x = cache

        dx = dout @ params['W'].T
        dW = x.T @ dout
        db = dout.sum(axis=0)

        grads = {
            'W': dW,
            'b': db
        }

        return dx, grads

    return {
        'params': params,
        'forward': forward,
        'backward': backward
    }

# Step 4 - make_activation
def make_activation(kind='relu'):
    """Create a genuinely nonlinear elementwise activation layer.

    Args:
        kind: str nonlinearity name. Default 'relu' must implement ReLU
              (zero negatives, pass non-negatives). Other kinds optional.

    Returns:
        Layer dict with:
          forward(x) -> (y, cache)
            x, y: np.ndarray shape (batch, dim)
          backward(dout, cache) -> (dx, {})
            dout, dx: np.ndarray shape (batch, dim)
            param grad dict is always empty (no learnable params)

    Must be elementwise and non-affine; analytic dx must match
    numerical_gradient / gradient_check.
    """
    
    if kind != 'relu':
        raise ValueError(f"Unsupported activation: {kind}")

    def forward(x):
        y = np.maximum(0, x)
        
        cache = x
        
        return y, cache

    def backward(dout, cache):
        x = cache
        
        dx = dout * (x > 0)
        
        param_grads = {}
        
        return dx, param_grads

    return {
        'params': {},
        'forward': forward,
        'backward': backward
    }

# Step 5 - initialize_weights (not yet solved)
# TODO: implement

# Step 6 - make_loss (not yet solved)
# TODO: implement

# Step 7 - make_sequential (not yet solved)
# TODO: implement

# Step 8 - forward_backward (not yet solved)
# TODO: implement

# Step 9 - make_optimizer (not yet solved)
# TODO: implement

# Step 10 - train_step (not yet solved)
# TODO: implement

# Step 11 - train (not yet solved)
# TODO: implement

# Step 12 - design_network (not yet solved)
# TODO: implement

# Step 13 - improve_generalization (not yet solved)
# TODO: implement

