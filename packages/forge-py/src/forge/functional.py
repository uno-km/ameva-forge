from typing import Tuple
from .tensor import Tensor
from .autograd import Function, Context
from .ops import exp_op, log_op, sum_op, sub, mul, neg, div, mean_op, add
import numpy as np

class SoftmaxFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, axis: int = -1) -> Tensor:
        ctx.axis = axis
        if x.device == 'cpu':
            from .ops import _require_cpu_data
            data = _require_cpu_data(x, 'x')
            max_val = np.max(data, axis=axis, keepdims=True)
            exp_data = np.exp(data - max_val)
            sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
            result = exp_data / sum_exp
            
            ctx.save_for_backward(Tensor(shape=result.shape, dtype='float32', device='cpu', data=result))
            return Tensor(shape=result.shape, dtype='float32', device='cpu', data=result)
        else:
            from .ops import exp_op, div, transpose, sum_axis, reshape
            e = exp_op(x)
            
            if axis == -1 or axis == 1:
                t = transpose(e)
                s = sum_axis(t, axis=0)
                s_reshaped = reshape(s, (x.shape[0], 1))
            elif axis == 0:
                s = sum_axis(e, axis=0)
                s_reshaped = reshape(s, (1, x.shape[1]))
            else:
                s = sum_axis(e, axis=0)
                s_reshaped = reshape(s, (1, x.shape[1]))
                
            res = div(e, s_reshaped)
            ctx.save_for_backward(res)
            return res

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        res, = ctx.saved_tensors
        if res.device == 'cpu':
            res_data = res.numpy()
            grad_data = grad_output.numpy()
            axis = ctx.axis
            sum_val = np.sum(grad_data * res_data, axis=axis, keepdims=True)
            grad_in = res_data * (grad_data - sum_val)
            return (Tensor(shape=res.shape, dtype='float32', device='cpu', data=grad_in),)
        else:
            from .ops import mul, sub, transpose, sum_axis, reshape
            axis = ctx.axis
            m = mul(grad_output, res)
            
            if axis == -1 or axis == 1:
                t = transpose(m)
                s = sum_axis(t, axis=0)
                sum_val = reshape(s, (res.shape[0], 1))
            elif axis == 0:
                s = sum_axis(m, axis=0)
                sum_val = reshape(s, (1, res.shape[1]))
            else:
                s = sum_axis(m, axis=0)
                sum_val = reshape(s, (1, res.shape[1]))
                
            diff = sub(grad_output, sum_val)
            return (mul(res, diff),)

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    return SoftmaxFunction.apply(x, axis=axis)

class LogSoftmaxFunction(Function):
    @staticmethod
    def forward(ctx: Context, x: Tensor, axis: int = -1) -> Tensor:
        ctx.axis = axis
        if x.device == 'cpu':
            from .ops import _require_cpu_data
            data = _require_cpu_data(x, 'x')
            max_val = np.max(data, axis=axis, keepdims=True)
            shifted = data - max_val
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
            result = shifted - log_sum_exp
            
            ctx.save_for_backward(Tensor(shape=result.shape, dtype='float32', device='cpu', data=np.exp(result)))
            return Tensor(shape=result.shape, dtype='float32', device='cpu', data=result)
        else:
            from .ops import exp_op, div, transpose, sum_axis, reshape, log_op, sub
            e = exp_op(x)
            
            if axis == -1 or axis == 1:
                t = transpose(e)
                s = sum_axis(t, axis=0)
                s_reshaped = reshape(s, (x.shape[0], 1))
            elif axis == 0:
                s = sum_axis(e, axis=0)
                s_reshaped = reshape(s, (1, x.shape[1]))
            else:
                s = sum_axis(e, axis=0)
                s_reshaped = reshape(s, (1, x.shape[1]))
                
            res = sub(x, log_op(s_reshaped))
            ctx.save_for_backward(div(e, s_reshaped))
            return res

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        softmax_x, = ctx.saved_tensors
        if grad_output.device == 'cpu':
            grad_data = grad_output.numpy()
            softmax_data = softmax_x.numpy()
            axis = ctx.axis
            sum_grad = np.sum(grad_data, axis=axis, keepdims=True)
            grad_in = grad_data - softmax_data * sum_grad
            return (Tensor(shape=grad_output.shape, dtype='float32', device='cpu', data=grad_in),)
        else:
            from .ops import mul, sub, transpose, sum_axis, reshape
            axis = ctx.axis
            
            if axis == -1 or axis == 1:
                t = transpose(grad_output)
                s = sum_axis(t, axis=0)
                sum_grad = reshape(s, (grad_output.shape[0], 1))
            elif axis == 0:
                s = sum_axis(grad_output, axis=0)
                sum_grad = reshape(s, (1, grad_output.shape[1]))
            else:
                s = sum_axis(grad_output, axis=0)
                sum_grad = reshape(s, (1, grad_output.shape[1]))
                
            return (sub(grad_output, mul(softmax_x, sum_grad)),)

def log_softmax(x, axis=-1):
    """Numerically stable log-softmax."""
    return LogSoftmaxFunction.apply(x, axis=axis)

class CrossEntropyFunction(Function):
    @staticmethod
    def forward(ctx: Context, predictions: Tensor, targets: Tensor) -> Tensor:
        ctx.save_for_backward(predictions, targets)
        
        if predictions.device == 'cpu':
            from .ops import _require_cpu_data
            pred_data = _require_cpu_data(predictions, 'pred')
            target_data = _require_cpu_data(targets, 'targets').astype(np.int64)
            
            max_val = np.max(pred_data, axis=-1, keepdims=True)
            shifted = pred_data - max_val
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            log_probs = shifted - log_sum_exp
            
            n = pred_data.shape[0]
            loss = -np.mean(log_probs[np.arange(n), target_data])
            
            probs = np.exp(log_probs)
            ctx.probs = Tensor(shape=probs.shape, dtype='float32', device='cpu', data=probs)
            ctx.target_data = target_data
            return Tensor(shape=(), dtype='float32', device='cpu', data=np.array(loss, dtype=np.float32))
        else:
            from .ops import _require_cpu_data, tensor, mul, sum_op, div, neg
            target_data = _require_cpu_data(targets, 'targets').astype(np.int64)
            n, c = predictions.shape
            
            one_hot = np.zeros((n, c), dtype=np.float32)
            one_hot[np.arange(n), target_data] = 1.0
            one_hot_t = tensor(one_hot, device='gpu')
            ctx.one_hot_t = one_hot_t
            
            log_sm = log_softmax(predictions)
            prod = mul(log_sm, one_hot_t)
            s = sum_op(prod)
            
            loss = neg(div(s, tensor(np.array([float(n)], dtype=np.float32), device='gpu')))
            return loss

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, type(None)]:
        if ctx.saved_tensors[0].device == 'cpu':
            probs = ctx.probs.numpy()
            target_data = ctx.target_data
            n = probs.shape[0]
            
            grad_pred = probs.copy()
            grad_pred[np.arange(n), target_data] -= 1.0
            grad_pred /= n
            
            grad_tensor = Tensor(shape=grad_pred.shape, dtype='float32', device='cpu', data=grad_pred.astype(np.float32))
            from .ops import mul
            return mul(grad_output, grad_tensor), None
        else:
            predictions, targets = ctx.saved_tensors
            n = predictions.shape[0]
            
            sm = softmax(predictions)
            from .ops import sub, div, mul, tensor
            import numpy as np
            diff = sub(sm, ctx.one_hot_t)
            grad_pred = div(diff, tensor(np.array([float(n)], dtype=np.float32), device='gpu'))
            
            return mul(grad_output, grad_pred), None

def cross_entropy(predictions, targets):
    """Cross-entropy loss. predictions: (N, C), targets: (N,) integer class indices."""
    return CrossEntropyFunction.apply(predictions, targets)

def mse_loss(predictions, targets):
    """Mean Squared Error loss."""
    diff = sub(predictions, targets)
    sq = mul(diff, diff)
    return mean_op(sq)

def batch_norm2d(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5):
    from .ops import sub, mul, div, add, reshape, mean_axis, tensor, sqrt, full
    import numpy as np
    
    if training:
        m_c = mean_axis(mean_axis(mean_axis(x, 0), 1), 1)
        m_view = reshape(m_c, (1, x.shape[1], 1, 1))
        
        diff = sub(x, m_view)
        diff_sq = mul(diff, diff)
        v_c = mean_axis(mean_axis(mean_axis(diff_sq, 0), 1), 1)
        v_view = reshape(v_c, (1, x.shape[1], 1, 1))
        
        n = x.shape[0] * x.shape[2] * x.shape[3]
        if x.device == 'cpu':
            unbiased_v = v_c._data * (n / (n - 1)) if n > 1 else v_c._data
            running_mean._data = (1 - momentum) * running_mean._data + momentum * m_c._data
            running_var._data = (1 - momentum) * running_var._data + momentum * unbiased_v
        else:
            new_rm = add(mul(running_mean, full(running_mean.shape, 1 - momentum, device='gpu')), mul(m_c, full(m_c.shape, momentum, device='gpu')))
            unbiased_v = mul(v_c, full(v_c.shape, n / (n - 1) if n > 1 else 1.0, device='gpu'))
            new_rv = add(mul(running_var, full(running_var.shape, 1 - momentum, device='gpu')), mul(unbiased_v, full(unbiased_v.shape, momentum, device='gpu')))
            running_mean.__dict__.update(new_rm.__dict__)
            running_var.__dict__.update(new_rv.__dict__)
            
        mean_use, var_use = m_view, v_view
    else:
        mean_use = reshape(running_mean, (1, x.shape[1], 1, 1))
        var_use = reshape(running_var, (1, x.shape[1], 1, 1))
        
    eps_t = full(var_use.shape, eps, device=x.device)
    denom = sqrt(add(var_use, eps_t))
    x_norm = div(sub(x, mean_use), denom)
    
    w_view = reshape(weight, (1, x.shape[1], 1, 1))
    b_view = reshape(bias, (1, x.shape[1], 1, 1))
    
    out = add(mul(x_norm, w_view), b_view)
    return out

def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    from .ops import sub, mul, div, add, mean_axis, full, sqrt, unsqueeze
    dim = -1
    
    m = mean_axis(x, dim)
    m_view = unsqueeze(m, dim)
    
    diff = sub(x, m_view)
    diff_sq = mul(diff, diff)
    
    v = mean_axis(diff_sq, dim)
    v_view = unsqueeze(v, dim)
    
    eps_t = full(v_view.shape, eps, device=x.device)
    denom = sqrt(add(v_view, eps_t))
    x_norm = div(diff, denom)
    
    out = x_norm
    if weight is not None:
        out = mul(out, weight)
    if bias is not None:
        out = add(out, bias)
        
    return out

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, training=False):
    from .ops import bmm, transpose, div, full, reshape, dropout
    import math
    
    orig_shape = query.shape
    if len(orig_shape) == 4:
        B, H, L, D = orig_shape
        query = reshape(query, (B * H, L, D))
        key = reshape(key, (B * H, key.shape[2], D))
        value = reshape(value, (B * H, value.shape[2], value.shape[3]))
        
    d_k = query.shape[-1]
    key_t = transpose(key)
    
    scores = bmm(query, key_t)
    scores = div(scores, full(scores.shape, math.sqrt(d_k), device=query.device))
    
    attn = softmax(scores, axis=-1)
    
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p, training)
        
    out = bmm(attn, value)
    
    if len(orig_shape) == 4:
        out = reshape(out, orig_shape)
        
    return out
