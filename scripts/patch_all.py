import re

def patch_ops():
    with open('packages/forge-py/src/forge/ops.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # We need to add the new functions at the end of the file.
    # But first, we replace Add, Mul, Sub, Div
    
    # Replacement for AddFunction
    add_target = '''class AddFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "add")
        if a.shape != b.shape:
            raise AMEVAForgeShapeError(
                f"Cannot add tensors with different shapes: {a.shape} and {b.shape}. "
                f"Broadcasting is not supported. Reshape tensors to match."
            )
            
        if _should_use_gpu(a, b):
            return Tensor(shape=a.shape, dtype="float32", device="gpu",
                          op='add', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a + data_b
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output'''
        
    add_replace = '''class AddFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "add")
        out_shape = _broadcast_shapes(a.shape, b.shape)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
            
        if _should_use_gpu(a, b):
            return Tensor(shape=out_shape, dtype="float32", device="gpu",
                          op='add', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a + data_b
            return Tensor(shape=out_shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return _unbroadcast(grad_output, ctx.a_shape), _unbroadcast(grad_output, ctx.b_shape)'''

    content = content.replace(add_target, add_replace)

    # Replacement for MulFunction
    mul_target = '''class MulFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "mul")
        if a.shape != b.shape:
            raise AMEVAForgeShapeError(
                f"Cannot multiply tensors with different shapes: {a.shape} and {b.shape}. "
                f"Broadcasting is not supported. Reshape tensors to match."
            )
            
        if _should_use_gpu(a, b):
            return Tensor(shape=a.shape, dtype="float32", device="gpu",
                          op='mul', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a * data_b
            return Tensor(shape=res.shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        return mul(grad_output, b), mul(grad_output, a)'''

    mul_replace = '''class MulFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, "mul")
        out_shape = _broadcast_shapes(a.shape, b.shape)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
            
        if _should_use_gpu(a, b):
            return Tensor(shape=out_shape, dtype="float32", device="gpu",
                          op='mul', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            res = data_a * data_b
            return Tensor(shape=out_shape, dtype="float32", device="cpu", data=res)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        return _unbroadcast(mul(grad_output, b), ctx.a_shape), _unbroadcast(mul(grad_output, a), ctx.b_shape)'''

    content = content.replace(mul_target, mul_replace)

    # SubFunction
    sub_target = '''class SubFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, 'sub')
        if a.shape != b.shape:
            raise AMEVAForgeShapeError(
                f"Cannot subtract tensors with different shapes: {a.shape} and {b.shape}. "
                f"Broadcasting is not supported. Reshape tensors to match."
            )
        if _should_use_gpu(a, b):
            return Tensor(shape=a.shape, dtype='float32', device='gpu', op='sub', parents=(a, b))
        else:
            return Tensor(shape=a.shape, dtype='float32', device='cpu', data=_require_cpu_data(a, "a") - _require_cpu_data(b, "b"))
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, neg(grad_output)'''
    
    sub_replace = '''class SubFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, 'sub')
        out_shape = _broadcast_shapes(a.shape, b.shape)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        if _should_use_gpu(a, b):
            return Tensor(shape=out_shape, dtype='float32', device='gpu', op='sub', parents=(a, b))
        else:
            return Tensor(shape=out_shape, dtype='float32', device='cpu', data=_require_cpu_data(a, "a") - _require_cpu_data(b, "b"))
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return _unbroadcast(grad_output, ctx.a_shape), _unbroadcast(neg(grad_output), ctx.b_shape)'''

    content = content.replace(sub_target, sub_replace)

    # DivFunction
    div_target = '''class DivFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, 'div')
        if a.shape != b.shape:
            raise AMEVAForgeShapeError(
                f"Cannot divide tensors with different shapes: {a.shape} and {b.shape}. "
                f"Broadcasting is not supported. Reshape tensors to match."
            )
        if _should_use_gpu(a, b):
            return Tensor(shape=a.shape, dtype='float32', device='gpu', op='div', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            return Tensor(shape=a.shape, dtype='float32', device='cpu', data=data_a / data_b)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        grad_a = div(grad_output, b)
        grad_b = neg(mul(div(grad_output, mul(b, b)), a))  # -grad * a / b^2
        return grad_a, grad_b'''

    div_replace = '''class DivFunction(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        _ensure_same_device(a, b, 'div')
        out_shape = _broadcast_shapes(a.shape, b.shape)
        ctx.a_shape = a.shape
        ctx.b_shape = b.shape
        if _should_use_gpu(a, b):
            return Tensor(shape=out_shape, dtype='float32', device='gpu', op='div', parents=(a, b))
        else:
            data_a = _require_cpu_data(a, "a")
            data_b = _require_cpu_data(b, "b")
            return Tensor(shape=out_shape, dtype='float32', device='cpu', data=data_a / data_b)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_tensors
        grad_a = div(grad_output, b)
        grad_b = neg(mul(div(grad_output, mul(b, b)), a))  # -grad * a / b^2
        return _unbroadcast(grad_a, ctx.a_shape), _unbroadcast(grad_b, ctx.b_shape)'''

    content = content.replace(div_target, div_replace)


    new_ops = '''

def _broadcast_shapes(a_shape, b_shape):
    ndim = max(len(a_shape), len(b_shape))
    a_padded = (1,) * (ndim - len(a_shape)) + a_shape
    b_padded = (1,) * (ndim - len(b_shape)) + b_shape
    result = []
    for ad, bd in zip(a_padded, b_padded):
        if ad == bd:
            result.append(ad)
        elif ad == 1:
            result.append(bd)
        elif bd == 1:
            result.append(ad)
        else:
            raise AMEVAForgeShapeError(f"Cannot broadcast {a_shape} and {b_shape}")
    return tuple(result)

def _unbroadcast(grad, target_shape):
    if grad.shape == target_shape:
        return grad
    # Sum along broadcast dimensions
    ndim_diff = len(grad.shape) - len(target_shape)
    padded = (1,) * ndim_diff + target_shape
    
    # For CPU: use numpy directly
    if grad.device == 'cpu':
        data = _require_cpu_data(grad, 'grad')
        axes = []
        for i, (g, t) in enumerate(zip(grad.shape, padded)):
            if t == 1 and g != 1:
                axes.append(i)
        for i in range(ndim_diff):
            axes.append(i) if i not in axes else None
        axes = sorted(set(axes))
        result = np.sum(data, axis=tuple(axes), keepdims=True)
        result = result.reshape(target_shape)
        return Tensor(shape=target_shape, dtype='float32', device='cpu', data=result)
    else:
        # For GPU: compose with sum_axis ops
        # Simplified: flatten dims and use sum
        # For now, fall back to composition
        result = grad
        for i in range(ndim_diff):
            result = sum_axis(result, axis=0)
        padded_target = (1,) * ndim_diff + target_shape
        for i in range(len(padded_target)):
            if padded_target[i] == 1 and result.shape[i] != 1:
                # Need to reduce this axis too
                pass  # handled by initial reduction
        return reshape(result, target_shape) if result.shape != target_shape else result

class SumFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        ctx.input_shape = x.shape
        if _should_use_gpu(x):
            return Tensor(shape=(), dtype='float32', device='gpu', op='sum', parents=(x,))
        else:
            return Tensor(shape=(), dtype='float32', device='cpu',
                         data=np.array(np.sum(_require_cpu_data(x, 'x')), dtype=np.float32))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # grad of sum is ones_like(input) * grad_output
        return (mul(grad_output, ones(x.shape, device=x.device)),)

def sum_op(x): return SumFunction.apply(x)


class MeanFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        n = 1
        for d in x.shape:
            n *= d
        ctx.numel = n
        if _should_use_gpu(x):
            # mean = sum / n, compose with existing ops
            s = sum_op(x)
            return div(s, tensor(np.array([float(n)], dtype=np.float32)))
        else:
            return Tensor(shape=(), dtype='float32', device='cpu',
                         data=np.array(np.mean(_require_cpu_data(x, 'x')), dtype=np.float32))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        n = ctx.numel
        return (mul(grad_output, full(x.shape, 1.0/n, device=x.device)),)

def mean_op(x): return MeanFunction.apply(x)


class ExpFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='exp', parents=(x,))
        else:
            result = Tensor(shape=x.shape, dtype='float32', device='cpu',
                          data=np.exp(_require_cpu_data(x, 'x')))
        ctx.save_for_backward(result)  # save output for backward
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        exp_x, = ctx.saved_tensors
        return (mul(grad_output, exp_x),)  # d/dx exp(x) = exp(x)

def exp_op(x): return ExpFunction.apply(x)


class LogFunction(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        if _should_use_gpu(x):
            return Tensor(shape=x.shape, dtype='float32', device='gpu', op='log', parents=(x,))
        else:
            return Tensor(shape=x.shape, dtype='float32', device='cpu',
                         data=np.log(_require_cpu_data(x, 'x')))
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return (div(grad_output, x),)  # d/dx log(x) = 1/x

def log_op(x): return LogFunction.apply(x)


class SigmoidFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='sigmoid', parents=(x,))
        else:
            data = 1.0 / (1.0 + np.exp(-_require_cpu_data(x, 'x')))
            result = Tensor(shape=x.shape, dtype='float32', device='cpu', data=data)
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        sig, = ctx.saved_tensors
        # sigmoid'(x) = sig * (1 - sig)
        one_minus = sub(ones(sig.shape, device=sig.device), sig)
        return (mul(grad_output, mul(sig, one_minus)),)

def sigmoid(x): return SigmoidFunction.apply(x)


class TanhFunction(Function):
    @staticmethod
    def forward(ctx, x):
        if _should_use_gpu(x):
            result = Tensor(shape=x.shape, dtype='float32', device='gpu', op='tanh', parents=(x,))
        else:
            result = Tensor(shape=x.shape, dtype='float32', device='cpu',
                          data=np.tanh(_require_cpu_data(x, 'x')))
        ctx.save_for_backward(result)
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        tanh_x, = ctx.saved_tensors
        # tanh'(x) = 1 - tanh²(x)
        one_minus_sq = sub(ones(tanh_x.shape, device=tanh_x.device), mul(tanh_x, tanh_x))
        return (mul(grad_output, one_minus_sq),)

def tanh_op(x): return TanhFunction.apply(x)


class ReshapeFunction(Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.original_shape = x.shape
        if x.device == 'gpu':
            # For GPU: create a new Tensor with same handle but different shape
            # This is a metadata-only operation
            return Tensor(shape=new_shape, dtype=x.dtype, device='gpu', op='reshape', parents=(x,),
                         op_params=list(new_shape))
        else:
            data = _require_cpu_data(x, 'x').reshape(new_shape)
            return Tensor(shape=new_shape, dtype='float32', device='cpu', data=data)
    
    @staticmethod
    def backward(ctx, grad_output):
        return (reshape(grad_output, ctx.original_shape),)

def reshape(x, new_shape):
    if isinstance(new_shape, list):
        new_shape = tuple(new_shape)
    return ReshapeFunction.apply(x, new_shape)


class SumAxisFunction(Function):
    @staticmethod
    def forward(ctx, x, axis):
        ctx.save_for_backward(x)
        ctx.axis = axis
        ctx.input_shape = x.shape
        
        if x.device == 'gpu':
            # GPU: emit sum_axis lazy node
            new_shape = list(x.shape)
            del new_shape[axis]
            new_shape = tuple(new_shape) if new_shape else ()
            return Tensor(shape=new_shape, dtype='float32', device='gpu',
                         op='sum_axis', parents=(x,), op_params=[x.shape[axis], 
                         1 if len(x.shape) == 1 else (x.shape[1] if axis == 0 else x.shape[0])])
        else:
            data = np.sum(_require_cpu_data(x, 'x'), axis=axis)
            return Tensor(shape=data.shape, dtype='float32', device='cpu', data=data)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Expand grad back to original shape
        # This is the reverse: unsqueeze + broadcast
        if grad_output.device == 'cpu':
            data = _require_cpu_data(grad_output, 'grad')
            expanded = np.expand_dims(data, axis=ctx.axis)
            tiled = np.broadcast_to(expanded, ctx.input_shape).copy()
            return (Tensor(shape=ctx.input_shape, dtype='float32', device='cpu', data=tiled),)
        return (grad_output,)  # simplified for now

def sum_axis(x, axis=0): return SumAxisFunction.apply(x, axis)

def randn(shape, device='cpu', dtype='float32', requires_grad=False):
    arr = np.random.randn(*shape).astype(np.float32)
    if device == 'cpu':
        return Tensor(shape=shape, dtype=dtype, device=device, requires_grad=requires_grad, data=arr)
    else:
        return Tensor(shape=shape, dtype=dtype, device='gpu', requires_grad=requires_grad, data=arr, op='upload')
'''
    
    content += new_ops
    with open('packages/forge-py/src/forge/ops.py', 'w', encoding='utf-8') as f:
        f.write(content)


def patch_tensor():
    with open('packages/forge-py/src/forge/tensor.py', 'r', encoding='utf-8') as f:
        content = f.read()

    tensor_methods = '''

    def sum(self):
        self._check_disposed()
        from .ops import sum_op
        return sum_op(self)

    def mean(self):
        self._check_disposed()
        from .ops import mean_op
        return mean_op(self)

    def reshape(self, *shape):
        self._check_disposed()
        from .ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def view(self, *shape):
        return self.reshape(*shape)

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def data(self):
        return self

    @data.setter
    def data(self, new_tensor):
        # In-place replacement for optimizer updates
        self._data = new_tensor._data if hasattr(new_tensor, '_data') else None
        self.shape = new_tensor.shape
        self._handle = getattr(new_tensor, '_handle', None)
        self._parents = ()
        self._grad_fn = None
        self.grad = None
        
    def exp(self):
        self._check_disposed()
        from .ops import exp_op
        return exp_op(self)
        
    def log(self):
        self._check_disposed()
        from .ops import log_op
        return log_op(self)
        
    def sigmoid(self):
        self._check_disposed()
        from .ops import sigmoid
        return sigmoid(self)
        
    def tanh(self):
        self._check_disposed()
        from .ops import tanh_op
        return tanh_op(self)
'''

    # Inject these methods before def __repr__
    content = content.replace('    def __repr__(self) -> str:', tensor_methods + '\\n    def __repr__(self) -> str:')
    
    with open('packages/forge-py/src/forge/tensor.py', 'w', encoding='utf-8') as f:
        f.write(content)


def patch_init():
    with open('packages/forge-py/src/forge/__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()

    new_imports = '''from .ops import sum_op as sum, mean_op as mean, exp_op as exp, log_op as log
from .ops import sigmoid, tanh_op as tanh, reshape, sum_axis
from . import nn
from . import optim  
from . import functional as F
from .data import DataLoader
'''
    
    # We append the imports right after __version__
    content = content.replace('__version__ = "0.1.0"', '__version__ = "0.1.0"\\n\\n' + new_imports)
    
    # Update __all__
    # we just append them to __all__ list
    extra_all = '''    "sum",
    "mean",
    "exp",
    "log",
    "sigmoid",
    "tanh",
    "reshape",
    "sum_axis",
    "nn",
    "optim",
    "F",
    "DataLoader",
'''
    content = content.replace('    "__version__",\\n]', '    "__version__",\\n' + extra_all + ']')
    
    with open('packages/forge-py/src/forge/__init__.py', 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    patch_ops()
    patch_tensor()
    patch_init()
    print("Patching complete.")
