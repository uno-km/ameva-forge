"""
functional.py - AMEVA-Forge 함수형 연산 모음

[역사적 메타데이터]
- Created: Wed Aug 12 12:14:52 2026 +0900 (초기 커밋)
- Modified:
  - Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
  - Wed Aug 12 12:59:35 2026 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
"""
from typing import Tuple
from .tensor import Tensor
from .autograd import Function, Context
from .ops import exp_op, log_op, sum_op, sub, mul, neg, div, mean_op, add, permute
import numpy as np

class SoftmaxFunction(Function):
    """
    무엇을: Softmax 연산과 그 역전파(gradient)를 정의하는 autograd Function 클래스이다.
    왜: 신경망의 출력층 등에서 값들을 확률 분포로 변환하기 위해 필요하며 자동 미분을 지원하기 위해 존재한다.
    어떻게: Function 베이스 클래스를 상속받고, forward와 backward 정적 메서드를 구현하여 연산을 정의한다.
    """
    @staticmethod
    def forward(ctx: Context, x: Tensor, axis: int = -1) -> Tensor:
        """
        무엇을: Softmax의 순전파 연산을 수행한다.
        왜: 입력 텐서 x를 지정된 축(axis)에 대해 확률값(0~1)으로 스케일링하기 위함이다.
        어떻게: 수치적 안정성을 위해 최대값을 빼고 exp를 취한 뒤, 그 합으로 나눈다. CPU와 GPU 디바이스에 따라 분기하여 처리한다.
        """
        # 무엇을: 축 정보를 컨텍스트에 저장한다.
        # 왜: backward 시 동일한 축(axis)을 기준으로 그래디언트를 계산해야 하기 때문이다.
        # 어떻게: ctx.axis 속성에 axis 값을 할당한다.
        ctx.axis = axis
        if x.device == 'cpu':
            from .ops import _require_cpu_data
            data = _require_cpu_data(x, 'x')
            
            # 무엇을: 유한수(Finite) 마스킹 기반 수치 안정화 Softmax를 계산한다.
            # 왜: All-Masked (-inf) 행 입력 시 (-inf - (-inf))로 인한 NaN 폭발을 방지하기 위함이다.
            finite_mask = np.isfinite(data)
            has_finite = np.any(finite_mask, axis=axis, keepdims=True)
            safe_data = np.where(finite_mask, data, -1e30)
            max_val = np.where(has_finite, np.max(safe_data, axis=axis, keepdims=True), 0.0)
            
            exp_data = np.where(finite_mask, np.exp(data - max_val), 0.0)
            sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
            
            # 무엇을: 정규화를 수행한다 (sum_exp == 0인 all-masked 행은 0.0 확률을 반환).
            # 왜: 0 / 0 NaN 발생을 원천 방어하기 위함이다.
            result = np.where(sum_exp > 0.0, exp_data / np.maximum(sum_exp, 1e-12), 0.0)
            
            # 무엇을: 결과 텐서를 컨텍스트에 저장한다.
            # 왜: backward 시 야코비안 계산을 위해 Softmax 결과값이 필요하기 때문이다.
            # 어떻게: ctx.save_for_backward에 텐서를 래핑하여 넘긴다.
            ctx.save_for_backward(Tensor(shape=result.shape, dtype='float32', device='cpu', data=result))
            return Tensor(shape=result.shape, dtype='float32', device='cpu', data=result)
        else:
            from .ops import exp_op, div, sub, sum_axis, max_axis, reshape
            norm_axis = axis if axis >= 0 else axis + len(x.shape)
            
            # 수치 안정성 (Numerical Stability): x - max(x) <= 0
            m = max_axis(x, axis=norm_axis)
            s_shape = list(x.shape)
            s_shape[norm_axis] = 1
            m_reshaped = reshape(m, tuple(s_shape))
            shifted_x = sub(x, m_reshaped)

            # 무엇을: 수치 안정화된 텐서에 대해 exp를 취한다 (exp(x - max(x)) <= 1.0, overflow 불가).
            e = exp_op(shifted_x)
            
            # 무엇을: N차원 범용 축소(sum_axis)를 수행하고 브로드캐스팅 형태로 변환한다.
            s = sum_axis(e, axis=norm_axis)
            s_reshaped = reshape(s, tuple(s_shape))
                
            # 무엇을: 분자(e)를 분모(s_reshaped)로 나눈다.
            res = div(e, s_reshaped)
            ctx.save_for_backward(res)
            return res

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """
        무엇을: Softmax의 역전파 연산을 수행한다.
        왜: 출력에 대한 그래디언트(grad_output)를 받아 입력(x)에 대한 그래디언트를 체인 룰을 통해 계산하기 위함이다.
        어떻게: 수식 `grad_in = res * (grad_out - sum(grad_out * res))`을 CPU와 GPU 각각의 방식으로 구현한다.
        """
        # 무엇을: 순전파에서 저장한 결과를 꺼낸다.
        # 왜: 역전파 수식 계산에 사용하기 위해서다.
        # 어떻게: 언패킹을 통해 할당한다.
        res, = ctx.saved_tensors
        if res.device == 'cpu':
            from .ops import _require_cpu_data
            res_data = _require_cpu_data(res, 'res')
            grad_data = _require_cpu_data(grad_output, 'grad_output')
            axis = ctx.axis
            
            # 무엇을: 그래디언트와 결과의 내적(sum)을 구한다.
            # 왜: 야코비안 행렬과 벡터의 곱셈을 스칼라 형태로 최적화하여 풀기 위함이다.
            # 어떻게: 원소별 곱 후 축(axis)에 대해 sum을 취한다.
            sum_val = np.sum(grad_data * res_data, axis=axis, keepdims=True)
            
            # 무엇을: 최종 입력 그래디언트를 구한다.
            # 왜: 이전 계층으로 오차를 전파하기 위해서다.
            # 어떻게: res * (grad - sum_val) 식을 numpy로 계산한다.
            grad_in = res_data * (grad_data - sum_val)
            return (Tensor(shape=res.shape, dtype='float32', device='cpu', data=grad_in),)
        else:
            from .ops import mul, sub, sum_axis, reshape
            norm_axis = ctx.axis if ctx.axis >= 0 else ctx.axis + len(res.shape)
            
            # 무엇을: 출력 그래디언트와 순전파 결과를 원소별로 곱한다.
            # 왜: 그래디언트 합(sum)을 구하기 위한 중간 단계이다.
            # 어떻게: mul 연산을 사용한다.
            m = mul(grad_output, res)
            
            s = sum_axis(m, axis=norm_axis)
            s_shape = list(res.shape)
            s_shape[norm_axis] = 1
            sum_val = reshape(s, tuple(s_shape))
                
            # 무엇을: grad_output에서 sum_val을 뺀다.
            # 왜: Softmax 야코비안 수식의 괄호 안 부분을 계산하기 위함이다.
            # 어떻게: sub 연산을 사용한다.
            diff = sub(grad_output, sum_val)
            return (mul(res, diff),)

def softmax(x, axis=None, dim=None):
    """
    Numerically stable softmax. Supports both PyTorch (dim) and NumPy (axis) naming.
    """
    if axis is None:
        axis = -1 if dim is None else dim
    return SoftmaxFunction.apply(x, axis=axis)

def log_softmax(x, axis=None, dim=None):
    """
    Numerically stable log_softmax. Supports both PyTorch (dim) and NumPy (axis) naming.
    """
    if axis is None:
        axis = -1 if dim is None else dim
    return LogSoftmaxFunction.apply(x, axis=axis)

class LogSoftmaxFunction(Function):
    """
    무엇을: Log-Softmax 연산과 그 역전파를 정의하는 클래스이다.
    왜: Softmax 후 Log를 취하는 것보다 수치적으로 훨씬 안정적이고 빠르기 때문이다.
    어떻게: Function을 상속받아 forward/backward를 구현한다.
    """
    @staticmethod
    def forward(ctx: Context, x: Tensor, axis: int = -1) -> Tensor:
        """
        무엇을: LogSoftmax의 순전파 연산을 수행한다.
        왜: 입력 텐서를 log 확률 형태로 안정적으로 변환하기 위함이다.
        어떻게: x - log(sum(exp(x - max))) 공식을 적용한다.
        """
        ctx.axis = axis
        if x.device == 'cpu':
            from .ops import _require_cpu_data
            data = _require_cpu_data(x, 'x')
            
            # 무엇을: 유한수(Finite) 마스킹 기반 Log-Sum-Exp 수치 안정화를 수행한다.
            # 왜: All-Masked (-inf) 행 입력 시 NaN 폭발을 방지하기 위함이다.
            finite_mask = np.isfinite(data)
            has_finite = np.any(finite_mask, axis=axis, keepdims=True)
            safe_data = np.where(finite_mask, data, -1e30)
            max_val = np.where(has_finite, np.max(safe_data, axis=axis, keepdims=True), 0.0)
            
            shifted = data - max_val
            exp_shifted = np.where(finite_mask, np.exp(shifted), 0.0)
            sum_exp = np.sum(exp_shifted, axis=axis, keepdims=True)
            log_sum_exp = np.where(sum_exp > 0.0, np.log(np.maximum(sum_exp, 1e-12)), 0.0)
            result = np.where(finite_mask, shifted - log_sum_exp, -1e30)
            
            # 무엇을: backward를 위해 softmax 확률을 저장한다.
            softmax_prob = np.where(finite_mask, np.exp(result), 0.0)
            ctx.save_for_backward(Tensor(shape=result.shape, dtype='float32', device='cpu', data=softmax_prob))
            return Tensor(shape=result.shape, dtype='float32', device='cpu', data=result)
        else:
            from .ops import exp_op, div, sum_axis, max_axis, reshape, log_op, sub
            norm_axis = axis if axis >= 0 else axis + len(x.shape)
            
            # 수치 안정성 (Numerical Stability): x - max(x) - log(sum(exp(x - max(x))))
            m = max_axis(x, axis=norm_axis)
            s_shape = list(x.shape)
            s_shape[norm_axis] = 1
            m_reshaped = reshape(m, tuple(s_shape))
            shifted_x = sub(x, m_reshaped)
            
            e = exp_op(shifted_x)
            s = sum_axis(e, axis=norm_axis)
            s_reshaped = reshape(s, tuple(s_shape))
            log_sum = log_op(s_reshaped)
            
            result = sub(shifted_x, log_sum)
            # backward를 위해 softmax 확률 저장
            res = div(e, s_reshaped)
            ctx.save_for_backward(res)
            return result

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        """
        무엇을: LogSoftmax의 역전파 연산을 수행한다.
        왜: 출력 오차를 입력으로 전파하기 위함이다.
        어떻게: 수식 `grad_in = grad_out - softmax(x) * sum(grad_out)`을 적용한다.
        """
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
    """
    Numerically stable log-softmax.
    
    무엇을: Log-Softmax를 호출하는 사용자 편의 함수이다.
    왜: 직접 LogSoftmaxFunction.apply를 타이핑하는 수고를 덜어주기 위함이다.
    어떻게: 함수 내부에서 apply를 위임 호출한다.
    """
    return LogSoftmaxFunction.apply(x, axis=axis)

class CrossEntropyFunction(Function):
    """
    무엇을: 교차 엔트로피 손실(Cross Entropy Loss) 연산 및 역전파 클래스이다.
    왜: 분류 문제에서 모델의 예측 분포와 정답 레이블 간의 차이를 측정하여 손실을 구하기 위해 존재한다.
    어떻게: 예측값에 Log-Softmax를 취한 뒤, 정답 레이블에 해당하는 인덱스의 값을 추출해 평균을 낸다.
    """
    @staticmethod
    def forward(ctx: Context, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        무엇을: Cross Entropy 손실값을 계산한다.
        왜: 모델 학습의 최적화 목표가 되는 단일 스칼라 손실(loss)을 얻기 위함이다.
        어떻게: 입력(predictions)을 수치 안정적 방식으로 처리하고, 정답 타겟 위치의 음의 로그 확률 평균을 구한다.
        """
        ctx.save_for_backward(predictions, targets)
        
        if predictions.device == 'cpu':
            from .ops import _require_cpu_data
            pred_data = _require_cpu_data(predictions, 'pred')
            target_data = _require_cpu_data(targets, 'targets').astype(np.int64)
            
            # 1. 수치 안정화 LogSoftmax (Finite-Masked)
            finite_mask = np.isfinite(pred_data)
            has_finite = np.any(finite_mask, axis=-1, keepdims=True)
            safe_data = np.where(finite_mask, pred_data, -1e30)
            max_val = np.where(has_finite, np.max(safe_data, axis=-1, keepdims=True), 0.0)
            
            shifted = pred_data - max_val
            exp_shifted = np.where(finite_mask, np.exp(shifted), 0.0)
            sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
            log_sum_exp = np.where(sum_exp > 0.0, np.log(np.maximum(sum_exp, 1e-12)), 0.0)
            log_probs = np.where(finite_mask, shifted - log_sum_exp, -1e30)
            
            n = pred_data.shape[0]
            # 2. ignore_index=-100 및 범위 유효성 마스킹
            valid_mask = (target_data != -100) & (target_data >= 0) & (target_data < pred_data.shape[-1])
            valid_count = max(int(np.sum(valid_mask)), 1)
            
            safe_targets = np.where(valid_mask, target_data, 0)
            selected_log_probs = log_probs[np.arange(n), safe_targets]
            masked_loss = np.where(valid_mask, -selected_log_probs, 0.0)
            loss = np.sum(masked_loss) / valid_count
            
            probs = np.where(finite_mask, np.exp(log_probs), 0.0)
            ctx.probs = Tensor(shape=probs.shape, dtype='float32', device='cpu', data=probs)
            ctx.target_data = target_data
            ctx.valid_mask = valid_mask
            ctx.valid_count = valid_count
            return Tensor(shape=(), dtype='float32', device='cpu', data=np.array(loss, dtype=np.float32))
        else:
            from .ops import tensor, sum_op, div, _require_cpu_data
            if targets.device == 'cpu':
                target_data = _require_cpu_data(targets, 'targets').astype(np.int32)
                targets_gpu = tensor(target_data, dtype='int32', device='gpu', requires_grad=False)
            else:
                targets_gpu = targets

            n, c = predictions.shape
            ctx.batch_size = float(n)
            ctx.num_classes = c
            ctx.targets_gpu = targets_gpu

            # WebGPU Native Sparse Cross-Entropy Forward: O(N) memory, No Dense One-Hot!
            loss_per_sample = Tensor(
                shape=(n,),
                dtype='float32',
                device='gpu',
                op='sparse_cross_entropy',
                parents=(predictions, targets_gpu),
                op_params=[c, -100, 0, 0],
                requires_grad=predictions.requires_grad
            )
            
            # Mean reduction over batch
            total_loss = sum_op(loss_per_sample)
            return div(total_loss, tensor(float(n), device=predictions.device, requires_grad=False))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, None]:
        """
        무엇을: Cross Entropy 손실 함수의 역전파를 수행한다.
        왜: 소프트맥스와 NLLLoss의 결합 도함수인 (probs - one_hot) / N 을 통해 입력 로짓의 그래디언트를 구하기 위함이다.
        어떻게: GPU 환경에서는 융합 sparse_cross_entropy_backward 커널을, CPU에서는 NumPy 역전파를 적용한다.
        """
        predictions, targets = ctx.saved_tensors
        if predictions.device == 'cpu':
            probs_data = ctx.probs.numpy()
            target_data = ctx.target_data
            valid_mask = ctx.valid_mask
            valid_count = ctx.valid_count
            n = probs_data.shape[0]
            
            grad_pred = probs_data.copy()
            safe_targets = np.where(valid_mask, target_data, 0)
            grad_pred[np.arange(n), safe_targets] -= 1.0
            
            # ignore_index=-100 토큰 위치의 기울기를 0으로 무효화
            grad_pred = np.where(valid_mask[:, None], grad_pred, 0.0)
            grad_pred = grad_pred / valid_count
            
            if grad_output.shape != ():
                grad_pred = grad_pred * grad_output.numpy()
            else:
                grad_pred = grad_pred * float(grad_output.numpy())
            return (Tensor(shape=grad_pred.shape, dtype='float32', device='cpu', data=grad_pred), None)
        else:
            targets_gpu = ctx.targets_gpu
            n = ctx.batch_size
            reduction_scale = 1.0 / n

            # WebGPU Native Sparse Cross-Entropy Backward: 1-Pass Fused Gradient
            grad_logits = Tensor(
                shape=predictions.shape,
                dtype='float32',
                device='gpu',
                op='sparse_cross_entropy_backward',
                parents=(predictions, targets_gpu, grad_output),
                op_params=[-100, reduction_scale]
            )
            return (grad_logits, None)

def cross_entropy(predictions, targets):
    """
    WHAT: 크로스 엔트로피 손실을 계산하며, 1D/2D/3D 정수 라벨 및 확률 분포(Soft Target)를 모두 지원합니다.
    WHY: 일반 분류뿐만 아니라 LLM 넥스트 토큰 예측(3D shape: [B, T, V]), Label Smoothing, Knowledge Distillation을 완벽 지원하기 위함입니다.
    """
    # 3D LLM Sequence Logits [B, T, C]와 [B, T] Targets 처리
    if len(predictions.shape) == 3 and len(targets.shape) == 2:
        from .ops import reshape
        B, T, C = predictions.shape
        flat_preds = reshape(predictions, (B * T, C))
        flat_targets = reshape(targets, (B * T,))
        return CrossEntropyFunction.apply(flat_preds, flat_targets)

    if len(targets.shape) == len(predictions.shape) and len(targets.shape) == 2:
        from .ops import mul, sum_axis, mean_op, neg
        log_p = log_softmax(predictions, axis=-1)
        loss_unreduced = neg(sum_axis(mul(targets, log_p), axis=-1))
        return mean_op(loss_unreduced)
    return CrossEntropyFunction.apply(predictions, targets)

def mse_loss(predictions, targets):
    """
    Mean Squared Error loss.
    
    무엇을: 평균 제곱 오차(MSE)를 계산한다.
    왜: 회귀(Regression) 문제 등에서 두 텐서 간의 값 차이를 손실로 산출하기 위함이다.
    어떻게: 예측값에서 정답값을 뺀 후(sub), 그 결과를 제곱하고(mul), 전체 평균(mean_op)을 구한다.
    """
    diff = sub(predictions, targets)
    sq = mul(diff, diff)
    return mean_op(sq)

def binary_cross_entropy_with_logits(input, target, weight=None, reduction: str = 'mean', pos_weight=None):
    """Measures Binary Cross Entropy with logits."""
    from .ops import binary_cross_entropy_with_logits as ops_bce
    return ops_bce(input, target, weight=weight, reduction=reduction, pos_weight=pos_weight)

def smooth_l1_loss(input, target, beta: float = 1.0, reduction: str = 'mean'):
    """Smooth L1 / Huber Loss."""
    from .ops import smooth_l1_loss as ops_smooth_l1
    return ops_smooth_l1(input, target, beta=beta, reduction=reduction)

def kl_div(input, target, reduction: str = 'mean', log_target: bool = False):
    """Kullback-Leibler divergence loss."""
    from .ops import kl_div as ops_kl_div
    return ops_kl_div(input, target, reduction=reduction, log_target=log_target)

def l1_loss(input, target, reduction: str = 'mean'):
    """L1 / MAE loss."""
    from .ops import l1_loss as ops_l1
    return ops_l1(input, target, reduction=reduction)

def cosine_similarity(x1, x2, dim: int = 1, eps: float = 1e-8):
    """
    무엇을: 두 텐서 x1과 x2 간의 코사인 유사도(Cosine Similarity)를 계산한다.
    왜: 임베딩 벡터 간 유사도 측정, 벡터 검색(RAG), 대조 학습(Contrastive Learning)을 지원하기 위함이다.
    어떻게: dim 축에 대해 내적을 구하고 L2 노름 제곱에 eps^2 클램프를 적용한 후 sqrt로 나누어 0-벡터 미분 NaN을 영구 방어한다.
    """
    from .ops import sum_axis, mul, sqrt, div, maximum, full
    dot = sum_axis(mul(x1, x2), axis=dim)
    w1 = sum_axis(mul(x1, x1), axis=dim)
    w2 = sum_axis(mul(x2, x2), axis=dim)
    
    eps_sq1 = full(w1.shape, eps * eps, device=x1.device)
    eps_sq2 = full(w2.shape, eps * eps, device=x2.device)
    norm1 = sqrt(maximum(w1, eps_sq1))
    norm2 = sqrt(maximum(w2, eps_sq2))
    
    denom = mul(norm1, norm2)
    return div(dot, denom)

def gelu(x, approximate: str = "none"):
    """Applies Gaussian Error Linear Units."""
    from .ops import gelu as ops_gelu
    return ops_gelu(x, approximate=approximate)

def silu(x):
    """Applies Sigmoid Linear Unit (Swish)."""
    from .ops import silu as ops_silu
    return ops_silu(x)

def leaky_relu(x, negative_slope: float = 0.01):
    """Applies LeakyReLU."""
    from .ops import leaky_relu as ops_leaky_relu
    return ops_leaky_relu(x, negative_slope=negative_slope)

def elu(x, alpha: float = 1.0):
    """Applies Exponential Linear Unit."""
    from .ops import elu as ops_elu
    return ops_elu(x, alpha=alpha)

def pad(input, pad, mode: str = 'constant', value: float = 0.0):
    """Pads tensor."""
    from .ops import pad as ops_pad
    return ops_pad(input, pad, mode=mode, value=value)

def adaptive_avg_pool2d(input, output_size):
    """Applies a 2D adaptive average pooling."""
    from .ops import adaptive_avg_pool2d as ops_adaptive_avg_pool2d
    return ops_adaptive_avg_pool2d(input, output_size)

def adaptive_max_pool2d(input, output_size):
    """Applies a 2D adaptive max pooling."""
    from .ops import adaptive_max_pool2d as ops_adaptive_max_pool2d
    return ops_adaptive_max_pool2d(input, output_size)

def conv1d(input, weight, bias=None, stride: int = 1, padding: int = 0):
    """Applies 1D convolution."""
    from .ops import conv1d as ops_conv1d
    return ops_conv1d(input, weight, bias=bias, stride=stride, padding=padding)

def pixel_shuffle(input, upscale_factor: int):
    """Pixel shuffle."""
    from .ops import pixel_shuffle as ops_pixel_shuffle
    return ops_pixel_shuffle(input, upscale_factor)

def pixel_unshuffle(input, downscale_factor: int):
    """Pixel unshuffle."""
    from .ops import pixel_unshuffle as ops_pixel_unshuffle
    return ops_pixel_unshuffle(input, downscale_factor)

def interpolate(input, size=None, scale_factor=None, mode: str = 'nearest'):
    """Interpolate."""
    from .ops import interpolate as ops_interpolate
    return ops_interpolate(input, size=size, scale_factor=scale_factor, mode=mode)

def _move_tensor_state(dst, src) -> None:
    """
    WHAT: src 텐서의 상태와 지연 연산 그래프를 dst 텐서로 안전하게 이동(Move)합니다.
    WHY: BatchNorm의 running_mean/running_var 같은 in-place 통계량 갱신 시,
         src의 식별자/그래프/데이터 소유권을 dst로 이전하여 dst 객체의 참조 동일성을 유지하기 위함입니다.
    HOW: 기존 dst GPU 버퍼 안전 해제 -> src의 _HandleCell 소유권 단일 인계 -> src 필드 None 초기화.
    """
    from .tensor import _gc_queue

    # 1. 기존 dst의 GPU 버퍼 참조 해제
    old_cell = getattr(dst, "_handle_cell", None)
    if dst.device == "gpu" and old_cell is not None and old_cell is not getattr(src, "_handle_cell", None):
        if old_cell.dec_ref() and old_cell.handle is not None:
            _gc_queue.add(old_cell.handle)

    # 2. src 버퍼 소유권을 dst로 단일 이전
    dst._data = src._data
    dst._handle_cell = getattr(src, "_handle_cell", None)
    
    if dst.device == "gpu" and dst._handle_cell is not None:
        if not getattr(dst, "_finalizer_registered", False):
            import weakref
            from .tensor import Tensor
            weakref.finalize(dst, Tensor._finalize_buffer, dst._handle_cell)
            dst._finalizer_registered = True

    dst._lazy_op = getattr(src, "_lazy_op", None)
    dst._op = getattr(src, "_op", None)
    dst._parents = getattr(src, "_parents", ())
    dst._op_params = getattr(src, "_op_params", None)

    dst.shape = src.shape
    dst.dtype = src.dtype
    dst.device = src.device
    dst.requires_grad = False
    dst.grad = None
    dst._disposed = False
    dst._version += 1

    # 3. src는 소유권을 dst에 완전히 넘겼으므로 None 초기화
    src._data = None
    src._handle_cell = None

def batch_norm2d(x, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-5):
    """
    무엇을: 2D 배치 정규화(Batch Normalization)를 수행한다.
    왜: 채널(Channel) 차원을 기준으로 정규화하여 학습 안정성을 부여한다.
    어떻게: weight/bias=None (affine=False) 및 running_stats=None (track_running_stats=False)을 완벽 지원한다.
    """
    from .ops import sub, mul, div, add, reshape, mean_axis, tensor, sqrt, full
    from .errors import AMEVAForgeShapeError
    import numpy as np
    
    if len(x.shape) != 4:
        raise AMEVAForgeShapeError(f"batch_norm2d expected 4D input [Batch, Channels, Height, Width], but got shape {x.shape}")
        
    num_channels = x.shape[1]
    
    if training or running_mean is None or running_var is None:
        m_c = mean_axis(mean_axis(mean_axis(x, 0), 1), 1)
        m_view = reshape(m_c, (1, num_channels, 1, 1))
        
        diff = sub(x, m_view)
        diff_sq = mul(diff, diff)
        v_c = mean_axis(mean_axis(mean_axis(diff_sq, 0), 1), 1)
        v_view = reshape(v_c, (1, num_channels, 1, 1))
        
        n = x.shape[0] * x.shape[2] * x.shape[3]
        if training and running_mean is not None and running_var is not None:
            if x.device == 'cpu':
                unbiased_v = v_c._data * (n / (n - 1)) if n > 1 else v_c._data
                running_mean._data = (1 - momentum) * running_mean._data + momentum * m_c._data
                running_var._data = (1 - momentum) * running_var._data + momentum * unbiased_v
                running_mean._version += 1
                running_var._version += 1
            else:
                old_rm = Tensor(
                    shape=running_mean.shape,
                    dtype=running_mean.dtype,
                    device=running_mean.device,
                    handle=running_mean._handle,
                    data=running_mean._data,
                    op=running_mean._lazy_op,
                    parents=running_mean._parents,
                    op_params=running_mean._lazy_params,
                    handle_cell=getattr(running_mean, "_handle_cell", None),
                )
                old_rv = Tensor(
                    shape=running_var.shape,
                    dtype=running_var.dtype,
                    device=running_var.device,
                    handle=running_var._handle,
                    data=running_var._data,
                    op=running_var._lazy_op,
                    parents=running_var._parents,
                    op_params=running_var._lazy_params,
                    handle_cell=getattr(running_var, "_handle_cell", None),
                )
                new_rm = add(mul(old_rm, full(running_mean.shape, 1 - momentum, device=x.device)), mul(m_c, full(m_c.shape, momentum, device=x.device)))
                unbiased_v = mul(v_c, full(v_c.shape, n / (n - 1) if n > 1 else 1.0, device=x.device))
                new_rv = add(mul(old_rv, full(running_var.shape, 1 - momentum, device=x.device)), mul(unbiased_v, full(unbiased_v.shape, momentum, device=x.device)))
                _move_tensor_state(running_mean, new_rm)
                _move_tensor_state(running_var, new_rv)
                
        mean_use, var_use = m_view, v_view
    else:
        mean_use = reshape(running_mean, (1, num_channels, 1, 1))
        var_use = reshape(running_var, (1, num_channels, 1, 1))
        
    eps_t = full(var_use.shape, eps, device=x.device)
    denom = sqrt(add(var_use, eps_t))
    out = div(sub(x, mean_use), denom)
    
    if weight is not None:
        w_view = reshape(weight, (1, num_channels, 1, 1))
        out = mul(out, w_view)
        
    if bias is not None:
        b_view = reshape(bias, (1, num_channels, 1, 1))
        out = add(out, b_view)
        
    return out

def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """
    무엇을: 레이어 정규화(Layer Normalization)를 수행한다.
    왜: 트랜스포머(Transformer), Vision 모델 등에서 지정된 차원들(normalized_shape) 전체에 걸쳐 스케일을 맞추기 위함이다.
    어떻게: normalized_shape에 해당하는 모든 후방 차원 축들에 대해 평균과 분산을 구하여 정규화한 뒤 아핀(affine) 변환을 수행한다.
    """
    from .ops import sub, mul, div, add, mean_axis, full, sqrt, unsqueeze
    if isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)

    ndim_norm = len(normalized_shape)
    norm_axes = list(range(len(x.shape) - ndim_norm, len(x.shape))) if len(x.shape) >= ndim_norm else [-1]

    # 1. 다차원 정규화 축에 대한 평균(Mean) 계산 (keepdim 형태 유지)
    m = x
    for ax in sorted(norm_axes, reverse=True):
        m = mean_axis(m, ax)
        m = unsqueeze(m, ax)

    diff = sub(x, m)
    diff_sq = mul(diff, diff)

    # 2. 다차원 정규화 축에 대한 분산(Variance) 계산 (keepdim 형태 유지)
    v = diff_sq
    for ax in sorted(norm_axes, reverse=True):
        v = mean_axis(v, ax)
        v = unsqueeze(v, ax)

    eps_t = full(v.shape, eps, device=x.device)
    denom = sqrt(add(v, eps_t))
    x_norm = div(diff, denom)

    out = x_norm
    if weight is not None:
        out = mul(out, weight)
    if bias is not None:
        out = add(out, bias)

    return out

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, training=False):
    """
    무엇을: 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention)을 계산한다.
    왜: 트랜스포머 구조에서 토큰 간의 연관성(Attention weight)을 구하고 정보를 집계하기 위함이다.
    어떻게: Q와 K의 전치를 내적하고 스케일링한 후, Softmax를 통과시켜 V와 가중합을 계산한다.
            추론(requires_grad=False) 시 1-Pass WebGPU FlashAttention을 디스패치하고,
            학습(requires_grad=True) 시 Autograd DAG를 정상 구성하여 역전파 미분 소실을 방지한다.
    """
    from .ops import bmm, transpose, div, full, reshape, dropout, permute, matmul, add, mul
    from .autograd import is_grad_enabled
    import math
    
    orig_shape = query.shape
    d_k = orig_shape[-1]
    effective_scale = scale if scale is not None else (1.0 / math.sqrt(d_k))
    needs_grad = is_grad_enabled() and (query.requires_grad or key.requires_grad or value.requires_grad)
    
    # 1. 추론 전용 (Inference / no_grad): 초고속 1-Pass Fused WebGPU FlashAttention 커널 디스패치
    if not needs_grad and len(orig_shape) == 4 and query.device == 'gpu' and key.device == 'gpu' and value.device == 'gpu' and dropout_p == 0.0 and attn_mask is None:
        from .tensor import Tensor
        B, H, N_q, d = query.shape
        H_kv = key.shape[1]
        N_kv = key.shape[2]
        return Tensor(
            shape=query.shape,
            dtype=query.dtype,
            device='gpu',
            op='flash_attention',
            parents=(query, key, value),
            op_params=[float(H_kv), float(effective_scale), 1.0 if is_causal else 0.0, float(N_kv)],
            requires_grad=False
        )
            
    # 2. 학습 모드 및 범용 Autograd 체인 (PyTorch 표준 역전파 지원)
    if len(orig_shape) == 4:
        B, H, L, D = orig_shape
        query = reshape(query, (B * H, L, D))
        key = reshape(key, (B * H, key.shape[2], D))
        value = reshape(value, (B * H, value.shape[2], value.shape[3]))
        if attn_mask is not None:
            from .ops import tensor
            if len(attn_mask.shape) == 4:
                m_B, m_H, m_L, m_S = attn_mask.shape
                m_data = attn_mask.numpy() if hasattr(attn_mask, 'numpy') else attn_mask
                if hasattr(m_data, 'dtype') and (m_data.dtype == bool or str(m_data.dtype) == 'bool'):
                    m_data = np.where(m_data, -1e9, 0.0).astype(np.float32)
                if m_H == 1 and H > 1:
                    m_data = np.broadcast_to(m_data, (B, H, m_L, m_S))
                m_data = np.ascontiguousarray(m_data.reshape(B * H, m_L, m_S))
                attn_mask = tensor(m_data, device=query.device)
        
    d_k = query.shape[-1]
    query_t = query
    key_t = permute(key, (0, 2, 1)) if len(key.shape) == 3 else permute(key, (0, 1, 3, 2))
    
    scores = bmm(query_t, key_t) if len(query_t.shape) == 3 else matmul(query_t, key_t)
    scores = mul(scores, full(scores.shape, effective_scale, device=query.device))
    
    if is_causal:
        from .ops import tensor
        L_q, L_k = scores.shape[-2], scores.shape[-1]
        diagonal_offset = (L_k - L_q) + 1
        mask_np = np.triu(np.full((L_q, L_k), -1e4, dtype=np.float32), k=diagonal_offset)
        causal_mask = tensor(mask_np, device=scores.device)
        scores = add(scores, causal_mask)
    elif attn_mask is not None:
        from .ops import tensor
        m_data = attn_mask.numpy() if hasattr(attn_mask, 'numpy') else attn_mask
        if hasattr(m_data, 'dtype') and (m_data.dtype == bool or str(m_data.dtype) == 'bool'):
            float_m = np.where(m_data, -1e9, 0.0).astype(np.float32)
            attn_mask_t = tensor(float_m, device=scores.device)
        else:
            attn_mask_t = attn_mask
        scores = add(scores, attn_mask_t)

    attn = softmax(scores, axis=-1)
    
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p, training)
        
    out = bmm(attn, value)
    
    if len(orig_shape) == 4:
        out = reshape(out, orig_shape)
        
    return out

def rms_norm(x, normalized_shape=None, weight=None, eps=1e-5):
    """
    무엇을: Root Mean Square Normalization (RMSNorm)을 적용한다.
    왜: LayerNorm 대비 평균 감산을 생략하여 계산 비용을 30% 절감하면서도 강력한 학습 안정성을 제공한다.
    어떻게: x / sqrt(mean(x^2, dim=normalized_shape) + eps) * weight
    PyTorch 2.4+ 시그니처 (x, normalized_shape, weight, eps) 및 LLaMA 시그니처 (x, weight, eps)를 모두 자동 수용한다.
    """
    from .tensor import Tensor
    # 시그니처 유연성 처리: 2번째 인자로 weight 텐서가 직접 전달된 경우
    if isinstance(normalized_shape, (Tensor,)):
        eps = 1e-5 if weight is None else float(weight) if isinstance(weight, (int, float)) else eps
        weight = normalized_shape
        normalized_shape = (x.shape[-1],)
    elif normalized_shape is None:
        if weight is not None and isinstance(weight, (int, float)):
            eps = float(weight)
            weight = None
        normalized_shape = (x.shape[-1],)
    elif isinstance(normalized_shape, int):
        normalized_shape = (normalized_shape,)
    else:
        normalized_shape = tuple(normalized_shape)

    from .autograd import is_grad_enabled
    needs_grad = is_grad_enabled() and (x.requires_grad or (weight is not None and weight.requires_grad))

    if not needs_grad and x.device == 'gpu' and len(normalized_shape) == 1:
        parents = (x,) if weight is None else (x, weight)
        return Tensor(shape=x.shape, dtype=x.dtype, device='gpu',
                      op='rmsnorm', parents=parents, op_params=[float(eps)],
                      requires_grad=False)
    
    from .ops import mul, div, add, mean_axis, full, sqrt, unsqueeze
    dims = tuple(range(-len(normalized_shape), 0))
    sq = mul(x, x)
    m = sq
    for d in dims:
        m = mean_axis(m, d)
        
    m_view = m
    for _ in range(len(normalized_shape)):
        m_view = unsqueeze(m_view, -1)
        
    eps_t = full(m_view.shape, float(eps), device=x.device)
    denom = sqrt(add(m_view, eps_t))
    out = div(x, denom)
    if weight is not None:
        out = mul(out, weight)
    return out

def swiglu(gate, up):
    """
    무엇을: SwiGLU 융합 활성화 함수(Swish(x) * y)를 적용한다.
    왜: LLaMA 및 Gemma 등의 최신 FFN 아키텍처 비선형성을 효율적으로 제공하기 위함이다.
    어떻게: (gate * sigmoid(gate)) * up
    """
    from .autograd import is_grad_enabled
    needs_grad = is_grad_enabled() and (gate.requires_grad or up.requires_grad)

    if gate.device == 'gpu' and up.device == 'gpu':
        from .tensor import Tensor
        return Tensor(shape=gate.shape, dtype=gate.dtype, device='gpu',
                      op='swiglu', parents=(gate, up),
                      requires_grad=needs_grad)
            
    from .ops import mul, sigmoid
    swish_g = mul(gate, sigmoid(gate))
    return mul(swish_g, up)

def rope(x, base_freq=10000.0, offset_pos=0):
    """
    무엇을: Rotary Position Embedding (RoPE) 2D 복소수 평면 회전을 수행한다.
    왜: 토큰 위치 정보를 Query와 Key 벡터에 인플레이스로 주입하기 위함이다.
    """
    from .autograd import is_grad_enabled
    needs_grad = is_grad_enabled() and x.requires_grad

    if x.device == 'gpu':
        from .tensor import Tensor
        return Tensor(shape=x.shape, dtype=x.dtype, device='gpu',
                      op='rope', parents=(x,), op_params=[float(base_freq), float(offset_pos)],
                      requires_grad=needs_grad)
    
    # CPU Reference Math
    from .ops import tensor
    data = x.numpy()
    orig_shape = data.shape
    B, H, N, d = orig_shape
    half_d = d // 2
    
    pos = np.arange(offset_pos, offset_pos + N, dtype=np.float32)[:, np.newaxis]
    exponent = -2.0 * np.arange(half_d, dtype=np.float32) / float(d)
    theta = pos * np.power(base_freq, exponent)
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    out_np = np.zeros_like(data)
    for b in range(B):
      for h in range(H):
        v0 = data[b, h, :, 0::2]
        v1 = data[b, h, :, 1::2]
        out_np[b, h, :, 0::2] = v0 * cos_theta - v1 * sin_theta
        out_np[b, h, :, 1::2] = v1 * cos_theta + v0 * sin_theta
        
    return tensor(out_np, device=x.device, dtype=x.dtype, requires_grad=x.requires_grad)


