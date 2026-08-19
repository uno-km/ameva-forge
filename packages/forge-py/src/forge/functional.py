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
            # 무엇을: CPU 데이터를 가져온다.
            # 왜: numpy 연산을 수행하기 위함이다.
            # 어떻게: _require_cpu_data를 호출한다.
            data = _require_cpu_data(x, 'x')
            
            # 무엇을: 해당 축에서 최대값을 찾는다.
            # 왜: exp() 계산 시 오버플로우(overflow)를 방지하기 위한 수치적 안정화 기법이다.
            # 어떻게: np.max를 사용한다.
            max_val = np.max(data, axis=axis, keepdims=True)
            
            # 무엇을: 지수 함수를 취한다.
            # 왜: 각 원소를 양수로 만들고 크기에 비례하게 증폭시키기 위함이다.
            # 어떻게: 데이터에서 최대값을 뺀 후 np.exp를 호출한다.
            exp_data = np.exp(data - max_val)
            
            # 무엇을: 지수 값들의 합을 구한다.
            # 왜: 확률 분포로 정규화하기 위한 분모를 얻기 위해서다.
            # 어떻게: np.sum을 사용한다.
            sum_exp = np.sum(exp_data, axis=axis, keepdims=True)
            
            # 무엇을: 정규화를 수행한다.
            # 왜: 합이 1이 되는 확률 값을 얻기 위함이다.
            # 어떻게: exp_data를 sum_exp로 나눈다.
            result = exp_data / sum_exp
            
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
            max_val = np.max(data, axis=axis, keepdims=True)
            shifted = data - max_val
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
            result = shifted - log_sum_exp
            
            # 무엇을: backward를 위해 softmax 확률을 저장한다.
            # 왜: log-softmax의 미분 시 exp(log_softmax) 즉 softmax 결과값이 필요하기 때문이다.
            # 어떻게: np.exp(result)를 취한 후 텐서화하여 저장한다.
            ctx.save_for_backward(Tensor(shape=result.shape, dtype='float32', device='cpu', data=np.exp(result)))
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
            
            max_val = np.max(pred_data, axis=-1, keepdims=True)
            shifted = pred_data - max_val
            log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
            log_probs = shifted - log_sum_exp
            
            n = pred_data.shape[0]
            # 무엇을: 배치 내의 정답 레이블에 해당하는 확률만 추출해 평균 음수 값을 취한다.
            # 왜: NLLLoss (Negative Log Likelihood Loss) 연산을 수행하기 위함이다.
            # 어떻게: numpy의 인덱싱 기법(fancy indexing)을 활용한다.
            loss = -np.mean(log_probs[np.arange(n), target_data])
            
            probs = np.exp(log_probs)
            ctx.probs = Tensor(shape=probs.shape, dtype='float32', device='cpu', data=probs)
            ctx.target_data = target_data
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
            n = probs_data.shape[0]
            grad_pred = probs_data.copy()
            grad_pred[np.arange(n), target_data] -= 1.0
            grad_pred = grad_pred / n
            
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
    WHAT: 크로스 엔트로피 손실을 계산하며, 1D 정수 라벨 및 2D 확률 분포(Soft Target)를 모두 지원합니다.
    WHY: 일반 분류뿐만 아니라 Label Smoothing, Knowledge Distillation(지식 증류) 등의 최신 LLM/Vision 학습을 지원하기 위함입니다.
    HOW: targets의 차원이 1D이면 고속 sparse_cross_entropy로, predictions와 동일한 2D이면 Soft Target 공식(-sum(y * log_softmax(z)))으로 디스패치합니다.
    """
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

def _move_tensor_state(dst, src) -> None:
    """
    WHAT: src 텐서의 상태와 지연 연산 그래프를 dst 텐서로 안전하게 이동(Move)합니다.
    WHY: BatchNorm의 running_mean/running_var 같은 in-place 통계량 갱신 시,
         src의 식별자/그래프/데이터 소유권을 dst로 이전하여 dst 객체의 참조 동일성을 유지하기 위함입니다.
    HOW: 기존 dst GPU 버퍼 안전 해제 -> src의 _HandleCell 및 AST 소유권 인계 -> src 필드 None 초기화.
    """
    if dst.device == "gpu" and getattr(dst, "_handle_cell", None) is not None:
        try:
            dst.dispose()
        except Exception:
            pass

    dst._data = src._data
    dst._handle_cell = getattr(src, "_handle_cell", None)
    if dst.device == "gpu" and dst._handle_cell is not None:
        dst._handle_cell.inc_ref()
        import weakref
        from .tensor import Tensor
        weakref.finalize(dst, Tensor._finalize_buffer, dst._handle_cell)

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
    dst._finalizer_registered = (dst.device == "gpu")
    dst._version += 1

    src._data = None
    src._handle_cell = None

def batch_norm2d(x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5):
    """
    무엇을: 2D 배치 정규화(Batch Normalization)를 수행한다.
    왜: 신경망 각 층의 입력을 정규화하여 학습(Internal Covariate Shift 방지)을 안정적이고 빠르게 만들기 위함이다.
    어떻게: 채널(Channel) 차원을 기준으로 배치, 높이, 너비에 대한 평균과 분산을 구하고, 이를 이용해 데이터를 정규화한 뒤 학습 가능한 weight와 bias를 적용한다.
    """
    from .ops import sub, mul, div, add, reshape, mean_axis, tensor, sqrt, full
    import numpy as np
    
    if training:
        # 무엇을: 배치 차원(0)과 공간 차원(2, 3)을 순차적으로 평균 내어 채널별 평균을 구한다.
        # 왜: 채널 단위의 분포 통계량을 얻기 위함이다.
        # 어떻게: mean_axis를 연쇄 호출한다.
        m_c = mean_axis(mean_axis(mean_axis(x, 0), 1), 1)
        m_view = reshape(m_c, (1, x.shape[1], 1, 1))
        
        diff = sub(x, m_view)
        diff_sq = mul(diff, diff)
        v_c = mean_axis(mean_axis(mean_axis(diff_sq, 0), 1), 1)
        v_view = reshape(v_c, (1, x.shape[1], 1, 1))
        
        n = x.shape[0] * x.shape[2] * x.shape[3]
        if x.device == 'cpu':
            unbiased_v = v_c._data * (n / (n - 1)) if n > 1 else v_c._data
            # 무엇을: 이동 평균(running stats)을 업데이트한다.
            # 왜: 추론(Inference) 시 현재 배치가 아닌 전체 데이터셋의 통계량을 사용하기 위해 모멘텀을 적용해 누적하기 위함이다.
            # 어떻게: 지수 이동 평균(EMA) 수식을 적용한다.
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
    """
    무엇을: 레이어 정규화(Layer Normalization)를 수행한다.
    왜: 트랜스포머(Transformer) 등에서 시퀀스나 토큰 단위로 데이터의 스케일을 맞춰주기 위함이다.
    어떻게: 가장 마지막 차원(dim=-1)을 기준으로 평균과 분산을 구하여 정규화한 뒤 아핀(affine) 변환을 수행한다.
    """
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

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, training=False):
    """
    무엇을: 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention)을 계산한다.
    왜: 트랜스포머 구조에서 토큰 간의 연관성(Attention weight)을 구하고 정보를 집계하기 위함이다.
    어떻게: Q와 K의 전치를 내적하고 스케일링한 후, Softmax를 통과시켜 V와 가중합을 계산한다.
            추론(requires_grad=False) 시 1-Pass WebGPU FlashAttention을 디스패치하고,
            학습(requires_grad=True) 시 Autograd DAG를 정상 구성하여 역전파 미분 소실을 방지한다.
    """
    from .ops import bmm, transpose, div, full, reshape, dropout, permute, matmul, add, mul
    from .autograd import _grad_mode
    import math
    
    orig_shape = query.shape
    d_k = orig_shape[-1]
    effective_scale = scale if scale is not None else (1.0 / math.sqrt(d_k))
    needs_grad = _grad_mode and (query.requires_grad or key.requires_grad or value.requires_grad)
    
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
        scores = add(scores, attn_mask)

    attn = softmax(scores, axis=-1)
    
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p, training)
        
    out = bmm(attn, value)
    
    if len(orig_shape) == 4:
        out = reshape(out, orig_shape)
        
    return out

def rms_norm(x, weight=None, eps=1e-5):
    """
    무엇을: Root Mean Square Normalization (RMSNorm)을 적용한다.
    왜: LayerNorm 대비 평균 계산을 생략하여 추론 및 학습 처리 속도를 20~30% 가속한다.
    어떻게: x / sqrt(mean(x^2) + eps) * weight
    """
    from .autograd import _grad_mode
    needs_grad = _grad_mode and (x.requires_grad or (weight is not None and weight.requires_grad))

    if not needs_grad and x.device == 'gpu':
        from .tensor import Tensor
        parents = (x,) if weight is None else (x, weight)
        return Tensor(shape=x.shape, dtype=x.dtype, device='gpu',
                      op='rmsnorm', parents=parents, op_params=[float(eps)],
                      requires_grad=False)
    
    from .ops import sub, mul, div, add, mean_axis, full, sqrt, unsqueeze
    sq = mul(x, x)
    m = mean_axis(sq, -1)
    m_view = unsqueeze(m, -1)
    eps_t = full(m_view.shape, eps, device=x.device)
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
    from .autograd import _grad_mode
    needs_grad = _grad_mode and (gate.requires_grad or up.requires_grad)

    if not needs_grad and gate.device == 'gpu' and up.device == 'gpu':
        from .tensor import Tensor
        return Tensor(shape=gate.shape, dtype=gate.dtype, device='gpu',
                      op='swiglu', parents=(gate, up),
                      requires_grad=False)
            
    from .ops import mul, sigmoid_op
    swish_g = mul(gate, sigmoid_op(gate))
    return mul(swish_g, up)

def rope(x, base_freq=10000.0, offset_pos=0):
    """
    무엇을: Rotary Position Embedding (RoPE) 2D 복소수 평면 회전을 수행한다.
    왜: 토큰 위치 정보를 Query와 Key 벡터에 인플레이스로 주입하기 위함이다.
    """
    from .autograd import _grad_mode
    needs_grad = _grad_mode and x.requires_grad

    if not needs_grad and x.device == 'gpu':
        from .tensor import Tensor
        return Tensor(shape=x.shape, dtype=x.dtype, device='gpu',
                      op='rope', parents=(x,), op_params=[float(base_freq), float(offset_pos)],
                      requires_grad=False)
    
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


