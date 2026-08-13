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
            from .ops import exp_op, div, transpose, sum_axis, reshape
            # 무엇을: GPU 텐서에 대해 exp를 취한다.
            # 왜: softmax 공식 분자를 계산하기 위함이다.
            # 어떻게: exp_op를 호출한다.
            e = exp_op(x)
            
            if axis == -1 or axis == 1:
                # 무엇을: 전치 연산 후 합을 구하고 형태를 맞춘다.
                # 왜: GPU 커널이 특정 차원의 리덕션만 지원할 경우 이를 우회하기 위함이다.
                # 어떻게: transpose -> sum_axis -> reshape 순으로 호출한다.
                t = transpose(e)
                s = sum_axis(t, axis=0)
                s_reshaped = reshape(s, (x.shape[0], 1))
            elif axis == 0:
                # 무엇을: 첫 번째 축에 대해 합을 구한다.
                # 왜: 배치 차원 등 지정된 축 방향으로 정규화하기 위함이다.
                # 어떻게: sum_axis 후 reshape한다.
                s = sum_axis(e, axis=0)
                s_reshaped = reshape(s, (1, x.shape[1]))
            else:
                # 무엇을: 그 외 축에 대해 합을 구한다.
                # 왜: 기본적으로 axis=0과 유사하게 동작하도록 폴백(fallback) 처리한다.
                # 어떻게: sum_axis 후 브로드캐스팅을 위해 reshape한다.
                s = sum_axis(e, axis=0)
                s_reshaped = reshape(s, (1, x.shape[1]))
                
            # 무엇을: 분자(e)를 분모(s_reshaped)로 나눈다.
            # 왜: 최종 softmax 확률을 구하기 위해서다.
            # 어떻게: div 연산을 사용한다.
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
            res_data = res.numpy()
            grad_data = grad_output.numpy()
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
            from .ops import mul, sub, transpose, sum_axis, reshape
            axis = ctx.axis
            
            # 무엇을: 출력 그래디언트와 순전파 결과를 원소별로 곱한다.
            # 왜: 그래디언트 합(sum)을 구하기 위한 중간 단계이다.
            # 어떻게: mul 연산을 사용한다.
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
                
            # 무엇을: grad_output에서 sum_val을 뺀다.
            # 왜: Softmax 야코비안 수식의 괄호 안 부분을 계산하기 위함이다.
            # 어떻게: sub 연산을 사용한다.
            diff = sub(grad_output, sum_val)
            return (mul(res, diff),)

def softmax(x, axis=-1):
    """
    Numerically stable softmax.
    
    무엇을: Softmax 연산을 수행하는 래퍼(wrapper) 함수이다.
    왜: 사용자가 Function.apply를 직접 호출하지 않고 직관적으로 함수를 사용할 수 있도록 하기 위함이다.
    어떻게: SoftmaxFunction.apply를 호출하여 텐서를 넘긴다.
    """
    return SoftmaxFunction.apply(x, axis=axis)

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
            from .ops import _require_cpu_data, tensor, mul, sum_op, div, neg
            target_data = _require_cpu_data(targets, 'targets').astype(np.int64)
            n, c = predictions.shape
            
            # 무엇을: 타겟 레이블을 원-핫 인코딩(One-hot encoding) 벡터로 변환한다.
            # 왜: GPU 텐서에서는 인덱싱 연산이 번거로우므로, 원-핫 행렬과의 내적(mul -> sum)으로 NLLLoss를 대체하기 위함이다.
            # 어떻게: numpy로 0 행렬을 만들고 정답 위치에 1.0을 넣은 뒤 GPU 텐서로 올린다.
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
        """
        무엇을: Cross Entropy 연산의 역전파를 수행한다.
        왜: 손실에 대한 모델 예측값의 미분값을 구하여 네트워크 파라미터를 업데이트하기 위해서다.
        어떻게: 수식 `(probs - one_hot) / N` 을 적용한다. 타겟 텐서는 미분 불가능하므로 None을 반환한다.
        """
        if ctx.saved_tensors[0].device == 'cpu':
            probs = ctx.probs.numpy()
            target_data = ctx.target_data
            n = probs.shape[0]
            
            # 무엇을: 그래디언트를 구한다.
            # 왜: CE loss와 Softmax가 결합된 야코비안은 단순히 정답 위치의 확률에서 1을 빼는 것으로 단순화되기 때문이다.
            # 어떻게: 정답 인덱스에서 1.0을 빼고 배치 크기(n)로 나눈다.
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
    """
    Cross-entropy loss. predictions: (N, C), targets: (N,) integer class indices.
    
    무엇을: 크로스 엔트로피 함수 래퍼.
    왜: 사용자 편의성을 위해 제공.
    어떻게: CrossEntropyFunction.apply 호출.
    """
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
        else:
            new_rm = add(mul(running_mean, full(running_mean.shape, 1 - momentum, device='gpu')), mul(m_c, full(m_c.shape, momentum, device='gpu')))
            unbiased_v = mul(v_c, full(v_c.shape, n / (n - 1) if n > 1 else 1.0, device='gpu'))
            new_rv = add(mul(running_var, full(running_var.shape, 1 - momentum, device='gpu')), mul(unbiased_v, full(unbiased_v.shape, momentum, device='gpu')))
            # 무엇을: 기존 텐서 인스턴스에 새로운 상태를 덮어씌운다.
            # 왜: call-by-reference로 넘어온 인자의 실제 데이터를 갱신해야 하기 때문이다.
            # 어떻게: __dict__.update() 매직 메서드를 이용해 객체 상태를 통째로 교체한다.
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

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, training=False):
    """
    무엇을: 스케일드 닷 프로덕트 어텐션(Scaled Dot-Product Attention)을 계산한다.
    왜: 트랜스포머 구조에서 토큰 간의 연관성(Attention weight)을 구하고 정보를 집계하기 위함이다.
    어떻게: Q와 K의 전치를 내적하고 스케일링한 후, Softmax를 통과시켜 V와 가중합을 계산한다.
    """
    from .ops import bmm, transpose, div, full, reshape, dropout
    import math
    
    orig_shape = query.shape
    if len(orig_shape) == 4:
        B, H, L, D = orig_shape
        # 무엇을: 배치와 헤드 차원을 하나로 합친다.
        # 왜: bmm(Batch Matrix Multiplication)을 3차원 텐서에 대해 쉽게 적용하기 위함이다.
        # 어떻게: reshape 연산을 통해 (B*H, L, D) 형태로 변환한다.
        query = reshape(query, (B * H, L, D))
        key = reshape(key, (B * H, key.shape[2], D))
        value = reshape(value, (B * H, value.shape[2], value.shape[3]))
        
    d_k = query.shape[-1]
    query_t = query
    key_t = permute(key, (0, 2, 1)) if len(key.shape) == 3 else permute(key, (0, 1, 3, 2))
    
    # 무엇을: Q와 K^T의 내적을 통해 어텐션 스코어를 구한다.
    # 왜: 토큰 간의 유사도를 측정하기 위함이다.
    # 어떻게: 차원에 따라 bmm 또는 matmul을 사용한다.
    scores = bmm(query_t, key_t) if len(query_t.shape) == 3 else matmul(query_t, key_t)
    
    # 무엇을: 스코어를 sqrt(d_k)로 나눈다.
    # 왜: 차원이 클수록 내적값이 커져 Softmax 기울기가 소실되는 것을 방지하기 위한 스케일링 작업이다.
    # 어떻게: full 텐서를 만들고 div 연산을 적용한다.
    scores = div(scores, full(scores.shape, math.sqrt(d_k), device=query.device))
    
    attn = softmax(scores, axis=-1)
    
    if dropout_p > 0.0:
        attn = dropout(attn, dropout_p, training)
        
    out = bmm(attn, value)
    
    if len(orig_shape) == 4:
        # 무엇을: 출력 형태를 원래의 4차원으로 복구한다.
        # 왜: 다중 헤드 어텐션(Multi-Head Attention)의 다음 단계 처리를 위해 형태를 맞춰주기 위함이다.
        # 어떻게: 저장해둔 orig_shape로 reshape한다.
        out = reshape(out, orig_shape)
        
    return out
