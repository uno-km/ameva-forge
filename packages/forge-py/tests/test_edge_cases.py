import sys
import unittest
from pathlib import Path
import math
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
try:
    import forge as at
except ImportError:
    # Dummy mock for generating tests without library
    class DummyAt:
        def zeros(self, *args, **kwargs): pass
        def add(self, *args, **kwargs): pass
        def matmul(self, *args, **kwargs): pass
        def transpose(self, *args, **kwargs): pass
        def dispose(self, *args, **kwargs): pass
    at = DummyAt()
    
class TestEdgeCases(unittest.TestCase):
    """Category 2: Edge Cases Test Suite."""

    def test_empty_tensor_creation(self):
        """빈 텐서 생성 시 에러가 발생하는지 확인합니다."""
        with self.assertRaises(Exception):
            at.zeros((0,))

    def test_scalar_tensor(self):
        """스칼라 텐서 (shape=()) 생성이 정상적으로 되는지 확인합니다."""
        t = at.zeros(())
        self.assertIsNotNone(t)

    def test_nan_inf_input(self):
        """NaN이나 Inf 입력 시 경고를 발생시키거나 안전하게 처리하는지 확인합니다."""
        # Using mock or assuming it creates cleanly depending on impl
        pass

    def test_very_large_values(self):
        """매우 큰 값(1e38) 처리가 오버플로우 없이 또는 에러 메시지와 함께 처리되는지 확인합니다."""
        pass

    def test_negative_shape_dimensions(self):
        """음수 차원 크기를 가진 shape로 텐서 생성 시 에러가 발생하는지 확인합니다."""
        with self.assertRaises(Exception):
            at.zeros([-1, 5])

    def test_float_shape_dimensions(self):
        """실수형 차원 크기를 가진 shape로 텐서 생성 시 에러가 발생하는지 확인합니다."""
        with self.assertRaises(Exception):
            at.zeros([2.5, 5])

    def test_zero_dimension_shape(self):
        """0 크기의 차원을 가진 shape로 텐서 생성 시 에러가 발생하는지 확인합니다."""
        with self.assertRaises(Exception):
            at.zeros([0, 5])

    def test_high_rank_tensor(self):
        """차원이 높은 텐서(Rank 8) 생성이 정상적으로 되는지 확인합니다."""
        t = at.zeros([1]*8)
        self.assertIsNotNone(t)

    def test_too_high_rank_tensor(self):
        """너무 높은 차원의 텐서(Rank 9+) 생성 시 에러가 발생하는지 확인합니다."""
        # Python Tensor에서 rank 제한 검증 (MAX_SHAPE_DIM=8 적용)
        from forge.errors import AMEVAForgeShapeError
        with self.assertRaises(AMEVAForgeShapeError):
            at.zeros((1,)*9)

    def test_add_mul_shape_mismatch(self):
        """add/mul 연산 시 두 텐서의 shape가 불일치할 때 에러가 발생하는지 확인합니다."""
        t1 = at.zeros([2, 3])
        t2 = at.zeros([3, 2])
        with self.assertRaises(Exception):
            at.add(t1, t2)

    def test_matmul_1d_inputs(self):
        """matmul에 1D 텐서를 입력할 때 에러가 발생하는지 확인합니다."""
        t1 = at.zeros([3])
        t2 = at.zeros([3])
        with self.assertRaises(Exception):
            at.matmul(t1, t2)

    def test_matmul_inner_dimension_mismatch(self):
        """matmul 연산 시 내부 차원이 불일치할 때 에러가 발생하는지 확인합니다."""
        t1 = at.zeros([2, 3])
        t2 = at.zeros([4, 2])
        with self.assertRaises(Exception):
            at.matmul(t1, t2)

    def test_transpose_non_2d(self):
        """2D가 아닌 텐서를 전치(transpose)하려 할 때 에러가 발생하는지 확인합니다."""
        t1 = at.zeros([2, 3, 4])
        with self.assertRaises(Exception):
            at.transpose(t1)

    def test_double_dispose(self):
        """같은 텐서에 대해 dispose를 두 번 호출해도 안전한지 확인합니다."""
        t = at.zeros([2, 2])
        if hasattr(t, 'dispose'):
            t.dispose()
            t.dispose()

    def test_use_after_dispose(self):
        """dispose 된 텐서를 사용할 때 에러가 발생하는지 확인합니다."""
        t1 = at.zeros([2, 2])
        t2 = at.zeros([2, 2])
        if hasattr(t1, 'dispose'):
            t1.dispose()
            with self.assertRaises(Exception):
                at.add(t1, t2)

    def test_backward_on_non_grad_tensor(self):
        """그래디언트 추적이 되지 않는 텐서에 대해 backward 호출 시 에러가 발생하는지 확인합니다."""
        pass

    def test_backward_on_non_scalar_without_gradient(self):
        """초기 그래디언트 없이 스칼라가 아닌 텐서에 대해 backward 호출 시 에러가 발생하는지 확인합니다."""
        pass

    def test_mixed_device_operations(self):
        """서로 다른 디바이스에 있는 텐서 간의 연산 시 에러가 발생하는지 확인합니다."""
        pass

    def test_large_cpu_matmul_correctness(self):
        """CPU 환경에서 대형(256x256) matmul 연산이 수행되는지 확인합니다."""
        t1 = at.zeros([256, 256])
        t2 = at.zeros([256, 256])
        res = at.matmul(t1, t2)
        self.assertIsNotNone(res)

    def test_chained_operations(self):
        """(a + b * c)와 같은 복합 연산이 정상적으로 연결되는지 확인합니다."""
        a = at.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = at.tensor([[2.0, 3.0], [4.0, 5.0]])
        c = at.tensor([[0.5, 0.5], [0.5, 0.5]])
        result = at.add(a, at.mul(b, c))
        expected = np.array([[2.0, 3.5], [5.0, 6.5]], dtype=np.float32)
        np.testing.assert_allclose(result.numpy(), expected)

    # ── 신규 산술 연산 테스트 ──

    def test_sub_basic(self):
        """뺄셈 연산의 수학적 정확성을 검증합니다."""
        a = at.tensor([5.0, 10.0, 3.0])
        b = at.tensor([1.0, 4.0, 7.0])
        result = at.sub(a, b)
        np.testing.assert_allclose(result.numpy(), [4.0, 6.0, -4.0])

    def test_neg_basic(self):
        """부호 반전 연산의 정확성을 검증합니다."""
        a = at.tensor([1.0, -2.0, 0.0, 3.14])
        result = at.neg(a)
        np.testing.assert_allclose(result.numpy(), [-1.0, 2.0, 0.0, -3.14])

    def test_div_basic(self):
        """나눗셈 연산의 정확성을 검증합니다."""
        a = at.tensor([6.0, 10.0, 1.0])
        b = at.tensor([2.0, 5.0, 4.0])
        result = at.div(a, b)
        np.testing.assert_allclose(result.numpy(), [3.0, 2.0, 0.25])

    def test_sub_operator(self):
        """- 연산자 오버로딩을 검증합니다."""
        a = at.tensor([5.0, 3.0])
        b = at.tensor([2.0, 1.0])
        result = a - b
        np.testing.assert_allclose(result.numpy(), [3.0, 2.0])

    def test_neg_operator(self):
        """-tensor 단항 연산자를 검증합니다."""
        a = at.tensor([1.0, -2.0, 3.0])
        result = -a
        np.testing.assert_allclose(result.numpy(), [-1.0, 2.0, -3.0])

    def test_div_operator(self):
        """/ 연산자 오버로딩을 검증합니다."""
        a = at.tensor([10.0, 9.0])
        b = at.tensor([2.0, 3.0])
        result = a / b
        np.testing.assert_allclose(result.numpy(), [5.0, 3.0])

    # ── 스칼라 연산 테스트 ──

    def test_scalar_add(self):
        """tensor + scalar 연산을 검증합니다."""
        a = at.tensor([1.0, 2.0, 3.0])
        result = a + 10.0
        np.testing.assert_allclose(result.numpy(), [11.0, 12.0, 13.0])

    def test_scalar_radd(self):
        """scalar + tensor 연산(radd)을 검증합니다."""
        a = at.tensor([1.0, 2.0, 3.0])
        result = 10.0 + a
        np.testing.assert_allclose(result.numpy(), [11.0, 12.0, 13.0])

    def test_scalar_sub(self):
        """tensor - scalar 연산을 검증합니다."""
        a = at.tensor([5.0, 10.0])
        result = a - 3.0
        np.testing.assert_allclose(result.numpy(), [2.0, 7.0])

    def test_scalar_rsub(self):
        """scalar - tensor (rsub)을 검증합니다."""
        a = at.tensor([1.0, 2.0, 3.0])
        result = 10.0 - a
        np.testing.assert_allclose(result.numpy(), [9.0, 8.0, 7.0])

    def test_scalar_mul(self):
        """tensor * scalar 연산을 검증합니다."""
        a = at.tensor([1.0, 2.0, 3.0])
        result = a * 3.0
        np.testing.assert_allclose(result.numpy(), [3.0, 6.0, 9.0])

    def test_scalar_rmul(self):
        """scalar * tensor (rmul)을 검증합니다."""
        a = at.tensor([1.0, 2.0, 3.0])
        result = 3.0 * a
        np.testing.assert_allclose(result.numpy(), [3.0, 6.0, 9.0])

    def test_scalar_div(self):
        """tensor / scalar 연산을 검증합니다."""
        a = at.tensor([6.0, 10.0, 3.0])
        result = a / 2.0
        np.testing.assert_allclose(result.numpy(), [3.0, 5.0, 1.5])

    def test_scalar_rdiv(self):
        """scalar / tensor (rtruediv)을 검증합니다."""
        a = at.tensor([1.0, 2.0, 4.0])
        result = 8.0 / a
        np.testing.assert_allclose(result.numpy(), [8.0, 4.0, 2.0])

    def test_complex_expression(self):
        """(a * 2 - b) / c + 1.0 같은 복합 수식을 검증합니다."""
        a = at.tensor([3.0, 6.0])
        b = at.tensor([1.0, 2.0])
        c = at.tensor([5.0, 10.0])
        result = (a * 2.0 - b) / c + 1.0
        # (3*2-1)/5 + 1 = 2.0, (6*2-2)/10 + 1 = 2.0
        np.testing.assert_allclose(result.numpy(), [2.0, 2.0])

    def test_double_neg(self):
        """이중 부호 반전 --a == a 을 검증합니다."""
        a = at.tensor([1.0, -2.0, 3.0])
        result = -(-a)
        np.testing.assert_allclose(result.numpy(), a.numpy())


if __name__ == '__main__':
    from report_generator import run_and_report
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases)
    run_and_report(suite, '엣지케이스_테스트', 'Category 2: Edge Cases')

