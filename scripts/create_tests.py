import os

tests_dir = r"c:\Users\GAME\Desktop\uno-km\dev\AMEVA-Forge\tests"
os.makedirs(tests_dir, exist_ok=True)

header = """import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor
"""

files = {}

files["test_tensor_creation.py"] = header + """
class TestTensorCreation(unittest.TestCase):
    def test_tensor_from_list(self):
        t = at.tensor([1, 2, 3])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.device, "cpu")
        np.testing.assert_array_equal(t.numpy(), np.array([1, 2, 3], dtype=np.float32))

    def test_tensor_from_numpy(self):
        arr = np.array([[1, 2], [3, 4]])
        t = at.tensor(arr)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_array_equal(t.numpy(), arr)
        
    def test_zeros(self):
        t = at.zeros((2, 3))
        self.assertEqual(t.shape, (2, 3))
        np.testing.assert_array_equal(t.numpy(), np.zeros((2, 3)))
        
    def test_ones(self):
        t = at.ones((2, 3))
        self.assertEqual(t.shape, (2, 3))
        np.testing.assert_array_equal(t.numpy(), np.ones((2, 3)))
        
    def test_full(self):
        t = at.full((2, 2), 5.0)
        self.assertEqual(t.shape, (2, 2))
        np.testing.assert_array_equal(t.numpy(), np.full((2, 2), 5.0))
        
    def test_random(self):
        t = at.random((3, 3))
        self.assertEqual(t.shape, (3, 3))
        self.assertTrue(np.all((t.numpy() >= 0) & (t.numpy() <= 1)))
        
    def test_requires_grad(self):
        t = at.tensor([1.0], requires_grad=True)
        self.assertTrue(t.requires_grad)
        t = at.tensor([1.0], requires_grad=False)
        self.assertFalse(t.requires_grad)

    def test_zeros_like(self):
        t1 = at.tensor([[1,2],[3,4]])
        t2 = at.zeros_like(t1)
        self.assertEqual(t2.shape, (2,2))
        np.testing.assert_array_equal(t2.numpy(), np.zeros((2,2)))
"""

files["test_arithmetic_ops.py"] = header + """
class TestArithmeticOps(unittest.TestCase):
    def test_add(self):
        t1 = at.tensor([1.0, 2.0])
        t2 = at.tensor([3.0, 4.0])
        res = t1 + t2
        np.testing.assert_allclose(res.numpy(), [4.0, 6.0])

    def test_sub(self):
        t1 = at.tensor([5.0, 6.0])
        t2 = at.tensor([3.0, 2.0])
        res = t1 - t2
        np.testing.assert_allclose(res.numpy(), [2.0, 4.0])
        
    def test_mul(self):
        t1 = at.tensor([2.0, 3.0])
        t2 = at.tensor([4.0, 5.0])
        res = t1 * t2
        np.testing.assert_allclose(res.numpy(), [8.0, 15.0])
        
    def test_div(self):
        t1 = at.tensor([10.0, 15.0])
        t2 = at.tensor([2.0, 3.0])
        res = t1 / t2
        np.testing.assert_allclose(res.numpy(), [5.0, 5.0])
        
    def test_neg(self):
        t1 = at.tensor([1.0, -2.0])
        res = -t1
        np.testing.assert_allclose(res.numpy(), [-1.0, 2.0])

    def test_add_scalar(self):
        t = at.tensor([1.0, 2.0])
        res = t + 2.0
        np.testing.assert_allclose(res.numpy(), [3.0, 4.0])
"""

files["test_broadcasting.py"] = header + """
class TestBroadcasting(unittest.TestCase):
    def test_broadcast_scalar(self):
        t1 = at.tensor([[1, 2], [3, 4]])
        res = t1 + 10
        np.testing.assert_allclose(res.numpy(), [[11, 12], [13, 14]])
        
    def test_broadcast_1d(self):
        t1 = at.tensor([[1, 2, 3], [4, 5, 6]])
        t2 = at.tensor([10, 20, 30])
        res = t1 + t2
        np.testing.assert_allclose(res.numpy(), [[11, 22, 33], [14, 25, 36]])
        
    def test_broadcast_2d(self):
        t1 = at.tensor([[1], [2], [3]])
        t2 = at.tensor([10, 20, 30])
        res = t1 + t2
        np.testing.assert_allclose(res.numpy(), [[11, 21, 31], [12, 22, 32], [13, 23, 33]])
        
    def test_broadcast_sub(self):
        t1 = at.tensor([[10, 20], [30, 40]])
        t2 = at.tensor([5, 5])
        res = t1 - t2
        np.testing.assert_allclose(res.numpy(), [[5, 15], [25, 35]])
        
    def test_broadcast_mul(self):
        t1 = at.tensor([[1, 2], [3, 4]])
        t2 = at.tensor([2, 3])
        res = t1 * t2
        np.testing.assert_allclose(res.numpy(), [[2, 6], [6, 12]])
"""

files["test_matrix_ops.py"] = header + """
class TestMatrixOps(unittest.TestCase):
    def test_matmul_2d(self):
        t1 = at.tensor([[1, 2], [3, 4]])
        t2 = at.tensor([[5, 6], [7, 8]])
        res = t1 @ t2
        np.testing.assert_allclose(res.numpy(), [[19, 22], [43, 50]])
        
    def test_transpose_2d(self):
        t1 = at.tensor([[1, 2, 3], [4, 5, 6]])
        res = at.transpose(t1)
        self.assertEqual(res.shape, (3, 2))
        np.testing.assert_allclose(res.numpy(), [[1, 4], [2, 5], [3, 6]])
        
    def test_matmul_shape_mismatch(self):
        t1 = at.tensor([[1, 2]])
        t2 = at.tensor([[1, 2, 3]])
        with self.assertRaises(at.AMEVAForgeShapeError):
            t1 @ t2

    def test_matmul_1d_not_supported(self):
        t1 = at.tensor([1, 2])
        t2 = at.tensor([3, 4])
        with self.assertRaises(at.AMEVAForgeShapeError):
            t1 @ t2
"""

files["test_reduction_ops.py"] = header + """
class TestReductionOps(unittest.TestCase):
    def test_sum_all(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = t.sum()
        self.assertEqual(res.shape, ())
        self.assertAlmostEqual(res.numpy().item(), 10.0)
        
    def test_mean_all(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = t.mean()
        self.assertEqual(res.shape, ())
        self.assertAlmostEqual(res.numpy().item(), 2.5)
        
    def test_sum_axis(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = at.sum_axis(t, axis=0)
        np.testing.assert_allclose(res.numpy(), [4, 6])
        
    def test_sum_axis_1(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = at.sum_axis(t, axis=1)
        np.testing.assert_allclose(res.numpy(), [3, 7])
"""

files["test_activation_functions.py"] = header + """
class TestActivationFunctions(unittest.TestCase):
    def test_relu(self):
        t = at.tensor([-1.0, 0.0, 1.0, 2.0])
        res = at.relu(t)
        np.testing.assert_allclose(res.numpy(), [0.0, 0.0, 1.0, 2.0])
        
    def test_sigmoid(self):
        t = at.tensor([0.0, 100.0, -100.0])
        res = at.sigmoid(t)
        np.testing.assert_allclose(res.numpy(), [0.5, 1.0, 0.0], atol=1e-5)
        
    def test_tanh(self):
        t = at.tensor([0.0, 100.0, -100.0])
        res = at.tanh(t)
        np.testing.assert_allclose(res.numpy(), [0.0, 1.0, -1.0], atol=1e-5)
        
    def test_relu_method(self):
        t = at.tensor([-2.0, 3.0])
        res = t.relu()
        np.testing.assert_allclose(res.numpy(), [0.0, 3.0])
"""

files["test_math_ops.py"] = header + """
class TestMathOps(unittest.TestCase):
    def test_exp(self):
        t = at.tensor([0.0, 1.0])
        res = at.exp(t)
        np.testing.assert_allclose(res.numpy(), [1.0, np.exp(1.0)])
        
    def test_log(self):
        t = at.tensor([1.0, np.exp(1.0)])
        res = at.log(t)
        np.testing.assert_allclose(res.numpy(), [0.0, 1.0])
        
    def test_exp_method(self):
        t = at.tensor([0.0, 1.0])
        res = t.exp()
        np.testing.assert_allclose(res.numpy(), [1.0, np.exp(1.0)])
        
    def test_log_method(self):
        t = at.tensor([1.0, np.exp(1.0)])
        res = t.log()
        np.testing.assert_allclose(res.numpy(), [0.0, 1.0])
"""

files["test_shape_ops.py"] = header + """
class TestShapeOps(unittest.TestCase):
    def test_reshape(self):
        t = at.tensor([[1, 2, 3], [4, 5, 6]])
        res = t.reshape((3, 2))
        self.assertEqual(res.shape, (3, 2))
        np.testing.assert_allclose(res.numpy(), [[1, 2], [3, 4], [5, 6]])
        
    def test_view(self):
        t = at.tensor([[1, 2], [3, 4]])
        res = t.view(4)
        self.assertEqual(res.shape, (4,))
        np.testing.assert_allclose(res.numpy(), [1, 2, 3, 4])
        
    def test_numel(self):
        t = at.tensor([[1, 2], [3, 4]])
        self.assertEqual(t.numel(), 4)
        
    def test_reshape_function(self):
        t = at.tensor([1, 2, 3, 4])
        res = at.reshape(t, (2, 2))
        self.assertEqual(res.shape, (2, 2))
"""

files["test_autograd.py"] = header + """
class TestAutograd(unittest.TestCase):
    def test_add_backward(self):
        x = at.tensor([2.0], requires_grad=True)
        y = x + 3.0
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [1.0])
        
    def test_mul_backward(self):
        x = at.tensor([2.0], requires_grad=True)
        y = x * 3.0
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [3.0])
        
    def test_matmul_backward(self):
        w = at.tensor([[1.0, 2.0]], requires_grad=True)
        x = at.tensor([[3.0], [4.0]])
        y = w @ x
        y.backward()
        np.testing.assert_allclose(w.grad.numpy(), [[3.0, 4.0]])
        
    def test_relu_backward(self):
        x = at.tensor([-2.0, 2.0], requires_grad=True)
        y = x.relu().sum()
        y.backward()
        np.testing.assert_allclose(x.grad.numpy(), [0.0, 1.0])
        
    def test_broadcast_backward(self):
        x = at.tensor([[1.0, 2.0]], requires_grad=True) # (1, 2)
        y = at.tensor([[3.0], [4.0]]) # (2, 1)
        z = (x + y).sum()
        z.backward()
        np.testing.assert_allclose(x.grad.numpy(), [[2.0, 2.0]])
"""

files["test_nn_module.py"] = header + """
class TestNNModule(unittest.TestCase):
    def test_linear(self):
        layer = at.nn.Linear(2, 3)
        x = at.tensor([[1.0, 2.0]])
        out = layer(x)
        self.assertEqual(out.shape, (1, 3))
        self.assertEqual(len(layer.parameters()), 2) # weight, bias
        
    def test_relu_module(self):
        layer = at.nn.ReLU()
        x = at.tensor([[-1.0, 2.0]])
        out = layer(x)
        np.testing.assert_allclose(out.numpy(), [[0.0, 2.0]])
        
    def test_sequential(self):
        model = at.nn.Sequential(
            at.nn.Linear(2, 4),
            at.nn.ReLU(),
            at.nn.Linear(4, 1)
        )
        x = at.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = model(x)
        self.assertEqual(out.shape, (2, 1))
        self.assertEqual(len(model.parameters()), 4)
"""

files["test_optimizers.py"] = header + """
class TestOptimizers(unittest.TestCase):
    def test_sgd(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1)
        y = x * x
        y.backward()
        opt.step()
        np.testing.assert_allclose(x.numpy(), [2.0 - 0.1 * 4.0])
        
    def test_sgd_momentum(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1, momentum=0.9)
        (x * x).backward()
        opt.step()
        opt.zero_grad()
        (x * x).backward()
        opt.step()
        self.assertTrue(x.numpy()[0] < 1.6) # Should move faster
        
    def test_adam(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.Adam([x], lr=0.1)
        (x * x).backward()
        opt.step()
        self.assertTrue(x.numpy()[0] < 2.0)
        
    def test_zero_grad(self):
        x = at.tensor([2.0], requires_grad=True)
        opt = at.optim.SGD([x], lr=0.1)
        (x * x).backward()
        self.assertIsNotNone(x.grad)
        opt.zero_grad()
        self.assertIsNone(x.grad)
"""

files["test_loss_functions.py"] = header + """
class TestLossFunctions(unittest.TestCase):
    def test_mse_loss(self):
        pred = at.tensor([1.0, 2.0])
        target = at.tensor([1.0, 3.0])
        loss = at.F.mse_loss(pred, target)
        self.assertAlmostEqual(loss.numpy().item(), 0.5)
        
    def test_softmax(self):
        x = at.tensor([[1.0, 2.0, 3.0]])
        res = at.F.softmax(x)
        self.assertAlmostEqual(np.sum(res.numpy()), 1.0)
        
    def test_log_softmax(self):
        x = at.tensor([[1.0, 2.0, 3.0]])
        res = at.F.log_softmax(x)
        np.testing.assert_allclose(res.numpy(), np.log(at.F.softmax(x).numpy()), atol=1e-5)
        
    def test_cross_entropy(self):
        pred = at.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        target = at.tensor([0, 1])
        loss = at.F.cross_entropy(pred, target)
        self.assertTrue(loss.numpy().item() < 0.01)
"""

files["test_dataloader.py"] = header + """
class TestDataLoader(unittest.TestCase):
    def test_dataloader_iteration(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        dl = at.DataLoader(x, y, batch_size=3, shuffle=False)
        batches = list(dl)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0][0].shape, (3,))
        self.assertEqual(batches[-1][0].shape, (1,))
        
    def test_dataloader_shuffle(self):
        x = np.arange(100)
        y = np.arange(100)
        dl1 = at.DataLoader(x, y, batch_size=10, shuffle=True)
        dl2 = at.DataLoader(x, y, batch_size=10, shuffle=True)
        b1 = next(iter(dl1))[0].numpy()
        b2 = next(iter(dl2))[0].numpy()
        # High probability they are different
        self.assertFalse(np.array_equal(b1, b2))
"""

files["test_edge_cases.py"] = header + """
class TestEdgeCases(unittest.TestCase):
    def test_nan(self):
        t = at.tensor([float('nan'), 1.0])
        res = t + 1
        self.assertTrue(np.isnan(res.numpy()[0]))
        
    def test_inf(self):
        t = at.tensor([float('inf'), 1.0])
        res = t * 2
        self.assertTrue(np.isinf(res.numpy()[0]))
        
    def test_zero_dim(self):
        t = at.tensor(5.0)
        self.assertEqual(t.shape, ())
        res = t + 2.0
        self.assertEqual(res.shape, ())
        self.assertEqual(res.numpy().item(), 7.0)
        
    def test_empty_tensor(self):
        with self.assertRaises(at.AMEVAForgeShapeError):
            at.zeros((0, 5))
            
    def test_negative_zero(self):
        t = at.tensor([-0.0])
        self.assertEqual(t.numpy()[0], 0.0)
        
    def test_denormals(self):
        t = at.tensor([1e-40])
        res = t * 2
        self.assertEqual(res.numpy()[0], 0.0)
"""

files["test_extreme.py"] = header + """
class TestExtreme(unittest.TestCase):
    def test_large_tensor(self):
        t1 = at.ones((100, 100))
        t2 = at.ones((100, 100))
        res = t1 @ t2
        self.assertEqual(res.numpy()[0,0], 100.0)
        
    def test_deep_computation_chain(self):
        t = at.tensor([1.0], requires_grad=True)
        curr = t
        for _ in range(100):
            curr = curr * 1.01
        curr.sum().backward()
        self.assertAlmostEqual(t.grad.numpy()[0], 1.01 ** 100, places=2)
        
    def test_gradient_accumulation(self):
        x = at.tensor([2.0], requires_grad=True)
        for _ in range(50):
            y = x * x
            y.backward()
        self.assertEqual(x.grad.numpy()[0], 4.0 * 50)
        
    def test_xor_training(self):
        x_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y_data = [[0], [1], [1], [0]]
        
        model = at.nn.Sequential(
            at.nn.Linear(2, 4),
            at.nn.Tanh(),
            at.nn.Linear(4, 1),
            at.nn.Sigmoid()
        )
        opt = at.optim.Adam(model.parameters(), lr=0.1)
        
        x = at.tensor(x_data)
        y = at.tensor(y_data)
        
        for epoch in range(2000):
            opt.zero_grad()
            pred = model(x)
            loss = at.F.mse_loss(pred, y)
            if loss.numpy().item() < 0.05:
                break
            loss.backward()
            opt.step()
            
        self.assertTrue(loss.numpy().item() < 0.05)
"""

files["test_error_handling.py"] = header + """
class TestErrorHandling(unittest.TestCase):
    def test_shape_mismatch_add(self):
        t1 = at.ones((2, 2))
        t2 = at.ones((3, 3))
        with self.assertRaises(at.AMEVAForgeShapeError):
            t1 + t2
            
    def test_disposed_tensor(self):
        t = at.tensor([1, 2])
        t.dispose()
        with self.assertRaises(at.AMEVAForgeDisposedError):
            t + 1
            
    def test_backward_no_grad(self):
        t = at.tensor([1.0], requires_grad=False)
        with self.assertRaises(RuntimeError):
            t.backward()
            
    def test_invalid_shape(self):
        with self.assertRaises(at.AMEVAForgeShapeError):
            at.zeros("invalid")
"""

runner = """import unittest
import json
import os
import io
import sys

# Safe stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

if __name__ == '__main__':
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    total = result.testsRun
    failed = len(result.failures)
    errors = len(result.errors)
    passed = total - failed - errors
    
    details = []
    for test, trace in result.failures + result.errors:
        details.append({
            'test': str(test),
            'traceback': trace
        })
        print(f"[FAIL] {test}")
        
    for test in result.successes if hasattr(result, 'successes') else []:
        print(f"[PASS] {test}")
        
    summary = {
        'total': total,
        'passed': passed,
        'failed': failed,
        'errors': errors,
        'details': details
    }
    
    with open(os.path.join(tests_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\\nTotal: {total}, Passed: {passed}, Failed: {failed}, Errors: {errors}")
"""

files["run_all_tests.py"] = runner

for fname, content in files.items():
    with open(os.path.join(tests_dir, fname), 'w', encoding='utf-8') as f:
        f.write(content)

print("Tests created successfully!")
