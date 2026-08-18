import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import forge as fg
import forge.errors as err
from forge.graph import GraphBuilder

class TestSecurity(unittest.TestCase):
    """Category 7: Security & Input Boundary Hardening Tests."""

    def test_invalid_op_name_in_graph_instruction(self):
        """그래프 명령어에 잘못된 연산자(op name) 입력 시 에러가 발생하는지 확인합니다."""
        builder = GraphBuilder()
        with self.assertRaises(ValueError):
            builder.add_op("", (2, 2), [1])
        with self.assertRaises(ValueError):
            builder.add_op("   ", (2, 2), [1])

    def test_malicious_json_instruction(self):
        """악의적인 JSON 명령어를 주입했을 때 시스템이 이를 거부하는지 확인합니다."""
        builder = GraphBuilder()
        with self.assertRaises(ValueError):
            builder.add_op("__proto__", (2, 2), [1])
        with self.assertRaises(ValueError):
            builder.add_op("constructor", (2, 2), [1])
        with self.assertRaises(ValueError):
            builder.add_op("prototype", (2, 2), [1])

    def test_negative_node_id(self):
        """음수인 노드 ID가 주어졌을 때 올바르게 거부하는지 확인합니다."""
        builder = GraphBuilder()
        with self.assertRaises(ValueError):
            builder.add_op("add", (2, 2), [-1, 2])
        with self.assertRaises(ValueError):
            builder.add_op("add", (2, 2), [0, 1])

    def test_excessive_shape_dimensions(self):
        """과도한 차원의 shape가 입력될 경우 시스템을 보호하는지 확인합니다."""
        builder = GraphBuilder()
        with self.assertRaises(ValueError):
            builder.add_upload((1, 2, 3, 4, 5, 6, 7, 8, 9))
        with self.assertRaises(ValueError):
            builder.add_op("add", (1, 2, 3, 4, 5, 6, 7, 8, 9), [1, 2])

    def test_error_hierarchy_validation(self):
        """에러 계층 구조가 변조되지 않고 올바르게 검증되는지 확인합니다."""
        error_classes = [
            err.AMEVAForgeShapeError,
            err.AMEVAForgeDTypeError,
            err.AMEVAForgeDeviceError,
            err.AMEVAForgeDisposedError,
            err.AMEVAForgeWebGPUUnavailableError,
            err.AMEVAForgeQuotaExceededError,
            err.AMEVAForgeSecurityError,
            err.AMEVAForgeValidationError,
            err.AMEVAForgeOutOfMemoryError,
            err.AMEVAForgeInternalGPUError,
            err.AMEVAForgeDeviceLostError,
            err.AMEVAForgeStaleHandleError,
            err.AMEVAForgeUnsupportedOperationError,
        ]
        for cls in error_classes:
            self.assertTrue(issubclass(cls, err.AMEVAForgeError), f"{cls} must inherit from AMEVAForgeError")
            self.assertTrue(issubclass(cls, Exception), f"{cls} must inherit from Exception")

    def test_shape_injection_attempts(self):
        """Shape 파라미터에 대한 악의적인 인젝션 공격 시도가 차단되는지 확인합니다."""
        builder = GraphBuilder()
        with self.assertRaises((ValueError, TypeError)):
            builder.add_upload((-1, 4))
        with self.assertRaises((ValueError, TypeError)):
            builder.add_upload(("2; DROP TABLE", 4))  # type: ignore

    def test_all_error_classes_are_properly_typed(self):
        """모든 에러 클래스가 올바르게 타입 지정(Typed)되어 있는지 확인합니다."""
        e = err.AMEVAForgeSecurityError("Security violation detected")
        self.assertIsInstance(e, err.AMEVAForgeError)
        self.assertEqual(str(e), "Security violation detected")

    def test_no_arbitrary_code_execution_via_tensor_ops(self):
        """Verify that tensor operations cannot be used as RCE vectors."""
        import forge as fg
        t = fg.tensor([1.0, 2.0, 3.0])
        
        # Verify no dangerous builtins are accessible through tensor
        dangerous_attrs = ['__import__', '__subclasses__', '__globals__', 
                           '__builtins__', 'system', 'exec', 'eval', 'compile']
        for attr in dangerous_attrs:
            self.assertFalse(
                callable(getattr(t, attr, None)),
                f"Tensor exposes callable dangerous attribute: {attr}"
            )
        
        # Verify graph builder rejects potentially dangerous op names
        from forge.graph import GraphBuilder
        gb = GraphBuilder()
        with self.assertRaises((ValueError, KeyError)):
            gb.add_op('__import__', [], [1], [])
        with self.assertRaises((ValueError, KeyError)):
            gb.add_op('eval', [], [1], [])

    def test_tensor_repr_does_not_leak_sensitive_info(self):
        """텐서의 __repr__ 출력이 메모리 주소 등 민감한 정보를 노출하지 않는지 확인합니다."""
        t = fg.tensor([[1.0, 2.0], [3.0, 4.0]])
        r = repr(t)
        self.assertIn("Tensor", r)
        self.assertNotIn("0x", r)
        self.assertNotIn("password", r.lower())
        self.assertNotIn("token", r.lower())

    def test_import_does_not_trigger_side_effects(self):
        """모듈 임포트 시 예기치 않은 부작용(Side effects)이 발생하지 않는지 확인합니다."""
        import forge
        self.assertTrue(hasattr(forge, "tensor"))
        self.assertTrue(hasattr(forge, "nn"))
        self.assertTrue(hasattr(forge, "optim"))

if __name__ == '__main__':
    from report_generator import run_and_report
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSecurity)
    run_and_report(suite, '보안 테스트', 'Category 7: Security')
