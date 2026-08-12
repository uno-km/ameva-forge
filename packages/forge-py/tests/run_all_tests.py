import unittest
import json
import os
import io
import sys

# Safe stdout
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
        
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {failed}, Errors: {errors}")
