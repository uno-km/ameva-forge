#!/usr/bin/env python3
"""AMEVA-Forge 전체 테스트 실행기 — 모든 카테고리를 실행하고 보고서를 생성한다."""
import sys
import os
import io
import unittest
from pathlib import Path
from datetime import datetime

# Windows cp949 인코딩 문제 방지
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from report_generator import run_and_report, MarkdownTestResult, generate_report

REPO_ROOT = str(Path(__file__).parent.parent.parent.parent)

CATEGORIES = [
    ('test_cpu_ops', 'CPU 단위 테스트', 'Category 1: CPU Unit Tests'),
    ('test_edge_cases', '엣지케이스 테스트', 'Category 2: Edge Cases'),
    ('test_exceptions', '예외 처리 테스트', 'Category 3: Exception Handling'),
    ('test_stress', '스트레스 테스트', 'Category 4: Stress Tests'),
    ('test_security', '보안 테스트', 'Category 7: Security'),
    ('test_compatibility', '호환성 테스트', 'Category 8: Compatibility'),
]

def main():
    print("=" * 70)
    print("🔬 AMEVA-Forge 종합 테스트 실행기")
    print(f"   실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = []
    
    for module_name, display_name, category in CATEGORIES:
        print(f"\n{'─' * 60}")
        print(f"▶ {display_name} ({module_name})")
        print(f"{'─' * 60}")
        
        try:
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            result = run_and_report(suite, display_name, category, REPO_ROOT)
            all_results.append((display_name, category, result))
        except Exception as e:
            print(f"  ⚠️ Failed to load {module_name}: {e}")
    
    # Generate combined summary
    print(f"\n{'=' * 70}")
    print("📊 종합 결과 요약")
    print(f"{'=' * 70}")
    
    total_tests = sum(r.testsRun for _, _, r in all_results)
    total_passed = sum(len(r.successes) for _, _, r in all_results)
    total_failed = sum(len(r.failures) for _, _, r in all_results)
    total_errors = sum(len(r.errors) for _, _, r in all_results)
    
    for name, cat, r in all_results:
        status = '✅' if len(r.failures) == 0 and len(r.errors) == 0 else '❌'
        print(f"  {status} {name}: {len(r.successes)}/{r.testsRun} passed")
    
    print(f"\n  TOTAL: {total_passed}/{total_tests} passed, {total_failed} failed, {total_errors} errors")
    
    # Generate combined report
    reports_dir = os.path.join(REPO_ROOT, 'reports', 'tests')
    os.makedirs(reports_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    combined_path = os.path.join(reports_dir, f"{date_str}_combined_all_categories.md")
    
    lines = []
    lines.append("# 🔬 AMEVA-Forge 종합 테스트 보고서")
    lines.append(f"")
    lines.append(f"**실행 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**총 테스트**: {total_tests}")
    lines.append(f"**총 통과**: {total_passed}")
    lines.append(f"**총 실패**: {total_failed + total_errors}")
    lines.append(f"**통과율**: {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%")
    lines.append(f"")
    lines.append(f"## 카테고리별 결과")
    lines.append(f"")
    lines.append(f"| 카테고리 | 총 | 성공 | 실패 | 에러 | 통과율 |")
    lines.append(f"|----------|---|------|------|------|--------|")
    for name, cat, r in all_results:
        p = len(r.successes)
        t = r.testsRun
        f_count = len(r.failures)
        e = len(r.errors)
        rate = (p/t*100) if t > 0 else 0
        status = '✅' if f_count == 0 and e == 0 else '❌'
        lines.append(f"| {status} {name} | {t} | {p} | {f_count} | {e} | {rate:.1f}% |")
    lines.append(f"")
    
    # Append individual reports
    for name, cat, r in all_results:
        lines.append(f"---")
        lines.append(f"")
        report_text = generate_report(r, name, cat)
        lines.append(report_text)
    
    with open(combined_path, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(lines))
    
    print(f"\n📄 Combined report: {combined_path}")
    
    return 0 if (total_failed + total_errors) == 0 else 1

if __name__ == '__main__':
    sys.exit(main())
