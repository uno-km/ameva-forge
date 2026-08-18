from pathlib import Path
import re
import sys

ROOTS = [Path('README.md'), Path('docs')]
PATTERNS = [
    r'perfect(?:ly)?', r'empirical proof', r'actual benchmark results',
    r'100%.*GPU', r'완벽(?:히|하게)?', r'실증(?:적| 데이터)',
    r'절대적', r'입증(?:했|된|한다)',
]
ALLOW_MARKERS = ('UNVERIFIED', '실측 아님', 'not measured', 'projected target', 'unverified')

violations = []
marker_count = 0

for root in ROOTS:
    paths = [root] if root.is_file() else list(root.rglob('*'))
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {'.md', '.html'}:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for line_no, line in enumerate(text.splitlines(), 1):
            if any(marker.lower() in line.lower() for marker in ALLOW_MARKERS):
                marker_count += 1
                continue
            if any(re.search(pattern, line, re.I) for pattern in PATTERNS):
                violations.append(f'{path}:{line_no}: {line.strip()}')

if violations:
    sys.stdout.reconfigure(encoding='utf-8')
    print("Claim linter found unverified hype words:")
    for v in violations:
        print(v)
    sys.exit(1)

if marker_count > 10:
    print(f"Warning: Excessive use of ALLOW_MARKERS ({marker_count} times). Please verify claims.")

print("Claim linter PASSED: All documentation claims are properly annotated.")
