import os
import glob

for f in glob.glob('packages/forge-py/tests/test_*.py'):
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace("'..', 'packages', 'forge-py', 'src'", "'..', 'src'")
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
