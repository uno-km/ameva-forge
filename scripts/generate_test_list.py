import os
import json

def generate_test_list():
    tests_dir = os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'tests')
    output_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'playgrounds', 'tests.json')
    
    test_files = [f for f in os.listdir(tests_dir) if f.startswith('test_') and f.endswith('.py')]
    test_files.sort()
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_files, f, indent=4)
        
    print(f"Generated {output_path} with {len(test_files)} test files.")

if __name__ == "__main__":
    generate_test_list()
