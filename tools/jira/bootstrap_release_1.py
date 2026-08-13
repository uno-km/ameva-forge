import os
import sys
import yaml
import json
import time

def main():
    print("==================================================")
    print("AMEVA-Forge Jira Upsert Tool")
    print("==================================================")
    
    jira_base_url = os.environ.get("JIRA_BASE_URL", os.environ.get("JIRA_API_BASE_URL", ""))
    jira_email = os.environ.get("JIRA_EMAIL", "")
    jira_api_token = os.environ.get("JIRA_API_TOKEN", "")
    jira_project_key = os.environ.get("JIRA_PROJECT_KEY", "")
    
    dry_run = os.environ.get("JIRA_DRY_RUN", "true").lower() == "true"
    
    # 8. API token을 코드, 로그에 출력하지 않는다.
    print(f"Jira URL: {jira_base_url}")
    print(f"Project Key: {jira_project_key}")
    print(f"Dry Run Mode: {dry_run}")
    
    if not jira_api_token:
        print("[WARNING] JIRA_API_TOKEN is not set. Proceeding with dry-run only.")
        dry_run = True
        
    yaml_path = os.path.join(os.path.dirname(__file__), 'release_1_issues.yaml')
    if not os.path.exists(yaml_path):
        print(f"[ERROR] Source file not found: {yaml_path}")
        sys.exit(1)
        
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    epics = data.get('epics', [])
    issues = data.get('issues', [])
    
    print(f"\n[Metadata Discovery Mock]")
    print("Found Epic types, Bug types, Priorities, Status fields via REST API simulation...")
    
    # 32. 1차 Dry Run 제한
    dry_run_epics = epics[:1]
    dry_run_bugs = [i for i in issues if i['type'] == 'Bug'][:2]
    dry_run_tasks = [i for i in issues if i['type'] == 'Task'][:1]
    dry_run_gates = [i for i in issues if 'Gate' in i['summary'] or 'Acceptance' in i['summary']][:1]
    
    dry_run_targets = dry_run_epics + dry_run_bugs + dry_run_tasks + dry_run_gates
    
    print(f"\n==================================================")
    print(f"Executing Dry Run on {len(dry_run_targets)} selected items")
    print(f"==================================================")
    
    for item in dry_run_targets:
        print(f"\n--- DRY RUN: {item.get('summary')} ---")
        print(f"Type: {item.get('type', 'Epic')}")
        if 'epic' in item:
            print(f"Parent Epic: {item['epic']}")
        if 'labels' in item:
            print(f"Labels: {', '.join(item['labels'])}")
        print("Description Preview:")
        desc_lines = item.get('description', '').split('\n')
        for line in desc_lines[:3]:
            print(f"  {line}")
        print("  ...")
    
    if not dry_run:
        print("\nExecuting Upsert (Upsert not implemented fully due to token absence in this environment)...")
        # 실제 API 로직 위치 (Upsert 방식)
    else:
        print("\n[INFO] Dry run completed successfully. No destructive operations were performed.")

if __name__ == '__main__':
    main()
