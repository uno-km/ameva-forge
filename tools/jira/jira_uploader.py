import os
import sys
import yaml
import json
import requests
from requests.auth import HTTPBasicAuth
import time

def build_adf(text):
    # Simple ADF wrapping
    paragraphs = []
    for line in text.split('\n'):
        if line.strip():
            paragraphs.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": line}]
            })
    
    if not paragraphs:
        paragraphs.append({"type": "paragraph", "content": []})
        
    return {
        "version": 1,
        "type": "doc",
        "content": paragraphs
    }

def main():
    print("==================================================")
    print("AMEVA-Forge Jira Real Upsert Tool")
    print("==================================================")
    
    jira_base_url = os.environ.get("JIRA_BASE_URL", "").rstrip('/')
    jira_email = os.environ.get("JIRA_EMAIL", "")
    jira_api_token = os.environ.get("JIRA_API_TOKEN", "")
    jira_project_key = os.environ.get("JIRA_PROJECT_KEY", "")
    
    if not all([jira_base_url, jira_email, jira_api_token, jira_project_key]):
        print("[ERROR] Missing required Jira environment variables.")
        sys.exit(1)
        
    auth = HTTPBasicAuth(jira_email, jira_api_token)
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    yaml_path = os.path.join(os.path.dirname(__file__), 'release_1_issues_phase1.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    epics = data.get('epics', [])
    issues = data.get('issues', [])
    
    print(f"Jira URL: {jira_base_url}")
    print(f"Project Key: {jira_project_key}")
    
    # Get Project details
    url = f"{jira_base_url}/rest/api/3/project/{jira_project_key}"
    resp = requests.get(url, headers=headers, auth=auth)
    if resp.status_code != 200:
        print(f"[ERROR] Failed to fetch project: {resp.text}")
        sys.exit(1)
        
    project_id = resp.json()['id']
    issue_types = resp.json()['issueTypes']
    
    issue_type_map = {it['name']: it['id'] for it in issue_types}
    print("Found Issue Types:", issue_type_map)
    
    # Helper to find issue type ID
    def get_issue_type_id(type_name):
        mapping = {
            "Epic": ["Epic", "에픽"],
            "Bug": ["Bug", "버그"],
            "Story": ["Story", "스토리"],
            "Task": ["Task", "작업"]
        }
        candidates = mapping.get(type_name, [type_name])
        for cand in candidates:
            if cand in issue_type_map:
                return issue_type_map[cand]
        return list(issue_type_map.values())[0] # Fallback to first available

    created_epics = 0
    created_issues = 0
    updated_issues = 0
    
    epic_key_map = {} # map epic 'key' from yaml to actual Jira key
    
    # 1. Upsert Epics
    for epic in epics:
        epic_yaml_key = epic['key']
        epic_label = epic_yaml_key.lower()
        
        jql = f'project = "{jira_project_key}" AND labels = "{epic_label}"'
        search_url = f"{jira_base_url}/rest/api/3/search"
        search_resp = requests.get(search_url, headers=headers, auth=auth, params={"jql": jql, "maxResults": 1})
        search_data = search_resp.json()
        
        if search_data.get('total', 0) > 0:
            existing_key = search_data['issues'][0]['key']
            epic_key_map[epic_yaml_key] = existing_key
            print(f"Epic {epic_yaml_key} already exists as {existing_key}")
            continue
            
        # Create Epic
        payload = {
            "fields": {
                "project": {"id": project_id},
                "summary": epic['summary'],
                "description": build_adf(epic.get('description', '')),
                "issuetype": {"id": get_issue_type_id("Epic")},
                "labels": ["release-1", epic_label]
            }
        }
        
        # Epic Name field might be required depending on Jira config. Try without it first.
        # If it fails, we will know. New Jira Cloud uses standard fields.
        create_url = f"{jira_base_url}/rest/api/3/issue"
        create_resp = requests.post(create_url, json=payload, headers=headers, auth=auth)
        if create_resp.status_code == 201:
            new_key = create_resp.json()['key']
            epic_key_map[epic_yaml_key] = new_key
            print(f"Created Epic: {new_key} for {epic_yaml_key}")
            created_epics += 1
        else:
            # If Epic Name is required (custom field usually format customfield_10011)
            # Find the Epic Name custom field
            meta_url = f"{jira_base_url}/rest/api/3/issue/createmeta?projectKeys={jira_project_key}&issuetypeNames=Epic,에픽&expand=projects.issuetypes.fields"
            meta_resp = requests.get(meta_url, headers=headers, auth=auth)
            epic_name_field = None
            if meta_resp.status_code == 200:
                try:
                    fields = meta_resp.json()['projects'][0]['issuetypes'][0]['fields']
                    for k, v in fields.items():
                        if 'Epic Name' in v['name'] or '에픽 이름' in v['name']:
                            epic_name_field = k
                            break
                except: pass
            
            if epic_name_field:
                payload["fields"][epic_name_field] = epic['summary']
                create_resp2 = requests.post(create_url, json=payload, headers=headers, auth=auth)
                if create_resp2.status_code == 201:
                    new_key = create_resp2.json()['key']
                    epic_key_map[epic_yaml_key] = new_key
                    print(f"Created Epic (with name): {new_key} for {epic_yaml_key}")
                    created_epics += 1
                else:
                    print(f"[ERROR] Failed to create Epic {epic_yaml_key}: {create_resp2.text}")
            else:
                print(f"[ERROR] Failed to create Epic {epic_yaml_key}: {create_resp.text}")

    # 2. Upsert Issues
    search_url = f"{jira_base_url}/rest/api/3/search"
    for issue in issues:
        issue_id = issue['id'].lower()
        jql = f'project = "{jira_project_key}" AND labels = "{issue_id}"'
        search_resp = requests.get(search_url, headers=headers, auth=auth, params={"jql": jql, "maxResults": 1})
        search_data = search_resp.json()
        
        issue_type_name = issue['type']
            
        labels = issue.get('labels', [])
        if issue_id not in labels:
            labels.append(issue_id)
            
        fields = {
            "project": {"id": project_id},
            "summary": issue['summary'],
            "description": build_adf(issue.get('description', '')),
            "issuetype": {"id": get_issue_type_id(issue_type_name)},
            "labels": labels
        }
        
        if 'epic' in issue and issue['epic'] in epic_key_map:
            # New Jira Cloud links parent Epic via "parent"
            fields["parent"] = {"key": epic_key_map[issue['epic']]}
            
        if search_data.get('total', 0) > 0:
            existing_key = search_data['issues'][0]['key']
            print(f"Updating existing issue: {existing_key} for {issue_id}")
            update_url = f"{jira_base_url}/rest/api/3/issue/{existing_key}"
            # Only update description/summary to not overwrite statuses
            update_payload = {
                "fields": {
                    "summary": fields["summary"],
                    "description": fields["description"],
                    "labels": fields["labels"]
                }
            }
            res = requests.put(update_url, json=update_payload, headers=headers, auth=auth)
            if res.status_code in [200, 204]:
                updated_issues += 1
            else:
                print(f"[ERROR] Failed to update {existing_key}: {res.text}")
        else:
            create_url = f"{jira_base_url}/rest/api/3/issue"
            res = requests.post(create_url, json={"fields": fields}, headers=headers, auth=auth)
            if res.status_code == 201:
                print(f"Created Issue: {res.json()['key']} for {issue_id}")
                created_issues += 1
            else:
                print(f"[ERROR] Failed to create issue {issue_id}: {res.text}")
                
    print("\n==================================================")
    print("Upsert Completed.")
    print(f"Created Epics: {created_epics}")
    print(f"Created Issues: {created_issues}")
    print(f"Updated Issues: {updated_issues}")
    print("==================================================")

if __name__ == '__main__':
    main()
