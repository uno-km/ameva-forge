#!/usr/bin/env python3
"""
==============================================================================
AMEVA-Forge Release & Jira Real-time Strict Tracking Harness
==============================================================================
Manages task lifecycles, real-time code modifications, metrics, error logs,
and comment tracking across Release 1.0 ~ Release 5.0.
"""

import sys
import os
import json
import argparse
from datetime import datetime, timezone

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRATCH_DIR = os.path.join(ROOT_DIR, "scratch")
STATE_FILE = os.path.join(SCRATCH_DIR, "jira_tasks_state.json")
MATRIX_FILE = os.path.join(ROOT_DIR, "JIRA_RELEASE_TASKS_MATRIX.md")

os.makedirs(SCRATCH_DIR, exist_ok=True)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"tasks": {}, "history": []}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def get_timestamp():
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

def cmd_start(args):
    state = load_state()
    ticket = args.ticket.upper()
    now = get_timestamp()
    
    if ticket not in state["tasks"]:
        state["tasks"][ticket] = {
            "ticket": ticket,
            "status": "IN_PROGRESS",
            "started_at": now,
            "logs": [],
            "errors": [],
            "metrics": {}
        }
    else:
        state["tasks"][ticket]["status"] = "IN_PROGRESS"
        state["tasks"][ticket]["updated_at"] = now

    entry = f"[{now}] START {ticket}: Initiated task development."
    state["tasks"][ticket]["logs"].append(entry)
    state["history"].append(entry)
    save_state(state)
    print(f"[*] Task {ticket} marked as IN_PROGRESS at {now}")

def cmd_log(args):
    state = load_state()
    ticket = args.ticket.upper()
    now = get_timestamp()
    
    if ticket not in state["tasks"]:
        state["tasks"][ticket] = {"ticket": ticket, "status": "IN_PROGRESS", "started_at": now, "logs": [], "errors": [], "metrics": {}}
    
    log_msg = f"[{now}] EDIT & TEST {ticket} | File: {args.file} | Action: {args.action}"
    if args.metrics:
        log_msg += f" | Metrics: {args.metrics}"
        state["tasks"][ticket]["metrics"][now] = args.metrics

    state["tasks"][ticket]["logs"].append(log_msg)
    state["tasks"][ticket]["updated_at"] = now
    state["history"].append(log_msg)
    save_state(state)
    print(f"[*] Recorded log for {ticket}: {log_msg}")

def cmd_fail(args):
    state = load_state()
    ticket = args.ticket.upper()
    now = get_timestamp()
    
    if ticket not in state["tasks"]:
        state["tasks"][ticket] = {"ticket": ticket, "status": "FAILED", "started_at": now, "logs": [], "errors": [], "metrics": {}}
    
    state["tasks"][ticket]["status"] = "FAILED"
    err_msg = f"[{now}] ERROR in {ticket}: {args.error}"
    state["tasks"][ticket]["errors"].append(err_msg)
    state["tasks"][ticket]["logs"].append(err_msg)
    state["tasks"][ticket]["updated_at"] = now
    state["history"].append(err_msg)
    save_state(state)
    print(f"[!] Recorded failure comment on {ticket}: {args.error}")

def cmd_done(args):
    state = load_state()
    ticket = args.ticket.upper()
    now = get_timestamp()
    
    if ticket not in state["tasks"]:
        state["tasks"][ticket] = {"ticket": ticket, "started_at": now, "logs": [], "errors": [], "metrics": {}}
    
    state["tasks"][ticket]["status"] = "DONE"
    state["tasks"][ticket]["completed_at"] = now
    done_msg = f"[{now}] COMPLETED {ticket} | Summary: {args.summary}"
    state["tasks"][ticket]["logs"].append(done_msg)
    state["history"].append(done_msg)
    save_state(state)
    print(f"[+] Task {ticket} successfully marked as DONE: {args.summary}")

def cmd_status(args):
    state = load_state()
    print("=" * 80)
    print(" AMEVA-Forge Master Jira Task Status Board")
    print("=" * 80)
    tasks = state.get("tasks", {})
    if not tasks:
        print("No active tasks registered in state database.")
    else:
        for t, data in tasks.items():
            status = data.get("status", "UNKNOWN")
            print(f"- [{status:12s}] {t:10s} (Started: {data.get('started_at', 'N/A')})")
            if data.get("errors"):
                print(f"    Errors: {len(data['errors'])} recorded")
            if data.get("metrics"):
                print(f"    Latest Metric: {list(data['metrics'].values())[-1]}")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(description="AMEVA-Forge Release & Jira Tracking Harness")
    subparsers = parser.add_subparsers(dest="command")

    # start
    p_start = subparsers.add_parser("start", help="Start a Jira task")
    p_start.add_argument("ticket", help="Jira ticket ID, e.g., SCRUM-201")

    # log
    p_log = subparsers.add_parser("log", help="Log a file edit and test metric")
    p_log.add_argument("ticket", help="Jira ticket ID")
    p_log.add_argument("--file", required=True, help="Modified file path")
    p_log.add_argument("--action", required=True, help="Action description")
    p_log.add_argument("--metrics", default="", help="Test metrics (e.g. Loss: 0.012, VRAM: 112B, PASS)")

    # fail
    p_fail = subparsers.add_parser("fail", help="Log an error on a task")
    p_fail.add_argument("ticket", help="Jira ticket ID")
    p_fail.add_argument("--error", required=True, help="Error details")

    # done
    p_done = subparsers.add_parser("done", help="Mark task as DONE")
    p_done.add_argument("ticket", help="Jira ticket ID")
    p_done.add_argument("--summary", required=True, help="Completion summary")

    # status
    subparsers.add_parser("status", help="Show all task statuses")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "start":
        cmd_start(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "fail":
        cmd_fail(args)
    elif args.command == "done":
        cmd_done(args)
    elif args.command == "status":
        cmd_status(args)

if __name__ == "__main__":
    main()
