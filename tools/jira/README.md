# AMEVA-Forge Jira Upsert Tools

This directory contains scripts and data files for bootstrapping the AMEVA-Forge Release 1 Jira project.

## Requirements
- Python 3.8+
- PyYAML

## Usage
1. Configure `.env` or set environment variables:
   - `JIRA_BASE_URL`
   - `JIRA_PROJECT_KEY`
   - `JIRA_EMAIL`
   - `JIRA_API_TOKEN`

2. Run dry run:
   ```bash
   python bootstrap_release_1.py
   ```

3. Run actual upsert (requires token):
   ```bash
   JIRA_DRY_RUN=false python bootstrap_release_1.py
   ```

## Design Principles
- External ID via labels for Upsert.
- Retries on 429 and 5xx.
- Dry-run by default.
- No secrets in logs.
