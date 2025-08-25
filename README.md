# Log_Agent

Autonomous AIOps agent that reads application logs, identifies critical errors, asks an LLM (Google Gemini) to perform root-cause analysis, generates a code fix, patches the code, creates a Git branch/commit, pushes it, and opens a Pull Request. Optionally sends email notifications.

Repository contents
- aiops_agent.py — main agent orchestrator
  - LogAnalyzer: parses logs, finds latest critical error, collects preceding events
  - CodeDiagnoser: calls Gemini to produce RCA and a code fix
  - CodePatcher: applies the generated fix to the culprit source file
  - GitManager: creates a branch, commits changes, pushes to origin
  - NotificationManager: creates a PR (GitHub REST) and can send email
- buggy_app.py — a small script that intentionally logs an error (division by zero) to exercise the agent
- README.md — this file

Note: The current code is a work-in-progress and includes TODOs and placeholder sections. See “Code Review and Known Issues” below.

Features
- Log parsing and error detection
- LLM-driven RCA and code remediation suggestion (Gemini 1.5 Flash)
- Patch application to local repo files
- Branch/commit/push automation (GitPython)
- GitHub PR creation (via REST)
- Optional SMTP email notification

Requirements
- Python 3.10+ recommended
- Google API key for Generative AI (Gemini)
- A local Git repository containing the code to patch (the agent operates on this repo)
- GitHub token with repo scope for PR creation (optional but recommended)
- SMTP credentials (optional, for email notifications)

Install
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # fill in values
```

Environment variables (.env)
- GOOGLE_API_KEY: Google Generative AI API key
- GITHUB_TOKEN: GitHub Personal Access Token (repo scope)
- GITHUB_REPO_OWNER: GitHub owner/org for PR creation (e.g., varunreddyGOPU)
- GITHUB_REPO_NAME: Repository name where PRs should be opened
- PR_ASSIGNEE: GitHub username to assign the PR to (optional)
- EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_RECIPIENT: SMTP settings (optional)

Usage

1) Generate a test log
- Run the sample buggy app to create a log entry:
```bash
python buggy_app.py
```

This writes a division-by-zero error to app_error.log.

2) Configure the agent
- Open aiops_agent.py and set the REPO_PATH variable (in the main block, if present) to point to your local repository that contains the file to patch (e.g., the directory where buggy_app.py resides and is a Git repo).
- Ensure your .env is configured (see above).

3) Run the agent
```bash
python aiops_agent.py
```

What the agent will try to do
- Parse app_error.log and detect the latest critical error
- Ask Gemini for an RCA and a code fix based on the culprit file and context
- Overwrite the culprit file with the generated fix
- Create a branch, commit with RCA details, push to origin
- Create a Pull Request on GitHub
- Optionally send an email summarizing the remediation

Notes on Git workflow
- The agent uses GitPython. It attempts to:
  - Create a new branch named agent-fix/remediation-YYYYMMDD-HHMMSS
  - Commit the patched file with a structured message (includes RCA)
  - Push the branch to origin
  - Open a PR against the default branch (assumed master/main)

API/Model defaults
- Gemini model: gemini-1.5-flash (in CodeDiagnoser)

Project status
- Prototype/WIP. See “Code Review and Known Issues” for current gaps and suggested improvements.

Code Review and Known Issues

Log parsing and format
- buggy_app.py uses logging format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
- LogAnalyzer.log_pattern expects: timestamp - level - message
- Action: Update regex to include the logger name field, for example:
  r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<logger>[\w\.]+) - (?P<level>\w+) - (?P<message>.*)'

Incomplete variables/blocks
- Several functions have placeholders or incomplete assignments (e.g., critical_errors =, traceback_lines =, preceding_events =). These will raise syntax errors.
- Action: Initialize lists properly (e.g., critical_errors = [], traceback_lines = [], preceding_events = []), and complete the logic blocks.

Latest error selection bug
- In analyze(), latest_error is assigned from sorted(...), which returns a list, then used like a dict (latest_error['timestamp']).
- Action: Use latest_error = sorted(..., reverse=True)[0].

Culprit file detection
- analyze() defaults culprit_file to "buggy_app.py" and optionally extracts from traceback.
- Action: Ensure traceback regex is robust and that the log message includes a traceback. If the format doesn’t include “File ..., line ...” patterns, augment detection.

LLM prompting and parsing
- CodeDiagnoser builds prompts for RCA and remediation. Responses are parsed with response.text and json.loads in places.
- Action: Use the official google-generativeai SDK response object fields (e.g., candidates) and handle JSON extraction/validation robustly. Add guardrails for model output (e.g., enforce fenced JSON or a schema check).

PR creation
- Title currently uses rca.splitlines() which produces a list string; prefer a concise single-line summary.
- Ensure base branch and head are specified if your repository’s default is “main” rather than “master”.
- Validate HTTP response and print errors with context.

Git workflow
- After pushing, code checks out self.repo.heads.master; this will fail if the default branch is main.
- Action: Detect the default branch dynamically (self.repo.active_branch.name before switching, or config setting) and return to it.

Stray snippet at bottom of aiops_agent.py
- There appears to be a pasted code example block inside the source (surrounded by backticks). Remove it; it will cause syntax errors.

Config discoverability
- Consolidate constants (paths like app_error.log and REPO_PATH) into environment variables or a config section for easier customization.

Email
- NotificationManager mentions SMTP but the code path is not fully shown. If used, ensure TLS/SSL, exceptions handling, and timeouts.

Testing and examples
- Consider adding:
  - A create_mock_files() helper (referenced in comments) to bootstrap a demo repo
  - Unit tests for LogAnalyzer and PR creation
  - A dry-run flag to skip patching/commits for safe testing

Security
- Do not log secrets. Ensure .env is not committed. Add a .gitignore if missing.

License
- Add a LICENSE file (e.g., Apache-2.0 or MIT) if you plan to share widely.

Roadmap
- Robust JSON schema validation for LLM outputs
- AST-based localized patching instead of overwriting entire files
- Provider abstraction to support other LLMs
- Richer PR description with diff and links to logs
- Dockerfile and CI workflow (lint, tests)

Contributing
- PRs welcome. Please open an issue describing bugs or enhancements.
