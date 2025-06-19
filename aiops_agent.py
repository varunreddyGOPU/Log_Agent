# aiops_agent.py
#
# An autonomous AIOps agent for end-to-end error remediation in Python.
#
# Author: Award-Winning Writer
# Version: 1.0.0
#
# Description:
# This script implements a multi-stage AIOps agent that can:
# 1.  Analyze application logs to detect critical errors and identify recurring patterns.
# 2.  Use a Large Language Model (LLM) via Google's Generative AI API to perform
#     root cause analysis (RCA) and generate a code fix.
# 3.  Programmatically apply the generated code fix as a patch.
# 4.  Automate the Git workflow: create a new branch, commit the fix, and push to remote.
# 5.  Create a pull request on GitHub and send an email notification to stakeholders.
#
# Setup Instructions:
# 1.  Install required Python libraries:
#     pip install google-generativeai GitPython requests python-dotenv
#
# 2.  Create a `.env` file in the same directory as this script and add your credentials:
#     GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
#     GITHUB_TOKEN="YOUR_GITHUB_PERSONAL_ACCESS_TOKEN"
#     GITHUB_REPO_OWNER="YOUR_GITHUB_USERNAME_OR_ORG"
#     GITHUB_REPO_NAME="YOUR_REPOSITORY_NAME"
#     PR_ASSIGNEE="GITHUB_USERNAME_TO_ASSIGN_PR"
#     EMAIL_HOST="your_smtp_server.com"
#     EMAIL_PORT=587
#     EMAIL_USER="your_email@example.com"
#     EMAIL_PASS="your_email_password"
#     EMAIL_RECIPIENT="recipient_email@example.com"
#
# 3.  Create a local Git repository for the agent to work with.
#     - Initialize a new repo: `git init my-test-app`
#     - `cd my-test-app`
#     - Create a `buggy_app.py` file with the content from the `create_mock_files()` function below.
#     - Create an `app.log` file with the content from `create_mock_files()`.
#     - Commit the initial files: `git add. && git commit -m "Initial commit"`
#     - Create a corresponding repository on GitHub and push the initial commit.
#       `git remote add origin https://github.com/YOUR_GITHUB_USERNAME_OR_ORG/YOUR_REPOSITORY_NAME.git`
#       `git push -u origin master` (or `main`)
#
# 4.  Update the `REPO_PATH` variable in the `main` block to point to your local repository.
#
# 5.  Run the agent:
#     python aiops_agent.py

import os
import re
import json
import difflib
import smtplib
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Third-party libraries
import google.generativeai as genai
from git import Repo, Actor
import requests
from dotenv import load_dotenv

# --- Configuration Loading ---
load_dotenv()

class LogAnalyzer:
    """
    Analyzes log files to find critical errors and preceding event sequences.
    """
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.log_keywords = ["ERROR", "FAILURE", "EXCEPTION", "TRACEBACK", "CRITICAL"]  # Assign an appropriate value or list of keywords
        self.log_pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (?P<level>\w+) - (?P<message>.*)'
        )

    def find_critical_errors(self) -> List:
        """Finds log entries containing critical keywords."""
        critical_errors =
        try:
            with open(self.log_file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: Log file not found at {self.log_file_path}")
            return

        for line in lines:
            if any(keyword in line for keyword in self.log_keywords):
                match = self.log_pattern.match(line)
                if match:
                    error_data = match.groupdict()
                    # Include full traceback if it follows
                    traceback_lines =
                    current_line_index = lines.index(line)
                    for next_line in lines[current_line_index + 1:]:
                        if self.log_pattern.match(next_line):
                            break # Start of a new log entry
                        traceback_lines.append(next_line.strip())
                    
                    if traceback_lines:
                        error_data['message'] += "\n" + "\n".join(traceback_lines)
                    
                    critical_errors.append(error_data)
        return critical_errors

    def find_preceding_events(self, error_timestamp_str: str, window_minutes: int = 5) -> List:
        """Finds events that occurred within a time window before a critical error."""
        preceding_events =
        error_timestamp = datetime.strptime(error_timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        window_start_time = error_timestamp - timedelta(minutes=window_minutes)

        with open(self.log_file_path, 'r') as f:
            for line in f:
                match = self.log_pattern.match(line)
                if match:
                    event_data = match.groupdict()
                    try:
                        event_timestamp = datetime.strptime(event_data['timestamp'], '%Y-%m-%d %H:%M:%S,%f')
                        if window_start_time <= event_timestamp < error_timestamp:
                            preceding_events.append(event_data)
                    except ValueError:
                        continue
        return preceding_events

    def analyze(self) -> Optional:
        """
        Performs a full analysis of the log file.
        Returns a dictionary with analysis results or None if no critical errors are found.
        """
        print("--- Starting Log Analysis ---")
        critical_errors = self.find_critical_errors()
        if not critical_errors:
            print("No critical errors found.")
            return None

        # Focus on the most recent critical error
        latest_error = sorted(critical_errors, key=lambda x: x['timestamp'], reverse=True)
        print(f"Found critical error at {latest_error['timestamp']}:\n{latest_error['message']}")

        preceding_events = self.find_preceding_events(latest_error['timestamp'])
        print(f"Found {len(preceding_events)} preceding events in the last 5 minutes.")

        # Extract culprit file from traceback for later use
        culprit_file = "buggy_app.py" # Default
        traceback_match = re.search(r'File "(.+?)", line \d+', latest_error['message'])
        if traceback_match:
            culprit_file = traceback_match.group(1)

        return {
            "latest_error": latest_error,
            "preceding_events": preceding_events,
            "culprit_file": culprit_file
        }

class CodeDiagnoser:
    """
    Uses a Generative AI model to diagnose issues and suggest code fixes.
    """
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def _create_diagnostic_prompt(self, analysis_data: Dict, source_code: str) -> str:
        """Constructs a detailed prompt for root cause analysis."""
        error = analysis_data['latest_error']
        events = "\n".join([f"{e['timestamp']} - {e['level']} - {e['message']}" for e in analysis_data['preceding_events']])
        
        return f"""
        You are an expert Python Site Reliability Engineer with 20 years of experience.
        Your task is to perform a root cause analysis based on the provided data.

        ### CONTEXT START ###

        #### PRECEDING LOG EVENTS (in chronological order) ####
        {events}

        #### CRITICAL ERROR LOG & STACK TRACE ####
        Timestamp: {error['timestamp']}
        Level: {error['level']}
        Message:
        {error['message']}

        #### RELEVANT SOURCE CODE from file '{analysis_data['culprit_file']}' ####
        ```python
        {source_code}
        ```

        ### CONTEXT END ###

        ### INSTRUCTIONS ###
        1. Analyze all the provided context.
        2. Think step-by-step to determine the single, most likely root cause of the error.
        3. Provide your final analysis ONLY in the following JSON format. Do not include any other text.

        {{
          "root_cause_analysis": "A detailed explanation of the root cause.",
          "culprit_file": "{analysis_data['culprit_file']}",
          "culprit_function_name": "The name of the function where the error occurred.",
          "confidence_score": 0.9
        }}
        """

    def _create_remediation_prompt(self, diagnosis: Dict, buggy_code: str) -> str:
        """Constructs a prompt to generate a code fix."""
        return f"""
        You are an expert Python developer specializing in writing safe, idiomatic, and bug-free code.
        Your task is to fix the provided buggy Python code based on the given root cause analysis.

        ### BUGGY CODE ###
        ```python
        {buggy_code}
        ```

        ### ROOT CAUSE ANALYSIS ###
        {diagnosis.get("root_cause_analysis", "No analysis provided.")}

        ### INSTRUCTIONS & CONSTRAINTS ###
        1. Rewrite the entire function to fix the bug described in the root cause analysis.
        2. The fix MUST be idiomatic Python and not introduce any new dependencies.
        3. You MUST NOT change the function's signature (name or arguments).
        4. Your response MUST ONLY contain the raw Python code for the corrected function. Do not include markdown, explanations, or any other text.

        ### CORRECTED CODE ###
        """

    def get_diagnosis_and_fix(self, analysis_data: Dict, repo_path: str) -> Optional:
        """
        Orchestrates the diagnosis and code generation process.
        """
        print("\n--- Starting AI Diagnosis and Code Generation ---")
        culprit_file_path = os.path.join(repo_path, analysis_data['culprit_file'])
        try:
            with open(culprit_file_path, 'r') as f:
                source_code = f.read()
        except FileNotFoundError:
            print(f"Error: Source code file not found at {culprit_file_path}")
            return None

        # 1. Get Diagnosis
        diagnostic_prompt = self._create_diagnostic_prompt(analysis_data, source_code)
        try:
            response = self.model.generate_content(diagnostic_prompt)
            # Clean up potential markdown formatting
            cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
            diagnosis = json.loads(cleaned_response_text)
            print(f"AI Diagnosis Received: {diagnosis['root_cause_analysis']}")
        except (Exception, json.JSONDecodeError) as e:
            print(f"Error during AI diagnosis: {e}")
            print(f"LLM Response was: {response.text}")
            return None

        # 2. Get Code Fix
        # For simplicity, we'll assume the whole file content is the buggy code scope
        # A more advanced agent would use AST to extract the specific function
        buggy_code = source_code
        remediation_prompt = self._create_remediation_prompt(diagnosis, buggy_code)
        try:
            response = self.model.generate_content(remediation_prompt)
            fixed_code = response.text.strip().replace("```python", "").replace("```", "").strip()
            print("AI-Generated Code Fix:\n" + "="*30 + f"\n{fixed_code}\n" + "="*30)
        except Exception as e:
            print(f"Error during code generation: {e}")
            return None

        diagnosis['original_code'] = source_code
        diagnosis['fixed_code'] = fixed_code
        return diagnosis

class CodePatcher:
    """
    Applies generated code fixes to the local file system.
    """
    def __init__(self, repo_path: str):
        self.repo_path = repo_path

    def apply_patch(self, culprit_file: str, new_content: str) -> Optional[str]:
        """
        Overwrites the culprit file with the new, fixed content.
        Returns the path to the patched file.
        """
        print("\n--- Applying Code Patch ---")
        file_path = os.path.join(self.repo_path, culprit_file)
        if not os.path.exists(file_path):
            print(f"Error: File to patch does not exist: {file_path}")
            return None
        
        try:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Successfully patched file: {file_path}")
            return file_path
        except IOError as e:
            print(f"Error writing to file {file_path}: {e}")
            return None

class GitManager:
    """
    Manages all Git operations using GitPython.
    """
    def __init__(self, repo_path: str):
        try:
            self.repo = Repo(repo_path)
        except Exception as e:
            print(f"Error initializing Git repository at {repo_path}: {e}")
            raise
        self.author = Actor("AIOps Agent", "aiops-agent@example.com")

    def create_commit_and_push(self, file_to_commit: str, rca: str) -> Optional[str]:
        """
        Creates a new branch, commits the patched file, and pushes to origin.
        """
        print("\n--- Starting Git Workflow ---")
        if self.repo.is_dirty(untracked_files=True):
            # 1. Create a new branch
            branch_name = f"agent-fix/remediation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            try:
                new_branch = self.repo.create_head(branch_name)
                new_branch.checkout()
                print(f"Created and checked out new branch: {branch_name}")

                # 2. Stage the patched file
                self.repo.index.add([file_to_commit])
                print(f"Staged file: {file_to_commit}")

                # 3. Create a structured commit message
                commit_message = f"fix: Automated remediation for system error\n\n"
                commit_message += f"This commit was generated automatically by the AIOps agent.\n\n"
                commit_message += f"Root Cause Analysis:\n{rca}"
                
                self.repo.index.commit(commit_message, author=self.author, committer=self.author)
                print("Committed changes.")

                # 4. Push the new branch to the remote
                origin = self.repo.remote(name='origin')
                origin.push(refspec=f'{branch_name}:{branch_name}')
                print(f"Pushed branch '{branch_name}' to remote.")
                
                # Return to main branch
                self.repo.heads.master.checkout()
                return branch_name

            except Exception as e:
                print(f"An error occurred during the Git workflow: {e}")
                return None
        else:
            print("No changes to commit. Working tree is clean.")
            return None

class NotificationManager:
    """
    Handles creation of pull requests and sending email notifications.
    """
    def __init__(self, config: Dict):
        self.config = config

    def create_pull_request(self, branch_name: str, rca: str) -> Optional[str]:
        """Creates a pull request on GitHub."""
        print("\n--- Creating GitHub Pull Request ---")
        url = f"https://api.github.com/repos/{self.config['owner']}/{self.config['repo']}/pulls"
        headers = {
            "Authorization": f"token {self.config['token']}",
            "Accept": "application/vnd.github.v3+json",
        }
        title = f"Automated Fix: {rca.splitlines()}"
        body = f"""
## Automated Remediation by AIOps Agent

**Root Cause Analysis:**
`````
{rca}
`````
**AI-Generated Code Fix:**
`````
# ...existing code...

try:
    print("Attempting to divide 10 by 0.")
    result = 10 / 0
except Exception as e:
    print("An error occurred. Check the log file 'app_error.log' for details.")
    with open('app_error.log', 'a') as log_file:
        log_file.write(f"{type(e).__name__}: {e}\n")

print("Application run finished.")