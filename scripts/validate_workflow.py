#!/usr/bin/env python3
"""
Validate GitHub Actions workflow files for common issues.
"""
import yaml
import sys
import re
from pathlib import Path


def check_file_format(filepath):
    """Check file encoding and line endings."""
    issues = []
    warnings = []
    
    with open(filepath, 'rb') as f:
        content = f.read()
    
    # Check for CRLF (Windows line endings)
    if b'\r\n' in content:
        issues.append("File has CRLF (Windows) line endings - should use LF (Unix) for GitHub Actions")
    
    # Check for final newline
    if not content.endswith(b'\n'):
        issues.append("File missing newline at end - required by POSIX")
    
    # Check for BOM
    if content.startswith(b'\xef\xbb\xbf'):
        warnings.append("File has UTF-8 BOM - not necessary for YAML")
    
    return issues, warnings


def check_yaml_syntax(filepath):
    """Validate YAML syntax."""
    issues = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        issues.append(f"YAML syntax error: {e}")
    
    return issues


def check_workflow_structure(filepath):
    """Validate GitHub Actions workflow structure."""
    issues = []
    warnings = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
    except Exception as e:
        return [f"Cannot parse workflow: {e}"], []
    
    # Check required top-level keys
    if 'name' not in workflow:
        warnings.append("Missing 'name' key (recommended but not required)")
    
    if 'on' not in workflow:
        issues.append("Missing required 'on' key (workflow triggers)")
    
    if 'jobs' not in workflow:
        issues.append("Missing required 'jobs' key")
        return issues, warnings
    
    # Validate jobs
    jobs = workflow.get('jobs', {})
    if not isinstance(jobs, dict):
        issues.append("'jobs' must be a dictionary")
        return issues, warnings
    
    if not jobs:
        issues.append("Workflow must have at least one job")
        return issues, warnings
    
    # Check each job
    for job_name, job in jobs.items():
        if not isinstance(job, dict):
            issues.append(f"Job '{job_name}' must be a dictionary")
            continue
        
        # Check required job keys
        if 'runs-on' not in job:
            issues.append(f"Job '{job_name}' missing required 'runs-on' key")
        
        if 'steps' not in job:
            issues.append(f"Job '{job_name}' missing required 'steps' key")
            continue
        
        # Check steps
        steps = job.get('steps', [])
        if not isinstance(steps, list):
            issues.append(f"Job '{job_name}' steps must be a list")
            continue
        
        if not steps:
            warnings.append(f"Job '{job_name}' has no steps")
            continue
        
        # Validate each step
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                issues.append(f"Job '{job_name}' step {i} must be a dictionary")
                continue
            
            # Each step must have either 'uses' or 'run'
            if 'uses' not in step and 'run' not in step:
                issues.append(f"Job '{job_name}' step {i} must have either 'uses' or 'run'")
            
            # Check for both uses and run (conflicting)
            if 'uses' in step and 'run' in step:
                issues.append(f"Job '{job_name}' step {i} cannot have both 'uses' and 'run'")
    
    return issues, warnings


def check_shell_scripts(filepath):
    """Check shell scripts in run blocks for common issues."""
    issues = []
    warnings = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            workflow = yaml.safe_load(f)
    except:
        return [], []  # Already caught by YAML validation
    
    for job_name, job in workflow.get('jobs', {}).items():
        for i, step in enumerate(job.get('steps', [])):
            if 'run' in step:
                script = step['run']
                step_name = step.get('name', f'step {i}')
                
                # Check for unmatched quotes
                single_quotes = script.count("'")
                double_quotes = script.count('"')
                
                if single_quotes % 2 != 0:
                    warnings.append(f"{job_name}/{step_name}: Unmatched single quotes in run block")
                if double_quotes % 2 != 0:
                    warnings.append(f"{job_name}/{step_name}: Unmatched double quotes in run block")
                
                # Check for common shell errors
                if 'fi' in script.split():
                    # Check if there's a matching if
                    if_count = len(re.findall(r'\bif\b', script))
                    fi_count = len(re.findall(r'\bfi\b', script))
                    if fi_count > if_count:
                        issues.append(f"{job_name}/{step_name}: Extra 'fi' without matching 'if'")
                
                # Check for set -e or set -euo pipefail (best practice)
                if '\n' in script and 'set -' not in script:
                    warnings.append(f"{job_name}/{step_name}: Multi-line script without 'set -e' (may hide errors)")
    
    return issues, warnings


def check_github_expressions(filepath):
    """Check for common issues with GitHub Actions expressions."""
    issues = []
    warnings = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for secret references
    secrets = re.findall(r'\$\{\{\s*secrets\.(\w+)\s*\}\}', content)
    if secrets:
        non_default = [s for s in secrets if s != 'GITHUB_TOKEN']
        if non_default:
            warnings.append(f"Workflow uses custom secrets: {', '.join(non_default)} - ensure they're configured")
    
    # Check for potentially undefined environment variables
    env_vars = re.findall(r'\$\{\{\s*env\.(\w+)\s*\}\}', content)
    if env_vars:
        warnings.append(f"Workflow references env variables: {', '.join(set(env_vars))} - ensure they're defined")
    
    return issues, warnings


def validate_workflow(filepath):
    """Run all validation checks."""
    filepath = Path(filepath)
    
    print(f"Validating: {filepath}")
    print("=" * 70)
    
    all_issues = []
    all_warnings = []
    
    # Run checks
    checks = [
        ("File Format", check_file_format),
        ("YAML Syntax", check_yaml_syntax),
        ("Workflow Structure", check_workflow_structure),
        ("Shell Scripts", check_shell_scripts),
        ("GitHub Expressions", check_github_expressions),
    ]
    
    for check_name, check_func in checks:
        print(f"\n{check_name}...")
        result = check_func(filepath)
        
        if len(result) == 2:
            issues, warnings = result
        else:
            issues = result
            warnings = []
        
        if issues:
            all_issues.extend(issues)
            print(f"  ❌ {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"     - {issue}")
        elif warnings:
            all_warnings.extend(warnings)
            print(f"  ⚠️  {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"     - {warning}")
        else:
            print(f"  ✓ OK")
    
    # Summary
    print("\n" + "=" * 70)
    if all_issues:
        print(f"❌ VALIDATION FAILED: {len(all_issues)} issue(s) found")
        print("\nCritical issues that must be fixed:")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        return False
    elif all_warnings:
        print(f"✓ Validation passed with {len(all_warnings)} warning(s)")
        print("\nWarnings (workflow will work but could be improved):")
        for i, warning in enumerate(all_warnings, 1):
            print(f"  {i}. {warning}")
        return True
    else:
        print("✓ Validation passed - no issues found!")
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python validate_workflow.py <workflow.yml>")
        sys.exit(1)
    
    success = validate_workflow(sys.argv[1])
    sys.exit(0 if success else 1)
