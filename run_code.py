# run_code.py

import os
import uuid
import requests
import re
import time
import logging
import subprocess
from typing import Dict
from datetime import datetime

from prompts import summarize_output, summarize_refinement
from prompts import extract_code_from_chunk
from docker_executor import update_executor_compose, force_cleanup_executor, EXECUTOR_COMPOSE_PATH
from faiss_cache import BASE_DIR
from colorama import Fore, Style

logger = logging.getLogger(__name__)

# Configuration
ENABLE_AUTO_RETRIES = True  # Set to False to disable automatic retries
MAX_RETRY_ATTEMPTS = 3      # Maximum number of retry attempts


def run_python_code_with_retries(code_str: str, user_prompt: str, max_retries: int = 3) -> Dict[str, str]:
    """
    Run Python code with automatic retries on failure.
    
    Args:
        code_str: The Python code to execute
        user_prompt: The original user prompt
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        Dictionary with execution results
    """
    attempt = 0
    best_result = None
    
    while attempt <= max_retries:
        # Execute the current code
        if attempt == 0:
            print_system_message(f"Executing initial code solution (attempt {attempt + 1}/{max_retries + 1})...")
        else:
            print_system_message(f"Executing refined solution (attempt {attempt + 1}/{max_retries + 1})...")
            
        exec_result = run_python_code_in_docker(code_str, user_prompt)
        
        # Check if execution was successful
        if evaluate_docker_output(exec_result["combined_output"]):
            # Success - return the result
            print_system_message("Execution successful!")
            return exec_result
        
        # Keep track of the best result so far (even if all fail)
        if best_result is None or (
            # Heuristic: fewer error messages might mean it's closer to working
            len(exec_result["combined_output"]) < len(best_result["combined_output"])
        ):
            best_result = exec_result
        
        # If we've reached max retries, break
        if attempt >= max_retries:
            break
            
        # Generate improved code for next attempt
        print_system_message(f"Execution failed. Generating improved solution...")
        
        # Generate improved prompt based on errors
        from prompts import generate_refined_prompt, stream_llm_response, ADVANCED_MODEL
        
        refined_messages = generate_refined_prompt(
            user_prompt=user_prompt,
            code_str=code_str,
            execution_result=exec_result,
            attempt=attempt + 1
        )
        
        # Get improved code
        print_system_message("Generating refined code solution...")
        refined_code = stream_llm_response(model=ADVANCED_MODEL, messages=refined_messages)
        
        # Update the code for the next attempt
        code_str = refined_code
        attempt += 1
        
        # Show a summary of what's being fixed
        from prompts import summarize_refinement
        refinement_summary = summarize_refinement(code_str, exec_result["combined_output"])
        print_system_message("Refinement approach:")
        print(refinement_summary)
        print()
    
    # If we get here, all attempts failed
    print_system_message(f"All {max_retries + 1} execution attempts failed. Returning best attempt.")
    return best_result

def run_python_code_in_service(code_str, user_prompt):
    """
    Run Python code using the persistent execution service
    instead of creating a new Docker container each time.
    """
    project_name = make_project_name(user_prompt)
    project_dir = prepare_project_dir(project_name)
    
    start_time = time.time()
    
    # Write code to file for reference
    script_path = os.path.join(project_dir, "script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_str)
    
    # Detect dependencies using the existing function
    packages = detect_missing_packages(code_str)
    
    # Ensure execution service is running
    from docker_executor import ensure_execution_service
    if not ensure_execution_service():
        return {
            "combined_output": "[Execution Error] Failed to start execution service.",
            "result_summary": "Execution service error.",
            "code_path": script_path
        }
    
    # Submit to execution service
    service_url = "http://localhost:5000"  # Adjust for your setup
    payload = {
        "code": code_str,
        "requirements": packages,
        "task_id": None  # Let the service generate one
    }
    
    try:
        # Submit job
        response = requests.post(f"{service_url}/execute", json=payload)
        response.raise_for_status()
        task_data = response.json()
        task_id = task_data["task_id"]
        
        # Poll for results (with timeout)
        timeout = 180  # 3 minutes
        while time.time() - start_time < timeout:
            result_response = requests.get(f"{service_url}/result/{task_id}")
            if result_response.status_code != 200:
                break
                
            result_data = result_response.json()
            if "status" not in result_data or result_data.get("status") != "running":
                # We have a result
                break
                
            time.sleep(1)  # Wait before polling again
        
        # Get final result (wait=true to ensure we get it)
        final_result = requests.get(f"{service_url}/result/{task_id}?wait=true").json()
        
        # Format combined output similar to current format
        combined_output = (
            f"```markdown\n"
            f"--- STDOUT ---\n{final_result.get('stdout', '')}\n"
            f"--- STDERR ---\n{final_result.get('stderr', '')}\n"
            f"--- RETURN CODE ---\n{final_result.get('returncode', '')}\n```"
        )
        
        # Process the results
        from prompts import summarize_output
        result_summary = summarize_output(combined_output)
        
        # Cleanup resources
        requests.post(f"{service_url}/cleanup/{task_id}")
        
        duration_ms = int((time.time() - start_time) * 1000)
        logger.debug(f"Code execution took {duration_ms}ms total.")
        
        write_result_files(project_dir, result_summary, combined_output)
        
        return {
            "combined_output": combined_output,
            "result_summary": result_summary,
            "code_path": script_path
        }
    except Exception as e:
        logger.error(f"Error executing code via execution service: {e}")
        return {
            "combined_output": f"[Execution Error] {str(e)}",
            "result_summary": f"Execution failed: {str(e)}",
            "code_path": script_path
        }

def print_system_message(message: str):
    """
    Helper to print a system-style message in cyan.
    """
    print(f"{Fore.CYAN}{message}{Style.RESET_ALL}")


def make_project_name(user_prompt: str) -> str:
    """
    Generate a short project name from the user prompt.
    """
    import re
    words = re.findall(r"[A-Za-z0-9]+", user_prompt)
    project_words = words[:4]
    project_name = "-".join(project_words).lower()
    return project_name or "default-project"


def prepare_project_dir(project_name: str) -> str:
    """
    Create a folder under docker_projects for this code run.
    """
    project_root = os.path.join(BASE_DIR, "docker_projects")
    full_path = os.path.join(project_root, project_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def detect_missing_packages(code_str: str):
    """
    Rudimentary logic to detect additional packages that might need installation.
    """
    import re
    std_libs = {
        "os", "time", "re", "json", "subprocess", "shutil", "uuid",
        "logging", "sqlite3", "datetime", "sys", "math"
    }

    pkgs = set()
    for line in code_str.splitlines():
        line = line.strip()
        if line.startswith("import "):
            # e.g. import cv2, re, time
            imports = line.replace("import ", "").split(",")
            for imp in imports:
                mod = imp.split()[0].strip()
                if mod not in std_libs:
                    if mod == "cv2":
                        pkgs.add("opencv-python")
                    else:
                        pkgs.add(mod)
        elif line.startswith("from "):
            # e.g. from requests import get
            parts = line.split()
            if len(parts) >= 2:
                mod = parts[1].strip()
                if mod not in std_libs:
                    if mod == "cv2":
                        pkgs.add("opencv-python")
                    else:
                        pkgs.add(mod)
    return list(pkgs)


def write_result_files(project_dir: str, result: str, scan: str):
    """
    Write result and scan files (like a moderation layer).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(project_dir, f"result_{timestamp}.txt")
    scan_path = os.path.join(project_dir, f"scan_{timestamp}.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result)

    with open(scan_path, "w", encoding="utf-8") as f:
        f.write(scan)

    logger.info(f"Wrote result to {result_path} & scan to {scan_path}")


def evaluate_docker_output(output: str) -> tuple[bool, str]:
    """
    Evaluates if docker execution was successful and extracts error details.
    
    Returns:
        Tuple of (success_bool, error_message)
    """
    # Check for obvious success signals
    if "exited with code 0" in output.lower():
        return True, ""
    
    # Extract error patterns
    error_patterns = [
        (r"ImportError: No module named '([^']+)'", "Missing module: {}"),
        (r"ModuleNotFoundError: No module named '([^']+)'", "Missing module: {}"),
        (r"NameError: name '([^']+)' is not defined", "Undefined variable: {}"),
        (r"SyntaxError: ([^(]+)", "Syntax error: {}"),
        (r"TypeError: ([^(]+)", "Type error: {}"),
        (r"IndexError: ([^(]+)", "Index error: {}"),
        (r"KeyError: ([^'\"]+)", "Key error: {}"),
        (r"FileNotFoundError: ([^(]+)", "File not found: {}"),
        (r"PermissionError: ([^(]+)", "Permission error: {}"),
        (r"ValueError: ([^(]+)", "Value error: {}"),
    ]
    
    error_messages = []
    for pattern, template in error_patterns:
        matches = re.findall(pattern, output)
        for match in matches:
            error_messages.append(template.format(match))
    
    if error_messages:
        return False, "; ".join(error_messages)
    
    # If no specific errors found but exit code wasn't 0
    return False, "Execution failed with non-zero exit code"

def run_python_code_in_docker(code_str, user_prompt):
    """
    Build & run user code using the persistent execution service
    instead of creating a new Docker container each time.
    """
    return run_python_code_in_service(code_str, user_prompt)