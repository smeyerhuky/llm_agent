# run_code.py

import os
import uuid
import time
import logging
import subprocess
from typing import Dict
from datetime import datetime

from prompts import summarize_output, refine_code, summarize_refinement
from prompts import extract_code_from_chunk
from docker_executor import update_executor_compose, force_cleanup_executor, EXECUTOR_COMPOSE_PATH
from faiss_cache import BASE_DIR
from colorama import Fore, Style

logger = logging.getLogger(__name__)

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


def evaluate_docker_output(output: str) -> bool:
    """
    We assume success if 'exited with code 0' is found in logs.
    """
    return "exited with code 0" in output.lower()


def run_python_code_in_docker(code_str: str, user_prompt: str) -> Dict[str, str]:
    """
    Build & run user code in a Docker container, capturing output and summarizing results.
    Returns { "combined_output", "result_summary", "code_path" }.
    """
    project_name = make_project_name(user_prompt)
    project_dir = prepare_project_dir(project_name)
    unique_id = uuid.uuid4().hex[:6]
    service_name = f"user_service_{project_name}_{unique_id}"
    logger.debug(f"Generated service name: {service_name}")

    start_time = time.time()

    script_path = os.path.join(project_dir, "script.py")
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code_str)
    code_path = script_path
    logger.debug(f"Wrote code to {script_path}")

    reqs_path = os.path.join(project_dir, "requirements.txt")
    packages = detect_missing_packages(code_str)
    with open(reqs_path, "w", encoding="utf-8") as f:
        for pkg in packages:
            f.write(pkg + "\n")
        f.write("requests\n")
    logger.debug(f"Wrote requirements.txt with packages: {packages}")

    update_executor_compose(service_name, project_dir)
    if not os.path.exists(EXECUTOR_COMPOSE_PATH):
        logger.error("docker-compose-executor.yml missing; can't proceed.")
        return {
            "combined_output": "[Execution Error] Executor compose file missing.",
            "result_summary": "",
            "code_path": code_path
        }

    # 1) pip install
    cmd_install = [
        "docker-compose", "-f", EXECUTOR_COMPOSE_PATH, "run", "--rm",
        service_name, "pip", "install", "-r", "requirements.txt"
    ]
    logger.debug(f"Running pip install via: {cmd_install}")

    try:
        proc_install = subprocess.run(cmd_install, cwd=project_dir, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {
            "combined_output": "[Execution Error] Docker container timed out during pip install.",
            "result_summary": "",
            "code_path": code_path
        }

    install_stdout, install_stderr = proc_install.stdout, proc_install.stderr

    # 2) docker-compose up
    cmd_up = [
        "docker-compose", "-f", EXECUTOR_COMPOSE_PATH, "up", "--build",
        "--abort-on-container-exit", "--always-recreate-deps", service_name
    ]
    logger.debug(f"Running code container via: {cmd_up}")

    try:
        proc_up = subprocess.run(cmd_up, cwd=project_dir, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return {
            "combined_output": "[Execution Error] Docker container timed out while running script.",
            "result_summary": "",
            "code_path": code_path
        }
    finally:
        # Always attempt to shut down
        try:
            subprocess.run(["docker-compose", "-f", EXECUTOR_COMPOSE_PATH, "down"], cwd=project_dir)
            logger.info("Docker Compose execution cleaned up.")
        except Exception as cleanup_e:
            logger.warning(f"Cleanup error: {cleanup_e}")

    stdout, stderr = proc_up.stdout, proc_up.stderr
    if "Conflict" in stderr:
        logger.warning(f"Container name conflict for {service_name}. Attempting forced removal.")
        subprocess.run(["docker", "rm", "-f", service_name])

    # Combine logs
    combined_output = (
        f"```markdown\n"
        f"--- PIP INSTALL STDOUT ---\n{install_stdout}\n"
        f"--- PIP INSTALL STDERR ---\n{install_stderr}\n\n"
        f"--- DOCKER COMPOSE STDOUT ---\n{stdout}\n"
        f"--- DOCKER COMPOSE STDERR ---\n{stderr}\n```"
    )

    from prompts import summarize_output
    result_summary = summarize_output(combined_output)
    print_system_message(f"**RESULT:** {result_summary}")
    print(combined_output)
    duration_ms = int((time.time() - start_time) * 1000)
    logger.debug(f"Docker execution took {duration_ms}ms total.")

    write_result_files(project_dir, result_summary, combined_output)

    # Check for success
    if not evaluate_docker_output(combined_output):
        refine_choice = input("Execution output indicates issues. Refine and re-run? (y/n): ").strip().lower()
        if refine_choice == "y":
            diff_summary = summarize_refinement(code_str, combined_output)
            print_system_message("Refinement Summary:")
            print(diff_summary)
            refined = refine_code(code_str, combined_output, user_prompt)
            print_system_message("Refined code generated. Re-running execution...\n")
            return run_python_code_in_docker(refined, user_prompt)

    return {
        "combined_output": combined_output,
        "result_summary": result_summary,
        "code_path": code_path
    }