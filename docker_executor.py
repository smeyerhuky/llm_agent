# docker_executor.py

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.getenv("BASE_DIR", ".")
EXECUTOR_COMPOSE_PATH = os.path.abspath(os.path.join(BASE_DIR, "docker-compose-executor.yml"))

def ensure_execution_service():
    """Ensure that the execution service is running"""
    logger.info("Checking execution service status...")
    try:
        subprocess.run([
            "docker-compose", 
            "-f", 
            EXECUTOR_COMPOSE_PATH, 
            "ps", 
            "-q", 
            "execution_service"
        ], check=True, capture_output=True)
        
        output = subprocess.check_output([
            "docker-compose", 
            "-f", 
            EXECUTOR_COMPOSE_PATH, 
            "ps", 
            "-q", 
            "execution_service"
        ]).decode().strip()
        
        if output:
            logger.info("Execution service is already running.")
            return True
        else:
            logger.info("Execution service not running, starting it...")
    except Exception as e:
        logger.info(f"Execution service check failed: {e}, starting it...")
    
    try:
        subprocess.run([
            "docker-compose", 
            "-f", 
            EXECUTOR_COMPOSE_PATH, 
            "up", 
            "-d", 
            "execution_service"
        ], check=True)
        logger.info("Execution service started successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to start execution service: {e}")
        return False

def force_cleanup_executor():
    """Forcefully cleanup the docker-compose executor environment but keep the execution service."""
    logger.info("Force cleaning up docker-compose executor (except execution_service)...")
    try:
        # Get all services except execution_service
        output = subprocess.check_output([
            "docker-compose", 
            "-f", 
            EXECUTOR_COMPOSE_PATH, 
            "ps", 
            "-q"
        ]).decode().strip()
        
        # We're now using a persistent service, so only remove other services if needed
        if output:
            services = output.split("\n")
            for service in services:
                if service and service != "llm_agent_executor":
                    try:
                        subprocess.run(["docker", "rm", "-f", service], check=True)
                    except Exception as e:
                        logger.warning(f"Error removing container {service}: {e}")
        
        logger.info("Docker Compose executor cleaned up.")
    except Exception as e:
        logger.error(f"Error during force cleanup: {e}")


def update_executor_compose(service_name: str, project_dir: str):
    """
    Update the docker-compose-executor.yml file so that
    a new service is defined that can run the user code.
    """
    if not os.path.exists(EXECUTOR_COMPOSE_PATH):
        content = f"""version: '3.8'
services:
  {service_name}:
    image: python:3.9-slim
    container_name: {service_name}
    working_dir: /app
    volumes:
      - {project_dir}:/app
      - {os.path.join(BASE_DIR, 'pip_cache')}:/root/.cache/pip
    command: ["python", "script.py"]

volumes:
  pip_cache: {{}}
"""
        with open(EXECUTOR_COMPOSE_PATH, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Created executor compose file with service {service_name}.")
    else:
        with open(EXECUTOR_COMPOSE_PATH, "r", encoding="utf-8") as f:
            original = f.read()

        if service_name in original:
            logger.debug(f"Service {service_name} already exists in the compose file.")
            return

        lines = original.splitlines()
        new_lines = []
        inserted = False
        for line in lines:
            new_lines.append(line)
            if not inserted and line.strip() == "services:":
                new_service_def = f"""  {service_name}:
    image: python:3.9-slim
    container_name: {service_name}
    working_dir: /app
    volumes:
      - {project_dir}:/app
      - {os.path.join(BASE_DIR, 'pip_cache')}:/root/.cache/pip
    command: ["python", "script.py"]
"""
                new_lines.append(new_service_def)
                inserted = True

        new_content = "\n".join(new_lines)
        with open(EXECUTOR_COMPOSE_PATH, "w", encoding="utf-8") as f:
            f.write(new_content)
        logger.debug(f"Appended new service {service_name} to docker-compose-executor.yml.")