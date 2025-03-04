# docker_executor.py

import os
import subprocess
import logging

logger = logging.getLogger(__name__)

BASE_DIR = os.getenv("BASE_DIR", ".")
EXECUTOR_COMPOSE_PATH = os.path.abspath(os.path.join(BASE_DIR, "docker-compose-executor.yml"))

def force_cleanup_executor():
    """
    Forcefully cleanup the docker-compose executor environment.
    """
    logger.info("Force cleaning up docker-compose executor...")
    try:
        subprocess.run(["docker-compose", "-f", EXECUTOR_COMPOSE_PATH, "down"], check=True)
        logger.info("Docker Compose executor cleaned up.")
    except Exception as e:
        logger.error(f"Error during force cleanup: {e}")

    default_template = """version: '3.8'
services:
  # Dynamically added services will appear here
volumes:
  pip_cache: {}
"""

    try:
        with open(EXECUTOR_COMPOSE_PATH, "w", encoding="utf-8") as f:
            f.write(default_template)
        logger.info("Executor compose file reset to default template.")
    except Exception as e:
        logger.error(f"Error resetting executor compose file: {e}")


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