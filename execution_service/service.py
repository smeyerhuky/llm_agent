import os
import uuid
import subprocess
import logging
import shutil
import json
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class ExecutionService:
    def __init__(self, max_workers=4, venv_dir="/app/venvs", pip_cache="/app/pip_cache"):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.venv_dir = venv_dir
        self.pip_cache = pip_cache
        self.active_jobs = {}
        self.requirements_manifest = {}
        
        # Create necessary directories
        os.makedirs(venv_dir, exist_ok=True)
        os.makedirs(pip_cache, exist_ok=True)
        
        # Load existing requirements manifest if it exists
        manifest_path = os.path.join(pip_cache, "requirements_manifest.json")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'r') as f:
                    self.requirements_manifest = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load requirements manifest: {e}")
        
        logger.info(f"Execution service initialized with {max_workers} workers")
    
    def save_manifest(self):
        """Save the current requirements manifest to disk"""
        manifest_path = os.path.join(self.pip_cache, "requirements_manifest.json")
        try:
            with open(manifest_path, 'w') as f:
                json.dump(self.requirements_manifest, f)
        except Exception as e:
            logger.error(f"Failed to save requirements manifest: {e}")
    
    def execute_code(self, code_str, requirements, task_id=None):
        """Execute code in an isolated environment"""
        if task_id is None:
            task_id = str(uuid.uuid4())
            
        # Create a dedicated directory for this execution
        job_dir = os.path.join(self.venv_dir, task_id)
        os.makedirs(job_dir, exist_ok=True)
        
        # Write code and requirements
        with open(os.path.join(job_dir, "script.py"), "w") as f:
            f.write(code_str)
            
        with open(os.path.join(job_dir, "requirements.txt"), "w") as f:
            f.write("\n".join(requirements))
        
        # Install any new requirements to the global pip cache first
        self.update_global_requirements(requirements)
        
        # Submit execution task to thread pool
        future = self.executor.submit(
            self._run_in_isolated_env, 
            job_dir,
            task_id
        )
        self.active_jobs[task_id] = {
            "future": future,
            "job_dir": job_dir,
            "status": "running"
        }
        return task_id
    
    def update_global_requirements(self, requirements):
        """Update the global pip cache with new requirements"""
        new_reqs = []
        for req in requirements:
            req = req.strip()
            if req and req not in self.requirements_manifest:
                new_reqs.append(req)
                self.requirements_manifest[req] = True
        
        if new_reqs:
            logger.info(f"Installing new requirements: {new_reqs}")
            try:
                # Install to the global pip cache
                result = subprocess.run(
                    ["pip", "install"] + new_reqs,
                    capture_output=True,
                    text=True,
                    check=False,
                    env={"PIP_CACHE_DIR": self.pip_cache}
                )
                
                if result.returncode == 0:
                    self.save_manifest()
                    logger.info("Successfully updated global requirements")
                else:
                    logger.error(f"Failed to install requirements: {result.stderr}")
                    # Remove failed requirements from manifest
                    for req in new_reqs:
                        if req in self.requirements_manifest:
                            del self.requirements_manifest[req]
            except Exception as e:
                logger.error(f"Error updating global requirements: {e}")
    
    def _run_in_isolated_env(self, job_dir, task_id):
        """Run the code in an isolated environment using Python's subprocess"""
        try:
            # Run the script with restricted permissions
            # Note: We're using Python's built-in subprocess isolation
            # For production, consider using a more robust sandbox like bubblewrap or seccomp
            result = subprocess.run(
                [
                    "python", "-c",
                    f"import sys; sys.path.insert(0, '{job_dir}'); "
                    f"exec(open('{os.path.join(job_dir, 'script.py')}').read())"
                ],
                capture_output=True,
                text=True,
                check=False,
                cwd=job_dir,
                # Time limit - 60 seconds
                timeout=60,
                # Restricted env
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": job_dir,
                    "PIP_CACHE_DIR": self.pip_cache,
                    # Remove potentially sensitive environment variables
                    "HOME": job_dir,
                    "TEMP": job_dir,
                    "TMP": job_dir,
                }
            )
            
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Execution timed out (60s limit)",
                "returncode": 124,
                "success": False
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": 1,
                "success": False,
                "error": str(e)
            }
        finally:
            # Mark job as completed
            if task_id in self.active_jobs:
                self.active_jobs[task_id]["status"] = "completed"
    
    def get_result(self, task_id, wait=True):
        """Get the result of a task"""
        if task_id not in self.active_jobs:
            return {"error": "Task not found"}
        
        job = self.active_jobs[task_id]
        if wait:
            result = job["future"].result()
            return result
        elif job["future"].done():
            return job["future"].result()
        else:
            return {"status": "running"}
            
    def cleanup_task(self, task_id):
        """Clean up resources for a completed task"""
        if task_id in self.active_jobs:
            job_dir = self.active_jobs[task_id]["job_dir"]
            # Only clean up if the job is completed
            if self.active_jobs[task_id]["status"] == "completed":
                shutil.rmtree(job_dir, ignore_errors=True)
                del self.active_jobs[task_id]
                return True
        return False