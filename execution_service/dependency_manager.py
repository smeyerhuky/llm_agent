# dependency_manager.py
import os
import json
import subprocess
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class PackageState(Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLING = "installing"
    INSTALLED = "installed"
    FAILED = "failed"

class DependencyManager:
    def __init__(self, cache_dir="/app/pip_cache", manifest_path=None):
        """
        Initialize the dependency manager.
        
        Args:
            cache_dir: Directory to use for pip cache
            manifest_path: Path to the manifest file (defaults to cache_dir/manifest.json)
        """
        self.cache_dir = cache_dir
        self.manifest_path = manifest_path or os.path.join(cache_dir, "manifest.json")
        self.packages = {}
        self._load_manifest()
        
    def _load_manifest(self):
        """Load the package manifest from disk."""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)
                    self.packages = {
                        pkg: PackageState(state) for pkg, state in data.items()
                    }
                logger.info(f"Loaded {len(self.packages)} packages from manifest")
            except Exception as e:
                logger.error(f"Failed to load manifest: {e}")
                self.packages = {}
        else:
            logger.info(f"No manifest found at {self.manifest_path}")
            self.packages = {}
            
    def _save_manifest(self):
        """Save the package manifest to disk."""
        try:
            os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
            with open(self.manifest_path, 'w') as f:
                data = {
                    pkg: state.value for pkg, state in self.packages.items()
                }
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.packages)} packages to manifest")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            
    def get_package_state(self, package):
        """Get the state of a package."""
        return self.packages.get(package, PackageState.NOT_INSTALLED)
        
    def set_package_state(self, package, state):
        """Set the state of a package."""
        self.packages[package] = state
        self._save_manifest()
        
    def ensure_packages(self, packages):
        """
        Ensure all packages are installed. Returns a list of failed packages.
        
        Args:
            packages: List of package names to install
            
        Returns:
            List of packages that failed to install
        """
        # Filter out packages that are already installed or being installed
        to_install = []
        for pkg in packages:
            pkg = pkg.strip()
            if not pkg:
                continue
                
            state = self.get_package_state(pkg)
            if state in (PackageState.NOT_INSTALLED, PackageState.FAILED):
                to_install.append(pkg)
                self.set_package_state(pkg, PackageState.INSTALLING)
                
        if not to_install:
            logger.info("All packages already installed")
            return []
            
        # Install packages
        failed = []
        try:
            logger.info(f"Installing packages: {to_install}")
            # Use Python's sys.executable to get the path to the Python interpreter
            python_path = subprocess.check_output(
                ["which", "python"], 
                text=True
            ).strip()
            pip_path = os.path.join(os.path.dirname(python_path), "pip")
            
            # Make sure pip exists
            if not os.path.exists(pip_path):
                pip_path = os.path.join(os.path.dirname(python_path), "pip3")
                
            if not os.path.exists(pip_path):
                # Try using Python -m pip
                cmd = [python_path, "-m", "pip", "install"] + to_install
            else:
                cmd = [pip_path, "install"] + to_install
                
            # Add pip cache directory
            cmd.extend(["--cache-dir", self.cache_dir])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install packages: {result.stderr}")
                # Mark each package as failed
                for pkg in to_install:
                    self.set_package_state(pkg, PackageState.FAILED)
                    failed.append(pkg)
            else:
                # Mark each package as installed
                for pkg in to_install:
                    self.set_package_state(pkg, PackageState.INSTALLED)
                logger.info("Successfully installed packages")
        except Exception as e:
            logger.error(f"Error installing packages: {e}")
            # Mark each package as failed
            for pkg in to_install:
                self.set_package_state(pkg, PackageState.FAILED)
                failed.append(pkg)
                
        return failed
        
    def get_all_installed_packages(self):
        """Get a list of all installed packages."""
        return [
            pkg for pkg, state in self.packages.items() 
            if state == PackageState.INSTALLED
        ]