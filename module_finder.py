import modulefinder
import sys
import pkgutil

def get_non_standard_modules(script_path):
    finder = modulefinder.ModuleFinder()
    finder.run_script(script_path)

    # Get all imported modules
    imported_modules = set(finder.modules.keys())

    # Get standard library modules
    std_lib_modules = {module.name for module in pkgutil.iter_modules()}
    
    # Find non-standard modules
    non_standard = imported_modules - std_lib_modules
    return non_standard

script_path = "your_script.py"  # Change this to the target script
missing_modules = get_non_standard_modules(script_path)

print("Non-standard libraries that may need installation:", missing_modules)
