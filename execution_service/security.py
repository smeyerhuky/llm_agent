import re

def scan_code_for_security_issues(code_str):
    """
    Scan code for potential security issues
    Returns (is_safe, message)
    """
    security_issues = []
    
    # Check for potentially dangerous imports
    dangerous_imports = [
        r'import\s+os(?:\s|$|\.)',
        r'from\s+os\s+import',
        r'import\s+subprocess(?:\s|$|\.)',
        r'from\s+subprocess\s+import',
        r'import\s+sys(?:\s|$|\.)',
        r'from\s+sys\s+import',
        r'import\s+shutil(?:\s|$|\.)',
        r'from\s+shutil\s+import',
        r'__import__\s*\(',
    ]
    
    for pattern in dangerous_imports:
        if re.search(pattern, code_str):
            security_issues.append(f"Contains potentially dangerous import: {pattern}")
    
    # Check for file operations
    file_operations = [
        r'open\s*\(',
        r'with\s+open',
        r'\.write\s*\(',
        r'\.read\s*\(',
        r'\.readline\s*\(',
    ]
    
    for pattern in file_operations:
        if re.search(pattern, code_str):
            security_issues.append(f"Contains file operations: {pattern}")
    
    # Check for network operations
    network_operations = [
        r'socket\.',
        r'urllib',
        r'requests\.',
        r'http\.',
    ]
    
    for pattern in network_operations:
        if re.search(pattern, code_str):
            security_issues.append(f"Contains network operations: {pattern}")
    
    # Check for eval/exec
    dangerous_functions = [
        r'eval\s*\(',
        r'exec\s*\(',
        r'compile\s*\(',
    ]
    
    for pattern in dangerous_functions:
        if re.search(pattern, code_str):
            security_issues.append(f"Contains dangerous function calls: {pattern}")
    
    # Decision based on findings
    is_safe = len(security_issues) == 0
    
    if is_safe:
        return True, "Code passed security scan"
    else:
        return False, "Security issues detected: " + ", ".join(security_issues)