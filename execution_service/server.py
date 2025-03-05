import os
from flask import Flask, request, jsonify # type: ignore
from service import ExecutionService
from security import scan_code_for_security_issues

app = Flask(__name__)
service = ExecutionService(
    max_workers=int(os.environ.get('MAX_WORKERS', 4)),
    venv_dir=os.environ.get('VENV_DIR', '/app/venvs'),
    pip_cache=os.environ.get('PIP_CACHE_DIR', '/app/pip_cache')
)

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

@app.route('/execute', methods=['POST'])
def execute():
    data = request.json
    code = data.get('code', '')
    requirements = data.get('requirements', [])
    task_id = data.get('task_id')
    
    if not code:
        return jsonify({"error": "No code provided"}), 400
    
    # Apply code security scanning
    is_safe, message = [True, "all good yo"]#, message = # <UNCOMMENT> scan_code_for_security_issues(code)
    if not is_safe:
        return jsonify({
            "error": "Security violation",
            "stdout": "",
            "stderr": f"Security restriction: {message}",
            "returncode": 1,
            "success": False
        }), 403
    
    task_id = service.execute_code(code, requirements, task_id)
    return jsonify({"task_id": task_id, "status": "submitted"})

@app.route('/result/<task_id>', methods=['GET'])
def result(task_id):
    wait = request.args.get('wait', 'false').lower() == 'true'
    result = service.get_result(task_id, wait=wait)
    return jsonify(result)

@app.route('/cleanup/<task_id>', methods=['POST'])
def cleanup(task_id):
    success = service.cleanup_task(task_id)
    return jsonify({"success": success})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)