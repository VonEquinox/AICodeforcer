"""Sandboxed code executor with resource limits."""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from codeforce_fucker.types import ExecutionResult

CODE_WRAPPER_TEMPLATE = '''
import sys
import resource

sys.setrecursionlimit(200000)

try:
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
except (ValueError, resource.error):
    pass

{user_code}
'''


def _create_resource_limiter(timeout_seconds: float, memory_mb: int):
    def set_limits():
        import resource

        cpu_limit = int(timeout_seconds) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))

        memory_bytes = memory_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except ValueError:
            pass

        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        except ValueError:
            pass

        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, AttributeError):
            pass

    return set_limits


def execute_code(
    code: str,
    stdin: str,
    timeout_seconds: float = 2.0,
    memory_mb: int = 256,
) -> ExecutionResult:
    """Execute Python code in a sandboxed environment."""
    wrapped_code = CODE_WRAPPER_TEMPLATE.format(user_code=code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapped_code)
        code_file = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(stdin)
        input_file = f.name

    try:
        start_time = time.perf_counter()

        preexec = None
        if sys.platform != "win32":
            preexec = _create_resource_limiter(timeout_seconds, memory_mb)

        with open(input_file, "r") as stdin_f:
            result = subprocess.run(
                [sys.executable, code_file],
                stdin=stdin_f,
                capture_output=True,
                text=True,
                timeout=timeout_seconds + 1,
                preexec_fn=preexec,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": "",
                    "HOME": tempfile.gettempdir(),
                },
            )

        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        if result.returncode == 0:
            return ExecutionResult(
                status="passed",
                actual_output=result.stdout.strip(),
                execution_time_ms=execution_time_ms,
            )
        elif result.returncode == -signal.SIGKILL:
            return ExecutionResult(
                status="memory_exceeded",
                error_message="Process killed (likely out of memory)",
                execution_time_ms=execution_time_ms,
            )
        elif result.returncode == -signal.SIGXCPU:
            return ExecutionResult(
                status="timeout",
                error_message=f"CPU time limit exceeded ({timeout_seconds}s)",
                execution_time_ms=execution_time_ms,
            )
        else:
            return ExecutionResult(
                status="runtime_error",
                actual_output=result.stdout.strip() if result.stdout else None,
                error_message=result.stderr.strip() if result.stderr else f"Exit code: {result.returncode}",
                execution_time_ms=execution_time_ms,
            )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            status="timeout",
            error_message=f"Time limit exceeded ({timeout_seconds}s)",
        )

    except Exception as e:
        return ExecutionResult(
            status="runtime_error",
            error_message=str(e),
        )

    finally:
        try:
            Path(code_file).unlink(missing_ok=True)
            Path(input_file).unlink(missing_ok=True)
        except Exception:
            pass
