"""Sandboxed code executor with resource limits."""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from AICodeforcer.types import ExecutionResult

CODE_WRAPPER_TEMPLATE = '''
import sys
import resource

# Conservative recursion limit to reduce crash risk
sys.setrecursionlimit(10000)

# Disable network access inside sandbox (best-effort)
try:
    import socket as _socket
    def _deny_network(*args, **kwargs):
        raise RuntimeError('Network access is disabled in the sandbox')
    _socket.socket = _deny_network
    _socket.create_connection = _deny_network
except Exception:
    pass

{user_code}
'''

# Max bytes captured from child stdout/stderr to protect parent memory
_MAX_STDOUT_BYTES = 256 * 1024  # 256 KB
_MAX_STDERR_BYTES = 128 * 1024  # 128 KB


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

        # Limit file size (1MB)
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        except ValueError:
            pass

        # Disallow spawning subprocesses
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
        except (ValueError, AttributeError):
            pass

        # Limit number of open file descriptors
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        except (ValueError, AttributeError):
            pass

        # Disable core dumps
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except (ValueError, AttributeError):
            pass

    return set_limits


def execute_code(
    code: str,
    stdin: str,
    timeout_seconds: float = 2.0,
    memory_mb: int = 256,
) -> ExecutionResult:
    """Execute Python code in a sandboxed environment with output caps."""
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
                text=False,  # Read raw bytes to control size
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

        # Truncate output to protect memory
        stdout_bytes = result.stdout[:_MAX_STDOUT_BYTES] if result.stdout else b""
        stderr_bytes = result.stderr[:_MAX_STDERR_BYTES] if result.stderr else b""

        stdout_text = stdout_bytes.decode(errors="replace").strip()
        stderr_text = stderr_bytes.decode(errors="replace").strip()

        if result.stdout and len(result.stdout) > _MAX_STDOUT_BYTES:
            stdout_text += f"\n... (stdout truncated, total {len(result.stdout)} bytes)"
        if result.stderr and len(result.stderr) > _MAX_STDERR_BYTES:
            stderr_text += f"\n... (stderr truncated, total {len(result.stderr)} bytes)"

        if result.returncode == 0:
            return ExecutionResult(
                status="passed",
                actual_output=stdout_text,
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
                actual_output=stdout_text or None,
                error_message=stderr_text or f"Exit code: {result.returncode}",
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
