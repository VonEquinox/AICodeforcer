"""Run Python code tool for Gemini Function Calling."""

from codeforce_fucker.tools.executor import execute_code

_timeout_seconds: float = 5.0
_memory_mb: int = 256
_max_output_chars: int = 32000


def configure_executor(timeout_seconds: float = 5.0, memory_mb: int = 256) -> None:
    global _timeout_seconds, _memory_mb
    _timeout_seconds = timeout_seconds
    _memory_mb = memory_mb


def run_python_code(code: str, test_input: str = "") -> str:
    """Execute Python code with the given input and return the result."""
    result = execute_code(
        code=code,
        stdin=test_input,
        timeout_seconds=_timeout_seconds,
        memory_mb=_memory_mb,
    )

    lines = ["=== Execution Result ==="]
    lines.append(f"Status: {result.status.upper().replace('_', ' ')}")

    if result.actual_output is not None:
        output = result.actual_output
        if len(output) > _max_output_chars:
            output = output[:_max_output_chars] + f"\n... (truncated, total {len(result.actual_output)} chars)"
        lines.append(f"Output:\n{output}")

    if result.error_message:
        lines.append(f"Error:\n{result.error_message}")

    lines.append(f"Execution Time: {result.execution_time_ms:.1f}ms")

    return "\n".join(lines)
