"""Stress test tool for comparing solution with brute force."""

from codeforce_fucker.tools.executor import execute_code

_timeout_seconds: float = 5.0
_memory_mb: int = 256


def stress_test(
    solution_code: str,
    brute_force_code: str,
    generator_code: str,
    num_tests: int = 2000,
) -> str:
    """对拍验证工具：比较优化算法和暴力算法的输出。"""
    for i in range(num_tests):
        gen_result = execute_code(
            code=generator_code,
            stdin="",
            timeout_seconds=_timeout_seconds,
            memory_mb=_memory_mb,
        )
        if gen_result.status != "passed":
            return f"""=== GENERATOR ERROR ===
Test #{i + 1}
Error:
{gen_result.error_message or 'Unknown error'}
Stdout:
{(gen_result.actual_output or '').strip()}
Status: {gen_result.status}"""

        test_input = gen_result.actual_output or ""

        brute_result = execute_code(
            code=brute_force_code,
            stdin=test_input,
            timeout_seconds=_timeout_seconds * 10,
            memory_mb=_memory_mb,
        )
        if brute_result.status != "passed":
            return f"""=== BRUTE FORCE ERROR ===
Test #{i + 1}
Input:
{test_input}
Error:
{brute_result.error_message or 'Unknown error'}
Stdout:
{(brute_result.actual_output or '').strip()}
Status: {brute_result.status}"""

        sol_result = execute_code(
            code=solution_code,
            stdin=test_input,
            timeout_seconds=_timeout_seconds,
            memory_mb=_memory_mb,
        )
        if sol_result.status != "passed":
            return f"""=== SOLUTION ERROR ===
Test #{i + 1}
Input:
{test_input}
Error:
{sol_result.error_message or 'Unknown error'}
Stdout:
{(sol_result.actual_output or '').strip()}
Status: {sol_result.status}"""

        brute_out = (brute_result.actual_output or "").strip()
        sol_out = (sol_result.actual_output or "").strip()

        if brute_out != sol_out:
            return f"""=== COUNTEREXAMPLE FOUND ===
Test #{i + 1} failed!

Input:
{test_input}

Brute Force Output:
{brute_out}

Solution Output:
{sol_out}

请分析差异并修正你的优化算法。"""

    return f"""=== STRESS TEST PASSED ===
All {num_tests} tests passed!
Your solution matches the brute force on all random inputs."""
