"""Interactive stress test tool for validating interactive solutions."""

from AICodeforcer.interactive.tools.interaction_runner import run_interaction
from AICodeforcer.standard.tools.executor import execute_code

# 日志截断配置
_LOG_MAX_CHARS = 3500


def _truncate_interaction_log(log: str, max_chars: int = _LOG_MAX_CHARS) -> str:
    """智能截断交互日志，保留关键信息。

    策略：
    1. 优先保留 STDERR 和 INFO 行（错误信息）
    2. 保留第一轮交互（了解初始状态）
    3. 保留最后几轮交互（通常是出错的地方）
    4. 截断过长的单行
    """
    if len(log) <= max_chars:
        return log

    lines = log.splitlines()
    original_len = len(log)
    original_line_count = len(lines)

    # 识别交互轮次：每轮以 [JUDGE -> SOLVER] 开始
    round_starts = [i for i, line in enumerate(lines) if line.startswith("[JUDGE -> SOLVER]")]

    # 构建轮次范围
    rounds: list[tuple[int, int]] = []
    for idx, start in enumerate(round_starts):
        end = round_starts[idx + 1] if idx + 1 < len(round_starts) else len(lines)
        rounds.append((start, end))

    # 选择要保留的行
    keep = [False] * len(lines)

    # 始终保留 STDERR 和 INFO 行
    for i, line in enumerate(lines):
        if "STDERR" in line or line.startswith("[INFO]"):
            keep[i] = True

    # 保留第一轮和最后几轮
    if rounds:
        # 第一轮
        start, end = rounds[0]
        for i in range(start, end):
            keep[i] = True
        # 最后 3 轮（或更少）
        for start, end in rounds[-3:]:
            for i in range(start, end):
                keep[i] = True
    else:
        # 没有识别到轮次，保留头尾各 20 行
        for i in range(min(20, len(lines))):
            keep[i] = True
        for i in range(max(0, len(lines) - 20), len(lines)):
            keep[i] = True

    # 构建输出
    output_lines = [
        "=== 日志已截断 ===",
        f"原始: {original_len} 字符, {original_line_count} 行, {len(rounds)} 轮交互",
    ]
    if rounds:
        output_lines.append(f"保留: 第1轮 + 最后{min(3, len(rounds))}轮 + STDERR/INFO")
    else:
        output_lines.append("保留: 头尾各20行 + STDERR/INFO")
    output_lines.append("")

    prev_idx = -1
    for i, line in enumerate(lines):
        if not keep[i]:
            continue
        # 显示省略标记
        if prev_idx >= 0 and i > prev_idx + 1:
            gap = i - prev_idx - 1
            output_lines.append(f"... 省略 {gap} 行 ...")
        # 截断过长的单行
        if len(line) > 200:
            line = line[:200] + " ...(行截断)"
        output_lines.append(line)
        prev_idx = i

    result = "\n".join(output_lines)

    # 最终硬截断保护
    if len(result) > max_chars:
        marker = "\n...(截断)...\n"
        # 确保 max_chars 足够容纳 marker
        if max_chars < len(marker) + 100:
            return result[:max_chars]
        head_len = int(max_chars * 0.6)
        tail_len = max_chars - head_len - len(marker)
        if tail_len > 0:
            result = result[:head_len] + marker + result[-tail_len:]
        else:
            result = result[:max_chars - len(marker)] + marker.strip()

    return result


def interactive_stress_test(
    solution_code: str,
    generator_code: str,
    judge_code: str,
    num_tests: int = 100,
) -> str:
    """Run interactive stress test.

    Args:
        solution_code: Python code for the solver
        generator_code: Python code for generating test data
        judge_code: Python code for the judge/interactor
        num_tests: Number of tests to run

    Returns:
        Result string: "INTERACTIVE STRESS TEST PASSED" or failure details with full log
    """
    for i in range(num_tests):
        # Generate test data
        gen_result = execute_code(
            code=generator_code,
            stdin="",
            timeout_seconds=5.0,
            memory_mb=256,
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

        # Run interaction
        result = run_interaction(
            judge_code=judge_code,
            solver_code=solution_code,
            test_input=test_input,
            timeout_total=30.0,
            timeout_per_turn=2.0,
        )

        if result.verdict != "AC":
            # 截断测试输入（如果过长）
            truncated_input = test_input
            if len(test_input) > 500:
                truncated_input = test_input[:500] + f"\n... (输入截断，原始 {len(test_input)} 字符)"

            return f"""=== INTERACTIVE TEST FAILED ===
Test #{i + 1}
Verdict: {result.verdict}
Time: {result.time_ms:.1f}ms
{f"Exit Code: {result.exit_code}" if result.exit_code is not None else ""}
{f"Error: {result.error_message}" if result.error_message else ""}

Test Input:
{truncated_input}

Interaction Log:
{_truncate_interaction_log(result.log)}

请分析交互日志并修正你的代码。"""

    return f"""=== INTERACTIVE STRESS TEST PASSED ===
All {num_tests} tests passed!
Your interactive solution works correctly on all random inputs."""
