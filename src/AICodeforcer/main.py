"""CLI entry point for the algorithm solver."""

import sys

from dotenv import load_dotenv

from AICodeforcer.agents import AlgorithmSolver

load_dotenv()


def print_solution(python_code: str | None, cpp_code: str | None, passed: bool) -> None:
    """打印解决方案（Python 和 C++ 两份代码）。"""
    print("\n" + "=" * 60)

    if passed and python_code:
        print("  对拍通过!")
        print("=" * 60)

        # 输出 Python 代码
        print("\n" + "=" * 60)
        print("  最终代码 (Python)")
        print("=" * 60)
        print(python_code)

        # 输出 C++ 代码
        if cpp_code:
            print("\n" + "=" * 60)
            print("  最终代码 (C++)")
            print("=" * 60)
            print(cpp_code)
        else:
            print("\n[注意] C++ 翻译失败，仅提供 Python 代码")
    else:
        print("  本轮求解未通过对拍")
        print("=" * 60)
        if python_code:
            print("\n当前代码 (Python):")
            print("-" * 40)
            print(python_code)
            print("-" * 40)

            if cpp_code:
                print("\n当前代码 (C++):")
                print("-" * 40)
                print(cpp_code)
                print("-" * 40)


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("  AICodeforcer - Gemini 算法题解 Agent")
    print("=" * 60)
    print()

    import os
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("错误: 请设置 GEMINI_API_KEY 环境变量")
        return 1

    print("请粘贴完整的题目 (输入 END 结束):")
    print("-" * 60)

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    text = "\n".join(lines)

    if not text.strip():
        print("错误: 题目不能为空")
        return 1

    print("-" * 60)
    print("开始求解...")
    print("=" * 60)

    try:
        solver = AlgorithmSolver(api_key=api_key)

        def on_attempt(attempt: int, code: str) -> None:
            print(f"\n--- 尝试 #{attempt} ---")
            print("-" * 40)
            code_lines = code.split("\n")
            for l in code_lines[:30]:
                print(l)
            if len(code_lines) > 30:
                print(f"... ({len(code_lines) - 30} more lines)")
            print("-" * 40)

        solution, cpp_code, passed = solver.solve(text, max_attempts=100, on_attempt=on_attempt)

        while True:
            print_solution(solution, cpp_code, passed)

            print("\n" + "-" * 60)
            print("请输入提交结果反馈 (输入 AC/done/quit 结束):")
            print("  例如: TLE on test 5, WA on test 3, MLE, RE")
            print("-" * 60)

            try:
                feedback = input("> ").strip()
            except EOFError:
                print("\n已结束")
                break

            if not feedback:
                continue

            feedback_lower = feedback.lower()
            if feedback_lower in ("ac", "done", "quit", "exit", "q"):
                print("\n" + "=" * 60)
                print("  恭喜 AC!" if feedback_lower == "ac" else "  已结束")
                print("=" * 60)
                break

            print("\n" + "=" * 60)
            print(f"  收到反馈: {feedback}")
            print("  继续优化中...")
            print("=" * 60)

            solution, cpp_code, passed = solver.continue_solving(
                feedback=feedback,
                max_attempts=50,
                on_attempt=on_attempt,
            )

        return 0

    except KeyboardInterrupt:
        print("\n\n已取消")
        return 130

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
