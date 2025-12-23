"""Brute force algorithm generator agent."""

import concurrent.futures
import os
import re
import time

from google import genai
from google.genai import types

from AICodeforcer.standard.tools.executor import execute_code

BRUTE_FORCE_PROMPT = """<role>
You are an assistant specialized in writing brute force algorithms. Your sole objective is to write absolutely correct brute force solutions for algorithm problems.
</role>

<core-principles title="Must Not Be Violated">
  <principle name="Must use the most naive exhaustive methods">
    <item>O(n!) enumerate all permutations</item>
    <item>O(2^n) enumerate all subsets</item>
    <item>O(n^3) or higher nested loops</item>
    <item>Recursive backtracking to enumerate all possibilities</item>
  </principle>

  <principle name="No optimizations allowed">
    <item>No greedy algorithms</item>
    <item>No pruning</item>
    <item>No dynamic programming</item>
    <item>No "clever" tricks</item>
  </principle>

  <principle name="Correctness is the only goal">
    <item>Do not consider time complexity</item>
    <item>Do not consider space complexity</item>
    <item>As long as the result is correct</item>
  </principle>
</core-principles>

<output-format-requirements priority="critical">
  <rule>Output format must strictly follow the problem requirements!</rule>
  <rule>If the problem requires outputting k integers, you must output k integers (space-separated)</rule>
  <rule>Use `print(*ans)` or `print(' '.join(map(str, ans)))`</rule>
  <forbidden>Never output aggregated values (e.g., `print(sum(ans))`, `print(len(ans))`)</forbidden>
  <forbidden>Never output debug information</forbidden>
  <warning>Wrong output format = completely wrong</warning>
</output-format-requirements>

<code-standards>
  <rule>Code must be complete and self-contained, able to run independently</rule>
  <rule>Read input from stdin, output to stdout</rule>
  <rule>Explicitly import required modules (standard library only)</rule>
  <rule>Handle multiple test cases (if the problem has multiple tests)</rule>
</code-standards>

<required-output>
  <code-block name="Brute Force Algorithm Code">
    <format>Wrapped in ```python, marked as BRUTE_FORCE</format>
  </code-block>

  <code-block name="Data Generator Code">
    <format>Wrapped in ```python, marked as GENERATOR</format>
    <requirement>Generate small-scale random data (n ≤ 6)</requirement>
    <requirement>Ensure the brute force algorithm can run in reasonable time</requirement>
    <requirement>Cover edge cases</requirement>
  </code-block>
</required-output>

<output-example>
```python
# BRUTE_FORCE
import sys
# ... brute force algorithm code ...
```

```python
# GENERATOR
import random
# ... data generator code ...
```
</output-example>
"""


class BruteForceGenerator:
    """Generate brute force algorithm and test data generator."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required.")

        self.base_url = base_url or os.environ.get("GEMINI_BASE_URL")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        if self.base_url:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(base_url=self.base_url),
            )
        else:
            self.client = genai.Client(api_key=self.api_key)

    def generate(self, problem_text: str) -> tuple[str, str] | None:
        """生成暴力算法和数据生成器。

        Args:
            problem_text: 题目描述

        Returns:
            (brute_force_code, generator_code) 元组，失败返回 None
        """
        print("\n" + "=" * 60)
        print("  生成暴力算法和数据生成器 (独立会话)")
        print("=" * 60)

        config = types.GenerateContentConfig(
            system_instruction=BRUTE_FORCE_PROMPT,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )

        user_prompt = f"""请为以下算法题编写暴力算法和数据生成器：

{problem_text}

记住：
1. 暴力算法必须使用最朴素的穷举，禁止任何优化
2. 输出格式必须严格符合题目要求
3. 用 # BRUTE_FORCE 和 # GENERATOR 标记两段代码"""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ]

        response = None
        for retry in range(30):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                print(f"[暴力生成] 请求失败 (重试 {retry + 1}/30): {e}")
                if retry == 29:
                    print("[暴力生成] 生成失败")
                    return None
                import time
                time.sleep(5)

        if not response:
            return None

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content:
            print("[暴力生成] 无响应内容")
            return None

        response_text = ""
        for part in candidate.content.parts:
            if part.text:
                response_text += part.text

        if not response_text.strip():
            print("[暴力生成] 无有效输出")
            return None

        # 提取两段代码
        brute_force_code = self._extract_code(response_text, "BRUTE_FORCE")
        generator_code = self._extract_code(response_text, "GENERATOR")

        if not brute_force_code:
            print("[暴力生成] 未能提取暴力算法代码")
            return None

        if not generator_code:
            print("[暴力生成] 未能提取数据生成器代码")
            return None

        print("[暴力生成] 成功生成暴力算法和数据生成器")
        print(f"  - 暴力算法: {len(brute_force_code)} 字符")
        print(f"  - 数据生成器: {len(generator_code)} 字符")

        return brute_force_code, generator_code

    def _extract_code(self, text: str, marker: str) -> str | None:
        """提取带有特定标记的代码块。

        Args:
            text: 响应文本
            marker: 代码标记 (BRUTE_FORCE 或 GENERATOR)

        Returns:
            提取的代码，失败返回 None
        """
        # 尝试匹配带标记的代码块
        patterns = [
            rf"```python\s*\n\s*#\s*{marker}\s*\n(.*?)```",
            rf"```py\s*\n\s*#\s*{marker}\s*\n(.*?)```",
            rf"```\s*\n\s*#\s*{marker}\s*\n(.*?)```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()

        # 备用方案：按顺序提取代码块
        code_blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)

        if marker == "BRUTE_FORCE" and len(code_blocks) >= 1:
            # 第一个代码块通常是暴力算法
            code = code_blocks[0].strip()
            # 移除可能的标记注释
            code = re.sub(r"^\s*#\s*(BRUTE_FORCE|GENERATOR)\s*\n", "", code)
            return code

        if marker == "GENERATOR" and len(code_blocks) >= 2:
            # 第二个代码块通常是生成器
            code = code_blocks[1].strip()
            code = re.sub(r"^\s*#\s*(BRUTE_FORCE|GENERATOR)\s*\n", "", code)
            return code

        return None

    def _generate_single(self, problem_text: str, agent_id: int) -> tuple[str, str, int] | None:
        """单个 agent 生成暴力解法。

        Args:
            problem_text: 题目描述
            agent_id: agent 标识

        Returns:
            (brute_force_code, generator_code, agent_id) 或 None
        """
        # 为每个线程创建独立的 client，避免并发安全问题
        if self.base_url:
            client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(base_url=self.base_url),
            )
        else:
            client = genai.Client(api_key=self.api_key)

        config = types.GenerateContentConfig(
            system_instruction=BRUTE_FORCE_PROMPT,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )

        user_prompt = f"""请为以下算法题编写暴力算法和数据生成器：

{problem_text}

记住：
1. 暴力算法必须使用最朴素的穷举，禁止任何优化
2. 输出格式必须严格符合题目要求
3. 用 # BRUTE_FORCE 和 # GENERATOR 标记两段代码"""

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt)],
            )
        ]

        response = None
        for retry in range(30):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as e:
                print(f"  [Agent {agent_id}] 请求失败 (重试 {retry + 1}/30): {e}")
                if retry == 29:
                    return None
                time.sleep(5)

        if not response:
            return None

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content:
            return None

        response_text = ""
        for part in candidate.content.parts:
            if part.text:
                response_text += part.text

        if not response_text.strip():
            return None

        brute_force_code = self._extract_code(response_text, "BRUTE_FORCE")
        generator_code = self._extract_code(response_text, "GENERATOR")

        if not brute_force_code or not generator_code:
            return None

        return brute_force_code, generator_code, agent_id

    def _validate_consensus(
        self,
        brute_codes: list[str],
        generator_code: str,
        validation_rounds: int,
    ) -> bool:
        """验证多个暴力解法的一致性。

        Args:
            brute_codes: 暴力解法代码列表
            generator_code: 数据生成器代码
            validation_rounds: 验证轮数

        Returns:
            True 如果所有解法一致，否则 False
        """
        print(f"\n[验证] 开始一致性验证 ({validation_rounds} 轮)...")

        for round_num in range(validation_rounds):
            # 1. 用 generator 生成测试数据
            gen_result = execute_code(generator_code, "", timeout_seconds=5.0, memory_mb=256)
            if gen_result.status != "passed":
                print(f"  轮次 {round_num + 1}/{validation_rounds}: 生成器执行失败")
                print(f"    状态: {gen_result.status}")
                print(f"    错误: {gen_result.error_message or 'Unknown'}")
                return False

            test_input = (gen_result.actual_output or "").strip()
            if not test_input:
                print(f"  轮次 {round_num + 1}/{validation_rounds}: 生成器输出为空")
                return False

            # 2. 运行所有暴力解法，检查状态并收集输出
            outputs = []
            for i, code in enumerate(brute_codes):
                result = execute_code(code, test_input, timeout_seconds=10.0, memory_mb=256)
                if result.status != "passed":
                    print(f"  轮次 {round_num + 1}/{validation_rounds}: Agent {i} 执行失败")
                    print(f"    状态: {result.status}")
                    print(f"    错误: {result.error_message or 'Unknown'}")
                    print(f"    输入: {test_input[:100]}...")
                    return False
                outputs.append((result.actual_output or "").strip())

            # 3. 检查输出是否完全一致
            if not all(o == outputs[0] for o in outputs):
                print(f"  轮次 {round_num + 1}/{validation_rounds}: ✗ 不一致")
                print(f"    输入: {test_input[:100]}...")
                for i, o in enumerate(outputs):
                    print(f"    Agent {i}: {o[:100]}...")
                return False

            print(f"  轮次 {round_num + 1}/{validation_rounds}: ✓ 一致")

        return True

    def generate_with_consensus(
        self,
        problem_text: str,
        num_agents: int = 3,
        validation_rounds: int = 10,
    ) -> tuple[str, str] | None:
        """并行生成多个暴力解法，通过一致性验证选择可信的解法。

        Args:
            problem_text: 题目描述
            num_agents: 并行生成的 agent 数量（默认 3）
            validation_rounds: 验证轮数（默认 10）

        Returns:
            (brute_force_code, generator_code) 或 None
        """
        print("\n" + "=" * 60)
        print(f"  启动 {num_agents} 个 Agent 并行生成暴力解法")
        print("=" * 60)

        # 并行生成
        results: list[tuple[str, str, int] | None] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_agents) as executor:
            futures = {
                executor.submit(self._generate_single, problem_text, i): i
                for i in range(num_agents)
            }

            for future in concurrent.futures.as_completed(futures):
                agent_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        brute_code, gen_code, aid = result
                        print(f"  Agent {aid}: 生成成功 ({len(brute_code)} 字符)")
                        results.append(result)
                    else:
                        print(f"  Agent {agent_id}: 生成失败")
                except Exception as e:
                    print(f"  Agent {agent_id}: 异常 - {e}")

        # 检查是否所有 agent 都成功
        if len(results) < num_agents:
            print(f"\n[错误] 只有 {len(results)}/{num_agents} 个 Agent 成功，需要全部成功")
            return None

        # 提取暴力代码和生成器
        brute_codes = [r[0] for r in results]
        generator_code = results[0][1]  # 使用第一个 agent 的生成器

        # 一致性验证
        if not self._validate_consensus(brute_codes, generator_code, validation_rounds):
            print("\n[错误] 暴力解法一致性验证失败，3 个解法输出不一致")
            return None

        print("\n[成功] 暴力解法验证通过，使用 Agent 0 的代码")
        return brute_codes[0], generator_code
