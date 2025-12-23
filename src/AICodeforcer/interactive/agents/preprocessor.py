"""Interactive problem preprocessor - generates data generator and judge."""

import os
import re

from google import genai
from google.genai import types

PREPROCESSOR_SYSTEM_PROMPT = """<role>
You are a top-tier ICPC / CCPC competitive programming problem setter.
Your task is to generate data generator and judge (Interactor) for interactive problems.
</role>

<judge-specification>
  <structure>
    <step>Get test data file path from command line argument: `sys.argv[1]`</step>
    <step>Read test data</step>
    <step>Interact with contestant's program (via stdin/stdout)</step>
    <step>Exit based on interaction result:
      <exit-code code="0">AC (Accepted)</exit-code>
      <exit-code code="1">WA (Wrong Answer)</exit-code>
      <exit-code code="2">PE (Protocol Error)</exit-code>
    </step>
  </structure>

  <template language="python">
import sys

def main():
    # Read test data
    with open(sys.argv[1], 'r') as f:
        # Parse test data
        ...

    # Interaction loop
    while not finished:
        # Send message to contestant
        print(message, flush=True)

        # Read contestant's response
        try:
            response = input()
        except EOFError:
            exit(2)  # Protocol Error

        # Process response
        ...

    # Determine result
    if correct:
        exit(0)  # AC
    else:
        exit(1)  # WA

if __name__ == "__main__":
    main()
  </template>
</judge-specification>

<generator-specification>
  <requirement>Generate random test data</requirement>
  <requirement>Output to stdout</requirement>
  <requirement>Use `random` module, generate different data each run</requirement>
</generator-specification>

<output-format>
  <code-block name="Data Generator">
    <format>Wrapped with ```generator and ```</format>
  </code-block>
  <code-block name="Judge">
    <format>Wrapped with ```judge and ```</format>
  </code-block>
  <rule>Code must be complete, self-contained, and directly runnable.</rule>
</output-format>
"""


class InteractivePreprocessor:
    """Generates data generator and judge for interactive problems."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required")

        self.base_url = base_url or os.environ.get("GEMINI_BASE_URL")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        if self.base_url:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(base_url=self.base_url),
            )
        else:
            self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        problem_text: str,
        max_attempts: int = 10,
    ) -> tuple[str, str] | None:
        """Generate data generator and judge code.

        Args:
            problem_text: The problem statement
            max_attempts: Maximum attempts to generate valid code

        Returns:
            (generator_code, judge_code) tuple or None if failed
        """
        from AICodeforcer.interactive.agents.judge_validator import JudgeValidator

        validator = JudgeValidator(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
        )

        config = types.GenerateContentConfig(
            system_instruction=PREPROCESSOR_SYSTEM_PROMPT,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )

        contents: list[types.Content] = []

        initial_prompt = f"""请为以下交互题生成数据生成器和评测机：

{problem_text}

请仔细分析题目的交互协议，然后生成：
1. 数据生成器（用 ```generator 包裹）
2. 评测机（用 ```judge 包裹）

确保：
- 数据生成器生成符合题目约束的随机数据
- 评测机正确实现交互协议
- 评测机使用正确的退出码（0=AC, 1=WA, 2=PE）
- 所有 print 语句都使用 flush=True
"""

        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=initial_prompt)],
        ))

        for attempt in range(max_attempts):
            print(f"\n[预处理] 生成评测机和数据生成器 (尝试 {attempt + 1}/{max_attempts})...")

            response = None
            for retry in range(10):
                try:
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )
                    break
                except Exception as e:
                    print(f"  请求失败 (重试 {retry + 1}/10): {e}")
                    if retry == 9:
                        return None
                    import time
                    time.sleep(3)

            if not response or not response.candidates:
                continue

            candidate = response.candidates[0]
            if not candidate.content:
                continue

            response_text = ""
            for part in candidate.content.parts:
                if part.text:
                    response_text += part.text

            # Extract generator and judge code
            generator_code = self._extract_code(response_text, "generator")
            judge_code = self._extract_code(response_text, "judge")

            if not generator_code or not judge_code:
                print("  未能提取到完整代码，重试...")
                contents.append(candidate.content)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="请确保输出完整的代码，用 ```generator 和 ```judge 分别包裹数据生成器和评测机代码。"
                    )],
                ))
                continue

            print(f"  生成器: {len(generator_code)} 字符")
            print(f"  评测机: {len(judge_code)} 字符")

            # Validate with fresh AI session
            print("  验证评测机...")
            is_valid, issues = validator.validate(problem_text, generator_code, judge_code)

            if is_valid:
                print("  验证通过!")
                return generator_code, judge_code

            print(f"  验证发现问题: {issues[:200]}...")

            # Add feedback and retry
            contents.append(candidate.content)
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_text(
                    text=f"""验证器发现以下问题：

{issues}

请修正这些问题，重新生成数据生成器和评测机代码。"""
                )],
            ))

        print("[预处理] 生成失败，已达最大尝试次数")
        return None

    def _extract_code(self, text: str, code_type: str) -> str | None:
        """Extract code block of specific type."""

        def _strip_leading_markers(code: str, marker: str) -> str:
            """Remove redundant leading marker lines like 'generator'/'judge'."""
            lines = code.splitlines()
            result = []
            for line in lines:
                # Skip lines that are just the marker word (with optional whitespace)
                if line.strip().lower() == marker.lower():
                    continue
                result.append(line)
            return "\n".join(result).strip()

        # Try specific marker first
        # Pattern captures: ```generator (optional same-line content) \n (code body) ```
        # Group 1: same-line content after marker, Group 2: code body
        pattern = rf"```[ \t]*{re.escape(code_type)}[ \t]*([^\n]*)\r?\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            same_line, body = matches[-1]
            # Prepend same-line content if it looks like code (not empty/whitespace)
            if same_line.strip():
                body = same_line.strip() + "\n" + body
            candidate = _strip_leading_markers(body, code_type)
            if candidate:
                return candidate

        # Fallback to python blocks if only one type requested
        if code_type == "generator":
            # Look for generator-related code
            pattern = r"```python[ \t]*([^\n]*)\r?\n(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for same_line, body in matches:
                if same_line.strip():
                    body = same_line.strip() + "\n" + body
                candidate = _strip_leading_markers(body, code_type)
                if "random" in candidate and "print" in candidate:
                    return candidate

        elif code_type == "judge":
            # Look for judge-related code
            pattern = r"```python[ \t]*([^\n]*)\r?\n(.*?)```"
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for same_line, body in matches:
                if same_line.strip():
                    body = same_line.strip() + "\n" + body
                candidate = _strip_leading_markers(body, code_type)
                if "sys.argv" in candidate or "exit(" in candidate:
                    return candidate

        return None
