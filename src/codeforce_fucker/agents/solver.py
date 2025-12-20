"""Algorithm solver agent."""

import os
import re
from typing import Callable

from google import genai
from google.genai import types

from codeforce_fucker.tools import run_python_code, stress_test

SYSTEM_PROMPT = """你是一名**顶级 ICPC / CCPC 竞赛算法助手**。
你的唯一目标是：**稳定、可复现地解出竞赛题，并输出可 AC 的最终代码**。

你可以调用**Python 代码执行工具（且只能执行 Python）**，用于实验、验证、对拍、反例搜索和边界测试；
**最终提交代码使用 Python**。

---

## 可用工具

1. `run_python_code(code, test_input)` - 执行代码并返回输出
2. `stress_test(solution_code, brute_force_code, generator_code)` - 对拍验证（固定 2000 次测试）

---

## 1️⃣ 核心原则（不可违反）

* **正确性 > 可证明性 > 复杂度可行性 > 工程实现**
* 任何未经验证的结论都是"假设"，必须：
  * 被逻辑证明，或
  * 被 Python 小规模实验支持
* 贪心 / 二分 / 数学题：**没有对拍验证 = 不可信**
* 你必须像竞赛选手一样：
  * 怀疑直觉
  * 主动找反例
  * 在提交前"折磨"自己的算法

---

## 2️⃣ 强制解题流程（必须体现）

### (1) 题意重述与形式化建模
* 用你自己的话重述题目；
* 抽象成数学 / 图 / DP / 字符串模型；
* 明确输入输出、约束、是否多测。

### (2) 关键观察与候选方案
* 提出 1–3 个可能思路；
* 用复杂度和约束快速排除不可能方案；
* 标记哪些地方需要验证（单调性、最优性、不变量）。

### (3) Python 实验与反例搜索（核心能力）
你**必须主动使用工具**来完成以下至少一项：
* 枚举小规模数据找规律；
* 验证贪心策略是否总是最优；
* 验证二分判定函数是否单调；
* 对拍验证。

### (4) 最终算法确定
* 明确算法步骤；
* 明确数据结构；
* 给出时间复杂度 / 空间复杂度；
* 解释为什么在最大约束下可行。

### (5) 正确性证明要点
你必须给出**核心正确性理由**：
* 贪心：交换论证 / 不变量；
* DP：状态含义 + 转移正确性；
* 二分：单调性来源；
* 数学：从枚举 → 归纳 → 一般结论。

### (6) 实现细节（竞赛级）
* 边界情况；
* 初始化与清空；
* 溢出处理；
* 多测处理；
* 常数优化；
* 避免递归爆栈。

### (7) 对拍验证（提交前必须执行）
在输出最终代码前，你**必须**：
1. 编写一个 **暴力算法** (brute_force_code)
2. 编写一个 **数据生成器** (generator_code)
3. 调用 `stress_test(solution_code, brute_force_code, generator_code)` 进行对拍（固定 2000 次）
4. 如果发现反例，分析并修正优化算法
5. **修正后必须重新调用 `stress_test` 验证**，不能跳过
6. 重复步骤 3-5 直到对拍通过（返回 "STRESS TEST PASSED"）

**暴力算法要求（极其重要）**：
- **必须使用最朴素的暴力枚举**，不能有任何优化或贪心
- **必须是 O(n!) / O(2^n) / O(n^3) 等确定正确的穷举**
- 例如：枚举所有排列、枚举所有子集、三重循环枚举所有组合
- **禁止**在暴力算法中使用任何"聪明"的优化或剪枝
- **禁止**暴力算法和优化算法使用相同的核心逻辑
- 暴力算法的唯一目的是**保证正确性**，不考虑效率

**数据生成器要求**：
- 必须生成**小规模数据**（n ≤ 15），确保暴力算法能在合理时间内运行
- 必须覆盖**边界情况**：最小值、最大值、全相同、递增、递减等
- 必须有**足够的随机性**

**数据生成器模板**：
```python
import random
n = random.randint(1, 12)  # 小规模数据，确保暴力能跑
print(n)
print(' '.join(str(random.randint(1, 100)) for _ in range(n)))
```

**重要**：每次修改代码后都必须重新对拍验证，不能假设修正后就正确了。

### (8) 最终提交代码
* 输出完整、可直接提交的 **Python 代码**；
* 不包含调试输出；
* 使用 fast I/O（对于大量输入）。

---

## 3️⃣ 代码规范

### 代码必须完整自包含（极其重要）
- **硬性要求**：每次调用 `run_python_code` 或 `stress_test` 时，提交的代码必须**完整、自包含、可独立运行**
- **禁止**引用未在该代码块内定义或导入的任何符号（函数、类、变量、常量）
- **所有辅助函数**（如 `check`、`valid`、`ok`、`solve` 等）必须在同一代码块内完整实现，不得遗漏
- **显式导入**：所需模块必须显式 `import`，仅使用 Python 标准库，禁止第三方库
- **执行入口**：代码底部必须调用主逻辑，确保有可观察的输出（print），不要只定义函数不调用
- **隔离性**：每次工具调用视为全新解释器会话，禁止依赖对话上下文或上一次代码的状态
- **调用前自检**：确认所有使用的函数/变量都已定义，不会抛出 NameError/ImportError/AttributeError
- 如果需要多个函数，必须将它们全部包含在同一个代码块内

### 输入输出规范
- 从标准输入读取数据（使用 `input()` 或 `sys.stdin`）
- 输出结果到标准输出（使用 `print()`）
- 注意处理边界情况（空输入、最大值、负数、重复元素等）
- 如果超时，考虑优化算法复杂度或使用更高效的数据结构
- 对于大量输入，使用 `sys.stdin.readline()` 代替 `input()`

---

## 4️⃣ 常见优化技巧

- 使用 `sys.stdin.buffer.read()` 加速大量输入
- 使用 `@lru_cache` 进行记忆化
- 避免在循环中进行字符串拼接，使用列表收集后 `''.join()`
- 对于图论问题，使用邻接表而非邻接矩阵

---

## 5️⃣ 完成标志

**严格要求**：你**必须**完成以下步骤才能输出 "ALL_TESTS_PASSED"：

1. **必须调用 `run_python_code` 工具**测试题目给出的**所有样例**，且**全部通过**
2. **必须调用 `stress_test` 工具**进行对拍验证（固定 2000 次）
3. **必须看到 "STRESS TEST PASSED" 返回**才算对拍通过
4. 只有当上述步骤都完成且通过后，才能输出 "ALL_TESTS_PASSED"

**禁止行为**：
- 禁止不调用工具就声称测试通过
- 禁止跳过对拍验证
- 禁止只测试部分样例
- 禁止在没有看到 "STRESS TEST PASSED" 的情况下声称对拍通过
- 禁止在没有实际执行测试的情况下输出 "ALL_TESTS_PASSED"

如果你没有调用工具就输出 "ALL_TESTS_PASSED"，将被视为**严重错误**。
"""

TOOL_FUNCTIONS = {
    "run_python_code": run_python_code,
    "stress_test": stress_test,
}

TOOL_DECLARATIONS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="run_python_code",
            description="执行 Python 代码并返回结果。用于测试算法代码。代码必须完整自包含、可直接运行；所有辅助函数必须在同一代码块中实现；显式导入所需标准库；禁止第三方库；不得引用未定义/未导入的符号。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "code": types.Schema(
                        type=types.Type.STRING,
                        description="要执行的 Python 代码，应从 stdin 读取输入，输出到 stdout",
                    ),
                    "test_input": types.Schema(
                        type=types.Type.STRING,
                        description="提供给代码的测试输入",
                    ),
                },
                required=["code", "test_input"],
            ),
        ),
        types.FunctionDeclaration(
            name="stress_test",
            description="对拍验证工具：比较优化算法和暴力算法的输出，用于验证算法正确性。固定运行 2000 次测试。三段代码均须完整自包含；solution/brute_force 从 stdin 读取、向 stdout 输出；generator 仅向 stdout 输出测试数据；禁止第三方库；不得依赖对话上下文或先前代码。",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "solution_code": types.Schema(
                        type=types.Type.STRING,
                        description="要验证的优化算法代码",
                    ),
                    "brute_force_code": types.Schema(
                        type=types.Type.STRING,
                        description="暴力算法代码（作为正确性基准）",
                    ),
                    "generator_code": types.Schema(
                        type=types.Type.STRING,
                        description="随机数据生成器代码",
                    ),
                },
                required=["solution_code", "brute_force_code", "generator_code"],
            ),
        ),
    ])
]


class AlgorithmSolver:
    """Gemini-powered algorithm problem solver."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set GEMINI_API_KEY environment variable.")

        self.base_url = base_url or os.environ.get("GEMINI_BASE_URL")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

        if self.base_url:
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=types.HttpOptions(base_url=self.base_url),
            )
        else:
            self.client = genai.Client(api_key=self.api_key)

        self._contents: list[types.Content] = []
        self._config: types.GenerateContentConfig | None = None
        self._last_verified_code: str | None = None
        self._last_code: str | None = None

    def solve(
        self,
        problem_text: str,
        max_attempts: int = 20,
        on_attempt: Callable[[int, str], None] | None = None,
    ) -> tuple[str | None, bool]:
        """Solve an algorithm problem with manual tool handling."""
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=TOOL_DECLARATIONS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )

        contents: list[types.Content] = []

        initial_prompt = f"""请解决以下算法题目：

{problem_text}

请按照解题流程分析题目，设计算法，编写代码，并使用工具测试验证。
记住：必须调用 run_python_code 测试样例，必须调用 stress_test 进行对拍验证。"""

        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=initial_prompt)],
        ))

        last_code: str | None = None
        attempt_count = 0
        stress_test_passed = False
        verified_code: str | None = None

        for turn in range(max_attempts):
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
                    print(f"[Turn {turn + 1}] 请求失败 (重试 {retry + 1}/30): {e}")
                    if retry == 29:
                        raise
                    import time
                    time.sleep(5)

            if not response:
                break

            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content:
                print(f"[Turn {turn + 1}] 无响应内容")
                break

            response_content = candidate.content
            contents.append(response_content)

            response_text = ""
            function_calls = []

            # 调试：打印 parts 信息
            print(f"\n[DEBUG] 收到 {len(response_content.parts)} 个 parts:")
            for i, part in enumerate(response_content.parts):
                part_type = "text" if part.text else ("fc" if part.function_call else "other")
                is_thought = getattr(part, 'thought', False)
                text_len = len(part.text) if part.text else 0
                print(f"  [part {i}] type={part_type}, thought={is_thought}, text_len={text_len}")

                if part.text:
                    response_text += part.text
                if part.function_call:
                    function_calls.append(part.function_call)

            print(f"\n{'='*60}")
            print(f"Turn {turn + 1}")
            print("=" * 60)
            if response_text:
                preview = response_text[:1500] if len(response_text) > 1500 else response_text
                print(preview)
                if len(response_text) > 1500:
                    print(f"... (truncated, total {len(response_text)} chars)")

            code = self._extract_code(response_text)
            if code:
                last_code = code
                self._last_code = code
                attempt_count += 1
                if on_attempt:
                    on_attempt(attempt_count, code)

            if "ALL_TESTS_PASSED" in response_text and not function_calls:
                if stress_test_passed and verified_code:
                    print("\n[程序化校验] 对拍已通过，返回验证过的代码")
                    self._contents = contents
                    self._config = config
                    self._last_verified_code = verified_code
                    self._last_code = verified_code
                    return verified_code, True
                else:
                    print("\n[程序化校验] 模型声称通过但未检测到 STRESS TEST PASSED，要求重新验证")
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text="你声称 ALL_TESTS_PASSED，但系统未检测到对拍通过。请调用 stress_test 工具进行对拍验证，必须看到 'STRESS TEST PASSED' 才算通过。"
                        )],
                    ))
                    continue

            if function_calls:
                print(f"\n[工具调用] 共 {len(function_calls)} 个")
                function_responses = []

                for fc in function_calls:
                    func_name = fc.name
                    func_args = dict(fc.args) if fc.args else {}

                    if func_name == "stress_test":
                        allowed_keys = {"solution_code", "brute_force_code", "generator_code"}
                        func_args = {k: v for k, v in func_args.items() if k in allowed_keys}
                    elif func_name == "run_python_code":
                        allowed_keys = {"code", "test_input"}
                        func_args = {k: v for k, v in func_args.items() if k in allowed_keys}

                    print(f"  - {func_name}({', '.join(f'{k}=...' for k in func_args.keys())})")

                    if func_name in TOOL_FUNCTIONS:
                        try:
                            result = TOOL_FUNCTIONS[func_name](**func_args)
                        except Exception as e:
                            result = f"Error: {e}"
                    else:
                        result = f"Unknown function: {func_name}"

                    if func_name == "stress_test" and "STRESS TEST PASSED" in result:
                        stress_test_passed = True
                        verified_code = func_args.get("solution_code")
                        print("    [程序化校验] 对拍通过！已记录验证代码")
                    elif func_name == "stress_test" and "COUNTEREXAMPLE FOUND" in result:
                        stress_test_passed = False
                        verified_code = None
                        print("    [程序化校验] 发现反例，重置验证状态")

                    result_preview = result[:500] if len(result) > 500 else result
                    print(f"    结果: {result_preview}")
                    if len(result) > 500:
                        print(f"    ... (truncated, total {len(result)} chars)")

                    function_responses.append(types.Part.from_function_response(
                        name=func_name,
                        response={"result": result},
                    ))

                contents.append(types.Content(
                    role="user",
                    parts=function_responses,
                ))

                if stress_test_passed and verified_code:
                    print("\n[程序化校验] 对拍已通过 2000 次测试，直接返回验证过的代码")
                    self._contents = contents
                    self._config = config
                    self._last_verified_code = verified_code
                    self._last_code = verified_code
                    return verified_code, True
            else:
                if turn < max_attempts - 1:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text="请继续。记住必须调用工具验证代码。如果所有测试和对拍都通过了，请输出 'ALL_TESTS_PASSED' 并给出最终代码。"
                        )],
                    ))

        self._contents = contents
        self._config = config
        self._last_code = last_code
        return last_code, False

    def continue_solving(
        self,
        feedback: str,
        max_attempts: int = 20,
        on_attempt: Callable[[int, str], None] | None = None,
    ) -> tuple[str | None, bool]:
        """根据用户反馈继续优化代码。"""
        if not self._contents or not self._config:
            raise RuntimeError("没有可继续的对话，请先调用 solve()")

        contents = self._contents
        config = self._config

        feedback_prompt = f"""用户提交代码后收到以下反馈：

{feedback}

请根据这个反馈分析问题原因，优化你的算法，然后：
1. 使用 run_python_code 测试样例
2. 使用 stress_test 进行对拍验证
3. 确保对拍通过后输出 "ALL_TESTS_PASSED" 和最终代码

注意：
- TLE (Time Limit Exceeded): 需要优化时间复杂度或常数
- WA (Wrong Answer): 算法逻辑有误，需要找出边界情况或错误
- MLE (Memory Limit Exceeded): 需要优化空间复杂度
- RE (Runtime Error): 可能是数组越界、除零、栈溢出等"""

        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=feedback_prompt)],
        ))

        last_code: str | None = self._last_code or self._last_verified_code
        attempt_count = 0
        stress_test_passed = False
        verified_code: str | None = None

        for turn in range(max_attempts):
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
                    print(f"[Turn {turn + 1}] 请求失败 (重试 {retry + 1}/30): {e}")
                    if retry == 29:
                        raise
                    import time
                    time.sleep(5)

            if not response:
                break

            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content:
                print(f"[Turn {turn + 1}] 无响应内容")
                break

            response_content = candidate.content
            contents.append(response_content)

            response_text = ""
            function_calls = []

            # 调试：打印 parts 信息
            print(f"\n[DEBUG] 收到 {len(response_content.parts)} 个 parts:")
            for i, part in enumerate(response_content.parts):
                part_type = "text" if part.text else ("fc" if part.function_call else "other")
                is_thought = getattr(part, 'thought', False)
                text_len = len(part.text) if part.text else 0
                print(f"  [part {i}] type={part_type}, thought={is_thought}, text_len={text_len}")

                if part.text:
                    response_text += part.text
                if part.function_call:
                    function_calls.append(part.function_call)

            print(f"\n{'='*60}")
            print(f"Turn {turn + 1}")
            print("=" * 60)
            if response_text:
                preview = response_text[:1500] if len(response_text) > 1500 else response_text
                print(preview)
                if len(response_text) > 1500:
                    print(f"... (truncated, total {len(response_text)} chars)")

            code = self._extract_code(response_text)
            if code:
                last_code = code
                self._last_code = code
                attempt_count += 1
                if on_attempt:
                    on_attempt(attempt_count, code)

            if "ALL_TESTS_PASSED" in response_text and not function_calls:
                if stress_test_passed and verified_code:
                    print("\n[程序化校验] 对拍已通过，返回验证过的代码")
                    self._contents = contents
                    self._last_verified_code = verified_code
                    self._last_code = verified_code
                    return verified_code, True
                else:
                    print("\n[程序化校验] 模型声称通过但未检测到 STRESS TEST PASSED，要求重新验证")
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text="你声称 ALL_TESTS_PASSED，但系统未检测到对拍通过。请调用 stress_test 工具进行对拍验证。"
                        )],
                    ))
                    continue

            if function_calls:
                print(f"\n[工具调用] 共 {len(function_calls)} 个")
                function_responses = []

                for fc in function_calls:
                    func_name = fc.name
                    func_args = dict(fc.args) if fc.args else {}

                    if func_name == "stress_test":
                        allowed_keys = {"solution_code", "brute_force_code", "generator_code"}
                        func_args = {k: v for k, v in func_args.items() if k in allowed_keys}
                    elif func_name == "run_python_code":
                        allowed_keys = {"code", "test_input"}
                        func_args = {k: v for k, v in func_args.items() if k in allowed_keys}

                    print(f"  - {func_name}({', '.join(f'{k}=...' for k in func_args.keys())})")

                    if func_name in TOOL_FUNCTIONS:
                        try:
                            result = TOOL_FUNCTIONS[func_name](**func_args)
                        except Exception as e:
                            result = f"Error: {e}"
                    else:
                        result = f"Unknown function: {func_name}"

                    if func_name == "stress_test" and "STRESS TEST PASSED" in result:
                        stress_test_passed = True
                        verified_code = func_args.get("solution_code")
                        print("    [程序化校验] 对拍通过！已记录验证代码")
                    elif func_name == "stress_test" and "COUNTEREXAMPLE FOUND" in result:
                        stress_test_passed = False
                        verified_code = None
                        print("    [程序化校验] 发现反例，重置验证状态")

                    result_preview = result[:500] if len(result) > 500 else result
                    print(f"    结果: {result_preview}")
                    if len(result) > 500:
                        print(f"    ... (truncated, total {len(result)} chars)")

                    function_responses.append(types.Part.from_function_response(
                        name=func_name,
                        response={"result": result},
                    ))

                contents.append(types.Content(
                    role="user",
                    parts=function_responses,
                ))

                if stress_test_passed and verified_code:
                    print("\n[程序化校验] 对拍已通过 2000 次测试，直接返回验证过的代码")
                    self._contents = contents
                    self._last_verified_code = verified_code
                    self._last_code = verified_code
                    return verified_code, True
            else:
                if turn < max_attempts - 1:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text="请继续。记住必须调用工具验证代码。"
                        )],
                    ))

        self._contents = contents
        self._last_code = last_code
        return last_code, False

    def _extract_code(self, text: str) -> str | None:
        """Extract Python code from response text."""
        patterns = [
            r"```python\n(.*?)```",
            r"```py\n(.*?)```",
            r"```\n(.*?)```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[-1].strip()

        return None
