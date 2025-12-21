"""C++ translator agent for converting Python to competitive programming style C++."""

import os
import re
import time

from google import genai
from google.genai import types

CPP_TRANSLATOR_PROMPT = """# Role
你是一个资深的 C++ 算法竞赛（Competitive Programming）选手。你的任务是将输入的 Python 算法代码翻译成一种特定的"竞赛个人模版风格" C++ 代码。

# Target Style Guidelines
请严格遵守以下代码风格和模版约定：

1. **头文件与宏定义（必须完全一致）：**
   代码必须以以下模版开头（不要修改任何定义）：
   ```cpp
   #include<bits/stdc++.h>
   using namespace std;
   #define pb emplace_back
   #define mp make_pair
   #define ALL(x) (x).begin(),(x).end()
   #define rALL(x) (x).rbegin(),(x).rend()
   #define srt(x) sort(ALL(x))
   #define rev(x) reverse(ALL(x))
   #define rsrt(x) sort(rALL(x))
   #define sz(x) (int)(x.size())
   #define inf 0x3f3f3f3f
   #define lb(v,x) (int)(lower_bound(ALL(v),x)-v.begin())
   #define ub(v,x) (int)(upper_bound(ALL(v),x)-v.begin())
   #define uni(v) v.resize(unique(ALL(v))-v.begin())
   using ll=long long;
   using ull=unsigned long long;
   using pii=pair<int,int>;
   using i128=__int128_t;
   void die(string S){puts(S.c_str());exit(0);}
   ```

2. **类型替换：**
   - 将所有的 `long long` 替换为 `ll`。
   - 将 `unsigned long long` 替换为 `ull`。
   - 将 `pair<int,int>` 替换为 `pii`。

3. **容器与算法操作替换（使用宏）：**
   - `vec.push_back(...)` 或 `vec.emplace_back(...)` -> `vec.pb(...)`
   - `vec.size()` -> `sz(vec)`
   - `sort(vec.begin(), vec.end())` -> `srt(vec)`
   - `reverse(vec.begin(), vec.end())` -> `rev(vec)`
   - `make_pair(...)` -> `mp(...)`
   - 离散化去重操作 -> 使用 `srt(v); uni(v);`

4. **输入逻辑（极其重要，必须严格遵守）：**

   **绝对禁止以下写法：**
   - `if (!(cin >> ...))` ❌
   - `if (cin.fail())` ❌
   - `if (!(cin >> t)) return 0;` ❌
   - `while (cin >> n)` ❌（除非 EOF 结束）
   - 任何形式的输入检查或防御性代码 ❌

   **正确写法：**
   - 读取变量：直接 `cin >> n;`，不要任何检查
   - 多组测试：
     ```cpp
     int t;
     cin >> t;
     while(t--) solve();
     ```
   - 单组测试：直接 `cin >> n;` 然后处理

5. **主函数模版：**
   `main` 函数开头必须包含 IO 加速：
   ```cpp
   ios_base::sync_with_stdio(false);
   cin.tie(0);
   cout.tie(0);
   ```
   将主要逻辑封装在 `void solve()` 函数中，`main` 函数只负责 IO 加速和调用 `solve`。

6. **代码格式风格：**
   - **大括号风格**：函数和循环的大括号 `{` 另起一行（Allman风格）。
   - **紧凑性**：如果是简单的 `if` 或 `while` 语句且只有一行执行体，**不要**加大括号。
   - **命名习惯**：快速幂函数名为 `ksm`。变量名用短命名（`n, m, t, ans, res`）。

# Output Format
- 只输出 C++ 代码，不要任何解释或说明
- 代码用 ```cpp 包裹
- 必须输出完整代码，不准截断
"""


class CppTranslator:
    """将 Python 算法代码翻译成 C++ 竞赛风格代码。"""

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

    def translate(self, python_code: str) -> str | None:
        """将 Python 代码翻译成 C++ 竞赛风格代码。

        Args:
            python_code: Python 源代码

        Returns:
            C++ 代码，失败返回 None
        """
        print("\n" + "=" * 60)
        print("  翻译 Python -> C++")
        print("=" * 60)

        config = types.GenerateContentConfig(
            system_instruction=CPP_TRANSLATOR_PROMPT,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )

        user_prompt = f"""```python
{python_code}
```"""

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
                print(f"[翻译] 请求失败 (重试 {retry + 1}/30): {e}")
                if retry == 29:
                    print("[翻译] 翻译失败")
                    return None
                time.sleep(5)

        if not response:
            return None

        candidate = response.candidates[0] if response.candidates else None
        if not candidate or not candidate.content:
            print("[翻译] 无响应内容")
            return None

        response_text = ""
        for part in candidate.content.parts:
            if part.text:
                response_text += part.text

        if not response_text.strip():
            print("[翻译] 无有效输出")
            return None

        # 提取 C++ 代码
        cpp_code = self._extract_cpp_code(response_text)

        if not cpp_code:
            print("[翻译] 未能提取 C++ 代码")
            return None

        print(f"[翻译] 成功 ({len(cpp_code)} 字符)")
        return cpp_code

    def _extract_cpp_code(self, text: str) -> str | None:
        """从响应文本中提取 C++ 代码。

        Args:
            text: 响应文本

        Returns:
            提取的 C++ 代码，失败返回 None
        """
        # 尝试匹配 ```cpp 代码块
        patterns = [
            r"```cpp\s*\n(.*?)```",
            r"```c\+\+\s*\n(.*?)```",
            r"```\s*\n(.*?)```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()

        # 如果没有代码块，尝试直接返回（可能整个响应就是代码）
        if "#include" in text:
            return text.strip()

        return None
