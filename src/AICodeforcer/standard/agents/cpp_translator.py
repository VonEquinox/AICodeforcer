"""C++ translator agent for converting Python to competitive programming style C++."""

import os
import re
import time

from google import genai
from google.genai import types

CPP_TRANSLATOR_PROMPT = """<role>
You are a senior C++ Competitive Programming contestant. Your task is to translate the input Python algorithm code into a specific "competitive programming personal template style" C++ code.
</role>

<style-guidelines>
  <guideline name="Header Files and Macro Definitions" priority="must-be-exact">
    <description>The code must start with the following template (do not modify any definitions):</description>
    <template language="cpp">
#include&lt;bits/stdc++.h&gt;
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
using pii=pair&lt;int,int&gt;;
using i128=__int128_t;
void die(string S){puts(S.c_str());exit(0);}
    </template>
  </guideline>

  <guideline name="Type Replacements">
    <replacement from="long long" to="ll"/>
    <replacement from="unsigned long long" to="ull"/>
    <replacement from="pair&lt;int,int&gt;" to="pii"/>
  </guideline>

  <guideline name="Container and Algorithm Operation Replacements">
    <replacement from="vec.push_back(...) or vec.emplace_back(...)" to="vec.pb(...)"/>
    <replacement from="vec.size()" to="sz(vec)"/>
    <replacement from="sort(vec.begin(), vec.end())" to="srt(vec)"/>
    <replacement from="reverse(vec.begin(), vec.end())" to="rev(vec)"/>
    <replacement from="make_pair(...)" to="mp(...)"/>
    <replacement from="Discretization and deduplication" to="srt(v); uni(v);"/>
  </guideline>

  <guideline name="Input Logic" priority="critical">
    <forbidden-patterns>
      <pattern>if (!(cin &gt;&gt; ...))</pattern>
      <pattern>if (cin.fail())</pattern>
      <pattern>if (!(cin &gt;&gt; t)) return 0;</pattern>
      <pattern>while (cin &gt;&gt; n) (unless EOF-terminated)</pattern>
      <pattern>Any form of input checking or defensive code</pattern>
    </forbidden-patterns>
    <correct-patterns>
      <pattern name="Reading variables">directly `cin &gt;&gt; n;`, no checks</pattern>
      <pattern name="Multiple test cases">
int t;
cin &gt;&gt; t;
while(t--) solve();
      </pattern>
      <pattern name="Single test case">directly `cin &gt;&gt; n;` then process</pattern>
    </correct-patterns>
  </guideline>

  <guideline name="Main Function Template">
    <description>The `main` function must start with IO acceleration:</description>
    <template language="cpp">
ios_base::sync_with_stdio(false);
cin.tie(0);
cout.tie(0);
    </template>
    <rule>Encapsulate the main logic in a `void solve()` function</rule>
    <rule>`main` function only handles IO acceleration and calling `solve`</rule>
  </guideline>

  <guideline name="Code Format Style">
    <rule name="Brace Style">Opening brace `{` for functions and loops on a new line (Allman style)</rule>
    <rule name="Compactness">For simple `if` or `while` statements with only one line of execution, do not add braces</rule>
    <rule name="Naming Conventions">Fast exponentiation function named `ksm`. Use short variable names (`n, m, t, ans, res`)</rule>
  </guideline>
</style-guidelines>

<output-format>
  <rule>Only output C++ code, no explanations or descriptions</rule>
  <rule>Code wrapped in ```cpp</rule>
  <rule>Must output complete code, no truncation allowed</rule>
</output-format>
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
