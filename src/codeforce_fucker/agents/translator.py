"""C++ translator agent."""

import os
import re

from google import genai
from google.genai import types

CPP_TRANSLATE_PROMPT = """将 Python 代码翻译成等价的 C++17 代码。

要求：
1. 保持算法逻辑完全一致
2. 使用 fast I/O
3. 直接输出完整的 C++ 代码，不要任何解释
"""


class CppTranslator:
    """Translate Python code to C++."""

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
        """将 Python 代码翻译成 C++ 代码。"""
        print("\n" + "=" * 60)
        print("  开始翻译 Python -> C++")
        print("=" * 60)

        config = types.GenerateContentConfig(
            system_instruction=CPP_TRANSLATE_PROMPT,
            temperature=1.0,
            thinking_config=types.ThinkingConfig(thinking_level="high"),
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=python_code)],
            )
        ]

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
                print(f"[翻译] 请求失败 (重试 {retry + 1}/10): {e}")
                if retry == 9:
                    print("[翻译] 翻译失败，跳过 C++ 输出")
                    return None
                import time
                time.sleep(3)

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

        cpp_code = self._extract_cpp_code(response_text)

        if cpp_code:
            print("[翻译] C++ 代码生成成功")
            return cpp_code
        else:
            print("[翻译] 未能提取 C++ 代码")
            return None

    def _extract_cpp_code(self, text: str) -> str | None:
        """Extract C++ code from response text."""
        patterns = [
            r"```cpp\n(.*?)```",
            r"```c\+\+\n(.*?)```",
            r"```\n(.*?)```",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[-1].strip()

        text = text.strip()
        if text.startswith("#include"):
            return text

        return None
