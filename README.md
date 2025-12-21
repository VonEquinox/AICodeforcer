# AICodeforcer

Gemini-powered algorithm problem solver agent.

使用 Gemini AI 自动解决 Codeforces 等 OJ 平台的算法竞赛题目，支持自动对拍验证和 Python 转 C++ 翻译。

## 功能特性

- 自动分析算法题目并生成解法
- 使用沙箱环境执行代码测试
- 三路共识暴力生成（3 个独立 Agent 必须一致）
- 自动对拍验证（1000 组随机测试）
- Python 代码自动翻译为竞赛风格 C++
- 支持交互式反馈优化

## 安装

```bash
# 克隆项目
git clone https://github.com/yourname/AICodeforcer.git
cd AICodeforcer

# 使用 uv 安装依赖
uv sync

# 或使用 pip
pip install -e .
```

## 配置

创建 `.env` 文件并配置 Gemini API：

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash  # 可选，默认 gemini-2.5-flash
GEMINI_BASE_URL=https://your-proxy.com  # 可选，自定义 API 地址
```

## 使用方法

```bash
# 运行
aicodeforcer

# 或
python -m AICodeforcer.main
```

运行后：
1. 粘贴完整的题目内容
2. 输入 `END` 结束输入
3. 等待 AI 分析、编写代码、对拍验证
4. 获得 Python + C++ 双份代码
5. 提交后输入反馈（如 `TLE on test 5`）继续优化
6. 输入 `AC` 或 `quit` 结束

## 项目结构

```
src/AICodeforcer/
├── __init__.py
├── main.py              # CLI 入口
├── types.py             # 类型定义
├── agents/
│   ├── solver.py        # 算法求解 Agent
│   ├── brute_force.py   # 三路共识暴力生成 Agent
│   └── cpp_translator.py # Python 转 C++ Agent
└── tools/
    ├── executor.py      # 沙箱代码执行器
    ├── run_python.py    # 代码执行工具
    └── stress_test.py   # 对拍验证工具
```

## 工作流程

1. **题意分析** - AI 重述题目，建立数学模型
2. **算法设计** - 提出候选方案，分析复杂度
3. **代码实现** - 编写 Python 解法
4. **样例测试** - 运行题目给出的样例
5. **暴力生成** - 三路共识生成暴力算法（3 个 Agent 必须一致）
6. **对拍验证** - 暴力算法 vs 优化算法，1000 组随机测试
7. **代码翻译** - Python 自动翻译为竞赛风格 C++
8. **反馈优化** - 根据 OJ 反馈继续迭代

## 依赖

- Python >= 3.10
- google-genai >= 1.0.0
- pydantic >= 2.0
- python-dotenv >= 1.2.1

## License

MIT
