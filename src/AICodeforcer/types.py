"""Data types for the algorithm solver."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TestCase:
    """A single test case with input and expected output."""

    input: str
    expected_output: str

    def __post_init__(self) -> None:
        self.input = self.input.strip()
        self.expected_output = self.expected_output.strip()


@dataclass
class Problem:
    """An algorithm problem with description and test cases."""

    description: str
    test_cases: list[TestCase] = field(default_factory=list)
    time_limit_seconds: float = 2.0
    memory_limit_mb: int = 256

    def format_for_prompt(self) -> str:
        """Format the problem for the LLM prompt."""
        lines = [
            "## Problem Description",
            self.description,
            "",
            "## Constraints",
            f"- Time Limit: {self.time_limit_seconds} seconds",
            f"- Memory Limit: {self.memory_limit_mb} MB",
            "",
            "## Test Cases",
        ]

        for i, tc in enumerate(self.test_cases, 1):
            lines.extend([
                f"### Test Case #{i}",
                "Input:",
                "```",
                tc.input,
                "```",
                "Expected Output:",
                "```",
                tc.expected_output,
                "```",
                "",
            ])

        return "\n".join(lines)


ExecutionStatus = Literal[
    "passed",
    "wrong_answer",
    "runtime_error",
    "timeout",
    "memory_exceeded",
    "dangerous_code",
]


@dataclass
class ExecutionResult:
    """Result of executing code against a test case."""

    status: ExecutionStatus
    actual_output: str | None = None
    error_message: str | None = None
    execution_time_ms: float = 0.0
    test_case_index: int = 0

    def format_feedback(self, test_case: TestCase | None = None) -> str:
        """Format the result as feedback for the LLM."""
        lines = [f"Status: {self.status.upper().replace('_', ' ')}"]

        if test_case:
            lines.extend([
                f"Input:\n{test_case.input}",
                f"Expected Output:\n{test_case.expected_output}",
            ])

        if self.actual_output is not None:
            lines.append(f"Actual Output:\n{self.actual_output}")

        if self.error_message:
            lines.append(f"Error:\n{self.error_message}")

        lines.append(f"Execution Time: {self.execution_time_ms:.1f}ms")

        return "\n".join(lines)
