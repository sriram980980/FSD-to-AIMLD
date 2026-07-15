# Dependencies: langchain>=0.2, langchain-community>=0.2, anthropic>=0.27
# Node: 5.2 — AI Agents — Orchestration, Tool Use & Memory
# Run: python starter.py
#
# All Tasks 1-6 use only Python's standard library (ast, re, textwrap).
# No API key is required until the Extension Challenge.
# Extension: swap ast-based tools for a real LLM via anthropic>=0.27.

import ast
import textwrap
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Constants — the code under review and the acceptance criteria
# ---------------------------------------------------------------------------

SAMPLE_CODE: str = textwrap.dedent("""\
    def add_numbers(a, b):
        return a + b

    def compute_area(radius: float) -> float:
        \"\"\"Compute the area of a circle given its radius.\"\"\"
        import math
        return math.pi * radius * radius

    def helper(data):
        total = 0
        for item in data:
            total += item
        return total

    def process(items):
        results = []
        for x in items:
            val = transform(x)
            results.append(val)
        return results
""")

REVIEW_CRITERIA: list[str] = [
    "All functions must have docstrings",
    "All function parameters and return values must have type annotations",
    "No undefined names called as functions",
]

# ---------------------------------------------------------------------------
# IMPLEMENTED HELPERS — read and understand these before writing your stubs
# ---------------------------------------------------------------------------


def load_sample_code() -> str:
    """Return the shared SAMPLE_CODE string for the pipeline to review."""
    return SAMPLE_CODE


def find_missing_docstrings(code: str) -> list[str]:
    """Return names of functions whose body does not begin with a string literal.

    Implementation reference: parse with ast.parse(), walk for ast.FunctionDef,
    check whether node.body[0] is an ast.Expr wrapping an ast.Constant str.
    """
    tree = ast.parse(code)
    missing: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            has_docstring = (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            )
            if not has_docstring:
                missing.append(node.name)
    return missing


def format_report(findings: list[dict[str, str]]) -> str:
    """Format a list of finding dicts into a human-readable aligned report string."""
    if not findings:
        return "  (no issues found)"
    lines: list[str] = []
    for finding in findings:
        kind = finding.get("kind", "UNKNOWN")
        detail = finding.get("detail", "")
        lines.append(f"  [{kind:<26}] {detail}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TASK 1 — implement check_undefined_variables
# ---------------------------------------------------------------------------


def check_undefined_variables(code: str) -> list[str]:
    """Return names that are called as plain functions but never defined in the module.

    Steps:
      1. Parse `code` with ast.parse().
      2. Walk all nodes — collect into `defined`:
           - ast.FunctionDef.name  (function definitions)
           - ast.Name nodes in Store context (assignment targets)
           - ast.alias.asname or ast.alias.name (imports)
      3. Walk all ast.Call nodes — if node.func is ast.Name, add node.func.id
         to `called`.
      4. Build BUILTINS = set(dir(__builtins__)) to exclude built-ins.
      5. Return sorted list of names in called - defined - BUILTINS.

    Expected: check_undefined_variables(SAMPLE_CODE) == ['transform']
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# TASK 2 — implement check_missing_type_hints
# ---------------------------------------------------------------------------


def check_missing_type_hints(code: str) -> list[str]:
    """Return names of functions missing a return annotation or any argument annotation.

    Steps:
      1. Parse `code` with ast.parse().
      2. Walk all ast.FunctionDef nodes.
      3. A function FAILS if:
           - node.returns is None  (no return type annotation), OR
           - any arg in node.args.args has arg.annotation is None,
             EXCLUDING 'self' and 'cls' parameters.
      4. Collect the name of every failing function.

    Expected: check_missing_type_hints(SAMPLE_CODE) == ['add_numbers', 'helper', 'process']
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# TASK 3 — implement build_review_plan
# ---------------------------------------------------------------------------


def build_review_plan(code: str, criteria: list[str]) -> list[str]:
    """Map each criterion to one tool name and return an ordered list of task strings.

    Each returned string must have the format: "<tool_name>: <code>"

    Keyword → tool name mapping:
      "docstring"            → "find_missing_docstrings"
      "annotation" or "type" → "check_missing_type_hints"
      "undefined"            → "check_undefined_variables"

    Steps:
      1. For each criterion in `criteria`, check which keyword it contains
         (case-insensitive substring match).
      2. Append f"{tool_name}: {code}" to the task list.
      3. Return the task list in criteria order.

    Expected: build_review_plan(SAMPLE_CODE, REVIEW_CRITERIA) returns a list
    of 3 strings, each starting with one of the tool names above.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# TASK 4 — implement run_reviewer_agent
# ---------------------------------------------------------------------------


def run_reviewer_agent(
    code: str,
    tasks: list[str],
    tools: dict[str, Callable[[str], list[str]]],
) -> list[dict[str, str]]:
    """Execute a step-traced ReAct loop and return all structured findings.

    For each task string in `tasks`:
      1. Parse the tool name — everything before the first ": ".
      2. Call tools[tool_name](code) → list of result strings.
      3. Convert each result item into a finding dict:
           {"kind": tool_name.upper(), "detail": item}
      4. Print a step-trace line (see Expected Output in assignment.md):
           f"Step {step} | tool: {tool_name:<28}| found: {count} issue(s)"
      5. Extend the findings list with this step's results.

    Return the complete findings list after all tasks.

    Expected: 3 trace lines printed; 7 dicts returned for SAMPLE_CODE.
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# TASK 5 — implement run_critic_agent
# ---------------------------------------------------------------------------


def run_critic_agent(
    findings: list[dict[str, str]],
    criteria: list[str],
) -> dict[str, str]:
    """Validate findings against criteria and return a verdict dict.

    Rules (apply in order):
      1. If `findings` is empty →
             {"verdict": "ACCEPTED", "reason": "No issues found."}
      2. For each criterion, check if at least one finding's "kind" field
         contains a keyword from that criterion (case-insensitive).
         Keywords to check per criterion:
           "docstrings"   → "DOCSTRING" in kind
           "annotations"  → "TYPE" in kind
           "undefined"    → "UNDEFINED" in kind
         If ANY criterion has zero matching findings →
             {"verdict": "REVISE", "reason": f"Unaddressed: {criterion}"}
      3. Otherwise →
             {"verdict": "ACCEPTED", "reason": "All criteria addressed."}

    Expected with 7-finding result set: verdict == "ACCEPTED"
    Expected with empty findings list:  verdict == "ACCEPTED", reason == "No issues found."
    Expected with findings that miss one criterion: verdict == "REVISE"
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# TASK 6 — wire run_code_review_pipeline
# ---------------------------------------------------------------------------


def run_code_review_pipeline(
    code: str,
    criteria: list[str],
) -> dict[str, Any]:
    """Orchestrate Planner → Reviewer → Critic and return the full report dict.

    Steps:
      1. tasks   = build_review_plan(code, criteria)
      2. Build tool_registry — a dict mapping each tool name (str) to its
         callable: find_missing_docstrings, check_missing_type_hints,
         check_undefined_variables.
      3. findings = run_reviewer_agent(code, tasks, tool_registry)
      4. verdict  = run_critic_agent(findings, criteria)
      5. Return {"tasks": tasks, "findings": findings, "verdict": verdict}
    """
    raise NotImplementedError("TODO: implement this")


# ---------------------------------------------------------------------------
# Entry point — runs as-is; NotImplementedError is caught per task
# ---------------------------------------------------------------------------


def main() -> None:
    separator = "=" * 60
    code = load_sample_code()

    print(f"\n{separator}")
    print("Code Review Pipeline — Node 5.2")
    print(separator)
    line_count = code.count("\n")
    print(f"Sample code: {line_count} lines, {len(code)} chars\n")

    # Implemented helper — always works
    missing_docs = find_missing_docstrings(code)
    print(f"[Helper] Functions missing docstrings : {missing_docs}")

    # Task 1
    try:
        undefined = check_undefined_variables(code)
        print(f"[Task 1] Undefined names called as functions: {undefined}")
    except NotImplementedError:
        print("[Task 1] check_undefined_variables    — NOT YET IMPLEMENTED")

    # Task 2
    try:
        no_hints = check_missing_type_hints(code)
        print(f"[Task 2] Functions missing type hints: {no_hints}")
    except NotImplementedError:
        print("[Task 2] check_missing_type_hints     — NOT YET IMPLEMENTED")

    # Tasks 3-6: full pipeline
    print(f"\n{separator}")
    print("Running full pipeline (Tasks 3-6)...")
    print(separator)
    print()
    try:
        report = run_code_review_pipeline(code, REVIEW_CRITERIA)
        print(f"\nPlan  : {len(report['tasks'])} tasks")
        verdict_dict: dict[str, str] = report["verdict"]
        print(f"Verdict: {verdict_dict['verdict']} — {verdict_dict['reason']}")
        print("\nFindings:")
        print(format_report(report["findings"]))
    except NotImplementedError:
        print("Pipeline not yet implemented — complete Tasks 3-6 to see full output.")

    print(f"\n{separator}")
    print("Done.")
    print(separator)


if __name__ == "__main__":
    main()
