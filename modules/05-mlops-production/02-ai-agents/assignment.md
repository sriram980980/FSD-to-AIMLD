# Assignment 5.2 — Build Multi-Agent Autonomous Code Reviewer

## Objective

Build a three-agent pipeline (Planner → Reviewer → Critic) that autonomously reviews
Python source code for missing docstrings, missing type hints, and undefined function
calls, then returns a structured verdict — proving you can wire ReAct tool use and
multi-agent orchestration without a hand-coded task sequence.

## Background

In the lesson you saw that a ReAct agent advances by emitting a Thought, selecting a
tool from its registry, and appending the Observation to its growing context. This
assignment replaces the single-agent math solver with a three-role pipeline: the
**Planner** decomposes the review goal into an ordered tool-call sequence, the
**Reviewer** executes those tools in a step-traced ReAct loop, and the **Critic**
validates that every acceptance criterion is covered before issuing a verdict. The
tool registry is built from Python's `ast` module — no API key is required for the
core tasks; the Extension Challenge wires in a real LLM.

## Setup

```bash
pip install langchain>=0.2 langchain-community>=0.2 anthropic>=0.27
```

The base tasks (1–6) use only Python's standard library (`ast`, `re`, `textwrap`).
No API key is needed until the Extension Challenge.

## Tasks

### Task 1 — Implement `check_undefined_variables(code)`

Use `ast.walk()` to collect every name that appears in a `Call` node's `func`
attribute with an `ast.Name` type (i.e., plain function calls, not method calls).
Separately collect every name that is *defined* in the module: function definitions
(`ast.FunctionDef.name`), assignments (`ast.Name` in `Store` context), and import
aliases (`ast.alias.asname or ast.alias.name`). Return names that are called but
not defined and not in Python's built-in namespace (`dir(__builtins__)`).

**Acceptance check:** `check_undefined_variables(SAMPLE_CODE)` returns `['transform']`.

---

### Task 2 — Implement `check_missing_type_hints(code)`

Walk all `ast.FunctionDef` nodes. A function *fails* if any of these hold:

- `node.returns is None` (no return annotation)
- Any argument in `node.args.args` (excluding `self` and `cls`) has `arg.annotation is None`

Return the list of failing function names.

**Acceptance check:** `check_missing_type_hints(SAMPLE_CODE)` returns
`['add_numbers', 'helper', 'process']`.

---

### Task 3 — Implement `build_review_plan(code, criteria)`

The Planner maps each criterion string to one tool name by keyword matching
(see the table below) and returns an ordered list of task strings in the format
`"<tool_name>: <code>"`.

| Criterion keyword | Tool name |
|-------------------|-----------|
| `docstring`       | `find_missing_docstrings` |
| `annotation` or `type` | `check_missing_type_hints` |
| `undefined`       | `check_undefined_variables` |

**Acceptance check:** `build_review_plan(SAMPLE_CODE, REVIEW_CRITERIA)` returns
a list of exactly 3 strings, each starting with one of the tool names above.

---

### Task 4 — Implement `run_reviewer_agent(code, tasks, tools)`

Execute a ReAct loop over the task list:

1. For each task string, parse the tool name (everything before `": "`).
2. Call `tools[tool_name](code)` to get a list of result strings.
3. Convert each result item into a finding dict:
   `{"kind": "<TOOL_NAME_UPPER>", "detail": "<item>"}`.
4. Print a step-trace line: `"Step N | tool: <name> | found: <count> issue(s)"`.
5. Append all findings and return the full list after all tasks complete.

**Acceptance check:** After Tasks 1–3 are implemented, calling
`run_reviewer_agent(SAMPLE_CODE, plan, tool_registry)` prints 3 step-trace lines
and returns 7 findings total (3 docstring + 3 type-hint + 1 undefined).

---

### Task 5 — Implement `run_critic_agent(findings, criteria)`

Apply these rules in order:

1. If `findings` is empty → `{"verdict": "ACCEPTED", "reason": "No issues found."}`.
2. For each criterion, check whether at least one finding's `"kind"` field contains
   a keyword from that criterion (case-insensitive). If any criterion has **zero**
   matching findings → `{"verdict": "REVISE", "reason": "Unaddressed: <criterion>"}`.
3. Otherwise → `{"verdict": "ACCEPTED", "reason": "All criteria addressed."}`.

**Acceptance check:** With the 7 findings from Task 4 and `REVIEW_CRITERIA`,
`run_critic_agent` returns `{"verdict": "ACCEPTED", "reason": "All criteria addressed."}`.

---

### Task 6 — Wire `run_code_review_pipeline(code, criteria)`

Orchestrate all agents in sequence:

1. `tasks = build_review_plan(code, criteria)`
2. Build `tool_registry = {"find_missing_docstrings": find_missing_docstrings, ...}`
3. `findings = run_reviewer_agent(code, tasks, tool_registry)`
4. `verdict = run_critic_agent(findings, criteria)`
5. Return `{"tasks": tasks, "findings": findings, "verdict": verdict}`

Run `python starter.py` and confirm the full output matches the Expected Output
section below.

## Expected Output

```
============================================================
Code Review Pipeline — Node 5.2
============================================================
Sample code: 16 lines, 312 chars

[Helper] Functions missing docstrings : ['add_numbers', 'helper', 'process']
[Task 1] Undefined names called as functions: ['transform']
[Task 2] Functions missing type hints: ['add_numbers', 'helper', 'process']

============================================================
Running full pipeline (Tasks 3-6)...
============================================================

Step 1 | tool: find_missing_docstrings    | found: 3 issue(s)
Step 2 | tool: check_missing_type_hints   | found: 3 issue(s)
Step 3 | tool: check_undefined_variables  | found: 1 issue(s)

Plan  : 3 tasks
Verdict: ACCEPTED — All criteria addressed.

Findings:
  [FIND_MISSING_DOCSTRINGS   ] add_numbers
  [FIND_MISSING_DOCSTRINGS   ] helper
  [FIND_MISSING_DOCSTRINGS   ] process
  [CHECK_MISSING_TYPE_HINTS  ] add_numbers
  [CHECK_MISSING_TYPE_HINTS  ] helper
  [CHECK_MISSING_TYPE_HINTS  ] process
  [CHECK_UNDEFINED_VARIABLES ] transform

============================================================
Done.
============================================================
```

Exact function names and finding counts must match. Verdict must be `ACCEPTED`.

## Evaluation Criteria

- [ ] `check_undefined_variables(SAMPLE_CODE)` returns `['transform']` and no other names
- [ ] `check_missing_type_hints(SAMPLE_CODE)` returns exactly `['add_numbers', 'helper', 'process']`
- [ ] `build_review_plan` returns a list of 3 strings, each prefixed with a valid tool name
- [ ] `run_reviewer_agent` prints a step-trace line for every task and returns 7 findings
- [ ] `run_critic_agent` returns `"ACCEPTED"` for the 7-finding result set and `"REVISE"` when given an empty findings list
- [ ] `run_code_review_pipeline` produces output that matches Expected Output exactly (function names, counts, verdict)
- [ ] `python starter.py` runs to completion with no unhandled exceptions after all tasks are implemented

## Extension Challenge

Replace the `ast`-based static-analysis tools with a **real LLM reviewer** using the
Anthropic Claude API:

1. Implement a `ClaudeReviewerLLM` class that wraps `anthropic.Anthropic().messages.create`.
2. Register three new tools — `llm_check_docstrings`, `llm_check_type_hints`,
   `llm_check_logic_errors` — where each tool sends the code to Claude with a
   targeted system prompt and returns a structured JSON list of findings.
3. Modify `run_reviewer_agent` to accept either the `ast`-based tools or the LLM
   tools interchangeably (duck-typed — same `Callable[[str], list[str]]` signature).
4. Add a **fourth agent role — Synthesizer** — that takes the Critic's `"REVISE"`
   verdict, generates a corrected version of the source code, and re-runs the full
   pipeline until the Critic emits `"ACCEPTED"` or a round limit of 3 is reached.

No starter code is provided. Refer to the
[Anthropic Messages API docs](https://docs.anthropic.com/en/api/messages) and the
LangChain `@tool` decorator for integration patterns.
