Generate a complete assignment.md for the following course node.

NODE METADATA:
- ID: {node_id}
- Title: {title}
- Assignment title: {assignment_title}
- Lesson topic: {lesson_topic}
- Python dependencies: {python_libs}
- Java applicable: {java_applicable}

OUTPUT FORMAT — produce exactly this structure:

# Assignment {node_id} — {assignment_title}

## Objective
[One sentence: what the student builds and what it proves]

## Background
[2-3 sentences linking back to lesson concepts. No new theory here.]

## Setup

```bash
pip install {python_libs}
```

## Tasks

1. [Atomic step — one action, one output]
2. [Atomic step]
3. [Atomic step]
[4-6 total tasks, each independently testable]

## Expected Output

```
[Exact expected output or clearly bounded range]
```

## Evaluation Criteria

- [ ] [Criterion 1 — binary pass/fail]
- [ ] [Criterion 2]
- [ ] [Criterion 3]
- [ ] [Criterion 4]

## Extension Challenge (Optional)

[One harder task. No starter code. Requires going beyond what the lesson covered.]

RULES:
- Each task must be testable in isolation
- Expected output must be exact or have explicit tolerance (e.g., "accuracy between 0.85 and 0.95")
- No theoretical questions — all tasks produce runnable artifacts
- Extension challenge must be genuinely harder, not just "do it again with different data"
- Java section only if java_applicable=True
