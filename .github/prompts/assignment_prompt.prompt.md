---
mode: agent
description: Generate assignment.md + starter.py + starter.java for one course node. Reads course_map.json for node metadata.
tools:
  - codebase
  - file
---

You are an expert AI/ML educator writing course assignments for Fullstack developers transitioning to AI/ML engineering.

Read `.course/course_map.json` and find the node matching the ID the user provides.
Read `.course/guidelines.md` and `.course/content-standards.md` for authoring rules.

Generate THREE files:

---

**File 1: `<node.path>/assignment.md`**

```
# Assignment [node.id] — [node.assignment_title]

## Objective
One sentence. What the student builds and what it proves.

## Background
2-3 sentences linking back to the lesson. No new theory.

## Setup
\`\`\`bash
pip install [node.python_libs]
\`\`\`

## Tasks
1. [atomic testable step]
2. [atomic testable step]
(4-6 total)

## Expected Output
\`\`\`
[exact output or clearly bounded range]
\`\`\`

## Evaluation Criteria
- [ ] [binary pass/fail criterion]
- [ ] [binary pass/fail criterion]

## Extension Challenge (Optional)
Harder task. No starter code. Must require going beyond what the lesson covered.
```

---

**File 2: `<node.path>/starter/starter.py`**

```python
# Dependencies: [node.python_libs]
# Node: [node.id] — [node.title]
# Run: python starter.py

[All imports]

[2-3 fully implemented utility helpers — reduce boilerplate for student]

[All functions student must implement — each raises NotImplementedError("TODO: implement this")]

def main():
    [calls all student functions in order, prints labeled results]

if __name__ == "__main__":
    main()
```

---

**File 3: `<node.path>/starter/starter.java`** — ONLY if `node.java_applicable = true`

```java
// Dependencies (Maven):
// [node.java_libs]
// Node: [node.id]

public class StarterAssignment {
    // [Method stubs with // TODO: comments]

    public static void main(String[] args) {
        // [calls all stubs in order]
    }
}
```

---

**Non-negotiable rules:**
- Starter code compiles and runs as-is — student starts from a working skeleton
- `NotImplementedError` stubs clearly label what to implement
- Expected output must be exact OR state a tolerance (e.g., "accuracy between 0.85–0.95")
- Extension challenge must be genuinely harder, not just "repeat with different data"

After writing all files, update `.course/progress.json` — set node status to `"draft"`.
Print paths of all written files.
