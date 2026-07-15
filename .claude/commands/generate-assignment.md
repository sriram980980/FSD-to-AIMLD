---
description: Generate assignment.md + starter code for a course node. Usage: /generate-assignment <node-id>
argument-hint: <node-id>  e.g. 1.1.1
---

Read `.course/course_map.json` and find the node with id matching `$ARGUMENTS`.

Then read `.course/guidelines.md` and `.course/content-standards.md`.

Generate TWO files:

**1. `<node.path>/assignment.md`** with:
- Objective (one sentence)
- Background (2-3 sentences linking to lesson)
- Setup block (`pip install <python_libs>`)
- Tasks (4-6 atomic, testable steps)
- Expected Output (exact or bounded)
- Evaluation Criteria (binary pass/fail checkboxes)
- Extension Challenge (harder, no starter code)

**2. `<node.path>/starter/starter.py`** with:
- All imports at top (with version comment)
- Function stubs — all raise `NotImplementedError("TODO: implement this")`
- 2-3 fully implemented utility helpers to reduce boilerplate
- `main()` that calls all stubs in order
- `if __name__ == "__main__": main()`

**3. `<node.path>/starter/starter.java`** — only if `java_applicable: true`:
- Standard Java class with `main`
- Method stubs with `// TODO:` comments
- Maven dependency block in a comment at top

After writing files, update `.course/progress.json` status to `"draft"` for that node (if not already).

Print paths of all written files when done.
