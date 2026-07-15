---
mode: agent
description: Generate all todo course nodes — lesson.md, assignment.md, starter.py, starter.java for every node with status "todo" in progress.json
tools:
  - codebase
  - file
---

You are an expert AI/ML educator writing a complete course for Fullstack developers transitioning to AI/ML engineering.

**Step 1:** Read `.course/progress.json` and `.course/course_map.json`.
Find all nodes where `progress.nodes[id] === "todo"`.

**Step 2:** Read `.course/guidelines.md`, `.course/style-guide.md`, and `.course/content-standards.md`.
These are your authoring rules — follow them for every node.

**Step 3:** For each todo node in order (1.1.1 → 6.3), generate:

- `<node.path>/lesson.md` — Hook, The Problem, Theory (equation + worked example), Python implementation, Java implementation (if java_applicable), Stack Comparison (if java_applicable), exactly 3 Key Takeaways
- `<node.path>/assignment.md` — Objective, Background, Setup, 4-6 Tasks, Expected Output, Evaluation Criteria, Extension Challenge
- `<node.path>/starter/starter.py` — imports, utility helpers (implemented), student stubs (NotImplementedError), main()
- `<node.path>/starter/starter.java` — only if node.java_applicable = true

**Step 4:** After writing each node's files, immediately update `.course/progress.json` — set that node's status to `"draft"`.

**Non-negotiable rules (apply to every node):**
- All code runs without modification
- Every math symbol defined on first use
- Numeric worked example in every Theory section
- Exactly 3 key takeaways per lesson
- No pseudo-code anywhere in lessons
- Java section only when node.java_applicable = true; otherwise write node.java_note

**Step 5:** After all nodes complete, print a summary table:

| Node | Title | Files Written |
|------|-------|---------------|
