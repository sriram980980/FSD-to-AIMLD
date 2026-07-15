---
description: Generate lesson + assignment + starter for ALL nodes with status "todo" in progress.json
---

Read `.course/progress.json` and `.course/course_map.json`.

Find all nodes where `progress.nodes[id] === "todo"`.

For each todo node in order (1.1.1 → 6.3):
1. Generate `lesson.md` following the structure in `.course/guidelines.md`
2. Generate `assignment.md` following the structure in `.course/guidelines.md`
3. Generate `starter/starter.py` (always) and `starter/starter.java` (if `java_applicable: true`)
4. Update that node's status to `"draft"` in `.course/progress.json` immediately after writing

Read `.course/style-guide.md` and `.course/content-standards.md` before generating any content.

After all nodes complete, print a summary table:
| Node | Status | Files Written |
