---
description: Show course content generation progress table
---

Read `.course/progress.json` and `.course/course_map.json`.

Print a markdown table:

| ID | Title | Status | lesson.md | assignment.md | starter.py |
|----|-------|--------|-----------|---------------|------------|

For each node, check if files actually exist on disk (not just what progress.json says):
- `<node.path>/lesson.md`
- `<node.path>/assignment.md`  
- `<node.path>/starter/starter.py`

Show ✓ or ✗ for each file. Show status from progress.json.

Print counts at end: todo / draft / reviewed / done.
