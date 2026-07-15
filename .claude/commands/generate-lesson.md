---
description: Generate lesson.md for a course node. Usage: /generate-lesson <node-id> e.g. /generate-lesson 1.1.1
argument-hint: <node-id>  e.g. 1.1.1
---

Read `.course/course_map.json` and find the node with id matching `$ARGUMENTS`.

Then read `.course/guidelines.md` and `.course/style-guide.md`.

Generate a complete `lesson.md` file for that node and write it to `<node.path>/lesson.md`.

Follow the exact lesson structure from guidelines.md:
1. Hook — one FSD analogy sentence (use or improve the `fsd_analogy` field from the node)
2. The Problem — 2-3 sentences motivation
3. Theory — equation, symbol definitions, numeric worked example
4. Python Implementation — runnable code using the node's `python_libs`
5. Java Implementation — only if `java_applicable: true`, else write the `java_note`
6. Stack Comparison table — only if `java_applicable: true`
7. Key Takeaways — exactly 3 bullets

After writing the file, update `.course/progress.json` to set that node's status to `"draft"`.

Print the path of the written file when done.
