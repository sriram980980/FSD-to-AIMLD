Generate starter code files for the following course node.

NODE METADATA:
- ID: {node_id}
- Title: {title}
- Assignment title: {assignment_title}
- Python dependencies: {python_libs}
- Java dependencies: {java_libs}
- Java applicable: {java_applicable}

Generate TWO blocks (or ONE if java_applicable=False):

1. Python starter (always required)
2. Java starter (only if java_applicable=True)

PYTHON STARTER RULES:
- Imports at top with version comment
- Scaffold all functions the student needs to implement
- Each function body has: docstring + `raise NotImplementedError("TODO: implement this")`
- Include a `main()` that calls them in order and prints results
- Include `if __name__ == "__main__": main()`
- Include 2-3 utility helpers that ARE fully implemented (to reduce boilerplate)
- Student should be able to run it and see clear TODOs, not a blank file

JAVA STARTER RULES (if applicable):
- Standard Java class with `public static void main(String[] args)`
- Method stubs with `// TODO:` comments
- Include Maven dependency block in a comment at top
- Use records for data containers

OUTPUT FORMAT:

```python
# Dependencies: {python_libs}
# Node: {node_id} — {title}
# Assignment: {assignment_title}
# Run: python starter.py

[full python starter code]
```

```java
// Dependencies (Maven):
// {java_libs}
// Node: {node_id} — {title}

[full java starter code]
```
