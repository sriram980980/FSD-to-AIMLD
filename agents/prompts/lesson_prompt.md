Generate a complete lesson.md for the following course node.

NODE METADATA:
- ID: {node_id}
- Title: {title}
- Topic: {lesson_topic}
- FSD Analogy (use this verbatim or improve it): {fsd_analogy}
- Python dependencies: {python_libs}
- Java dependencies: {java_libs}
- Java applicable: {java_applicable}
- Java note: {java_note}

OUTPUT FORMAT — produce exactly this structure:

# [{node_id}] — {title}

## Hook
[One sentence FSD analogy — must map the ML concept to a known Fullstack pattern]

## The Problem
[2-3 sentences: what breaks or becomes impossible without this concept?]

## Theory

### [Main concept name]
[Math equation in LaTeX inline syntax]

[Plain-language breakdown: define each symbol]

[Worked numeric example — small numbers, show the calculation step by step]

## Python Implementation

```python
# Dependencies: {python_libs}
[full runnable code — imports, function definitions, example call, printed output with labels]
```

[3-5 sentences explaining what to notice in the output]

## Java Implementation
[Only if java_applicable=True. Otherwise write:]
> **Java:** {java_note}

```java
// Dependencies: {java_libs}
[full runnable code]
```

## Stack Comparison
[Only if java_applicable=True — a markdown table comparing Python vs Java approach on: Library, API style, Performance notes, When to use]

## Key Takeaways
- [Takeaway 1]
- [Takeaway 2]
- [Takeaway 3]

RULES:
- Code must run without modification
- Define every math symbol on first use
- Numeric worked example required
- Max 3 takeaways
- No pseudo-code
- No hedging language ("might", "could be", "one way to think about it")
