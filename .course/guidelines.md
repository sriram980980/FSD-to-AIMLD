# Course Authoring Guidelines

## Core Principles

1. **WHY before HOW** — Motivation precedes implementation. Never show code before the reader understands what problem it solves.
2. **FSD analogy first** — Every new concept gets one sharp analogy from the reader's existing Fullstack world.
3. **Runnable code** — Every snippet runs as-is. No pseudo-code. No "fill in the blanks" mid-snippet.
4. **Production focus** — Prefer production patterns over academic ones. Show the thing you'd actually deploy.
5. **Dual stack** — Python primary. Java parallel only where real tooling exists. Note gaps explicitly.

---

## Lesson Structure (lesson.md)

```
# [Section X.X.X] — [Topic Name]

## Hook
One sentence. FSD analogy that maps this ML concept to something the reader already knows.
Example: "Gradient descent is like webpack's bundle optimizer — it tweaks knobs (weights)
iteratively until the output (loss) is minimal."

## The Problem
What breaks without this concept? Why does it matter? (2-3 sentences max)

## Theory
Math → intuition → code progression.
- Start with the equation
- Explain each term in plain language
- Show how it connects to the FSD analogy

## Python Implementation
\`\`\`python
# runnable code, no pseudo-code
\`\`\`

## Java Implementation (if ecosystem exists)
Library: [library name + version]
\`\`\`java
// runnable code
\`\`\`

## Stack Comparison (if both exist)
| Dimension | Python | Java |
|-----------|--------|------|
| Library | ... | ... |

## Key Takeaways
- Bullet 1 (max 3 total)
- Bullet 2
- Bullet 3
```

---

## Assignment Structure (assignment.md)

```
# Assignment [X.X.X] — [Assignment Title]

## Objective
One sentence. What the student builds and what they prove by building it.

## Background
What they need to know. Link back to lesson.md.

## Task
Numbered steps. Each step is atomic and testable.

## Starter Code
See starter/starter.py (and starter/starter.java where applicable).

## Expected Output
Exact output or acceptable range. Student knows when they're done.

## Evaluation Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Extension Challenge (Optional)
One harder task for students who finish early. No starter code provided.
```

---

## Writing Rules

### Language
- Active voice. Imperative for instructions ("Run this", "Notice that").
- No passive hedging ("it might be noted that", "one could argue").
- Technical terms exact — never paraphrase `gradient` as "the slope thingy".

### Code
- Python 3.10+. Type hints on function signatures.
- Java 17+. Use records where appropriate.
- All imports shown. All dependencies listed at top of file as comments.
- Variable names descriptive — `learning_rate` not `lr` in lesson code (assignments can abbreviate).
- Output printed with labels: `print(f"Loss: {loss:.4f}")` not just `print(loss)`.

### Math
- Use LaTeX notation in markdown: `$\nabla L = \frac{\partial L}{\partial w}$`
- Define every symbol on first use.
- Always show a worked numeric example alongside the formula.

### FSD Analogies (approved list, use or extend)
| ML Concept | FSD Analogy |
|------------|-------------|
| Gradient descent | Webpack optimizer tuning bundle size |
| Loss function | HTTP 4xx/5xx rate — lower is better |
| Overfitting | Memorizing test IDs instead of query logic |
| Embedding | Hashing a user object into a fixed-size JWT |
| Attention | SQL JOIN with learned weights |
| Transformer | Event-driven architecture with global pub/sub |
| RAG | API gateway routing queries to specialized microservices |
| Fine-tuning | Forking a framework and patching one layer |
| Batch normalization | Request rate normalization across microservices |
| Tokenization | Lexer/parser splitting source into AST tokens |

---

## Java Ecosystem Rules

Include Java section ONLY when the library is production-grade and maintained:

| Domain | Include | Library |
|--------|---------|---------|
| Math/stats | YES | Apache Commons Math 3.x |
| Classical ML | YES | Tribuo 4.x, Weka 3.x |
| Deep learning | YES | Deep Java Library (DJL) |
| GenAI/LLM | YES | LangChain4j, Spring AI |
| MLOps/serving | NO | Gap — note it explicitly |
| Distributed training | NO | Gap — Python-only |

When Java section is omitted, add:
> **Java:** No production-equivalent in Java ecosystem. Python-only for this topic.

---

## Progress Tracking

All leaf nodes tracked in `.course/progress.json`. States:
- `"todo"` — not started
- `"draft"` — content generated, needs review
- `"reviewed"` — human-reviewed
- `"done"` — finalized

Update progress after every content generation run.
