# FSD → AI/ML Developer Course — Copilot Instructions

You are helping author a course that transforms Fullstack developers into AI/ML engineers.

## Audience Context

Students are experienced Fullstack developers (React/Node/Spring/SQL). They know OOP, REST APIs, SQL. They do NOT know tensors, gradients, or transformers. Never assume ML knowledge. Always bridge from their existing skills.

## Content Generation Rules

### Always do
- Lead every new concept with a 1-sentence FSD analogy (e.g., "gradient descent is like webpack's bundle size optimizer")
- Follow order: WHY → math → intuition → code
- Write runnable code — every snippet executes as-is, no pseudo-code
- Define every math symbol on first use
- Include a numeric worked example alongside every formula
- End lessons with exactly 3 key takeaways

### Never do
- Start with ML jargon before the FSD analogy
- Write theoretical snippets with `# pseudo-code` comments
- Use hedging language: "might", "could be", "one way to think about it"
- Add `# TODO: implement this` in lesson code (assignments only)
- Write Java sections without real production tooling

## File Structure

Each lesson node lives at: `modules/<module>/<section>/<topic>/`
Files:
- `lesson.md` — theory, code, takeaways
- `assignment.md` — objective, tasks, rubric, extension
- `starter/starter.py` — stubbed Python with NotImplementedError TODOs
- `starter/starter.java` — stubbed Java (only when java_applicable)

## Lesson Format

```markdown
# [X.X.X] — Title

## Hook
[One FSD analogy sentence]

## The Problem
[2-3 sentences: why this concept exists]

## Theory
### [Concept]
$$[formula]$$
Where: $symbol$ = definition

**Worked example:** [numeric walkthrough]

## Python Implementation
\`\`\`python
# Dependencies: library>=version
[runnable code with labeled print output]
\`\`\`

## Java Implementation
[real library or: > **Java:** no equivalent — Python only]

## Stack Comparison
| Dimension | Python | Java |
|-----------|--------|------|

## Key Takeaways
- [max 3 bullets]
```

## Assignment Format

```markdown
# Assignment X.X.X — Title

## Objective
[one sentence]

## Tasks
1. [atomic testable step]

## Expected Output
\`\`\`
[exact or bounded output]
\`\`\`

## Evaluation Criteria
- [ ] binary pass/fail criterion

## Extension Challenge (Optional)
[harder task, no starter code]
```

## Java Policy

Show Java only when real production tooling exists:
| Domain | Library |
|--------|---------|
| Math | Apache Commons Math 3.x |
| ML | Tribuo 4.x, Weka |
| Deep Learning | Deep Java Library (DJL) |
| GenAI | LangChain4j, Spring AI |
| MLOps/serving | **No Java equivalent — note the gap** |
| Distributed training | **No Java equivalent — note the gap** |

## Code Style

- Python 3.10+, type hints on all function signatures
- Java 17+, records for data containers
- All imports shown, dependencies in top comment
- Variable names: `learning_rate` not `lr` in lesson code
- Print output with labels: `print(f"Loss: {loss:.4f}")`

## Terminology

Use exact: `weights`, `learning rate`, `loss`, `batch`, `epoch`, `embedding`, `token`, `fine-tune`, `inference`.
Never substitute with informal synonyms.

## Approved FSD Analogies

| ML Concept | FSD Analogy |
|------------|-------------|
| Gradient descent | Webpack optimizer tuning bundle size |
| Loss function | HTTP 4xx/5xx rate — lower is better |
| Overfitting | Memorizing test IDs instead of query logic |
| Embedding | Hashing a user object into a fixed-size JWT |
| Attention | SQL JOIN with learned weights |
| Transformer | Event-driven architecture with global pub/sub |
| RAG | API gateway routing to specialized microservices |
| Fine-tuning | Forking a framework and patching one layer |
| Tokenization | Lexer splitting source into AST tokens |
