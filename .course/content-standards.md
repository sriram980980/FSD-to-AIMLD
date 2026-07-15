# Content Standards

## Lesson Quality Checklist

Before marking a lesson as `reviewed`:

### Structure
- [ ] Starts with FSD analogy (Hook section)
- [ ] Theory follows: equation → plain-language breakdown → worked example
- [ ] Python snippet runs without modification (test it)
- [ ] Java snippet present if ecosystem has tooling
- [ ] Stack comparison table present if both stacks covered
- [ ] Exactly 3 key takeaways (no more, no fewer)

### Content
- [ ] Every math symbol defined on first use
- [ ] At least one numeric worked example
- [ ] No passive hedging language
- [ ] All jargon from ML introduced via FSD analogy first
- [ ] No pseudo-code — real runnable code only

### Code
- [ ] Python 3.10+ with type hints
- [ ] All imports shown
- [ ] Output printed with labels (not bare values)
- [ ] Runs in under 60 seconds on a CPU laptop
- [ ] No external API calls (use local/offline data only in lessons)

---

## Assignment Quality Checklist

Before marking an assignment as `reviewed`:

- [ ] Objective is one clear sentence
- [ ] Starter code compiles/runs as-is (student starts from working state)
- [ ] Expected output is exact or clearly bounded
- [ ] Evaluation criteria are binary (pass/fail per criterion)
- [ ] Extension challenge is present and harder than the base task
- [ ] Python starter code always present
- [ ] Java starter code present only where ecosystem exists

---

## Difficulty Calibration

| Level | Expectation |
|-------|-------------|
| Module 1–2 | Student completes assignment in 1–2 hours. Math review acceptable. |
| Module 3 | Student completes in 2–4 hours. Debugging expected. |
| Module 4 | Student completes in 4–8 hours. External downloads (model weights) OK. |
| Module 5–6 | Student completes in 4–8 hours. Cloud account may be needed. Provide free-tier path. |

---

## Dependency Policy

### Python
- Pin major versions in comments: `# numpy>=1.24`
- No dependency on private packages
- GPU-optional: code must run on CPU with degraded but correct output
- Prefer: `numpy`, `scipy`, `scikit-learn`, `torch`, `transformers`, `langchain`

### Java
- Maven coordinates in comments: `<!-- ai.djl:api:0.26.0 -->`
- Prefer: Apache Commons Math, Tribuo, DJL, LangChain4j
- Spring Boot acceptable for Module 5+ production patterns

---

## Forbidden Patterns

- Showing a formula without a worked numeric example
- Starting code from scratch without starter template
- Using a dataset that requires login or paid access
- Writing `# TODO: implement this` in lesson code (assignments only)
- Analogy that requires ML knowledge to understand (that's circular)
- Java section that just wraps a Python call via Jython/Py4J

---

## Review Process

1. Agent generates content → status `draft`
2. Human reviews against this checklist → status `reviewed`  
3. Human makes corrections if needed
4. Mark `done` when no further changes expected

Track all statuses in `.course/progress.json`.
