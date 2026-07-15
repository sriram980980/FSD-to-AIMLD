# FSD → AI/ML Developer

> Transform your Fullstack engineering career into AI/ML engineering — with real code, not slides.

**Audience:** React/Node/Spring/SQL developers. Strong programmer. Zero ML.
**Goal:** Production AI/ML engineer. Not a data scientist.
**Stack:** Python (primary) + Java (parallel) where ecosystem exists.

---

## Why This Course

Fullstack developers already know the hard parts: systems thinking, APIs, data flow, abstraction. This course translates that foundation into ML/AI vocabulary and practice — anchoring every concept in a FSD analogy before showing the math and code.

---

## Prerequisites

- Comfortable with OOP, REST APIs, SQL
- Python basics (or willing to pick up fast)
- Git, terminal, basic Linux commands

---

## Course Index

### Module 1 — Math Foundations
| Section | Topic | Lesson | Assignment |
|---------|-------|--------|------------|
| 1.1.1 | Vectors & Matrices | [lesson](modules/01-math-foundations/01-linear-algebra/01-vectors-matrices/lesson.md) | [assignment](modules/01-math-foundations/01-linear-algebra/01-vectors-matrices/assignment.md) |
| 1.1.2 | Calculus & Derivatives | [lesson](modules/01-math-foundations/01-linear-algebra/02-calculus-derivatives/lesson.md) | [assignment](modules/01-math-foundations/01-linear-algebra/02-calculus-derivatives/assignment.md) |
| 1.2.1 | Distributions & Bayesian Stats | [lesson](modules/01-math-foundations/02-probability-statistics/01-distributions-bayesian/lesson.md) | [assignment](modules/01-math-foundations/02-probability-statistics/01-distributions-bayesian/assignment.md) |

### Module 2 — Machine Learning Core
| Section | Topic | Lesson | Assignment |
|---------|-------|--------|------------|
| 2.1.1 | Linear Models (Regression & Classification) | [lesson](modules/02-ml-core/01-supervised-learning/01-linear-models/lesson.md) | [assignment](modules/02-ml-core/01-supervised-learning/01-linear-models/assignment.md) |
| 2.1.2 | Tree-Based Models & Ensembles | [lesson](modules/02-ml-core/01-supervised-learning/02-tree-based-models/lesson.md) | [assignment](modules/02-ml-core/01-supervised-learning/02-tree-based-models/assignment.md) |
| 2.2.1 | PCA & Clustering | [lesson](modules/02-ml-core/02-unsupervised-learning/01-dimensionality-reduction/lesson.md) | [assignment](modules/02-ml-core/02-unsupervised-learning/01-dimensionality-reduction/assignment.md) |

### Module 3 — Deep Learning
| Section | Topic | Lesson | Assignment |
|---------|-------|--------|------------|
| 3.1.1 | MLP — Forward & Backward Pass | [lesson](modules/03-deep-learning/01-neural-networks/01-mlp-forward-backward/lesson.md) | [assignment](modules/03-deep-learning/01-neural-networks/01-mlp-forward-backward/assignment.md) |
| 3.1.2 | Optimization — Gradient Descent Variants | [lesson](modules/03-deep-learning/01-neural-networks/02-optimization/lesson.md) | [assignment](modules/03-deep-learning/01-neural-networks/02-optimization/assignment.md) |
| 3.2.1 | Attention Mechanisms — Self-Attention | [lesson](modules/03-deep-learning/02-sequence-models/01-attention-mechanisms/lesson.md) | [assignment](modules/03-deep-learning/02-sequence-models/01-attention-mechanisms/assignment.md) |

### Module 4 — Generative AI & LLMs
| Section | Topic | Lesson | Assignment |
|---------|-------|--------|------------|
| 4.1 | Transformer Architecture (GPT vs BERT) | [lesson](modules/04-genai-llms/01-transformer-architecture/lesson.md) | [assignment](modules/04-genai-llms/01-transformer-architecture/assignment.md) |
| 4.2 | Fine-Tuning — LoRA & QLoRA | [lesson](modules/04-genai-llms/02-fine-tuning/lesson.md) | [assignment](modules/04-genai-llms/02-fine-tuning/assignment.md) |
| 4.3 | RAG — Vector Search & Retrieval | [lesson](modules/04-genai-llms/03-augmented-generation/lesson.md) | [assignment](modules/04-genai-llms/03-augmented-generation/assignment.md) |

### Module 5 — MLOps & GenAI Production
| Section | Topic | Lesson | Assignment |
|---------|-------|--------|------------|
| 5.1 | Inference Serving — vLLM & TensorRT | [lesson](modules/05-mlops-production/01-model-deployment/lesson.md) | [assignment](modules/05-mlops-production/01-model-deployment/assignment.md) |
| 5.2 | AI Agents — LangChain, AutoGen, ReAct | [lesson](modules/05-mlops-production/02-ai-agents/lesson.md) | [assignment](modules/05-mlops-production/02-ai-agents/assignment.md) |

### Module 6 — Hardware, Cloud & FinOps
| Section | Topic | Lesson | Assignment |
|---------|-------|--------|------------|
| 6.1 | GPU/TPU Hardware — NVLink, HBM | [lesson](modules/06-hardware-cloud-finops/01-compute-cluster/lesson.md) | [assignment](modules/06-hardware-cloud-finops/01-compute-cluster/assignment.md) |
| 6.2 | Distributed Training — DDP, TP, PP | [lesson](modules/06-hardware-cloud-finops/02-distributed-scale/lesson.md) | [assignment](modules/06-hardware-cloud-finops/02-distributed-scale/assignment.md) |
| 6.3 | Cloud FinOps — Cost & Infrastructure | [lesson](modules/06-hardware-cloud-finops/03-cloud-finops/lesson.md) | [assignment](modules/06-hardware-cloud-finops/03-cloud-finops/assignment.md) |

---

## Repository Layout

```
FSD-to-AIMLD/
├── README.md
├── CLAUDE.md                     # AI assistant instructions
├── modules/                      # One folder per lesson node
│   ├── 01-math-foundations/
│   ├── 02-ml-core/
│   ├── 03-deep-learning/
│   ├── 04-genai-llms/
│   ├── 05-mlops-production/
│   └── 06-hardware-cloud-finops/
├── assignments/                  # Runnable code archives
├── resources/
│   ├── diagrams/                 # Visual aids
│   ├── datasets/                 # Sample data
│   └── reference-sheets/        # Cheat sheets
├── stack-comparisons/            # Python vs Java tables
├── agents/                       # Content generation agents
│   ├── orchestrator.py
│   ├── lesson_agent.py
│   ├── assignment_agent.py
│   └── prompts/
└── .course/                      # Course meta
    ├── guidelines.md
    ├── style-guide.md
    ├── content-standards.md
    └── progress.json
```

---

## Each Lesson Folder Contains

```
<lesson-folder>/
├── lesson.md          # Theory, analogies, code snippets
├── assignment.md      # Objective, rubric, extension challenge
└── starter/
    ├── starter.py     # Python starter code
    └── starter.java   # Java starter (where stack exists)
```

---

## Running Content Generation Agents

```bash
# Install dependencies
pip install anthropic

# Generate single leaf node
python agents/orchestrator.py --node "1.1.1"

# Generate all pending nodes
python agents/orchestrator.py --all

# Check progress
python agents/orchestrator.py --status
```

---

## Stack Comparison Policy

Java parallel content shown **only** where real production tooling exists:

| Domain | Python | Java |
|--------|--------|------|
| Math | NumPy, SciPy | Apache Commons Math |
| ML | scikit-learn | Tribuo, Weka |
| Deep Learning | PyTorch, TensorFlow | Deep Java Library (DJL) |
| GenAI | LangChain, Hugging Face | LangChain4j, Spring AI |
| MLOps | MLflow, BentoML | — (Python only, gap noted) |

---

## Content Standards

See [.course/guidelines.md](.course/guidelines.md) for full authoring rules.

**Quick reference:**
- Hook every lesson with 1-sentence FSD analogy
- Math → intuition → code order, always
- Code snippets must run without modification
- Max 3 key takeaways per lesson
- Assignments require starter code + expected output + extension challenge
