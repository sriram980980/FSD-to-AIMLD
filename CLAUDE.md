# FSD → AI/ML Developer Course

## Role
Educator & course author. Target: Fullstack developers transitioning to AI/ML engineering.
Dual-stack approach: Python (primary) + Java (parallel comparison) where stacks differ meaningfully.

## Audience
- Background: React/Node/Spring/SQL — strong programming foundations
- Goal: Production AI/ML engineer (not data scientist)
- Assumption: Knows loops, OOP, REST APIs. Does NOT know tensors, gradients, transformers

## Teaching Philosophy
- Explain WHY before HOW. Motivation before implementation.
- Always anchor new concept to FSD analogy (e.g., "gradient descent = optimizer tuning your learning rate like webpack tunes bundle size")
- Python first. Java parallel where ecosystem exists (Tribuo, DJL, Spring AI, LangChain4j).
- Show stack comparison tables when both ecosystems have meaningful solutions.
- Assignments are runnable, not theoretical. Real code, real data.

## Course Structure

```
Module 1 — Math Foundations
  1.1 Linear Algebra
      1.1.1 Vectors & Matrices → Core Operations
            Lesson: Matrix multiplication, dot products, eigenvalues
            Assignment: Implement NumPy vectorization scripts
      1.1.2 Calculus → Derivatives
            Lesson: Chain rule, partial derivatives, gradients
            Assignment: Code manual gradient descent calculation
  1.2 Probability & Statistics
      1.2.1 Distributions → Bayesian Stats
            Lesson: Normal distribution, Bayes theorem
            Assignment: Build SciPy statistical model

Module 2 — Machine Learning Core
  2.1 Supervised Learning
      2.1.1 Linear Models → Regression & Classification
            Lesson: Logistic regression, loss functions (MSE, Cross-Entropy)
            Assignment: Build regressor from scratch, evaluate metrics
      2.1.2 Tree-Based Models → Ensembles
            Lesson: Random Forest, XGBoost architecture
            Assignment: Train/tune classifier on tabular dataset
  2.2 Unsupervised Learning
      2.2.1 Dimensionality Reduction → PCA & Clustering
            Lesson: K-Means, Principal Component Analysis
            Assignment: Implement customer segmentation pipeline

Module 3 — Deep Learning
  3.1 Neural Networks
      3.1.1 Multi-Layer Perceptrons → Forward/Backward Pass
            Lesson: Activation functions, backpropagation math
            Assignment: Build basic PyTorch MLP
      3.1.2 Optimization → Gradient Descent Variants
            Lesson: Adam, SGD, learning rate scheduling
            Assignment: Execute hyperparameter tuning experiment
  3.2 Sequence Models
      3.2.1 Attention Mechanisms → Self-Attention
            Lesson: RNNs to Attention, context vectors
            Assignment: Code isolated attention block

Module 4 — Generative AI & LLMs
  4.1 Transformer Architecture → Encoder-Decoder
            Lesson: GPT (Decoder-only) vs BERT (Encoder-only)
            Assignment: Construct minGPT from scratch
  4.2 Fine-Tuning → PEFT → LoRA & QLoRA
            Lesson: Parameter efficient tuning math, quantization
            Assignment: Fine-tune Llama-3 on domain dataset
  4.3 Augmented Generation → RAG → Vector Search
            Lesson: Embeddings, chunking, semantic retrieval
            Assignment: Build full-stack RAG document QA API

Module 5 — MLOps & GenAI Production
  5.1 Model Deployment → Inference Serving → High-Throughput Engines
            Lesson: vLLM, continuous batching, TensorRT-LLM
            Assignment: Deploy quantized model via API endpoint
  5.2 AI Agents → Orchestration → Tool Use & Memory
            Lesson: LangChain, AutoGen, ReAct prompting
            Assignment: Build multi-agent autonomous code reviewer

Module 6 — Hardware, Cloud & FinOps
  6.1 Compute & Cluster Architecture → GPU/TPU Hardware → Interconnect & Memory
            Lesson: NVLink, InfiniBand, VRAM constraints, HBM bandwidth
            Assignment: Profile memory/network bottlenecks during distributed run
  6.2 Distributed Scale → Multi-Node Training → Parallelism Strategies
            Lesson: Data Parallelism (DDP), Tensor Parallelism (TP), Pipeline Parallelism
            Assignment: Configure Megatron-LM / DeepSpeed multi-GPU configuration
  6.3 Cloud FinOps → Resource Allocation → Cost Metrics & Infrastructure Design
            Lesson: Spot instances, serverless inference cost, GPU vs CPU compilation trade-offs
            Assignment: Build cost-optimization dashboard mapping throughput/dollar on AWS/GCP
```

## Directory Layout

```
FSD-to-AIMLD/
├── CLAUDE.md
├── modules/
│   ├── 01-math-foundations/
│   │   ├── 01-linear-algebra/
│   │   └── 02-probability-statistics/
│   ├── 02-ml-core/
│   ├── 03-deep-learning/
│   ├── 04-genai-llms/
│   ├── 05-mlops-production/
│   └── 06-hardware-cloud-finops/
├── assignments/           # Runnable code, one folder per assignment
├── resources/             # Diagrams, datasets, reference sheets
└── stack-comparisons/     # Python vs Java side-by-side tables
```

## Content Standards

### Lesson Files (lesson.md)
- Hook: 1-sentence FSD analogy
- Theory: Math → intuition → code
- Python snippet (runnable)
- Java snippet (when stack exists, with library noted)
- Stack comparison table (when both have meaningful solutions)
- Key takeaways: 3 bullets max

### Assignment Files (assignment.md + starter code)
- Clear objective
- Starter code provided (Python + Java where applicable)
- Expected output / evaluation criteria
- Extension challenge (optional hard mode)

## Stack Comparison Policy
Show Java parallel ONLY when Java ecosystem has real production tooling:
- ML/DL: Tribuo, DJL (Deep Java Library), Weka
- GenAI: LangChain4j, Spring AI
- MLOps: BentoML has no Java equiv → Python only, note gap
- Math: Apache Commons Math vs NumPy/SciPy

## Commands

### Claude Code Slash Commands
Run these from Claude Code in this project directory:

| Command | What it does |
|---------|-------------|
| `/generate-lesson 1.1.1` | Generate `lesson.md` for node 1.1.1 |
| `/generate-assignment 1.1.1` | Generate `assignment.md` + starter code for node 1.1.1 |
| `/generate-all` | Generate all nodes with status `"todo"` in `progress.json` |
| `/course-status` | Print progress table — checks files exist on disk |

Node IDs: `1.1.1`, `1.1.2`, `1.2.1`, `2.1.1`, `2.1.2`, `2.2.1`, `3.1.1`, `3.1.2`, `3.2.1`, `4.1`, `4.2`, `4.3`, `5.1`, `5.2`, `6.1`, `6.2`, `6.3`

### Python Orchestrator (Claude API)
```bash
# One-time setup
pip install -r agents/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # Windows: set ANTHROPIC_API_KEY=sk-ant-...

# Generate one node
python agents/orchestrator.py --node 1.1.1

# Generate all todo nodes
python agents/orchestrator.py --all

# Check progress
python agents/orchestrator.py --status
```

### VS Code Tasks
`Terminal → Run Task` (or `Ctrl+Shift+P → Tasks: Run Task`):
- **Course: Generate Node** — prompts for node ID, runs orchestrator
- **Course: Generate ALL Nodes** — runs all todo nodes
- **Course: Status** — prints progress table
- **Course: Install Agent Dependencies** — `pip install -r agents/requirements.txt`

Requires `ANTHROPIC_API_KEY` in your shell environment before opening VS Code.

### GitHub Copilot
`.github/copilot-instructions.md` is loaded automatically in this workspace.
Copilot Chat follows course style rules for all content generation.
Use Copilot Chat to generate individual sections, review drafts, or suggest FSD analogies.
Example prompts:
- `Generate lesson.md for node 3.1.1 following the course guidelines`
- `Write starter.py for the attention mechanism assignment`
- `Suggest a better FSD analogy for backpropagation`
