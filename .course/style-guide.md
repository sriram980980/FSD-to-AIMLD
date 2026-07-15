# Style Guide

## Markdown Conventions

- H1 (`#`) — lesson/assignment title only
- H2 (`##`) — major sections (Hook, Theory, Implementation)
- H3 (`###`) — subsections within a major section
- No H4+ — flatten structure instead
- Blank line between every section heading and content
- Code blocks always have language tag: ` ```python `, ` ```java `, ` ```bash `

## File Naming

- All lowercase, hyphen-separated: `lesson.md`, `assignment.md`, `starter.py`
- Folder names: `01-vectors-matrices`, `02-calculus-derivatives` (zero-padded index + slug)
- No spaces anywhere in paths

## Code Style

### Python
```python
# Dependencies: numpy>=1.24, scipy>=1.11
import numpy as np
from scipy import stats
from typing import Tuple

def compute_gradient(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Single docline only. No multi-paragraph blocks."""
    predictions = X @ weights
    error = predictions - y
    return (2 / len(y)) * X.T @ error
```

Rules:
- `ruff` formatting (88 char line limit)
- Type hints on all public functions
- Single-line docstring only (no multi-paragraph)
- `f-string` for all interpolation
- No `%` formatting, no `.format()`

### Java
```java
// Dependencies: djl-api:0.26, tribuo:4.3
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

public record GradientResult(float[] weights, float loss) {}

public GradientResult computeGradient(float[] weights, float[][] X, float[] y) {
    // implementation
}
```

Rules:
- Java 17+ records for data containers
- Full import paths shown
- No Lombok (adds complexity for learners)
- `var` acceptable for obvious types in assignment code

## Terminology Consistency

| Use | Never use |
|-----|-----------|
| `weights` | `parameters`, `thetas`, `w` |
| `learning rate` | `lr`, `alpha`, `step size` |
| `loss` | `cost`, `objective`, `error` (when used as the optimization target) |
| `batch` | `mini-batch` (unless distinguishing from full-batch) |
| `epoch` | `pass`, `iteration` (epoch = full dataset pass) |
| `embedding` | `vector representation`, `dense vector` |
| `token` | `word piece` (unless discussing WordPiece specifically) |
| `fine-tune` | `retrain`, `adapt` |
| `inference` | `prediction time`, `serving time` |

## Tone

- Peer-to-peer, not professor-to-student
- No "obviously", "simply", "trivially", "just"
- No rhetorical questions unless immediately answered
- Imperative for instructions: "Run the script", "Notice the output"
- Past tense for results: "The model converged after 50 epochs"

## Diagrams

- Save as SVG in `resources/diagrams/`
- Name: `[module]-[topic]-[description].svg` e.g. `03-attention-qkv-flow.svg`
- Reference from lesson with relative path
- Alt text required: `![Attention QKV flow showing Q, K, V projections](../../../resources/diagrams/03-attention-qkv-flow.svg)`

## Reference Sheets

- One sheet per major concept cluster
- Save in `resources/reference-sheets/`
- Format: table-heavy, formula-reference, no prose
- Name: `[topic]-cheatsheet.md`
