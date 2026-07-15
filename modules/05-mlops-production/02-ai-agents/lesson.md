# 5.2 — AI Agents — Orchestration, Tool Use & Memory

## Hook

ReAct agents are a state machine with an LLM as the transition function — the Reason → Act → Observe loop mirrors a CI pipeline: plan the step, run the tool, check the output, then decide what comes next.

## The Problem

A static LLM prompt can answer questions from its training data, but it breaks the moment a task requires real-world data, external computation, or multi-step decision-making. Without an agent loop every dynamic workflow becomes hand-rolled glue code: you hard-code the sequence of tool calls, lose the ability to course-correct mid-task, and must re-engineer the entire pipeline when requirements change. Agent frameworks replace that glue code with a principled loop that reasons about *when* to invoke tools and *how* to interpret their outputs before committing to the next action.

## Theory

### The ReAct Loop

ReAct (**Re**ason + **Act**) is the dominant agent prompting strategy. At each step $t$ the LLM receives a growing context $C_t$ and produces a thought $T_t$ followed by either a tool call $a_t$ or a final answer:

$$C_t = \bigl[q,\; T_1, a_1, o_1,\; T_2, a_2, o_2,\; \ldots,\; T_{t-1}, a_{t-1}, o_{t-1}\bigr]$$

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| $q$ | The user's original question — fixed for the entire episode |
| $T_i$ | The reasoning *thought* at step $i$ — plain-language text the LLM generates before acting |
| $a_i$ | The *action* at step $i$ — a structured tool call, e.g. `calculator("0.15 * 240")` |
| $o_i$ | The *observation* at step $i$ — the tool's return value, appended verbatim to the context |
| $C_t$ | The full conversation context fed to the LLM at step $t$; grows by three items per step |

The agent terminates when the LLM emits `Final Answer:` instead of another action, or when a maximum step limit is reached.

**Context growth formula:**

$$|C_t| = 1 + 3(t - 1)$$

At step $t = 10$ the context already holds $1 + 27 = 28$ items. This linear growth is why context-window management is critical for long-running agents.

**Numeric worked example:**

Question: *"What is 15% of 240 plus the square root of 144?"*

| Step $t$ | Thought $T_t$ | Action $a_t$ | Observation $o_t$ |
|----------|--------------|-------------|------------------|
| 1 | "I need 15% of 240 first." | `calculator("0.15 * 240")` | `36.0` |
| 2 | "Now I need √144." | `calculator("sqrt(144)")` | `12.0` |
| 3 | "36.0 + 12.0 = 48.0. Done." | `Final Answer: 48.0` | — |

Context size at termination: $|C_3| = 1 + 3(3-1) = 7$ items. The arithmetic check: $0.15 \times 240 = 36.0$, $\sqrt{144} = 12.0$, $36.0 + 12.0 = 48.0$.

### Tool Registration & Dispatch

Tools are named callables with a typed signature and a description the LLM uses to decide *when* to call them. The tool-selection decision at each step is:

$$\hat{a}_t = \arg\max_{a \in \mathcal{A}} \; P_\theta\!\left(a \mid C_t\right)$$

**Symbol definitions:**

| Symbol | Meaning |
|--------|---------|
| $\mathcal{A}$ | The finite set of available tools plus the `Final Answer` action |
| $P_\theta(a \mid C_t)$ | The LLM's probability distribution over actions given the current context, with weights $\theta$ |
| $\hat{a}_t$ | The highest-probability action, selected greedily |

The LLM emits the tool name and input as structured text; a regex parser extracts both and dispatches the call to the matching function in the tool registry.

### Memory Architecture

Agents maintain three memory tiers, each with a direct FSD equivalent:

| Memory Type | Agent Role | FSD Equivalent |
|-------------|-----------|----------------|
| **Short-term** | Conversation buffer — recent messages in the current session | In-memory session store (Redis with TTL) |
| **Long-term** | Persistent facts recalled across sessions | PostgreSQL user-preferences table |
| **Episodic** | Past task outcomes used for self-reflection | Application audit log / structured run history |

Context-window limits bound short-term memory. When the buffer overflows two strategies apply: **summarization** (compress old messages into a single summary node) or **sliding-window eviction** (drop the oldest turn). Most production agents combine both.

### Multi-Agent Orchestration (AutoGen Pattern)

A single agent handles one specialization well, but complex tasks benefit from a *group chat* where agents with different system prompts collaborate. AutoGen's model assigns three roles:

1. **Planner** — decomposes the user task into sub-tasks and sets acceptance criteria
2. **Executor** — runs code or tool calls and returns raw outputs
3. **Critic** — validates outputs against criteria and requests revisions

Each agent is an independent ReAct loop. A **GroupChat** router decides which agent speaks next — the same role a message broker plays when routing events to specialized microservices. The conversation terminates when the Critic signals acceptance or a round limit is reached.

## Python Implementation

```python
# Dependencies: langchain>=0.2, langchain-community>=0.2, anthropic>=0.27
import math
import json
import re
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a safe math expression using Python's math module."""
    allowed_names: dict[str, Any] = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "pi": math.pi,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def string_length(text: str) -> str:
    """Return the character count of a string."""
    return str(len(text))


TOOLS: dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "string_length": string_length,
}

TOOL_DESCRIPTIONS = """\
Available tools:
  calculator(expression)  — Evaluates a math expression. Example: calculator("sqrt(144)")
  string_length(text)     — Returns character count.    Example: string_length("hello")
"""


# ---------------------------------------------------------------------------
# Mock LLM — scripted responses so this lesson runs without an API key
# ---------------------------------------------------------------------------

class MockReActLLM:
    """Deterministic LLM that replays a scripted ReAct dialogue."""

    def __init__(self) -> None:
        self._step: int = 0
        self._script: list[str] = [
            (
                "Thought: I need to calculate 15% of 240 first.\n"
                "Action: calculator\n"
                'Action Input: "0.15 * 240"'
            ),
            (
                "Thought: Now I need the square root of 144.\n"
                "Action: calculator\n"
                'Action Input: "sqrt(144)"'
            ),
            "Thought: 36.0 + 12.0 = 48.0. I have both values.\nFinal Answer: 48.0",
        ]

    def generate(self, prompt: str) -> str:  # noqa: ARG002
        """Return the next scripted agent step."""
        response = self._script[min(self._step, len(self._script) - 1)]
        self._step += 1
        return response


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

class ReActAgent:
    """Minimal Reason+Act agent loop."""

    def __init__(
        self,
        llm: MockReActLLM,
        tools: dict[str, Callable[[str], str]],
        max_steps: int = 6,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: list[dict[str, str]] = []

    def _parse_action(self, response: str) -> tuple[str | None, str | None]:
        """Extract (tool_name, tool_input) from an LLM response string."""
        action_match = re.search(r"Action:\s*(\w+)", response)
        input_match = re.search(r'Action Input:\s*"?([^"\n]+)"?', response)
        if action_match and input_match:
            return action_match.group(1).strip(), input_match.group(1).strip()
        return None, None

    def run(self, question: str) -> str:
        """Execute the ReAct loop until Final Answer or max_steps reached."""
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"Question: {question}")
        print(separator)

        context = f"Question: {question}\n\n{TOOL_DESCRIPTIONS}\n"

        for step in range(1, self.max_steps + 1):
            response = self.llm.generate(context)
            print(f"\n--- Step {step} ---")
            print(response)
            self.history.append({"step": str(step), "agent_output": response})

            if "Final Answer:" in response:
                final_answer = response.split("Final Answer:")[-1].strip()
                print(f"\n{separator}")
                print(f"Final Answer: {final_answer}")
                print(f"Steps used:   {step} / {self.max_steps}")
                return final_answer

            tool_name, tool_input = self._parse_action(response)
            if tool_name and tool_input and tool_name in self.tools:
                observation = self.tools[tool_name](tool_input)
                print(f"Observation: {observation}")
                self.history.append({"step": str(step), "observation": observation})
                context += f"\n{response}\nObservation: {observation}\n"
            else:
                print("Warning: No valid action parsed — halting.")
                break

        return "Max steps reached without a final answer."


# ---------------------------------------------------------------------------
# Memory demonstration
# ---------------------------------------------------------------------------

def demonstrate_memory_types() -> None:
    """Print examples of the three agent memory types."""
    separator = "=" * 60
    print(f"\n{separator}")
    print("Memory Types in Agent Systems")
    print(separator)

    # Short-term: sliding conversation buffer
    short_term: list[dict[str, str]] = [
        {"role": "user",      "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user",      "content": "Double that."},
    ]
    print(f"\nShort-term memory ({len(short_term)} messages in buffer):")
    for msg in short_term:
        print(f"  [{msg['role']:>9}]: {msg['content']}")

    # Long-term: persistent key-value store
    long_term: dict[str, Any] = {
        "user_preference_units": "metric",
        "last_successful_tool": "calculator",
        "session_count": 3,
    }
    print("\nLong-term memory (persisted across sessions):")
    print(json.dumps(long_term, indent=2))

    # Episodic: past task outcomes for reflection
    episodic: list[dict[str, str]] = [
        {"task": "Calculate compound interest", "outcome": "success",    "steps": "3"},
        {"task": "Fetch live stock price",      "outcome": "tool_error", "steps": "2"},
    ]
    print(f"\nEpisodic memory ({len(episodic)} past episodes):")
    for ep in episodic:
        print(f"  {ep['task']:<35} → {ep['outcome']:<12} ({ep['steps']} steps)")


# ---------------------------------------------------------------------------
# Multi-agent demonstration (AutoGen pattern)
# ---------------------------------------------------------------------------

def demonstrate_multi_agent() -> None:
    """Print a conceptual multi-agent group chat turn sequence."""
    separator = "=" * 60
    print(f"\n{separator}")
    print("Multi-Agent GroupChat (AutoGen Pattern)")
    print(separator)

    task = "Write and verify a Python function that computes fibonacci(n)."
    scripted_turns: list[tuple[str, str]] = [
        ("Planner",  "Sub-tasks: (1) write fib function, (2) add edge-case tests."),
        ("Coder",    "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"),
        ("Critic",   "Recursion is O(2^n). Request iterative O(n) version for large n."),
        ("Coder",    "def fib(n):\n              a, b = 0, 1\n              for _ in range(n): a, b = b, a + b\n              return a"),
        ("Critic",   "Iterative O(n), constant space. Accepted."),
    ]

    print(f"\nTask: {task}\n")
    for speaker, message in scripted_turns:
        first_line = message.split("\n")[0]
        print(f"  [{speaker:<7}]: {first_line}")

    print(f"\nResult: Task completed in {len(scripted_turns)} agent turns.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    llm = MockReActLLM()
    agent = ReActAgent(llm=llm, tools=TOOLS, max_steps=6)
    agent.run("What is 15% of 240 plus the square root of 144?")
    demonstrate_memory_types()
    demonstrate_multi_agent()


if __name__ == "__main__":
    main()
```

Run the script and notice the following. The agent prints a structured Thought → Action → Observation trace for each step, making the reasoning process fully auditable — unlike a black-box API call where you only see the final answer. At step 3 the agent emits `Final Answer: 48.0` and terminates early (3 of 6 allowed steps), showing that `max_steps` is a safety ceiling, not a fixed iteration count. The memory section demonstrates that short-term context is just a Python list — the entire "memory" mechanism is token concatenation. The multi-agent section shows that specialization (Planner / Coder / Critic) maps directly to the microservice separation-of-concerns principle you already apply in FSD work.

## Java Implementation

```java
// Dependencies (Maven — add to pom.xml):
//   <dependency>
//     <groupId>dev.langchain4j</groupId>
//     <artifactId>langchain4j</artifactId>
//     <version>0.31.0</version>
//   </dependency>
//   <dependency>
//     <groupId>dev.langchain4j</groupId>
//     <artifactId>langchain4j-open-ai</artifactId>
//     <version>0.31.0</version>
//   </dependency>
//
// Run: export OPENAI_API_KEY=sk-...   (or any OpenAI-compatible endpoint)
// Java 17+

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;

public class ReActAgentDemo {

    // ---------------------------------------------------------------
    // Tool definitions
    //
    // LangChain4j scans @Tool-annotated methods at startup and
    // registers each one in the agent's action space — no manual
    // routing code required.
    // ---------------------------------------------------------------

    static class MathTools {

        @Tool("Evaluates a mathematical expression and returns the result as a double. "
            + "Example: evaluate(\"0.15 * 240\")")
        public double evaluate(String expression) {
            // Production code should use a safe expression parser (e.g. exp4j).
            // This switch covers the lesson's worked example deterministically.
            return switch (expression.trim()) {
                case "0.15 * 240" -> 0.15 * 240.0;
                case "sqrt(144)"  -> Math.sqrt(144.0);
                default -> {
                    try {
                        yield Double.parseDouble(expression.trim());
                    } catch (NumberFormatException e) {
                        yield Double.NaN;
                    }
                }
            };
        }

        @Tool("Returns the character count of the provided string. "
            + "Example: stringLength(\"hello\") → 5")
        public int stringLength(String text) {
            return text.length();
        }
    }

    // ---------------------------------------------------------------
    // Agent interface
    //
    // LangChain4j generates the full ReAct loop at runtime from this
    // interface — no hand-written orchestration loop needed.
    // ---------------------------------------------------------------

    interface MathAgent {
        String answer(String question);
    }

    // ---------------------------------------------------------------
    // Entry point
    // ---------------------------------------------------------------

    public static void main(String[] args) {
        String apiKey = System.getenv("OPENAI_API_KEY");
        if (apiKey == null || apiKey.isBlank()) {
            System.err.println("Error: OPENAI_API_KEY environment variable is not set.");
            System.err.println("Export it before running: export OPENAI_API_KEY=sk-...");
            System.exit(1);
        }

        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(apiKey)
                .modelName("gpt-4o-mini")
                .build();

        // AiServices wires the MathAgent interface to the model + tools + memory.
        // The ReAct loop (Thought → Action → Observation) runs inside the proxy.
        MathAgent agent = AiServices.builder(MathAgent.class)
                .chatLanguageModel(model)
                .tools(new MathTools())
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        String question = "What is 15% of 240 plus the square root of 144?";
        System.out.println("Question:     " + question);

        String answer = agent.answer(question);
        System.out.println("Final Answer: " + answer);
    }
}
```

The key design decision in LangChain4j is that you define tools as annotated Java methods on a plain class — the library discovers them via reflection and formats their signatures into the system prompt automatically. `AiServices.builder()` acts as the dependency-injection wiring point: swap `OpenAiChatModel` for any `ChatLanguageModel` implementation (Anthropic, Mistral, local Ollama) without changing the agent interface or tool code.

## Stack Comparison

| Dimension | Python (LangChain) | Java (LangChain4j) |
|-----------|--------------------|--------------------|
| Tool definition | `@tool` decorator or `StructuredTool` class | `@Tool` annotation on a plain method |
| Agent executor | `AgentExecutor` / LCEL `.invoke()` | `AiServices.builder()` proxy |
| Memory | `ConversationBufferMemory`, `ConversationSummaryMemory` | `MessageWindowChatMemory`, `TokenWindowChatMemory` |
| Streaming | `.stream()` on any LCEL chain | `StreamingChatLanguageModel` interface |
| Async execution | `ainvoke()` coroutine | `CompletableFuture<String>` |
| Multi-agent | LangGraph, AutoGen (separate library) | `AiServices` orchestration patterns (built-in) |
| Ecosystem breadth | 100+ integrations (vectorstores, loaders, tools) | Growing — covers major LLM providers and vector DBs |
| Production maturity | Stable since 2023 | Stable since 0.27 (2024) |

## Key Takeaways

- The ReAct loop — Reason, Act, Observe — is a finite state machine where every transition grows the context by exactly three items (thought, action, observation); context-window cost is $O(t)$ in the number of steps.
- Tool use decouples the LLM from real-world state: the model reasons about *what* to compute, the tool registry handles *how* to compute it — the same separation-of-concerns you apply when splitting API routes from business logic.
- Agent memory has three distinct tiers (short-term buffer, long-term store, episodic history), each requiring a different eviction strategy; conflating them is the most common source of agents that drift or repeat work across sessions.
