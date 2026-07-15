#!/usr/bin/env python3
"""
Course content orchestrator.
Reads course_map.json, dispatches sub-agents (via Claude API) to generate
lesson.md, assignment.md, and starter code for each leaf node.

Usage:
    python agents/orchestrator.py --node 1.1.1
    python agents/orchestrator.py --all
    python agents/orchestrator.py --status
"""

# Dependencies: anthropic>=0.27, rich>=13.0
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import anthropic
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

ROOT = Path(__file__).parent.parent
COURSE_MAP = ROOT / ".course" / "course_map.json"
PROGRESS_FILE = ROOT / ".course" / "progress.json"
GUIDELINES = ROOT / ".course" / "guidelines.md"
STYLE_GUIDE = ROOT / ".course" / "style-guide.md"

console = Console()
NodeStatus = Literal["todo", "draft", "reviewed", "done"]


def load_course_map() -> dict:
    return json.loads(COURSE_MAP.read_text())


def load_progress() -> dict:
    return json.loads(PROGRESS_FILE.read_text())


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def load_system_context() -> str:
    guidelines = GUIDELINES.read_text()
    style = STYLE_GUIDE.read_text()
    return f"""You are an expert AI/ML educator writing course content for Fullstack developers transitioning to AI/ML engineering.

GUIDELINES:
{guidelines}

STYLE GUIDE:
{style}

CRITICAL RULES:
- All code must run without modification
- Every concept needs a FSD analogy before ML explanation
- Python snippets always. Java snippets only when java_applicable=true in node metadata.
- Exactly 3 key takeaways per lesson
- Starter code must compile/run as-is with placeholder TODOs for student to fill
"""


def dispatch_lesson_agent(client: anthropic.Anthropic, node: dict, system: str) -> str:
    """Call Claude to generate lesson.md content for a node."""
    prompt_path = Path(__file__).parent / "prompts" / "lesson_prompt.md"
    template = prompt_path.read_text()

    user_prompt = template.format(
        node_id=node["id"],
        title=node["title"],
        lesson_topic=node["lesson_topic"],
        fsd_analogy=node["fsd_analogy"],
        python_libs=", ".join(node["python_libs"]),
        java_libs=", ".join(node.get("java_libs", [])),
        java_applicable=node.get("java_applicable", False),
        java_note=node.get("java_note", ""),
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def dispatch_assignment_agent(client: anthropic.Anthropic, node: dict, system: str) -> str:
    """Call Claude to generate assignment.md content for a node."""
    prompt_path = Path(__file__).parent / "prompts" / "assignment_prompt.md"
    template = prompt_path.read_text()

    user_prompt = template.format(
        node_id=node["id"],
        title=node["title"],
        assignment_title=node["assignment_title"],
        lesson_topic=node["lesson_topic"],
        python_libs=", ".join(node["python_libs"]),
        java_applicable=node.get("java_applicable", False),
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=3000,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def dispatch_starter_agent(client: anthropic.Anthropic, node: dict, system: str) -> tuple[str, str | None]:
    """Call Claude to generate starter.py (and optionally starter.java)."""
    prompt_path = Path(__file__).parent / "prompts" / "starter_prompt.md"
    template = prompt_path.read_text()

    user_prompt = template.format(
        node_id=node["id"],
        title=node["title"],
        assignment_title=node["assignment_title"],
        python_libs=", ".join(node["python_libs"]),
        java_libs=", ".join(node.get("java_libs", [])),
        java_applicable=node.get("java_applicable", False),
    )

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text
    py_code, java_code = parse_starter_response(raw)
    return py_code, java_code


def parse_starter_response(raw: str) -> tuple[str, str | None]:
    """Extract python and java blocks from agent response."""
    import re
    py_match = re.search(r"```python\n(.*?)```", raw, re.DOTALL)
    java_match = re.search(r"```java\n(.*?)```", raw, re.DOTALL)
    py_code = py_match.group(1) if py_match else "# TODO: starter code generation failed\n"
    java_code = java_match.group(1) if java_match else None
    return py_code, java_code


def generate_node(client: anthropic.Anthropic, node: dict, system: str, progress: dict) -> None:
    """Run all three sub-agents for one leaf node and write files."""
    node_path = ROOT / node["path"]
    starter_path = node_path / "starter"
    starter_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Node {node['id']}[/bold cyan] — {node['title']}")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress_bar:
        task = progress_bar.add_task("Generating lesson.md...", total=None)
        lesson_content = dispatch_lesson_agent(client, node, system)
        (node_path / "lesson.md").write_text(lesson_content)
        progress_bar.update(task, description="lesson.md done")

        progress_bar.update(task, description="Generating assignment.md...")
        assignment_content = dispatch_assignment_agent(client, node, system)
        (node_path / "assignment.md").write_text(assignment_content)
        progress_bar.update(task, description="assignment.md done")

        progress_bar.update(task, description="Generating starter code...")
        py_code, java_code = dispatch_starter_agent(client, node, system)
        (starter_path / "starter.py").write_text(py_code)
        if java_code and node.get("java_applicable"):
            (starter_path / "starter.java").write_text(java_code)
        progress_bar.update(task, description="starter code done")

    progress["nodes"][node["id"]] = "draft"
    save_progress(progress)
    console.print(f"[green]✓[/green] {node['id']} → draft")


def cmd_status(course_map: dict, progress: dict) -> None:
    table = Table(title="Course Progress")
    table.add_column("ID", style="cyan")
    table.add_column("Title")
    table.add_column("Status", style="bold")

    status_colors = {"todo": "red", "draft": "yellow", "reviewed": "blue", "done": "green"}

    for node in course_map["nodes"]:
        status = progress["nodes"].get(node["id"], "todo")
        color = status_colors.get(status, "white")
        table.add_row(node["id"], node["title"], f"[{color}]{status}[/{color}]")

    console.print(table)

    counts = {}
    for v in progress["nodes"].values():
        counts[v] = counts.get(v, 0) + 1
    total = len(course_map["nodes"])
    done = counts.get("done", 0) + counts.get("reviewed", 0)
    console.print(f"\n[bold]Progress:[/bold] {done}/{total} complete")


def main() -> None:
    parser = argparse.ArgumentParser(description="Course content orchestrator")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--node", metavar="ID", help="Generate content for one node (e.g. 1.1.1)")
    group.add_argument("--all", action="store_true", help="Generate all todo nodes")
    group.add_argument("--status", action="store_true", help="Show progress table")
    args = parser.parse_args()

    course_map = load_course_map()
    progress = load_progress()

    if args.status:
        cmd_status(course_map, progress)
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error:[/red] ANTHROPIC_API_KEY not set")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    system = load_system_context()

    nodes_by_id = {n["id"]: n for n in course_map["nodes"]}

    if args.node:
        if args.node not in nodes_by_id:
            console.print(f"[red]Error:[/red] Node {args.node!r} not in course map")
            sys.exit(1)
        generate_node(client, nodes_by_id[args.node], system, progress)

    elif args.all:
        todo_nodes = [
            n for n in course_map["nodes"]
            if progress["nodes"].get(n["id"], "todo") == "todo"
        ]
        if not todo_nodes:
            console.print("[green]All nodes already generated.[/green]")
            return
        console.print(f"Generating {len(todo_nodes)} todo nodes...")
        for node in todo_nodes:
            generate_node(client, node, system, progress)
        console.print("\n[bold green]All done.[/bold green]")


if __name__ == "__main__":
    main()
