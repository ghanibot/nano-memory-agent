from __future__ import annotations

import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from .config.schema import MemoryConfig, EmbedderConfig, StoreConfig, BudgetConfig
from .core.memory import Memory, load_config
from .store.schema import MemoryType

app = typer.Typer(
    name="nano-mem",
    help="nano-memory — persistent memory layer for AI agents.",
    add_completion=False,
)
console = Console()

_CONFIG_ENV = "NANO_MEMORY_CONFIG"


def _get_memory(
    namespace: str | None,
    config: str | None,
    store: str | None,
) -> Memory:
    cfg_path = config or os.environ.get(_CONFIG_ENV)
    if cfg_path:
        mem = Memory(cfg_path)
    else:
        mem = Memory(
            MemoryConfig(
                namespace=namespace or "default",
                store=StoreConfig(path=store or "~/.nano-memory"),
            )
        )
    if namespace:
        mem.switch_namespace(namespace)
    return mem


@app.command("save")
def cmd_save(
    text: str = typer.Argument(..., help="Text to save as a memory."),
    type: MemoryType = typer.Option("fact", "--type", "-t", help="Memory type."),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to YAML config."),
    store: Optional[str] = typer.Option(None, "--store", help="Override store directory."),
    metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="JSON metadata string."),
) -> None:
    """Save a new memory."""
    mem = _get_memory(namespace, config, store)
    meta: dict = json.loads(metadata) if metadata else {}
    result = mem.save(text, type=type, metadata=meta)
    if isinstance(result, list):
        console.print(f"[green]Saved {len(result)} chunks[/green]")
        for rid in result:
            console.print(f"  {rid}")
    else:
        console.print(f"[green]Saved[/green] id=[bold]{result}[/bold]")


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Search query."),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    type_filter: Optional[MemoryType] = typer.Option(None, "--type", "-t"),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    store: Optional[str] = typer.Option(None, "--store"),
    cross_namespace: bool = typer.Option(False, "--cross-namespace", "-x"),
) -> None:
    """Search memories by semantic similarity."""
    mem = _get_memory(namespace, config, store)
    records = mem.search(query, top_k=top_k, type_filter=type_filter, cross_namespace=cross_namespace)

    if not records:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Score", style="cyan", width=7)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Namespace", style="blue", width=14)
    table.add_column("Text", style="white")
    table.add_column("ID", style="dim", width=34)

    for r in records:
        table.add_row(
            f"{r.score:.3f}",
            r.type,
            r.namespace,
            r.text[:200] + ("…" if len(r.text) > 200 else ""),
            r.id,
        )

    console.print(table)


@app.command("list")
def cmd_list(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    type_filter: Optional[MemoryType] = typer.Option(None, "--type", "-t"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    store: Optional[str] = typer.Option(None, "--store"),
) -> None:
    """List all memories in the namespace."""
    mem = _get_memory(namespace, config, store)
    records = mem.list(type_filter=type_filter)

    if not records:
        console.print("[yellow]No memories found.[/yellow]")
        return

    table = Table(box=box.ROUNDED, show_lines=True)
    table.add_column("Type", style="magenta", width=12)
    table.add_column("Text", style="white")
    table.add_column("Created", style="dim", width=28)
    table.add_column("ID", style="dim", width=34)

    for r in records:
        table.add_row(
            r.type,
            r.text[:200] + ("…" if len(r.text) > 200 else ""),
            r.created_at,
            r.id,
        )

    console.print(f"[bold]{len(records)} memor{'y' if len(records) == 1 else 'ies'}[/bold] in namespace [cyan]{mem.namespace!r}[/cyan]")
    console.print(table)


@app.command("forget")
def cmd_forget(
    record_id: str = typer.Argument(..., help="Memory ID to delete."),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    store: Optional[str] = typer.Option(None, "--store"),
) -> None:
    """Delete a memory by ID."""
    mem = _get_memory(namespace, config, store)
    deleted = mem.forget(record_id)
    if deleted:
        console.print(f"[green]Deleted[/green] {record_id}")
    else:
        console.print(f"[red]Not found:[/red] {record_id}")
        raise typer.Exit(code=1)


@app.command("clear")
def cmd_clear(
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    store: Optional[str] = typer.Option(None, "--store"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete ALL memories in the namespace."""
    mem = _get_memory(namespace, config, store)
    if not yes:
        confirmed = typer.confirm(
            f"Delete all memories in namespace {mem.namespace!r}?", default=False
        )
        if not confirmed:
            raise typer.Abort()
    count = mem.clear()
    console.print(f"[green]Cleared[/green] {count} memor{'y' if count == 1 else 'ies'} from [cyan]{mem.namespace!r}[/cyan]")


@app.command("cost")
def cmd_cost(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    store: Optional[str] = typer.Option(None, "--store"),
) -> None:
    """Show embedding cost report."""
    mem = _get_memory(None, config, store)
    report = mem.cost_report()

    console.print(f"\n[bold]Budget:[/bold] ${report['budget_usd']:.2f}  |  "
                  f"[bold]Remaining:[/bold] ${report['budget_remaining_usd']:.4f}")

    for scope in ("session", "global"):
        rows = report[scope]
        total = report[f"{scope}_total_usd"]
        console.print(f"\n[bold]{scope.title()} usage[/bold] (total: ${total:.6f})")
        if not rows:
            console.print("  [dim]No usage recorded.[/dim]")
            continue
        table = Table(box=box.SIMPLE)
        table.add_column("Model")
        table.add_column("Calls", justify="right")
        table.add_column("Tokens in", justify="right")
        table.add_column("Cost USD", justify="right")
        for model, stats in rows.items():
            table.add_row(
                model,
                str(stats["calls"]),
                str(stats["tokens_in"]),
                f"${stats['cost_usd']:.6f}",
            )
        console.print(table)


@app.command("export")
def cmd_export(
    path: str = typer.Argument(..., help="Output JSON file path."),
    namespace: Optional[str] = typer.Option(None, "--namespace", "-n"),
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    store: Optional[str] = typer.Option(None, "--store"),
) -> None:
    """Export all memories in the namespace to a JSON file."""
    mem = _get_memory(namespace, config, store)
    mem.export(path)
    console.print(f"[green]Exported[/green] namespace [cyan]{mem.namespace!r}[/cyan] to [bold]{path}[/bold]")


if __name__ == "__main__":
    app()
