# main.py — Beautiful Rich CLI for RAG Bakery System
"""
Command-line interface with 4 modes:
    --ingest      : Run the full PDF ingestion pipeline
    --query "..." : Single query, print formatted results
    --interactive : Loop mode, ask questions until Ctrl+C
    --demo        : Run the official challenge example question

Uses the 'rich' library for beautiful, colored terminal output.
"""

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Configure logging BEFORE importing any src modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)

# ──────────────────────────────────────────────────────────────
# Rich Console
# ──────────────────────────────────────────────────────────────

console = Console()

# ──────────────────────────────────────────────────────────────
# Challenge Example Question (official from jury)
# ──────────────────────────────────────────────────────────────

DEMO_QUESTION = (
    "Améliorant de panification : quelles sont les quantités recommandées "
    "d'alpha-amylase, xylanase et d'Acide ascorbique ?"
)

# ──────────────────────────────────────────────────────────────
# Score Bar Generator
# ──────────────────────────────────────────────────────────────

def score_bar(score: float, width: int = 20) -> str:
    """
    Generate a visual score bar using block characters.

    Example: score=0.85, width=20 → '█████████████████░░░'

    Args:
        score: Similarity score between 0 and 1.
        width: Total width of the bar in characters.

    Returns:
        String of filled (█) and empty (░) blocks.
    """
    filled = int(round(score * width))
    empty = width - filled
    return '█' * filled + '░' * empty


# ──────────────────────────────────────────────────────────────
# Display Results — Rich Table + Challenge Format
# ──────────────────────────────────────────────────────────────

def display_results(question: str, results: list):
    """
    Display search results in both Rich table AND challenge format.

    Shows:
        - Question in a blue panel
        - Results in a rounded table with colored columns
        - Score bar visualization
        - Fragment truncated at 300 chars
        - ALSO prints the exact challenge-format output

    Args:
        question: The user's search question.
        results:  List of result dicts from semantic_search().
    """
    # Question panel
    console.print()
    console.print(Panel(
        f"[bold white]{question}[/bold white]",
        title="🔍 Question",
        border_style="blue",
        padding=(1, 2)
    ))

    if not results:
        console.print(
            Panel(
                "[bold red]Aucun résultat trouvé.[/bold red]",
                border_style="red",
                padding=(1, 2)
            )
        )
        return

    # Results table (Rich visual)
    table = Table(
        title="📊 Résultats de recherche sémantique",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        padding=(0, 1),
        expand=True
    )

    table.add_column("Rank", style="cyan", justify="center", width=6)
    table.add_column("Score", style="green", justify="center", width=8)
    table.add_column("Score Bar", style="green", width=22)
    table.add_column("Fragment", style="white", ratio=1)
    table.add_column("Doc ID", style="yellow", justify="center", width=8)

    for result in results:
        # Truncate fragment at 300 chars
        fragment = result['fragment']
        if len(fragment) > 300:
            fragment = fragment[:297] + "..."

        table.add_row(
            f"#{result['rank']}",
            f"{result['score']:.4f}",
            score_bar(result['score']),
            fragment,
            str(result['doc_id'])
        )

    console.print(table)
    console.print()

    # ALSO print exact challenge format output
    console.print(Panel(
        "[bold green]Format de sortie officiel (challenge)[/bold green]",
        border_style="green",
        padding=(0, 2)
    ))
    from src.search import format_results
    console.print(format_results(results))
    console.print()


# ──────────────────────────────────────────────────────────────
# Mode Handlers
# ──────────────────────────────────────────────────────────────

def run_ingest():
    """Run the full ingestion pipeline."""
    from src.ingest import ingest_all_pdfs

    console.print(Panel(
        "[bold green]Starting PDF Ingestion Pipeline[/bold green]\n"
        "This will extract, chunk, embed, and store all PDFs.",
        title="🏭 RAG Bakery — Ingestion",
        border_style="green",
        padding=(1, 2)
    ))

    result = ingest_all_pdfs()

    if result:
        console.print(Panel(
            f"[bold green]Pipeline complete![/bold green]\n\n"
            f"  ✓ Processed: {result['processed']}\n"
            f"  ⊘ Skipped:   {result['skipped']}\n"
            f"  ✗ Failed:    {result['failed']}\n"
            f"  ◉ Chunks:    {result['total_chunks']}",
            title="📋 Summary",
            border_style="green",
            padding=(1, 2)
        ))


def run_query(question: str):
    """Run a single search query."""
    from src.search import semantic_search
    results = semantic_search(question)
    display_results(question, results)


def run_demo():
    """
    Run the official challenge example question.
    Shows the exact output format the jury expects.
    """
    console.print(Panel(
        "[bold yellow]Running official challenge demo question[/bold yellow]\n"
        "This is the exact example from the challenge specification.",
        title="🎯 Challenge Demo",
        border_style="yellow",
        padding=(1, 2)
    ))
    run_query(DEMO_QUESTION)


def run_interactive():
    """Run interactive search mode — loop until Ctrl+C."""
    from src.search import semantic_search

    console.print(Panel(
        "[bold cyan]Interactive Search Mode[/bold cyan]\n"
        "Type your questions in natural language (French or English).\n"
        "Press [bold]Ctrl+C[/bold] to exit.",
        title="🔄 RAG Bakery — Interactive",
        border_style="cyan",
        padding=(1, 2)
    ))

    while True:
        try:
            console.print()
            question = console.input("[bold cyan]❓ Votre question : [/bold cyan]")

            if not question.strip():
                console.print("[yellow]Veuillez entrer une question.[/yellow]")
                continue

            from src.search import semantic_search
            results = semantic_search(question.strip())
            display_results(question.strip(), results)

        except KeyboardInterrupt:
            console.print("\n")
            console.print(Panel(
                "[bold yellow]Session terminée. Au revoir ![/bold yellow]",
                border_style="yellow",
                padding=(0, 2)
            ))
            break
        except EOFError:
            break


# ──────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────

def main():
    """
    Parse command-line arguments and dispatch to the correct mode.

    Modes:
        --ingest      : Run full PDF ingestion pipeline
        --query "..." : Run a single search query
        --demo        : Run the official challenge example
        --interactive : Interactive question-answer loop
    """
    parser = argparse.ArgumentParser(
        description="🍞 RAG Bakery — Semantic Search for Boulangerie & Pâtisserie",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --ingest
  python main.py --query "Quelles sont les quantités recommandées d'alpha-amylase ?"
  python main.py --demo
  python main.py --interactive
        """
    )

    parser.add_argument(
        '--ingest',
        action='store_true',
        help='Run the full PDF ingestion pipeline (extract → chunk → embed → store)'
    )
    parser.add_argument(
        '--query',
        type=str,
        metavar='QUESTION',
        help='Run a single semantic search query'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run the official challenge example question'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive search mode (loop until Ctrl+C)'
    )

    args = parser.parse_args()

    # Banner
    console.print(Panel(
        Text("RAG BAKERY", style="bold white", justify="center"),
        subtitle="Semantic Search for Boulangerie & Pâtisserie │ Model: all-MiniLM-L6-v2 │ Top K: 3 │ Cosine Similarity",
        border_style="bright_magenta",
        padding=(1, 4)
    ))

    if args.ingest:
        run_ingest()
    elif args.query:
        run_query(args.query)
    elif args.demo:
        run_demo()
    elif args.interactive:
        run_interactive()
    else:
        parser.print_help()
        console.print("\n[yellow]💡 Tip: Use --ingest first, then --demo or --interactive to search.[/yellow]")


if __name__ == '__main__':
    main()
