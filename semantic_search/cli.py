
import typer
import numpy as np
from rich import print
from rich.panel import Panel
from rich.table import Table

from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import cosine_similarity
from semantic_search.similarity import dot_product
from semantic_search.similarity import euclidean_distance
from semantic_search.similarity import interpret_cosine
from semantic_search.similarity import interpret_euclidean
from semantic_search.similarity import interpret_dot_product    

app = typer.Typer(help="Semantic Search CLI using FastEmbed")


@app.command()
def embed(
    text: str = typer.Argument(..., help="Text to embed"),
    model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Embedding model name"),
):
    """
    Generate embedding for a single text.
    """
    gen = EmbeddingGenerator(model)
    vec = gen.embed_single(text)

    # Convert to numpy for stats
    arr = np.array(vec, dtype=np.float32)

    # Header box
    print(Panel.fit("Text Embedding", style="bold cyan"))

    # Input preview
    print(f'[bold] Input:[/bold] "{text}"\n')

    # Model info
    print(f"[bold]Model:[/bold] {model}")
    print(f"[bold]Dimensions:[/bold] {len(vec)}\n")

    # First 10 values
    first_10 = [round(float(x), 4) for x in vec[:10]]
    print("[bold green]Embedding (first 10 values):[/bold green]")
    print(f"  {first_10}\n")

    # Stats
    print("[bold magenta]Stats:[/bold magenta]")
    print(f"  Min: {arr.min(): .4f}")
    print(f"  Max: {arr.max(): .4f}")
    print(f"  Mean:{arr.mean(): .4f}")



@app.command()
def compare(
    text1: str = typer.Argument(..., help="First text"),
    text2: str = typer.Argument(..., help="Second text"),
    model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Embedding model name"),
):
    """
    Compare two texts and compute their cosine similarity.
    """
    gen = EmbeddingGenerator(model)
    vec1 = gen.embed_single(text1)
    vec2 = gen.embed_single(text2)

    cos= cosine_similarity(vec1, vec2)
    euc= euclidean_distance(vec1, vec2)
    dot= dot_product(vec1, vec2)

    # Header box
    print(Panel.fit("Text Comparison", style="bold cyan"))

    # Input preview
    print(f'[bold] Input 1:[/bold] "{text1}"\n')
    print(f'[bold] Input 2:[/bold] "{text2}"\n')

    #Table for results
    table = Table(title="Similarity Scores", show_lines=True)

    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Interpretation", style="green")

    table.add_row("Cosine Similarity", f"{cos:.4f}", interpret_cosine(cos))
    table.add_row("Euclidean Distance", f"{euc:.4f}", interpret_euclidean(euc))
    table.add_row("Dot Product", f"{dot:.4f}", interpret_dot_product(dot))

    print(table)


if __name__ == "__main__":
    app()
