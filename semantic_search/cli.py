import time
import typer
import os
from typing import List
import numpy as np
from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

from semantic_search.embeddings import EmbeddingGenerator
from semantic_search.similarity import cosine_similarity
from semantic_search.similarity import dot_product
from semantic_search.similarity import euclidean_distance
from semantic_search.similarity import interpret_cosine
from semantic_search.similarity import interpret_euclidean
from semantic_search.similarity import interpret_dot_product
from semantic_search.similarity import find_top_k    
from semantic_search.index import DocumentIndex

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




@app.command()
def search(
    query: str = typer.Argument(..., help="Query text"),
    corpus: list[str] = typer.Option(..., "--corpus", help="Add corpus document (repeatable)"),
    top_k: int = typer.Option(5, help="How many top results to show"),
    model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Embedding model name"),
):
    """
    Find similar texts from a given list.
    """
    gen = EmbeddingGenerator(model)

    # embed query + corpus
    query_vec = gen.embed_single(query)
    corpus_vecs = gen.embed_batch(corpus)

    # get top matches (index, score)
    matches = find_top_k(query_vec, corpus_vecs, k=top_k)

    # UI header
    print(Panel.fit(" Semantic Search", style="bold cyan"))
    print(f'[bold] Query:[/bold] "{query}"\n')

    # table
    table = Table(title="Results (ranked by similarity)", show_lines=True)
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Document", style="green")

    for rank, (idx, score) in enumerate(matches, start=1):
        table.add_row(str(rank), f"{score:.3f}", corpus[idx])

    print(table)

    # relevant results count
    relevant = [s for _, s in matches if s > 0.5]
    print(f"\n Found {len(relevant)} relevant results (score > 0.5)")


#This is for index commands

index_app = typer.Typer(help="Build and search a document index")
app.add_typer(index_app, name="index")


def _read_docs(file_path: str) -> List[str]:
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = line.strip()
            if doc:
                docs.append(doc)
    return docs


@index_app.command("build")
def index_build(
    file_path: str = typer.Argument(..., help="Text file (1 document per line)"),
    name: str = typer.Option(..., "--name", help="Index name (saved as <name>.npz)"),
    model: str = typer.Option("BAAI/bge-small-en-v1.5", help="Embedding model name"),
):
    """
    Build index from a file and save as NPZ.
    """
    print(Panel.fit(" Building Index", style="bold cyan"))

    print(f"[bold]Loading documents from:[/bold] {file_path}")
    docs = _read_docs(file_path)
    print(f"[bold]Documents found:[/bold] {len(docs)}\n")

    if len(docs) == 0:
        raise typer.BadParameter("No documents found in file.")

    # build index object
    idx = DocumentIndex(name=name, model_name=model)

    print("[bold]Generating embeddings...[/bold]")
    # embed with progress bar (simple but effective)
    for doc in track(docs, description="Embedding documents"):
        idx.add_documents([doc])  # add one by one to show progress

    # save
    idx.save(name)
    out_file = f"{name}.npz"

    size_bytes = os.path.getsize(out_file)
    size_mb = size_bytes / (1024 * 1024)

    dims = idx.embeddings.shape[1] if idx.embeddings is not None else 0

    print(f"\n[bold green]Index saved:[/bold green] {out_file}")
    print(f"  - Documents: {len(idx.documents)}")
    print(f"  - Dimensions: {dims}")
    print(f"  - Size: {size_mb:.2f} MB")


@index_app.command("search")
def index_search(
    name: str = typer.Argument(..., help="Index name (without .npz)"),
    query: str = typer.Argument(..., help="Query text"),
    top: int = typer.Option(5, "--top", help="Top results to show"),
):
    """
    Search an existing index saved as <name>.npz
    """
    print(Panel.fit("Index Search", style="bold cyan"))

    idx = DocumentIndex.load(name)

    print(f"[bold]Index:[/bold] {idx.name} ({len(idx.documents)} documents)")
    print(f'[bold]Query:[/bold] "{query}"\n')

    results = idx.search(query, top_k=top)

    table = Table(title=f"Top {min(top, len(results))} Results", show_lines=True)
    table.add_column("Rank", justify="center", style="bold")
    table.add_column("Score", justify="right", style="magenta")
    table.add_column("Document", style="green")

    for rank, r in enumerate(results, start=1):
        table.add_row(str(rank), f"{r['score']:.3f}", r["document"])

    print(table)


# Benchmark command

MODEL_SIZES_MB = {
    "sentence-transformers/all-MiniLM-L6-v2": "90 MB",
    "BAAI/bge-small-en-v1.5": "130 MB",
    "BAAI/bge-base-en-v1.5": "420 MB",
}


@app.command()
def benchmark(
    text: str = typer.Argument(..., help="Text to benchmark embedding models on"),
):
    """
    Compare embedding models: speed + dims (+ approximate model size).
    """
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "BAAI/bge-base-en-v1.5",
    ]

    print(Panel.fit("Model Benchmark", style="bold cyan"))
    print(f'[bold]Testing models with:[/bold] "{text}"\n')

    table = Table(show_lines=True)
    table.add_column("Model", style="cyan")
    table.add_column("Dims", justify="right", style="magenta")
    table.add_column("Time (ms)", justify="right", style="green")
    table.add_column("Size", justify="right", style="yellow")

    best_model = None
    best_time = float("inf")

    for model_name in models_to_test:
        gen = EmbeddingGenerator(model_name)

        # warmup (model download/load may happen)
        gen.embed_single(text)

        # timed run
        start = time.perf_counter()
        vec = gen.embed_single(text)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        dims = len(vec)

        # best speed
        if elapsed_ms < best_time:
            best_time = elapsed_ms
            best_model = model_name

        size = MODEL_SIZES_MB.get(model_name, "Unknown")

        table.add_row(model_name, str(dims), f"{elapsed_ms:.0f}", size)

    print(table)

    # Recommendation
    recommendation = best_model if best_model is not None else "N/A"
    
    print(f"\n💡 [bold]Recommendation:[/bold] {recommendation} (best speed/quality balance)")
    


if __name__ == "__main__":
    app()
