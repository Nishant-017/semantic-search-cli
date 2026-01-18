# Semantic Search CLI 🔍

A command-line tool that performs **semantic search** using **text embeddings** and **vector similarity**.  
It converts text into embeddings (vectors) and finds similar texts using cosine similarity.

Built using:
- **FastEmbed** (local embeddings)
- **NumPy** (vector operations)
- **Typer** (CLI)
- **Rich** (beautiful CLI output)

---

## 📁 Project Structure

semantic-search/
├── semantic_search/
│ ├── init.py
│ ├── cli.py # CLI commands
│ ├── embeddings.py # Embedding generation (FastEmbed)
│ ├── similarity.py # Similarity calculations
│ └── index.py # In-memory + persistent document index (.npz)
├── tests/
│ ├── test_embeddings.py
│ └── test_similarity.py
├── data/
│ └── excuses.txt # Sample documents file (one per line)
├── requirements.txt
└── README.md


---

## ✅ Features

### CLI Commands:
1. **embed** → Generate embedding for a text
2. **compare** → Compare similarity of two texts
3. **search** → Find similar texts from a given list
4. **index** → Build and search a persistent document index
5. **benchmark** → Compare multiple embedding models (speed + dimensions)

---

## ⚙️ Setup Instructions

### 1) Create and activate virtual environment

**Windows (PowerShell)**:

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt



### 1) Embed a Text

Generate an embedding vector for a sentence.

python -m semantic_search.cli embed "I love semantic search"

### 2) Compare Two Texts

python -m semantic_search.cli compare "It's not a bug" "It's a feature"

### 3) Search from a List (Semantic Search)

python -m semantic_search.cli search "why is my code not working" `
  --corpus "How to mass mass effectively" `
  --corpus "Best pizza recipes in Mumbai" `
  --corpus "Debugging tips for Python" `
  --corpus "How to mass a meeting productively" `
  --corpus "Stack Overflow error solutions"

### 4) Index: Build and Search a Document Index

python -m semantic_search.cli index build data/excuses.txt --name dev_excuses

### B) Search from an Index

python -m semantic_search.cli index search dev_excuses "the build is broken" --top 5

### 5) Benchmark Embedding Models

python -m semantic_search.cli benchmark "why do programmers prefer dark mode"


## 📌 Notes

The first run may be slower because FastEmbed downloads models locally.

Index files are generated outputs. Do not commit them to GitHub.