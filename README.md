# Semantic Search CLI 🔍

A command-line tool that performs **semantic search** using **text embeddings** and **vector similarity**.  
It converts text into embeddings (vectors) and finds similar texts using cosine similarity.

Built using:
- **FastEmbed** (local ONNX embeddings)
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

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
