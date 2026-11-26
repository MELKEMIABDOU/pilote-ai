# Copilote ADV

Copilote qui prépare et contrôle les bons de commande B2B pour les équipes ADV.

**État actuel :** seule l'extraction documentaire est faite (Docling → Markdown + passages).

## Installation

```powershell
uv sync --extra rag   # --extra rag installe Docling (nécessaire pour extraire de vrais documents)
```

## Lancer l'API

```powershell
uv run uvicorn app.main:app --reload
```

Puis ouvrir http://127.0.0.1:8000/health et http://127.0.0.1:8000/docs.

## Extraire un document

```python
from pathlib import Path

from app.adapters.extraction.docling import DoclingExtractor

result = DoclingExtractor().extract(Path("documents/commande.pdf"))

print(result.markdown)
for chunk in result.chunks:
    print(chunk.chunk_id, chunk.page_number, chunk.text)
```

- Formats acceptés : PDF, Word, Excel, HTML, emails (`.eml`, `.msg`).
- Fichiers locaux uniquement : les URLs sont refusées.

## Tests et qualité

```powershell
uv run pytest                 # tests rapides
uv run pytest -m integration  # test sur un vrai PDF (lent)
uv run ruff check .
uv run mypy
```

## Structure

```
app/
├── domain/     # règles métier (sans dépendance externe)
├── ports/      # interfaces
├── adapters/   # implémentations (Docling, ERP, base de données…)
├── application/# cas d'usage
└── api/        # routes FastAPI
```

Les documents clients vont dans `documents/` (ignoré par git).
