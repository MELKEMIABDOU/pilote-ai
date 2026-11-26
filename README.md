# Copilote ADV

Copilote qui prépare et contrôle les bons de commande B2B pour les équipes ADV.

**État actuel :** l'outil RAG (recherche dans les documents) fonctionne en local.
Pas encore fait : accès MCP/API pour les agents, bons de commande et règles de contrôle.

## Installation

```powershell
uv sync --extra rag   # --extra rag installe Docling, PyTorch et Transformers
```

Après un `uv add` ou `uv remove`, relancer `uv sync --extra rag` (sinon ces paquets sont retirés).

## Tester le RAG soi-même

**1. Indexer un document** pour un client (la première fois, le modèle Qwen3 ≈ 1,2 Go est
téléchargé) :

```powershell
uv run python -m app.cli index tests/fixtures/horizon.pdf --client demo
```

Relancer la même commande répond « déjà indexé » sans rien recalculer.
`--reindex` force le recalcul.

**2. Poser une question :**

```powershell
uv run python -m app.cli search "Quel est le prix d'achat du riz ?" --client demo
uv run python -m app.cli search "Quel fournisseur est à surveiller ?" --client demo --top-k 3
```

Chaque résultat affiche le fichier, la page et le score.

**3. Vérifier l'isolation des clients** : un autre client ne voit rien.

```powershell
uv run python -m app.cli search "prix du riz" --client autre
```

La base est dans `data/vector_store.db` (ignorée par git). La supprimer remet tout à zéro.

## Mesurer la qualité

```powershell
uv run python -m scripts.eval_retrieval --store sqlite --save
```

Le script pose les 30 questions de `tests/fixtures/eval/horizon_questions.json` et affiche,
pour chacune, le rang de la bonne page. Avec `--save`, le résultat est enregistré dans
`outputs/eval/` pour comparer les essais.

- **Recall@1** : la bonne page est en 1re position.
- **Recall@5** : la bonne page est dans les 5 premiers résultats.
- **MRR** : 1 = toujours 1re, 0,5 = toujours 2e.

### Résultats (rapport Horizon, 30 passages, 26 questions avec réponse + 4 sans)

| Version | Recall@1 | Recall@5 | MRR | Temps embeddings |
|---|---|---|---|---|
| Départ | 85 % | 100 % | 0,91 | — |
| + titre de section conservé après nettoyage | 85 % | 100 % | 0,92 | 735 s |
| Stockage sqlite-vec (au lieu de la mémoire) | 85 % | 100 % | 0,92 | 1 451 s |
| **+ float32 sur CPU (actuel)** | **81 %** | **100 %** | **0,90** | **149 s** |

- **sqlite-vec donne exactement les mêmes résultats** que la recherche en mémoire.
- **float32 : 10× plus rapide.** Les poids Qwen3 sont en bfloat16, émulé (lent) sur les CPU
  sans instructions bfloat16. La baisse à 81 % vient d'une seule question (q24) dont les deux
  meilleurs passages sont à égalité (écart de 0,0001) : c'est du bruit, pas une régression.
- **Pas de seuil « rien trouvé »** : les questions sans réponse ont des scores aussi hauts que
  les bonnes réponses (jusqu'à 0,76). C'est l'agent qui juge si les passages répondent.
- **Limite** : un seul petit document, donc le recall@5 est facile. Il faudra ajouter d'autres
  documents pour une mesure plus exigeante.

## Tests et qualité du code

```powershell
uv run pytest                 # tests rapides (sans modèle ni téléchargement)
uv run pytest -m integration  # vrais Docling, Qwen3 et sqlite-vec (lent)
uv run ruff check .
uv run mypy
```

## Lancer l'API

```powershell
uv run uvicorn app.main:app --reload
```

Puis ouvrir http://127.0.0.1:8000/health et http://127.0.0.1:8000/docs.

## Structure

```
app/
├── domain/          # règles métier et erreurs (sans dépendance externe)
├── ports/           # interfaces : extraction, embeddings, stockage vectoriel
├── adapters/        # implémentations : Docling, Qwen3, sqlite-vec
├── application/     # cas d'usage : ingest_document, search_documents
├── api/             # routes FastAPI
├── rag_container.py # seul endroit qui choisit les implémentations
└── cli.py           # utilisation en ligne de commande
scripts/             # évaluation de la qualité
```

Les documents clients vont dans `documents/` (ignoré par git).
