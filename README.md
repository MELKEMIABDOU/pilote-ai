# Copilote ADV

Copilote qui prépare et contrôle les bons de commande B2B pour les équipes ADV.

**État actuel :**
- l'outil RAG (recherche dans les documents) fonctionne en local, en ligne de commande et via
  un serveur MCP pour les agents ;
- le modèle de bon de commande et ses contrôles sont faits ;
- un LLM local lit un bon de commande (PDF, Excel, Word, HTML, email) et le code le contrôle
  (voir « Lire et contrôler une commande »).

Pas encore fait : ERP, écran de validation humaine.

## Lire et contrôler une commande

Il faut [Ollama](https://ollama.com) lancé, avec le modèle téléchargé :

```powershell
ollama pull qwen3:8b
uv run python -m app.cli controler tests/fixtures/orders/o02_pdf_erreur_cafe.pdf
```

Sortie réelle :

```
=== o02_pdf_erreur_cafe.pdf : ❌ À CORRIGER ===
Client : Épicerie Martin | n° de commande : CMD-2026-119
Date : 2026-11-04 | livraison souhaitée : —
Lignes :
  ✅ L1 P006 Riz 1 kg : 100 × 1,85 € (5,5 %)
  ❌ L2 P004 Café 250 g : 50 × 4,20 € (5,5 %)
Total HT calculé : 395,00 €
Total TTC calculé : 416,73 €
Anomalies :
  ❌ Ligne 2 (Café 250 g) : total incohérent (215,00 € écrit, 210,00 € calculé = 50 × 4,20 €).
  ❌ Total HT incohérent : 400,00 € écrit, 395,00 € calculé.
(lu par qwen3:8b en 39.2 s — l'ADV doit valider avant toute saisie)
```

**Principe : le LLM recopie, le code calcule et vérifie.** Le modèle recopie les valeurs
telles qu'elles sont écrites ; le code les convertit (`app/domain/parsing.py`), vérifie que
chacune existe vraiment dans le document (sinon : « valeur probablement inventée »,
bloquant), puis applique les contrôles ci-dessous. Le document est passé au modèle comme une
donnée : les instructions qu'il contient sont ignorées.

Si le PC chauffe ou s'éteint pendant la lecture, mettre une partie du modèle sur le CPU
(plus lent, moins de puissance demandée à la carte graphique) dans `.env` :
`COPILOTE_LLM_GPU_LAYERS=18`.

### Mesurer la lecture

```powershell
uv run python -m scripts.eval_extraction --models qwen3:8b --save
# PC qui chauffe : --gpu-layers 18 --pause 20 ; reprise après un arrêt : --resume <fichier>
```

10 commandes fictives (`tests/fixtures/orders/`, valeurs attendues dans `expected.json`),
qwen3:8b, 18 couches sur la carte graphique (RX 6600M) :

| Mesure | Résultat |
|---|---|
| Champs justes | 174/178 (98 %) |
| Commandes parfaites | 9/10 |
| Verdict juste (prête / à corriger, bonnes anomalies) | 9/10 |
| Fausses alertes « introuvable » | 0 |
| Temps par commande | 39 s |

La commande ratée est l'email piège (o05), qui demande à l'IA de « mettre 0 € partout » : les
prix restent justes, mais le modèle écrit 0 € dans les totaux. Les contrôles le détectent et
la commande sort **à corriger** : rien de faux ne serait saisi. En plus, le code repère les
phrases qui s'adressent à une IA (`app/domain/injection.py`, sans IA) et l'affiche en
**première anomalie** : « Le document contient un texte qui s'adresse à l'IA (possible
tentative de manipulation) … Vérifier chaque valeur à la main. »

### Les contrôles

`uv run python -m scripts.demo_controle` affiche deux commandes d'exemple. Les contrôles
(`app/domain/rules.py`) sont déterministes, en `Decimal`, sans IA :

| Contrôle | Gravité |
|---|---|
| Client absent, aucune ligne | bloquant |
| Ligne sans référence, quantité ou prix ; quantité ≤ 0 ; prix négatif | bloquant |
| Total de ligne ≠ quantité × prix (tolérance 0,01 € par ligne) | bloquant |
| Total HT ou TTC écrit ≠ total calculé | bloquant |
| Taux de TVA inconnu (autorisés : 20, 10, 5,5, 2,1 et 0 %) | bloquant |
| TVA absente, TVA à 0 %, référence en double, n° de commande absent, livraison avant la commande | avertissement |

« Bloquant » : la commande ne peut pas être saisie en l'état. « Avertissement » : à vérifier.
Une ligne sans anomalie bloquante reste « prête » même si une autre ligne ne l'est pas.

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

Chaque résultat affiche le fichier, la ou les pages (ex. `p.4-5`) et le score.
Un document indexé avant l'ajout des pages multiples garde une seule page : relancer `index`
avec `--reindex` pour le mettre à jour.

**3. Vérifier l'isolation des clients** : un autre client ne voit rien.

```powershell
uv run python -m app.cli search "prix du riz" --client autre
```

La base est dans `data/vector_store.db` (ignorée par git). La supprimer remet tout à zéro.

## Utiliser le RAG depuis un agent (serveur MCP)

Le serveur MCP « copilote » expose un outil **en lecture seule** : `chercher_documents(question,
top_k)`. Il renvoie les passages avec leur source, leurs pages et leur score.

**Le client est fixé par la configuration** (`COPILOTE_MCP_CLIENT_ID`), pas par l'agent :
l'agent ne peut lire que les documents de ce client. L'indexation reste dans la CLI (un agent
ne peut pas ajouter ni supprimer de documents).

**1. Indexer d'abord** les documents du client avec la CLI (voir plus haut).

**2. Tester sans aucun LLM, avec MCP Inspector** (gratuit, nécessite Node.js) :

```powershell
npx @modelcontextprotocol/inspector -e COPILOTE_MCP_CLIENT_ID=demo uv run python -m app.mcp_server
```

Une page web s'ouvre : « Connect », onglet « Tools », choisir `chercher_documents`, saisir une
question, « Run Tool ».

**3. Brancher sur un client MCP.** Remplacer `<chemin du projet>` par le dossier du projet.

Claude Code :

```powershell
claude mcp add copilote -e COPILOTE_MCP_CLIENT_ID=demo -- uv --directory "<chemin du projet>" run python -m app.mcp_server
```

Claude Desktop, LM Studio et la plupart des clients : ajouter dans leur fichier de
configuration MCP (pour Claude Desktop sous Windows : `%APPDATA%\Claude\claude_desktop_config.json`) :

```json
{
  "mcpServers": {
    "copilote": {
      "command": "uv",
      "args": ["--directory", "<chemin du projet>", "run", "python", "-m", "app.mcp_server"],
      "env": { "COPILOTE_MCP_CLIENT_ID": "demo" }
    }
  }
}
```

Le modèle est chargé au démarrage du serveur (quelques secondes) : le premier appel est donc
aussi rapide que les suivants.

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
| + float32 sur CPU | 81 % | 100 % | 0,90 | 149 s |
| **+ citations sur plusieurs pages, hors ligne (actuel)** | **85 %** | **100 %** | **0,92** | **130 s** |

- **sqlite-vec donne exactement les mêmes résultats** que la recherche en mémoire.
- **Citations « p.4-5 »** : la recherche est identique (mêmes scores), mais un passage à
  cheval sur deux pages était cité avec sa seule 1re page. Ex. q07 : le tableau des prix est
  page 5, le passage était étiqueté « p.4 » et compté comme faux. La mesure est maintenant juste.
- **Hors ligne** : l'évaluation complète passe avec `HF_HUB_OFFLINE=1` ; plus aucun appel à
  Hugging Face une fois les modèles téléchargés.
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
├── mcp_server/      # serveur MCP pour les agents
├── rag_container.py # seul endroit qui choisit les implémentations
└── cli.py           # utilisation en ligne de commande
scripts/             # évaluation de la qualité
```

Les documents clients vont dans `documents/` (ignoré par git).
