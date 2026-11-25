# Copilote ADV — extraction documentaire

Premier module documentaire : Docling convertit un fichier local en Markdown structuré,
puis HybridChunker prépare des passages avec le tokenizer Qwen3 Embedding.

Formats de fichiers acceptés dans ce module : PDF, Word (`.doc`, `.docx`), Excel
(`.xls`, `.xlsx`), HTML (`.html`, `.htm`) et emails (`.eml`, `.msg`). Les fichiers `.xls`
et `.doc` peuvent nécessiter LibreOffice selon le système. Docling reçoit directement les
URLs HTTP(S), ce qui permet aussi de convertir des pages web.

Installer Docling :

```powershell
uv sync --extra rag
```

Exemple :

```python
from pathlib import Path

from app.rag.extraction import DoclingExtractor

extractor = DoclingExtractor()

pdf = extractor.extract(Path("documents/procedure.pdf"))
excel = extractor.extract(Path("documents/tarifs.xlsx"))
email = extractor.extract(Path("documents/message.eml"))

print(pdf.markdown)
for chunk in pdf.chunks:
    print(chunk.chunk_id, chunk.page_number, chunk.text)
```

Chaque passage contient un identifiant local, sa page source lorsqu’elle est connue et le
texte contextualisé produit par HybridChunker. Le post-traitement retire les sommaires et
séparateurs de tableau résiduels, propage le titre de section et réunit les coupures qui
semblent tomber au milieu d’une ligne ou séparer une courte phrase de son tableau. La
limite initiale est de 512 tokens.
