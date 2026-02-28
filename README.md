# 🍞 RAG Bakery — Module de Recherche Sémantique
## Boulangerie & Pâtisserie Intelligence Platform

Un module **RAG (Retrieval-Augmented Generation)** de recherche sémantique de niveau production pour une base de données d'ingrédients de boulangerie. Extrait le texte de fiches techniques PDF, génère des embeddings, et permet la recherche en langage naturel sur les spécifications d'ingrédients.

---

## ✅ Conformité Challenge

| Contrainte imposée | Implémentation | Statut |
|---|---|---|
| Modèle d'embedding : `all-MiniLM-L6-v2` | `src/embedder.py` — singleton, `normalize_embeddings=True` | ✅ |
| Dimension : 384 | `config.py` — `EMBEDDING_DIM = 384` | ✅ |
| Méthode de similarité : Cosine Similarity | `src/db.py` — opérateur pgvector `<=>` | ✅ |
| Top K = 3 résultats | `config.py` — `TOP_K = 3` | ✅ |
| Table : `embeddings` | `src/db.py` — `CREATE TABLE embeddings` | ✅ |
| Colonnes : `id`, `id_document`, `texte_fragment`, `vecteur` | `src/db.py` — schéma exact | ✅ |
| Type vecteur : `VECTOR(384)` | `src/db.py` — `vecteur VECTOR(384)` | ✅ |
| Langage : Python | Tout le projet | ✅ |
| Format de sortie : `Résultat N / Texte / Score` | `src/search.py` — `format_results()` | ✅ |

---

## 🏗️ Architecture

```
PDF Files → Extract → Chunk → Embed → Store → Search → Display
   │          │         │        │        │        │        │
   │     pdfplumber  Sliding  MiniLM   pgvector  Cosine   Rich
   │     PyMuPDF     Window   L6-v2    PostgreSQL Similarity CLI
   │     Tesseract   300w/50w 384-dim   HNSW     Top 3
   └─────────────────────────────────────────────────────────┘
```

### Pipeline en 7 étapes

| Étape | Module | Description |
|-------|--------|-------------|
| 1 | `src/extractor.py` | Extraction PDF 3 couches (pdfplumber → PyMuPDF → Tesseract OCR) |
| 2 | `src/chunker.py` | Découpage par fenêtre glissante respectant les limites de phrases (300 mots, 50 overlap) |
| 3 | `src/embedder.py` | Embedding avec `all-MiniLM-L6-v2` (384 dimensions, normalisé) |
| 4 | `src/db.py` | PostgreSQL + pgvector avec index HNSW |
| 5 | `src/ingest.py` | Pipeline d'orchestration avec suivi de progression |
| 6 | `src/search.py` | Recherche par similarité cosinus, retourne les 3 meilleurs résultats |
| 7 | `main.py` | CLI Rich (ingestion / requête / démo / interactif) |

---

## 📋 Prérequis

- **Python 3.10+**
- **Docker** (pour PostgreSQL + pgvector)
- **Tesseract OCR** installé sur le système

### Installation de Tesseract

**Ubuntu/Debian :**
```bash
sudo apt install tesseract-ocr tesseract-ocr-fra tesseract-ocr-eng
```

**Windows :**
Télécharger l'installateur depuis : https://github.com/UB-Mannheim/tesseract/wiki
Ajouter au PATH et installer le pack de langue français.

**macOS :**
```bash
brew install tesseract tesseract-lang
```

---

## 🚀 Installation

### 1. Démarrer PostgreSQL avec pgvector
```bash
docker run -d --name pgvector_bakery \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=bakery_rag \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### 2. Créer un environnement virtuel et installer les dépendances
```bash
python -m venv venv

# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configurer l'environnement
```bash
cp .env.example .env
# Modifier .env si vos identifiants de base de données diffèrent
```

### 4. Placer vos PDFs
```bash
# Copier toutes les fiches techniques PDF dans data/pdfs/
cp /vos/pdfs/*.pdf data/pdfs/
```

---

## ▶️ Utilisation

### Ingérer tous les PDFs (exécuter une fois)
```bash
python main.py --ingest
```
Ceci va :
- Initialiser la base de données (extension pgvector, table, index HNSW)
- Extraire le texte de tous les PDFs (fallback 3 couches)
- Découper le texte par fenêtre glissante (300 mots, 50 overlap)
- Générer les embeddings avec `all-MiniLM-L6-v2`
- Stocker tout dans PostgreSQL

### Requête unique
```bash
python main.py --query "Quelles sont les quantités recommandées d'alpha-amylase ?"
```

### Démo officielle du challenge
```bash
python main.py --demo
```
Exécute la question exemple officielle du challenge :
> *Améliorant de panification : quelles sont les quantités recommandées d'alpha-amylase, xylanase et d'Acide ascorbique ?*

### Mode interactif
```bash
python main.py --interactive
```
Tapez vos questions en langage naturel. Appuyez sur `Ctrl+C` pour quitter.

### Tests de validation
```bash
python test_search.py
python test_standalone.py
```

---

## 📁 Structure du projet

```
rag_bakery/
├── data/
│   └── pdfs/              ← Les fiches techniques PDF vont ici
├── src/
│   ├── __init__.py
│   ├── extractor.py       ← Extraction PDF 3 couches
│   ├── chunker.py         ← Découpage par fenêtre glissante
│   ├── embedder.py        ← Embeddings MiniLM-L6-v2
│   ├── db.py              ← Opérations PostgreSQL + pgvector
│   ├── ingest.py          ← Pipeline d'orchestration
│   └── search.py          ← Recherche sémantique + formatage
├── config.py              ← Toutes les constantes & configuration
├── main.py                ← Point d'entrée CLI Rich
├── test_search.py         ← Suite de tests de validation
├── test_standalone.py     ← Tests sans base de données
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Configuration

Tous les paramètres sont dans `config.py` et peuvent être surchargés via `.env` :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | **Imposé par le challenge** — ne pas modifier |
| `EMBEDDING_DIM` | `384` | Dimensions du vecteur |
| `TOP_K` | `3` | Nombre de résultats de recherche |
| `CHUNK_SIZE` | `300` | Mots cibles par chunk |
| `CHUNK_OVERLAP` | `50` | Mots de chevauchement entre chunks |
| `MIN_CHUNK_WORDS` | `30` | Minimum de mots pour un chunk valide |
| `BATCH_SIZE` | `64` | Chunks par lot d'embedding |

---

## 🗄️ Schéma de base de données

```sql
-- Extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Table embeddings (spécification exacte du challenge)
CREATE TABLE IF NOT EXISTS embeddings (
    id           SERIAL PRIMARY KEY,
    id_document  INTEGER NOT NULL,
    texte_fragment TEXT NOT NULL,
    vecteur      VECTOR(384)
);

-- Index HNSW pour recherche rapide par similarité cosinus
CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
ON embeddings
USING hnsw (vecteur vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## 🔍 Format de sortie

Le module retourne les résultats dans le format exact exigé par le challenge :

```
Résultat 1
Texte : "Dosage recommandé : 0.005% à 0.02% du poids de farine."
Score : 0.91

Résultat 2
Texte : "Alpha-amylase : utilisation entre 5 et 20 ppm selon la farine."
Score : 0.87

Résultat 3
Texte : "Xylanase : améliore l'extensibilité de la pâte..."
Score : 0.82
```

---

## 🔬 Fonctionnement du module de recherche

1. **Réception de la question** — L'utilisateur formule une question en langage naturel
2. **Génération de l'embedding** — La question est convertie en vecteur 384D via `all-MiniLM-L6-v2`
3. **Calcul de similarité cosinus** — Comparaison du vecteur question avec tous les vecteurs stockés via l'opérateur `<=>` de pgvector
4. **Classement** — Les résultats sont triés par score de similarité décroissant
5. **Sélection Top 3** — Seuls les 3 fragments les plus pertinents sont retournés
6. **Affichage** — Chaque résultat affiche le texte du fragment et le score de similarité

---

## 📄 Licence

Développé pour le Challenge RAG — Plateforme d'Intelligence Boulangerie & Pâtisserie.
