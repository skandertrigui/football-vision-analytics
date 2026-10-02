# 📄 AI Research Assistant — RAG Pipeline

<p align="center">
  <img src="APERCU.png" alt="Aperçu de l'application AI Research Assistant" width="100%" />
</p>

> Une application web RAG (Retrieval-Augmented Generation) de niveau production conçue pour ingérer, analyser et interroger des articles scientifiques et documents PDF académiques avec une précision contextuelle élevée et une traçabilité stricte des sources.

---

## 🚀 Fonctionnalités Clés

* **Interface SaaS Épurée :** Design moderne sous Streamlit avec masquage des éléments natifs et customisation CSS poussée.
* **Isolation Multi-Sessions :** Gestion de plusieurs discussions en parallèle. Chaque session possède son propre espace documentaire et sa propre collection ChromaDB dédiée pour éviter tout croisement de données.
* **Pipeline LangChain Robuste :** 
  * Découpage intelligent du texte (*RecursiveCharacterTextSplitter*).
  * Reformulation autonome des requêtes ambigües selon l'historique de conversation (`CONDENSE_PROMPT`).
  * Récupération sémantique ultra-précide (*Top-K retrieval*).
* **Traçabilité & Citations :** Affichage de badges de pages interactifs et expandeurs de sources pour chaque réponse générée afin d'éliminer les hallucinations.
* **Export de Session :** Module complet d'exportation de l'historique de discussion au format PDF (`fpdf2`).
* **Starter Prompts :** Suggestions dynamiques pour démarrer l'exploration d'un nouveau document en un clic.

---

## 🛠️ Stack Technique

* **Langage :** Python 3.10+
* **Interface Graphique :** Streamlit
* **Orchestration RAG :** LangChain (`langchain-core`, `langchain-community`, `langchain-chroma`, `langchain-google-genai`)
* **Base de Données Vectorielle :** ChromaDB (persistance locale)
* **Modèles IA (Google Gemini) :**
  * LLM : `gemini-3.8-flash` (génération rapide en streaming)
  * Embeddings : `gemini-embedding-2-preview` (vectorisation fine et multilingue)
* **Export PDF :** `fpdf2`

---

## ⚙️ Installation & Laisement en Local

1. **Cloner le repository :**
   ```bash
   git clone [https://github.com/ton-username/ai-research-assistant-rag.git](https://github.com/ton-username/ai-research-assistant-rag.git)
   cd ai-research-assistant-rag
