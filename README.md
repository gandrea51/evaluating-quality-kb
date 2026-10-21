# Definizione di un modello di calcolo degli indicatori di qualità della conoscenza in una Knowledge Base aziendale
## Caso di studio: la piattaforma CK (Company Knowledge)

Questo repository ospita codice sorgente, architettura dei dati e pipeline sperimentali per la definizione, l'implementazione e la validazione di un **motore di misura della qualità della conoscenza** integrato nella piattaforma aziendale **CK**.

L'obiettivo del progetto è trasformare la valutazione di una Knowledge Base aziendale da un processo empirico e soggettivo a una **misura oggettiva, quantitativa e ripetibile**, offrendo una dashboard analitica per i Knowledge Manager e gli amministratori di sistemi RAG (Retrieval-Augmented Generation).

---

## 🎯 Obiettivi del Progetto

Il sistema è progettato per analizzare i dati di esercizio della piattaforma e rispondere a quesiti critici:
* **Copertura:** Quanto è completa la copertura documentale di un’area tematica rispetto alla tassonomia aziendale?
* **Efficienza:** Quanta parte del contenuto è ridondante, obsoleta o frammentata a causa di strategie di chunking errate?
* **Accessibilità:** Quali documenti sono geometricamente "invisibili" o mai utilizzati dal retriever semantico?
* **Integrità:** Esistono informazioni contrastanti o esplicitamente contraddittorie tra fonti diverse?
* **Affidabilità:** In quale misura le risposte generate dall'AI sono effettivamente fondate sulle fonti documentali citate?

---

## 📐 Quadro delle Dimensioni di Qualità

Il motore analizza la Knowledge Base attraverso **8 macro-dimensioni di qualità**, combinando analisi geometrica dello spazio latente (embedding densi di ultima gen.), analisi logistica dei dati di log ed inferenza tramite modelli di linguaggio locali (LLM).

| Dimensione | Obiettivo Analitico | Approccio / Metrica | Stato Implementazione |
| :--- | :--- | :--- | :--- |
| **1. Copertura tematica** | Individuazione aree scoperte. | Clustering K-Means + Silhouette su spazio semantico. | 🟡 To be finished |
| **2. Ridondanza e quasi-duplicati** | Quota di unità informative sovrapposte. | Analisi dei grafi e componenti connesse (Soglia Cosine Similarity (0.92). | 🟡 To be finished |
| **3. Granularità e Integrità** | Adeguatezza e autoconsistenza dei chunk. | Distribuzione lunghezze, analisi troncamenti e maschere frammenti. | 🟡 To be finished |
| **4. Coerenza interna** | Rilevazione di affermazioni incompatibili. | Verifica di implicazione testuale (NLI) e discrepanza numerica su coppie 0.85. | 🟠 In progress (Background Thread) |
| **5. Freschezza e obsolescenza** | Attualità pesata sulla rilevanza reale. | Età media dei contenuti ponderata sulle frequenze di recupero. | ⏳ Work in Progress |
| **6. Recuperabilità** | Raggiungibilità delle unità informative. | Analisi della concentrazione del retrieval su log di esercizio. | ⏳ Work in Progress |
| **7. Fondatezza delle risposte** | Validazione delle risposte generate (Faithfulness). | Quota di claim verificabili sui passaggi citati (campionamento interazioni). | ⏳ Work in Progress |
| **8. Qualità percepita** | Riscontro dall'uso quotidiano in esercizio. | Analisi di risposte non conclusive, riformulazioni e feedback utente. | ⏳ Work in Progress |

---

## 📈 Indice Sintetico di Qualità (KB-IQI)

Le misure elementari estratte dalle 8 dimensioni confluiscono in un **indice sintetico di qualità della Knowledge Base**, calcolabile sia per singola area tematica sia per l'intera organizzazione. 

> 🔬 **Focus della Ricerca:** La determinazione dei pesi di aggregazione delle metriche costituisce l'oggetto centrale della tesi. Verrà proposto e giustificato un criterio di ponderazione matematico, validandone la tenuta rispetto alla qualità reale delle risposte osservate in esercizio.

---

## 🚀 Architettura del Software e Pipeline Corrente
*(Sezione in fase di aggiornamento)*

Attualmente la pipeline implementata esegue:
1. **Semantic Chunking:** Segmentazione logica dei documenti sorgente basata sulle distanze semantiche tra frasi.
2. **Dense Embedding:** Proiezione nello spazio latente a 1024 dimensioni tramite modello open-source **`BAAI/bge-m3`** (Hugging Face).
3. **Dimensionality Reduction:** Compressione dello spazio tramite *Singular Value Decomposition (SVD / PCA)* a 3 dimensioni per la diagnostica visiva e il calcolo della coesione.
4. **Analisi Asincrona LLM:** Validazione delle coppie candidate alla contraddizione in background tramite server locale **Ollama**.

---
*Nota: Questo progetto è un Work in Progress legato allo sviluppo della tesi magistrale per la piattaforma CK.*
