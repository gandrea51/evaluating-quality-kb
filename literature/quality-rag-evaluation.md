# 1. Qualità delle informazioni e dei dati

    Concetto generale: 
        cosa significa qualità dell'informazione
    Framework generici:
        quali dimensioni vengono proposte

    - Databricks - Che cosa si intende per qualità dei dati? - https://www.databricks.com/it/blog/what-is-data-quality

    - ZeroUno - La piramide DIKW, cos’è e cosa rappresenta - https://www.zerounoweb.it/big-data/la-piramide-dikw-cose-e-cosa-rappresenta/

    - International Monetary Fund - Data Quality Assessment Framework and Data Quality Program - https://www.imf.org/external/np/sta/dsbb/2003/eng/dqaf.htm

    - Digna - Total Data Quality Management (TDQM): A Practical Guide - https://www.digna.ai/total-data-quality-management-tdqm

    - ScienceDirect - Total Data Quality Management - https://www.sciencedirect.com/topics/computer-science/total-data-quality-management


# 2. Wang e Strong ('96) e Wang ('98) come ponte tra: (qualità delle info e dei dati) e (standard iso/iec)

    come viene formalizzata accademicamente la multidimensionalità

    - Wang, R. Y., & Strong, D. M. (1996). Beyond accuracy: What data quality means to data consumers. Journal of Management Information Systems, 12(4), pp 5-33 - https://www.tandfonline.com/doi/abs/10.1080/07421222.1996.11518099

    - Wang, R.Y. (1998) A Product Perspective on Total Data Quality Management. Communications of the ACM, 41, pp 58-65 - https://doi.org/10.1145/269012.269022

  
# 3. ISO / IEC 25012 (come viene standardizzata), ISO / IEC 25024 (come viene misurata)

    - ISO/IEC 25012:2008. Software engineering — Software product Quality Requirements and Evaluation (SQuaRE) — Data quality model. https://www.iso.org/standard/35736.html

    - AGID + Team digitale - Qualità dei dati - https://docs.italia.it/AgID/documenti-in-consultazione/lg-opendata-docs/it/bozza/aspetti-organizzativi-e-qualit%C3%A0-dei-dati/qualit%C3%A0-dei-dati.html

    - Agenzia per l'Italia digitale - Misurazione della qualità dei dati - https://www.agid.gov.it/sites/agid/files/2024-05/iso_25024_agid_misurazione_della_qualita_dei_dati.pdf

    - ISO/IEC 25024:2015. Systems and software engineering — Systems and software Quality Requirements and Evaluation (SQuaRE) — Measurement of data quality - https://www.iso.org/standard/35749.html

    Esempi per tre feature:
        Comprensibilità: Se possiedi un database con 10 tabelle ma solo 8 hanno una documentazione associata che ne spiega i 
            campi, il punteggio di comprensibilità sarà 0,8.
        Coerenza: Se in un database di e-commerce ci sono 1.000 ordini, ma in 50 di questi la "Data di Consegna" è antecedente 
            alla "Data di Acquisto" (contraddizione logica), i record coerenti sono 950. Il punteggio di consistenza sarà 0,95.
        Aggiornamento: Se l'azienda richiede che i saldi dei conti correnti siano aggiornati entro un massimo di 2 ore dall'operazione, 
            e su 100 transazioni totali ce ne sono 90 che rispettano questa tempistica (mentre 10 sono in ritardo), il valore di aggiornamento sarà 0,9.


# 5. ISO / IEC 5259: cosa cambia spostandosi verso Analytic/ML/not structered data

    - ISO/IEC 5259-2:2024 – Artificial intelligence – Data quality for analytics and machine learning (ML) - https://www.iso.org/standard/81860.html

    - ISO/IEC 5259 - Artificial intelligence. Data quality for analytics and machine learning - https://theiso5259.com/

    - Pebblous - 15 AI Data Quality Metrics You Need to Know - Data Quality Measures (QM) - https://blog.pebblous.ai/project/ISO5259/5259-2-cheetsheet-01/en/

    - Artificial intelligence — Data quality for analytics and machine learning (ML) — Part 2 - Data quality measures https://cdn.standards.iteh.ai/samples/81860/6ac019c359644309afa716572bb73edc/ISO-IEC-5259-2-2024.pdf

    perché è stata introdotta, quale problema vuole risolvere rispetto ai precedenti, cosa cambia quando si entra nel mondo analytic/ML e dei dati
    non strutturati


# 6. RAG

    - Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems, 33, pp. 9459-9474 - https://arxiv.org/abs/2005.11401 (def. rag)

    - Marco Pericci - RAG (Retrieval-Augmented Generation): cos’è, come funziona e come usarlo nel 2026 - https://www.marcopericci.com/rag/

    - LlamaIndex – Document Chunking Strategies - https://www.llamaindex.ai/glossary/document-chunking-strategies (def. chunking)

    - Günther, M., Mohr, I., Williams, D. J., Wang, B., Xiao, H. (2024). Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models - https://arxiv.org/abs/2409.04701 (def. del late chunking)

    - Brådland, H., Goodwin, M., Andersen, P.-A., Nossum, A. S., Gupta, A. (2025). A New HOPE: Domain-agnostic Automatic Evaluation of Text Chunking. Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1-10 - https://arxiv.org/abs/2505.02171 (def di hope)

    - Medium - Types of Retrieval in RAG - GenAI Lego Kit - https://medium.com/@upadhya.saumya/types-of-retrieval-in-rag-bb4e4a914e7c (type retriever)
    
    - AiBerto - Reranking per RAG: il retriever trova i candidati, il reranker sceglie la risposta - https://aiberto.it/reranking-per-rag/ (def. reranking)

    - Nogueira, R., Cho, K. (2019). Passage Re-ranking with BERT - https://arxiv.org/abs/1901.04085 (def. bert as reranking)

    - Apxml - Component-Level Evaluation: Retrieval - https://apxml.com/courses/getting-started-rag/chapter-6-evaluating-improving-rag-systems/evaluating-retrieval (def. precision, recall, mrr)

    - Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press, pp 1-482 - https://doi.org/10.1017/CBO9780511809071 (def. map, mrr)
    
    - Medium - Evaluating RAG Systems with Mean Reciprocal Rank (MRR) - DhanushKumar - https://blog.gopenai.com/evaluating-rag-systems-with-mean-reciprocal-rank-mrr-2e5a51227871 (def. mrr)

    - Järvelin, K., & Kekäläinen, J. (2002). Cumulated Gain-Based Evaluation of IR Techniques. ACM Transactions on Information Systems, 20(4), pp 422-446 - https://doi.org/10.1145/582415.582418 (def. ndcg)

    - Medium - NDCG vs. MRR: Ranking Metrics for Information Retrieval in RAG’s - BavalpreetSinghh - https://blog.stackademic.com/ndcg-vs-mrr-ranking-metrics-for-information-retrieval-in-rags-2061b04298a6 (def. ndcg)

    - Geeksforgeels - Evaluation Metrics for Retrieval-Augmented Generation (RAG) Systems - https://www.geeksforgeeks.org/nlp/evaluation-metrics-for-retrieval-augmented-generation-rag-systems/ (def. map)

    - Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics (TACL) - https://doi.org/10.48550/arXiv.2307.03172 (def. lost in middle - integration layer)

    - Ragas - Faithfulness - https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/ (def. metrica)

    - IBM – Cos’è la Retieval Augmented Generation (RAG)? - By Ivan Belcic - https://www.ibm.com/it-it/think/topics/retrieval-augmented-generation

    - Google Cloud - Che cos'è la Retrieval-Augmented Generation (RAG)? - https://cloud.google.com/use-cases/retrieval-augmented-generation?hl=it

    - Databricks - Workflow RAG end-to-end: come funziona la Retrieval Augmented Generation - https://www.databricks.com/it/blog/rag-workflow 

    Esempio di un prompt creato nell'integration layer = '''
        [System]
        You are an expert and precise virtual assistant. Your task  is to answer the user's question based EXCLUSIVELY on the information provided in the "Context" area.

        Fundamental Rules:
            - If the answer isn't found in the context, reply verbatim: "I don't have the sufficient information to answer this question based on the available documents."
            - Don't invent, extrapolate or use external or prior knowledge that isn't explicitly stated in the context.
            - Cite the source or the name of the reference document in square brackets (e.g., [Document_N]) when starting a specific fact.
            - Maintain a professional, clear and concise tone.

        [User Question]
        {{ user_query }}

        [Context]
        {% for chunk in reordered_chunks %}

            --- Start of Document: {{ chunk.metadata.source }} ---
            {{ chunk.page }}
            # ...
            # ...

            --- End of Document ---

        {% endfor %}

        [User Question - repeated]
        {{ user_query }}

        [Answer]
    '''

    Visualizzazione grafica del flusso rag:
    
    indexing
    ------------------------------------------------------------
    KB -> chunking -> embeddings -> memorizzazione nel Vector DB
                                            |
                    -------------------------             
                    |                                                                                            retrieval 
                ----------------------------------------------------------------------------------------------------------
       query -> Vector DB -> similarity search -> recupero chunk pertinenti (-> reranker) -> contesto sui chunk recuperati
        |                                                                                            |
        ----------------------------------------------------------------------------------------------
                    |
                prompt finale
                    |                              generation
                    -----------------------------------------
                    LLM -> output analysis (-> output handler) -> risposta finale

# RAG evaluation

    - Zheng, L., Chiang, W. L., Sheng, C., Li, S., Zhuang, Z., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Xing, E. P., Gonzalez, J. E., Stoica, I., & Zhang, H. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems, 36, pp 1-22 - https://arxiv.org/abs/2306.05685 (def. llmaaj)

    - Amplitude - LLM as a Judge Evals: How They Work and When to Trust Them - https://amplitude.com/explore/product/llm-as-a-judge (def. LLMaaj)

    - Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024). RAGAs: Automated Evaluation of Retrieval Augmented Generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pp 150-158 - https://arxiv.org/abs/2309.15217 (def. ragas)

    - Ragas - https://www.ragas.io/ (doc.)

    - Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp 338-354 - https://aclanthology.org/2024.naacl-long.20/ (def. ares)

    - ARES Documentation - ARES: An Automated Evaluation Framework for RAG Systems - https://ares-ai.vercel.app/ (doc.) (+ https://github.com/stanford-futuredata/ARES)

    - Medium - Evaluating RAG Models with ARES: A Scalable Approach to Automated Retrieval and Generation Scoring - Rupak (Bob) Roy - II - https://bobrupakroy.medium.com/evaluating-rag-models-with-ares-a-scalable-approach-to-automated-retrieval-and-generation-scoring-02ac9ea047ac (def. ares)

    - DeepEval - RAG Evaluation - https://deepeval.com/guides/guides-rag-evaluation (def. deepeval)

    - DeepEval - The LLM Evaluation Framework - https://deepeval.com/ 

    - Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment - https://doi.org/10.48550/arXiv.2303.16634 (def. geval)
    
    - QASkills.sh - TruLens LLM Evaluation Framework Complete Guide 2026 - https://qaskills.sh/blog/trulens-llm-evaluation-framework-guide (def. trulens)

    - Trulens - Evaluate and trace AI agents. Move from vibes to metrics - https://www.trulens.org/

    - What is Arize Phoenix? AI Observability and Evaluation - https://arize.com/docs/phoenix (+ Evaluation https://arize.com/docs/phoenix/evaluation/llm-evals) (def. phoenix)



    Esempio di prompt (LLM as a Judge) per valutazione "Faithfulness" = '''
        You will be provided with a "RAG_context" and a "system_answer". 
        Your task is divided in two phases.

        Step 1: Decompose
        Split "system_answer" into a list of individual atomic factual data statements. Each statement must express exactly one fact and be understandable on its own.

        Step 2: Verify each complaint
        For each claim, judge whether it can be inferred from or is supported directly by "RAG_context" alone, not from outside or prior knowledge. Use this scale:
            1. Directly contradicts the context or is entirely manufactured / not present in it.
            2. Partially supported: Context-related, but adds unsupported details or overstate what the context says.
            3. Supported, but requires free or indirect inference from the context.
            4. Directly and clearly supported by an explicit statement in context.
        
        From each claim rated 1 or 2, cite the specific part of "system_answer" is responsible and briefly explains why it isn't supported, before giving the score.
        Then calculate an overall loyalty score like: (numbers of complaints assessed 3 or 4) / (total number of complaints). 

        Context: {{ RAG_context }}
        Answer: {{ system_answer }}
    
        Output format: {
            "statements": [
                {
                    "statement": "...",
                    "score": 1 - 4,
                    "justification": "..."
                },
                ...
            ],
            "fidelity_score": 0.0 - 1.0
        }

    '''

    - DeepEval vs RAGAS vs TruLens: scegli il tuo stack di valutazione RAG - by Sebastian Mondragon - https://particula.tech/blog/deepeval-vs-ragas-vs-trulens-rag-evaluation-stack (confronto a 3)

    Tabella comparativa

                   RAGAS                    |                   ARES                     |               DeepEval                 |                  Trulens

Origine |      Es et. al. 2024                          Saad-Falcon et al 2024                         Confident AI                              TruEra --> Snowflake

Approccio | LLM as a judge (prompting          Fine tunend judge su dati sintetici, ppi        Pytest native: metriche, poi                  Tracing (OpenTelemetry), feedback function                      
            diretto, media a doppio judge)     intervalli di confidenza                        asserzioni pass su soglia                      agganciate all'esecuzione del codice

Cosa misura | Faithfulness, answer relevancy   context relevance, answer faithfulness,   stesse metriche dei due precendenti, 50 metriche     rag triad: context relevance, groundedness,
              context (precison, recall)        answer relevance                           (agents, tool-use, multi-turno, sicurezza)           answer relevance

Necessita di  | solo per context recall         piccolo set per calibrare i judges         a seconda della metrica: (contextual                funzionamento anche senza risposte ideali
ground truth? |                                                                            precision e recall si)

Output |       4 valori appartenti [0, 1]         score, intervallo di confidenza         pass/fail su soglia (e score, motivazione)          traccia completa (span by span), score per span

Collocamento | segmentazione (chunk size,       validazione metodologica/statistica              gate regressione in ci/cd                    monitoring e debug in produzione
                embedding, reranker)            risultati

Pro |        rapido, zero attrito         rigore statistico, meno dipendenza da un           ampiezza, integrazione come test unitario      osservability: voto e dove nella pipeline si è rotto
                                           judge rumoroso                                    nel test codebase esistente

Contro |    Varianza giudice singolo        annotazione umana iniziale, +complesso        no deep tracing, resta a livello di set statico        sviluppo lento  

Licenza |    open source (apache 2.0)                ricerca accademica                                        apache 2.0                                   mit

                   |                                          |                                            |                                                 |
                   -------------------------------------------------------------------------------------------------------------------------------------------
                        |
                        non distinguono un contesto fedelmente riportato da uno fattualmente falso

# Logica classica

    - Argomentare - I principi della logica classica - http://www.argomentare.it/logica/i-principi-della-logica

    - Wikipedia - Base di Herbrand - https://it.wikipedia.org/wiki/Base_di_Herbrand

    - Programmazione Logica -- Lezione 1: Sintassi e Semantica Dichiarativa - https://person.dibris.unige.it/moggi-eugenio/LP/Palamidessi/PL-lezione1.html

    Esempio di insieme di fatti:
        capitale(Brasile, Brasila).
        capitale(Germania, Berlino).
        capitale(Francia, Parigi).
        capitale(Brasile, Rio).
        capitale(Francia, Parigi).

        auto(Mario, Rossa).
        auto(Giovanni, Verde).

        Universo di Herbrand: {Brasile, Brasilia, Germania, Berlino, Francia, Parigi, Rio, Mario, Rossa, Giovanni, Verde}
        Predicati: {capitale/2, auto/2} 
        Base di Herbrand: {universo x predicati}
        
    Con clausole di Horn - semantica di Herbrand:
        La KB è un database chiuso: Vale l'Assunzione di Mondo Chiuso (CWA). Tutto ciò che non è estratto o deducibile è falso.
        Le metriche di qualità non sono stime teoriche. Diventano query algoritmiche. Si può contare quante istanze della Base di Herbrand violano i vincoli.
        Efficienza logica: clingo non deve fare congetture su elementi sconosciuti. Lavora solo sui fatti concreti derivati dal testo, rendendo il calcolo del "punteggio di qualità" (Coerenza, Accuratezza, Consistenza) estremamente veloce e deterministico.

        
