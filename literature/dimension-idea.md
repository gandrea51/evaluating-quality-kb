# Copertura tematica
    = i topic che dovrebbero essere presenti, sono effettivamente rappresentati?

    Embeddings 
    Clustering
    (con tassonomia: embedding anche degli attesi -> capire topic che "hanno poco" + "mancano completamente")
    Analisi della distribuzione 
    Dimensionality reduction -> Nuvola di punti -> (per human comprehension)

    --> soluzione possibile: traduzione in fatti/predicati fol - inferenza asp


# Ridondanza e quali Duplicazione
    = ho tanti documenti, che dicono sostanzialmente la stessa cosa.

    Embeddings
    Vectorial index 
    Nearest Neighbors
    cosine-similarity > threshold
    grafo di similarità
    union find / connected component

    --> soluzione possibile: traduzione in fatti/predicati fol - inferenza asp


# Freschezza e Obsolescenza
    = quanto è vecchia la conoscenza che viene recuperata?

    sum (retrieval_count * age) / sum (retrieval_count)
    -- media ponderata dell'età dei documenti --
    -- significato: in media i documenti che il sistema sta servendo hanno quasi X anni --
    
    -> percentuale di rischio: def thresh critica di veccchiaia (5y) -> quanti recuperi vanno a finire su doc che superano la thresh rispetto al totale

    (documenti > thresh) & (recuperati almeno X volte)
    
    -> alto retrieval_count & contenuto vecchio => obsolescenza & mancanza di alternative (bassa copertura su quel topic)

    --> esercizio: simulate freschezza

    ! (piccola nota): ragas per misurare l'effetto della degradazione causato dai dati obsoleti (non l'età, ma le conseguenze)


# Recuperabilità
    = i contenuti nella kb sono realmente raggiungibili dal retriever o ci sono parti "morte"?

    mai recuperati / totali

    recuperati almeno una volta / totali

    distribuzione frequenza di recupero

    -> Topic A,B,C presenti (=> ottima copertura) & Retriever usa sempre A (=> recuperabilità pessima) 


# Coerenza interna
    
    ! (nota da esperimento)
    L'embedding space è efficace per individuare contenuti che trattano lo stesso oggetto informativo. 
    Tuttavia la similarità semantica non consente di distinguere in modo affidabile relazioni di entailment da relazioni di contradiction.   
    Pertanto la ricerca dei vicini semantici può costituire una fase di filtraggio preliminare, da completare successivamente tramite tecniche di Natural Language Inference o mediante LLM.