# Esperimento analisi di Herbrand

Flusso dell'esercitazione:
    - chunk testuali
    - LLM (con schema Pydantic vincolato)
    - fatti strutturati -> normalizzazione -> atomi ASP con fonte inclusa
    - regole ASP -> Clingo
    - occorrenze/non_coperti/discordanze
    - metriche numeriche (accuratezza, compattezza, copertura)


## Bug critici

    - Ridondanza: contare a coppie vs occorrenze (cresce in modo combinatorio invece che linearmente)
    - Union Pydantic: unire i fatti (AutoFact | ColoreFact | ...) porta a mescolare campi diversi, meglio uno schema unico (tipo + argN generici)
    - Normalizzazione di genere necessaria: possibilità di falsi conflitti (es: bianca/bianco, nera/nero, ="case sensitive")
    - Specializzazione del prompt: richieste troppo precise all'LLM producono artefatti, così come richieste troppo "leggere" ne producono altri

## Lista delle osservazioni

    - La metrica "abita" nella definizione di equivalenza decisa a monte. Si devono decidere regole (es. in python) per stabilire che due "entità"
        sono la stssa persona ("Citroen" != "citroen", "nera" != "nero")
    - Discordanza è nella regola, non nel dato. (es. ASP non conosce che Parigi/Bordeaux è un conflitto, è la regola che lo decide, 
        colore(X, Y, S1), colore(X, Z, S2), Y != Z, S1 < S2)
    - L'atomo in se, può non essere in grado di rilevare conflitti (es: 
        orario(negozio, sabato, 10:00, 13:00) vs orario(negozio, sabato, 10:00, 16:00) può essere un vero conflitto, o può avere validità diverse
        che l'atomo attuale non riesce a separare -> Servono più argomenti?)
    - Fonte di variabilità: chunk identico, modello identico, temperature: 0 -> risultati diversi a seconda delle parole del prompt (un chunk 
        può sparire completamente in un run) -> Quanto sono robuste le metriche rispetto alla variabilità dell'estrazione stessa?
    - Diversi livelli, ognuno con il suo errore: text, LLM's interpretation, Logic representation. 
        ASP può essere perfetto, ma la metrica errata perché l'errore può nascere "prima".
    - Prompt: vincoli sulla struttura (pydantic) - meglio stringenti, vincoli sul contenuto (richieste fatte direttamente all'LLM) - meglio toglierli 
        dal prompt e "spostarli" verso python, per un maggior controllo
    - Approccio logic-based vs embedding-based: non meno "decisioni" - (schema dei predicati, dict di normalizzazione, scelta del modello) 
        vs (threshold cosine sim.); ogni passaggio, un artefatto analizzabile (white-box), rispetto a similarity=0.89 (black-box)
    . Mondo chiuso: ASP -p = non derivabile, non "falso" (concetto di copertura)

## Questions

    - Chi decide le equivalenze?
    - Quali differenze tra fatti portano a vere discordanze vs cambiamenti legittimi?
    - Robustezza delle metriche risptto alla variabilità dell'estrazione LLM?
    - Misurare la correttazza dell'estrazione? 
    - Copertura minima (def)?
    - Quantità di semantica da delegare all'LLM vs quanta impostata dal sistema (schema, regole)?
    - Come mantenere la provenienza dei fatti in modo utile per la fondatezza?

    - La versione embedding-base non è in grado di rilevare la vera copertura dei topic e molto dipendente dalla threshold
    - Nella versione logic-based tutto si basa sull'estrazione llm-guided dei fatti