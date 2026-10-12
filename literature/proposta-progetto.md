## PROPOSTA DI TESI

## Definizione di un modello di calcolo degli indicatori di qualità della conoscenza in una Knowledge Base aziendale

Caso di studio: la piattaforma CK – Company Knowledge

## 1. Contesto

CK – Company Knowledge è una piattaforma di knowledge management aziendale che acquisisce, indicizza e rende interrogabile in linguaggio naturale il patrimonio documentale di un’organizzazione. I documenti vengono segmentati in unità informative (chunk), rappresentati come vettori semantici in uno spazio di embedding e organizzati in Knowledge Base distinte per area di competenza, sulle quali operano assistenti conversazionali basati su architettura RAG (Retrieval-Augmented Generation).

La qualità delle risposte prodotte da un sistema di questo tipo dipende in misura determinante dalla qualità della base di conoscenza sottostante: un contenuto obsoleto, ridondante, contraddittorio o mai raggiunto dal motore di recupero degrada le prestazioni del sistema in modo silenzioso, senza generare errori applicativi. Ad oggi la piattaforma dispone di metriche di utilizzo e di consumo, ma non di un impianto strutturato di misura della qualità intrinseca del sapere archiviato. Il presente lavoro di tesi si colloca esattamente in questo spazio.

## 2. Obiettivo formativo

L’attività di tesi ha l’obiettivo di far acquisire al tesista le competenze metodologiche e tecniche necessarie per affrontare, con approccio ingegneristico e verificabile, il problema della misura della qualità dell’informazione all’interno di un sistema di knowledge management basato su intelligenza artificiale generativa. Il percorso è concepito affinché il candidato impari a tradurre un’esigenza aziendale espressa in termini qualitativi – « la nostra conoscenza è affidabile? » – in un modello formale, calcolabile su dati reali, validato sperimentalmente e comunicabile a interlocutori non tecnici. Al termine dell’esperienza il tesista dovrà essere in grado di operare con autonomia sull’intero ciclo che va dall’analisi dello stato dell’arte alla progettazione della metrica, dalla sua implementazione alla discussione critica dei risultati ottenuti e dei limiti del modello proposto.

In particolare, il lavoro è finalizzato allo sviluppo delle seguenti competenze:

- Modellazione della qualità dell'informazione: capacità di selezionare, adattare e formalizzare dimensioni di qualità a partire dai framework di riferimento della letteratura scientifica e normativa, motivando le scelte rispetto al dominio applicativo.

- Progettazione di metriche misurabili: capacità di trasformare una dimensione qualitativa in un indicatore quantitativo dotato di definizione operativa, dominio di variazione, fonte del dato e procedura di calcolo riproducibile.


- Comprensione delle architetture RAG e della rappresentazione semantica: padronanza dei meccanismi di segmentazione, embedding, ricerca vettoriale e generazione condizionata, e della relazione fra le proprietà della base documentale e le prestazioni del sistema.

- Analisi quantitativa dei dati e validazione sperimentale: capacità di impostare un protocollo di verifica, costruire un insieme di riferimento, misurare la correlazione fra gli indicatori proposti e la qualità osservata delle risposte, e valutare la robustezza dei risultati.

- Progettazione software e trattamento dei dati su larga scala: realizzazione di un motore di calcolo efficiente su volumi documentali reali, con attenzione ai costi computazionali e alla sostenibilità dell’esecuzione periodica.

- Comunicazione dei risultati e orientamento alla decisione: capacità di sintetizzare misure eterogenee in una rappresentazione leggibile dal responsabile della conoscenza aziendale e di derivarne indicazioni concrete di intervento.

- Autonomia e metodo di ricerca: gestione autonoma della ricerca bibliografica, della pianificazione delle attività e della redazione di un elaborato tecnico-scientifico strutturato.

## 3. Obiettivo del progetto

Il progetto ha per oggetto la definizione di un modello di calcolo degli indicatori di qualità della conoscenza presente nelle Knowledge Base della piattaforma CK, la sua implementazione come motore di misura eseguibile sui dati di esercizio e la sua validazione sperimentale. Il risultato atteso è un sistema di indicatori che consenta di rispondere, in modo oggettivo e ripetibile, a domande quali: quanto è completa la copertura documentale di un’area tematica? quanta parte del contenuto è ridondante o obsoleta? quali documenti non vengono mai utilizzati dal motore di recupero? esistono informazioni fra loro contraddittorie? in quale misura le risposte generate sono effettivamente fondate sulle fonti?

## 4. Modello degli indicatori: dimensioni candidate

La tabella seguente riporta le dimensioni di qualità individuate come candidate in fase di proposta. Il perimetro definitivo, la formalizzazione delle metriche e la loro eventuale estensione costituiscono parte integrante del lavoro di tesi.

| Dimensione | Definizione | Metrica proposta | Fonte del dato |
| --- | --- | --- | --- |
| Copertura tematica | Ampiezza con cui la base documentale copre gli argomenti di pertinenza dell’area. | Distribuzione dei contenuti su una mappa tematica derivata per clustering dello spazio semantico; individuazione delle aree scoperte. | Vettori di embedding, tassonomia di area |
| Ridondanza e quasi-duplicazione | Presenza di contenuti sovrapposti che disperdono il recupero e moltiplicano le versioni della stessa informazione. | Quota di unità informative con similarità coseno superiore a una soglia calibrata; dimensione dei gruppi di duplicati. | Indice vettoriale |
| Freschezza e obsolescenza | Attualità dell’informazione, pesata sulla rilevanza effettiva del contenuto. | Età media dei contenuti ponderata sulla frequenza di recupero; quota di contenuti oltre la soglia di obsolescenza. | Metadati documentali, log di recupero |


| Recuperabilità | Effettiva raggiungibilità del contenuto da parte del motore di ricerca semantica. | Frazione di unità informative mai comparse fra i risultati; concentrazione del recupero su un sottoinsieme ristretto della base. | Log di recupero |
| --- | --- | --- | --- |
| Granularità e integrità della segmentazione | Adeguatezza delle unità informative in termini di autoconsistenza e completezza rispetto al documento di origine. | Distribuzione della lunghezza delle unità; incidenza di troncamenti e frammenti privi di contesto; completezza dell’acquisizione. | Archivio documentale, esito dell’acquisizione |
| Coerenza interna | Assenza di affermazioni fra loro incompatibili su uno stesso oggetto informativo. | Rilevazione di contraddizioni su coppie di unità ad alta similarità mediante verifica di implicazione testuale. | Indice vettoriale, modello di linguaggio |
| Fondatezza delle risposte | Grado in cui le risposte generate risultano sostenute dalle fonti effettivamente recuperate. | Quota di affermazioni verificabili sui passaggi citati, misurata su un campione di interazioni reali. | Storico delle conversazioni, fonti citate |
| Qualità percepita in esercizio | Riscontro proveniente dall’uso quotidiano della piattaforma. | Incidenza di risposte non conclusive, di riformulazioni successive e delle valutazioni esplicite degli utenti. | Log applicativi, riscontri utente |

Le misure elementari confluiscono in un indice sintetico di qualità della Knowledge Base, calcolabile per singola area e per l’intera organizzazione. La determinazione dei pesi di aggregazione non è assunta come dato, ma costituisce essa stessa una questione di ricerca: il tesista dovrà proporre e giustificare un criterio di ponderazione, verificandone la tenuta rispetto alla qualità osservata delle risposte.


## 5. Struttura del progetto

Il lavoro è articolato in sette fasi, ciascuna con un prodotto verificabile. La progressione è incrementale: ogni fase produce un risultato utilizzabile anche in caso di ridefinizione del perimetro delle fasi successive.

| Fase | Attività | Prodotto atteso |
| --- | --- | --- |
| 1. Stato dell’arte | Ricognizione della letteratura sulla qualità dell’informazione (framework multidimensionali di riferimento, standard ISO/IEC 25012 sulla qualità dei dati) e sui metodi di valutazione dei sistemi RAG. | Capitolo di rassegna critica e selezione motivata dei riferimenti adottati. |
| 2. Analisi del dominio | Studio dell’architettura di CK, del modello dati della Knowledge Base, del ciclo di acquisizione dei documenti e delle informazioni di esercizio effettivamente disponibili. | Documento di analisi con mappa delle fonti dati utilizzabili per il calcolo. |
| 3. Definizione del modello | Formalizzazione delle dimensioni di qualità e delle rispettive metriche: definizione operativa, dominio di variazione, normalizzazione e criterio di aggregazione nell’indice sintetico. | Specifica formale del modello degli indicatori. |
| 4. Implementazione | Realizzazione del motore di calcolo su dati reali, con esecuzione periodica e persistenza storica delle misure per l’analisi dell’andamento nel tempo. | Componente software funzionante e documentato. |
| 5. Validazione sperimentale | Costruzione di un insieme di riferimento di domande e risposte attese; analisi della relazione fra i valori degli indicatori e la qualità osservata; prove di sensibilità su soglie e pesi. | Rapporto sperimentale con evidenze quantitative e discussione dei limiti. |
| 6. Restituzione e interventi correttivi | Progettazione della rappresentazione degli indicatori per il responsabile della conoscenza e definizione delle azioni correttive associate a ciascuna condizione critica rilevata. | Prototipo di cruscotto e linee guida operative di intervento. |
| 7. Redazione dell’elaborato | Stesura della tesi, sistematizzazione dei risultati, individuazione dei possibili sviluppi futuri. | Elaborato finale e presentazione. |

## 6. Metodo di lavoro e strumenti

- Approccio: iterativo e guidato dall-evidenza; ogni indicatore proposto viene ritenuto valido solo dopo essere stato calcolato su dati reali e sottoposto a verifica sperimentale.

- Ambiente tecnico: piattaforma CK, base dati relazionale con estensione vettoriale, modelli di embedding e modelli di linguaggio di grandi dimensioni, ambiente di analisi in linguaggio Python.

- Dati: Knowledge Base di esercizio e informazioni di utilizzo della piattaforma, trattati nel rispetto delle politiche di riservatezza e, ove necessario, in forma anonimizzata.

- Verifica di avanzamento: riesame periodico con il referente aziendale al termine di ciascuna fase, sul prodotto atteso corrispondente.


## 7. Risultati attesi

- Un modello formale e documentato degli indicatori di qualità della conoscenza, trasferibile ad altre installazioni della piattaforma.

- Un motore di calcolo eseguibile periodicamente sulle Knowledge Base di esercizio.

- Evidenza sperimentale della relazione fra gli indicatori proposti e la qualità delle risposte prodotte dal sistema.

- Un cruscotto di sintesi e un insieme di linee guida per il miglioramento continuo del patrimonio informativo aziendale.

## 8. Prerequisiti e durata indicativa

Sono richieste conoscenze di programmazione in Python, di basi di dati relazionali e nozioni fondamentali di apprendimento automatico ed elaborazione del linguaggio naturale; la familiarità con le architetture RAG e con la ricerca vettoriale sarà acquisita nella prima parte del percorso. La durata indicativa dell’attività è stimata in sei mesi, con impegno continuativo e affiancamento del referente aziendale.
