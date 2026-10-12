# Possibile estensione: Explainable AI

## Giudice LLM - motore di Explainability

   - La matematica veloce trova il punto nello spazio latente (il dove), e l'LLM spiega all'utente in linguaggio naturale il motivo del conflitto (il perché)
   Questo approccio straccia l'albero di derivazione di ASP, che sputerebbe fuori solo un freddo ed esoterico conflict(daily_upi_limit, upi, 100000, 250000)


## Proiezioni Geometriche Guidate (SVD / PCA Feature Loading)
   
   - XAI geometrica consiste nell'analizzare la matrice dei pesi (loadings) delle componenti principali.
   
   Estrazione matematica delle parole (token) che hanno il peso maggiore nello spostare i chunk lungo l'asse PC1 o PC2. Se si scopre che l'asse PC1 è 
   dominato da parole (es. branch, atm, address) -> spiegare all'utente che quella zona dello spazio 3D identifica l'area tematica della rete fisica.
   Spiegabilità intrinseca della riduzione dimensionale

## Integrated Gradients / SHAP per i Transformer
   
   - SHAP (SHapley Additive exPlanations) / Integrated Gradients applicati ai modelli di embedding
   
   - Spengono una alla volta le parole del chunk di partenza (es. mascherano la parola "India" o "UPI") -> calcolo di quanto si sposta il vettore nello spazio latente. Se togliendo la parola "UPI" il vettore si sposta tantissimo, il sistema dimostra matematicamente che l'attenzione del modello era focalizzata su quel termine.


