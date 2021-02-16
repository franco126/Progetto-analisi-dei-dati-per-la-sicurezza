# Progetto analisi dei dati per la sicurezza
Piccolo progetto di machine learning realizzato per l'esame di Analisi dei dati per la sicurezza.
L'obiettivo principale è quello di apprendere 3 pattern di classificazione in grado di identificare, se un'applicazione android è un malware a partire dall'analisi del traffico di rete generato. L'addestramento prevede l'utilizzo della Stratified Cross Validation.

## Steps principali per la realizzazione del progetto
1. Calcolo dei migliori parametri di configurazione per gli algoritmi di classificazione random forest e K nearest Neighbours 
   - metrica utilizzata la scelta dei parametri: fscore
2. Addestramento random forest su Training Set
3. Addestramento random forest sulle (10) Top componenti principali
   - applicata PCA su Training Set
4. Stacking
   - Creazione di un nuovo dataset le cui variabili indipendenti (attributi) sono le predizioni di classe dei pattern precedenti
   - Algoritmo K Nearest Neighbours addestrato sul dataset di predizioni
   - Il pattern risultante prendo il nome di Stacker
5. Valutazione dell'accuratezza dei pattern appresi utilizzando il Testing Set
   - metrica di riferimento per la valutazione: fscore
