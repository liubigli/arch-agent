# Benchmark tool-use scena4_VAL

## Report tecnico per presentazione aziendale

Data del report: 17 settembre 2026  
Run analizzato: `benchmark_outputs_20260916`  
Scena: `scena4_VAL`  
Campione: 60 domande per modello  
Modelli: `llama3.1`, `gemma4:31b`, `qwen3.5`, `command-r`, `gpt-oss:20b`

## Executive summary

Il benchmark conferma che tutti e cinque i modelli sanno attivare gli strumenti della scena, ma con differenze rilevanti nella selezione del tool, nella lettura del risultato e nella produzione della risposta finale.

`gemma4:31b` mostra il routing migliore: 55,8% di selezioni ottimali e 86,5% almeno accettabili. La lettura manuale dei raw indica inoltre che quasi tutte le sue penalizzazioni di grounding sono falsi positivi del validatore. Il limite principale e la latenza media di 14,38 secondi.

`llama3.1` e il piu veloce, con 1,31 secondi per domanda, e ottiene il grounded-rate automatico piu alto, 86,7%. Tuttavia questo dato nasconde due risposte senza tool e un errore grave sul materiale delle colonne, indicato come marmo anziche breccia policroma. Il routing almeno accettabile e il piu basso del gruppo, 69,2%.

`gpt-oss:20b` offre il miglior compromesso alternativo: 2,97 secondi, nessuna risposta vuota, 75% di routing almeno accettabile e buona gestione delle classi assenti. Il grounded-rate automatico del 70% e fortemente sottostimato da falsi positivi del valutatore.

`qwen3.5` raggiunge un buon 84,6% di routing almeno accettabile, ma lascia sette risposte finali vuote dopo aver eseguito correttamente i tool. Il problema e quindi nel completamento del ciclo agentico, non soltanto nella selezione dello strumento.

`command-r` completa tutte le risposte, ma usa spesso un formato annidato non conforme per gli argomenti (`tool_name` e `parameters`). Questo causa conteggi pari a zero per classi presenti e numerose allucinazioni sulle classi assenti.

## Contesto della scena

Il run del 16 settembre usa il seguente stato della pipeline:

| Dato | Valore |
|---|---:|
| Oggetti | 33 |
| Relazioni | 262 |
| Nodi spatial graph | 33 |
| Archi spatial graph | 178 |
| `column` | 6 |
| `door_window` | 8 |
| `floor` | 1 |
| `moldings` | 6 |
| `vault` | 1 |
| `wall` | 11 |
| Classi assenti | `arch`, `other`, `roof`, `stairs` |

Le valutazioni devono riferirsi a questi numeri. I benchmark precedenti riportavano 34 oggetti e non sono direttamente confrontabili, perche prodotti con una diversa versione della pipeline.

## Risultati automatici

| Modello | Tool call medie | Latenza media | Zero tool | Risposte vuote | Errori runtime | Grounded automatico | Routing ottimale | Routing accettabile o migliore |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| llama3.1 | 0,97 | 1,31 s | 2 | 0 | 0 | 86,7% | 40,4% | 69,2% |
| gemma4:31b | 1,28 | 14,38 s | 0 | 0 | 0 | 83,3% | 55,8% | 86,5% |
| qwen3.5 | 1,23 | 3,27 s | 0 | 7 | 0 | 68,3% | 55,8% | 84,6% |
| command-r | 1,27 | 4,31 s | 0 | 0 | 0 | 66,7% | 46,2% | 76,9% |
| gpt-oss:20b | 1,07 | 2,97 s | 0 | 0 | 0 | 70,0% | 48,1% | 75,0% |

Il grounded-rate non deve essere letto come accuratezza finale: il validatore presenta errori sistematici descritti più avanti.

## Uso dei tool

| Modello | Tool piu usato | Chiamate totali | Osservazione |
|---|---|---:|---|
| llama3.1 | `find_relationships` (15) | 58 | Routing rapido, ma due risposte senza tool |
| gemma4:31b | `find_relationships` (21) | 77 | Migliore copertura e uso consistente dei tool semantici |
| qwen3.5 | `find_relationships` (16) | 74 | Buon routing, sette risposte finali mancanti |
| command-r | `list_relationships` (20) | 76 | Argomenti spesso incapsulati in uno schema errato |
| gpt-oss:20b | `find_relationships` (14) | 64 | Routing compatto e buona gestione multi-classe |

Il set analizzato non richiama `find_pattern`, `find_focal_points` o `discover_functional_areas`. Questo conferma che il benchmark usa il set ristretto atteso.

## Analisi per modello

### gemma4:31b

Punti di forza:

- Usa correttamente `count_objects_by_class` con `semantic_labels` per il conteggio multi-classe.
- Distingue correttamente le classi presenti dalle classi con conteggio zero.
- Usa `get_object_semantic_details` o annotazioni CSV per funzione, materiale e tipologia.
- Gestisce correttamente le classi assenti senza creare istanze inesistenti.
- Identifica le due aperture in legno come `door_window_3` e `door_window_7`.
- Non produce risposte vuote o errori di protocollo.

Criticita:

- Latenza di 14,38 secondi, quasi undici volte quella di Llama.
- In alcune domande di elenco usa `list_objects` ma viene classificato come subottimale dal benchmark, probabilmente perche il riferimento si aspetta un tool piu specifico. Il risultato ottenuto e comunque corretto.
- Per le relazioni dei wall effettua cinque paginazioni. La risposta e accurata, ma costosa.
- Una risposta non rispetta la lingua attesa.

Revisione delle dieci risposte dichiarate ungrounded:

- Q5, Q6 e Q7 sono corrette: il validatore penalizza la semplice presenza testuale di classi con conteggio zero.
- Q15, Q28, Q34 e Q54 elencano correttamente le istanze; il parser numerico confonde i numeri degli ID o il conteggio della classe con il totale della scena.
- Q24 riporta correttamente che dieci wall hanno annotazioni funzionali e che `wall_2` non le possiede.
- Q56 identifica correttamente due aperture in legno; il validatore interpreta erroneamente il numero 2 come totale degli oggetti della scena.

Conclusione: il valore automatico dell'83,3% sottostima sensibilmente la qualita reale del run. Tra i cinque modelli e quello con il comportamento piu affidabile sui raw.

### llama3.1

Punti di forza:

- Migliore latenza: 1,31 secondi.
- Risponde nella lingua della domanda in tutti i 60 casi.
- Buona gestione delle domande relazionali e delle classi assenti nella maggior parte dei casi.
- Identifica correttamente `door_window_3` e `door_window_7` come aperture in legno.

Criticita reali:

- Q5 chiede l'inventario completo, ma il modello estrae soltanto i sinonimi `colonna` e `tetto` e chiama `count_objects_by_class` con quelle due classi. La risposta non contiene l'inventario richiesto.
- Q7 legge correttamente quasi tutte le classi ma omette `moldings` nella risposta finale.
- Q16 non esegue il tool, simula verbalmente la chiamata e inventa che le colonne siano in marmo. Il CSV indica breccia policroma.
- Q24 non esegue il tool e fornisce una descrizione generica della funzione dei muri. La risposta e plausibile, ma non verificata.
- Il routing ottimale e solo 40,4%; molte domande su elenco e annotazioni usano strumenti accettabili ma non preferiti.

Falsi positivi del validatore:

- Q6 riporta correttamente la distribuzione e le classi assenti.
- Q38 elenca relazioni reali della volta; il raw mostrato non usa la relazione `contains` contestata dal validatore.
- Q54 elenca correttamente otto `door_window`.
- Q56 riporta correttamente le due aperture in legno.

Conclusione: il grounded-rate automatico dell'86,7% e troppo ottimistico rispetto al comportamento critico su Q16 e troppo severo su alcuni conteggi. Llama resta la migliore baseline prestazionale, ma non la piu affidabile semanticamente.

### gpt-oss:20b

Punti di forza:

- Usa correttamente gli input multi-classe, ad esempio `semantic_labels=["column", "roof"]`.
- Gestisce in modo esplicito l'assenza di `roof` e `arch`.
- Produce tutte le 60 risposte senza errori runtime.
- Mantiene una latenza media contenuta di 2,97 secondi.
- Usa un solo tool nella maggior parte delle domande.

Criticita:

- Routing ottimale al 48,1%, inferiore a Gemma e Qwen.
- Alcune domande descrittive attivano tool piu estesi del necessario.
- Q24 afferma che tutti gli undici wall sono annotati come pareti perimetrali portanti; questa generalizzazione va confrontata con il fatto che altri modelli rilevano un'annotazione mancante per `wall_2`.
- Q26 sintetizza molte relazioni e usa conteggi aggregati che meritano un controllo diretto sul tool output prima di essere presentati come misura.

Gran parte delle diciotto penalizzazioni automatiche e falsa:

- Q4, Q6 e Q7 indicano esplicitamente `arch`, `roof` e `stairs` con conteggio zero.
- Q15, Q22, Q28, Q34 e Q54 elencano correttamente le istanze.
- Q41-Q44 dichiarano chiaramente che `roof` e assente.
- Q49 dichiara correttamente che `arch` e assente.
- Q56 identifica correttamente due aperture in legno.

Conclusione: il 70% automatico non rappresenta la qualita osservata. GPT-OSS e il miglior compromesso dopo Gemma quando velocita e correttezza del routing hanno peso simile.

### qwen3.5

Punti di forza:

- Routing ottimale al 55,8%, pari a Gemma.
- Routing almeno accettabile all'84,6%.
- Buona selezione di `get_object_semantic_details` e `find_relationships`.
- Corretta gestione delle classi assenti nelle risposte effettivamente prodotte.
- Corretta identificazione delle aperture in legno.

Criticita reali:

- Sette risposte finali vuote: Q1, Q4, Q7, Q12, Q15, Q20 e Q34.
- In tutti questi casi sono presenti tool call; il fallimento avviene dopo il ritorno del tool.
- Q24 dichiara dieci wall invece di undici. La probabile origine e la disponibilita di annotazioni CSV per dieci oggetti, scambiata per conteggio geometrico della classe.
- Q32 presenta un riepilogo strutturale molto assertivo; deve distinguere chiaramente relazione del grafo e descrizione CSV.

Falsi positivi del validatore:

- Q5 e Q6 riportano correttamente 33 oggetti distribuiti in sei classi.
- Q28 elenca correttamente `floor_0`.
- Q44, Q47, Q49 e Q51 gestiscono correttamente classi assenti.
- Q54 e Q56 riportano correttamente otto aperture e due elementi in legno.

Conclusione: il modello comprende bene gli strumenti, ma il 11,7% di risposte vuote lo rende inadatto a esecuzioni non supervisionate finche non viene risolto il passaggio tool-result -> final answer.

### command-r

Punti di forza:

- Completa tutte le risposte.
- Nessun errore runtime o mismatch linguistico.
- Numero medio di chiamate ragionevole, molto migliore rispetto ai vecchi benchmark.

Criticita reali:

- Produce frequentemente argomenti come `{"tool_name": ..., "parameters": ...}` invece dei parametri diretti previsti dallo schema.
- Q14, Q21, Q27, Q53 e Q58 restituiscono zero per classi presenti proprio a causa di questo formato.
- Q38 attribuisce alla volta tutte le 262 relazioni della scena.
- Q40 inventa relazioni per `roof`, classe assente.
- Q46 inventa relazioni per `arch`, classe assente.
- Q49 crea l'oggetto inesistente `arch_0`.
- Q51 inventa relazioni per `stairs`, classe assente.

Falsi positivi del validatore:

- Q4 elenca correttamente le classi assenti.
- Q5 sembra fornire correttamente inventario e conteggi; il validatore confonde il numero di istanze di una classe con il totale.
- Q15, Q22, Q28, Q34 e Q54 elencano correttamente gli oggetti.
- Q42 e Q44 rispondono correttamente che non esiste evidenza o oggetto `roof`.
- Q26 usa la parola inglese “other” in senso naturale, non come classe semantica `other`.

Conclusione: il problema principale non e la conoscenza architettonica, ma l'aderenza allo schema degli argomenti. Serve verificare la compatibilita del formato tool-call di Command-R con l'adapter Ollama.

## Ambiguita e cause degli errori

### Classe assente menzionata in una negazione

Il validatore tratta spesso la parola `roof` come prova che il modello ne dichiari la presenza. Frasi come “roof: 0” o “roof non e presente” vengono quindi marcate erroneamente come `absent_class_claimed_present`.

Correzione consigliata: il controllo deve riconoscere conteggio zero, negazioni e formule come “classe assente” prima di emettere il flag.

### Conteggio della classe contro totale della scena

Il validatore interpreta frequentemente “8 door_window” come affermazione “8 oggetti totali”. Lo stesso accade con “2 aperture in legno” e con elenchi numerati.

Correzione consigliata: associare ogni numero al nome della classe più vicino e controllare il totale soltanto quando compaiono espressioni come “totale”, “in tutto” o `total objects`.

### Oggetti geometrici contro righe CSV annotate

La scena contiene undici wall, ma il CSV puo avere dieci corrispondenze. Dire “dieci wall annotati” e corretto; dire “la scena contiene dieci wall” e errato. Qwen confonde i due insiemi nella Q24.

### Ruolo semantico contro relazione osservata

Una classe puo essere definita strutturale, ma questo non dimostra che una specifica istanza supporti un'altra. `above`, `near` e `adjacent_to` non devono essere trasformate automaticamente in `supports`.

### Roof contro vault

La scena contiene una `vault` e nessun `roof`. Anche se la volta svolge una funzione di copertura, non deve sostituire la classe richiesta. La risposta deve prima dichiarare `roof=0`, poi eventualmente aggiungere una nota separata sulla volta.

### Tool eseguito contro tool simulato nel testo

Llama Q16 scrive che chiamera `get_object_annotation`, ma il raw contiene zero chiamate. La frase del modello non equivale a un'esecuzione e non deve essere considerata evidence.

### Risposta vuota dopo una chiamata valida

Qwen completa la selezione e l'esecuzione del tool ma non emette il messaggio finale. Questo va classificato come failure del ciclo agentico, non come errore di comprensione della classe o del tool.

### Argomenti annidati non conformi

Command-R inserisce il payload vero dentro `parameters`. Se il wrapper non normalizza questo formato, il tool riceve i default e restituisce zero o dati globali. Il modello sembra aver scelto il tool corretto, ma la chiamata effettiva e semanticamente vuota.

## Valutazione del validatore automatico

Il validatore e utile per individuare rapidamente risposte sospette, ma non e ancora adeguato come metrica aziendale autonoma.

Problemi osservati:

- falsi positivi sulle classi assenti nominate in forma negativa;
- confusione tra conteggio di classe e totale degli oggetti;
- confusione tra numeri negli ID e conteggi;
- flag su una relazione non visibile nel raw esaminato;
- incapacita di distinguere “other” come parola comune dalla classe `other`;
- nessuna penalizzazione sufficientemente forte per alcune risposte plausibili ma prive di tool.

Di conseguenza, i grounded-rate automatici non devono essere usati per ordinare i modelli senza revisione dei raw.

## Classifica ragionata

### Qualita complessiva delle risposte

1. `gemma4:31b`: migliore affidabilita osservata, completa e semanticamente disciplinata.
2. `gpt-oss:20b`: buon equilibrio tra accuratezza, completezza e velocita.
3. `llama3.1`: eccellente velocita, ma errori gravi senza tool su materiale e funzione.
4. `qwen3.5`: buon routing, penalizzato dalle sette risposte vuote.
5. `command-r`: completo, ma formato tool-call incompatibile e classi assenti inventate.

### Efficienza

1. `llama3.1`: 1,31 s.
2. `gpt-oss:20b`: 2,97 s.
3. `qwen3.5`: 3,27 s.
4. `command-r`: 4,31 s.
5. `gemma4:31b`: 14,38 s.

### Routing dei tool

1. `gemma4:31b`: 86,5% accettabile o migliore.
2. `qwen3.5`: 84,6%.
3. `command-r`: 76,9%, ma con argomenti spesso malformati.
4. `gpt-oss:20b`: 75,0%.
5. `llama3.1`: 69,2%.

## Raccomandazioni

1. Usare Gemma come baseline di qualita e Llama come baseline di latenza.
2. Considerare GPT-OSS come candidato principale per il compromesso produzione/prestazioni.
3. Correggere il validatore prima di usare il grounded-rate in presentazioni decisionali.
4. Rendere equivalenti `semantic_label` e `semantic_labels` oppure rimuovere il parametro legacy per evitare ambiguita.
5. Validare gli argomenti annidati di Command-R prima dell'esecuzione e restituire un errore esplicito invece di applicare valori di default.
6. Aggiungere un controllo che obblighi una risposta finale dopo ogni tool result, per Qwen.
7. Marcare come non grounded qualsiasi materiale o funzione dichiarati senza tool, anche se plausibili.
8. Salvare per ogni run commit Git, toolset esposto, versione del prompt, soglia spaziale e hash del CSV.
9. Ripetere ogni modello almeno tre volte per misurare la variabilita.
10. Estendere la validazione ad altre scene, incluse scene senza CSV e scene contenenti davvero `roof`, `arch` e `stairs`.

## Conclusione aziendale

Il sistema ha raggiunto una buona maturita nel routing degli strumenti: tutti i modelli, salvo specifici fallimenti di completamento, interrogano il contesto della scena invece di rispondere esclusivamente per conoscenza generale. La differenza competitiva si sposta ora dalla semplice capacita di chiamare un tool alla capacita di scegliere quello giusto, passare argomenti validi, distinguere dati geometrici e CSV e trasformare il risultato in una risposta fedele.

Gemma fornisce oggi la qualita più consistente; Llama garantisce la migliore velocita ma richiede maggiore disciplina sul grounding; GPT-OSS e il compromesso più promettente; Qwen necessita di stabilizzazione del ciclo finale; Command-R necessita di compatibilita con lo schema delle chiamate. Il prossimo intervento con il maggiore ritorno non e modificare il modello, ma migliorare validatore e adapter dei tool, perche entrambi alterano sensibilmente la misura finale.
