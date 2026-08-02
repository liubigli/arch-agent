---
tags: [biblio, benchmark, hallucination, tool-calling, llm-agents]
---

# Bibliografia — Metriche di groundedness & costruzione di benchmark per agenti/tool-calling

Sommario di struttura per il vault Obsidian. Ogni voce sotto è pensata come
"scheda rapida" (titolo, riassunto, pro/contro, keyword, link PDF). Quando un
paper viene letto in modo approfondito, l'idea è creare una nota dedicata
(es. `paper - Gorilla.md`) e linkarla qui sostituendo il placeholder
`[[nota approfondita]]` con il link reale — così questo file resta l'indice
di struttura e le note di lettura vivono separate.

Le keyword sono pensate come tag Obsidian (`#keyword`) per collegare i paper
trasversalmente, indipendentemente dalla sezione in cui sono elencati qui.

---

## Parte 1 — Metriche per misurare hallucination/groundedness

### Survey of Hallucination in Natural Language Generation

- **Autori / Anno / Venue:** Ji, Lee, Frieske, Yu, Su, Xu, Ishii, Bang, Chen, Dai, Chan, Madotto, Fung — ACM Computing Surveys 55(12), 2022/2023
- **Link PDF:** https://arxiv.org/abs/2202.03629 (preprint; versione pubblicata su ACM CSUR, DOI 10.1145/3571730)
- **Riassunto abstract:** Survey fondazionale sull'hallucination nella NLG: propone una tassonomia (intrinsic vs extrinsic hallucination), rassegna metriche generali di misurazione e strategie di mitigazione, poi analizza il fenomeno per task specifico (summarization astrattiva, dialogo, QA, data-to-text, traduzione, generazione visione-linguaggio) e infine nei LLM.
- **Pro:** tassonomia chiara e ancora oggi standard di riferimento; copre sia metriche automatiche sia mitigazioni; buona bibliografia per orientarsi nel campo.
- **Contro:** pre-ChatGPT/GPT-4 nella sua prima versione (2022), quindi la parte sui LLM è meno aggiornata rispetto a survey più recenti; è una rassegna, non introduce un metodo o benchmark proprio.
- **Keyword:** #hallucination-taxonomy #survey #intrinsic-extrinsic #nlg
- **Nota approfondita:** [[nota approfondita]]

### A Survey on Hallucination in Large Language Models

- **Autori / Anno / Venue:** vari — arXiv preprint, novembre 2023
- **Link PDF:** https://arxiv.org/abs/2311.05232
- **Riassunto abstract:** Survey specifico sui LLM (non NLG in generale): propone una tassonomia aggiornata dell'hallucination nell'era LLM, analizza i fattori che la causano, rassegna metodi e benchmark di detection, poi le metodologie di mitigazione. Discute anche i limiti dei sistemi retrieval-augmented nel contrastare l'hallucination e chiude su direzioni di ricerca (incluso hallucination nei modelli visione-linguaggio).
- **Pro:** aggiornato rispetto a Ji et al.; utile ponte tra tassonomia teorica e benchmark/metriche concrete; discute esplicitamente i limiti del RAG, rilevante per il vostro caso (agente grounded su scene graph).
- **Contro:** essendo una survey ampia, resta a livello di rassegna — per i dettagli implementativi delle metriche bisogna comunque andare ai paper originali (FActScore, SelfCheckGPT, ecc.).
- **Keyword:** #hallucination-taxonomy #survey #llm #retrieval-augmented #detection-benchmarks
- **Nota approfondita:** [[nota approfondita]]

### FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation

- **Autori / Anno / Venue:** Min, Krishna, Lyu, Lewis, Yih, Koh, Iyyer, Zettlemoyer, Hajishirzi — arXiv, maggio 2023
- **Link PDF:** https://arxiv.org/abs/2305.14251
- **Riassunto abstract:** Propone FActScore: scompone un testo generato in "atomic facts" (singole affermazioni verificabili) e calcola la percentuale di quelle supportate da una fonte affidabile, invece di un giudizio binario "corretto/scorretto" sull'intera risposta. Include sia una valutazione umana su biografie generate da modelli commerciali (ChatGPT arriva solo al 58%) sia una versione automatica del metodo (retrieval + LLM) con meno del 2% di errore rispetto all'umano, usata poi per valutare 6.500 generazioni da 13 modelli recenti.
- **Pro:** metrica granulare (per-asserzione, non per-risposta intera) — è concettualmente il pattern più vicino al vostro `check_groundedness`; ha una versione automatica economica da riprodurre; libreria pubblica (`pip install factscore`).
- **Contro:** pensato per testo libero (biografie) contro una fonte testuale (Wikipedia), non per dati strutturati come un grafo di scena — va adattato per verificare asserzioni contro entità/relazioni enumerabili invece che contro un corpus testuale.
- **Keyword:** #atomic-facts #factual-precision #fine-grained-metric #automated-evaluation
- **Nota approfondita:** [[nota approfondita]]

### SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models

- **Autori / Anno / Venue:** Manakul, Liusie, Gales — arXiv, marzo 2023
- **Link PDF:** https://arxiv.org/abs/2303.08896
- **Riassunto abstract:** Metodo "zero-resource" (nessun database esterno, nessun accesso alle probabilità del modello): campiona più risposte allo stesso prompt e misura quanto sono consistenti tra loro. L'idea è che un fatto reale tende a essere ripetuto in modo simile in più campionamenti, mentre un'allucinazione tende a divergere/contraddirsi tra un campione e l'altro. Validato su biografie generate con GPT-3, con AUC-PR più alta dei baseline "grey-box".
- **Pro:** non richiede ground truth né accesso a logits/API privilegiate — funziona anche con modelli black-box dietro API; utile come "seconda opinione" indipendente dal checker deterministico basato su ground truth.
- **Contro:** costoso in termini di chiamate LLM (richiede N campionamenti extra per risposta); misura consistenza, non correttezza — un modello può essere consistentemente sbagliato (bias sistematico) e questo metodo non lo rileverebbe.
- **Keyword:** #zero-resource #sampling-consistency #black-box #self-consistency
- **Nota approfondita:** [[nota approfondita]]

### Ragas: Automated Evaluation of Retrieval Augmented Generation

- **Autori / Anno / Venue:** Es, James, Espinosa-Anke, Schockaert — arXiv, settembre 2023
- **Link PDF:** https://arxiv.org/abs/2309.15217
- **Riassunto abstract:** Framework di valutazione reference-free per pipeline RAG. Propone metriche per tre dimensioni distinte: capacità del retriever di trovare passaggi rilevanti/focalizzati, capacità del LLM di sfruttarli in modo fedele ("faithfulness" — la risposta è supportata dal contesto recuperato?), e qualità generale della generazione — tutto senza richiedere annotazioni umane di ground truth.
- **Pro:** scompone la valutazione in dimensioni distinte (retrieval vs faithfulness vs qualità) invece di un unico punteggio aggregato; reference-free, quindi economico da applicare ripetutamente durante lo sviluppo.
- **Contro:** pensato per RAG testuale (contesto recuperato = testo), va reinterpretato per il vostro caso dove il "contesto" è l'output strutturato dei tool, non un passaggio di testo libero.
- **Keyword:** #rag #faithfulness #reference-free #retrieval-evaluation
- **Nota approfondita:** [[nota approfondita]]

---

## Parte 2 — Costruzione di benchmark per tool-calling / agenti

### Gorilla: Large Language Model Connected with Massive APIs

- **Autori / Anno / Venue:** Patil, Zhang, Wang, Gonzalez — arXiv, maggio 2023 (NeurIPS 2024)
- **Link PDF:** https://arxiv.org/abs/2305.15334
- **Riassunto abstract:** Introduce Gorilla, un modello fine-tuned (basato su LLaMA) specializzato nello scrivere chiamate API corrette, e APIBench, il dataset di valutazione (API HuggingFace, TorchHub, TensorHub). Il problema centrale che affrontano è proprio l'hallucination nelle tool-call: i LLM generici tendono a inventare argomenti o usi scorretti delle API. La valutazione usa un confronto **AST-based** (Abstract Syntax Tree) tra la chiamata generata e quella attesa, non solo string-match sul nome del tool.
- **Pro:** è il paper che ha introdotto la valutazione AST-based per le tool-call, oggi standard de facto (vedi BFCL sotto); affronta esplicitamente l'hallucination come problema centrale, non secondario.
- **Contro:** valuta la generazione della singola chiamata API isolata, non un intero ciclo agentico multi-turno con ragionamento intermedio (per quello serve BFCL o AgentBench).
- **Keyword:** #ast-evaluation #api-hallucination #function-calling #retrieval-aware-training
- **Nota approfondita:** [[nota approfondita]]

### Berkeley Function-Calling Leaderboard (BFCL)

- **Autori / Anno / Venue:** progetto Gorilla (UC Berkeley), evoluzione continua dal 2024
- **Link PDF:** non è un singolo paper arXiv, ma un leaderboard/blog + repo continuamente aggiornato — https://gorilla.cs.berkeley.edu/leaderboard.html · repo: https://github.com/ShishirPatil/gorilla
- **Riassunto abstract:** Evoluzione di Gorilla/APIBench: valuta la capacità di function-calling su scenari single-turn, parallel (più chiamate nello stesso turno), multi-turno e agentici, sempre con matching AST-based degli argomenti oltre che del nome del tool.
- **Pro:** oggi lo standard più citato per confrontare modelli diversi sul tool-calling; copre scenari multi-turno e paralleli, più vicini al vostro caso d'uso reale.
- **Contro:** non è un paper peer-reviewed ma un leaderboard vivo — la metodologia può cambiare versione dopo versione, va controllata la release specifica quando si cita.
- **Keyword:** #ast-evaluation #leaderboard #multi-turn #function-calling
- **Nota approfondita:** [[nota approfondita]]

### API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs

- **Autori / Anno / Venue:** Li, Zhao, Yu, Song, Li, Yu, Li, Huang, Li — EMNLP 2023
- **Link PDF:** https://arxiv.org/abs/2304.08244 · https://aclanthology.org/2023.emnlp-main.187/
- **Riassunto abstract:** Benchmark "runnable" con 73 tool API reali; 314 dialoghi annotati con 753 chiamate API per valutare *planning*, *retrieval* e *calling* separatamente. Include anche un training set enorme (1.888 dialoghi, 2.138 API, 1.000 domini) usato per addestrare Lynx (da Alpaca), che supera Alpaca di oltre 26 punti avvicinandosi a GPT-3.5. L'error analysis identifica gli ostacoli principali per la ricerca futura.
- **Pro:** scompone esplicitamente la valutazione in planning / retrieval / calling — schema riusabile per strutturare le vostre metriche per fase, invece di un unico punteggio aggregato.
- **Contro:** dominio generico (API varie, non un grafo di scena strutturato); il training set è pensato per fine-tuning, non necessariamente rilevante se usate solo modelli off-the-shelf via Ollama.
- **Keyword:** #planning-retrieval-calling #runnable-benchmark #tool-augmented-llm #error-analysis
- **Nota approfondita:** [[nota approfondita]]

### ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs

- **Autori / Anno / Venue:** Qin, Liang, Ye, Zhu, Yan, et al. — arXiv, luglio 2023 (ICLR 2024)
- **Link PDF:** https://arxiv.org/abs/2307.16789
- **Riassunto abstract:** Framework con tre componenti: ToolBench (dataset di istruzioni generato via ChatGPT su 16.464 API reali), ToolLLaMA (modello fine-tuned) e ToolEval (valutatore automatico). Introduce un algoritmo decision-tree depth-first-search che permette al modello di esplorare più tracce di ragionamento invece di un singolo percorso lineare, più un retriever neurale per suggerire le API più adatte. ToolLLaMA raggiunge prestazioni comparabili a ChatGPT e generalizza bene su API mai viste.
- **Pro:** ToolEval usa Pass Rate (task completato correttamente?) e Win Rate (confronto pairwise via LLM-as-judge) — metriche riusabili se volete giudicare la qualità della risposta finale oltre alla sola correttezza della tool-call; il decision-tree per esplorare più percorsi di ragionamento è un'idea utile se in futuro volete un agente che non si blocca al primo tentativo fallito.
- **Contro:** scala enorme (16k+ API) pensata per generalizzazione ampia, non per un dominio chiuso come il vostro; ToolEval con LLM-as-judge introduce dipendenza da un altro LLM (costo + possibile bias del giudice).
- **Keyword:** #pass-rate #win-rate #llm-as-judge #decision-tree-search #tool-retrieval
- **Nota approfondita:** [[nota approfondita]]

### T-Eval: Evaluating the Tool Utilization Capability of Large Language Models Step by Step

- **Autori / Anno / Venue:** Chen et al. — arXiv, dicembre 2023 (ACL 2024)
- **Link PDF:** https://arxiv.org/abs/2312.14033
- **Riassunto abstract:** Invece di un punteggio unico "outcome-oriented", scompone la capacità di usare i tool in sei sotto-processi valutati separatamente: instruction following, planning, reasoning, retrieval, understanding, review. Questo permette un'analisi granulare di *dove* un modello fallisce (es. sceglie il tool giusto ma sbaglia gli argomenti, oppure ragiona bene ma non segue le istruzioni sul formato). Gli esperimenti mostrano che i risultati di T-Eval sono coerenti con la valutazione end-to-end ma offrono una prospettiva più fine.
- **Pro:** è il paper più direttamente riusabile per il vostro caso: il vostro nodo "think" (chain-of-thought prima della tool-call) corrisponde concettualmente alle fasi "reasoning/planning" di T-Eval — potete prendere in prestito la loro scomposizione per strutturare `summarize()` in sotto-metriche invece di un unico "num_tool_calls".
- **Contro:** la scomposizione in 6 fasi richiede annotazioni/riferimenti per ciascuna fase (non solo per l'output finale), quindi più lavoro di costruzione del ground truth rispetto a un semplice controllo pass/fail.
- **Keyword:** #step-by-step-evaluation #fine-grained-decomposition #reasoning #planning #instruction-following
- **Nota approfondita:** [[nota approfondita]]

### AgentBench: Evaluating LLMs as Agents

- **Autori / Anno / Venue:** Liu et al. — arXiv, agosto 2023 (ICLR 2024)
- **Link PDF:** https://arxiv.org/abs/2308.03688
- **Riassunto abstract:** Benchmark multi-dimensionale con 8 ambienti interattivi distinti (tra cui uno **knowledge-graph, code-grounded** — il più vicino concettualmente al vostro scene graph) per valutare ragionamento e decision-making di un LLM-agente. I risultati mostrano un forte divario tra i migliori modelli commerciali e i modelli open-source sotto i 70B, e identificano ragionamento a lungo termine, decision-making e instruction-following come i principali colli di bottiglia.
- **Pro:** unico tra questi paper ad avere un ambiente esplicitamente knowledge-graph/code-grounded, quindi il più vicino come "genere" di task al vostro agente su scene graph; identifica ostacoli comuni (instruction-following, ragionamento lungo) utili come categorie di errore da cercare anche nel vostro benchmark.
- **Contro:** gli ambienti sono generici (OS, database, gioco di carte, ecc.), non specifici per grafi di scena architettonici — va usato più come ispirazione metodologica che come benchmark da riusare direttamente.
- **Keyword:** #multi-environment #knowledge-graph-grounded #agent-evaluation #failure-analysis
- **Nota approfondita:** [[nota approfondita]]

---

## Come si collega tutto al vostro benchmark (`arch_agent/benchmark.py`, `grounding_checks.py`)

- Il vostro `check_groundedness` è concettualmente un **FActScore/Ragas-faithfulness in miniatura**, ma contro un grafo strutturato invece che contro testo libero — quindi più vicino, come tipo di verifica, a un controllo AST-based (Gorilla/BFCL) che a un fact-checker testuale.
- Il nodo "think" + `chain_of_thought` che avete aggiunto è l'equivalente pratico della fase "reasoning/planning" di **T-Eval** — se volete affinare la metodologia, T-Eval è la lettura più direttamente applicabile.
- Se in futuro volete estendere `grounding_checks.py` oltre alle regex (es. verificare affermazioni più complesse sulle relazioni), il pattern "scomponi la risposta in asserzioni atomiche e verifica ciascuna" di **FActScore** è il riferimento da adattare.