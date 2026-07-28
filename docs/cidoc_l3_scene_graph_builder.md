# CIDOC L3 Scene Graph Builder

Questo progetto definisce una logica a tre livelli per descrivere scene point cloud/HBIM.

- L1 e' fisso, non modificato in questa sessione, e si fonda sullo scenegraph geometrico/spaziale della scena.
- L2 riporta annotazioni sulle classi della scena: elementi strutturali, decorazioni e altri elementi non strutturali.
- L3 costruisce una lettura ontologica CIDOC-CRM per beni culturali a partire da L1 e L2.

Le modifiche di questa sessione riguardano L3, non L1.

Il grafo L3 varia da scena a scena: ogni scena passa un proprio `scene_id`, le annotazioni L2 e, quando disponibili, misure geometriche derivate da L1 o da tool spaziali.

## Livelli

### L1: Scenegraph Geometrico

L1 e' il livello fisso della pipeline. Rappresenta lo scenegraph della scena e contiene dati geometrici/spaziali derivati dalla point cloud, dall'HBIM o da tool spaziali.

Esempi di informazioni L1:

- coordinate e centroidi;
- bounding box;
- relazioni spaziali calcolate;
- misure geometriche derivate;
- struttura di scena gia' disponibile.

L1 non viene reinterpretato semanticamente dal builder CIDOC: viene usato come fonte dati geometrica.

### L2: Annotazioni Semantiche

L2 contiene le annotazioni sulle classi riconosciute nella scena.

Include:

- classi strutturali, per esempio `column`, `wall`, `vault`, `floor`, `stair`;
- decorazioni o feature, per esempio `molding`;
- aperture e accessi, normalizzati come `door_window`;
- materiale, tipologia, funzione e descrizione quando forniti da CSV/JSON.

L2 non e' ancora ontologia CIDOC: e' il livello di annotazione semantica della scena.

### L3: Ontologia CIDOC per Beni Culturali

L3 e' il livello ontologico. Traduce le informazioni di L1 e L2 in un grafo CIDOC-CRM per beni culturali.
Il knowledge graph CIDOC e' costruito sopra lo scenegraph L1: le relazioni tra elementi devono essere supportate da prossimita', contatto, adiacenza o altra evidenza spaziale.

L3 non inventa dati mancanti: formalizza solo informazioni esplicite provenienti da L1, L2 o da regole CIDOC configurate.
Non collega elementi lontani solo perche' semanticamente compatibili. Per esempio, una colonna non viene collegata a una volta o a una parete dall'altra parte della stanza se lo scenegraph L1 non supporta quella relazione.

## Fonti Dati Ammesse

Il tool non deve inventare informazioni. Formalizza solo dati espliciti provenienti da:

- point cloud, L1, HBIM o tool spaziali: coordinate, centroidi, bounding box, misure e relazioni geometriche calcolate;
- CSV o JSON di annotazione: classe semantica, materiale, tipologia, funzione e descrizione;
- regole CIDOC configurate nel builder: mapping classi e predicati ontologici.

Il tipo di materiale viene determinato esclusivamente dal CSV allegato alla scena. La point cloud, L1, HBIM e i tool spaziali non sono fonti valide per dedurre automaticamente il materiale.

Se un materiale, una tipologia, una funzione o una misura non sono presenti negli input, il tool non crea automaticamente valori dedotti.

## Input Annotazioni

I CSV/JSON di scena sono input forniti dall'utente per la comprensione della scena, non output del progetto.

Schema minimo atteso per ogni elemento:

```text
class_semantic_label
global_box_center_x
global_box_center_y
global_box_center_z
material
typology
function
description
```

`door_window` e' una classe ombrello: comprende porte, finestre e porta-finestra.

## Mapping Classi CIDOC

| Classe point cloud | Nodo L3 | Classe locale | Classe CIDOC |
| --- | --- | --- | --- |
| column | Colonna_N | Column | crm:E22_Human-Made_Object |
| colonnade | Colonnato_N | Colonnade | crm:E22_Human-Made_Object |
| portico | Portico_N | Portico | crm:E22_Human-Made_Object |
| loggia | Loggia_N | Loggia | crm:E22_Human-Made_Object |
| wall | Muro_N | Wall | crm:E22_Human-Made_Object |
| vault | Volta_N | Vault | crm:E22_Human-Made_Object |
| roof | Roof_N | Roof | crm:E22_Human-Made_Object |
| floor | Pavimento_N | Floor | crm:E22_Human-Made_Object |
| stair | Scala_N | Stair | crm:E22_Human-Made_Object |
| door_window | PortaFinestra_N | DoorWindow | crm:E22_Human-Made_Object |
| molding | Modanatura_N | Molding | crm:E26_Physical_Feature |
| architrave | Architrave_N | Architrave | crm:E22_Human-Made_Object |
| pillar | Pilastro_N | Pillar | crm:E22_Human-Made_Object |

## Pattern L3 Principale

Ogni elemento architettonico viene collegato a tre nodi satellite:

```text
Elemento_N crm:P2_has_type Tipo_*
Elemento_N crm:P45_consists_of Materiale_*
Elemento_N crm:P103_was_intended_for Funzione_*
```

Classi CIDOC dei satelliti:

```text
Tipo_*       a crm:E55_Type
Materiale_* a crm:E57_Material
Funzione_*  a crm:E55_Type
```

Esempio:

```text
Colonna_1
  a Column
  a crm:E22_Human-Made_Object
  crm:P2_has_type Tipo_ColonnaDorica
  crm:P45_consists_of Materiale_BrecciaPolicroma
  crm:P103_was_intended_for Funzione_SostegnoDelleVolte
```

## Relazioni Fisiche Contestuali

Le relazioni fisiche non sostituiscono la struttura principale L3, ma la completano quando sono supportate da dati geometrici o regole esplicite.
Ogni relazione elemento-elemento deve essere locale: deriva dallo scenegraph L1 o da relazioni spaziali calcolate da tool geometrici.

Pattern per classe a livello di soli elementi:

| Classe | Pattern CIDOC ammessi |
| --- | --- |
| column | `P2_has_type`, `P45_consists_of`, `P103_was_intended_for`, opzionale `P46i_forms_part_of` |
| colonnade | `P46_is_composed_of Column` |
| portico | `P2_has_type`, `P103_was_intended_for`, `P46_is_composed_of Arch/Column/Architrave/Cover` |
| loggia | `P2_has_type`, `P103_was_intended_for`, `P46_is_composed_of Arch/Column/Pillar/Cover` |
| wall | `P2_has_type`, `P45_consists_of`, `P56_bears_feature` |
| vault | `P46i_forms_part_of Roof`, `P46_is_composed_of Arch`, `P2_has_type`, `P45_consists_of`, `P103_was_intended_for` |
| stair | `P45_consists_of`, `P103_was_intended_for`, `P46_is_composed_of` |
| door_window | `P46i_forms_part_of`, `P2_has_type`, `P45_consists_of`, `P103_was_intended_for` |
| molding | `P56i_is_found_on`, `P45_consists_of`, `P2_has_type` |

Pattern per porte, finestre e porta-finestra:

```text
PortaFinestra_N crm:P46i_forms_part_of Muro_N
```

Pattern per modanature che insistono su una parete:

```text
Muro_N crm:P56_bears_feature Modanatura_N
Modanatura_N crm:P56i_is_found_on Muro_N
```

Pattern per componenti di volte e scale, se presenti in L2 o nel grafo geometrico:

```text
Volta_N crm:P46_is_composed_of Arco_N
Volta_N crm:P46_is_composed_of Costolone_N
Scala_N crm:P46_is_composed_of Rampa_N
Scala_N crm:P46_is_composed_of Gradino_N
```

Queste relazioni elemento-elemento devono provenire da L1, L2 o da una regola esplicita con supporto spaziale. Il builder non crea archi contestuali quando sorgente o target non sono presenti nei dati, e non collega elementi distanti senza evidenza spaziale.

## Regola L3 per Colonnato

Il builder puo' definire `Colonnato_N` solo quando ci sono evidenze esplicite nei dati geometrici e semantici.

Condizioni minime:

- almeno 4 colonne;
- colonne con coordinate `global_box_center_x/y`;
- colonne ordinate lungo una direttrice e approssimativamente equispaziate;
- colonne con funzione esplicita di sostegno strutturale tramite `crm:P103_was_intended_for`.

Pattern CIDOC:

```text
Colonnato_N
  a Colonnade
  a crm:E22_Human-Made_Object
  crm:P46_is_composed_of Colonna_1
  crm:P46_is_composed_of Colonna_2
  crm:P46_is_composed_of Colonna_3
  crm:P46_is_composed_of Colonna_4
  crm:P46_is_composed_of Colonna_5
```

Il colonnato non viene creato se mancano coordinate, regolarita' spaziale o funzione di sostegno. La regola deriva da L1/L2 e non introduce un'interpretazione non documentata.

## Regola L3 per Portico / Porticato

Il builder puo' definire `Portico_N` solo quando la scena fornisce evidenze esplicite. Non basta avere archi o colonne: devono essere presenti anche le condizioni spaziali e d'uso.

Condizioni minime:

- insieme di archi allineati, oppure colonne piu' architravi allineati lungo una direttrice retta o curva;
- copertura continua sopra, per esempio solaio, terrazza, tetto o altro orizzontamento strutturale;
- spazio praticabile coperto immediatamente dietro gli archi;
- collocazione al piano terreno;
- lato degli archi aperto verso uno spazio esterno, per esempio via, piazza, cortile o giardino;
- lato opposto addossato a un edificio o chiuso da pareti.

Se manca la copertura, lo spazio coperto retrostante o la condizione di piano terreno, il builder non classifica l'insieme come portico.

Le condizioni non geometriche devono arrivare come input esplicito in `scene_evidence`, per esempio:

```text
rule = portico
continuous_cover_above = true
covered_walkable_space_behind = true
ground_floor = true
open_to_external_space = true
opposite_side_against_building_or_walls = true
source = L1_spatial_tool_or_researcher_validation
description = Portico al piano terreno aperto verso il cortile e addossato alla parete dell'edificio.
```

Pattern CIDOC:

```text
Portico_N
  a Portico
  a crm:E22_Human-Made_Object
  crm:P2_has_type Tipo_PorticoPorticato
  crm:P46_is_composed_of Arco_N
  crm:P46_is_composed_of Colonna_N
  crm:P46_is_composed_of Architrave_N
  crm:P46_is_composed_of Roof_N
```

Anche questa regola resta vincolata a L1: i componenti devono essere allineati o spazialmente supportati. Il builder non crea un portico se le evidenze richieste non sono presenti negli input.

## Regola L3 per Loggia / Loggiato

Il builder puo' definire `Loggia_N` solo quando la scena documenta un ambiente coperto, inserito nel volume dell'edificio, con almeno un lato integralmente aperto verso l'esterno tramite una serie di archi su colonne o pilastri.

Condizioni minime:

- ambiente coperto, stanza o galleria;
- almeno un lato integralmente aperto verso l'esterno;
- lato aperto costituito da una serie di archi su colonne o pilastri;
- spazio inserito nel volume dell'edificio o comunque parte dell'edificio;
- funzione prevalente di spazio intermedio o rappresentativo tra interno ed esterno;
- quota libera: puo' essere al piano terreno o ai piani superiori.

Se e' al piano terreno ma la funzione e' chiaramente percorso di accesso diretto all'edificio, e la parte aperta e' solo un filtro verso il portone, il builder non crea `Loggia_N`; in quel caso la classificazione preferita resta `Portico_N`.

Le condizioni non geometriche devono arrivare come input esplicito in `scene_evidence`, per esempio:

```text
rule = loggia
covered_room_or_gallery = true
fully_open_side_to_external_space = true
arches_on_columns_or_pillars = true
inside_building_volume = true
intermediate_representative_function = true
ground_floor = false
direct_building_access_function = false
simple_entrance_filter = false
source = L1_spatial_tool_or_researcher_validation
description = Loggia inserita nel volume dell'edificio, aperta verso il cortile tramite archi su colonne.
```

Pattern CIDOC:

```text
Loggia_N
  a Loggia
  a crm:E22_Human-Made_Object
  crm:P2_has_type Tipo_LoggiaLoggiato
  crm:P103_was_intended_for Funzione_SpazioIntermedioRappresentativo
  crm:P46_is_composed_of Arco_N
  crm:P46_is_composed_of Colonna_N
  crm:P46_is_composed_of Pilastro_N
  crm:P46_is_composed_of Roof_N
```

Anche questa regola resta vincolata a L1: archi e sostegni devono risultare allineati e spazialmente supportati. Il builder non crea una loggia se le evidenze richieste non sono presenti negli input.

## Funzioni Multiple

Un elemento puo' avere piu' funzioni. Nel campo `function` usa `;` oppure `|` come separatore:

```text
Passaggio; Illuminazione
```

Il grafo genera:

```text
PortaFinestra_1 crm:P103_was_intended_for Funzione_Passaggio
PortaFinestra_1 crm:P103_was_intended_for Funzione_Illuminazione
```

## Misure da L1, HBIM e Tool Spaziali

Le misure geometriche derivate dalla point cloud o da strumenti spaziali sono modellate come:

```text
Measurement_* a crm:E16_Measurement
Measurement_* crm:P39_measured Elemento_N
Measurement_* crm:P43_has_dimension Dimension_*

Dimension_* a crm:E54_Dimension
Dimension_* crm:P90_has_value valore_numerico
```

Pattern specifico per `vault`:

```text
Volta_N
  a Vault
  a crm:E22_Human-Made_Object
  crm:P46i_forms_part_of Roof_N
  crm:P46_is_composed_of Arco_N
  crm:P2_has_type Tipo_Volta*
  crm:P45_consists_of Materiale_*
  crm:P103_was_intended_for Funzione_*

Measurement_VoltaN_*
  a crm:E16_Measurement
  crm:P39_measured Volta_N
  crm:P43_has_dimension Dimension_VoltaN_*

Dimension_VoltaN_*
  a crm:E54_Dimension
  crm:P90_has_value valore_numerico
  unit "m"
```

Campi minimi per una misura:

```text
target_node_id
```

oppure:

```text
target_local_class
target_index
```

piu':

```text
dimension_type
value
unit
source
```

Esempio:

```text
target_local_class = column
target_index = 1
dimension_type = height
value = 3.20
unit = m
source = point_cloud_spatial_tool
```

## Uso

La funzione principale e':

```python
scene_graph = build_scene_graph(
    scene_id="scena4_VAL",
    annotations=annotations,
    measurements=measurements,
    contextual_relations=contextual_relations,
    spatial_relations=spatial_relations,
    scene_evidence=scene_evidence,
)
```

Restituisce un oggetto `SceneGraph` con:

```text
scene_id
nodes
edges
```

Gli elementi sono namespacizzati per scena, ad esempio:

```text
Scena4Val_Colonna_1
Scena4Val_Muro_1
Scena4Val_PortaFinestra_1
```

I nodi satellite `Tipo_*`, `Materiale_*` e `Funzione_*` restano concetti riusabili.

## File Principali

- `work/l3_cidoc_graph_builder.py`: logica Python per costruire il grafo CIDOC L3 in memoria.
- `work/cidoc_second_labelling.md`: specifica della seconda labellizzazione CIDOC.
- `outputs/scena4_VAL_annotations.csv`: input di annotazione scena usato in questa sessione.
- `outputs/scena4_VAL_annotations_no_source.csv`: variante input senza colonna `source_file`.

Il builder non esporta CSV automaticamente e non produce output di scena se non richiesto esplicitamente.
I CSV di scena attualmente presenti sono da considerare input/prototipi di input, non output finali.
Eventuali export verso RDF/Turtle, JSON-LD, Neo4j o CSV vanno aggiunti come step separato e richiesto esplicitamente.
