# Seconda Labellizzazione CIDOC per Classi Point Cloud

Questa labellizzazione affianca la classe semantica originale della point cloud con una lettura CIDOC-CRM.
Il grafo L3 non e' mereologico: ogni elemento e' un nodo centrale collegato a tre nodi satellite.
Il knowledge graph CIDOC e' comunque fondato sullo scenegraph L1: le relazioni tra elementi sono considerate valide solo quando hanno supporto spaziale locale.

## Nodi Centrali

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

## Nodi Satellite

| Campo annotazione | Nodo satellite | Classe CIDOC | Relazione dal nodo centrale |
| --- | --- | --- | --- |
| typology | Tipo_* | crm:E55_Type | crm:P2_has_type |
| material | Materiale_* | crm:E57_Material | crm:P45_consists_of |
| function | Funzione_* | crm:E55_Type | crm:P103_was_intended_for |

## Pattern

```text
Elemento_N
  a crm:E22_Human-Made_Object oppure crm:E26_Physical_Feature
  crm:P2_has_type Tipo_*
  crm:P45_consists_of Materiale_*
  crm:P103_was_intended_for Funzione_*
```

## Relazioni Fisiche Contestuali

La seconda labellizzazione CIDOC usa tipo, materiale e funzione come struttura principale.
Le relazioni parte/tutto possono comunque essere mantenute come relazioni fisiche secondarie quando servono a descrivere il contesto architettonico.
Queste relazioni non sono globali: devono riguardare elementi vicini, adiacenti, in contatto o comunque collegati nello scenegraph L1.

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

Per il pattern semplice adottato qui, una porta/finestra fisicamente integrata nel muro e' espressa sempre con `crm:P46i_forms_part_of`, orientata da `PortaFinestra_N` verso `Muro_N`.

La classe `door_window` e' una classe ombrello: comprende porte, finestre e porta-finestra. Nel grafo L3 vengono tutte normalizzate come `DoorWindow` e come `crm:E22_Human-Made_Object`.

```text
PortaFinestra_1
  a DoorWindow
  a crm:E22_Human-Made_Object
  crm:P46i_forms_part_of Muro_1
```

Per una modanatura che insiste su una parete:

```text
Muro_1
  a Wall
  a crm:E22_Human-Made_Object
  crm:P56_bears_feature Modanatura_1
```

Equivalente inverso lato modanatura:

```text
Modanatura_1
  a Molding
  a crm:E26_Physical_Feature
  crm:P56i_is_found_on Muro_1
```

Per componenti di volte e scale, se presenti in L1/L2:

```text
Volta_1 crm:P46_is_composed_of Arco_1
Volta_1 crm:P46_is_composed_of Costolone_1
Scala_1 crm:P46_is_composed_of Rampa_1
Scala_1 crm:P46_is_composed_of Gradino_1
```

Per il colonnato:

```text
Colonnato_1
  a Colonnade
  a crm:E22_Human-Made_Object
  crm:P46_is_composed_of Colonna_1
  crm:P46_is_composed_of Colonna_2
  crm:P46_is_composed_of Colonna_3
  crm:P46_is_composed_of Colonna_4
  crm:P46_is_composed_of Colonna_5
```

La regola crea `Colonnato_1` solo se:

```text
numero_colonne >= 4
coordinate global_box_center_x/y presenti
sequenza allineata su direttrice e approssimativamente equispaziata
funzione delle colonne = sostegno/supporto strutturale tramite P103
```

Senza queste evidenze da L1/L2, il colonnato non viene creato.

Per il portico / porticato:

```text
Portico_1
  a Portico
  a crm:E22_Human-Made_Object
  crm:P2_has_type Tipo_PorticoPorticato
  crm:P46_is_composed_of Arco_1
  crm:P46_is_composed_of Arco_2
  crm:P46_is_composed_of Roof_1
```

La regola crea `Portico_1` solo se:

```text
archi allineati oppure colonne + architravi allineati
continuous_cover_above = true
covered_walkable_space_behind = true
ground_floor = true
open_to_external_space = true
opposite_side_against_building_or_walls = true
```

Se manca la copertura, lo spazio coperto retrostante o il piano terreno, il portico non viene creato.
Le condizioni non deducibili dalla sola geometria devono arrivare in `scene_evidence`.

Per la loggia / loggiato:

```text
Loggia_1
  a Loggia
  a crm:E22_Human-Made_Object
  crm:P2_has_type Tipo_LoggiaLoggiato
  crm:P103_was_intended_for Funzione_SpazioIntermedioRappresentativo
  crm:P46_is_composed_of Arco_1
  crm:P46_is_composed_of Colonna_1
  crm:P46_is_composed_of Pilastro_1
  crm:P46_is_composed_of Roof_1
```

La regola crea `Loggia_1` solo se:

```text
covered_room_or_gallery = true
fully_open_side_to_external_space = true
arches_on_columns_or_pillars = true
inside_building_volume = true
intermediate_representative_function = true
archi allineati su colonne o pilastri
```

La quota puo' essere piano terra o piano superiore.
Se al piano terra la funzione e' ingresso diretto all'edificio o semplice filtro verso il portone, non viene creata una loggia: la classificazione preferita e' portico.

Per funzioni multiple sullo stesso elemento, usa valori separati da `;` oppure `|`.

```text
PortaFinestra_1
  crm:P103_was_intended_for Funzione_Passaggio
  crm:P103_was_intended_for Funzione_Illuminazione
```

Uso consigliato:

```text
Elemento -> Tipo / Materiale / Funzione      struttura L3 principale
PortaFinestra -> Muro tramite P46i           relazione fisica contestuale
Muro -> Modanatura tramite P56               feature fisica sulla parete
```

## Funzione Scene-by-Scene

La costruzione del grafo per ogni scena passa da:

```python
scene_graph = build_scene_graph(
    scene_id,
    annotations,
    measurements=measurements,
    contextual_relations=contextual_relations,
    spatial_relations=spatial_relations,
    scene_evidence=scene_evidence,
)
```

Dove:

```text
scene_id      identifica la scena, per esempio scena4_VAL
annotations   contiene le righe con classe semantica, coordinate, materiale, tipologia, funzione e descrizione
measurements  contiene misure derivate da L1, HBIM, grafo geometrico o tool spaziali
contextual_relations contiene relazioni elemento-elemento esplicite, per esempio P46, P46i, P56, P56i
spatial_relations contiene il supporto L1/scenegraph per prossimita', contatto, adiacenza, intersezione o contenimento
scene_evidence contiene evidenze di scena non sempre deducibili dalle sole coordinate, per esempio le condizioni per riconoscere un portico
```

Regola: una relazione CIDOC tra elementi viene creata solo se e' supportata da `spatial_relations` oppure se l'input dichiara esplicitamente `spatially_supported=true`.

Gli elementi della scena vengono namespacizzati, per esempio:

```text
scena4VAL_Colonna_1
scena4VAL_Muro_1
scena4VAL_PortaFinestra_1
```

I nodi satellite condivisi, come `Tipo_*`, `Materiale_*` e `Funzione_*`, restano concetti riusabili.

## Misure da L1, HBIM e Tool Spaziali

Le misure geometriche non sono attributi semplici dell'elemento. Vengono modellate come eventi/atti di misurazione e dimensioni CIDOC.

Pattern:

```text
Measurement_Column1_Height
  a crm:E16_Measurement
  crm:P39_measured Colonna_1
  crm:P43_has_dimension Dimension_Height_Column1

Dimension_Height_Column1
  a crm:E54_Dimension
  crm:P90_has_value "3.20"
  value "3.20"
  unit "m"
```

Pattern specifico per ogni `vault`:

```text
Volta_1
  a Vault
  a crm:E22_Human-Made_Object
  crm:P46i_forms_part_of Roof_1
  crm:P46_is_composed_of Arco_1
  crm:P2_has_type Tipo_Volta*
  crm:P45_consists_of Materiale_*
  crm:P103_was_intended_for Funzione_*

Measurement_Volta1_*
  a crm:E16_Measurement
  crm:P39_measured Volta_1
  crm:P43_has_dimension Dimension_Volta1_*

Dimension_Volta1_*
  a crm:E54_Dimension
  crm:P90_has_value valore_numerico
```

Campi minimi per una misura:

```text
target_node_id oppure target_local_class + target_index
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

## Esempi

```text
Colonna_1
  a Column
  a crm:E22_Human-Made_Object
  crm:P2_has_type Tipo_ColonnaDorica
  crm:P45_consists_of Materiale_BrecciaPolicroma
  crm:P103_was_intended_for Funzione_SostegnoDelleVolte
```

```text
Modanatura_1
  a Molding
  a crm:E26_Physical_Feature
  crm:P2_has_type Tipo_ApparatoDecorativoBarocco
  crm:P45_consists_of Materiale_Stucco
  crm:P103_was_intended_for Funzione_MonumentaleRappresentativa
```
