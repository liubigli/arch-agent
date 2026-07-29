# CIDOC Scene Graph Instructions

Questo documento raccoglie le classi e le proprieta' CIDOC-CRM usate negli esempi e nella costruzione del grafo ontologico L3.

L1 resta il grafo spaziale della scena. L2 resta uno step informativo intermedio basato su CSV/JSON, tool geometrici e validazione. L3 usa queste informazioni per costruire il knowledge graph CIDOC.

## Classi CIDOC Usate

### Oggetti e Luoghi

| Classe CIDOC | Uso nel progetto |
|---|---|
| `crm:E22_Man-Made_Object` | Oggetto costruito materiale: colonne, muri, archi, porte/finestre, scale, pavimenti, volte, tetti, padiglioni. |
| `crm:E24_Physical_Man-Made_Thing` | Variante per insiemi fisici complessi: interi edifici, colonnati, campate. |
| `crm:E19_Physical_Object` | Oggetto fisico generico, per esempio supporto di feature come un muro su cui insistono modanature. |
| `crm:E26_Physical_Feature` | Caratteristica fisica non separabile facilmente dal supporto: modanature, nicchie, aperture come vuoti nel muro. |
| `crm:E53_Place` | Luogo o spazio: scena, piazza, sala, loggia, chiesa come luogo, cappella, padiglione come posto. |

### Tipi, Materiali, Misure, Informazione

| Classe CIDOC | Uso nel progetto |
|---|---|
| `crm:E55_Type` | Qualsiasi classificazione/tipo: tipologia di elemento, funzione, tipo di scena. Esempi: colonna dorica, muro portante, supporto strutturale, spazio espositivo, piazza, chiesa a navate, cappella, padiglione. |
| `crm:E57_Material` | Materiali fisici: breccia, mattoni, intonaco, stucco, legno, vetro, metallo. Il materiale viene letto solo dal CSV della scena. |
| `crm:E54_Dimension` | Misure con valore numerico e unita': altezza colonna, diametro, intercolumnio, raggio volta, spessore muro. |
| `crm:E16_Measurement` | Evento di misurazione da rilievi HBIM/LiDAR o tool geometrici che osserva dimensioni di un elemento architettonico. |
| `crm:E73_Information_Object` | Oggetti informativi: tesi LiDAR-HBIM, schede storiche, documenti di progetto, modelli HBIM come documenti digitali. |
| `crm:E89_Propositional_Object` | Oggetto concettuale che rappresenta una regola o pattern, per esempio una regola di classificazione di piazza, chiesa a navate, padiglione, portico o loggia. |

## Proprieta' CIDOC Usate

### Identificazione, Titoli, Tipi

| Proprieta' | Uso nel progetto |
|---|---|
| `crm:P1_is_identified_by` | Etichette e testi descrittivi, come nome scena, identificativo elemento o tipo. |
| `crm:P102_has_title` | Titoli di oggetti o luoghi, per esempio `Sala delle Colonne del Castello del Valentino`. |
| `crm:P2_has_type` | Associazione a `E55_Type`: elemento -> tipologia architettonica, scena -> tipo di scena. |

Esempi:

```text
Column_1 crm:P2_has_type Type_DoricColumn
Vault_1 crm:P2_has_type Type_CrossVault
DoorWindow_1 crm:P2_has_type Type_DoorWindow
Scene_1 crm:P2_has_type Type_UrbanSquare
```

### Parte-Tutto e Feature

| Proprieta' | Uso nel progetto |
|---|---|
| `crm:P46_is_composed_of` / `crm:P46i_forms_part_of` | Parte-tutto fra oggetti fisici: colonnato -> colonne, volta -> archi/costoloni, scala -> rampe/gradini, muro -> sotto-elementi oggetto. |
| `crm:P56_bears_feature` / `crm:P56i_is_found_on` | Supporto-feature: muro/volta -> modanature, nicchie, aperture trattate come feature; colonna -> decorazione sul capitello. |

Esempi:

```text
Colonnade_1 crm:P46_is_composed_of Column_1
DoorWindow_1 crm:P46i_forms_part_of Wall_1
Wall_1 crm:P56_bears_feature Molding_1
Molding_1 crm:P56i_is_found_on Wall_1
```

### Materiali e Funzione

| Proprieta' | Uso nel progetto |
|---|---|
| `crm:P45_consists_of` | Oggetto -> materiale (`E57_Material`): colonna -> breccia, muro -> mattoni e intonaco, volta -> mattoni, molding -> stucco. |
| `crm:P103_was_intended_for` | Oggetto o scena -> funzione prevista (`E55_Type`): colonna -> supporto strutturale, door_window -> passaggio/illuminazione/ventilazione, stairs -> circolazione verticale, piazza -> spazio pubblico, chiesa -> culto/liturgia. |

Il materiale non viene dedotto dalla point cloud, dal colore, da L1, dalla classe semantica o dalle relazioni spaziali. Viene usato solo se presente nel CSV/JSON di annotazione della scena.

Esempi:

```text
Column_1 crm:P45_consists_of Material_Breccia
Wall_1 crm:P45_consists_of Material_Brick
Molding_1 crm:P45_consists_of Material_Stucco

Column_1 crm:P103_was_intended_for Type_StructuralSupport
DoorWindow_1 crm:P103_was_intended_for Type_Passage
DoorWindow_1 crm:P103_was_intended_for Type_Lighting
```

### Luoghi, Scena, Posizione

| Proprieta' | Uso nel progetto |
|---|---|
| `crm:P89_falls_within` | Luogo o scena contenuto in altro luogo: sala -> castello, cappella laterale -> chiesa, padiglione -> complesso principale. |

Esempi:

```text
SalaDelleColonne crm:P89_falls_within CastelloDelValentino
CappellaLaterale_1 crm:P89_falls_within Chiesa_1
Padiglione_1 crm:P89_falls_within ComplessoPrincipale_1
```

### Misure

| Proprieta' | Uso nel progetto |
|---|---|
| `crm:P39_measured` | Misurazione (`E16_Measurement`) -> elemento misurato (`E22`). |
| `crm:P40_observed_dimension` / `crm:P43_has_dimension` | Misurazione -> dimensione (`E54_Dimension`); oggetto -> dimensione osservata. |

Pattern:

```text
Measurement_Column1_Height a crm:E16_Measurement
Measurement_Column1_Height crm:P39_measured Column_1
Measurement_Column1_Height crm:P40_observed_dimension Dimension_Height_Column1
Column_1 crm:P43_has_dimension Dimension_Height_Column1

Dimension_Height_Column1 a crm:E54_Dimension
Dimension_Height_Column1 crm:P90_has_value "3.42"
Dimension_Height_Column1 crm:P91_has_unit Unit_Meter
```

### Documentazione e Regole

| Proprieta' | Uso nel progetto |
|---|---|
| `crm:P70_documents` | Documento (`E73_Information_Object`) che documenta un oggetto, luogo o scena. |
| `crm:P94_has_created` | Oggetto informativo, come un modello HBIM, creato da un'attivita' di modellazione o ricerca. |

Esempi:

```text
Thesis_LiDAR_HBIM crm:P70_documents SalaDelleColonne
ResearchActivity_HBIM crm:P94_has_created HBIM_Model_1
```
