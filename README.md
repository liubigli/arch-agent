# arch-agent

An LLM-powered agent for interactive analysis of 3D architectural point clouds.

Given a semantically labelled point cloud (LAZ), the system builds a spatial scene graph, enriches it with scene annotations, and derives a CIDOC-CRM ontology graph for cultural-heritage interpretation. A conversational agent lets you explore the resulting scene in natural language.

## How it works

```text
LAZ point cloud
      |
      v
Load and sample points
      |
      v
DBSCAN segmentation -> individual objects
      |
      v
Spatial graph
      - object geometry
      - centroids and bounding boxes
      - near / adjacent_to / above / below
      - supports / rests_on
      - has_part / part_of / is_opening_in / is_ornament_of
      |
      v
CSV annotation and enrichment step
      - CSV element metadata
      - material, typology, function, descriptions
      - explicit relationship evidence from CSV or validation
      - colonnade, portico, loggia, nave, pavilion, etc.
      |
      v
CIDOC ontology graph
      - CIDOC element nodes
      - type/material/function satellites
      - measurements and dimensions
      - aggregate architectural entities
      |
      v
LangGraph Agent
      |
      v
Interactive chat
```

The key distinction is that the spatial graph is the operational object-to-object graph, where geometric/spatial relations have priority. CSV annotations collect metadata and researcher-provided evidence required to build the final CIDOC interpretation.

## Input format

The input can be a LAZ file or a directory containing `.laz` files. When a directory is provided, the CLI lists the available `.laz` files and asks which one to load. The selected file must contain semantic labels in either a `semantic_label` extra dimension or the standard `classification` dimension. Optional RGB channels and normals are preserved when available.

```
semantic_label or classification
optional: red;green;blue;nx;ny;nz
```

Optional CSV annotations can be linked with `--annotation-csv`. Use comma
separation and match objects by global AABB box center:

```csv
semantic_label,global_box_center_x,global_box_center_y,global_box_center_z,material,typology,function,description
column,631.367,813.088,231.604,"Stone supplied by researcher","Column type","Structural role","User description"
```

Material, typology, function, and historical/descriptive notes come from this
CSV metadata only; they are not inferred from point-cloud visual features.
Material type is determined exclusively from the CSV attached to the scene.

Supported semantic labels are integer-encoded. The class registry is defined in
`arch_agent/semantic_schema.py` and reused by the loader, relationship rules and
agent tools.

| ID | Class | Category | Architectural role |
|----|---|---|---|
| 0 | arch | structural | structural |
| 1 | column | structural | structural |
| 2 | moldings | finishing | ornamental |
| 3 | floor | finishing | support_surface |
| 4 | door_window | finishing | opening |
| 5 | wall | structural | structural |
| 6 | stairs | finishing | circulation |
| 7 | vault | structural | structural |
| 8 | roof | structural | structural |
| 9 | other | finishing | unknown |


## Object segmentation

The pipeline uses DBSCAN segmentation for all semantic classes.

| Semantic class | Method | Reason |
|---|---|---|
| `column` | DBSCAN | density-based object extraction |
| `arch` | DBSCAN | density-based object extraction |
| `door_window` | DBSCAN | density-based object extraction |
| `wall` | DBSCAN | continuous or irregular geometry |
| `floor` | DBSCAN | continuous surface |
| `vault` | DBSCAN | irregular/continuous curved geometry |
| `roof` | DBSCAN | irregular/continuous geometry |
| `stairs` | DBSCAN | variable topology |
| `moldings` | DBSCAN | often continuous decorative geometry |
| `other` | DBSCAN | unknown or mixed topology |

Each detected object stores the segmentation method used:

```python
"segmentation_method": "dbscan"
```

## Scene Understanding Layers

The scene is no longer described as three equivalent graphs. The current model separates the operational spatial graph, annotation/detail metadata, and CIDOC ontology:

| Layer | Output | Meaning |
|---|---|---|
| Spatial graph | `networkx.DiGraph` / scenegraph | Object-to-object spatial relations between segmented objects. Geometric/spatial relations have priority (`near`, `adjacent_to`, `above`, `below`); validated architectural relations can also be stored when supported by spatial rules or CSV/user metadata (`supports`, `rests_on`, `has_part`, `part_of`, `is_opening_in`, `is_ornament_of`, etc.). |
| CSV annotation/enrichment | CSV/JSON annotations linked to objects and aggregate evidence | User/researcher metadata: material, typology, function, descriptions, historical notes, plus explicit relationship evidence and validated aggregate labels such as colonnade, portico, loggia, nave, pavilion. |
| CIDOC ontology graph | CIDOC-oriented knowledge graph | Cultural-heritage interpretation built from spatial graph + CSV detail: elements, types, materials, functions, measurements, and aggregate architectural entities. |

### Spatial Graph

The spatial graph is the operational scenegraph derived from point-cloud geometry, architectural class rules, and optional CSV/user metadata. It contains object geometry and object-to-object spatial relations. Geometric/spatial relations are evaluated first; architectural relations such as support, appoggio, opening, ornament, or part-of are added only when they are spatially valid or explicitly supported by CSV/user metadata. It does not assign material, historical interpretation, or architectural aggregate identity by itself.

Examples:

```text
column_1 near column_2
vault_1 above floor_1
column_1 supports vault_1
vault_1 rests_on column_1
moldings_1 is_ornament_of wall_1
wall_1 has_part moldings_1
door_window_1 is_opening_in wall_1
```

### CSV Annotation And Enrichment Step

CSV annotation is not a graph. It is the intermediate information layer that collects and links external knowledge to the scene.

CSV annotation includes two kinds of information:

1. Element-level annotations from the scene CSV:
   - material
   - typology
   - function
   - description
   - historical/material notes

2. Scene/aggregate annotations from geometric tools or researcher validation:
   - colonnade / colonnato
   - portico / porticato
   - loggia / loggiato
   - nave / navata
   - pavilion / padiglione

Material type is determined only from the CSV attached to the scene. It is not inferred from the point cloud, geometry, semantic class, or spatial-graph relations.

Aggregate annotations can be derived from geometric tools or validation and then added to the scene annotation data. They are used to support the final scene-level description.

### CIDOC Ontology Graph

CIDOC/KG is the ontological layer. It uses CIDOC-CRM patterns to formalize the information from the spatial graph and CSV annotations.

The CIDOC/KG graph contains:

```text
Element_N crm:P2_has_type Tipo_*
Element_N crm:P45_consists_of Materiale_*
Element_N crm:P103_was_intended_for Funzione_*
Aggregate_N crm:P46_is_composed_of Element_N
```

CIDOC element-to-element or aggregate-to-element relations are created only when they are supported by local spatial evidence from the spatial graph or explicit aggregate evidence from CSV/user metadata. The system must not connect distant elements only because they are semantically compatible.

For relationship queries, `list_relationships` is the primary tool for inspecting spatial relations and existing graph relations. CIDOC aggregate interpretation is handled by the CIDOC/KG builder and documented in `docs/cidoc_l3_scene_graph_builder.md`.

## Requirements

- [Ollama](https://ollama.com/) running locally with `llama3` pulled
- [Pixi](https://prefix.dev/) for environment management

```bash
ollama pull llama3
ollama serve          # in a separate terminal
```

Install Python dependencies:

```bash
pixi install
```

## Usage

```bash
# Basic usage with default directory and interactive file selection
python main.py

# From WSL, Windows paths are converted automatically when using the default.
# You can also pass an explicit /mnt/c/... path.
python main.py /mnt/c/Users/Utente/Desktop/Lucrezia/Lu_test_project/laz_archdataset_palette_originale/scena19_KAS_pavillion_2.laz

# Tune DBSCAN clustering (smaller eps = tighter clusters)
python main.py --eps 0.3 --min-samples 10

# Extend the spatial relationship radius and use a different model
python main.py --distance-threshold 5.0 --model llama3.1

# Use Poisson reconstruction for more accurate surface area estimates
python main.py --use-normals

# Use another LAZ file or directory
python main.py path/to/scene.laz
```

### Open3D visualization

The project includes a small Open3D viewer with the same workflow used by the
notebook-style prototype: semantic point cloud first, DBSCAN clusters second.

```bash
# Semantic classes with flat colors
pixi run view-pointcloud /mnt/c/Users/Utente/Desktop/Lucrezia/Lu_test_project/laz_archdataset_segmented/scena4_VAL.laz

# DBSCAN output with one color per detected object
pixi run view-dbscan /mnt/c/Users/Utente/Desktop/Lucrezia/Lu_test_project/laz_archdataset_segmented/scena4_VAL.laz --eps 0.5 --min-samples 15

# Open both viewers in sequence; close the first window to open the second
pixi run view-pointcloud-both /mnt/c/Users/Utente/Desktop/Lucrezia/Lu_test_project/laz_archdataset_segmented/scena4_VAL.laz

# Optional: show only selected semantic classes and their AABB boxes
pixi run view-dbscan path/to/scene.laz --classes column wall --with-boxes
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--eps` | `0.5` | DBSCAN epsilon for object segmentation |
| `--min-samples` | `15` | DBSCAN min_samples (lower for sparse clouds) |
| `--distance-threshold` | `3.0` | Max centroid distance (m) for spatial relationships |
| `--sample-n` | `150000` | Max points to load (0 = no limit) |
| `--use-normals` | `False` | Poisson-based surface area (slower, more accurate) |
| `--model` | `llama3` | Ollama model to use |

## Example interaction

```
You: How many structural elements are in the scene?
Agent: The scene contains 8 structural elements: 3 columns, 2 walls, 2 arches and 1 vault.

You: Which element is the most central in the scene?
Agent: The most spatially central element is column_1, with a centrality score of 0.82. ...

You: Reload the scene with eps=0.3 to get finer clusters
Agent: [calls reload_scene] Scene reloaded. Objects: 24 | Relationships: 41 ...
```

## Configuration

The semantic taxonomy is kept in Python, not YAML:

- `arch_agent/semantic_schema.py` defines semantic IDs, categories, roles,
  DBSCAN segmentation notes, support rules and part/object relations.
- `arch_agent/settings.py` exposes the same class list and visualization colors
  to the loader and tools.
- `prompts/system.md` defines the conversational behavior of the agent.

## Project structure

```
prompts/
│   └── system.md          # agent system prompt (editable)
arch_agent/
├── semantic_schema.py     # semantic IDs, roles, DBSCAN notes and relation rules
├── settings.py            # runtime config and visualization colors
├── pipeline/
│   ├── loader.py          # LAZ → DataFrame
│   ├── segmentation.py    # DBSCAN object extraction
│   ├── features.py        # geometric feature computation
│   ├── relationships.py   # spatial graph rules
│   ├── graph.py           # NetworkX DiGraph builders
│   └── pipeline.py        # PipelineParams, SceneContext, run_pipeline()
├── tools/
│   └── scene_tools.py     # LangChain tools wrapping the scene graph
├── agent.py               # LangGraph agent + conversation loop
└── __init__.py
main.py                    # CLI entry point
```

## CIDOC Ontology Layer

The CIDOC/KG layer is a CIDOC-CRM based knowledge graph for cultural heritage interpretation.

- The spatial graph is the operational graph derived from geometry, class rules,
  CSV/user metadata, and validated local evidence.
- CSV annotation/enrichment is an information step, not a graph: it links element metadata and validated aggregate evidence to the scene.
- Material, typology and function values in CSV detail and CIDOC/KG come only from the scene annotation CSV.
- CIDOC/KG builds a CIDOC-oriented ontology graph from spatial graph + CSV detail without inventing missing values.
- Element-to-element CIDOC relations are created only when supported by local spatial evidence or explicit scene evidence.

Implemented CIDOC patterns include element nodes connected to type, material and function satellites:

```text
Elemento_N crm:P2_has_type Tipo_*
Elemento_N crm:P45_consists_of Materiale_*
Elemento_N crm:P103_was_intended_for Funzione_*
```

Additional CIDOC/KG rules currently documented/implemented:

- `colonnade`: inferred from at least 4 aligned, approximately equispaced structural-support columns.
- `portico`: inferred only with aligned arches or columns+architraves, continuous cover, covered walkable ground-floor space, open external side and opposite side attached to/closed by the building.
- `loggia`: inferred only with a covered room/gallery integrated in the building volume, one open side made of arches on columns or pillars, and intermediate/representative function.

Full documentation:

- `docs/cidoc_l3_scene_graph_builder.md`
- `docs/cidoc_scene_graph.md`

Python implementation:

- `arch_agent/pipeline/l3_cidoc_graph_builder.py`
