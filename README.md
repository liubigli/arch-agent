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
      - supports / rests_on and composition links when validated
      |
      v
CSV / metadata enrichment
      - CSV element metadata
      - material, typology, function, descriptions
      - aggregate evidence from geometric tools or validation
      - colonnade, portico, loggia, nave, pavilion, etc.
      |
      v
CIDOC / knowledge graph
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

The key distinction is that the spatial graph is the operational scene graph, CSV metadata is an enrichment layer, and CIDOC/KG is the cultural-heritage knowledge graph built from the available evidence.

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

The CSV mixes two granularities, and both are matched. A row whose
`global_box_center` lands on one object describes that object: six columns get
six rows. A row whose centre coincides with the centre of *all* objects of its
class describes the whole semantic region and is attached to each of them -
which is how a single `wall` row covers the eleven fragments DBSCAN extracts
from one continuous wall. `--annotation-class-region-tolerance 0` turns the
second rule off and restores strict per-object matching.

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

The scene is not described as three equivalent graphs. The current model separates spatial relations, user/researcher annotations, and ontology:

| Layer | Internal key | Output | Meaning |
|---|---|---|---|
| Spatial graph | `L1` | `networkx.DiGraph` / scenegraph | Spatial and locally validated architectural relations between segmented objects: `near`, `adjacent_to`, `above`, `below`, `supports`, `rests_on`, `is_opening_in`, `is_ornament_of`, `part_of`, plus object geometry, distances, bounding boxes and centroids. |
| CSV/detail enrichment | `L2_DETAIL` | CSV/JSON annotations linked to objects and aggregate evidence | User/researcher metadata: material, typology, function, descriptions, historical notes, plus validated aggregate labels such as colonnade, portico, loggia, nave, pavilion. This is not a graph. |
| CIDOC/KG | `L3` | CIDOC-oriented knowledge graph | Cultural-heritage interpretation built from the spatial graph and CSV/detail enrichment: elements, types, materials, functions, measurements, and aggregate architectural entities. |

### Spatial Graph

The spatial graph is derived from the point cloud, object geometry, spatial tools and architectural rules. It gives priority to geometric/spatial evidence, then records locally validated architectural relations. It does not assign material, historical interpretation, or architectural aggregate identity by itself.

Examples:

```text
column_1 near column_2
vault_1 above floor_1
door_window_1 adjacent_to wall_1
floor_1 supports column_1
door_window_1 is_opening_in wall_1
moldings_1 is_ornament_of wall_1
```

### CSV/detail Enrichment

CSV/detail enrichment is not a graph. It collects and links external knowledge to the scene.

Element-level annotations from the scene CSV:
   - material
   - typology
   - function
   - description
   - historical/material notes

Material type is determined only from the CSV attached to the scene. It is not inferred from the point cloud, geometry, semantic class, RGB, roughness, or spatial graph relations.

Aggregate annotations can be derived from geometric tools or validation and then added to the scene annotation data. They are used to support the final scene-level description.

### CIDOC/KG

CIDOC/KG is the ontological layer. It uses CIDOC-CRM patterns to formalize the information from the spatial graph and CSV/detail enrichment.

The CIDOC/KG graph contains:

```text
Element_N crm:P2_has_type Tipo_*
Element_N crm:P45_consists_of Materiale_*
Element_N crm:P103_was_intended_for Funzione_*
Aggregate_N crm:P46_is_composed_of Element_N
```

CIDOC element-to-element or aggregate-to-element relations are created only when they are supported by local spatial evidence from the spatial graph or explicit aggregate evidence from CSV/detail enrichment. The system must not connect distant elements only because they are semantically compatible.

For relationship queries, `list_relationships` is the primary tool for inspecting spatial graph relations and existing graph relations. CIDOC aggregate interpretation is handled by the CIDOC/KG builder and documented in `docs/cidoc_l3_scene_graph_builder.md`.

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
| `--distance-threshold` | `2.0` | Max centroid distance (m) for spatial relationships |
| `--sample-n` | `150000` | Max points to load (0 = no limit) |
| `--use-normals` | `False` | Poisson-based surface area (slower, more accurate) |
| `--annotation-match-threshold` | `2.0` | Max distance (m) to match a CSV row to a single object |
| `--annotation-class-region-tolerance` | `0.5` | Max distance (m) for a CSV row to count as describing a whole semantic region (0 disables) |
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

## Benchmark

`benchmark.py` runs the official question set against one or more Ollama
models and scores the answers against an approved structured reference. It
binds a restricted tool set to the agent, so diagnostic and pipeline-mutation
tools stay out of the measurement.

### Ablation conditions

The benchmark runs in one of two conditions, selected with `--condition`.
Both use the same point cloud, the same DBSCAN segmentation and the same
spatial graph — on `scena4_VAL` both give 33 objects and 170 edges — so the
only variable is whether the CSV knowledge layer exists at all.

| Condition | Flag | Pipeline | Tools bound | Reachable by the model |
|---|---|---|---:|---|
| With CSV | `--condition full` (default) | annotation CSV loaded and matched to objects | 11 | geometry, spatial relations, material, typology, function, description |
| Without CSV | `--condition graph` | annotation CSV never resolved | 8 | geometry and spatial relations only |

Withholding the CSV takes more than hiding the four CSV tools: matched values
are written into the scene-graph nodes, and `get_object_info` prints them
from there. `--condition graph` therefore skips loading the CSV entirely, so
material, typology, function and description are absent from the graph.

### Running both conditions

```bash
# With CSV - the standard run
python benchmark.py /path/to/scena4_VAL.laz \
  --annotation-csv /path/to/scena4_VAL_annotations.csv \
  --condition full \
  --models llama3.1 gemma4:31b qwen3.5 command-r gpt-oss:20b

# Without CSV - the ablation
python benchmark.py /path/to/scena4_VAL.laz \
  --condition graph \
  --models llama3.1 gemma4:31b qwen3.5 command-r gpt-oss:20b
```

Pass `--annotation-csv` explicitly on `full` runs. Without it the loader
auto-discovers a CSV next to the LAZ file, and two different CSVs give two
different `full` conditions. The flag has no effect under `--condition graph`.

Use `--limit 5` for a smoke run before a full sweep. On the September 2026
runs, mean latency per question ranged from 1.3 s (`llama3.1`) to 14.4 s
(`gemma4:31b`), so the 60-question set takes roughly 2 to 15 minutes per
model.

### Reports

Reports land in `benchmark_results/` unless `--output-dir` says otherwise,
named `benchmark_<kind>_<scene>_<model>_<date>_test_<n>.{json,csv}` with
`kind` one of `raw`, `evaluation` and `manual_review`. Runs under
`--condition graph` add a `_cond_graph` suffix to the model segment, so the
two conditions never overwrite each other, and every record carries a
`condition` field.

### Scoring

Answers are scored against
`benchmark/references/<scene>_reference_draft.json`, which must carry
`"status": "approved"`. Per question it pins the expected facts, the
preferred and acceptable tools, and any forbidden claims. Use
`--reference-file` to point at a different one.

Under `--condition graph` the CSV values are unreachable, so questions that
depend on them are scored on whether the model says so instead of supplying
a value. The reference field `expected_without_csv` declares the expectation:

| Value | Expected answer | Flagged when violated |
|---|---|---|
| `abstention` | the value is not available | `missing_abstention` when a value is asserted anyway, `csv_value_without_source` when the CSV value itself is reproduced |
| `abstention_or_role_only` | not available, or the architectural role from `semantic_schema` | as above |
| `manual_review` | the question changes meaning without the CSV | not scored automatically |

Questions whose preferred tools are withheld in this condition are reported
as `not_applicable` for tool routing rather than penalised for it.

## Configuration

The main project configuration is split between Python schema/rules, CSV scene metadata, and the agent prompt:

**`arch_agent/semantic_schema.py`** - semantic class registry, roles, aliases and architectural rule hints. This is where class IDs, structural/finishing categories, support/resting pairs and part/ornament/opening relationships are defined.

**Scene annotation CSV** - optional per-scene metadata linked through `--annotation-csv`. Use it for material, typology, function, historical notes and object descriptions. Matching is based on `semantic_label` plus `global_box_center_x/y/z`.

**`prompts/system.md`** - system prompt for the agent, edit freely to change its tone or instructions.

## Project structure

```
arch_agent/semantic_schema.py  # semantic class registry and architectural rule hints
prompts/
    system.md              # agent system prompt (editable)
arch_agent/
    settings.py            # cached runtime config derived from semantic_schema.py
    pipeline/
        loader.py          # LAZ -> DataFrame
        segmentation.py    # DBSCAN object extraction
        features.py        # geometric feature computation
        relationships.py   # spatial graph relationship detection
        graph.py           # NetworkX DiGraph builders
        pipeline.py        # PipelineParams, SceneContext, run_pipeline()
    tools/
        registry.py        # tool assembly, TOOL_ORDER and the benchmark tool sets
        inventory_tools.py # semantic classes, object counts, object lists
        geometry_tools.py  # dimensions, areas, volumes, distances
        annotation_tools.py# material, typology, function (CSV layer only)
        relationship_tools.py # spatial-graph queries
        scene_state_tools.py  # scene statistics and reload
        _shared.py         # helpers used by more than one tool module
        scene_tools.py     # compatibility shim re-exporting the public names
    benchmark/
        harness.py         # benchmark run loop, scoring and reports
        grounding_checks.py# deterministic groundedness checks
        structured_reference.py # loading of the approved per-question reference
    agent.py               # LangGraph agent + conversation loop
    __init__.py
main.py                    # CLI entry point
benchmark.py               # benchmark CLI (see the Benchmark section)
```

## CIDOC/KG ontology layer

The CIDOC/KG layer is a CIDOC-CRM based knowledge graph for cultural heritage interpretation.

- The spatial graph remains the main scene graph derived from the point cloud and validated architectural rules.
- CSV/detail enrichment is not a graph: it links CSV element metadata and validated aggregate evidence to the scene.
- Material, typology and function values in CSV/detail enrichment and CIDOC/KG come only from the scene annotation CSV.
- CIDOC/KG builds a CIDOC-oriented ontology graph from the spatial graph and CSV/detail enrichment without inventing missing values.
- Element-to-element CIDOC relations are created only when supported by local spatial evidence from the spatial graph or explicit scene evidence.

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
