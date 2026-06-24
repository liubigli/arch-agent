# arch-agent

An LLM-powered agent for interactive analysis of 3D architectural point clouds.

Given a semantically labelled point cloud (LAZ), the system builds a **three stratified scene graphs** of the architectural space and launches a conversational agent that lets you explore it in natural language.

## How it works

```
LAZ point cloud
      │
      ▼
┌───────────────────────────────────────────────────────┐
│               Pipeline                                │
│  1. Load & sample points                              │
│  2. DBSCAN segmentation -> individual objects         |
|   - DBSCAN for all semantic classes                   |
│  3. Geometric features (volume, area, …)              │
│  4. Spatial relationships between objects             │
|      - L1 geometric                                   |
|      - L2 structural                                  |
|     - L3 mereological                                 │
│  5. Scene graph (NetworkX DiGraph)                    │
│      - geometric graph                                │
|      - structural graph                               │
|      - mereological graph                             │
└───────────────────────────────────────────────────────┘

      │
      ▼
┌─────────────────────────────────────────────┐
│         LangGraph Agent (Llama 3)           │
│  Tools:                                     │
│  • list_objects                             │
│  • get_object_info                          │
│  • find_relationships                       │
│  • list_relationships                       │
│  • find_relationship_anomalies              │
│  • get_scene_statistics                     │
│  • get_point_cloud_info                     │
│  • get_color_summary                        │
│  • estimate_room_volume                     │
│  • measure_distance                         │
│  • find_nearest_objects                     │
│  • find_focal_points                        │
│  • find_pattern                             │
│  • discover_functional_areas                │
│  • reload_scene                             │
└─────────────────────────────────────────────┘
      │
      ▼
  Interactive chat (terminal)
```

## Input format

The input can be a LAZ file or a directory containing `.laz` files. When a directory is provided, the CLI lists the available `.laz` files and asks which one to load. The selected file must contain semantic labels in either a `semantic_label` extra dimension or the standard `classification` dimension. Optional RGB channels and normals are preserved when available.

```
semantic_label or classification
optional: red;green;blue;nx;ny;nz
```

Supported semantic labels (integer-encoded):

| ID | Class       | Type       |
|----|-------------|------------|
| 0  | arch        | structural |
| 1  | column      | structural |
| 2  | moldings    | finishing  |
| 3  | floor       | finishing  |
| 4  | door_window | finishing  |
| 5  | wall        | structural |
| 6  | stairs      | finishing  |
| 7  | vault       | structural |
| 8  | roof        | structural |
| 9  | other       | finishing  |


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


```md
## Stratified relationship model

The scene is represented through three relationship levels. Each level is stored as a separate `networkx.DiGraph`, avoiding duplicated edges inside a graph while preserving multiple interpretations of the same object pair across levels.

| Level | Graph | Relations | Meaning |
|---|---|---|---|
| L1 | geometric | `near`, `adjacent_to`, `above`, `below` | spatial position and proximity |
| L2 | structural | `supports`, `rests_on` | vertical support and load-bearing interpretation |
| L3 | mereological | `part_of`, `has_part`, `is_opening_in`, `is_rib_of`, `is_ornament_of`, `is_attached_to`, `is_placed_on`, `is_connected_to` | architectural composition and functional membership |

The three graphs are available in the scene context:

```python
ctx.scene_graphs["L1"]  # geometric graph
ctx.scene_graphs["L2"]  # structural graph
ctx.scene_graphs["L3"]  # mereological graph




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

Two files can be customized without touching Python code:

**`config.yaml`** — semantic class definitions:
```yaml
semantic_classes:
  names: [arch, column, moldings, ...]   # label id → class name
  structural: [arch, column, wall, ...]  # used for element type classification
  finishing: [moldings, floor, ...]
  colors:                                # RGB in [0, 1], used for visualization
    arch: [0.85, 0.37, 0.01]
    ...
```

**`prompts/system.md`** — system prompt for the agent, edit freely to change its tone or instructions.

## Project structure

```
config.yaml                # semantic class definitions (editable)
prompts/
│   └── system.md          # agent system prompt (editable)
arch_agent/
├── settings.py            # YAML config loader (lru_cache)
├── pipeline/
│   ├── loader.py          # LAZ → DataFrame
│   ├── segmentation.py    # DBSCAN object extraction
│   ├── features.py        # geometric feature computation
│   ├── relationships.py   # spatial relationship detection L1/L2/L3
│   ├── graph.py           # NetworkX DiGraph builders
│   └── pipeline.py        # PipelineParams, SceneContext, run_pipeline()
├── tools/
│   └── scene_tools.py     # LangChain tools wrapping the scene graph
├── agent.py               # LangGraph agent + conversation loop
└── __init__.py
main.py                    # CLI entry point
```
