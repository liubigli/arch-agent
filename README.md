# arch-agent

An LLM-powered agent for interactive analysis of 3D architectural point clouds.

Given a semantically labelled point cloud (CSV), the system builds a **scene graph** of the architectural space and launches a conversational agent that lets you explore it in natural language.

## How it works

```
CSV point cloud
      │
      ▼
┌─────────────────────────────────────────────┐
│               Pipeline                      │
│  1. Load & sample points                    │
│  2. DBSCAN clustering → individual objects  │
│  3. Geometric features (volume, area, …)    │
│  4. Spatial relationships between objects   │
│  5. Scene graph (NetworkX DiGraph)          │
└─────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────┐
│         LangGraph Agent (Llama 3)           │
│  Tools:                                     │
│  • list_objects                             │
│  • get_object_info                          │
│  • find_relationships                       │
│  • get_scene_statistics                     │
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

The input CSV must use `;` as delimiter with the following columns:

```
x;y;z;R;G;B;nx;ny;nz;semantic_label
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
# Basic usage with default parameters
python main.py path/to/scene.csv

# Tune DBSCAN clustering (smaller eps = tighter clusters)
python main.py path/to/scene.csv --eps 0.3 --min-samples 10

# Extend the spatial relationship radius and use a different model
python main.py path/to/scene.csv --distance-threshold 5.0 --model llama3.1

# Use Poisson reconstruction for more accurate surface area estimates
python main.py path/to/scene.csv --use-normals
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
│   ├── loader.py          # CSV → DataFrame
│   ├── segmentation.py    # DBSCAN object extraction
│   ├── features.py        # geometric feature computation
│   ├── relationships.py   # spatial relationship detection
│   ├── graph.py           # NetworkX scene graph builder
│   └── pipeline.py        # PipelineParams, SceneContext, run_pipeline()
├── tools/
│   └── scene_tools.py     # LangChain tools wrapping the scene graph
├── agent.py               # LangGraph agent + conversation loop
└── __init__.py
main.py                    # CLI entry point
```
