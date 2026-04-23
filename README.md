# TrajGenAgent
 Official Code Implementation of TrajGenAgent

## Overview
This repository contains the official implementation of our human mobility trajectory generation framework TrajGenAgent, as submitted to **IEEE MDM 2026**. This framework generates realistic, context-coherent human mobility trajectories by leveraging Large Language Models (LLMs) and agent workflows. It assumes all modeled days are "active days" (i.e., multiple visits outside the home) and utilizes a robust LLM workflow of in-context learning, similar agent data augmentation, semantic-aware activity chain generation and physics/time-budget-aware spatiotemporal grounding.

By treating trajectory generation as a hierarchical process, our pipeline transform LLM-generated activity chains into high-fidelity daily trajectories with spatiotemporal grounding workflow.

## Pipeline Workflow

### Phase 1: Colocation Extraction & Agent Similarity Matching
Before generation, the framework enriches the target agent's context by finding historically similar peers, which serves as data augmentation for the LLM's in-context learning.
* **`colocation_extract.py`**: Extracts spatiotemporal colocation features from the raw ground truth trajectories to capture interpersonal mobility interactions.
* **`peer_agents_match.py`**: Utilizes the extracted spatiotemporal features and colocation indices to match target agents with similar peer agents. This forms an augmented ground truth dataset used to guide the LLM.

### Phase 2: LLM-Based Activity Chain Generation
* **`llm_activity_chain_gen.py`**: Generates the logical sequence of activities for a given day/agent. By employing LLM in-context learning over the augmented agent data (Individual Contexts) and validity constraints, the model outputs semantically logical and realistic daily activity chains.
* **`retrieve_act_chain.py`**: Serves as a retrieval-based baseline or utility for fetching existing activity sequences for comparative analysis. (Not applied in the paper)

### Phase 3: Spatiotemporal Grounding Workflow
* **`location_time_tools_llm.py`**: Translates the abstract activity chains into concrete trajectory data with exact coordinates and timestamps. This operates as a state machine with three core components:
  * **Location Selection:** A rule-based retrieval system combined with spatial gravity scoring to select exact POIs.
  * **Travel Time Estimation:** Kinematics-aware calculations based on historical modality speeds and calculated geographical distances.
  * **Duration Assignment (Time Budget-aware LLM):** An LLM generation node that dynamically determines stay duration by weighing the agent's historical activity preferences against the remaining daily time budget in the current context.
* **`location_time_tools_ablation.py`**: Scripts dedicated to running ablation studies on the pure rule-based grounding modules.

### Phase 4: Post-Processing & Finalization
* **`process_gen_traj.py`**: Post-processes the raw generated outputs.

## Execution Instructions

**Phase 3 (Spatiotemporal Grounding) requires a persistent `vLLM` server instance.** Because activity durations and travel times are calculated on-the-fly and depend strictly on the immediate context of the previously generated step, the workflow operates as an interactive state machine. Then it must provision a dedicated, persistent vLLM API server endpoint (see `run_stage2_workflow_vllm_config_example`) that the grounding scripts can query continuously during execution.

**Hardware Recommendations:**
The default settings are calibrated for high-performance GPUs. To achieve optimal performance, **NVIDIA H100** GPUs are highly recommended, though A100s can be utilized with specific compromises (e.g., smaller LLM variants, reduced precision, or shorter contexts).

***

### Directory & File Structure Map
* **Data Directories:** * `raw_ground_truth_trajectory/` / `data/` - Initial input datasets.
  * `colocation/` / `extracted_colocation/` - Intermediary interaction data.
  * `generated_activity_chain/` / `generated_raw_trajectory/` - Intermediate LLM outputs.
  * `statistical_eval/` - Output directory for evaluation metrics.
* **Core Scripts:**
  * `colocation_extract.py`
  * `peer_agents_match.py`
  * `llm_activity_chain_gen.py`
  * `retrieve_act_chain.py`
  * `location_time_tools_llm.py`
  * `location_time_tools_ablation.py`
  * `process_gen_traj.py`
* **Infrastructure:**
  * `run_stage2_workflow_vllm_config_example` - Reference configuration for setting up the persistent vLLM server.
