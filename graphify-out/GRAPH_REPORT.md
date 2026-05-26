# Graph Report - .  (2026-05-26)

## Corpus Check
- 40 files · ~119,528 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 419 nodes · 769 edges · 26 communities detected
- Extraction: 66% EXTRACTED · 34% INFERRED · 0% AMBIGUOUS · INFERRED: 259 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `LowRankStateGenerator` - 31 edges
2. `BeatSegmenter` - 27 edges
3. `BeatEncoder` - 26 edges
4. `VCGPseudoInverse` - 25 edges
5. `BeatDecoder` - 25 edges
6. `BeatStitcher` - 25 edges
7. `GlobalRREmbedding` - 25 edges
8. `ECGRefinementDecoder` - 25 edges
9. `GeometricLeadProjection` - 24 edges
10. `Loss functions for LVCG training.` - 21 edges

## Surprising Connections (you probably didn't know these)
- `LVCG pretraining on MIMIC-IV ECG (self-supervised reconstruction).` --uses--> `Config`  [INFERRED]
  scripts/train.py → lvcg/utils/config.py
- `Loss functions for LVCG training.` --uses--> `AIREADIConditionProvider`  [INFERRED]
  lvcg/models/losses/__init__.py → probing/datasets/aireadi/provider.py
- `Loss functions for LVCG training.` --uses--> `DatasetBundle`  [INFERRED]
  lvcg/models/losses/__init__.py → probing/datasets/base.py
- `Loss functions for LVCG training.` --uses--> `DatasetProvider`  [INFERRED]
  lvcg/models/losses/__init__.py → probing/datasets/base.py
- `Loss functions for LVCG training.` --uses--> `BenchmarkProvider`  [INFERRED]
  lvcg/models/losses/__init__.py → probing/datasets/benchmark_provider.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.1
Nodes (52): BeatDecoder, BeatEncoder, BeatStitcher, GlobalRREmbedding, ResNet1D encoder for VCG beats.          Input: [B, N, 3, P=128] - N beats, each, Encode VCG beats to state vectors.                  Args:             beats: [B,, ResNet-style decoder from beat state to VCG waveform.          Input: z [B, N, D, Decode state vectors to VCG beats.                  Args:             z: [B, N, (+44 more)

### Community 1 - "Community 1"
Cohesion: 0.07
Nodes (39): ABC, DatasetBundle, DatasetProvider, Abstract base class for dataset providers used by the probing pipeline.  A ``Dat, Output of :meth:`DatasetProvider.build`.      Attributes:         train_loader:, Backward-compatible tuple unpacking used by ``run_probing.py``., Source of train/val/test ECG :class:`~torch.utils.data.DataLoader` triples., BenchmarkProvider (+31 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (20): ContextMixer, DecoderResBlock, DynamicsHead, EmbeddingReadout, Beat Modules for LVCG.  Contains all beat-level processing modules: - BeatEncode, Transformer encoder for cross-beat context aggregation.          Input: z_fused, Contextualize beat states.                  Args:             z_fused: [B, N, D], Residual block for 1D signal.          Structure: x -> Conv -> BN -> GELU -> Con (+12 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (26): load_mimic_manifest(), load_wfdb_record(), MimicLoader, normalize_ecg(), preprocess_mimic_record(), MIMIC-IV ECG data loading utilities., Preprocess raw ECG data: reorder leads, resample, bandpass filter, and normalize, Convenience class to iterate over MIMIC records from a manifest. (+18 more)

### Community 4 - "Community 4"
Cohesion: 0.06
Nodes (22): Beat Segmentation for LVCG.  Pipeline Role: VCG [B, 3, T] + ECG [B, 12, T] -> V_, ClassificationHead, Classification heads for downstream tasks., Simple classification head on top of W [B, D].      Supports:         - 'linear', masked_reconstruction_loss(), random_lead_mask(), Loss utilities for LVCG training., Generate random lead masks for training.          Args:         batch_size: Numb (+14 more)

### Community 5 - "Community 5"
Cohesion: 0.16
Nodes (19): already_done(), append_csv(), _collect_probs_labels(), compute_all_metrics(), _compute_auroc_cached(), extract_embeddings(), _find_optimal_thresholds(), init_csv() (+11 more)

### Community 6 - "Community 6"
Cohesion: 0.09
Nodes (21): compute_lead_directions(), compute_lead_directions_np(), directions_to_angles(), get_lead_angles(), get_lead_directions(), get_visible_lead_angles(), get_visible_lead_directions(), Lead direction vectors and reorder utilities for ECG multi-lead reconstruction. (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (13): BaseEncoder, Frozen ECG encoder with a uniform ``ext_ecg_emb`` API.      Subclasses must set, BaseEncoder, Loss functions for LVCG training., ext_ecg_emb(), LVCGEncoder, LVCG encoder wrapper for linear probing., Frozen LVCG backbone for embedding extraction.      Operates at 100 Hz with ``ti (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (12): add_cli_overrides(), apply_overrides(), Config, load_config(), Configuration loading and management., Structured config backed by a plain dict.      Notes:         - We intentionally, Load a config file (YAML or JSON).      Notes:         - We prefer stdlib; howev, Add minimal CLI overrides for convenience.      We allow changing the config pat (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (10): DepthwiseConvDecoder, MultiScaleDecoder, ECG Refinement Decoder for post-projection signal enhancement.  This module prov, Depthwise separable convolution decoder for efficiency.          Uses depthwise, Args:             num_leads: Number of leads             hidden_dim: Hidden dime, Args:             x: [B, L, T]                  Returns:             out: [B, L,, Multi-scale convolution decoder with different kernel sizes.          Captures b, Args:             num_leads: Number of leads             hidden_dim: Hidden dime (+2 more)

### Community 10 - "Community 10"
Cohesion: 0.18
Nodes (7): BenchmarkECGDataset, _normalize_ecg(), Native PyTorch datasets for PTB-XL, ICBEB (CPSC 2018), and Chapman (CSN).  Split, Match MIMIC lead order: swap aVL and aVF (indices 4 and 5)., Single-split ECG dataset for linear probing benchmarks., _swap_avl_avf(), Dataset

### Community 11 - "Community 11"
Cohesion: 0.15
Nodes (5): VCG (Vectorcardiogram) modules for ECG multi-lead reconstruction.  This module p, Initialize with direction vectors (preferred) or angles (backward compatible)., Generate latent VCG trajectories V [B, 3, T'] from state W [B, D].          DEPR, Args:             eps: Regularization coefficient for pseudo-inverse stability, VCGGenerator

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (6): Predict next state directly from current state.                  Args:, Compute gradients for U and V using chain rule.                  Loss: L = ||z_p, Update U, V with gradient descent (with gradient clipping).                  Arg, Direct state prediction with TTT (NOT delta prediction!).                  Data, Inference mode (same as forward, TTT still active).                  Args:, Low-rank MLP forward pass.                  Args:             z_proj: [B, D'] -

### Community 13 - "Community 13"
Cohesion: 0.43
Nodes (6): _load_task_config(), main(), _person_condition_matrix(), Return every person_id mentioned in ``condition_occurrence.csv``., Build a person × condition binary matrix.      A person is positive for a condit, _scan_persons()

### Community 14 - "Community 14"
Cohesion: 0.53
Nodes (5): main(), Return person_ids that have at least one ECG recording., _read_label_persons(), _read_manifest_persons(), _read_participants()

### Community 15 - "Community 15"
Cohesion: 0.33
Nodes (3): Segment VCG into beats.                  Args:             vcg: [B, 3, T] - VCG, Detect R-peaks in a single lead signal.                  Args:             lead_, Get beat boundaries from R-peaks, ensuring full coverage of [0, T].

### Community 16 - "Community 16"
Cohesion: 0.7
Nodes (4): _check_chapman(), _check_icbeb(), _check_ptbxl(), main()

### Community 17 - "Community 17"
Cohesion: 0.5
Nodes (2): Recover VCG from visible leads via pseudo-inverse.                  Supports two, Recover VCG from visible leads using direction vectors (preferred API).

### Community 18 - "Community 18"
Cohesion: 0.5
Nodes (2): Args:             state_dim: State dimension D (default: 256)             proj_d, Initialize projection weights with Xavier uniform.

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (2): _iter_wfdb_stems(), main()

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (1): Extract a fixed-size embedding from raw 12-lead ECG.          Args:

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): Return a :class:`DatasetBundle` for the requested configuration.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (1): Penalize curvature via second differences along time.

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (1): Penalize overall energy to prevent scale drift.

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Encourage start ~ end if applicable (optional).

## Knowledge Gaps
- **120 isolated node(s):** `Convert ``cfg['datasets']`` into a uniform ``{name: provider_cfg}`` map.      A`, `Build train/val/test loaders via a :class:`DatasetProvider`.      Training subse`, `Compute AUROC, F1, Accuracy, Precision, Recall on cached embeddings.      If ``v`, `Abstract base class for dataset providers used by the probing pipeline.  A ``Dat`, `Frozen ECG encoder with a uniform ``ext_ecg_emb`` API.      Subclasses must set` (+115 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 20`** (2 nodes): `evaluate.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `Extract a fixed-size embedding from raw 12-lead ECG.          Args:`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `Return a :class:`DatasetBundle` for the requested configuration.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `Penalize curvature via second differences along time.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `Penalize overall energy to prevent scale drift.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Encourage start ~ end if applicable (optional).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Loss functions for LVCG training.` connect `Community 7` to `Community 0`, `Community 1`, `Community 2`, `Community 3`, `Community 4`?**
  _High betweenness centrality (0.291) - this node is a cross-community bridge._
- **Why does `LowRankStateGenerator` connect `Community 0` to `Community 18`, `Community 4`, `Community 12`, `Community 7`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `AIREADIConditionProvider` connect `Community 1` to `Community 7`?**
  _High betweenness centrality (0.057) - this node is a cross-community bridge._
- **Are the 21 inferred relationships involving `LowRankStateGenerator` (e.g. with `StateGRU` and `LinearPredictor`) actually correct?**
  _`LowRankStateGenerator` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `BeatSegmenter` (e.g. with `Loss functions for LVCG training.` and `StateGRU`) actually correct?**
  _`BeatSegmenter` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 21 inferred relationships involving `BeatEncoder` (e.g. with `StateGRU` and `LinearPredictor`) actually correct?**
  _`BeatEncoder` has 21 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `VCGPseudoInverse` (e.g. with `StateGRU` and `LinearPredictor`) actually correct?**
  _`VCGPseudoInverse` has 20 INFERRED edges - model-reasoned connections that need verification._