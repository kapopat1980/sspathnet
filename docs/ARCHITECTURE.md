# Paper concept -> code map

All model code lives in `scripts/plantvillage_part1.py` and `scripts/plantvillage_part2.py` (the model/engine
section is byte-identical in both files, so that each can run alone in its own Kaggle session).

| Paper (Section 3.2) | Class / function | Notes |
|---|---|---|
| Dual-stream chromatic-frequency stem | `ChromaFreqStem(out_ch, mode)` | `mode` in `rgb`, `rgb+hsv`, `rgb+dct`, `full` (ablations A1-A3) |
| Continuous HSV `[s cos 2*pi*h, s sin 2*pi*h, s, v]` | `RGB2HSVFeat` | no learnable parameters |
| Fixed 4x4 block-DCT on Y/Cb/Cr + `sign(z)*log(1+4|z|)` | `BlockDCT` | stride 4, 48 channels |
| Stream-wise softmax gating, Eq. (1) | `ChromaFreqStem.forward` | `self.gate` |
| Deformable depthwise 3x3 conv, Eq. (2) | `DeformDWConv(deform=True/False)` | implemented with `grid_sample`; offsets zero-initialised, bounded by `3*tanh(o/3)`; `deform=False` = ablation A4 |
| Deformable-inception mixer | `InceptionDeformMixer` | deformable DW3x3 / dilated DW3x3 / strip 1x7+7x1 / avg-pool |
| Windowed cross-attention | `WindowCrossAttention` | queries from a depthwise-conv "local stream", keys/values from tokens; 5x5 windows |
| DIT block | `DITBlock(kind='conv'|'attn')` | alternates; `use_attn=False` = ablation A5 |
| TCFA, Eq. (3) | `TCFA`, `ChannelSpatialGate` | `agg='tcfa'`; `'none'` = A6, `'concat'` = A7 |
| Adaptive activation | `AdaptiveAct` | `head_act='gelu'` = A9 |
| Center loss, Eq. (4) | `CenterLoss` | weight 0.005; `use_center=False` = A8 |
| Full network | `SSPathNet(num_classes, stem, deform, use_attn, agg, head_act, ...)` | 9.797 M parameters |
| Baselines | `build_baseline(name, K)` | torchvision, `weights=None` (random init) |
| Robustness suite | `PERTURBATIONS`, `robustness_eval` | 8 conditions, fixed random streams |
| Grad-CAM | `gradcam_figure` | stage-3 feature map (`cam_stage=2`), predicted class |
| Training loop / time guard / resume | `run_experiment`, `safe_run` | AdamW, cosine schedule, AMP, best-val checkpoint |

Class indices in Grad-CAM titles and confusion matrices follow `results/part1/classes.json` (alphabetical folder order).
