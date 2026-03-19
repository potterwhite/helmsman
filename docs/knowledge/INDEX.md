# Knowledge Base Index — Edge AI Deployment Engineer Curriculum

> **Goal**: Systematic self-study material for becoming an edge deployment engineer
> **Focus**: RKNN (RK3588), portrait matting, model optimization
> **Last Updated**: 2026-03-19

---

## Learning Path (Recommended Reading Order)

### Tier 1: Absolute Fundamentals (Read First)
| ID | Topic | File | Key Concepts |
|----|-------|------|-------------|
| KB-001 | Input Shape vs Conv Kernel | `KB-001_input-shape-vs-conv-kernel.md` | NCHW, resolution, aliasing, pixel budget |
| KB-002 | Backbone, ImageNet, MobileNetV2 | `KB-002_backbone-imagenet-mobilenetv2.md` | Feature extractor, lightweight arch, enc_channels |
| KB-003 | Epoch, Batch, Iteration | `KB-003_epoch-batch-iteration.md` | Training loop structure, BN vs batch size, LR schedule |
| KB-004 | Forward/Backward Pass | `KB-004_forward-backward-pass.md` | Loss, gradient, backpropagation, SGD, optimizer |

### Tier 2: Training Deep Dive
| ID | Topic | File | Key Concepts |
|----|-------|------|-------------|
| KB-006 | Pretrained Weights & Transfer Learning | `KB-006_pretrained-weights-transfer-learning.md` | Checkpoint, state_dict, fine-tuning, from scratch |
| KB-008 | Overfitting & Regularization | `KB-008_overfitting-underfitting-regularization.md` | Generalization, augmentation, dropout, early stopping |

### Tier 3: Model Compression & Deployment
| ID | Topic | File | Key Concepts |
|----|-------|------|-------------|
| KB-005 | Distillation, Pruning, Quantization | `KB-005_distillation-pruning-quantization.md` | INT8, calibration, teacher-student, structured pruning |
| KB-007 | ONNX, RKNN, TensorRT | `KB-007_onnx-rknn-tensorrt.md` | Model formats, op fusion, static shape, Netron |

### Tier 3.5: Industry Literacy
| ID | Topic | File | Key Concepts |
|----|-------|------|-------------|
| KB-009 | AI Buzzword Dictionary | `KB-009_ai-buzzword-dictionary.md` | Transformer, GAN, edge AI, inference, latency, TOPS |

### Tier 4: Advanced (Coming Soon)
| ID | Topic | Status |
|----|-------|--------|
| KB-010 | Zero-Copy vs Non-Zero-Copy Inference | Planned |
| KB-010 | Memory Bandwidth Bottleneck on NPU | Planned |
| KB-011 | Guided Filter & AI+CV Hybrid Pipeline | Planned |
| KB-012 | Temporal Smoothing & Video Stability | Planned |
| KB-013 | RKNN Profiling & Performance Analysis | Planned |
| KB-014 | SOC Adaptation (Self-Supervised Domain Transfer) | Planned |
| KB-015 | Trimap, Alpha Matte, Compositing Basics | Planned |

---

## Quick Reference: Project-Specific Terms

| Term | Meaning in Our Project |
|------|------------------------|
| **MODNet** | The portrait matting model we're deploying |
| **Pure BN** | Our modified MODNet with InstanceNorm replaced by BatchNorm |
| **P3M-10k** | Training dataset: 9,421 portrait images with alpha masks |
| **Phase 1** | Model retraining (current) |
| **Phase 2** | INT8 quantization for RKNN |
| **Phase 3** | Guided Filter post-processing for hair detail |
| **Phase 4** | Video temporal stability |
| **helmsman** | The parent project / inference engine |
| **RK3588** | Target edge device (6 TOPS NPU) |

---

## How to Use This Knowledge Base

1. **Sequential study**: Follow the reading order above for structured learning
2. **Reference lookup**: Search for specific terms in the vocabulary tables
3. **Self-test**: Every KB ends with 5 self-check questions with answers
4. **Cross-reference**: KBs link to each other where concepts overlap
5. **Living document**: New KBs will be added as we progress through phases
