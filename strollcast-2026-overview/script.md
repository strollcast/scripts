---
title: "Strollcast 2026 Overview: Your Guide to ML Systems Research"
summary: "A comprehensive introduction to Strollcast and guided tour through our entire catalog of ML systems research episodes, covering training at scale, attention mechanisms, memory optimization, and serving infrastructure."
---

# Strollcast 2026 Overview: Your Guide to ML Systems Research

**ERIC:** Welcome to Strollcast! We're your AI hosts, here to make cutting-edge machine learning research accessible while you're on the move. I'm Eric.

**MAYA:** And I'm Maya. Today we're doing something special—a guided tour through our entire catalog of episodes, giving you the perfect introduction to what Strollcast is all about.

**ERIC:** Think of this as your roadmap to the ML systems landscape. Whether you're a researcher, engineer, or just ML-curious, we've got you covered.

**MAYA:** And if you're wondering what Strollcast actually is, we transform dense academic papers into conversational audio episodes. No jargon walls, no intimidation—just clear explanations of the ideas that are reshaping how we build and deploy machine learning systems.

**ERIC:** We focus on systems research—the infrastructure, optimization techniques, and architectural innovations that make modern AI possible. From training trillion-parameter models to serving them efficiently in production.

**MAYA:** So let's dive into our catalog. We've organized our episodes into four main themes: training systems, memory optimization, attention mechanisms, and serving infrastructure.

## Training at Scale: The Foundation

**ERIC:** Let's start with the big question: how do you train models with hundreds of billions of parameters across thousands of GPUs? Our episodes on training systems tackle exactly that.

**MAYA:** First up is [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://released.strollcast.com/episodes/rajbhandari-2020-zero_memory_optimiz/rajbhandari-2020-zero_memory_optimiz.mp3) from Microsoft Research. This 2020 paper is foundational—it introduced the idea of eliminating memory redundancy by partitioning optimizer states, gradients, and even parameters across GPUs.

**ERIC:** ZeRO showed that you could train massive models without custom parallelism strategies, just by being smarter about what each GPU stores. It's the foundation that enabled so much of what came after.

**MAYA:** Building on that, we have [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://released.strollcast.com/episodes/narayanan-2021-efficient_large_scal/narayanan-2021-efficient_large_scal.mp3) from NVIDIA. This is the comprehensive playbook for combining three types of parallelism hierarchically.

**ERIC:** Megatron-LM showed how to mix tensor parallelism within nodes using fast NVLink connections, pipeline parallelism across nodes, and data parallelism for final scaling. They hit 52% of theoretical peak performance for 175 billion parameter models.

**MAYA:** Then there's [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://released.strollcast.com/episodes/zhao-2023-pytorch_fsdp_experi/zhao-2023-pytorch_fsdp_experi.mp3) from Meta. This is where ZeRO's ideas got productionized into PyTorch itself.

**ERIC:** What I love about the FSDP paper is that it's brutally honest about the engineering challenges. Sharding strategies, communication optimization, debugging distributed training—all the messy reality of making this work at scale.

**MAYA:** And if you want to see how Google approaches the same problems differently, check out [Pathways: Asynchronous Distributed Dataflow for ML](https://released.strollcast.com/episodes/barham-2022-pathways_asynchrono/barham-2022-pathways_asynchrono.mp3). Instead of the tightly synchronized approach of Megatron, Pathways uses futures and asynchronous dataflow.

**ERIC:** Pathways is fascinating because it breaks away from the traditional SPMD model—single program, multiple data. With asynchronous dataflow, different TPUs can run completely different computations while still coordinating efficiently.

**MAYA:** For those who want automation instead of manual tuning, we have [Alpa: Automating Inter- and Intra-Operator Parallelism](https://released.strollcast.com/episodes/zheng-2022-alpa_automating_int/zheng-2022-alpa_automating_int.mp3). This compiler automatically generates distributed training strategies.

**ERIC:** Alpa reorganizes the problem hierarchically: intra-operator parallelism for partitioning individual operations, and inter-operator parallelism for pipeline-style scheduling. It uses integer linear programming and dynamic programming to match hand-tuned Megatron performance automatically.

**MAYA:** And for those interested in architectural foundations, check out [mHC: Manifold-Constrained Hyper-Connections](https://released.strollcast.com/episodes/xie-2025-mhc_manifold_constr/xie-2025-mhc_manifold_constr.mp3). This 2025 paper extends the residual connection paradigm that's been fundamental to deep learning for over a decade.

**ERIC:** The key insight is that while hyper-connections improve performance by diversifying connectivity patterns, they compromise the identity mapping property that makes residual networks stable. mHC projects these connections onto a manifold to restore that property while maintaining the performance gains, enabling better scalability for training at scale.

## Attention Mechanisms: The Core Innovation

**MAYA:** Now let's talk about attention—the mechanism at the heart of transformers. We've got two groundbreaking episodes here.

**ERIC:** First is [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://released.strollcast.com/episodes/dao-2023-flashattention_2_fa/dao-2023-flashattention_2_fa.mp3). This is all about making attention computation blazingly fast by being smart about GPU memory hierarchy.

**MAYA:** FlashAttention-2 achieves 73% of theoretical GPU performance—up from 25-40% in the original FlashAttention. The trick is minimizing memory transfers between high-bandwidth memory and SRAM by reorganizing how the computation flows through GPU thread blocks.

**ERIC:** It's a masterclass in GPU optimization. By reducing non-matrix-multiply operations and improving parallelism across sequence length, they doubled the speed of already-optimized attention.

**MAYA:** And then we have [Gated Attention for Large Language Models](https://released.strollcast.com/episodes/qiu-2025-gated_attention_for/qiu-2025-gated_attention_for.mp3), the NeurIPS 2025 Best Paper Award winner.

**ERIC:** Gated attention is beautifully simple—just add a sigmoid gate after the attention computation. But that simple change solves multiple problems: eliminates attention sinks, prevents massive activations, improves training stability, and enables better long-context extrapolation.

**MAYA:** It's already deployed in production Qwen3-Next models supporting up to 1 million tokens. Sometimes the best innovations are the simplest ones.

## Memory Optimization and Efficient Inference

**ERIC:** Training is one thing, but what about when you need to actually use these massive models? Our episodes on memory optimization and inference tackle that challenge.

**MAYA:** [QLoRA: Efficient Finetuning of Quantized LLMs](https://released.strollcast.com/episodes/dettmers-2023-qlora_efficient_fin/dettmers-2023-qlora_efficient_fin.mp3) showed how to fine-tune 65 billion parameter models on a single consumer GPU. That's democratizing access to large model training.

**ERIC:** The key innovation is combining 4-bit quantization with LoRA adapters. QLoRA introduces 4-bit NormalFloat quantization, double quantization for scaling factors, and paged optimizers. You get full fine-tuning quality at a fraction of the memory cost.

**MAYA:** For inference, [FlexGen: High-Throughput Generative Inference on a Single GPU](https://released.strollcast.com/episodes/sheng-2023-flexgen_high_throug/sheng-2023-flexgen_high_throug.mp3) takes a completely different approach—trading latency for throughput.

**ERIC:** FlexGen lets you run GPT-175B scale models on commodity GPUs by leveraging the full memory hierarchy: GPU memory, CPU memory, and even disk. The zig-zag block schedule and 4-bit compression achieve 100x higher throughput on batch workloads.

**MAYA:** It's perfect for applications like document processing or running benchmarks where you don't need real-time responses but want to maximize work completed.

## Serving and Production Systems

**ERIC:** Finally, let's talk about serving models in production. This is where the rubber meets the road.

**MAYA:** [Punica: Multi-Tenant LoRA Serving](https://released.strollcast.com/episodes/chen-2023-punica_multi_tenant/chen-2023-punica_multi_tenant.mp3) addresses a critical problem: how do you efficiently serve hundreds or thousands of LoRA adapters on shared infrastructure?

**ERIC:** The breakthrough is the SGMV kernel—Segmented Gather Matrix-Vector multiplication. It enables batching requests across different LoRA adapters, achieving 12x higher throughput compared to naive approaches with only 2 milliseconds of additional latency.

**MAYA:** This is huge for companies deploying fine-tuned models at scale. Instead of dedicating separate infrastructure for each adapter, you can serve them all from shared GPUs efficiently.

**ERIC:** And that's our catalog! Eleven episodes covering the full spectrum of ML systems research, from training to serving, from attention mechanisms to memory optimization.

## Why These Papers Matter

**MAYA:** What ties all these papers together is a focus on practical impact. These aren't just theoretical improvements—they're the systems running in production at Google, Meta, Microsoft, NVIDIA, and across the industry.

**ERIC:** Every episode breaks down complex ideas into digestible explanations. We cover the motivation, the key insights, the technical details that matter, and the real-world impact.

**MAYA:** Whether you're optimizing your own training runs, debugging distributed systems, or just trying to understand how modern AI infrastructure works, we've got an episode for you.

**ERIC:** You can listen to them in any order, but we recommend starting with ZeRO and Megatron-LM if you're new to distributed training, or FlashAttention-2 and Gated Attention if you're focused on model architecture.

**MAYA:** All our episodes are available at strollcast.com, and you can find the original papers linked in the episode descriptions. We also have transcripts with timestamps if you prefer to read along.

**ERIC:** New episodes come out regularly as important papers are published. We're always looking for the next breakthrough in ML systems research.

**MAYA:** Before we wrap up, let's test your knowledge with a couple of quick questions!

## Quiz Time!

**ERIC:** First question: Which paper introduced the idea of partitioning optimizer states, gradients, and parameters across GPUs to eliminate memory redundancy in data parallelism?

**PAUSE:1**

**MAYA:** The answer is ZeRO from Microsoft Research! That 2020 paper laid the foundation for training trillion-parameter models by showing how to be smarter about what each GPU stores.

**ERIC:** Second question: FlashAttention-2 achieves what percentage of theoretical GPU performance for attention computation—25%, 50%, or 73%?

**PAUSE:1**

**MAYA:** It's 73%! That's a massive improvement from the 25-40% achieved by the original FlashAttention, making attention computation blazingly fast by optimizing memory access patterns.

## Closing

**ERIC:** That's a wrap on our Strollcast overview! We hope this gives you a clear map of our catalog and inspires you to dive deeper into the episodes that match your interests.

**MAYA:** Remember, we're here to make research accessible. No matter your background, these ideas are within reach.

**ERIC:** Until next time, keep strolling.

**MAYA:** And may your learning rate be just right!
