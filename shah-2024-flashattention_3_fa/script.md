# FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

## Introduction

**ERIC:** Welcome to Strollcast! I'm Eric.

**MAYA:** And I'm Maya. We're your AI hosts, here to make research accessible while you're on the move.

**ERIC:** Today we're diving into FlashAttention-3, a paper that's pushing the boundaries of what's possible with transformer attention. If you've been following the AI space, you know that attention is both the superpower and the bottleneck of modern language models.

**MAYA:** Exactly. This work comes from Jay Shah, Ganesh Bikshandi, and the team including Tri Dao, who's been leading the FlashAttention series. And the results are pretty remarkable - they're getting 1.5 to 2x speedups over FlashAttention-2, reaching up to 75% utilization on H100 GPUs. {{page: 1, section: "Abstract", excerpt: "We demonstrate that our method, FlashAttention-3, achieves speedup on H100 GPUs by 1.5-2.0× with FP16 reaching up to 740 TFLOPs/s (75% utilization)"}}

**ERIC:** To put that in perspective, FlashAttention-2 was only achieving about 35% utilization on the same hardware. So we're talking about more than doubling the hardware efficiency. {{page: 1, section: "Abstract", excerpt: "FlashAttention-2 achieving only 35% utilization on the H100 GPU"}}

**MAYA:** Before we jump into the technical details, let's set the stage. Why does attention performance matter so much right now?

**ERIC:** Great question! Attention is the core operation in transformers, and it's what's holding us back from longer contexts. Think about it - if you want to process a long document, analyze an entire codebase, or work with high-resolution images, you need attention to scale efficiently. {{page: 2, section: "1 Introduction", excerpt: "Scaling attention to longer context will unlock new capabilities (modeling and reasoning over multiple long documents and files in large codebases, new modalities (high-resolution images, audio, video)"}}

## Background: The Hardware Revolution

**MAYA:** So what's changed since FlashAttention-2 that makes these improvements possible?

**ERIC:** The key insight is that FlashAttention-2 was designed for an older model of how GPUs work - essentially a synchronous world where you do one operation, wait for it to finish, then do the next one. But modern GPUs, especially NVIDIA's Hopper architecture in the H100, are fundamentally asynchronous machines.

**MAYA:** Can you break that down? What does "asynchronous" mean in this context?

**ERIC:** Think of it like a restaurant kitchen. In the old synchronous model, you'd have one chef who prepares the appetizer, waits for it to be completely done, then starts the main course. In the asynchronous model, you have specialized stations - one chef handling appetizers, another on mains, another on desserts - all working simultaneously and coordinating through a system. {{page: 4, section: "2.2", excerpt: "Asynchrony is a result of hardware specialization to accelerate the most important operations in a ML workload: specific hardware units performing matrix multiplication (Tensor Cores) or memory loading (Tensor Memory Accelerator – TMA)"}}

**MAYA:** And the H100 has these specialized units?

**ERIC:** Exactly! The H100 has Tensor Cores for matrix multiplication and something called the Tensor Memory Accelerator, or TMA, for moving data around. These can work independently and asynchronously. {{page: 4, section: "2.2", excerpt: "For async memory copy between GMEM and SMEM, Hopper has the Tensor Memory Accelerator (TMA) as a dedicated hardware unit"}}

**MAYA:** There's also this trend toward lower precision arithmetic, right? The paper mentions FP8 support.

**ERIC:** Right! We've seen this progression over the years - from FP32 to FP16, then BF16, and now FP8. Each step roughly doubles your throughput for the same power and silicon area. The H100's FP8 Tensor Cores can deliver about 2x the performance of FP16. {{page: 4, section: "2.2", excerpt: "Low precision such as FP8 in Hopper and FP4 in Blackwell, continuing the trend of FP16 (Pascal in 2017) and BF16 (Ampere in 2020), is a proven technique to get double or quadruple throughput"}}

## Core Technical Contributions

**MAYA:** So FlashAttention-3 has three main technical innovations to exploit these hardware features. Let's start with the first one - producer-consumer asynchrony.

**ERIC:** This is where they implement that restaurant kitchen model I mentioned. They split the GPU warps - think of these as small teams of threads - into specialized roles. Some warps become "producers" that are only responsible for loading data from memory, while others become "consumers" that focus purely on computation. {{page: 5, section: "3.1", excerpt: "We define a warp-specialized software pipelining scheme that exploits the asynchronous execution of data movement and Tensor Cores by splitting producers and consumers of data into separate warps"}}

**MAYA:** And this works because the hardware can do these operations simultaneously?

**ERIC:** Exactly. While the producer warps are using the TMA to load the next batch of data, the consumer warps are using the Tensor Cores to crunch numbers on the current batch. It's like having a conveyor belt where loading and processing happen in parallel.

**MAYA:** The second innovation is about overlapping different types of computation within the attention calculation itself.

**ERIC:** Right, this gets into the nitty-gritty of how attention works. Remember that attention involves matrix multiplications and softmax operations. The matrix multiplications are very efficient on modern hardware - the H100 can do about 989 teraflops of FP16 matrix multiplication. But softmax involves operations like exponentials that are much slower - only about 3.9 teraflops. {{page: 8, section: "3.2", excerpt: "the H100 SXM5 GPU has 989 TFLOPS of FP16 matmul but only 3.9 TFLOPS of special functions such as exponential"}}

**MAYA:** So the idea is to hide the slow softmax computation behind the fast matrix multiplications?

**ERIC:** Exactly! They redesigned the algorithm so that while one warpgroup is computing softmax for one block of data, another warpgroup is simultaneously doing matrix multiplication on a different block. They call this "pingpong scheduling" because the roles alternate back and forth. {{page: 8, section: "3.2", excerpt: "we use synchronization barriers to force the GEMMs of warpgroup 1 to be scheduled before the GEMMs of warpgroup 2. As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs"}}

**MAYA:** That's quite clever. And the third innovation deals with low-precision arithmetic?

**ERIC:** Yes, and this is where things get really tricky. Using FP8 instead of FP16 can theoretically double your performance, but there are two major challenges. First, there are strict layout requirements - the hardware expects data to be arranged in memory in very specific ways for FP8 operations.

**MAYA:** What kind of layout requirements?

**ERIC:** Think of it like organizing books on a shelf. FP16 operations are flexible - you can arrange your "books" either by author or by title. But FP8 operations are picky - they only work efficiently if everything is arranged by title, let's say. {{page: 10, section: "3.3", excerpt: "for FP16 WGMMA, both mn-major and k-major input operands are accepted for operands in SMEM, but for FP8 WGMMA, only the k-major format is supported"}}

**ERIC:** So they had to implement an in-kernel transpose operation to rearrange data on the fly, using specialized instructions called LDSM and STSM.

**MAYA:** And the second challenge with FP8?

**ERIC:** Accuracy. When you compress numbers into fewer bits, you lose precision. This is especially problematic with large language models because they tend to have "outlier" features - a small number of values that are much larger than the rest. {{page: 12, section: "3.3", excerpt: "large models typically have outlier values that are much larger in magnitude than most other values, making quantization difficult"}}

**MAYA:** How do they solve that?

**ERIC:** Two clever techniques. First, "block quantization" - instead of using one scaling factor for an entire matrix, they use different scaling factors for different blocks. This lets them handle outliers more precisely in each local region.

**ERIC:** Second, "incoherent processing" - they multiply the input matrices by a random orthogonal matrix before quantization. This scrambles the data in a way that spreads out the outliers, making them easier to quantize accurately. {{page: 12, section: "3.3", excerpt: "This serves to 'spread out' the outliers since each entry of QM or KM is a random sum of entries of Q or K, thus reducing quantization error"}}

**MAYA:** And because the matrix is orthogonal, it doesn't change the final result?

**ERIC:** Exactly! It's mathematically equivalent but numerically more stable.

## Experimental Results

**MAYA:** Let's talk about how well this actually works in practice. What kind of speedups are they seeing?

**ERIC:** The results are quite impressive. In FP16, they're getting 1.5 to 2x speedups in the forward pass and 1.5 to 1.75x speedups in the backward pass compared to FlashAttention-2. {{page: 13, section: "4.1", excerpt: "FlashAttention-3 is around 1.5-2.0× faster than FlashAttention-2 in the forward pass and 1.5-1.75× faster in the backward pass"}}

**MAYA:** And they're competitive with NVIDIA's own optimized implementation?

**ERIC:** For longer sequences, FlashAttention-3 actually outperforms cuDNN, NVIDIA's highly optimized library. That's particularly impressive because cuDNN is closed-source and specifically tuned for H100 hardware. {{page: 13, section: "4.1", excerpt: "For medium and long sequences (1k and above), FlashAttention-3 even surpasses the speed of a vendor's library (cuDNN – closed source) that has been optimized for H100 GPUs"}}

**MAYA:** What about the FP8 results?

**ERIC:** FP8 reaches close to 1.2 petaflops per second - that's 1,200 teraflops! To put that in perspective, some entire data centers from a few years ago couldn't achieve that kind of compute throughput. {{page: 1, section: "Abstract", excerpt: "with FP8 reaching close to 1.2 PFLOPs/s"}}

**MAYA:** And the accuracy with FP8?

**ERIC:** This is where their block quantization and incoherent processing really shine. They show that FP8 FlashAttention-3 has 2.6x lower numerical error compared to standard FP8 attention implementations. {{page: 1, section: "Abstract", excerpt: "FP8 FlashAttention-3 achieves 2.6× lower numerical error than a baseline FP8 attention"}}

## Ablation Studies and Analysis

**MAYA:** Did they break down which optimizations contribute most to the speedup?

**ERIC:** Yes, they did a nice ablation study. Starting from a baseline of 570 teraflops per second, just adding warp specialization gets them to 582 teraflops. Adding the GEMM-softmax pipelining takes them to 661 teraflops. So both optimizations contribute significantly. {{page: 15, section: "4.2", excerpt: "our algorithmic improvements (asynchrony with warp-specialization and overlapping between GEMM and softmax) lead to significant speedup, from 570 to 661 TFLOPs"}}

**MAYA:** What about numerical accuracy compared to FlashAttention-2?

**ERIC:** In FP16, FlashAttention-3 maintains exactly the same numerical accuracy as FlashAttention-2, and both are actually more accurate than standard attention implementations. This is because they keep intermediate computations in higher precision. {{page: 15, section: "4.3", excerpt: "In FP16, both FlashAttention-2 and FlashAttention-3 achieves 1.7× lower RMSE compared to the standard implementation since intermediate results (softmax) are kept in FP32"}}

## Quizzes

**MAYA:** Let's test your understanding with a couple of questions. First one: FlashAttention-3 achieves better hardware utilization on H100 GPUs primarily by exploiting what key characteristic of modern GPU architecture?

**ERIC:** Take a moment to think about that...

**MAYA:** The answer is asynchrony! The key insight is that modern GPUs like the H100 have specialized hardware units that can work independently - the Tensor Cores for computation and TMA for memory operations. FlashAttention-3 exploits this through warp specialization, where some warps focus on data movement while others focus on computation, allowing both to happen simultaneously.

**ERIC:** Here's the second quiz question: When using FP8 precision, FlashAttention-3 employs "incoherent processing" to improve accuracy. What does this technique do and why does it help?

**MAYA:** Think about the outlier problem in large language models...

**ERIC:** The answer is that incoherent processing multiplies the input matrices by a random orthogonal matrix before quantization. This spreads out the outlier values that are typically concentrated in a few features, making them easier to quantize accurately. Since the matrix is orthogonal, it doesn't change the mathematical result, but it makes the numerical computation more stable in low precision.

## Implications and Future Directions

**MAYA:** What does this mean for the broader AI landscape?

**ERIC:** This is really about unlocking longer context lengths in practical applications. When attention runs 2x faster, you can process 2x longer sequences for the same computational budget. That means better performance on tasks like document analysis, code understanding, and multimodal applications with high-resolution images or videos.

**MAYA:** And the techniques themselves seem quite general.

**ERIC:** Absolutely. While they focused on the H100, the principles of exploiting asynchrony and hardware specialization should apply to other accelerators too. The paper mentions this could work for any GPU architecture with robust asynchronous execution capabilities. {{page: 5, section: "3", excerpt: "our algorithm is operative for any GPU architecture with sufficiently robust asynchronous execution and low-precision capabilities"}}

**MAYA:** What about limitations? The authors mention a few areas they'd like to improve.

**ERIC:** They call out three main areas: optimizing for LLM inference specifically, integrating persistent kernel designs into the FP8 implementation, and better understanding the effects of low-precision attention in large-scale training. {{page: 18, section: "5", excerpt: "optimizing for LLM inference, integrating a persistent kernel design into the FP8 kernel, and understanding the effects of low-precision attention in large-scale training"}}

**MAYA:** The inference optimization is particularly important given how much of the AI workload is shifting toward inference rather than training.

**ERIC:** Right, and they mention that for small sequence lengths and causal masking, their FP8 implementation doesn't quite match cuDNN's performance yet, partly because they haven't implemented the same persistent kernel optimizations.

## Broader Context

**MAYA:** How does this fit into the broader trend of hardware-software co-design in AI?

**ERIC:** This is a perfect example of why you need to understand your hardware deeply to get maximum performance. The FlashAttention series has always been about matching algorithms to hardware characteristics - the original FlashAttention was about minimizing memory transfers, FlashAttention-2 added better parallelization, and now FlashAttention-3 is about exploiting asynchrony and low precision.

**MAYA:** It's also interesting how they're making this available to the community. They mention open-sourcing with a permissive license and integrating with PyTorch and Hugging Face.

**ERIC:** That's crucial for adoption. No matter how fast your algorithm is, if it's hard to use, it won't have impact. By integrating with the standard ML frameworks, they're making these optimizations accessible to everyone building transformer models.

**MAYA:** And the timing is perfect with the push toward longer context windows across the industry. We're seeing 1 million token context windows becoming more common, and attention efficiency is a key bottleneck there.

## Alternative Architectures and Future

**ERIC:** Speaking of context lengths, it's worth noting that while attention is still king for the highest quality models, we're seeing interesting alternatives emerge. The paper mentions models like Mamba, xLSTM, and hybrid approaches that combine attention with other mechanisms.

**MAYA:** But even those alternative architectures often still use attention layers for the highest quality, right?

**ERIC:** Exactly. Models like Jamba and Zamba use a mixture of attention and state space models. So making attention faster benefits even these hybrid architectures. {{page: 19, section: "A", excerpt: "For the highest quality, these SSM- and RNN-based models still employ many layers of attention"}}

**MAYA:** Looking ahead, what do you think the next frontier is for attention optimization?

**ERIC:** I think we'll see more focus on inference optimization, better support for dynamic sequence lengths, and probably more aggressive quantization techniques. The move from FP8 to FP4 is already happening in newer hardware like Blackwell.

**MAYA:** And potentially more sophisticated sparsity patterns. While exact attention is still preferred for quality, there might be ways to combine the best of both worlds.

**ERIC:** Absolutely. The fundamental insight from FlashAttention-3 - that you need to deeply understand and exploit your hardware's capabilities - will remain relevant regardless of the specific optimizations.

## Conclusion

**MAYA:** Let's wrap up. What are the key takeaways from FlashAttention-3?

**ERIC:** First, modern hardware is asynchronous by design, and algorithms need to be redesigned to exploit that. Second, the devil is really in the details when it comes to performance optimization - things like memory layouts, instruction scheduling, and register allocation matter enormously.

**MAYA:** Third, you can achieve significant accuracy improvements in low-precision computation with the right techniques, like block quantization and incoherent processing. And finally, the best optimizations come from deep hardware-software co-design.

**ERIC:** The results speak for themselves - 2x speedups, 75% hardware utilization, and maintaining or improving accuracy. This is the kind of work that enables the next generation of AI applications.

**MAYA:** And it's a great reminder that performance optimization is far from a solved problem. There's still huge headroom for improvement when you're willing to dig deep into the hardware details.

**ERIC:** For anyone working on transformer models or performance optimization more generally, this paper is definitely worth a deep read. The techniques are sophisticated, but the principles are broadly applicable.

**MAYA:** Until next time, keep strolling.

**ERIC:** And may your gradients never explode.