# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

## Introduction

**ERIC:** Welcome to Strollcast! I'm Eric.

**MAYA:** And I'm Maya. We're your AI hosts, here to make research accessible while you're on the move.

**ERIC:** Today we're diving into FlashAttention-2, a paper by Tri Dao that's all about making attention mechanisms blazingly fast. {{page: 1, section: "Abstract", excerpt: "FlashAttention-2, with better work partitioning to address these issues"}}

**MAYA:** And when I say blazingly fast, I mean it. We're talking about doubling the speed of an already fast algorithm. But before we get into the technical details, let me ask you something, Eric - when you think about attention in transformers, what's the first thing that comes to mind?

**ERIC:** Honestly? Memory problems. You know how attention works - for every token in your sequence, you need to look at every other token. That means if you have a sequence of length N, you need N squared memory. {{page: 3, section: "2.2", excerpt: "Given input sequences Q,K,V∈ℝ^(N×d)... The standard attention implementation materializes the matrices S and P to HBM, which takes O(N²) memory"}}

**MAYA:** Exactly! And that quadratic scaling is what makes it so expensive to work with long sequences. Think about it - if you want to process a book or a high-resolution image, you're talking about thousands or tens of thousands of tokens.

**ERIC:** It's like trying to remember every single conversation you've ever had with every person you know, all at the same time. Your brain would explode!

## Background: The Memory Wall

**MAYA:** So let's set the stage. The original [FlashAttention](link:arxiv/2205.14135), also by Tri Dao, was already a game-changer. It solved the memory problem by being clever about how it moved data around on the GPU. {{page: 6, section: "2.3", excerpt: "FlashAttention applies the classical technique of tiling to reduce memory IOs"}}

**ERIC:** Think of it like organizing your kitchen. Instead of pulling out every ingredient for a complex recipe all at once and cluttering your counter, you work in stages - get what you need for the sauce, make it, put those ingredients away, then get what you need for the next step.

**MAYA:** That's a perfect analogy! FlashAttention breaks the attention computation into blocks, loads each block into fast memory, computes attention for that block, then moves on. This way, you never need to store those massive N squared matrices in slow memory.

**ERIC:** And the results were impressive - 2 to 4 times faster than standard attention implementations, with linear memory usage instead of quadratic. But here's the thing that motivated FlashAttention-2... {{page: 2, section: "1", excerpt: "FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s"}}

**MAYA:** Right! So FlashAttention was good, but it wasn't great. When you compare it to highly optimized matrix multiplication - which is what GPUs are really designed for - it was only using about a third of the GPU's potential.

**ERIC:** It's like having a Ferrari but only driving it at 30 mph. You're getting where you need to go, but you're not using the full power under the hood.

## Core Contribution: Three Key Improvements

**MAYA:** So what did Tri Dao do to unlock more of that potential? FlashAttention-2 makes three main improvements. Eric, want to walk us through them?

**ERIC:** Absolutely! First up is reducing non-matrix-multiply operations. {{page: 10, section: "3.1", excerpt: "We tweak the algorithm from FlashAttention to reduce the number of non-matmul FLOPs"}} Now, this might sound like a minor detail, but it's actually huge.

**MAYA:** Why is that?

**ERIC:** Well, modern GPUs have specialized hardware called Tensor Cores that are absolutely incredible at matrix multiplication. On an A100 GPU, these can do 312 trillion operations per second for matrix math, but only 19.5 trillion for other operations. {{page: 10, section: "3.1", excerpt: "the A100 GPU has a max theoretical throughput of 312 TFLOPs/s of FP16/BF16 matmul, but only 19.5 TFLOPs/s of non-matmul FP32"}}

**MAYA:** So every non-matrix operation is about 16 times more expensive! That means even small optimizations can have a big impact. What specific changes did they make?

**ERIC:** Two clever tweaks to the online softmax algorithm. First, instead of rescaling intermediate results at every step, they keep an "unscaled" version and only do the final scaling at the very end. Second, they simplified the statistics they need to track - instead of storing both the maximum and the sum separately, they just store the log-sum-exponential. {{page: 11, section: "3.1.1", excerpt: "We do not have to rescale both terms of the output update... We only need to store the logsumexp L^(j)=m^(j)+log(ℓ^(j))"}}

**MAYA:** These sound like small changes, but I imagine they add up. What's the second major improvement?

**ERIC:** Better parallelism! The original FlashAttention only parallelized across batch size and number of attention heads. But what happens when you have long sequences? {{page: 13, section: "3.2", excerpt: "In the case of long sequences (which usually means small batch sizes or small number of heads), to make better use of the multiprocessors on the GPU, we now additionally parallelize over the sequence length dimension"}}

**MAYA:** Ah, I see the problem. If you have a really long sequence, you probably have a smaller batch size to fit in memory. So you're not using all the GPU's processing units.

**ERIC:** Exactly! It's like having a factory with 100 assembly lines, but your current job only needs 20 of them. The other 80 are just sitting idle. FlashAttention-2 parallelizes over the sequence length too, so even with small batches, you can keep all those processing units busy.

**MAYA:** And the third improvement?

**ERIC:** This one's really in the weeds, but it's about how work gets divided among warps within each thread block. {{page: 14, section: "3.3", excerpt: "even within each thread block, we also have to decide how to partition the work between different warps"}}

**MAYA:** Okay, let's unpack this for our listeners. A GPU has thousands of threads, and these get organized into groups. The smallest group is a warp - 32 threads that work together. Multiple warps form a thread block.

**ERIC:** Right! And the original FlashAttention used what's called a "split-K" scheme for the forward pass. Think of it like having four teams work on different parts of a calculation, then they all have to meet up, share their intermediate results, add everything up, and move on. {{page: 14, section: "3.3", excerpt: "Each warp multiplies to get a slice of QK^⊤, then they need to multiply with a slice of V and communicate to add up the result... However, this is inefficient since all warps need to write their intermediate results out to shared memory"}}

**MAYA:** That communication step is expensive because it involves writing to and reading from shared memory, which takes time and energy.

**ERIC:** Exactly! So FlashAttention-2 reorganizes things so each warp can work independently without needing to communicate. It's like giving each team a complete, independent task instead of making them coordinate.

## Results: Doubling Performance

**MAYA:** So what do all these optimizations add up to? The results are pretty impressive. {{page: 16, section: "4", excerpt: "FlashAttention-2 is around 2× faster than FlashAttention and FlashAttention in xformers... FlashAttention-2 reaches up to 73% of the theoretical max throughput in the forward pass"}}

**ERIC:** We're talking about a 2x speedup over an already fast algorithm! And they're now reaching 73% of the theoretical maximum performance on A100 GPUs. Remember, the original FlashAttention was only hitting 25-40%.

**MAYA:** Those aren't just academic numbers either. When they tested this end-to-end on actual GPT-style models - we're talking 1.3 billion and 2.7 billion parameter models - they saw significant real-world speedups. {{page: 19, section: "4.2", excerpt: "FlashAttention-2 yields 2.8× speedup compared to a baseline without FlashAttention and 1.3× speedup compared to FlashAttention-2, reaching up to 225 TFLOPs/s per A100 GPU"}}

**ERIC:** And here's what that means in practical terms: you can now train a model with 16,000 token context length for the same cost as training an 8,000 token model before. That's the difference between processing a short article versus a whole research paper or book chapter.

**MAYA:** The implications are huge. We're already seeing language models with much longer context windows - [GPT-4](link:arxiv/2303.08774) with 32k tokens, some models going up to 100k tokens or more. {{page: 2, section: "1", excerpt: "Just within the last year, there have been several language models with much longer context than before: GPT-4 with context length 32k, MosaicML's MPT with context length 65k, and Anthropic's Claude with context length 100k"}}

## Technical Deep Dive: How It Actually Works

**ERIC:** Let's get a bit more technical about how this actually works. The key insight behind both FlashAttention and FlashAttention-2 is something called "online softmax." {{page: 7, section: "2.3.1", excerpt: "online softmax can split the attention computation into blocks, and rescale the output of each block to finally get the right result"}}

**MAYA:** This is a clever mathematical trick. Normally, to compute softmax, you need to see all the values at once - you find the maximum, subtract it from everything, take exponentials, sum them up, and divide. But online softmax lets you do this incrementally as you see new chunks of data.

**ERIC:** It's like calculating a running average. You can update your average each time you see a new number, without having to remember all the previous numbers. The math is a bit involved, but the key insight is that you can maintain just a few statistics and update them as you go.

**MAYA:** And this is what enables the tiling approach. Instead of loading the entire attention matrix into memory, you can process it block by block, updating your running statistics and partial results.

**ERIC:** FlashAttention-2 makes this even more efficient by reducing the amount of rescaling that happens during this process. {{page: 11, section: "3.1.1", excerpt: "We can instead maintain an 'un-scaled' version of O^(2) and keep around the statistics ℓ^(2)"}} Instead of rescaling at every step, they defer the final rescaling until the very end.

**MAYA:** It's a bit like doing your taxes. Instead of calculating the exact amount owed after every single deduction, you might keep running totals and only do the final calculation at the end. Less arithmetic overall.

## Beyond Forward Pass: The Backward Challenge

**MAYA:** Now, we've been talking mostly about the forward pass - computing attention outputs given inputs. But neural network training also requires a backward pass to compute gradients. This is actually even more challenging.

**ERIC:** Right! The backward pass is where FlashAttention really had to get creative. In the forward pass, you're just computing two matrix multiplications. But in the backward pass, you need to compute gradients for all the inputs, which involves five matrix multiplications. {{page: 20, section: "4.1", excerpt: "To get the FLOPs of the backward pass, we multiply the forward pass FLOPs by 2.5 (since there are 2 matmuls in the forward pass and 5 matmuls in the backward pass)"}}

**MAYA:** And you need to be careful about numerical stability and memory usage. The backward pass of the original FlashAttention was already quite complex - much more so than the forward pass.

**ERIC:** FlashAttention-2 keeps most of the backward pass the same, but makes that same optimization about storing just the logsumexp instead of separate max and sum statistics. {{page: 12, section: "3.1.2", excerpt: "The backward pass of FlashAttention-2 is almost the same as that of FlashAttention. We make a minor tweak to only use the row-wise logsumexp L"}}

**MAYA:** They also had to be clever about the parallelization in the backward pass. Unlike the forward pass where each block can work independently, the backward pass has dependencies where different thread blocks need to coordinate when updating the query gradients.

**ERIC:** They solve this using atomic operations - basically, a way for different processors to safely update the same memory location without stepping on each other's toes. {{page: 14, section: "3.2", excerpt: "We use atomic adds to communicate between different thread blocks to update dQ"}}

## Real-World Impact and Applications

**MAYA:** Let's talk about what this means in the real world. We mentioned that this enables longer context lengths, but what are the practical applications?

**ERIC:** Think about document analysis. Instead of having to break a long report into chunks and analyze each piece separately, you could potentially feed the entire document to a model at once. This preserves context and relationships that might be lost when chopping things up.

**MAYA:** Or consider code generation. Modern codebases are huge - being able to give a model context about an entire file or even multiple related files could lead to much better code completion and bug detection.

**ERIC:** And then there are applications we probably haven't even thought of yet. High-resolution image analysis, long-form video understanding, analyzing entire books or research papers... {{page: 2, section: "1", excerpt: "promising to improve performance in language modeling and high-resolution image understanding, as well as to unlock new applications in code, audio, and video generation"}}

**MAYA:** The efficiency gains also matter for deployment. If you can get 2x better performance with the same hardware, that means you can serve twice as many users, or the same number of users at half the cost.

**ERIC:** And for researchers, it means you can experiment with longer sequences without breaking the bank on compute costs. That could accelerate research into new architectures and applications.

## Technical Challenges and Future Directions

**MAYA:** Now, implementing something like this isn't trivial. This is very low-level GPU programming we're talking about. The paper mentions they built on NVIDIA's CUTLASS library for high-performance GPU kernels. {{page: 21, section: "Acknowledgments", excerpt: "We are grateful to the Nvidia CUTLASS team... for their CUTLASS library, in particular the CUTLASS 3.x release, which provides clean abstractions and powerful building blocks"}}

**ERIC:** And there are a lot of implementation details that matter. For example, they have to manually tune block sizes for different head dimensions and GPU memory configurations. {{page: 15, section: "3.3", excerpt: "We manually tune for each head dimensions since there are essentially only 4 choices for block sizes, but this could benefit from auto-tuning"}}

**MAYA:** That's something they mention as future work - automatic tuning instead of manual optimization. They also want to extend this to newer GPUs like the H100, which have additional features they could potentially exploit.

**ERIC:** Speaking of future work, they mention that even running the same code on H100 GPUs gets them up to 335 TFLOPs per second, and they expect another 1.5 to 2x speedup if they optimize for the H100's specific features. {{page: 18, section: "4.1", excerpt: "Just running the same implementation on H100 GPUs... we obtain up to 335 TFLOPs/s... We expect that by using new instructions, we can obtain another 1.5x-2x speedup"}}

**MAYA:** The authors also envision combining these low-level optimizations with high-level algorithmic changes - things like sparse attention patterns or local attention. {{page: 20, section: "5", excerpt: "Combining the low-level optimizations in FlashAttention-2 with high-level algorithmic changes (e.g., local, dilated, block-sparse attention) could allow us to train AI models with much longer context"}}

## Quizzes

**MAYA:** Alright, let's test your understanding with a couple of quick quizzes from the paper.

**ERIC:** First question: Why is reducing non-matrix-multiply operations so important for GPU performance? Think about it for a moment...

**MAYA:** The answer is that modern GPUs have specialized hardware for matrix multiplication - like Tensor Cores on NVIDIA GPUs. These can perform matrix operations up to 16 times faster than general arithmetic operations. So even if non-matrix operations are a small fraction of total operations, they can dominate the runtime because they're so much slower per operation.

**ERIC:** Here's the second quiz: FlashAttention-2 parallelizes over the sequence length dimension. Why wasn't this done in the original FlashAttention, and when is it most beneficial?

**MAYA:** The original FlashAttention only parallelized over batch size and number of heads. This works well when you have large batches, but with very long sequences, you typically have smaller batch sizes to fit in memory. Parallelizing over sequence length helps utilize more of the GPU's processing units when batch sizes are small, which is exactly the scenario where you need long sequences most.

## Conclusion

**ERIC:** So let's wrap up the key takeaways. FlashAttention-2 doubles the performance of an already efficient attention algorithm through three main optimizations: reducing non-matrix operations, better parallelization, and smarter work partitioning within GPU thread blocks.

**MAYA:** The real-world impact is significant - 2x speedup means you can train models with twice the context length for the same cost, or train existing models much faster. We're talking about enabling new applications in document analysis, code generation, and multimedia understanding.

**ERIC:** And this work represents the kind of cross-layer optimization that's becoming increasingly important in AI. It's not just about better algorithms or bigger models - it's about understanding the hardware deeply and co-designing algorithms and implementations.

**MAYA:** The fact that this gets us from 25-40% of theoretical GPU performance to 50-73% shows there's still room for improvement, but also demonstrates how much impact careful optimization can have.

**ERIC:** Looking forward, as context lengths continue to grow and new applications emerge, work like this becomes crucial for making advanced AI capabilities accessible and affordable.

**MAYA:** It's also worth noting that this kind of systems research - bridging the gap between algorithms and hardware - is what enables the rapid progress we're seeing in AI. The models get the headlines, but the infrastructure work makes it all possible.

**ERIC:** Until next time, keep strolling.

**MAYA:** And may your gradients never explode.