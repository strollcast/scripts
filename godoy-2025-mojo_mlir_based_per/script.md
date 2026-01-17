# Mojo: MLIR-Based Performance-Portable HPC Science Kernels on GPUs for the Python Ecosystem

## Introduction

**ERIC:** Welcome to Strollcast! I'm Eric.

**MAYA:** And I'm Maya. We're your AI hosts, here to make research accessible while you're on the move.

**ERIC:** Today we're exploring a fascinating paper about Mojo—no, not the Austin Powers kind of mojo, but a brand new programming language that's trying to solve one of the biggest headaches in scientific computing. {{page: 1, section: "Introduction"}}

**MAYA:** Right! The paper is titled "Mojo: MLIR-Based Performance-Portable HPC Science Kernels on GPUs for the Python Ecosystem" by William Godoy and his team at Oak Ridge National Laboratory. Think of this as asking: can we finally have our cake and eat it too when it comes to Python and high-performance computing? {{page: 1, section: "Abstract"}}

**ERIC:** That's the million-dollar question! See, scientists love Python because it's incredibly productive—you can prototype algorithms quickly, the syntax is clean, and there's this massive ecosystem of libraries. But when you need serious performance, especially on GPUs, you often have to drop down to CUDA or HIP, which is like switching from a comfortable sedan to a Formula 1 car mid-journey.

**MAYA:** Exactly! And what makes Mojo particularly interesting is that it's the first language built entirely on MLIR—the Multi-Level Intermediate Representation from the LLVM project. {{page: 1, section: "Introduction", excerpt: "As the first language based on the LLVM's Multi-Level Intermediate Representation (MLIR) compiler infrastructure"}} Think of MLIR as a sophisticated translation system that can optimize code at multiple levels simultaneously.

**ERIC:** The researchers tested Mojo on four different scientific workloads that cover the spectrum of what you typically see in high-performance computing. They've got memory-bound kernels like seven-point stencils and BabelStream, plus compute-bound kernels like miniBUDE and Hartree-Fock. {{page: 2, section: "2.2"}}

## Background and Motivation

**MAYA:** Let's step back and understand why this matters. The scientific computing world has been living with this painful fragmentation for decades. You start prototyping in Python, it works great, but then you need to scale up or run on GPUs...

**ERIC:** And suddenly you're rewriting everything in CUDA or HIP! It's like being fluent in Spanish but having to conduct all your important business meetings in Mandarin. The cognitive overhead is enormous, and you lose all that beautiful Python interoperability.

**MAYA:** The paper mentions that 9 out of the top 10 supercomputers in the world are GPU-accelerated. {{page: 1, section: "Introduction", excerpt: "The TOP500's Linpack ranking from June 2025 shows that 9 out of the top 10 fastest supercomputers in the world are accelerated by GPUs"}} So this isn't just about convenience—it's about accessing the computing power that's driving modern science and AI.

**ERIC:** What's clever about Mojo's approach is that it doesn't try to optimize Python itself. Instead, it creates a superset language that can compile high-performance code while still talking to the Python ecosystem at runtime. {{page: 3, section: "2.1"}} It's like having a bilingual friend who can seamlessly switch between languages depending on the conversation.

**MAYA:** The key insight is that Mojo separates performance-critical sections that get compiled with MLIR from the Python interoperability parts that stay dynamic. This is shown beautifully in their code example where GPU kernel programming happens at compile-time with fixed types and sizes, while Python interactions remain flexible. {{page: 4, section: "2.1", excerpt: "Mojo introduces this fragmentation in the language by keeping a clear separation between (i) JIT- or AOT-MLIR compiled high-performance code and (ii) Python interoperability as a run-time-only entity"}}

**ERIC:** But here's the trade-off: Mojo requires you to specify performance-critical information like problem sizes, data types, and memory layouts at compile time. For traditional HPC applications that discover these things at runtime, this could be a significant constraint.

## Mojo's Technical Approach

**MAYA:** Let's dive into how Mojo actually works. The language uses a CUDA-like programming model for GPU kernels, with explicit memory management and kernel launching. {{page: 4, section: "2.1"}} So if you're familiar with CUDA, the syntax will feel quite natural.

**ERIC:** What's really interesting is the memory safety aspect. Mojo uses value ownership—similar to Rust—where every heap-allocated value has exactly one owner. When the owner goes out of scope, the memory gets automatically cleaned up without needing a garbage collector. {{page: 5, section: "2.1"}} It's like having a very responsible roommate who always cleans up after themselves.

**MAYA:** The researchers tested Mojo on both NVIDIA H100 and AMD MI300A GPUs. The H100 has 94GB of memory with 3,900 GB/s of bandwidth, while the MI300A is even more impressive with 128GB and 5,300 GB/s bandwidth. {{page: 8, section: "3"}} These are seriously powerful machines.

**ERIC:** Now, let's talk about their test applications. The seven-point stencil is a classic computational pattern used in solving partial differential equations—think weather modeling or heat diffusion. {{page: 6, section: "2.2"}} It's memory-bound because you're essentially reading from neighboring grid points and writing back results.

**MAYA:** BabelStream is like a stress test for memory bandwidth. It implements five fundamental array operations: Copy, Multiply, Add, Triad, and Dot product. {{page: 6, section: "2.2"}} These are the building blocks of many linear algebra computations, so if you can make these fast, you're in good shape for a lot of scientific computing.

**ERIC:** The compute-bound applications are more complex. miniBUDE simulates molecular docking—essentially trying to predict how drug molecules will bind to proteins. {{page: 6, section: "2.2"}} This is computationally intensive because it involves lots of floating-point operations per memory access.

**MAYA:** And Hartree-Fock implements a quantum mechanical approximation method. What makes this particularly challenging is that it uses atomic operations extensively—six atomic updates to matrices for every computation iteration. {{page: 7, section: "2.2"}} Atomic operations are notoriously difficult to parallelize effectively.

## Performance Results

**ERIC:** So how did Mojo actually perform? The results are quite nuanced and paint an interesting picture of where the language stands today.

**MAYA:** For memory-bound kernels, Mojo performs remarkably well on AMD GPUs—essentially matching HIP performance across the board. On NVIDIA GPUs, it's a bit slower than CUDA but still quite competitive, averaging about 87% of CUDA performance for the seven-point stencil. {{page: 10, section: "4", excerpt: "Mojo is fairly competitive for both single and double precision, but is slightly slower on NVIDIA's H100, averaging 87% of the CUDA performance"}}

**ERIC:** The BabelStream results were actually surprising. Except for the Dot product operation, Mojo was slightly *faster* than CUDA on NVIDIA hardware! {{page: 12, section: "4"}} The researchers dug into this with profiling tools and found that Mojo's MLIR optimizations were generating more efficient memory operations in some cases.

**MAYA:** But the compute-bound kernels tell a different story. For miniBUDE, Mojo couldn't match the performance of vendor-optimized CUDA and HIP implementations, particularly when fast-math optimizations were enabled. {{page: 13, section: "4"}} This highlights a current limitation—Mojo doesn't yet support fast-math compiler flags that are crucial for compute-intensive workloads.

**ERIC:** The Hartree-Fock results were particularly puzzling. On NVIDIA GPUs, Mojo was actually 2.5 times faster than CUDA for smaller problem sizes, but performance deteriorated dramatically for larger problems. On AMD GPUs, it consistently underperformed HIP. {{page: 14, section: "4"}} This suggests there are still some rough edges in how Mojo handles atomic operations across different hardware.

**MAYA:** The researchers used a performance portability metric to quantify their results. This metric measures how consistently an application performs across different hardware platforms. {{page: 15, section: "4.1"}} Mojo scored 0.96 for BabelStream and 0.92 for the seven-point stencil—that's quite good! But it only managed 0.54 for miniBUDE, indicating significant room for improvement on compute-bound workloads.

## Profiling Deep Dive

**ERIC:** What I love about this paper is how thoroughly the researchers profiled their code to understand performance differences. They used NVIDIA's Nsight Compute profiler to analyze kernel execution at a very detailed level.

**MAYA:** For the seven-point stencil, they found that CUDA-generated code made more efficient use of memory resources, particularly at the register level. Mojo kernels used 24 registers per thread compared to CUDA's 21, which explains the performance gap in memory-bound workloads. {{page: 10, section: "Table 2"}}

**ERIC:** But for BabelStream, the story was different. Mojo actually achieved higher memory throughput and lower compute utilization, which is exactly what you want for memory-bandwidth limited operations. {{page: 12, section: "Table 3"}} This shows that MLIR's optimizations can sometimes outperform vendor compilers.

**MAYA:** They even looked at the generated assembly code! Figure 5 in the paper shows a side-by-side comparison of Mojo and CUDA assembly for the Triad kernel. {{page: 13, section: "4"}} Mojo produced fewer constant memory load operations and used constant memory more efficiently, but also generated more integer arithmetic operations.

## Broader Implications and Ecosystem

**ERIC:** Let's zoom out and think about what this means for the broader scientific computing ecosystem. Mojo isn't trying to compete directly with languages like Julia or replace established HPC languages. Instead, it's positioning itself as a bridge between Python's productivity and GPU performance.

**MAYA:** The Python ecosystem in scientific computing is incredibly rich—NumPy, SciPy, pandas, scikit-learn, and countless domain-specific libraries. But when you need serious performance, you often have to abandon this ecosystem entirely. {{page: 16, section: "4.2"}} Mojo could potentially keep you in the Python world while still accessing high-performance computing resources.

**ERIC:** However, there are still some significant challenges. The paper mentions that Mojo's interoperability with MPI—the Message Passing Interface used for distributed computing—is still an open question. {{page: 17, section: "4.2"}} For large-scale scientific applications that need to run across hundreds or thousands of nodes, this is crucial.

**MAYA:** There's also the tooling ecosystem to consider. The researchers found that debugging and profiling Mojo code works well with NVIDIA's tools but has limited support for AMD's profiling tools. {{page: 5, section: "2.1"}} As the language matures, this kind of tooling support will be essential for adoption.

**ERIC:** And let's not forget the learning curve. While Mojo's syntax is designed to be familiar to Python developers, writing high-performance GPU kernels still requires understanding low-level concepts like memory hierarchies, thread blocks, and data layout optimization. {{page: 17, section: "4.2"}}

## Comparison with Other Approaches

**MAYA:** The paper does a nice job contextualizing Mojo within the broader landscape of performance-portable programming approaches. There's been a lot of work in this space over the years.

**ERIC:** On the Python side, you have efforts like NVIDIA's cuda-python and AMD's hip-python-as-cuda, which provide direct bindings to GPU APIs. {{page: 18, section: "5"}} There's also Numba, which uses JIT compilation to accelerate Python code, and PyOpenCL for writing GPU kernels.

**MAYA:** Julia has been particularly successful in this space. Projects like [JACC](link:arxiv/2401.15644) and [KernelAbstractions.jl](link:arxiv/2106.12636) provide high-level, performance-portable abstractions while maintaining near-zero overhead compared to vendor-specific implementations. {{page: 18, section: "5"}}

**ERIC:** What makes Mojo unique is its deep integration with MLIR and its approach of separating compile-time performance code from runtime Python interoperability. Most other solutions either sacrifice performance for portability or require you to work entirely within their ecosystem.

**MAYA:** The researchers also mention historical efforts like the DARPA High Productivity Computing Systems program, which funded languages like Chapel, X10, and Fortress. {{page: 19, section: "5"}} Only Chapel remains actively developed today, and it's also evolving toward vendor-neutral GPU programming.

## Current Limitations and Future Directions

**ERIC:** While the results are promising, the paper is honest about Mojo's current limitations. The lack of fast-math optimizations significantly impacts compute-bound workloads, and the atomic operations behavior is inconsistent across different GPU vendors.

**MAYA:** The compile-time requirements could also be challenging for traditional HPC applications. Many scientific codes need to adapt problem sizes, data structures, or algorithmic approaches based on runtime information. {{page: 17, section: "4.2"}} Mojo's MLIR-based approach requires this information at compile time.

**ERIC:** There's also the question of ecosystem maturity. While Mojo can interoperate with Python libraries, it doesn't yet have the rich collection of high-performance libraries that languages like C++ and Fortran enjoy in the HPC world. Building and optimizing fundamental libraries like BLAS operations will be crucial for wider adoption.

**MAYA:** But the potential is exciting! The researchers suggest that Mojo could help unify the fragmented Python ecosystem, especially at the intersection of scientific computing and AI. {{page: 1, section: "Abstract"}} Given how much overlap there is between these fields today, this could be incredibly valuable.

**ERIC:** It's worth noting that Mojo is still very much a work in progress. The language is developed by Modular Inc. and isn't fully open-source yet, though they've committed to making it open-source by 2026. {{page: 19, section: "6"}} This could significantly impact adoption in the academic and open-source communities.

## Practical Implications for Researchers

**MAYA:** So what does this mean if you're a researcher or practitioner today? Should you start learning Mojo?

**ERIC:** I think it depends on your specific use case. If you're working on memory-bound problems and primarily targeting AMD GPUs, Mojo looks quite promising even in its current state. The performance is competitive, and you get the benefit of staying within a Python-like ecosystem.

**MAYA:** For compute-intensive workloads, you might want to wait until fast-math optimizations and better atomic operations support are available. The performance gaps are still significant for these use cases. {{page: 13-14, section: "4"}}

**ERIC:** But even if you're not ready to adopt Mojo today, it's worth watching the development closely. The underlying approach of using MLIR for performance-portable compilation while maintaining high-level language interoperability could influence future language designs.

**MAYA:** The paper also provides all their code implementations as open-source artifacts, so researchers can experiment with the kernels and potentially extend them to other scientific applications. {{page: 20, section: "Appendix"}} This is valuable for understanding how to effectively structure performance-critical code in Mojo.

## The Bigger Picture

**ERIC:** Stepping back even further, this work represents part of a broader trend toward heterogeneous computing in scientific applications. With the rise of AI and machine learning, many scientific workflows now involve both traditional simulation and data analysis components.

**MAYA:** Exactly! And these often have very different computational characteristics. Traditional scientific simulations might be memory-bound and involve complex geometries, while machine learning inference might be compute-bound with regular data access patterns. {{page: 1, section: "Introduction"}}

**ERIC:** Mojo's approach of providing performance-portable GPU programming within a Python-compatible language could be particularly valuable for these hybrid workflows. You could potentially write your entire application stack in one language ecosystem rather than juggling Python, CUDA, and various specialized libraries.

**MAYA:** The timing is interesting too. This work comes as the hardware landscape is becoming increasingly diverse. We're not just talking about NVIDIA versus AMD anymore—there are new architectures like Intel's GPU offerings, and potentially very different approaches emerging for AI-specific workloads.

**ERIC:** And the researchers' use of a formal performance portability metric is valuable for the community. {{page: 15, section: "4.1"}} Rather than just claiming "good performance," they provide quantitative measures that can be compared across different studies and programming approaches.

## Quizzes

**MAYA:** Alright, let's test your understanding with a couple of questions from the paper. First question: What makes Mojo unique compared to other approaches for accelerating Python code?

**ERIC:** Take a moment to think about it... 

**MAYA:** The key insight is that Mojo is the first language built entirely on MLIR, and it separates compile-time performance-critical code from runtime Python interoperability. {{page: 4, section: "2.1"}} Unlike other approaches that try to optimize Python itself or require you to abandon the Python ecosystem, Mojo provides a superset language that can compile high-performance kernels while still talking to Python libraries at runtime.

**ERIC:** Here's another one: The researchers tested four different scientific kernels. Can you categorize them by their computational characteristics?

**MAYA:** Think about whether each kernel is primarily limited by memory bandwidth or computation...

**ERIC:** The seven-point stencil and BabelStream are memory-bound kernels—they're limited by how fast you can move data to and from memory. miniBUDE and Hartree-Fock are compute-bound, meaning they perform many operations on each piece of data they load. {{page: 6, section: "2.2"}} Hartree-Fock has the additional complexity of using atomic operations extensively, which makes parallelization particularly challenging.

## Conclusion

**MAYA:** This paper gives us a fascinating glimpse into what could be the future of scientific computing. While Mojo isn't perfect yet, it represents a genuinely novel approach to bridging the performance and productivity gap that has plagued scientific software development for decades.

**ERIC:** The results show that performance-portable programming is achievable, especially for memory-bound workloads. And even where Mojo falls short today—like compute-bound kernels without fast-math optimizations—these seem like solvable problems rather than fundamental limitations.

**MAYA:** What I find most exciting is the potential for reducing fragmentation in scientific software development. If researchers can write their entire application stack in a single, coherent language ecosystem while still accessing cutting-edge hardware performance, that could accelerate scientific discovery significantly.

**ERIC:** The open-source commitment for 2026 will be crucial for adoption in academic and research communities. And the detailed profiling work in this paper provides a roadmap for where the language needs to improve to become truly competitive across all workload types.

**MAYA:** For anyone working at the intersection of scientific computing and AI—which is increasingly everyone in computational science—Mojo is definitely worth keeping on your radar. Even if you're not ready to adopt it today, the underlying approaches and design principles could influence how you structure your own software projects.

**ERIC:** Plus, the performance portability metrics and detailed benchmarking methodology in this paper provide a valuable framework for evaluating future programming language developments in the HPC space.

**MAYA:** Until next time, keep strolling through the fascinating world of high-performance computing.

**ERIC:** And may your gradients never explode—whether they're in your AI models or your numerical simulations!