# Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning

## Introduction

**ERIC:** Welcome to Strollcast! I'm Eric.

**MAYA:** And I'm Maya. We're your AI hosts, here to make research accessible while you're on the move.

**ERIC:** Today we're diving into a paper that tackles one of the biggest headaches in machine learning: training massive models across distributed systems. We're talking about "Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning."

**MAYA:** This comes from an impressive team including researchers from UC Berkeley, Google, Amazon, and several other institutions. The lead authors are Lianmin Zheng, Zhuohan Li, and Hao Zhang. {{page: 1, section: "Title", excerpt: "Lianmin Zheng, Zhuohan Li, Hao Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu, Danyang Zhuo, Eric P. Xing, Joseph E. Gonzalez, Ion Stoica"}}

**ERIC:** And boy, does this paper address a real pain point. Think about trying to train a model like GPT-3 with its 175 billion parameters. You can't just throw it on a single GPU and call it a day - you need dozens or hundreds of GPUs working together.

**MAYA:** But here's the catch: figuring out how to split up the work optimally has traditionally required deep expertise and tons of manual tuning. It's like trying to organize a massive group project where everyone needs to coordinate perfectly, but the optimal strategy depends on who's in your group and what resources they have.

**ERIC:** Exactly! And that's where Alpa comes in. The key insight is brilliant in its simplicity: instead of treating all parallelization strategies as separate, competing approaches, they reorganize everything into just two hierarchical levels. {{page: 2, section: "Abstract", excerpt: "Alpa distributes the training of large DL models by viewing parallelisms as two hierarchical levels: inter-operator and intra-operator parallelisms"}}

## Background: The Parallelism Puzzle

**MAYA:** Let's set the stage here. When you're training a huge model, you typically have three main strategies: data parallelism, operator parallelism, and pipeline parallelism. Each has its own trade-offs.

**ERIC:** Think of data parallelism like having multiple study groups working on the same problem set. Each group works on different data, then they share their answers. {{page: 4, section: "2.1", excerpt: "In data parallelism, the training data is partitioned across distributed workers, but the model is replicated"}}

**MAYA:** Operator parallelism is more like splitting up a single math problem across your study group - one person handles part of the equation, another handles a different part. {{page: 4, section: "2.1", excerpt: "Operator parallelism refers to approaches that partition the computation of a specific operator...along non-batch axes"}}

**ERIC:** And pipeline parallelism? That's like an assembly line where different groups handle different stages of the same problem sequentially. {{page: 4, section: "2.1", excerpt: "Pipeline parallelism places different groups of ops from the model graph, referred as stages, on different workers"}}

**MAYA:** Now here's where it gets tricky. The state-of-the-art systems like Megatron-LM manually combine these approaches, but they're incredibly specialized. {{page: 5, section: "2.1", excerpt: "The state-of-the-art training systems, such as Megatron-LM, manually design a specialized execution plan that combines these parallelisms for transformer language models"}} They work great for transformer models, but good luck adapting them to anything else.

**ERIC:** And the automatic systems? They're limited too. Most can only handle one or maybe two types of parallelism at once. It's like having a Swiss Army knife with only one blade.

## The Alpa Insight: A New Way to Think About Parallelism

**MAYA:** So here's where the Alpa team had their eureka moment. Instead of thinking about data, operator, and pipeline parallelism as three separate things, they reorganized everything into two categories based on a fundamental question: are you splitting up individual operations or not?

**ERIC:** They call these "intra-operator" and "inter-operator" parallelism. Intra-operator is anything where you're taking a single mathematical operation - like a matrix multiplication - and splitting it across multiple devices. {{page: 5, section: "2.2", excerpt: "We define all parallelization approaches using this workflow as intra-operator parallelism"}}

**MAYA:** So both data parallelism and operator parallelism are actually intra-operator parallelism. In data parallelism, you're splitting the input tensors along the batch dimension. In operator parallelism, you're splitting along other dimensions. But in both cases, you're partitioning the computation of individual operators.

**ERIC:** Inter-operator parallelism, on the other hand, is when you assign completely different operations to different devices. Pipeline parallelism is the classic example - different stages of your model run on different devices. {{page: 6, section: "2.2", excerpt: "We define inter-operator parallelism as the orthogonal class of approaches that do not perform operator partitioning, but instead, assign different operators of the graph to execute on distributed devices"}}

**MAYA:** This reorganization is more than just academic - it reveals something crucial about communication patterns. Intra-operator parallelism requires lots of communication between devices for every operation. Inter-operator parallelism only requires communication between pipeline stages.

**ERIC:** And here's the brilliant part: this maps perfectly onto how computer clusters are actually built! Within a single machine, GPUs are connected by high-bandwidth links like NVLink. Between machines, you have slower network connections. {{page: 6, section: "2.2", excerpt: "devices communicate only between pipeline stages, typically using point-to-point communication between device pairs"}}

## The Alpa System Architecture

**MAYA:** So how does Alpa turn this insight into a working system? They built a compiler with three main passes that work hierarchically.

**ERIC:** First, there's the intra-operator pass. Given a chunk of computation and a group of devices with fast connections between them - what they call a "device mesh" - this pass figures out the optimal way to split up individual operations. {{page: 7, section: "3", excerpt: "At the intra-op level, Alpa minimizes the cost of executing a stage...on a given device mesh"}}

**MAYA:** They formulate this as an Integer Linear Programming problem, which sounds fancy but basically means they can find the mathematically optimal solution. They enumerate all the possible ways to split each operation, figure out the communication costs, and let the solver find the best combination.

**ERIC:** Then there's the inter-operator pass, which decides how to slice the entire model into stages and how to assign those stages to different device meshes. {{page: 7, section: "3", excerpt: "At the inter-op level, Alpa minimizes the inter-op parallelization latency, with respect to how to slice the model and device cluster into stages and device meshes"}}

**MAYA:** This one uses dynamic programming to solve what's essentially a complex scheduling problem. The clever part is that it calls the intra-op pass to get cost estimates for each possible assignment of a stage to a device mesh.

**ERIC:** Finally, there's the runtime orchestration pass that handles the messy details of actually executing everything - generating the right communication patterns, scheduling pipeline execution, and dealing with the fact that different stages might need different device configurations.

## Technical Deep Dive: The Intra-Operator Magic

**MAYA:** Let's dig into how the intra-operator optimization actually works, because this is where some real algorithmic magic happens.

**ERIC:** The key insight is that they represent tensor layouts using something called "sharding specs." Think of it like a recipe for how to slice up a multi-dimensional array across devices. {{page: 10, section: "4.1", excerpt: "We use sharding spec to define the layout of a tensor"}}

**MAYA:** For example, if you have a 2D tensor - basically a matrix - you might partition it along rows, columns, both, or not at all. And then you need to specify which devices get which pieces.

**ERIC:** Here's where it gets interesting. For every operation in your model - matrix multiplication, convolution, whatever - there are multiple valid ways to parallelize it. Each way requires the input tensors to have specific layouts and produces outputs with specific layouts.

**MAYA:** But if one operation outputs a tensor in format A and the next operation needs it in format B, you need to do a "resharding" operation - essentially rearranging the data across devices. {{page: 11, section: "4.1", excerpt: "When an input tensor of an operator does not satisfy the sharding spec of the chosen parallel algorithm for the operator, a layout conversion, namely resharding, is required"}}

**ERIC:** And this resharding can be expensive! Sometimes it requires all-to-all communication where every device needs to talk to every other device. Sometimes it's just local slicing. The optimization algorithm needs to balance computation efficiency with communication costs.

**MAYA:** The ILP formulation captures all of this. For each operation, you have decision variables representing which parallelization strategy to use. The objective function minimizes the total cost including both computation and resharding communication. {{page: 12, section: "4.2", excerpt: "The total execution cost of a computational graph G=(V,E) is the sum of the compute and communication costs on all nodes v∈V and the resharding costs on all edges e∈E"}}

## Inter-Operator Optimization: The Scheduling Challenge

**ERIC:** The inter-operator pass tackles a different but equally complex problem. Given that you've optimally parallelized individual stages, how do you partition your model and assign stages to device groups to minimize overall training time?

**MAYA:** The objective function here is really elegant. Pipeline training time has two components: the time for the first batch to flow through all stages, and then additional time for subsequent batches limited by the slowest stage. {{page: 14, section: "5.1", excerpt: "T*=min{∑ti+(B-1)·max{tj}}"}}

**ERIC:** So if you have 4 stages taking 1, 2, 3, and 2 seconds respectively, and 10 batches total, the total time is (1+2+3+2) + 9×3 = 35 seconds. That bottleneck stage taking 3 seconds dominates everything.

**MAYA:** The dynamic programming algorithm tries different ways of partitioning the model and different device assignments, querying the intra-op pass for cost estimates each time. But here's a practical challenge: the full search space is enormous.

**ERIC:** So they use several clever optimizations. First, they limit the device mesh shapes to configurations that make sense for real clusters - like using all GPUs in a machine, or powers of 2. {{page: 15, section: "5.2", excerpt: "we reduce the available submesh shapes into two options: (1) one-dimensional submeshes...and (2) two-dimensional submeshes...that fully use the second dimension"}}

**MAYA:** They also use "operator clustering" to group lightweight operations together, reducing the graph size. {{page: 17, section: "Performance optimization #2", excerpt: "We develop another DP algorithm to cluster neighboring operators to reduce the total size of the graph"}} No point in optimizing the placement of a simple ReLU activation when it takes microseconds anyway.

## Real-World Results: Does It Actually Work?

**ERIC:** Alright, enough theory - does this thing actually work in practice? The results are pretty impressive.

**MAYA:** They tested on three different model families: GPT-3 style language models, Mixture of Experts models, and Wide-ResNet for image classification. These cover both homogeneous architectures where every layer is similar, and heterogeneous ones with very different layer types. {{page: 20, section: "8.1", excerpt: "We target three types of models...covering models with both homogeneous and heterogeneous architectures"}}

**ERIC:** For GPT-3, they went head-to-head with Megatron-LM, which is basically the gold standard for training large transformer models. Megatron-LM has years of hand-tuned optimizations specific to transformers. And Alpa matched or slightly beat it! {{page: 21, section: "GPT-3 results", excerpt: "compared to Megatron-LM, Alpa automatically generates execution plans and even achieves slightly better scaling on several settings"}}

**MAYA:** That's remarkable because Megatron-LM is incredibly specialized. The Alpa team found that the automatically generated strategies closely matched the hand-tuned ones - mostly using data parallelism with pipeline stages when memory gets tight.

**ERIC:** But here's where Alpa really shines: on models that don't have hand-tuned strategies. For Mixture of Experts models, they compared against DeepSpeed and achieved 3.5× speedup on 2 nodes and 9.7× speedup on 4 nodes! {{page: 22, section: "MoE results", excerpt: "Compared to DeepSpeed, Alpa achieves 3.5× speedup on 2 nodes and a 9.7× speedup on 4 nodes"}}

**MAYA:** The reason is that DeepSpeed doesn't include pipeline parallelism, so it can't scale well across multiple machines with slower inter-node connections. Alpa automatically figures out how to combine intra-operator parallelism within nodes and inter-operator parallelism across nodes.

**ERIC:** And for Wide-ResNet, there was no existing specialized system at all. Alpa achieved 80% scaling efficiency on 32 GPUs for a model architecture that nobody had manually optimized before. {{page: 22, section: "Wide-ResNet results", excerpt: "Alpa achieves a scalable performance on 32 GPUs with 80% scaling"}}

## Under the Hood: What Strategies Does Alpa Find?

**MAYA:** One of the coolest parts of the paper is seeing the actual strategies Alpa discovers. For Wide-ResNet, they visualize how the system partitions different layers.

**ERIC:** With 4 GPUs, it uses only intra-operator parallelism, mostly partitioning along the batch dimension early on, then switching to channel partitioning for later layers where the weights are larger. {{page: 27, section: "8.6", excerpt: "On 4 GPUs, Alpa uses only intra-operator parallelism. The intra-operator solution partitions along the batch axis for the first dozens of layers and then switches to partitioning the channel axis"}}

**MAYA:** With 16 GPUs, it creates a 3-stage pipeline with 4, 4, and 8 GPUs per stage. The first two stages use data parallelism because activations are large, but the third stage uses more complex partitioning strategies. {{page: 27, section: "8.6", excerpt: "On 16 GPUs, Alpa slices the model into 3 stages and assigns 4, 4, 8 GPUs to stage 1, 2, 3, respectively"}}

**ERIC:** The researchers note that creating such a strategy manually would be incredibly difficult, even for experts. The system is finding non-obvious optimizations that humans would struggle to discover.

## Compilation Time and Practical Considerations

**MAYA:** Now, there's always a catch with these sophisticated optimization systems - how long does it take to actually generate these strategies?

**ERIC:** For the largest GPT model they tested - 39 billion parameters on 64 GPUs - the total compilation time was about 40 minutes. {{page: 26, section: "8.4", excerpt: "compilation time grows linearly with the size of the model and the number of GPUs in the cluster"}} Most of that is spent profiling different stage-mesh combinations to get accurate performance estimates.

**MAYA:** Forty minutes sounds like a lot, but remember, these models typically train for days or weeks. If spending 40 minutes on optimization saves you 10% on a week-long training run, that's over 16 hours saved.

**ERIC:** Plus, they have several optimizations to speed things up, like using a cost model instead of actual profiling for some estimates, and parallelizing the compilation across multiple workers.

**MAYA:** The compilation time grows linearly with model and cluster size, which suggests it should scale reasonably well to even larger systems.

## Limitations and Future Directions

**ERIC:** Now, Alpa isn't perfect - the authors are refreshingly honest about current limitations.

**MAYA:** One big limitation is that they don't model communication costs between pipeline stages in their optimization. {{page: 18, section: "7", excerpt: "Alpa does not model the communication cost between different stages because the cross-stage communication cost is by nature small"}} They argue this is okay because inter-stage communication is typically small, but it could matter for some models.

**ERIC:** They also use a static pipeline schedule rather than more dynamic approaches that could potentially parallelize different branches of a computation graph simultaneously.

**MAYA:** And like most current systems, Alpa requires knowing all tensor shapes at compile time - it can't handle dynamic models where shapes change during execution.

**ERIC:** But perhaps the biggest limitation is philosophical: they're optimizing within the current paradigm of synchronous gradient descent. Future breakthroughs might require completely different training algorithms that this framework doesn't capture.

**MAYA:** Still, for the current state of the art, these limitations seem reasonable. The system solves a real, immediate problem that's blocking a lot of ML research and production work.

## Broader Impact: Democratizing Large-Scale ML

**ERIC:** Let's zoom out and talk about why this work matters beyond just the technical details.

**MAYA:** Right now, training massive models is basically restricted to a handful of organizations with deep systems expertise. Google has their custom TPU systems, OpenAI has their specialized infrastructure, Facebook has their own approaches.

**ERIC:** But most researchers and companies don't have teams of systems experts who can spend months hand-tuning parallelization strategies. Alpa could democratize access to large-scale training. {{page: 1, section: "1", excerpt: "Automating the parallelization of large-scale models would significantly accelerate ML research and production by enabling model developers to quickly explore new model designs"}}

**MAYA:** Think about it - if you're a researcher with a new architecture idea, you shouldn't have to become an expert in distributed systems just to test whether your idea works at scale. You should be able to focus on the ML and let the compiler handle the systems optimization.

**ERIC:** This could accelerate the pace of ML research significantly. How many good ideas never get properly tested because the systems engineering barrier is too high?

**MAYA:** And it's not just about research. As companies want to train custom models on their proprietary data, they need tools that can adapt to new architectures and cluster configurations automatically.

## Quiz Time!

**ERIC:** Alright, let's test your understanding with a couple of questions from the paper. First one: What are the two main categories that Alpa uses to reorganize all parallelization strategies?

**MAYA:** Think about it for a moment... what's the fundamental distinction the authors make?

**ERIC:** The answer is intra-operator and inter-operator parallelism! {{page: 5, section: "2.2", excerpt: "we re-catalog existing parallelization approaches into two orthogonal categories: intra-operator and inter-operator parallelisms"}} Intra-operator involves partitioning individual operations across devices, while inter-operator assigns different operations to different devices without partitioning them.

**MAYA:** Here's question two: In the pipeline parallelism execution time formula, what are the two main components that determine total training time?

**ERIC:** Take a second to think about how pipeline parallelism works...

**MAYA:** The answer is: the time for the first batch to go through all stages, plus the time for remaining batches which is limited by the slowest stage. {{page: 14, section: "5.1", excerpt: "T*=min{∑ti+(B-1)·max{tj}}"}} So if you have stages taking 1, 2, 3, 2 seconds and 10 batches, it's (1+2+3+2) + 9×3 = 35 seconds total.

## Conclusion

**ERIC:** Alpa represents a significant step forward in making large-scale machine learning more accessible. By reorganizing our thinking about parallelism and building sophisticated optimization algorithms, the authors have created a system that can automatically generate strategies that match or exceed hand-tuned solutions.

**MAYA:** What I find most exciting is how this opens up possibilities for exploring new model architectures. When the systems optimization is handled automatically, researchers can focus on the fundamental ML questions rather than getting bogged down in distributed systems details.

**ERIC:** The hierarchical approach - optimizing intra-operator parallelism within device meshes and inter-operator parallelism between meshes - is elegant and maps naturally onto real cluster topologies. It's one of those insights that seems obvious in retrospect but required real creativity to recognize.

**MAYA:** And the results speak for themselves. Matching Megatron-LM on its home turf while generalizing to completely different architectures is impressive. The 9× speedup on MoE models shows what's possible when you're not constrained by manual optimization strategies.

**ERIC:** There's still work to be done - handling dynamic shapes, optimizing communication between stages, exploring different training algorithms. But this feels like a major step toward the broader goal of making AI development more accessible and efficient.

**MAYA:** Until next time, keep strolling.

**ERIC:** And may your gradients never explode.