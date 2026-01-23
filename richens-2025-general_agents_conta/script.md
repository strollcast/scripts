# General agents contain world models

## Introduction

**ERIC:** Welcome to Strollcast! I'm Eric.

**MAYA:** And I'm Maya. We're your AI hosts, here to make research accessible while you're on the move.

**ERIC:** Today we're diving into a fascinating paper that tackles one of the biggest debates in AI: do intelligent agents need world models to be truly general? The paper is called "General agents contain world models" by Jonathan Richens, David Abel, Alexis Bellot, and Tom Everitt.

**MAYA:** This is a question that's been dividing the AI community for decades. On one side, you have folks advocating for explicit world models - internal representations of how the environment works. On the other side, there's the "model-free" camp, inspired by Rodney Brooks' famous "Intelligence without representation" idea. {{page: 1, section: "Introduction"}}

**ERIC:** Right! Brooks argued that the world is its own best model, and that intelligent behavior could emerge just from action-perception loops without needing explicit representations. It's a compelling vision - sidestep all the complexity of learning world models and just let intelligence emerge naturally. {{page: 1, section: "Introduction", excerpt: "Brooks famously proposed that the world is its own best model"}}

**MAYA:** But these authors provide a formal answer to this debate. They prove that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. Not only that - they show you can actually extract this world model from the agent's policy! {{page: 1, section: "Abstract"}}

**ERIC:** Think of it this way - imagine you're teaching someone to navigate a new city. If they can handle complex instructions like "go to the coffee shop, then the bookstore, then meet me at the park," they must have built up some internal map of how the streets connect. The authors prove this intuition mathematically.

## Background and Setup

**MAYA:** Let's break down their setup. They consider agents operating in what they call controlled Markov processes - basically environments where the current state and action determine the probability distribution over next states. Think of a robot navigating rooms in a building, or a game character moving through a grid world. {{page: 3, section: "2.2 Environment"}}

**ERIC:** The key innovation is how they define goals using Linear Temporal Logic, or LTL. Instead of just saying "reach state X," they can express complex sequential goals like "eventually reach the kitchen, then next go to the living room, then eventually return to the charging station." {{page: 6, section: "2.3 Goals", excerpt: "LTL extends classical propositional logic by introducing operators for reasoning about sequences of states over time"}}

**MAYA:** LTL gives them three main temporal operators: "Next" for immediate goals, "Eventually" for goals that must happen sometime in the future, and "Now" for current requirements. By chaining these together, they can create arbitrarily complex sequential tasks. {{page: 7, section: "Definition 2"}}

**ERIC:** Here's where it gets interesting - they define what makes an agent "general." A bounded goal-conditioned agent is one that can achieve goals with some maximum failure rate compared to the optimal agent, but only up to a certain complexity level. It's like saying "this robot can handle any task with up to 5 steps, and it succeeds at least 90% as often as a perfect robot would." {{page: 8, section: "Definition 5"}}

**MAYA:** This definition is crucial because it doesn't make strong assumptions about how the agent works internally - no rationality axioms or specific architectures. It just says the agent has some level of competence at goal-directed tasks of bounded complexity.

## The Main Result

**ERIC:** Now for the big reveal - Theorem 1. The authors prove that any such bounded agent fully determines a world model with bounded error. The accuracy of this model increases as the agent gets better at tasks or can handle more complex goals. {{page: 11, section: "Theorem 1"}}

**MAYA:** Let me unpack what this means. They show there's an algorithm that can look at just the agent's policy - how it chooses actions given different goals - and reconstruct the transition probabilities of the environment. The error in this reconstruction scales roughly as the agent's failure rate divided by the square root of goal complexity.

**ERIC:** The proof is elegant. They construct clever either-or goals where the agent has to choose between incompatible sub-goals. Since the agent satisfies a performance bound, its choice reveals information about which transitions are more likely in the environment. It's like inferring someone's knowledge of traffic patterns by watching them choose between different routes. {{page: 33, section: "A.5 Overview of proof"}}

**MAYA:** They also prove an interesting negative result for myopic agents - those that only optimize immediate outcomes. Theorem 2 shows that agents focusing only on one-step goals don't need to learn world models at all! The world model only becomes necessary for multi-step planning. {{page: 12, section: "Theorem 2"}}

**ERIC:** This connects to the classic exploration-exploitation tradeoff. If you only care about immediate rewards, you can succeed with simple reactive policies. But the moment you need to coordinate actions across multiple time steps - that's when world knowledge becomes essential.

## Experimental Validation

**MAYA:** The authors test their theory on a randomly generated environment with 20 states and 5 actions. They train agents on trajectories and measure how well their extraction algorithm recovers the true transition probabilities. {{page: 13, section: "3.1 Experiments"}}

**ERIC:** The results are striking. Even when agents strongly violated the theoretical assumptions - achieving maximum failure on some goals while succeeding on others - the algorithm still recovered accurate world models on average. The error scaled as predicted: roughly proportional to one over the square root of goal complexity.

**MAYA:** This suggests the result is quite robust. Real agents often have uneven capabilities, excelling at some tasks while completely failing at others. But as long as they achieve reasonable average performance on complex goals, the world model extraction still works.

## Implications and Consequences

**ERIC:** This theorem has profound implications. First, there's no model-free shortcut to general AI. Any agent that can generalize to long-horizon tasks must have learned a world model, whether explicitly or implicitly. The complexity of world modeling can't be avoided - it's mathematically necessary. {{page: 15, section: "Discussion", excerpt: "there is no way to train an agent capable of generalizing to long horizon tasks without learning a world model"}}

**MAYA:** This reframes the model-based versus model-free debate. Both approaches are ultimately learning world models - the difference is whether you do it explicitly or let it emerge implicitly in the policy. The authors argue this motivates explicit model-based approaches since you can't avoid the complexity anyway.

**ERIC:** There's a fascinating connection to emergent capabilities in large language models. The authors suggest their result provides a mechanism for how these capabilities could arise. To minimize failure across diverse training tasks, models are forced to learn implicit world models, which then support generalization to tasks they were never explicitly trained on. {{page: 15, section: "Emergent capabilities"}}

**MAYA:** The safety implications are equally significant. Many AI safety proposals require accurate predictive models of the agent-environment system. A common concern is that model-free agents might outpace our ability to understand them. But this result suggests we can always extract world models from capable agents. {{page: 16, section: "Safety"}}

**ERIC:** It's like having a guarantee that no matter how opaque an AI system becomes, if it's genuinely capable of complex goal-directed behavior, we can reverse-engineer its understanding of the world. The fidelity of this extracted model actually increases with the agent's capabilities.

**MAYA:** There are also limits this imposes on artificial intelligence. The authors argue that learning accurate world models is fundamentally constrained by the complexity and unpredictability of real-world systems. This means general agents are effectively limited to domains that are "solvable" in some sense. {{page: 17, section: "Limits on strong AI"}}

## Connections to Related Work

**ERIC:** This work beautifully complements existing research directions. In reinforcement learning, you have the planning triangle: given any two of environment model, goal, and optimal policy, you can determine the third. This paper fills in the missing direction - extracting the environment model from the policy and goal. {{page: 19, section: "Related work", excerpt: "Our result fills in the remaining direction, recovering the transition function given the agent's goal and their regret-bounded policy"}}

**MAYA:** It also connects to mechanistic interpretability research, where scientists try to uncover implicit world models in neural networks. But there's a key difference - this work shows you can extract world models from just the agent's behavior, not its internal representations. That makes it applicable even when you can't peek inside the black box. {{page: 19, section: "Related work"}}

**ERIC:** The authors also relate their work to causal inference and representation theorems from decision theory. Classical results like Savage's theorem show that rational agents behave as if they have beliefs and preferences. But this work goes further - it shows that capable agents must have learned accurate beliefs about the actual environment, not just any consistent belief system.

## Limitations and Future Work

**MAYA:** The current results are limited to fully observed environments where the agent can see the complete state. It's unclear what agents would need to learn about hidden variables in partially observable settings - that's an important direction for future work. {{page: 17, section: "Limitations"}}

**ERIC:** There are also questions about the specific extraction algorithms. Algorithm 1 that they derive is universal and unsupervised, but might not be the most efficient approach for real-world applications. Developing more scalable extraction methods could be valuable for AI safety and interpretability.

**MAYA:** The authors also point toward identifying "universal" task sets - collections of simple goals that are sufficient to guarantee an agent has learned a world model. These could be incredibly useful for training and evaluating general AI systems.

## Quizzes

**ERIC:** Alright, time for our first quiz. According to Theorem 2, what happens when you try to extract world models from myopic agents - those that only optimize for immediate outcomes?

**MAYA:** Take a moment to think about it... 

**ERIC:** The answer is that you can't extract any meaningful information about transition probabilities from myopic agents! The bounds you can derive are trivial - essentially saying the probability could be anywhere from 0 to 1. World models only become necessary when agents need to coordinate actions across multiple time steps.

**MAYA:** Here's our second quiz: In the experimental validation, what happened when agents strongly violated the theoretical assumptions by achieving maximum failure on some goals while succeeding on others?

**ERIC:** Think about whether the extraction algorithm would still work...

**MAYA:** Surprisingly, the algorithm still recovered accurate world models on average! Even though individual goal performance was inconsistent, as long as average performance was reasonable for complex goals, the world model extraction worked. This suggests the result is quite robust to realistic agent limitations.

## Conclusion

**ERIC:** This paper resolves a fundamental question that's been debated since the early days of AI. The authors prove that world models aren't just helpful for general intelligence - they're mathematically necessary. Any agent capable of flexible goal-directed behavior must have learned to predict how its environment works.

**MAYA:** The implications ripple across AI safety, interpretability, and our understanding of intelligence itself. We now have theoretical guarantees that world knowledge can be extracted from capable agents, regardless of their internal architecture. This provides new tools for understanding and controlling advanced AI systems.

**ERIC:** Perhaps most intriguingly, this work suggests that the remarkable capabilities emerging in large language models might stem from implicit world model learning. As these systems minimize failure across diverse tasks, they're forced to develop rich internal representations of how the world works.

**MAYA:** The paper also sets important limits - general intelligence is constrained by the fundamental difficulty of learning accurate world models. But it reframes this challenge, showing that both model-based and model-free approaches ultimately face the same underlying problem.

**ERIC:** For the AI safety community, this provides both reassurance and a research agenda. The reassurance is that capable agents can't hide their world knowledge from us. The agenda is developing better methods to extract and utilize these models for safety and alignment.

**MAYA:** Until next time, keep strolling.

**ERIC:** And may your gradients never explode.