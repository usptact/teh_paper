# Interviewer Notes — Hierarchical Pitman-Yor Processes

> Companion to `PYP_PRESENTATION_DISTILLED.md`.  
> Structure per slide: **Main message → If asked for more detail → Anticipated questions**.

---

## Slide 1 — The Dirichlet Process: Nonparametric Baseline

**Main message.**  
The DP is the standard nonparametric Bayesian prior, but its cluster growth is logarithmic — structurally too slow to model any domain with a long tail of rare types.

**If asked for more detail.**  
The key quantity is the expected number of distinct types after N observations: $E[K_N] \sim \theta \log N$ for the DP. Logarithmic growth means that even at N = 10 million, the model expects fewer than 17 distinct clusters (for θ = 1). Real language, ontologies, and knowledge bases grow much faster. This is not a tuning problem — it is a structural mismatch between the prior and the data-generating process.

**Anticipated questions.**

- *"Why not just use a larger finite K?"*  
  Fixing K forces a decision before seeing data. If K is too small, rare-but-real types are discarded. If K is too large, inference is wasteful and parameters are unidentifiable. The nonparametric approach lets K adapt to the evidence — you don't pay for types you don't need, and you're never capped.

- *"What is the base distribution G₀ in practice?"*  
  It is the prior belief about what a new atom looks like before any data. For a language model it is Uniform(1/V) over the vocabulary. For a neural-grounded model it becomes the encoder output distribution. The math is indifferent to what G₀ is — that is precisely what makes the model composable.

---

## Slide 2 — The Pitman-Yor Process: Power-Law Growth

**Main message.**  
Adding a single discount parameter d to the DP changes cluster growth from $O(\theta \log N)$ to $O(\theta N^d)$ — a qualitative difference that matches Zipf's law in every domain where symbolic structure matters.

**If asked for more detail.**  
The numerical comparison is the clearest way to make this concrete: at N = 10,000, the DP expects ~9 unique types, the PYP expects ~1,000. At N = 10 million the gap widens further. The discount d is not a free tuning knob — it is a measurable property of the domain. For natural language, $d \approx 0.75$; this same range has been observed in medical ontologies, software API surface areas, and gene ontologies. The strength θ controls how tightly the distribution concentrates around the most common types, but it does not change the asymptotic growth rate.

**Anticipated questions.**

- *"How do you estimate d in practice?"*  
  Via maximum likelihood over the held-out count distribution, or by placing a Beta(a, b) prior on d and resampling using an auxiliary variable Gibbs step. In the HPYLM, d is shared across all contexts of the same length, so you have at most n discount parameters for an n-gram model — very little to estimate.

- *"Doesn't a neural language model like GPT solve this automatically?"*  
  Neural models implicitly smooth through parameter sharing, but they do not explicitly maintain a power-law prior over the type inventory. They also do not give you a posterior over which types exist. The PYP gives you interpretable, countable uncertainty about the symbol inventory itself — something dense networks cannot natively provide.

---

## Slide 3 — Stick-Breaking Construction

**Main message.**  
Stick-breaking shows *why* d produces a heavy tail: the Beta distribution for each weight shifts with k, so later atoms decay more slowly than they would under the DP. When d = 0 every Beta is the same and you recover the DP; when d > 0 the tail never collapses.

**If asked for more detail.**  
The Beta parameters are $(1-d, \theta + k \cdot d)$. As k grows, the $k \cdot d$ term shifts mass toward 1 in the Beta, which means later atoms receive a less-aggressively-discounted share of remaining probability. For the DP (d = 0), the Beta reduces to Beta(1, θ) — the same for every k, so weights decay geometrically. For d > 0 the decay slows, producing the heavier tail. The construction also clarifies the roles of the two parameters: d sets the rate at which later atoms remain significant; θ sets how rapidly probability mass concentrates on the first few atoms.

**Anticipated questions.**

- *"Is stick-breaking used in inference, or just for analysis?"*  
  In collapsed Gibbs inference (which is what the HPYLM uses), you never explicitly draw stick fractions — the beta variables are integrated out. Stick-breaking is used when you need an explicit finite truncation, as in variational inference or sequential Monte Carlo. It is also the natural way to prove the power-law type-growth result analytically.

- *"What happens when d → 1?"*  
  The process becomes degenerate — every new atom gets an infinitesimally small weight, and the number of atoms with non-negligible probability grows without bound. In practice d is restricted to [0, 1). For language, empirically estimated d stays in [0.6, 0.85].

---

## Slide 4 — Chinese Restaurant Process: Generative Intuition

**Main message.**  
The CRP is the same PYP expressed as a step-by-step seating rule: popular tables grow sub-linearly, new tables open at a rate that keeps pace with the data. Each CRP node takes a base distribution as input and returns a predictive distribution as output — a clean interface that lets nodes be stacked, chained, or connected to anything.

**If asked for more detail.**  
The key contrast is between d = 0 and d > 0 in the new-table term. At d = 0, the numerator is just θ — a constant that vanishes relative to the growing denominator θ + N. At d > 0, the numerator is θ + d·T: as more tables open, T grows, and the new-table probability stabilises rather than collapsing. This is exactly the mechanism that produces $O(\theta N^d)$ growth instead of $O(\theta \log N)$. The CRP metaphor also prepares the audience for the HPYLM: stacking CRP nodes means each node's "dishes" are drawn from the parent node's distribution rather than from G₀ directly.

**Anticipated questions.**

- *"What do 'tables' represent in a language model?"*  
  In the HPYLM, each table at a context node represents one independent evidence contribution toward a particular word. Multiple observations of the same word can occupy different tables — this is what the table count t_{uw} tracks. The distinction between c_{uw} (raw occurrences) and t_{uw} (tables) is what generates the discounting behaviour in the predictive formula.

- *"Is the CRP exchangeable?"*  
  Yes — the joint distribution over cluster assignments is invariant to the order in which customers are seated. This is the de Finetti-type property that justifies the Bayesian interpretation. In practice it means you can update the model with new observations in any order and get the same posterior.

---

## Slide 5 — The n-Gram Problem and Why Dirichlet Priors Fail

**Main message.**  
N-gram sparsity is severe; IKN was a 20-year heuristic with no theoretical basis; and Dirichlet priors can't fix the underlying problem because their tails are too thin. Teh's answer: replace the Dirichlet at every level with a PYP and arrange the PYPs in a suffix tree over n-gram contexts — that model is the HPYLM.

**If asked for more detail.**  
The scale makes the problem concrete: V = 17,000 gives V² = 289 million trigram contexts, most unobserved. The Dirichlet prior fails not because of its smoothing properties but because of its structural form — a Dirichlet places a symmetric prior over a fixed vocabulary, which has exponential tails, not power-law tails. Common words get reasonable probability but rare words get systematically too little regardless of how you tune the Dirichlet hyperparameters. The HPYLM fixes this at the root by replacing the Dirichlet with a PYP at every level of the hierarchy, which naturally concentrates mass on frequent types while maintaining a heavy tail for rare ones.

**Anticipated questions.**

- *"Modern LLMs have largely replaced n-gram models. Why does this matter?"*  
  Two reasons. First, LLMs operate on a fixed, static vocabulary with a fixed parameter count — they cannot natively handle open-ended symbol inventories or continuously update a type count. Second, the HPYLM analysis motivates the full neuro-symbolic architecture: the exact same principle (hierarchical PYP prior over contexts) scales from word n-grams to abstract concept hierarchies. The language model is the proof of concept, not the final destination.

- *"Why couldn't you just smooth more aggressively with the Dirichlet?"*  
  Because the mismatch is structural, not parametric. Changing the Dirichlet concentration parameter makes the prior broader or narrower, but it stays an exponential-family distribution whose tails decay faster than any power law. No value of the Dirichlet hyperparameter produces the right distributional shape for rare types.

---

## Slide 6 — Suffix Tree Architecture: Three Benefits of Hierarchy

**Main message.**  
The HPYLM's suffix tree delivers three properties for free — regularization, count-adaptive weighting, and compositionality — using only 2n scalar parameters for an n-gram model of any vocabulary size.

**If asked for more detail.**  
The three benefits follow directly from the CRP's seating rule applied hierarchically. Regularization: a context with zero observations gets zero weight in its own node; the interpolation coefficient $\lambda_\mathbf{u}$ becomes 1, and the full probability mass is delegated to the parent. Count-adaptive weighting: $\lambda_\mathbf{u}$ is a Bayesian shrinkage coefficient — large when evidence is sparse, small when evidence is abundant, computed automatically from the observed (c, t) counts. Compositionality: the "G₀ in, posterior predictive out" interface of each CRP node is the same regardless of depth, which is what allows the tree to be extended (HPYP topic model), coupled to external components (TNTM), or grounded in neural embeddings.

**Anticipated questions.**

- *"Does the suffix tree become intractable for large n or large vocabularies?"*  
  The tree has at most one node per observed context — it is sparse. In practice, a trigram model with a 100,000-word vocabulary has far fewer than V² = 10 billion trigram nodes; only observed contexts appear. The model is linear in corpus size, not in the number of potential contexts. Inference time per observation is proportional to the context length n, not to the vocabulary.

- *"What exactly are the 6 scalar parameters for a trigram model?"*  
  Three discount–strength pairs, one per context-length level: $(d_0, \theta_0)$ for the root (uniform), $(d_1, \theta_1)$ for unigram contexts, $(d_2, \theta_2)$ for bigram contexts. These are shared across every node at that level — so every bigram context node in the entire model uses the same $d_2$ and $\theta_2$. This sharing is what keeps the model low-dimensional regardless of vocabulary size.

---

## Slide 7 — The Predictive Probability Recursion

**Main message.**  
One recursive formula, bottoming out at a uniform distribution over the vocabulary, encodes everything the HPYLM knows; interpolated Kneser-Ney is this formula under two integer approximations — the field's 20-year heuristic was principled Bayesian inference in disguise.

**If asked for more detail.**  
The formula has two additive terms. The first is the discounted direct evidence at context node u: raw occurrence count $c_{\mathbf{u}w}$ minus $d \cdot t_{\mathbf{u}w}$ (a "table-opening charge" that redistributes mass toward novel words). The second term is the complementary weight multiplied by the parent's predictive distribution — and it recurses up the tree. The two fractions always sum to 1, so this is a proper probability. The entire model state is the integer pairs $(c_{\mathbf{u}w}, t_{\mathbf{u}w})$ — no probability vectors are stored. The Kneser-Ney connection (Theorem 1 in Teh 2006) follows from setting $t_{\mathbf{u}w} := \min(1, c_{\mathbf{u}w})$ and $\theta_{|\mathbf{u}|} := 0$; both are drastic simplifications that collapse the full Bayesian posterior to a point estimate, but they preserve the qualitative shape.

**Anticipated questions.**

- *"What is the difference between c_{uw} and t_{uw}, and why does it matter?"*  
  $c_{\mathbf{u}w}$ counts raw occurrences — how many times word w appeared after context u. $t_{\mathbf{u}w}$ counts the number of CRP tables serving w at that node, which is between 1 and $c_{\mathbf{u}w}$. Multiple occurrences of the same word can sit at different tables, reflecting the prior belief that repeated observations may come from different generative "causes." The discounting term $d \cdot t_{\mathbf{u}w}$ penalises based on the number of distinct contributing events, not just the raw count — this is what produces the better generalisation to rare words.

- *"How is $t_{uw}$ updated during inference?"*  
  Via a collapsed blocked Gibbs sampler that alternates between: (a) removing one observation's contribution and (b) resampling which table it sits at using the CRP seating probabilities. The table assignment is a latent variable; the sampler marginalises over it. No explicit probability distribution is ever stored — only the (c, t) integer counters are maintained.

---

## Slide 8 — HPYP as a Composable Module: The TNTM

**Main message.**  
The KN result confirms the HPYLM's inductive biases were correct all along. That same hierarchy of PYP nodes is also architecturally modular: plug it into a larger model and only one factor in the joint likelihood changes. The TNTM (HPYP + GP, jointly trained) proves the point with a 40% perplexity gain over HDP-LDA.

**If asked for more detail.**  
The key modularity property: the joint likelihood factorises into one factor per CRP node, and each factor depends only on the local (c, t) counts at that node. Plugging the HPYP into the TNTM changes exactly those factors — nothing else. The coupling between the two components is through the per-author topic vector $\nu_i$, which simultaneously drives the HPYP's word distributions and the GP's link probability predictions. Joint inference alternates between collapsed Gibbs (for the HPYP) and elliptical slice sampling (for the GP); neither disturbs the other's state except through $\nu_i$. The 40% perplexity reduction over HDP-LDA occurs because the GP's social-network information regularises topic assignments in document-sparse contexts — exactly the count-adaptive weighting mechanism from Slide 6, now amplified by external signal.

**Anticipated questions.**

- *"Is 40% perplexity reduction large in practice?"*  
  Perplexity is exponential in cross-entropy, so a 40% reduction corresponds to a large improvement in bits-per-word. In language modelling, gains above 10–15% over a strong baseline are considered significant. Going from HDP-LDA (840) to TNTM (505) is not a marginal tweak — it reflects genuinely different information being captured. The ablation table in the slide confirms that every component contributes: removing hashtags, authorship, or the power-law prior each degrades performance.

- *"Why does combining HPYP with a GP outperform using either alone?"*  
  The HPYP captures lexical structure (which words co-occur in topics) but treats each document independently unless they share authors. The GP captures social structure (who links to whom) but is indifferent to word content. The $\nu_i$ bridge forces both components to agree on a shared topic representation — each one regularises the other's parameter estimates in data-sparse regions.

---

## Slide 9 — Three-Layer Neuro-Symbolic Architecture

**Main message.**  
The TNTM already showed Layers 2 and 3 working together. This slide maps the full target: add a neural encoder as Layer 1, deepen the concept hierarchy, and let each layer keep its own inference algorithm. How Layer 1 actually connects to Layer 2 is the subject of the next slide.

**If asked for more detail.**  
The TNTM is already Layers 2 + 3 of the full architecture. Adding Layer 1 (the encoder) is the step that grounds abstract symbols in raw perception. Each layer interfaces to the next through a well-defined embedding: encoder output embeddings flow down to HPYP leaf nodes (as the base measure G₀); HPYP concept embeddings $\nu_i$ flow up to the GP (as the kernel input). No gradient crosses the discrete CRP boundary during inference; the layers are coupled by the shared representation, not by a continuous loss signal. The system is designed so that each layer can be upgraded or replaced independently — swapping the encoder from BERT to a vision transformer changes Layer 1 without touching Layers 2 or 3.

**Anticipated questions.**

- *"Can you train all three layers jointly end-to-end?"*  
  Layers 1 and 3 can receive gradient updates (their parameters are continuous). Layer 2 does not — CRP assignments are discrete and non-differentiable. The gradient interface between Layer 1 and Layer 2 requires either a straight-through estimator, a Gumbel-softmax relaxation, or an EM formulation where Layer 2 runs the E-step and Layer 1 updates on the expected sufficient statistics. This is an open engineering problem, not a fundamental barrier.

- *"What are the failure modes of this architecture?"*  
  Three main ones: (1) scale — Gibbs sampling on the HPYP is practical up to ~1M documents before mixing time becomes a bottleneck (variational or streaming alternatives exist); (2) the discrete-continuous gradient interface adds complexity during joint training; (3) the GP in Layer 3 is $O(N^2)$ in the number of entity pairs, which requires sparse approximations at large scale. None of these is unique to HPYP — they are well-studied problems in the Bayesian nonparametrics and GP literatures.

---

## Slide 10 — Layer 1: Neural Grounding via G₀

**Main message.**  
The single substitution $G_0 \leftarrow \text{Encoder}_\omega(X)$ turns the HPYLM into a multi-modal neuro-symbolic system: symbols are now grounded in perception, their inventory grows automatically, and inference stays CPU-native after the encoder is precomputed.

**If asked for more detail.**  
In the original HPYLM, $G_0$ is Uniform(1/V) — a flat prior over a fixed vocabulary. Replacing it with the encoder output does two things: (1) atoms are now points in continuous embedding space, so the HPYP can cluster observations that are semantically similar rather than just textually identical; (2) the encoder's representations carry cross-domain information — a word embedding from a Transformer and an object embedding from a CNN are both admissible atoms at the same HPYP node. The GPU is used only during encoder training (gradient updates on $\omega$). Once embeddings are precomputed and cached, every Gibbs step in Layer 2 operates on integer counters against a fixed lookup table — entirely CPU-resident.

**Anticipated questions.**

- *"What happens if the encoder representations drift — e.g., after fine-tuning?"*  
  The HPYP's (c, t) counts reference atoms defined by the encoder's embedding at the time they were assigned. If the encoder is retrained, the atoms shift in embedding space and the existing counts become stale. The clean solution is to treat encoder retraining as a full re-inference pass — which is feasible because the Gibbs sampler can restart from the current (c, t) state and converge quickly when the shift is small. This is analogous to how topic models handle corpus updates.

- *"Why not just use a neural model end-to-end and skip the symbolic layer?"*  
  Because the symbolic layer provides things the encoder cannot: an explicit, inspectable inventory of symbols with uncertainty estimates; power-law priors over symbol frequency; continuous learning via count increments without retraining; and a structured interface for relational reasoning in Layer 3. The encoder handles the hard perceptual grounding problem; the HPYP handles the hard combinatorial abstraction problem. Conflating both into a single dense network solves neither as well.

---

## Slide 11 — Layer 2: Symbolic HPYP Scaffold

**Main message.**  
Four levels of abstraction — from root abstract categories down to grounded symbol instances — emerge automatically from data via the CRP franchise, using the same hierarchical sharing mechanism as the HPYLM suffix tree.

**If asked for more detail.**  
The CRP franchise is the key mechanism: when a new context (document, scene, agent) arrives, its topic distribution $\theta_d$ is drawn from the shared parent $\nu$ via the CRP seating rule. Contexts with few observations stay close to the global prior, inheriting the full existing symbol inventory. As evidence accumulates, $\lambda_\mathbf{u}$ shrinks and the context develops its own specialised distribution. The depth-4 hierarchy is not hand-engineered — the levels emerge from the data's count structure. The entire inference state across all levels is integer (c, t) pairs: adding a new observation increments exactly those counters at the nodes on its path from leaf to root.

**Anticipated questions.**

- *"How does the system decide which level a new concept belongs to?"*  
  It does not explicitly assign concepts to levels — the hierarchy is over contexts (documents, agents, scenes), not over concepts. Each level is a different granularity of context. A concept like "vehicle" appearing at level 1 ($\nu$) means the word-distribution for "vehicle" is shared across all documents, while its appearance at level 2 ($\theta_d$) reflects how a specific document uses the vehicle concept. The abstraction hierarchy emerges from which contexts share topic distributions, not from an explicit category assignment.

- *"What prevents the model from inventing spurious concepts from noise?"*  
  The power-law prior. A new concept (table) can only be opened at cost $(θ + d \cdot T)/(θ + N)$. When evidence is sparse relative to the existing inventory, this probability is low — the model prefers to seat a noisy observation at an existing table. New concepts emerge only when they are consistently supported across many observations. This is the same regularisation mechanism described in Slides 6–7, now operating over abstract concepts rather than words.

---

## Slide 12 — Layer 3: Relational GP

**Main message.**  
Symbols need relations; the GP over HPYP concept embeddings models knowledge-graph edges, logical entailments, and causal links — using the same ν_i bridge that coupled the TNTM's text and social-graph components.

**If asked for more detail.**  
The GP prior mean is cosine similarity between entity concept embeddings $\nu_i$ and $\nu_j$: entities with similar topic distributions are a priori more likely to be related. The RBF kernel encodes that nearby pairs in concept space have correlated link probabilities. The GP posterior, conditioned on observed edges, propagates relational information across the knowledge graph — filling in missing edges and adjusting confidence on existing ones. The relation type inventory is itself an HPYP node in Layer 2, so the number of distinct relation types is inferred automatically with a power-law prior over relation frequency.

**Anticipated questions.**

- *"Doesn't the GP scale as O(N³) in the number of entity pairs?"*  
  Full GP inference is O(N³), which is prohibitive for large entity sets. In practice, sparse GP approximations (inducing points, Nyström approximation) reduce this to O(NM²) where M ≪ N is the number of inducing points. For knowledge graphs with sparse observed edges, further reductions come from only evaluating the GP over observed and candidate edges, not the full N × N matrix.

- *"Can you add new entity types or relation types at inference time without retraining?"*  
  Yes — this is one of the architectural advantages of the nonparametric approach. A new entity enters as a new node in Layer 3 with an uninitialised $\nu_i$ that inherits the global topic prior from Layer 2. A new relation type is a new atom in the Layer 2 HPYP node for relation types. Both integrate into the existing inference loop without modifying any other parameters.

---

## Slide 13 — Joint Inference: What Runs Where

**Main message.**  
The three layers run three different inference algorithms in an alternating loop — Gibbs (Layer 2, CPU), gradient updates (Layer 1, GPU, training only), ESS (Layer 3, CPU) — with no shared state between algorithms except through the shared embeddings.

**If asked for more detail.**  
The collapsed blocked Gibbs sampler for Layer 2 is the heart of the system: for each observation, remove its contribution from the (c, t) counts, compute the conditional posterior over topic assignments (a product of per-node likelihood ratios using only integer arithmetic), resample, and update the counts. This is mathematically exact, requires no floating-point probability vectors, and handles new observations by increment. Layer 3's elliptical slice sampling is efficient for posteriors of the form $p(Q) \propto \text{Likelihood}(Q) \times \mathcal{N}(Q; \mu_{\text{GP}}, \kappa)$ — it requires no step-size tuning and mixes well when the likelihood is smooth. The gradient step in Layer 1 is the only part that requires a GPU, and only during training — at inference time, Layer 1 is frozen and its output is a precomputed lookup.

**Anticipated questions.**

- *"How do you handle the gradient interface between the discrete Layer 2 and the continuous Layer 1?"*  
  Three options exist: (1) straight-through estimator — treat the discrete CRP assignment as if it were differentiable during the backward pass (a common approximation in discrete latent variable models); (2) Gumbel-softmax relaxation — replace discrete assignments with a continuous approximation that anneals to hard assignments during training; (3) EM with a neural E-step — run Layer 2 Gibbs as the E-step, compute expected sufficient statistics, and do a gradient step on Layer 1 using those expectations. Option 3 is the most principled but requires holding the two optimisers in sync.

- *"How does this compare to variational inference for the same model?"*  
  Gibbs sampling is exact asymptotically (in the number of samples) but slow to mix for large models. Variational inference introduces a bias (the mean-field or structured approximation) but converges faster and scales better. For production systems, variational or stochastic variational inference (SVI) would likely replace Gibbs for Layer 2. The key architectural point — integer-state symbolic layer, relational GP, neural encoder — is agnostic to which approximate inference algorithm is used.

---

## Slide 14 — Conclusions

**Main message.**  
PYP was the correct prior all along; the HPYLM hierarchy delivers regularisation, adaptive weighting, and compositionality at minimal cost; one substitution ($G_0 \leftarrow \text{Encoder}_\omega(X)$) makes the system neuro-symbolic; and the integer inference state enables continuous learning without catastrophic forgetting.

**If asked for more detail.**  
The six conclusion bullets trace a single chain of reasoning: power-law prior (Slide 2) → hierarchical structure with 2n parameters (Slide 6) → Kneser-Ney as approximate Bayesian inference (Slide 7) → composability proven by TNTM (Slide 8) → neuro-symbolic grounding via G₀ substitution (Slide 10) → CPU-native continuous learning (Slide 13). Each step follows from the previous. The Kneser-Ney result is the strongest empirical validation: it shows that the field was computing approximate posteriors under the HPYLM for 20 years, which means the model's inductive biases were already correct — what was missing was the explicit probabilistic framing that allows extension to topics, to relations, and to neural grounding.

**Anticipated questions.**

- *"If this framework was available in 2006, why didn't neuro-symbolic AI happen then?"*  
  Three reasons: (1) in 2006 there were no large pre-trained neural encoders whose embeddings were good enough to serve as G₀ — BERT, GPT, and ViT came a decade later; (2) GPU-accelerated deep learning was not mature enough to make the encoder training tractable; (3) the HPYLM community and the neural community developed largely in parallel without cross-pollination. The architectural insight (G₀ = encoder output) is straightforward in hindsight, but it required both sides to mature before it became obviously correct to pursue.

- *"What is the main limitation of the HPYP approach at scale?"*  
  Gibbs sampling mixes slowly for large document collections — this is the primary bottleneck. Variational or streaming alternatives (online VB, stochastic gradient MCMC) are known and applicable, but they introduce approximation bias. A secondary limitation is the discrete-continuous gradient interface for joint training, which requires approximation. Neither limitation is fundamental — both have active research communities working on them — but they do mean that the architecture described here is closer to a research prototype than a production system as of 2024.

---

*End of interviewer notes.*
