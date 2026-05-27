# Hierarchical Pitman-Yor Processes
## From n-Gram Language Models to Neuro-Symbolic AI

> **Primary source:** Teh, Y.W. (2006). *A Hierarchical Bayesian Language Model based on Pitman-Yor Processes.* ACL.
> **Secondary source:** Lim, Buntine, Chen & Du (2016). *Nonparametric Bayesian Topic Modelling with Hierarchical Pitman-Yor Processes.* IJAR.

---

## Table of Contents

| # | Section |
|---|---------|
| 1 | The Dirichlet Process — Nonparametric Baseline |
| 2 | The Pitman-Yor Process — Power-Law Growth |
| 3 | Stick-Breaking Construction |
| 4 | Chinese Restaurant Process — Generative Intuition |
| 5 | The n-Gram Problem and Why Dirichlet Priors Fail |
| 6 | Suffix Tree Architecture — Three Benefits of Hierarchy |
| 7 | The Predictive Probability Recursion |
| 8 | HPYP as a Composable Module: The TNTM |
| 9 | Three-Layer Neuro-Symbolic Architecture |
| 10 | Layer 1: Neural Grounding via G₀ |
| 11 | Layer 2: Symbolic HPYP Scaffold |
| 12 | Layer 3: Relational GP |
| 13 | Joint Inference: What Runs Where |
| 14 | Conclusions |

---

---

# Part A — The Right Prior

---

## Slide 1 — The Dirichlet Process: Nonparametric Baseline

A **nonparametric model** does not fix the number of latent components K in advance — K grows with the data. The **Dirichlet Process (DP)** (Ferguson, 1973) is the standard nonparametric prior. It is defined by:

- **θ > 0** — concentration parameter (controls how quickly new clusters form)
- **G₀** — base distribution (the prior mean; any distribution over the domain)

A sample **G ~ DP(θ, G₀)** is itself a random discrete probability distribution.

**Pólya urn — one-parameter CRP:** Given N observations assigned to clusters so far (with c_k observations in cluster k):

$$P(\text{join table } k) = \frac{c_k}{\theta + N}, \qquad P(\text{open new table}) = \frac{\theta}{\theta + N}$$

**The fundamental limitation.** Under the DP, the expected number of unique clusters after N observations grows as:

$$E[K_N] \sim \theta \cdot \log(N) \qquad \text{(logarithmic growth)}$$

Natural language, concept inventories, and knowledge bases all exhibit **power-law** growth in unique types — $O(N^d)$ for $d \in (0,1)$. The DP predicts growth that is structurally too slow: it systematically under-represents the long tail of rare items. One additional parameter fixes this entirely.

---

## Slide 2 — The Pitman-Yor Process: Power-Law Growth

The **Pitman-Yor Process (PYP)** (Pitman & Yor, 1997) extends the DP with a second parameter:

| Parameter | Symbol | Range | Role |
|-----------|--------|-------|------|
| Discount | d | [0, 1) | Controls power-law tail weight |
| Strength | θ | > −d | Controls spread around base G₀ |
| Base measure | G₀ | any distribution | Prior mean over the support |

**G ~ PYP(d, θ, G₀)** is again a random discrete distribution. When **d = 0**, the PYP reduces exactly to the DP. When **d > 0**, the expected unique-type count after N observations grows as:

$$E[K_N] \sim \theta \cdot N^d \qquad \text{(power-law growth = Zipf's law)}$$

**Numerical comparison** ($\theta = 1$, $d = 0.75$, $N = 10{,}000$):

$$\begin{aligned}
\text{DP } (d=0): &\quad E[K_N] \approx \ln(10{,}000) \approx 9 \text{ unique types} \\
\text{PYP } (d>0): &\quad E[K_N] \approx 10{,}000^{0.75} \approx 1{,}000 \text{ unique types}
\end{aligned}$$

The difference is qualitative, not marginal. With the DP, the model implicitly treats rare items as noise; with the PYP, they are correctly modelled as legitimate low-frequency atoms. This matters most precisely where a symbolic AI system needs to be reliable: domain-specific entities, rare rules, novel concepts.

> **d is not a tuning knob.** It is a description of how knowledge is distributed in the world. For natural language, $d \approx 0.75$. The same range fits medical ontologies and software API surface areas.

---

## Slide 3 — Stick-Breaking Construction

The PYP can be constructed explicitly by the **stick-breaking** procedure, which generates the probability weights p₁, p₂, … over a countably infinite set of atoms:

**Step 1** — draw stick fractions:

$$V_k \sim \text{Beta}(1 - d,\; \theta + k \cdot d), \quad k = 1, 2, 3, \ldots$$

**Step 2** — assign weights:

$$p_1 = V_1, \qquad p_k = V_k \prod_{i=1}^{k-1}(1 - V_i), \quad k \geq 2$$

**Step 3** — draw atoms from the base:

$$\text{Atom}_k \sim G_0 \quad \text{(i.i.d.)}$$

**Step 4** — the random discrete distribution:

$$G = \sum_k p_k \, \delta_{\text{Atom}_k}$$

where δ_{x} is the point mass at x (equals 1 if the argument matches x, 0 otherwise). The construction reveals why d matters: the Beta distribution for V_k shifts with k in a way that slows the decay of p_k for larger k — producing heavier tails and more non-negligible atoms. When d = 0 the Beta reduces to Beta(1, θ) and the Sethuraman (1994) stick-breaking for the DP is recovered.

**Parameters in context:**
- **d** (discount) — higher d → Beta mass shifts toward 1 for each V_k → more atoms with non-trivial probability → heavier power-law tail
- **θ** (strength) — higher θ → more exploratory; lower θ → rich-gets-richer dominates

---

## Slide 4 — Chinese Restaurant Process: Generative Intuition

The **two-parameter Chinese Restaurant Process (CRP)** (Pitman, 1996) is the sequential generative procedure equivalent to the PYP. It makes the model's behaviour tangible.

**Setup:** Customers (observations) arrive one by one at a restaurant with infinitely many tables. Each table serves one "dish" (a value from G₀). Let:

- c_k — number of customers currently at table k
- T — total number of occupied tables
- N — total customers so far (= Σ_k c_k)

**Seating rule for customer N+1:**

$$P(\text{join table } k) = \frac{c_k - d}{\theta + N} \qquad \text{(rich-gets-richer, penalised by } d\text{)}$$

$$P(\text{open new table}) = \frac{\theta + d \cdot T}{\theta + N} \qquad \text{(new table draws a dish from } G_0\text{)}$$

**Why d changes everything.** Without the discount (d = 0), the new-table probability is θ/(θ + N) — a fixed fraction that decays to zero. With d > 0, the new-table probability is (θ + d·T)/(θ + N): as T grows, the numerator keeps pace with N, continuously creating room for new tables. This is the mechanism that produces O(θN^d) cluster growth.

**The CRP as a reusable interface.** Each CRP node takes a base distribution G₀ and produces a posterior predictive for new observations. This clean interface is what allows CRP nodes to be stacked (HPYLM), embedded in larger models (TNTM), or connected to neural encoders (neuro-symbolic extension).

---

---

# Part B — HPYLM: Hierarchy in Practice

---

## Slide 5 — The n-Gram Problem and Why Dirichlet Priors Fail

A **language model** assigns probabilities to word sequences. The standard **n-gram** approach estimates:

$$P(w_i \mid w_{i-n+1}, \ldots, w_{i-1}) \qquad \text{(probability of } w_i \text{ given the } n{-}1 \text{ preceding words)}$$

**The sparsity problem.** With a vocabulary of size V = 17,000 and trigrams (n = 3), the context space has V² ≈ 290 million entries. Most are never observed in training. Direct maximum-likelihood estimation assigns zero probability to unseen trigrams and catastrophically overfits.

**The classical answer: smoothing.** Blend high-order estimates with lower-order ones when counts are sparse. **Interpolated Kneser-Ney (IKN)** (Kneser & Ney, 1995) was the empirical gold standard for two decades — but it was purely heuristic. No principled derivation explained why it worked.

**Why Dirichlet priors fail.** Prior Bayesian attempts placed Dirichlet distributions over word probabilities. The Dirichlet cannot model power-law word-frequency distributions — it predicts near-uniform distributions over the vocabulary, systematically misallocating probability mass. Common words are fine; rare words receive far too little mass, and no amount of additional data corrects this structural bias.

**Teh's (2006) contribution:** replace the Dirichlet at every level with a PYP, organise the PYPs hierarchically over n-gram contexts, and show that the resulting model — the **Hierarchical Pitman-Yor Language Model (HPYLM)** — not only is principled but explains why IKN worked.

---

## Slide 6 — Suffix Tree Architecture: Three Benefits of Hierarchy

The HPYLM assigns one PYP node to each observed n-gram context in a **suffix tree** — a tree where each node is a word-sequence context and edges represent dropping the earliest word. For trigrams (n = 3):

```
  G_∅           ← PYP(d₀, θ₀, Uniform(1/V))       root: uniform over vocabulary
      │
  G_{w}         ← PYP(d₁, θ₁, G_∅)                unigram context: "cat", "the", ...
      │
  G_{u,w}       ← PYP(d₂, θ₂, G_{w})              bigram context: "the cat", ...
      │
  G_{v,u,w}     ← PYP(d₃, θ₃, G_{u,w})            trigram context: "the big cat", ...
```

**Structural rule:** each context's PYP uses its parent context's distribution as its own base measure G₀. The discount d_m and strength θ_m are **shared across all contexts of the same length** m. For a trigram model, the entire parameter set is **6 scalars** {d₀, θ₀, d₁, θ₁, d₂, θ₂} regardless of vocabulary size.

The hierarchy delivers **three benefits for free:**

**Benefit 1 — Regularization.** A trigram never seen in training gets c_{u··} = 0 at its node; the full interpolation weight shifts to the bigram parent. Probability never equals zero. No manual back-off rules needed.

**Benefit 2 — Automatic count-adaptive weighting.** The interpolation coefficient toward the parent,

$$\lambda_\mathbf{u} = \frac{\theta_{|\mathbf{u}|} + d_{|\mathbf{u}|} \cdot t_{\mathbf{u}\cdot\cdot}}{\theta_{|\mathbf{u}|} + c_{\mathbf{u}\cdot\cdot}}$$

is large when $c_{\mathbf{u}\cdot\cdot}$ is small (few observations → lean on prior) and approaches 0 when $c_{\mathbf{u}\cdot\cdot}$ is large (abundant evidence → trust direct counts). The model continuously recalibrates trust in each context based purely on the observed data. This is principled Bayesian shrinkage, not a fixed discount schedule.

**Benefit 3 — Compositionality.** Each CRP node exposes a standard interface: takes G₀ (the parent distribution) as input, produces a posterior predictive p(w | context) as output. This modularity means nodes can be stacked to any depth, replaced, or connected to external components — the same property that makes HPYP work inside a larger model (next slide).

---

## Slide 7 — The Predictive Probability Recursion

The probability of word w given context **u** is computed by a single recursive formula. Let the context length be |**u**| (number of words in **u**), and define:

| Symbol | Meaning |
|--------|---------|
| **u** | Context: word sequence immediately preceding the target word |
| w | Target word to predict |
| π(**u**) | Parent context: **u** with its earliest word dropped |
| \|**u**\| | Length of context **u** (ranges 0 to n−1) |
| c_{**u**w} | Number of times word w was observed after context **u** in training |
| t_{**u**w} | Number of CRP tables serving word w at node **u** (1 ≤ t_{**u**w} ≤ c_{**u**w}) |
| c_{**u**··} | Total word tokens after context **u**: Σ_w c_{**u**w} |
| t_{**u**··} | Total tables at node **u**: Σ_w t_{**u**w} |
| d_{\|**u**\|} | Discount parameter at context-length level |**u**| |
| θ_{\|**u**\|} | Strength parameter at context-length level |**u**| |

All counts c and t are non-negative integers. The formula:

$$p(w \mid \mathbf{u}) = \underbrace{\frac{c_{\mathbf{u}w} - d_{|\mathbf{u}|} \cdot t_{\mathbf{u}w}}{\theta_{|\mathbf{u}|} + c_{\mathbf{u}\cdot\cdot}}}_{\text{discounted direct evidence}} + \underbrace{\frac{\theta_{|\mathbf{u}|} + d_{|\mathbf{u}|} \cdot t_{\mathbf{u}\cdot\cdot}}{\theta_{|\mathbf{u}|} + c_{\mathbf{u}\cdot\cdot}}}_{\text{adaptive weight}} \cdot p(w \mid \pi(\mathbf{u}))$$

The recursion bottoms out at the root (context ∅): p(w | ∅) = 1/V.

**Reading the formula:**
- **First term:** direct evidence from context **u**, discounted by d times the table count. Intuitively: c_{**u**w} raw occurrences, minus a "table opening charge" that redistributes probability toward new words.
- **Second term:** the complementary weight (the two fractions always sum to 1) applied to the parent context — which itself recurses upward. When c_{**u**··} is small, this weight is near 1; when large, near 0.
- **Memory:** the entire model state is the integer pairs (c_{**u**w}, t_{**u**w}) for each observed (context, word) pair. No probability vectors are ever stored. No matrix is formed.

**The Kneser-Ney connection (Teh 2006, Theorem 1):** interpolated Kneser-Ney is exactly the HPYLM predictive distribution under two approximations: (1) $t_{\mathbf{u}w} := \min(1, c_{\mathbf{u}w})$ and (2) $\theta_{|\mathbf{u}|} := 0$. Under these constraints the formula reduces algebraically to IKN. The twenty-year gold standard was performing approximate Bayesian inference in the HPYLM, without knowing it.

---

## Slide 8 — HPYP as a Composable Module: The TNTM

**The key property of a CRP node is modularity:** its posterior factorises into exactly one likelihood factor that depends only on the local counts (c, t). Adding or removing a node from the model changes exactly one factor in the joint likelihood. This makes HPYP architecturally composable — it can be embedded inside any larger generative model.

**Proof of concept: the Twitter-Network Topic Model (TNTM)** (Lim, Buntine, Chen & Du, 2016) embeds a full HPYP topic model inside a Gaussian Process (GP) social-network model:

Text & hashtags → HPYP topic model, coupled via per-author topic distributions $\nu_i$

Social graph → GP over $\nu_i$ embeddings:

$$x_{ij} \sim \text{Bernoulli}(\sigma(Q_{ij})), \qquad Q_{ij} \sim \text{GP}\!\left(\cos(\nu_i, \nu_j),\; \kappa_{\text{RBF}}\right)$$

Here ν_i ∈ Δ^K is author i's topic distribution (a K-simplex vector inferred by the HPYP), which simultaneously serves as the GP kernel input. **Joint inference** alternates:
- **Collapsed blocked Gibbs** on the HPYP side — integer arithmetic only, CPU-native
- **Elliptical slice sampling (ESS)** on the GP side — efficient MCMC for Gaussian-prior posteriors

**Results on a Twitter corpus:**

| Model | Test Perplexity |
|-------|----------------|
| HDP-LDA (nonparametric baseline) | 840 |
| Author Topic Model (+ authorship, nonparametric) | 664 |
| **TNTM (HPYP + GP, jointly trained)** | **505** |

Every ablation — removing hashtags, authorship, the power-law prior, or the miscellany node — degrades performance. The 40% perplexity reduction over HDP-LDA shows that the HPYP, when coupled to a continuous component via a shared embedding, outperforms purely discrete or purely continuous alternatives.

**The architectural lesson:** the ν_i side-channel that connects HPYP to GP in the TNTM is the same coupling mechanism used in the full neuro-symbolic architecture (Layers 2 → 3). The TNTM is already a two-layer instance of that architecture.

---

---

# Part C — Toward Neuro-Symbolic AI

---

## Slide 9 — Three-Layer Neuro-Symbolic Architecture

The three-layer architecture generalises the HPYLM's suffix tree and the TNTM's hybrid design into a full neuro-symbolic system:

```
┌────────────────────────────────────────────────────────────────┐
│  LAYER 1 — NEURAL  (Perceptual Grounding)                      │
│                                                                │
│  Encoder_ω : X  →  ℝ^d      (CNN, Transformer, GNN, ...)      │
│  Encoder output serves as base distribution G₀                 │
│  for leaf PYP nodes in Layer 2                                 │
└─────────────────────────────┬──────────────────────────────────┘
                              │  continuous embeddings e_x ∈ ℝ^d
                              ▼
┌─────────────────────────────────────────────────────────────── ┐
│  LAYER 2 — SYMBOLIC  (HPYP Scaffold)                           │
│                                                                │
│  μ   ~ PYP(·, H_abstract)    ← abstract concept inventory     │
│  ν   ~ PYP(·, μ)             ← domain-level concepts          │
│  θ_d ~ PYP(·, ν)             ← context-specific mixtures      │
│  φ_k ~ PYP(·, Encoder_ω(X))  ← grounded symbol distributions  │
│                                                                │
│  Inference: collapsed blocked Gibbs on integer (c, t) counts   │
│  Automatic K, power-law prior, full posterior uncertainty       │
└─────────────────────────────┬──────────────────────────────────┘
                              │  concept embeddings ν_i ∈ ℝ^K
                              ▼
┌────────────────────────────────────────────────────────────────┐
│  LAYER 3 — RELATIONAL  (GP / Graph Network)                    │
│                                                                │
│  Q_ij ~ GP over concept embeddings ν_i                         │
│  x_ij ~ Bernoulli(sigmoid(Q_ij))                               │
│                                                                │
│  Models: KG edges, logical entailment, causal links,           │
│          spatial / temporal relations                          │
│  Inference: Metropolis-Hastings / Elliptical Slice Sampling    │
└────────────────────────────────────────────────────────────────┘
```

The TNTM (Slide 8) is a two-layer instance: HPYP (Layer 2) + GP (Layer 3), with author embeddings ν_i linking them. Connecting a neural encoder as Layer 1 and deepening the concept hierarchy gives the full system.

---

## Slide 10 — Layer 1: Neural Grounding via G₀

**The single key insight that makes neuro-symbolic integration possible:**

> The PYP base distribution G₀ is mathematically arbitrary. In the HPYLM it was Uniform(1/V). It can be anything — including the output of a neural encoder.

$$\text{Encoder}_\omega : X \to \mathbb{R}^d$$

where $\omega$ = encoder parameters, $X$ = raw perceptual input (tokens, pixels, audio, ...), $d$ = embedding dimension.

$$\phi_k \sim \text{PYP}(\alpha,\, \beta,\, \text{Encoder}_\omega(X))$$

Setting G₀ = Encoder_ω(X) turns each leaf PYP node into a **nonparametric cluster in neural embedding space**. The HPYP learns how many symbols are needed from data; the power-law prior ensures dominant symbols handle most observations with a long tail of rare but valid ones.

**Cross-modality grounding — same HPYP, different encoder:**

| Modality | Input X | Encoder | Atoms at HPYP leaves |
|----------|---------|---------|---------------------|
| Language | Tokens / sentences | Transformer (BERT, LLaMA) | Word / phrase embeddings |
| Vision | Image patches | CNN / ViT | Object / scene embeddings |
| Audio | Waveforms | Wav2Vec / Whisper | Phoneme / event embeddings |
| Graphs | Node neighbourhoods | GNN | Entity / relation embeddings |

**Where the GPU is needed:** only during encoder training (Layer 1 gradient updates). Once embeddings are precomputed and cached, Layers 2 and 3 operate on integer count arithmetic — entirely CPU-resident.

---

## Slide 11 — Layer 2: Symbolic HPYP Scaffold

The HPYP scaffold extends the HPYLM's suffix tree beyond language to general concept abstraction. Four levels emerge from data rather than being hand-engineered:

```
Level 0 — Root μ:    Abstract categories        (inferred automatically)
                     e.g., "physical object", "event", "relation"

Level 1 — ν:         Domain-level concepts
                     e.g., "vehicle", "animal", "speech act"

Level 2 — θ_d:       Context-specific mixtures   (one per input document d)
                     e.g., "this scene", "this utterance", "this agent"

Level 3 — φ_k:       Grounded symbol instances
                     Distributions over neural embeddings from Layer 1
```

Note: θ_d here denotes a per-document topic distribution (a K-simplex vector); d is the document index, not the PYP discount parameter.

**How sharing works (the CRP franchise):** when a new context (document d) arrives, it draws its topic distribution θ_d from the shared parent ν via the CRP. New contexts with few observations stay close to the global prior — they inherit the full global symbol inventory automatically. This is the same hierarchical mechanism as the HPYLM suffix tree, now operating over abstract concepts rather than word sequences.

**Inference state:** two integers per (node, label) pair: the customer count c and the table count t. Adding or removing a node changes exactly one factor in the joint posterior — the same modularity property that made the TNTM possible.

---

## Slide 12 — Layer 3: Relational GP

**Symbols are necessary but not sufficient.** A symbolic AI system requires **relations** between concepts — edges in a knowledge graph, logical entailments, causal links, spatial and temporal orderings. Layer 3 models these over the concept embeddings ν_i produced by Layer 2.

Let $\nu_i \in \mathbb{R}^K$ denote the topic (concept) distribution for entity $i$, inferred by the HPYP. Following the TNTM architecture:

$$Q_{ij} \sim \text{GP}\!\left(\mu = \cos(\nu_i, \nu_j),\; \kappa_{\text{RBF}}\right), \qquad x_{ij} \sim \text{Bernoulli}\!\left(\sigma(Q_{ij})\right)$$

Here:
- $Q_{ij} \in \mathbb{R}$: latent strength of the directed relation from entity $i$ to entity $j$
- $\cos(\nu_i, \nu_j) = \frac{\nu_i \cdot \nu_j}{\|\nu_i\| \cdot \|\nu_j\|}$: the GP prior mean
- $\kappa$: covariance kernel encoding that entity pairs with similar concept overlap have correlated link probabilities
- $x_{ij} \in \{0, 1\}$: observed edge (1 = relation present, 0 = absent)

**Applications of Layer 3:**

| Use case | Layer 3 configuration |
|----------|----------------------|
| Knowledge graph completion | Entity pairs; GP posterior fills missing edges |
| Logical entailment | Antecedent-consequent pairs; implication strength via Q |
| Causal links | Intervention × outcome; counterfactual via GP posterior |
| Scene graphs | Object pairs from vision encoder; spatial relation type |
| Discourse | Speaker-utterance pairs; discourse relation inventory from HPYP |

The **relation type inventory** is itself modelled by an HPYP prior at Layer 2 — the number of distinct relation types is inferred from data with a power-law prior over their frequency.

---

## Slide 13 — Joint Inference: What Runs Where

The three-layer system is trained jointly in an alternating loop that follows the TNTM's inference strategy directly:

```
Repeat until convergence:

  1. LAYER 2 — Collapsed blocked Gibbs  [CPU]
     For each observation x_dn (e.g., token n in document d):
       a. Decrement: remove x_dn's contribution from counts (c, t)
          at all relevant HPYP nodes, propagating up the hierarchy
       b. Block-sample: draw new topic assignment z_dn from
          conditional posterior p(z | all other assignments)
          = product of per-node likelihood ratios (integer arithmetic)
       c. Increment: update (c, t) counts with new assignment
     Resample concentration hyperparameters β via auxiliary variable sampler

  2. LAYER 1 — Gradient update  [GPU, training only]
     Compute ∂ log p(data | Z) / ∂ω conditioned on current assignments Z
     Update encoder weights ω
     (Note: gradients do not flow through discrete CRP assignments;
      straight-through estimator or EM with neural E-step bridges this)

  3. LAYER 3 — Elliptical Slice Sampling  [CPU]
     Sample GP latent values Q given current concept embeddings ν_i
     and observed edges X; ESS is designed for posteriors of the form
     p(Q) ∝ Likelihood(Q) × N(Q; μ_GP, κ)
```

**Hardware allocation:**

| Component | Core operation | Hardware |
|-----------|---------------|----------|
| Layer 2 Gibbs | Integer count increment / decrement | CPU |
| Layer 1 gradient | Dense matrix multiply (encoder) | GPU — training only |
| Layer 3 ESS | O(N²) GP covariance ops | CPU (small N), GPU (large N) |

At inference time with precomputed Layer 1 embeddings, Layers 2 and 3 are entirely CPU-resident. The inference state across all three layers is compact: integer (c, t) pairs for Layer 2, cached float vectors for Layer 1, and a GP posterior over the edge matrix for Layer 3.

---

---

# Conclusions

---

## Slide 14 — What This All Means

**The Pitman-Yor Process is the correct prior for symbolic knowledge:**
- Unique type growth is $O(\theta N^d)$, matching Zipf's law in language, ontologies, rule libraries, and any other domain with symbolic structure. The DP ($d = 0$) predicts $O(\theta \log N)$ — qualitatively wrong at any real scale.

**The HPYLM hierarchy delivers three properties for free — with only 2n scalar parameters:**
1. **Regularization:** never assign zero probability; hierarchical fallback from trigram → bigram → unigram is automatic
2. **Count-adaptive prior weighting:** the interpolation coefficient $\lambda_\mathbf{u} = (\theta + d \cdot t_{\mathbf{u}\cdot\cdot}) / (\theta + c_{\mathbf{u}\cdot\cdot})$ shrinks toward the parent exactly when evidence is sparse, and defers to direct counts when evidence is abundant
3. **Compositionality:** each CRP node exposes a standard interface (G₀ in, posterior predictive out) and can be chained, stacked, or connected to external components

**The Kneser-Ney result validates the framework:**
- IKN — the twenty-year empirical gold standard — is HPYLM under two approximations ($t_{\mathbf{u}w} := \min(1, c_{\mathbf{u}w})$, $\theta := 0$). Principled symbolic inference was right all along; the field had been approximating it without knowing it.

**HPYP is a composable building block, not just a language model:**
- The TNTM (Lim et al., 2016) proves that a full HPYP topic model can be jointly trained with a GP social-network model in a single inference loop, achieving 40% lower perplexity than HDP-LDA.

**One change turns HPYLM into neuro-symbolic AI:**
- Replace $G_0$ (uniform over vocabulary) with $\text{Encoder}_\omega(X)$ (neural encoder output). Symbols are now grounded in perception, their inventory grows automatically with a power-law prior, and inference remains integer-arithmetic and CPU-native.

**The inference state is just integers:**
- The entire Layer 2 state is $(c_{\mathbf{u}w}, t_{\mathbf{u}w})$ pairs — no probability vectors, no matrices. New observations increment counts directly. The model updates continuously without retraining; there is no catastrophic forgetting.

---

## Appendix — Notation Summary

| Symbol | Definition |
|--------|-----------|
| d | PYP discount parameter, d ∈ [0, 1); controls power-law tail weight |
| θ | PYP strength parameter, θ > −d; controls spread around G₀ |
| G₀ | Base distribution (prior mean); can be any distribution over the domain |
| G ~ PYP(d, θ, G₀) | A random discrete distribution drawn from the PYP |
| c_k | Customer count at CRP table k |
| T | Total number of occupied tables in a CRP node |
| N | Total number of customers in a CRP node (= Σ_k c_k) |
| **u** | n-gram context: the (n−1) words preceding the target word |
| w | Target word |
| π(**u**) | Parent context: **u** with its earliest (leftmost) word dropped |
| \|**u**\| | Length of context **u** in words (ranges from 0 to n−1) |
| c_{**u**w} | Count of word w observed after context **u** in training |
| t_{**u**w} | CRP table count for word w at context node **u** |
| c_{**u**··} | Σ_w c_{**u**w} — total token count at node **u** |
| t_{**u**··} | Σ_w t_{**u**w} — total table count at node **u** |
| d_m | Discount at context-length level m (shared across all contexts of length m) |
| θ_m | Strength at context-length level m |
| V | Vocabulary size |
| n | n-gram order (context length ranges from 0 to n−1) |
| μ | Global topic prior (root PYP in topic model) |
| ν | Intermediate topic prior (child of μ) |
| θ_d | Per-document topic distribution (subscript d = document index) |
| φ_k | Per-topic (or per-symbol) word distribution |
| z_{dn} | Latent topic assignment for word n in document d |
| ω | Neural encoder parameters |
| Encoder_ω | Neural encoder: X → ℝ^d (maps raw input to embedding) |
| ν_i | Concept embedding for entity i; output of HPYP Layer 2 |
| Q_{ij} | GP-distributed latent link strength between entities i and j |
| x_{ij} | Observed edge: x_{ij} ∈ {0, 1} |

---

## Appendix — Key References

| Paper | Contribution |
|-------|-------------|
| Pitman & Yor (1997). *Two-parameter Poisson-Dirichlet distribution.* Ann. Prob. | Mathematical foundation of PYP; stick-breaking, type growth |
| Teh (2006). *A hierarchical Bayesian language model based on Pitman-Yor processes.* ACL. | HPYLM; suffix tree architecture; Kneser-Ney correspondence |
| Lim, Buntine, Chen & Du (2016). *Nonparametric Bayesian topic modelling with hierarchical Pitman-Yor processes.* IJAR. | HPYP topic model; TNTM (HPYP + GP jointly trained) |
| Goldwater, Griffiths & Johnson (2006). *Interpolating between types and tokens.* NeurIPS. | Independent PYP–Kneser-Ney correspondence |
| Ishwaran & James (2001). *Gibbs sampling for stick-breaking priors.* JASA. | Practical MCMC for PYP; auxiliary variable hyperparameter sampler |
