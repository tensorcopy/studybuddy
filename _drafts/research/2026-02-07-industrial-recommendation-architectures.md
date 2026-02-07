---
topic: "Industrial Recommendation System Architectures (2023-2026)"
date: 2026-02-07
status: research-brief
---

# Research Brief: Industrial Recommendation System Architectures (2023-2026)

## Executive Summary

The industrial recommendation systems landscape has undergone a paradigm shift between 2023 and 2026. The dominant trend is the move from traditional Deep Learning Recommendation Models (DLRMs) --- which rely on manually engineered features, embedding tables, and shallow feature interaction networks --- toward **generative recommendation** architectures that treat the recommendation problem as sequence transduction. The key players driving this shift are Meta (HSTU, Wukong, GEM, Andromeda), ByteDance (OneTrans, HLLM, Monolith), Kuaishou (OneRec), Meituan (MTGR, DFGR), and Google (PLUM).

Three architectures were requested in depth: **Wukong** (Meta), **HSTU** (Meta), and **OneTrans** (ByteDance). Additionally, this brief covers six other notable systems: **GEM** (Meta), **Andromeda** (Meta), **OneRec** (Kuaishou), **HLLM** (ByteDance), **MTGR** (Meituan), **DFGR** (Meituan), and **PLUM** (Google/YouTube).

---

## 1. Wukong --- Meta (ICML 2024)

### Paper Details

- **Title**: "Wukong: Towards a Scaling Law for Large-Scale Recommendation"
- **Authors**: Buyun Zhang, Liang Luo, Yuxin Chen, Jade Nie, Xi Liu, Shen Li, Yanli Zhao, Yuchen Hao, Yantao Yao, Ellie Dingqiao Wen, Jongsoo Park, Maxim Naumov, Wenlin Chen
- **Venue**: ICML 2024 (41st International Conference on Machine Learning)
- **arXiv**: [2403.02545](https://arxiv.org/abs/2403.02545)
- **Published proceedings**: [PMLR v235](https://proceedings.mlr.press/v235/zhang24ao.html)

### Core Problem

Recommendation models have not exhibited scaling laws analogous to those observed in LLMs. As you scale up existing architectures (DCN, DLRM, DeepFM), quality improvements plateau or degrade. The problem is architectural: prior feature-interaction networks do not scale effectively with increased compute. Wukong asks: **can we design a recommendation architecture that exhibits predictable, monotonic quality improvement as we increase model size?**

### Architecture: Stacked Factorization Machines

Wukong is built entirely from stacked **Factorization Machine Blocks (FMBs)** and **Linear Compression Blocks (LCBs)**, organized into an "Interaction Stack."

**Each layer $i$ consists of two parallel blocks:**

1. **Factorization Machine Block (FMB)**: Computes pairwise feature interactions.
   - Input: embeddings $X_i \in \mathbb{R}^{n_i \times d}$ from the previous layer
   - Compute FM interaction matrix: $X_i X_i^T$ (an $n_i \times n_i$ matrix of pairwise interactions)
   - Flatten, normalize, and pass through an MLP:
     $$\text{FMB}(X_i) = \text{MLP}(\text{LN}(\text{flatten}(X_i X_i^T)))$$
   - Reshape output to $n_F$ embeddings of dimension $d$

2. **Linear Compression Block (LCB)**: Linearly projects input embeddings.
   $$\text{LCB}(X_i) = W_L X_i$$
   where $W_L \in \mathbb{R}^{n_L \times n_i}$, producing $n_L$ compressed embeddings.

**Layer output**: The outputs of FMB and LCB are concatenated:
$$X_{i+1} = [\text{FMB}(X_i); \text{LCB}(X_i)]$$

So the next layer receives $n_F + n_L$ embeddings.

### Key Insight: Binary Exponentiation of Interaction Order

The critical design insight is that **stacking FM layers causes the interaction order to grow exponentially**:

- Layer 1: captures 2nd-order interactions (and passes through 1st-order via LCB)
- Layer 2: captures up to 4th-order interactions (2nd-order of 2nd-order results)
- Layer 3: captures up to 8th-order interactions
- Layer $L$: captures up to $2^L$-th order interactions

This "binary exponentiation" means Wukong can model extremely high-order feature interactions with relatively few layers, something that DCN (which grows interaction order linearly with depth) cannot achieve.

### Upscaling Strategy

Wukong can be scaled along multiple axes:
- **Depth**: More stacked layers (increases interaction order exponentially)
- **Width of FMB** ($n_F$): More output embeddings from the FM block
- **Width of LCB** ($n_L$): More compressed embeddings
- **FM compression rank** ($k$): Rank of the optimized FM computation
- **MLP size**: Larger MLPs within FMB

The paper proposes a **synergistic upscaling strategy** that scales all dimensions jointly, finding that balanced scaling (as opposed to scaling only one dimension) yields the best quality-vs-compute tradeoff.

### Results

- **Public datasets**: Wukong consistently outperforms DCNv2, DLRM, DeepFM, AutoInt, and other SOTA models across six public datasets (Criteo, Avazu, KDD Cup 2012, etc.).
- **Scaling law**: Wukong exhibits a **smooth, monotonic quality improvement across two orders of magnitude** in model complexity (measured in GFLOP/example), extending beyond 100 GFLOP/example. Prior architectures plateau or degrade well before this point.
- **Internal Meta data**: Maintains superiority on large-scale internal datasets, where the scaling law holds even more clearly.

### Role in the Pipeline

Wukong is designed as a **ranking model** --- it scores and ranks candidates that have already been retrieved. It replaces the feature-interaction component (DCNv2, DeepFM, etc.) in the traditional DLRM stack.

### Production Deployment

Wukong's architecture was subsequently adopted and enhanced in **Meta's GEM** (Generative Ads Model), which uses "stackable factorization machines with cross-layer attention connections" as its core building block (see GEM section below). GEM is deployed across Facebook and Instagram ads recommendation.

---

## 2. HSTU --- Meta (ICML 2024)

### Paper Details

- **Title**: "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
- **Authors**: Jiaqi Zhai et al.
- **Venue**: ICML 2024 (PMLR 235:58484-58509)
- **arXiv**: [2402.17152](https://arxiv.org/abs/2402.17152)
- **Code**: [github.com/meta-recsys/generative-recommenders](https://github.com/meta-recsys/generative-recommenders)

### Core Problem

Traditional DLRMs treat recommendation as a discriminative scoring problem: given a (user, item) pair, predict the probability of engagement. This requires extensive manual feature engineering and separate models for retrieval and ranking. HSTU instead reformulates recommendation as a **sequential transduction** problem: given a user's history of actions, generate the next actions. This is the "Generative Recommender" (GR) paradigm.

### Architecture: Hierarchical Sequential Transduction Unit

HSTU is a modified Transformer architecture specifically designed for recommendation data, which differs from NLP data in critical ways:
- **High cardinality**: Billions of items (vs. ~100K vocabulary in NLP)
- **Non-stationary vocabulary**: New items constantly appear and old ones expire
- **Heterogeneous tokens**: Each "token" in a user's history consists of (item, action_type, timestamp) --- not just an item ID
- **Intensity matters**: The *number* of interactions with a topic signals preference intensity, which softmax normalization destroys

**Each HSTU layer has three sub-layers:**

1. **Pointwise Projection**: Projects input into four components:
   $$U(X), V(X), Q(X), K(X) = \text{Split}(\varphi_1(f_1(X)))$$
   where $f_1$ is a linear projection and $\varphi_1$ is the SiLU activation.

2. **Spatial Aggregation** (the modified attention):
   $$A(X) V(X) = \frac{\varphi_2(Q(X) K(X)^T + r_{ab}^{p,t}) \cdot V(X)}{N}$$
   where:
   - $\varphi_2$ is **SiLU** (not softmax!) --- this is the key departure from standard Transformers
   - $r_{ab}^{p,t}$ is the sum of **relative positional attention bias** and **relative temporal attention bias**
   - $N$ is a normalization constant (not row-wise softmax)

3. **Pointwise Transformation** (feature interaction via gating):
   $$Y(X) = f_2(\text{Norm}(A(X) V(X)) \odot U(X))$$
   The Hadamard product ($\odot$) with $U(X)$ acts as a gating mechanism, and $f_2$ is a linear output projection.

### Why Not Softmax?

Softmax normalization converts attention scores into a probability distribution that sums to 1. This destroys information about the **absolute magnitude** of attention --- in recommendation, a user who has interacted with 100 cooking videos should get a different signal than one who interacted with 5, even if the relative distribution is similar. Pointwise SiLU activation preserves this intensity information.

### M-FALCON: Efficient Multi-Candidate Inference

**M-FALCON** (Microbatched-Fast Attention Leveraging Cacheable Operations) solves the inference efficiency problem. When ranking $m$ candidates given a user history of length $n$:
- The user's history (the "prefix") is processed once, and its K/V representations are cached
- Candidates are divided into micro-batches with modified attention masks to prevent information leakage between candidates
- Complexity scales **linearly** with the number of candidates (unlike DLRMs where each candidate requires a full forward pass)
- Result: a **285x more complex GR model** can be served at **3x higher QPS** than a traditional DLRM

### Results

- **Public datasets**: Up to **65.8% improvement in NDCG** over baselines on synthetic and public benchmarks
- **Speed**: 5.3x to 15.2x faster than FlashAttention2-based Transformers on 8192-length sequences
- **Production**: 1.5 trillion parameter model deployed on **multiple surfaces** of Meta's platform (billions of daily active users), achieving **12.4% metric improvement** in online A/B tests
- **Scaling**: Demonstrates LLM-like scaling behavior --- larger models consistently improve quality

### Role in the Pipeline

HSTU-based Generative Recommenders are designed to **replace the entire DLRM stack**, handling both retrieval and ranking in a unified model. The generative formulation means the model directly outputs next-item predictions rather than scoring individual candidates.

### Production Deployment

Deployed at Meta across multiple product surfaces. The combination of HSTU + M-FALCON enables serving trillion-parameter models within the latency budget of a few hundred milliseconds per request.

---

## 3. OneTrans --- ByteDance (2025)

### Paper Details

- **Title**: "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender"
- **Authors**: Zhaoqi Zhang, Haolei Pei, Jun Guo, Tianyu Wang, Yufei Feng, Hui Sun, Shaowei Liu, Aixin Sun
- **Affiliation**: ByteDance (with Aixin Sun from Nanyang Technological University)
- **arXiv**: [2510.26104](https://arxiv.org/abs/2510.26104) (October 2025)

### Core Problem

Traditional industrial recommenders separate **user behavior sequence modeling** (what the user did in the past) from **feature interaction** (combining user, item, and context features). These are typically handled by different sub-networks with different architectures, limiting information flow between them. HSTU (Meta) addressed sequence modeling but didn't deeply integrate non-sequential features. OneTrans asks: **can a single Transformer backbone handle both tasks simultaneously?**

### Architecture: Unified Tokenization + Pyramid Blocks

**Step 1: Unified Tokenization**

OneTrans introduces a unified tokenizer that converts all input attributes into a single token sequence:
- **S-tokens** (Sequential tokens): Represent user behavior history items. These share parameters across the sequence since they come from the same semantic space.
- **NS-tokens** (Non-Sequential tokens): Represent non-sequential features (user demographics, item attributes, context features). Each NS-token gets its own **token-specific** parameters.

The unified sequence is: $[s_1, s_2, \ldots, s_T, ns_1, ns_2, \ldots, ns_K]$

**Step 2: OneTrans Block (Mixed Parameterization)**

Each OneTrans Block is a causal pre-norm Transformer block with **mixed parameterization**:
- **Mixed Causal Attention**: S-tokens share one set of Q/K/V projection weights; each NS-token has its own token-specific Q/K/V weights. Causal masking ensures S-tokens only attend to past S-tokens.
- **Mixed FFN**: Same idea --- shared FFN for S-tokens, token-specific FFN for each NS-token.
- Normalization: RMSNorm (following LLM conventions)

**Step 3: Pyramid Architecture**

Stacked OneTrans blocks use **pyramid-style tail truncation**: at each layer, the number of S-tokens is progressively reduced (truncated from the earliest/oldest tokens). This progressively distills the sequential information into fewer, higher-level representations, until the final layer has only NS-tokens remaining --- the S-token information has been fully absorbed into the NS-token representations through cross-attention.

### Cross-Request KV Caching

The critical efficiency innovation: In industrial recommenders, a single request asks the model to score many candidate items. All candidates share the same user behavior sequence (S-tokens) but differ in item features (NS-tokens).

OneTrans exploits this with **cross-request KV caching**:
1. Process all S-tokens once per request using causal attention
2. Cache the S-token key/value pairs
3. For each candidate item, compute only the NS-token representations and cross-attend against the cached S-token KVs
4. Extend across requests: reuse previous caches for incremental behavior updates

**Efficiency gains**: ~30% reduction in runtime, ~50% reduction in memory usage. Mixed precision with recomputation further improved p99 latency by ~69%.

### Results

- **Offline**: OneTrans scales near-log-linearly with increased parameters, consistently outperforming DLRM baselines, DIN, DCN, DIEN, and other models
- **Online A/B tests at ByteDance**:
  - **Feeds scenario**: +4.35% orders per user
  - **Mall scenario**: +5.68% Gross Merchandise Volume (GMV) per user
- **Production deployment**: OneTrans-L deployed on two large-scale ByteDance industrial scenarios (Feeds and Mall), maintaining production-grade latency

### Role in the Pipeline

OneTrans serves as a **ranking model** that unifies sequence modeling and feature interaction. It replaces both the sequence encoder (DIN/DIEN) and the feature interaction network (DCN/DeepFM) in the traditional DLRM pipeline.

### Comparison to HSTU

| Aspect | HSTU (Meta) | OneTrans (ByteDance) |
|--------|-------------|---------------------|
| **Focus** | Sequential transduction (generative) | Unified sequence + feature interaction |
| **Token types** | All tokens in a single stream | Explicit S-token / NS-token distinction |
| **Attention** | Pointwise SiLU (not softmax) | Standard causal attention (softmax) |
| **Feature interaction** | Via pointwise gating ($U \odot$) | Via mixed parameterization + cross-attention |
| **Inference optimization** | M-FALCON (micro-batched candidates) | Cross-request KV caching |
| **Architecture shape** | Uniform depth | Pyramid (progressive S-token truncation) |
| **Primary formulation** | Generative (next-item prediction) | Discriminative (CTR/CVR prediction) |

---

## 4. Other Notable Industrial Architectures (2023-2026)

### 4a. GEM --- Meta (2025)

- **Title**: "Meta's Generative Ads Model (GEM): The Central Brain Accelerating Ads Recommendation AI Innovation"
- **Source**: [Meta Engineering Blog, November 2025](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)
- **What it is**: Meta's largest ads foundation model, built on the Wukong architecture. GEM enhances Wukong by adding **cross-layer attention connections** to the stacked factorization machines, enabling each block to attend to representations from all prior blocks (not just the immediately preceding one).
- **Scale**: Trained across thousands of GPUs at LLM scale. Achieved **23x increase in effective training throughput** using 16x more GPUs with 1.43x improved hardware efficiency. Multi-dimensional parallelism, custom GPU kernels, and memory optimizations.
- **Knowledge Transfer**: GEM acts as a "central brain" that feeds knowledge to downstream models via:
  - **Direct transfer**: Pass knowledge to vertical models within the same data space
  - **Hierarchical transfer**: Distill from GEM into domain-specific foundation models, which then teach vertical models
- **Results**: +5% ad conversions on Instagram, +3% on Facebook Feed
- **Significance**: Demonstrates that the Wukong architecture can scale to foundation-model level and serve as a knowledge hub for an entire ads ecosystem.

### 4b. Andromeda --- Meta (2024)

- **Title**: "Meta Andromeda: Supercharging Advantage+ automation with the next-gen personalized ads retrieval engine"
- **Source**: [Meta Engineering Blog, December 2024](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/)
- **What it is**: Meta's next-generation **retrieval engine** for ads recommendation, designed for NVIDIA Grace Hopper Superchip and Meta's MTIA accelerator.
- **Key innovations**:
  - **Deep neural retrieval**: Replaces expert-engineered retrieval features with GPU-native dynamic feature reconstruction. 100x improvement in feature extraction latency and throughput vs. previous CPU-based components.
  - **Hierarchical ad index**: Jointly trained with the retrieval model so the index aligns with the neural network's understanding of relevance. Enables sub-linear inference cost.
  - **Scale**: 10,000x increase in model capacity, 3x+ jump in inference QPS
- **Results**: +6% recall at retrieval, +8% ad-quality improvement on selected segments
- **Role**: Candidate generation / retrieval stage (upstream of ranking).

### 4c. OneRec --- Kuaishou (2025)

- **Title**: "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment"
- **Authors**: Shiyao Wang et al. (Kuaishou)
- **arXiv**: [2502.18965](https://arxiv.org/abs/2502.18965) (February 2025)
- **What it is**: The first end-to-end generative recommender that **unifies retrieval and ranking** and demonstrably surpasses complex cascaded recommendation systems in production.
- **Architecture**:
  - **Encoder-decoder**: Encoder processes user's historical behavior sequence; decoder generates videos the user may be interested in
  - **Sparse MoE**: Mixture-of-Experts layers scale model capacity without proportional compute increase. Only 13% of parameters are activated during inference.
  - **Session-wise generation**: Instead of predicting one item at a time, OneRec generates an entire session of items at once, considering the relative content and ordering within the session
  - **Iterative Preference Alignment**: Uses Direct Preference Optimization (DPO) with a reward model to align generated recommendations with user preferences
- **Results**: +1.6% watch-time increase in Kuaishou's main video feed. Training MFU of 23.7%, inference MFU of 28.8% (5.2x and 2.6x higher than traditional systems).
- **Significance**: First real production validation that a single generative model can outperform an entire cascaded pipeline (retrieval + pre-ranking + ranking + re-ranking).

### 4d. HLLM --- ByteDance (2024)

- **Title**: "HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling"
- **Authors**: Junyi Chen, Lu Chi, Bingyue Peng, Zehuan Yuan (ByteDance)
- **arXiv**: [2409.12740](https://arxiv.org/abs/2409.12740) (September 2024)
- **Code**: [github.com/bytedance/HLLM](https://github.com/bytedance/HLLM)
- **What it is**: A hierarchical architecture that uses **two separate LLMs** (with non-shared parameters) for recommendation:
  - **Item LLM**: Takes an item's text description as input, compresses it into a dense embedding representation
  - **User LLM**: Takes the sequence of Item LLM embeddings for a user's history and models the user profile / preference trajectory
- **Key innovation**: Decouples item understanding from user modeling, allowing each LLM to specialize. Uses open-source pre-trained LLMs (up to 7B parameters each) and fine-tunes them for recommendation.
- **Results**: Significantly outperforms ID-based models on PixelRec and Amazon Reviews datasets. Validated with tangible benefits in real-world industrial settings at ByteDance.
- **Significance**: Demonstrates that pre-trained LLM world knowledge (from text) can complement or replace traditional ID-based embeddings, especially for cold-start items.

### 4e. MTGR --- Meituan (2025)

- **Title**: "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan"
- **arXiv**: [2505.18654](https://arxiv.org/abs/2505.18654) (May 2025)
- **Venue**: CIKM 2025
- **What it is**: Meituan's adaptation of the HSTU-based generative recommendation paradigm, with a critical modification: **it retains traditional DLRM cross-features** alongside the generative sequence model.
- **Key problem solved**: Prior generative approaches (HSTU) require abandoning carefully engineered cross-features, which significantly degrades performance. MTGR shows that scaling up the generative model cannot compensate for the loss of these features.
- **Architecture innovations**:
  - **User-item level data organization**: Allows user-item interactions to be added as candidate item features
  - **Group-Layer Normalization (GLN)**: Enhances encoding within different semantic spaces
  - **Dynamic masking**: Prevents information leakage
  - **User-level compression**: Enables efficient training and inference
- **Results**: 65x FLOPs for single-sample inference vs. the base DLRM model. Largest quality gain in nearly two years, both offline and online. Deployed on Meituan (world's largest food delivery platform) handling main traffic.

### 4f. DFGR --- Meituan (2025)

- **Title**: "Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation"
- **arXiv**: [2505.16752](https://arxiv.org/abs/2505.16752) (May 2025)
- **What it is**: A more efficient alternative to HSTU that addresses a key inefficiency: HSTU interleaves item tokens and action tokens in the sequence, doubling the effective sequence length.
- **Architecture**: DFGR duplicates the user behavior sequence into two parallel flows:
  - **Real flow**: Contains actual action information
  - **Fake flow**: Contains placeholder/masked action information
  - A novel interaction method between flows within the QKV module of self-attention
- **Results**: 2x faster training, 4x faster inference vs. HSTU. Outperforms HSTU by 0.31-1.2% AUC on both public and industrial datasets.
- **Significance**: Shows that HSTU's sequence interleaving design is suboptimal and that dual-stream approaches can be both faster and more accurate.

### 4g. PLUM --- Google/YouTube (2025)

- **Title**: "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations"
- **arXiv**: [2510.07784](https://arxiv.org/abs/2510.07784) (October 2025)
- **What it is**: Google's framework for adapting pre-trained LLMs (from the Gemini family) for large-scale recommendation through generative retrieval.
- **Three-stage pipeline**:
  1. **Semantic ID Tokenization (SID-v2)**: Uses a Residual-Quantized Variational AutoEncoder (RQ-VAE) to convert each item into a sequence of discrete tokens --- turning the item corpus into a "language" the LLM can understand
  2. **Continued Pre-training (CPT)**: Fine-tunes a Gemini-family LLM on domain-specific corpus to align world knowledge with Semantic IDs and user behavior patterns
  3. **Generative Retrieval Fine-tuning**: The model autoregressively generates the IDs of next items a user will engage with
- **Results**: +4.96% Panel CTR lift for YouTube Shorts in live A/B tests. 2.6x larger unique video coverage for Long-Form Video, 13.24x larger for Shorts. 95% impression coverage.
- **Role**: Candidate generation / retrieval (the model generates candidate IDs directly).
- **Significance**: First major demonstration that pre-trained LLMs can be effectively adapted for industrial-scale generative retrieval at YouTube scale.

---

## 5. Meta's Ads Recommendation Evolution (System-Level Context)

Meta has published a detailed account of their ads recommendation system's evolution from traditional DLRMs to sequence-based learning:

- **Source**: [Sequence learning: A paradigm shift for personalized ads recommendations](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/) (Meta Engineering Blog, November 2024)

**Two foundational transformations**:
1. **Event-based learning**: Learning representations directly from engagement and conversion events, rather than thousands of human-engineered features
2. **Sequence learning**: Developing new sequence architectures (custom Transformer) to replace traditional DLRM neural network architectures

**Production results**: 2-4% more conversions on select segments. The system must rank thousands of ads in a few hundred milliseconds per request.

This provides important context: the architectures above (HSTU, Wukong, GEM) are not academic exercises --- they are components of Meta's production ads infrastructure serving billions of users.

---

## 6. Synthesis: The Generative Recommendation Paradigm Shift

### What Changed (2023-2026)

| Dimension | Traditional DLRM (pre-2024) | Generative Recommender (2024+) |
|-----------|---------------------------|-------------------------------|
| **Formulation** | Discriminative: $P(\text{click} \mid \text{user}, \text{item})$ | Generative: $P(\text{next\_item} \mid \text{history})$ |
| **Features** | Thousands of hand-engineered features | Raw user action sequences |
| **Architecture** | Embedding tables + feature interaction (DCN, DeepFM) | Transformer variants (HSTU, OneTrans) |
| **Pipeline** | Cascaded (retrieval -> ranking -> re-ranking) | Unified (or at least fewer stages) |
| **Scaling behavior** | Plateaus at moderate compute | LLM-like scaling laws (Wukong, HSTU) |
| **Inference** | Independent scoring per candidate | KV-cached sequence processing (M-FALCON) |
| **Model size** | ~Billions of parameters (mostly embedding tables) | Up to 1.5 trillion parameters |

### Key Architectural Themes

1. **Sequence-first design**: User behavior is a sequence of (item, action, timestamp) tuples. The model should process this sequence natively, not collapse it into aggregate features.

2. **Pointwise vs. softmax attention**: HSTU's replacement of softmax with SiLU is a genuine innovation --- preserving intensity information matters for recommendation in a way that doesn't matter for NLP.

3. **Unified tokenization**: OneTrans and OneRec demonstrate that unifying sequential and non-sequential features into a single token stream, processed by a single Transformer backbone, eliminates information bottlenecks between previously separate sub-networks.

4. **Efficient inference through KV caching**: M-FALCON (Meta) and cross-request KV caching (ByteDance) are critical for making these large models servable. Without them, the models would be too slow for production.

5. **Scaling laws arrive in RecSys**: Wukong and HSTU are the first convincing demonstrations that recommendation models can exhibit the smooth, predictable quality-vs-compute scaling laws that have driven progress in NLP.

6. **Foundation model paradigm**: GEM (Meta) and PLUM (Google) treat the recommendation model as a foundation model that transfers knowledge to downstream tasks, rather than training specialized models from scratch.

7. **Preference alignment from LLMs**: OneRec's use of DPO (from the RLHF playbook) for recommendation is a direct import of LLM alignment techniques.

---

## 7. References

### Primary Papers (Requested Architectures)

1. Zhang, B. et al. "Wukong: Towards a Scaling Law for Large-Scale Recommendation." ICML 2024. [arXiv:2403.02545](https://arxiv.org/abs/2403.02545)

2. Zhai, J. et al. "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations." ICML 2024. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)

3. Zhang, Z. et al. "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender." 2025. [arXiv:2510.26104](https://arxiv.org/abs/2510.26104)

### Additional Notable Systems

4. Meta Engineering. "Meta's Generative Ads Model (GEM)." [Engineering Blog, Nov 2025](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)

5. Meta Engineering. "Meta Andromeda: Supercharging Advantage+ automation." [Engineering Blog, Dec 2024](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/)

6. Wang, S. et al. "OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment." 2025. [arXiv:2502.18965](https://arxiv.org/abs/2502.18965)

7. Chen, J. et al. "HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models." 2024. [arXiv:2409.12740](https://arxiv.org/abs/2409.12740)

8. "MTGR: Industrial-Scale Generative Recommendation Framework in Meituan." CIKM 2025. [arXiv:2505.18654](https://arxiv.org/abs/2505.18654)

9. "Action is All You Need: Dual-Flow Generative Ranking Network for Recommendation." 2025. [arXiv:2505.16752](https://arxiv.org/abs/2505.16752)

10. He et al. "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations." 2025. [arXiv:2510.07784](https://arxiv.org/abs/2510.07784)

11. Meta Engineering. "Sequence learning: A paradigm shift for personalized ads recommendations." [Engineering Blog, Nov 2024](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/)

### Surveys and Overviews

12. "Generative Recommendation: A Survey of Models, Systems, and Industrial Advances." TechRxiv, 2025. [Link](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176523089.94266134/v2)

13. "Scaling New Frontiers: Insights into Large Recommendation Models." 2024. [arXiv:2412.00714](https://arxiv.org/html/2412.00714v1)

14. "The Rise of Generative Recommenders." ML Frontiers (Substack). [Link](https://mlfrontiers.substack.com/p/the-rise-of-generative-recommenders)

15. "Is this the ChatGPT moment for recommendation systems?" Shaped Blog. [Link](https://www.shaped.ai/blog/is-this-the-chatgpt-moment-for-recommendation-systems)
