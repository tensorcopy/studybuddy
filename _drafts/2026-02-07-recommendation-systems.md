---
title: "The Complete Guide to Recommendation Systems: From Collaborative Filtering to LLM-Enhanced Pipelines"
date: 2026-02-07
categories: [machine-learning, recommendation-systems]
tags: [collaborative-filtering, matrix-factorization, deep-learning, transformers, graph-neural-networks, llm, two-tower, recommendation-systems, hstu, wukong, generative-recommendation]
mathjax: true
toc: true
description: "A deep dive into recommendation systems covering 30+ years of evolution — from collaborative filtering and matrix factorization through neural models, GNNs, and the generative recommendation era (Wukong, HSTU, OneTrans, OneRec, PLUM), with runnable code examples and practical guidance."
---

## Introduction

Recommendation systems are the quiet engines behind nearly every digital experience you interact with daily. When you open YouTube, its recommendation algorithm is responsible for over 70% of total watch time — not the search bar, not your subscriptions, but the algorithm deciding what you should see next. Netflix estimates that its recommendation system saves the company roughly $1 billion per year in customer retention by keeping subscribers engaged enough that they don't cancel. Amazon has reported that approximately 35% of its revenue comes from its recommendation engine. Spotify's Discover Weekly, TikTok's For You page, Amazon's "Customers who bought this also bought" — these systems collectively influence billions of decisions every single day.

And yet, despite their outsized impact, recommendation systems occupy a strange position in the ML landscape. They are not as glamorous as large language models or as visually striking as diffusion models. Most ML courses spend a lecture or two on collaborative filtering and move on. This is a mistake. **Recommendation systems** represent one of the most commercially important applications of machine learning, and the engineering challenges they pose — extreme sparsity, real-time latency requirements, feedback loops, cold starts, multi-objective optimization — are among the most demanding in the field.

What makes recommendation systems particularly fascinating from a technical standpoint is that they sit at the intersection of almost every major ML paradigm. Classical approaches draw on nearest-neighbor methods and linear algebra. The Netflix Prize era brought matrix factorization and regularization into the spotlight. The deep learning revolution introduced representation learning, sequence modeling, and attention mechanisms. And the most recent wave has seen graph neural networks capturing complex relational structure, reinforcement learning optimizing long-term engagement, and large language models being repurposed as general-purpose reasoning engines for recommendations. If you understand recommendation systems deeply, you understand a significant cross-section of modern machine learning.

This matters beyond pure ML practice. If you are building any product where users interact with a catalog of items — articles, products, songs, videos, job postings, potential dates — you are building a recommendation problem, whether you frame it that way or not. The difference between a naive approach (show popular items) and a well-designed recommendation system can be the difference between a product that retains users and one that doesn't. I have seen startups where improving the recommendation layer had a larger impact on core metrics than any feature change the product team shipped in a year.

The arc of recommendation systems spans from the earliest collaborative filtering ideas in the early 1990s, through the matrix factorization breakthroughs catalyzed by the Netflix Prize, into the deep learning era that reshaped the field starting around 2016, and finally to the frontier approaches of 2024-2026 where transformers, graph neural networks, and LLMs are being integrated into production recommendation pipelines. Understanding this arc means formalizing the problem, implementing key algorithms from scratch, dissecting the architectures that power real-world systems at scale, and confronting the evaluation challenges that make this field uniquely tricky.

Whether you are an ML engineer looking to build or improve a recommendation system, a researcher exploring the space, or a product builder trying to understand what these systems can and cannot do, the goal is to leave you with both theoretical grounding and practical intuition.

Let us start where it all began.

## A Brief History

The term **collaborative filtering** was coined in 1992 by David Goldberg and colleagues at Xerox PARC in their paper introducing Tapestry, a system designed to help people manage the flood of email and Usenet messages. The core idea was elegantly simple: rather than filtering messages based on their content alone, you could leverage the reactions of other people. If someone with similar interests to you found a document useful, it was probably useful to you too. Tapestry required users to manually annotate documents, which limited its scalability, but the conceptual seed was planted.

Two years later, the **GroupLens** project at the University of Minnesota (Resnick et al., 1994) took the critical next step: automating collaborative filtering. GroupLens applied CF to Usenet news articles, automatically computing similarity between users based on their rating histories and generating predictions without manual annotation. This was the birth of what we now call **user-based collaborative filtering** — find users similar to you, and recommend what they liked. Around the same time, the MIT Media Lab's Ringo system (Shardanand and Maes, 1995) applied similar ideas to music recommendation. The mid-1990s saw an explosion of interest, with systems like Amazon's BookMatcher and early movie recommendation engines appearing on the nascent web.

The scalability limitations of user-based CF became apparent quickly. As user bases grew, computing pairwise similarities between all users became prohibitively expensive, and user profiles were volatile — a user's taste today might differ from last month. In 2001, Badrul Sarwar and colleagues published their influential work on **item-based collaborative filtering**, and Amazon's Linden, Smith, and York followed in 2003 with their landmark paper describing Amazon's item-to-item collaborative filtering at scale. The key insight was that item-item similarities were far more stable than user-user similarities — the relationship between *The Matrix* and *Inception* changes slowly compared to any individual user's evolving preferences. This made precomputation feasible and dramatically improved scalability. Amazon's approach became one of the most commercially successful ML systems in history.

Then came the **Netflix Prize**. In October 2006, Netflix released a dataset of 100 million movie ratings and offered a $1 million prize to anyone who could improve their recommendation algorithm (Cinematch) by 10% in RMSE. The competition ran for nearly three years and attracted over 40,000 teams. It was a watershed moment for the field. The competition popularized **matrix factorization** techniques — particularly Simon Funk's influential SVD approach — and demonstrated that ensemble methods combining dozens of models could squeeze out incremental gains. The winning solution by BellKor's Pragmatic Chaos blended over 100 models. Perhaps more importantly, the Netflix Prize brought recommendation systems into the mainstream ML consciousness and created a shared benchmark that accelerated research.

The deep learning revolution reached recommendation systems around 2016. Google's **Wide & Deep** model (Cheng et al., 2016) combined memorization (wide linear model) with generalization (deep neural network) for app recommendations on Google Play. He et al.'s **Neural Collaborative Filtering** (NCF, 2017) replaced the dot product in matrix factorization with a neural network, learning non-linear user-item interactions. YouTube's deep recommendation system (Covington et al., 2016) revealed the two-stage retrieve-and-rank architecture that would become industry standard.

From 2018 onward, **transformer-based** sequential models like SASRec (Kang and McAuley, 2018) and BERT4Rec (Sun et al., 2019) began modeling user behavior as sequences, borrowing the self-attention mechanism to capture long-range dependencies in interaction histories. **Graph neural networks** entered the picture with PinSage (Ying et al., 2018) at Pinterest and LightGCN (He et al., 2020), modeling the user-item interaction graph directly. And most recently, from 2024 to the present, **large language models** have been integrated as feature extractors, reasoning engines, and even end-to-end recommenders, blurring the line between recommendation systems and natural language understanding.

Each of these transitions was driven by the limitations of what came before — and understanding those limitations is key to understanding the field. To do that, we need to formalize what the recommendation problem actually is.

## The Recommendation Problem

At its core, the recommendation problem can be stated simply: given a set of users $U$ and a set of items $I$, predict which items each user will prefer. In practice, this deceptively simple framing hides enormous complexity.

The foundational data structure is the **user-item interaction matrix** $R \in \mathbb{R}^{|U| \times |I|}$, where entry $r_{ui}$ represents user $u$'s interaction with item $i$. In a movie rating system, $r_{ui}$ might be a 1-5 star rating. In an e-commerce system, it might be a purchase indicator. In a music streaming service, it might be a play count.

The first critical distinction is between **explicit feedback** and **implicit feedback**. Explicit feedback consists of direct user evaluations — star ratings, thumbs up/down, reviews. It is clean and interpretable: a 5-star rating on a movie clearly signals enjoyment. But explicit feedback is scarce. Most users never rate most items. On Netflix during the Netflix Prize era, only about 1.2% of the user-item matrix was filled.

**Implicit feedback**, by contrast, consists of behavioral signals: clicks, views, purchases, time spent, scroll depth, add-to-cart actions. Implicit feedback is abundant — every user interaction generates it — but it is noisy and asymmetric. A purchase suggests interest, but the absence of a purchase does not necessarily imply disinterest. A user might not have purchased an item simply because they never saw it.

> The critical insight that underlies much of modern recommendation research is the **"missing not at random" (MNAR)** problem: the absence of an interaction between a user and an item does not mean the user dislikes the item. It usually means the user has never been exposed to it. This distinction between "dislike" and "unaware" is fundamental, and ignoring it leads to biased models that systematically undervalue items with less exposure.

In practice, implicit feedback dominates modern recommendation systems. Most platforms do not ask users to rate items explicitly (and when they do, response rates are low and biased toward extreme opinions). The challenge is to extract meaningful preference signals from noisy behavioral data.

The second key distinction is between **rating prediction** and **ranking**. Rating prediction asks: what score would user $u$ give to item $i$? The Netflix Prize was framed as a rating prediction problem, optimizing RMSE. But in practice, users do not care about predicted ratings — they care about which items appear at the top of their feed. **Ranking** asks: given a set of candidate items, what is the optimal ordering for user $u$? This is fundamentally a different optimization problem. A model that accurately predicts ratings of 3.1 vs 3.2 vs 3.3 is wasting capacity on distinctions that don't matter, while a model that reliably puts the best item in the top position is far more useful, even if its predicted scores are miscalibrated.

This shift from pointwise prediction to pairwise or listwise ranking has had profound implications for model design and evaluation. Modern systems optimize metrics like **Normalized Discounted Cumulative Gain (NDCG)**, **Mean Reciprocal Rank (MRR)**, and **Hit Rate at K (HR@K)** rather than RMSE.

Now, the **sparsity problem**. Real-world interaction matrices are extraordinarily sparse. Netflix had roughly 480,000 users and 17,770 movies during the Prize era, giving a matrix with over 8.5 billion entries — of which only about 100 million (roughly 1.2%) were observed. On Amazon, the sparsity is far worse: with millions of users and millions of products, observed interactions might represent less than 0.01% of the matrix. On a platform like Spotify with 600+ million users and 100+ million tracks, you can do the math — the interaction matrix is overwhelmingly empty.

This extreme sparsity means that most users share very few co-rated items, making similarity computation unreliable. It means that most items have very few ratings, making their representations poorly estimated. It is the fundamental challenge that drives nearly every architectural decision in recommendation systems.

Closely related is the **cold-start problem**, which manifests in two forms. **User cold start**: a new user arrives with no interaction history. What do you recommend? You have no basis for collaborative filtering, no learned user embedding. **Item cold start**: a new item is added to the catalog with no interaction data. How do you surface it to relevant users? Content-based approaches can help with item cold start (you can describe the item by its features), but user cold start often requires falling back to popularity-based recommendations or active exploration strategies.

The interplay between sparsity, cold start, and the MNAR problem defines the landscape within which all recommendation algorithms operate. The classical approaches we examine next were the first systematic attempts to navigate this landscape, and understanding their strengths and failure modes is essential context for everything that followed.

## Classical Approaches: Content-Based and Collaborative Filtering

### Content-Based Filtering

The most intuitive approach to recommendation is **content-based filtering**: recommend items that are similar to what the user has liked before, based on item features. If you enjoyed *The Matrix*, and *Inception* shares features like "sci-fi," "mind-bending plot," and "action," then the system recommends *Inception*.

In a content-based system, each item is represented as a feature vector. For text-based items (articles, books, product descriptions), a classical representation is **TF-IDF** (Term Frequency–Inverse Document Frequency). Each item becomes a sparse vector in a vocabulary-sized space, where dimensions correspond to words weighted by how distinctive they are. The similarity between items is then computed using **cosine similarity**:

$$\text{sim}(a, b) = \frac{\mathbf{v}_a \cdot \mathbf{v}_b}{\|\mathbf{v}_a\| \|\mathbf{v}_b\|}$$

This measures the cosine of the angle between two vectors, producing a value between -1 and 1 (or 0 and 1 for non-negative TF-IDF vectors). Two items with similar feature distributions will have cosine similarity close to 1.

A user's profile is then typically constructed by aggregating the feature vectors of items they have interacted with, weighted by their ratings or engagement levels. Recommendations are generated by finding items whose feature vectors are most similar to this user profile.

Content-based filtering has clear advantages: it requires no data from other users (solving the user cold-start problem partially), it is transparent and explainable ("we recommended this because it is similar to X"), and it works even for items with very few interactions. But its limitations are equally clear. It can only recommend items similar to what the user has already seen — it creates a **filter bubble** and cannot produce serendipitous discoveries. If you have only watched action movies, a content-based system will never suggest the documentary that would change your perspective. It also requires good item features, which may not be available or may not capture what actually drives user preferences. Two movies might share the same genre tags but feel completely different.

These limitations are precisely what motivated collaborative filtering: instead of asking "what is similar to what you liked?" ask "what did similar people like?"

### User-Based Collaborative Filtering

**User-based collaborative filtering** operates on a simple but powerful premise: users who agreed in the past will agree in the future. To predict what user $u$ will think of item $i$, find other users who rated items similarly to $u$, and see what they thought of $i$.

Let us walk through a concrete example. Suppose we have five users and five movies, with ratings on a 1-5 scale (0 means unrated):

| | Toy Story | Matrix | Titanic | Inception | Notebook |
|---|---|---|---|---|---|
| Alice | 5 | 4 | 1 | 4 | ? |
| Bob | 4 | 5 | 2 | 5 | 1 |
| Carol | 1 | 2 | 5 | 1 | 5 |
| Dave | 2 | 3 | 4 | 2 | 4 |
| Eve | 4 | ? | 1 | 4 | ? |

To predict Alice's rating for *The Notebook*, we first compute similarity between Alice and every other user. Using cosine similarity on their rating vectors (considering only co-rated items), we might find that Bob is most similar (both love sci-fi/action, dislike romance) while Carol and Dave are dissimilar (opposite preferences). We then take a weighted average of the similar users' ratings for *The Notebook*, weighted by their similarity to Alice:

$$\hat{r}_{u,i} = \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot r_{v,i}}{\sum_{v \in N(u)} |\text{sim}(u, v)|}$$

where $N(u)$ is the set of $u$'s nearest neighbors who have rated item $i$. Since Bob (highly similar to Alice) rated *The Notebook* a 1, while Carol (dissimilar) rated it a 5, the weighted prediction would be low — the system correctly predicts Alice would not enjoy *The Notebook*.

User-based CF was elegant and worked well in early systems, but it ran into serious problems at scale. Computing pairwise similarity between all users is $O(|U|^2)$, which becomes intractable for millions of users. More fundamentally, user profiles are non-stationary — people's tastes evolve, and the similarity computation needs to be refreshed constantly. A new rating from any user potentially changes their similarity to every other user.

### Item-Based Collaborative Filtering

**Item-based collaborative filtering** flips the perspective: instead of finding similar users, find similar items. Two items are similar if users who rated one highly tended to rate the other highly as well. This seemingly small change has profound practical implications.

The key insight, first articulated clearly by Sarwar et al. (2001) and deployed at massive scale by Amazon (Linden et al., 2003), is that item-item similarities are far more stable than user-user similarities. The relationship between *The Matrix* and *Inception* is a property of the items themselves, grounded in the aggregate behavior of all users who rated both. It changes slowly as new ratings arrive, making it feasible to precompute and cache the similarity matrix. Meanwhile, individual user profiles shift frequently, making user-user similarities expensive to maintain.

To predict user $u$'s rating for item $i$, item-based CF looks at the items similar to $i$ that $u$ has already rated, and computes a weighted average:

$$\hat{r}_{u,i} = \frac{\sum_{j \in S(i)} \text{sim}(i, j) \cdot r_{u,j}}{\sum_{j \in S(i)} |\text{sim}(i, j)|}$$

where $S(i)$ is the set of items most similar to $i$.

Here is a runnable implementation to make this concrete:

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Simple movie ratings matrix: rows=users, cols=movies
# Movies: Toy Story, Matrix, Titanic, Inception, Notebook
ratings = np.array([
    [5, 4, 1, 4, 0],  # User 0: likes sci-fi/action
    [4, 5, 2, 5, 1],  # User 1: likes sci-fi/action  
    [1, 2, 5, 1, 5],  # User 2: likes romance/drama
    [0, 3, 4, 2, 4],  # User 3: likes drama
    [4, 0, 1, 4, 0],  # User 4: likes sci-fi (sparse)
])
movies = ["Toy Story", "Matrix", "Titanic", "Inception", "Notebook"]

# Compute item-item similarity (only using non-zero ratings)
# Replace 0s with NaN for proper similarity computation
mask = ratings > 0
ratings_masked = np.where(mask, ratings, 0)

item_sim = cosine_similarity(ratings_masked.T)
np.fill_diagonal(item_sim, 0)  # Don't recommend item to itself

print("Item-Item Similarity Matrix:")
for i, movie in enumerate(movies):
    similar = sorted(zip(movies, item_sim[i]), key=lambda x: -x[1])[:2]
    print(f"  {movie:12s} → most similar: {similar[0][0]} ({similar[0][1]:.2f}), "
          f"{similar[1][0]} ({similar[1][1]:.2f})")

# Predict rating for User 4, Matrix (index 1) — currently unrated
user = 4
target_item = 1  # Matrix
user_ratings = ratings[user]

# Weighted sum of user's ratings on similar items
numerator = 0
denominator = 0
for j in range(len(movies)):
    if user_ratings[j] > 0 and j != target_item:
        numerator += item_sim[target_item][j] * user_ratings[j]
        denominator += abs(item_sim[target_item][j])

predicted = numerator / denominator if denominator > 0 else 0
print(f"\nPredicted rating for User 4 on '{movies[target_item]}': {predicted:.2f}")
print("(Above midpoint — dragged down by Titanic similarity due to naive zero-handling)")
```

Running this code, you will see that the similarity matrix captures the intuitive groupings: *Toy Story*, *Matrix*, and *Inception* cluster together (users who like one tend to like the others), while *Titanic* and *The Notebook* form their own cluster. The predicted rating for User 4 on *The Matrix* is above the midpoint — their high ratings on *Toy Story* and *Inception* pull it up, but notice how the naive zero-handling inflates *Titanic*'s similarity to *Matrix*, dragging the prediction down. This is a real limitation of simple cosine similarity on sparse matrices, and one of the reasons the field moved toward learned representations.

> The elegance of item-based CF lies in its **asymmetry of stability**: while individual humans are fickle and their tastes shift over weeks and months, the aggregate relationship between two items — shaped by thousands or millions of users — changes glacially. This asymmetry is what makes item-based CF scalable where user-based CF is not.

### Limitations That Demanded Something New

Despite their success, neighborhood-based collaborative filtering methods hit a ceiling. The sparsity problem remains brutal: if two items share very few co-raters, their similarity estimate is unreliable. These methods cannot capture **latent factors** — the hidden dimensions of taste (like "cerebral sci-fi" vs "popcorn action") that explain why users group items the way they do. They operate on raw similarity in the observed space rather than learning a compressed, generalizable representation. And they scale poorly to the most extreme data regimes: when you have hundreds of millions of users and tens of millions of items, even item-based CF with precomputed similarities strains under the weight of storage and lookup.

What was needed was a way to compress the massive, sparse interaction matrix into a dense, low-dimensional representation that could capture these latent factors directly — a representation where similar users and similar items would naturally cluster together, and where predictions could be generated by simple operations in this learned space. This is exactly what matrix factorization provides.

## Matrix Factorization: The Netflix Prize Breakthrough

The Netflix Prize, as discussed earlier, catalyzed a paradigm shift in collaborative filtering. The winning insight was not some exotic algorithm — it was a clean, elegant idea rooted in linear algebra: **matrix factorization**.

The core intuition is disarmingly simple. We have a massive user-item rating matrix $R$, where most entries are missing (Netflix's matrix was about 98.8% empty). Rather than trying to find similar users or items directly in this sparse space — the approach that classical collaborative filtering takes — we assume that both users and items can be described by a small number of hidden, or **latent**, factors. Maybe a movie is 70% "cerebral thriller," 20% "dark comedy," and 10% "visually stunning." Maybe a user has strong affinity for the first two qualities but is indifferent to the third. If we can learn these latent descriptions, predicting a rating becomes a matter of checking how well a user's taste profile aligns with a movie's attribute profile.

Mathematically, we decompose the rating matrix into two low-rank matrices:

$$R \approx P Q^T$$

Here, $P \in \mathbb{R}^{m \times k}$ is the **user latent factor matrix** — each of the $m$ users gets a $k$-dimensional vector $p_u$ describing their preferences. $Q \in \mathbb{R}^{n \times k}$ is the **item latent factor matrix** — each of the $n$ items gets a $k$-dimensional vector $q_i$ describing its characteristics. The predicted rating is the dot product $\hat{r}_{ui} = p_u^T q_i$, which measures how well user $u$'s taste aligns with item $i$'s profile in this shared latent space. The number of factors $k$ (typically 20–200) is a hyperparameter that controls the expressiveness of the model.

But raw dot products miss something important: not all users use the rating scale the same way. Some people are generous raters who give 4 stars to anything they mildly enjoy. Others reserve 5 stars for true masterpieces and rate everything else a 3. Similarly, some items are universally well-regarded (The Shawshank Redemption hovers near 4.5 on every platform), while others are polarizing. To capture these tendencies, the model that became standard in the Netflix Prize literature adds **bias terms**:

$$\hat{r}_{ui} = \mu + b_u + b_i + p_u^T q_i$$

Let me unpack every symbol. $\mu$ is the **global mean rating** across all observed ratings — a baseline that says "the average rating on this platform is, say, 3.5 stars." $b_u$ is the **user bias** — how much user $u$ deviates from the global mean (a generous rater might have $b_u = +0.4$). $b_i$ is the **item bias** — how much item $i$ deviates from the global mean (a beloved film might have $b_i = +0.8$). And $p_u^T q_i$ captures the **personalized interaction** — the part of the rating that can't be explained by global trends or individual tendencies, the part that's truly about the match between this specific user and this specific item.

> The bias decomposition is worth internalizing deeply. In practice, $\mu + b_u + b_i$ alone explains a surprising amount of variance in ratings. The latent factor interaction $p_u^T q_i$ adds personalization on top of this strong baseline. Many recommendation bugs come from ignoring biases, not from having too few latent factors.

To learn the parameters $P$, $Q$, $b_u$, and $b_i$, we minimize the **regularized squared error** over all observed ratings:

$$\min_{P, Q, b} \sum_{(u,i) \in \mathcal{K}} \left(r_{ui} - \hat{r}_{ui}\right)^2 + \lambda\left(\|p_u\|^2 + \|q_i\|^2 + b_u^2 + b_i^2\right)$$

Here, $\mathcal{K}$ is the set of all observed (user, item) pairs — we only compute loss on ratings we actually have, not on the missing entries. The first term is straightforward: minimize prediction error. The second term is **L2 regularization** (controlled by $\lambda$, typically 0.01–0.1), which prevents the model from memorizing the training data by penalizing large parameter values. Without regularization, the model would overfit catastrophically on the sparse observed entries.

The standard optimization approach is **Stochastic Gradient Descent (SGD)**. For each observed rating $(u, i, r_{ui})$, compute the prediction error $e_{ui} = r_{ui} - \hat{r}_{ui}$, then update each parameter in the direction that reduces this error:

$$p_u \leftarrow p_u + \eta(e_{ui} \cdot q_i - \lambda \cdot p_u)$$
$$q_i \leftarrow q_i + \eta(e_{ui} \cdot p_u - \lambda \cdot q_i)$$

where $\eta$ is the learning rate. The bias updates follow the same pattern. SGD is favored here because it scales linearly with the number of observed ratings, making it practical even for Netflix-scale data. An alternative is **Alternating Least Squares** (**ALS**), which fixes $P$ and solves for $Q$ (a convex problem), then fixes $Q$ and solves for $P$, alternating until convergence. ALS parallelizes better and is the default in distributed systems like Apache Spark's MLlib.

Here is a complete, runnable Python implementation of matrix factorization with bias terms, trained via SGD:

```python
import numpy as np

class MatrixFactorization:
    """Matrix Factorization with bias terms, trained via SGD."""
    
    def __init__(self, n_users, n_items, n_factors=20, lr=0.005, reg=0.02):
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.mu = 0  # Global mean
        self.lr = lr
        self.reg = reg
    
    def predict(self, u, i):
        return self.mu + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
    
    def fit(self, ratings, epochs=30):
        """ratings: list of (user, item, rating) tuples."""
        self.mu = np.mean([r for _, _, r in ratings])
        
        for epoch in range(epochs):
            np.random.shuffle(ratings)
            total_loss = 0
            for u, i, r in ratings:
                err = r - self.predict(u, i)
                total_loss += err ** 2
                
                # Update biases
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                
                # Update latent factors (save P[u] before update)
                p_u = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u - self.reg * self.Q[i])
            
            rmse = np.sqrt(total_loss / len(ratings))
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | RMSE: {rmse:.4f}")
        return self

# Create a synthetic movie ratings dataset
np.random.seed(42)
n_users, n_items = 100, 50
# Two latent clusters: action fans and romance fans
true_P = np.random.randn(n_users, 3) * 0.5
true_Q = np.random.randn(n_items, 3) * 0.5
true_ratings = 3 + true_P @ true_Q.T  # Center around 3
true_ratings = np.clip(true_ratings, 1, 5)

# Sample 20% of entries as observed ratings (simulating sparsity)
ratings = []
for u in range(n_users):
    for i in range(n_items):
        if np.random.random() < 0.2:
            r = true_ratings[u, i] + np.random.normal(0, 0.3)  # Add noise
            ratings.append((u, i, np.clip(r, 1, 5)))

print(f"Dataset: {n_users} users, {n_items} items, {len(ratings)} ratings")
print(f"Sparsity: {1 - len(ratings)/(n_users*n_items):.1%}\n")

# Split into train/test
np.random.shuffle(ratings)
split = int(0.8 * len(ratings))
train, test = ratings[:split], ratings[split:]

# Train the model
mf = MatrixFactorization(n_users, n_items, n_factors=10)
mf.fit(list(train), epochs=30)

# Evaluate on test set
test_errors = [(r - mf.predict(u, i))**2 for u, i, r in test]
print(f"\nTest RMSE: {np.sqrt(np.mean(test_errors)):.4f}")

# Show top recommendations for a user
user = 0
rated_items = {i for u, i, _ in train if u == user}
scores = [(i, mf.predict(user, i)) for i in range(n_items) if i not in rated_items]
top5 = sorted(scores, key=lambda x: -x[1])[:5]
print(f"\nTop 5 recommendations for User {user}:")
for item, score in top5:
    print(f"  Item {item:3d} — predicted rating: {score:.2f}")
```

One important extension deserves mention. The formulation above assumes explicit ratings — numerical scores that users deliberately assign. But in most real-world systems, you do not have ratings. You have **implicit feedback**: clicks, purchases, watch time, page views. A user clicking on an item is a weak positive signal; not clicking could mean disinterest or simply not seeing the item. For implicit feedback, the standard approach shifts from predicting rating values to ranking: you want items a user interacted with to score higher than items they did not. **Bayesian Personalized Ranking (BPR)**, introduced by Rendle et al. (2009), formalizes this by optimizing a pairwise ranking loss — for each user, sample an item they interacted with (positive) and one they did not (negative), and push the positive item's score above the negative item's score via a sigmoid-based objective. BPR can be applied on top of MF and most other recommendation architectures.

> Do not dismiss matrix factorization as outdated just because newer methods exist. In a rigorous 2020 study, Rendle et al. showed that carefully tuned MF with dot-product interaction matches or outperforms Neural Collaborative Filtering on standard benchmarks. The lesson: don't assume deep equals better. A well-regularized linear model with the right inductive biases is a formidable baseline.

Matrix factorization gave us the language of latent factor models that still underpins modern recommendations. But it has clear limitations: the interaction between user and item latent vectors is a simple dot product, which assumes a linear relationship. It also cannot easily incorporate rich side information — user demographics, item descriptions, contextual signals like time of day or device type. These limitations set the stage for the next generation of models, which replaced the dot product with neural networks.


## The Deep Learning Revolution

By the mid-2010s, deep learning was rewriting the rules in computer vision and NLP. It was only a matter of time before the recommendation community asked the obvious question: can we replace matrix factorization's linear dot product with a neural network that learns arbitrarily complex user-item interactions? The answer spawned a family of models that, while not always superior to well-tuned MF in controlled experiments, fundamentally expanded what recommendation systems could represent.

### Neural Collaborative Filtering

**Neural Collaborative Filtering (NCF)**, proposed by He et al. (2017), is the canonical paper in this line of work. The core argument is that the dot product in MF is a fixed, linear function — it cannot capture complex, non-linear interactions between user and item latent factors. NCF replaces it with a learnable neural architecture.

The model takes a user ID and an item ID as input, maps each to a dense embedding vector (analogous to the latent factors in MF), and then processes these embeddings through two parallel pathways:

1. **Generalized Matrix Factorization (GMF)**: Computes the element-wise product of the user and item embeddings, then passes the result through a linear layer. This is a strict generalization of MF — if you set the output weights to all ones, you recover the dot product exactly. But because the weights are learned, GMF can assign different importance to different latent dimensions.

2. **Multi-Layer Perceptron (MLP)**: Concatenates the user and item embeddings and feeds them through a stack of fully connected layers with nonlinear activations (typically ReLU). This pathway can learn arbitrary, non-linear interaction patterns between user and item features — something a dot product cannot do.

The final model, **NeuMF**, combines both pathways by concatenating their outputs and passing the combined vector through a final prediction layer. The intuition is that GMF handles the structured, linear interactions that MF already captures well, while the MLP pathway discovers additional non-linear patterns. The two pathways use separate embeddings, so they can learn different representations suited to their respective functions.

In practice, NCF showed improvements over vanilla MF on several benchmarks, though later work (including the Rendle et al. 2020 study mentioned above) questioned whether these gains survive rigorous hyperparameter tuning of the MF baseline. My intuition is that NCF's real contribution was not raw accuracy — it was opening the door to thinking about recommendations as a representation learning problem where architecture choices matter.

### Wide & Deep Learning

While NCF focused on pure collaborative filtering, **Wide & Deep Learning** (Cheng et al., 2016) tackled a different problem that dominates industrial recommendation: how do you combine the strengths of memorization and generalization?

The **wide component** is a linear model that operates on hand-crafted **feature crosses** — for example, "user has installed app X AND the impression app is in the same category." These crosses memorize specific, high-value co-occurrences from training data. If users who installed Netflix also frequently install Hulu, the wide component captures this directly. Memorization is powerful but brittle: it only works for patterns explicitly encoded in the feature crosses, and it cannot generalize to unseen combinations.

The **deep component** is a standard feed-forward neural network that takes dense embeddings of categorical features (user demographics, app attributes, contextual signals) and learns to generalize across them. A deep network can discover that users who like streaming apps might also like media players, even if this specific combination was never seen in training. But deep networks can over-generalize — recommending vaguely related items that a user would never actually want.

Wide & Deep combines both components by feeding their outputs into a joint prediction layer, and it was deployed in Google Play's app recommendation system. The model works well in practice, but it has a significant weakness: the wide component requires careful, manual feature engineering. Someone has to decide which feature crosses to include, and getting this wrong leaves value on the table.

### DeepFM: Automating Feature Interactions

**DeepFM** (Guo et al., 2017) elegantly solves Wide & Deep's manual engineering problem by replacing the wide component with a **Factorization Machine (FM)**. An FM models all pairwise feature interactions through latent vectors — the interaction between features $i$ and $j$ is computed as the dot product of their respective latent vectors, $\langle v_i, v_j \rangle$. This captures the same low-order interactions that the wide component's feature crosses capture, but automatically and without manual engineering.

The key architectural decision in DeepFM is that the FM component and the deep component **share the same embedding layer**. This means the raw features are embedded once, and both pathways operate on the same learned representations. Shared embeddings have two benefits: they reduce the total number of parameters (important for models with millions of categorical features), and they allow the FM and deep components to jointly influence the feature representations during training.

The result is a model that automatically captures both low-order feature interactions (through the FM, analogous to the wide component) and high-order feature interactions (through the deep network), without any manual feature engineering. DeepFM consistently outperformed Wide & Deep in experiments and became a staple architecture in click-through rate prediction.

> The deep learning revolution in recommendations was less about raw predictive accuracy and more about representational flexibility. Classical MF operates on user and item IDs alone. Neural models can seamlessly incorporate heterogeneous side information — user demographics, item text descriptions, temporal signals, device type, session context — through shared embedding layers. This ability to fuse diverse feature types into a unified prediction framework is what makes deep models indispensable in production systems, even when their collaborative filtering component alone might not beat well-tuned MF.

These feed-forward architectures treat each user-item interaction as an independent prediction. But real user behavior is sequential — a person who just watched three horror movies in a row is probably in the mood for another, regardless of their long-term preference profile. Capturing this sequential structure requires a different class of models.


## Sequential and Transformer-Based Models

Classical collaborative filtering and even the neural models discussed above share a fundamental assumption: a user's preference is a static thing. You have a user embedding, an item embedding, you compute a score. But anyone who has used a streaming service knows this is wrong. Your taste on a Friday night after a long week is different from your taste on a Sunday morning. More importantly, your preferences *evolve*: the action movies you loved at 20 might give way to documentaries at 35. And within a single session, there is momentum — watching one film noir often leads to wanting another.

This observation led to a paradigm shift: modeling user behavior as a **sequence of interactions** rather than a static profile. The goal becomes **next-item prediction** — given the ordered history of items a user has interacted with, predict what they will engage with next.

### SASRec: Self-Attention for Sequential Recommendation

Early sequential models used RNNs and GRUs to process interaction histories, but they suffered from the usual RNN limitations: difficulty with long-range dependencies and sequential (non-parallelizable) computation. **SASRec** (Kang & McAuley, 2018) brought the **self-attention mechanism** — already transforming NLP via the Transformer architecture — to sequential recommendation.

SASRec treats a user's interaction history as a sequence of item embeddings and applies **unidirectional (left-to-right) self-attention**, exactly like GPT in language modeling. At each position in the sequence, the model attends only to previous items (using a causal mask) and produces a representation that encodes the user's evolving preference up to that point. The prediction for the next item is made from the final position's representation.

Why is self-attention natural for this task? Because not all past interactions are equally relevant to the next one. If you watched The Godfather, then a cooking show, then Goodfellas, the attention mechanism can learn to assign high weight to The Godfather and Goodfellas (both mob films) and low weight to the cooking show when predicting your next movie. Traditional RNNs would struggle with this because the cooking show sits between the two relevant items, creating a bottleneck in the sequential hidden state. Self-attention bypasses this entirely — every past item can directly influence the current prediction, with learned attention weights determining relevance.

SASRec also adds **positional embeddings** (learned, not sinusoidal) so the model knows the ordering of interactions, and it uses a relatively shallow architecture — typically 2 self-attention blocks — because user interaction sequences are much shorter than text documents.

### BERT4Rec: Bidirectional Context for Recommendations

**BERT4Rec** (Sun et al., 2019) took the next logical step: if SASRec is GPT for recommendations, why not try BERT? Instead of unidirectional attention that only looks at past items, BERT4Rec uses **bidirectional self-attention** that can attend to items on both sides of any position.

The training objective mirrors BERT's **masked language model**: randomly mask some items in the interaction sequence (the **Cloze task**), and train the model to predict the masked items from the surrounding context. At inference time, you append a mask token to the end of the sequence and predict the next item using bidirectional context over the entire history.

The argument for bidirectionality is that context from future items (in the training sequence) can help disambiguate past items. If a user watched item A, then item B, then item C, a unidirectional model predicting B can only use A. A bidirectional model can use both A and C, potentially learning that A and C together indicate a specific interest that makes B's role clearer. BERT4Rec showed improvements over SASRec on several benchmarks, particularly for users with longer interaction histories where bidirectional context provides more signal.

> Sequential models reveal something fundamental about recommendation: user preference is not a noun, it is a verb. It is a process that unfolds over time, shaped by context, mood, and momentum. Any model that treats users as static vectors is throwing away information — and the attention mechanism gives us a principled, interpretable way to capture this dynamic nature.

I suspect the sequential paradigm still has significant room to grow, particularly as interaction sequences get longer and richer with multi-modal signals (text queries, image clicks, dwell times). But there is another structural insight that sequential models do not exploit: the global connectivity pattern between all users and all items, which naturally forms a graph.


## Graph Neural Networks for Recommendation

If you step back and look at the totality of user-item interactions in a recommendation system, a natural structure emerges: a **bipartite graph** where users and items are nodes, and edges represent interactions (ratings, clicks, purchases). This graph encodes collaborative signals that extend well beyond direct neighbors. If user A and user B both liked items X, Y, and Z, and user B also liked item W, then the graph structure implicitly suggests item W for user A — not through first-order similarity, but through **multi-hop reasoning** on the graph.

**Graph Neural Networks (GNNs)** formalize this intuition by learning node representations through iterative **message passing**: each node aggregates information from its neighbors, then its neighbors' neighbors, and so on. After $L$ layers of propagation, each node's embedding captures collaborative signals from its $L$-hop neighborhood in the interaction graph.

### From NGCF to LightGCN

**Neural Graph Collaborative Filtering (NGCF)**, proposed by Wang et al. (2019), was among the first to apply GNNs to collaborative filtering in a principled way. NGCF initializes user and item embeddings, then refines them by propagating information along the edges of the user-item interaction graph. Each propagation layer applies a learned feature transformation (a weight matrix) and a nonlinear activation function — standard GNN design borrowed from models like GCN and GraphSAGE.

NGCF demonstrated clear improvements over MF, validating the idea that explicitly encoding multi-hop collaborative signals into embeddings is valuable. But then something surprising happened.

**LightGCN** (He et al., 2020) stripped away nearly everything from NGCF — the feature transformation matrices, the nonlinear activation functions, the self-connection — and kept only the simplest possible operation: **normalized sum aggregation**. The propagation rule for a user node $u$ at layer $l+1$ is:

$$e_u^{(l+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}} e_i^{(l)}$$

Here, $e_u^{(l+1)}$ is the embedding of user $u$ at layer $l+1$, $\mathcal{N}_u$ is the set of items that user $u$ has interacted with, $|\mathcal{N}_u|$ is the degree of user $u$ (how many items they interacted with), and $|\mathcal{N}_i|$ is the degree of item $i$ (how many users interacted with it). The normalization factor $\frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}}$ ensures that users and items with many connections do not dominate the aggregation — it is the symmetric normalization from spectral graph theory. The same equation applies symmetrically for item nodes aggregating from their user neighbors.

The final embedding for each node is a **weighted sum across all layers**:

$$e_u = \sum_{l=0}^{L} \alpha_l \, e_u^{(l)}$$

where $\alpha_l$ can be uniform ($1/(L+1)$) or learned. This layer combination is crucial — it ensures the final representation contains information from different neighborhood scopes: the 0th-layer embedding captures the node itself, the 1st-layer captures direct neighbors, the 2nd-layer captures two-hop neighbors, and so on.

The counterintuitive finding was that LightGCN, despite being dramatically simpler than NGCF, consistently outperformed it across benchmarks. The authors argued that feature transformations and nonlinear activations — standard in GNNs for tasks with rich node features — actually hurt performance in collaborative filtering, where the only input features are learnable ID embeddings. There is no meaningful "feature transformation" to apply to a randomly initialized embedding vector. The value of the GNN here is purely in the propagation structure — spreading collaborative signals through the graph — not in the nonlinear transformations.

> Simplicity as a feature, not a bug: LightGCN's success is a powerful reminder that architectural complexity must be justified by the problem structure. In collaborative filtering, the signal is in the graph topology, not in nonlinear feature transformations. Adding complexity that does not align with the data's inductive biases does not just waste compute — it can actively hurt performance.

There is an important caveat with GNN-based recommendation models: **over-smoothing**. As you stack more propagation layers, each node's embedding becomes an average over an exponentially growing neighborhood. After too many hops (typically more than 3-4 layers), all node embeddings start to converge toward a common vector, destroying the discriminative information needed for personalized recommendation. This is not merely a theoretical concern — it is the primary reason most GNN recommendation models use only 2-3 layers in practice. Techniques like residual connections, layer dropout, and the multi-scale layer combination in LightGCN help mitigate this, but over-smoothing remains an active research problem.

These algorithmic advances — from matrix factorization through deep models, sequential architectures, and graph neural networks — each capture different aspects of the recommendation problem. But deploying any of them in a real production system requires something that academic papers rarely discuss in depth: the engineering pipeline that connects raw data to served recommendations at scale.

## The Industrial Pipeline

Everything we have discussed so far — collaborative filtering, matrix factorization, deep interaction models, sequential transformers, graph neural networks — operates in a somewhat idealized setting. You have a user, you have items, you score them, you rank them. But what happens when "items" means a billion videos, "users" means two billion accounts, and "score them" needs to happen in under 200 milliseconds while a human waits for a page to load?

The answer, refined through hard-won engineering at YouTube, Netflix, Meta, LinkedIn, and others, is a **multi-stage funnel architecture**. No single model can simultaneously be fast enough to scan billions of candidates and expressive enough to capture nuanced user preferences. Instead, production systems decompose the problem into stages, each trading off differently between coverage and precision.

### Stage 1: Candidate Generation

The first stage has one job: reduce billions of items to a few hundred plausible candidates, and do it *fast*. The workhorse architecture here is the **two-tower model** (also called a dual encoder).

The idea is elegant in its simplicity. You build two independent neural networks — one for users, one for items — that map their respective inputs into a shared embedding space:

- **User tower**: $e_u = f_\theta(x_u)$, where $x_u$ includes user demographics, watch history, and other features
- **Item tower**: $e_i = g_\phi(x_i)$, where $x_i$ includes item metadata, content features, and engagement statistics

The relevance score is just a dot product:

$$s(u, i) = e_u^T e_i$$

Training uses **in-batch negatives** with a softmax loss and temperature scaling:

$$\mathcal{L} = -\log \frac{\exp(s(u, i^+) / \tau)}{\sum_{j \in \text{batch}} \exp(s(u, i_j) / \tau)}$$

Here $i^+$ is the positive (interacted) item, the sum runs over all items in the batch (treating other users' positives as negatives), and $\tau$ is a temperature parameter that controls how peaked the distribution is. Lower temperature makes the model focus harder on distinguishing the true positive from close negatives.

Now here is the key insight that makes this architecture production-viable: **the user and item towers never interact during encoding**. The score is just a dot product between two independently computed vectors. This seems like a limitation — and in terms of expressiveness, it is. A cross-attention model that jointly attends over user and item features would capture richer interactions. But the late interaction is the *feature*, not a bug, because it enables a critical serving optimization.

Since item embeddings $e_i$ do not depend on the user, you can **pre-compute all item embeddings offline** and index them in an **approximate nearest neighbor (ANN)** data structure. Libraries like **FAISS** (Facebook AI Similarity Search) and **ScaNN** (Google) can search over billion-scale vector indices in single-digit milliseconds. At serving time, you only need to run the user tower once, get $e_u$, and then perform an ANN lookup to retrieve the top few hundred nearest item embeddings. This takes the retrieval problem from $O(N)$ brute-force scoring down to something closer to $O(\log N)$.

> The two-tower architecture's apparent weakness — that user and item representations cannot attend to each other — is precisely what makes billion-scale retrieval possible. The expressiveness you sacrifice is recovered in the next stage.

In practice, systems often run **multiple candidate generators in parallel**: one for collaborative filtering embeddings, one for content-based embeddings, one for trending/popular items, and perhaps a knowledge-graph-based retriever. Their outputs are merged before passing downstream.

### Stage 2: Ranking

With the candidate set reduced from billions to hundreds, the **ranking stage** can afford to be far more expressive. This is where the feature-rich models we discussed earlier — Wide & Deep, DeepFM, or gradient-boosted decision trees like XGBoost/LightGBM — earn their keep.

At this stage, the model can incorporate hundreds of features that would be prohibitively expensive to compute for every item in the catalog:

- **User features**: demographics, subscription tier, historical engagement rates, time-of-day patterns
- **Item features**: content metadata, creator information, freshness, quality scores
- **Cross features**: does this user tend to watch videos from this category? Has the user seen similar items recently?
- **Contextual features**: device type, time of day, day of week, geographic location

The ranking model scores each candidate with a rich feature vector and produces a refined ordering. Because it only needs to score a few hundred items, it can spend 1-10 milliseconds per item — a luxury that the candidate generation stage could never afford.

Many production systems actually split ranking into a **pre-ranking** (or "light ranking") stage that scores thousands of candidates with a simpler model, filtering down to hundreds for the **heavy ranker**. This creates a three-tier funnel within what I have described as two tiers, illustrating how real systems often have more nuance than their textbook descriptions suggest.

### Stage 3: Re-ranking

The final stage is often overlooked in academic treatments, but practitioners will tell you it is where some of the most consequential decisions are made. **Re-ranking** applies business logic, policy constraints, and user experience objectives on top of the raw relevance scores.

Consider what happens without re-ranking: if a user loves action movies, the ranker will dutifully return ten action movies. That is technically accurate but a terrible user experience. Re-ranking enforces **diversity** — spreading recommendations across genres, creators, and content types. It boosts **freshness**, ensuring that new content gets exposure rather than the catalog calcifying around proven hits. It enforces **fairness**, preventing the system from systematically under-recommending content from certain creators. And it applies **content policy**, filtering out items that may be technically relevant but violate platform guidelines.

This is not an afterthought. At companies like Netflix and YouTube, the re-ranking layer is where critical business objectives — subscriber retention, creator ecosystem health, regulatory compliance — get translated into concrete algorithmic adjustments.

The seminal paper that crystallized this pipeline thinking for the deep learning era was Covington et al.'s 2016 work, "Deep Neural Networks for YouTube Recommendations." Their two-stage system (candidate generation with a deep neural network, ranking with a separate deep model) became the template that most large-scale systems still follow today, with the re-ranking stage growing more sophisticated over time.

With the industrial pipeline architecture established, we can examine the most significant recent development: the emergence of architectures that exhibit scaling laws comparable to LLMs — and in some cases, collapse the multi-stage pipeline into a single model.

## The Generative Recommendation Era

Between 2024 and early 2026, a paradigm shift swept through industrial recommendation systems. The core insight: recommendation can be reformulated from a discriminative problem (score each user-item pair) to a **generative** one (given a user's history, generate the next items they will engage with). This shift, combined with the discovery that recommendation models can exhibit **scaling laws** similar to LLMs, has led to architectures that are fundamentally more capable than anything that came before.

### Wukong: Scaling Laws Arrive in RecSys

**Wukong** (Zhang et al., Meta, ICML 2024) asked a question that had not been rigorously tested before: do recommendation models exhibit scaling laws? In NLP, we know that increasing model size predictably improves performance — the insight that launched the LLM era. Wukong demonstrated that the same principle holds for recommendation, *if* you use the right architecture.

The architecture is built on **stacked factorization machines** — a network of alternating Factorization Machine Blocks (FMB) and Linear Compression Blocks (LCB). Each FMB captures pairwise feature interactions and processes them through an MLP. Each LCB linearly compresses the representation. The critical property is that stacking these layers causes the interaction order to grow **exponentially**: layer 1 captures 2nd-order interactions, layer 2 captures up to 4th-order, layer $L$ captures up to $2^L$-th order. This "binary exponentiation" means Wukong models extremely high-order feature interactions with relatively few layers — something that Deep & Cross Network (which grows interaction order *linearly* with depth) cannot match.

The key result: Wukong exhibits a **smooth, monotonic quality improvement across two orders of magnitude** in model complexity, extending beyond 100 GFLOP/example. Prior architectures plateau or degrade well before this point. This is not a marginal finding — it means that for the first time, recommendation teams can predictably trade compute for quality, just as NLP teams do with LLMs.

> The discovery of scaling laws in recommendation models is arguably as consequential for the field as the discovery of scaling laws was for NLP. It means we are no longer architecture-hunting — we are scaling, and the returns are predictable.

Wukong operates as a **ranking model**, replacing the feature interaction component (DCN, DeepFM) in the traditional DLRM stack. Its architecture was subsequently adopted in Meta's **GEM** (Generative Ads Model), a production foundation model that powers ads recommendation across Facebook and Instagram, achieving +5% ad conversions on Instagram and +3% on Facebook Feed.

### HSTU: Trillion-Parameter Generative Recommenders

If Wukong brought scaling laws to recommendation ranking, **HSTU** (Hierarchical Sequential Transduction Unit, Zhai et al., Meta, ICML 2024) brought them to the entire recommendation pipeline. HSTU reformulates recommendation as **sequential transduction**: given a user's history of actions, generate the next actions. This is the "Generative Recommender" (GR) paradigm — essentially treating recommendation as a language modeling problem over user behavior sequences.

HSTU is a modified Transformer, but the modifications are not cosmetic. Recommendation data differs from text in ways that demand architectural changes:

- **Intensity matters**: A user who interacted with 100 cooking videos has stronger signal than one who interacted with 5, even if the relative distribution is similar. Standard softmax attention normalizes away this intensity. HSTU replaces softmax with **pointwise SiLU activation**, preserving absolute magnitude information.
- **Heterogeneous tokens**: Each "token" in a user's history is a (item, action_type, timestamp) tuple, not just an item ID. HSTU processes these with relative positional and temporal attention biases.
- **Gated feature interaction**: Each HSTU layer applies a Hadamard product gating mechanism ($U(X) \odot \text{Attention}(X)$) that serves as the feature interaction component — replacing the separate feature interaction networks in traditional DLRMs.

The inference bottleneck is solved by **M-FALCON** (Microbatched-Fast Attention Leveraging Cacheable Operations): the user's history is processed once and its K/V representations are cached, while candidates are scored in micro-batches. The result: a 285x more complex model can be served at 3x higher QPS than a traditional DLRM.

The production results are striking: a **1.5 trillion parameter** HSTU model deployed across multiple Meta surfaces achieves a **12.4% metric improvement** in online A/B tests, with up to 65.8% improvement in NDCG over baselines.

### OneTrans: Unified Tokenization at ByteDance

While HSTU focused on sequential transduction, **OneTrans** (Zhang et al., ByteDance, 2025) tackled a different unification problem. Traditional industrial recommenders separate user behavior sequence modeling (what the user did) from feature interaction (combining user, item, and context features). These are typically handled by different sub-networks — DIN/DIEN for sequences, DCN/DeepFM for features — limiting information flow between them.

OneTrans introduces a **unified tokenizer** that converts all input attributes into a single token sequence:
- **S-tokens** (Sequential): Represent user behavior history items, sharing parameters since they come from the same semantic space
- **NS-tokens** (Non-Sequential): Represent non-sequential features (user demographics, item attributes, context), each with its own token-specific parameters

These are processed by a causal Transformer with **mixed parameterization** — shared Q/K/V weights for S-tokens but token-specific weights for each NS-token — and a **pyramid architecture** that progressively truncates older S-tokens at each layer, distilling sequential information into the NS-token representations through cross-attention.

The efficiency innovation is **cross-request KV caching**: all candidates in a request share the same user behavior sequence, so S-token K/V pairs are computed once and reused across candidates (~30% runtime reduction, ~50% memory reduction). Online A/B tests at ByteDance showed +4.35% orders per user in Feeds and +5.68% GMV per user in Mall scenarios.

### OneRec: The End of the Pipeline?

Perhaps the most provocative result comes from **OneRec** (Wang et al., Kuaishou, 2025) — the first production system where a single generative model demonstrably **outperforms an entire cascaded pipeline** (retrieval + pre-ranking + ranking + re-ranking).

OneRec uses an encoder-decoder architecture with **sparse Mixture-of-Experts** layers (only 13% of parameters activated during inference) and a novel **session-wise generation** approach: instead of predicting one item at a time, it generates an entire session of recommendations at once, considering the relative content and ordering within the session. It then applies **Direct Preference Optimization (DPO)** — borrowed directly from the LLM alignment playbook — to align generated recommendations with user preferences.

The +1.6% watch-time increase on Kuaishou's main video feed may sound modest, but at Kuaishou's scale (300+ million daily active users), this represents enormous value. More importantly, it validates the thesis that the multi-stage pipeline — which we have treated as an engineering necessity — may ultimately be an artifact of insufficient model capacity.

> OneRec's result challenges a fundamental assumption of the field: that the multi-stage pipeline is architecturally necessary. If a single generative model can outperform a carefully optimized cascade of specialized models, the engineering complexity of maintaining separate retrieval, ranking, and re-ranking systems may become a liability rather than an advantage.

I should note that this remains early evidence. OneRec operates on a single product surface, and whether the approach generalizes across all recommendation domains remains an open question. But the directional signal is clear.

### The Paradigm Shift in Summary

| Dimension | Traditional DLRM | Generative Recommender |
|-----------|------------------|----------------------|
| **Formulation** | Discriminative: $P(\text{click} \mid \text{user}, \text{item})$ | Generative: $P(\text{next\_item} \mid \text{history})$ |
| **Features** | Thousands of hand-engineered features | Raw user action sequences |
| **Architecture** | Embedding tables + feature interaction (DCN, DeepFM) | Transformer variants (HSTU, OneTrans) |
| **Pipeline** | Cascaded (retrieval → ranking → re-ranking) | Unified (or fewer stages) |
| **Scaling** | Plateaus at moderate compute | LLM-like scaling laws |
| **Inference** | Independent scoring per candidate | KV-cached sequence processing |
| **Model size** | ~Billions (mostly embedding tables) | Up to 1.5 trillion parameters |

## The LLM Wave

The generative recommendation architectures described above — HSTU, Wukong, OneTrans — are Transformer-based but trained from scratch on behavioral data. A parallel and equally important development is the integration of **pre-trained large language models** into recommendation systems, bringing world knowledge, natural language understanding, and zero-shot reasoning capabilities that behavioral models cannot match.

### LLMs as Feature Extractors and Metadata Generators

The most immediately practical use of LLMs in recommendation systems is not glamorous: generating better features for existing models. **Bing** used GPT-4 to produce improved webpage summaries that fed into their search ranking pipeline, then distilled this capability into a smaller Mistral-7B model for cost-effective serving. **Indeed** used LLM-generated labels for job-match quality, essentially replacing expensive human annotation with LLM judgments that could be produced at scale.

This pattern — use a large model to generate enriched metadata or pseudo-labels, then train a smaller, specialized model on those outputs — turns out to be remarkably effective. It sidesteps the latency and cost problems of serving LLMs directly while still capturing their language understanding capabilities.

### HLLM: Hierarchical LLMs for Recommendation

ByteDance's **HLLM** (Chen et al., 2024) pushes the LLM-as-feature-extractor pattern further with a hierarchical architecture using **two separate LLMs**: an **Item LLM** that compresses an item's text description into a dense embedding, and a **User LLM** that processes the sequence of Item LLM embeddings to model the user's preference trajectory. By decoupling item understanding from user modeling, each LLM specializes at its task. HLLM demonstrates that pre-trained LLM world knowledge (from text) can complement or replace traditional ID-based embeddings, particularly for cold-start items.

### PLUM: Generative Retrieval at YouTube Scale

Google's **PLUM** (2025) represents perhaps the most ambitious attempt to use pre-trained LLMs for recommendation at scale. PLUM adapts a **Gemini-family** LLM for generative retrieval through three stages:

1. **Semantic ID Tokenization**: A Residual-Quantized Variational AutoEncoder (RQ-VAE) converts each item into a sequence of discrete tokens — turning the item corpus into a "language" the LLM can understand
2. **Continued Pre-training**: Fine-tunes the Gemini model on domain-specific data to align world knowledge with Semantic IDs and user behavior patterns
3. **Generative Retrieval**: The model autoregressively generates the token IDs of next items a user will engage with

The production results at YouTube are compelling: +4.96% Panel CTR lift for YouTube Shorts, 2.6x larger unique video coverage for Long-Form Video, and 13.24x larger coverage for Shorts. PLUM is the first major demonstration that pre-trained LLMs can be effectively adapted for industrial-scale generative retrieval.

### Unified LLM-Based Rankers

The more ambitious application is using LLMs as the ranking model itself. **LinkedIn's 360Brew** is perhaps the most striking example: they replaced over 30 separate ranking tasks (feed ranking, job recommendations, ad ranking, notifications) with a single 150-billion-parameter Mixtral-based model. This dramatically simplified their ML infrastructure and improved performance across the board.

**Netflix's UniCoRn** (Unified Contextual Recommendation) similarly unified search and recommendation into a single model, eliminating the artificial boundary between "the user typed a query" and "the user is browsing."

### Conversational Recommendation

LLMs also open up an entirely new interaction paradigm: **conversational recommendation**, where users describe what they want in natural language rather than relying on implicit behavioral signals. "I'm in the mood for something like Fleabag but darker" is a query that traditional recommender systems cannot handle but that an LLM-augmented system can reason about.

### Where Things Stand

The picture that emerges in early 2026 is more nuanced than either "LLMs will replace everything" or "LLMs are just hype for recommendations." The evidence suggests a clear segmentation by pipeline stage:

- **Retrieval**: PLUM shows pre-trained LLMs can work, but purpose-built behavioral models (HSTU, two-tower + ANN) remain dominant for latency-critical serving at billion-item scale
- **Ranking**: Wukong, HSTU, and OneTrans — Transformer architectures trained from scratch on behavioral data — are winning here, with LLM-based rankers (360Brew) viable for platforms that can afford the compute
- **Feature enrichment and cold-start**: This is where pre-trained LLMs add the most unambiguous value, through HLLM-style hierarchical encoding or metadata generation
- **End-to-end**: OneRec provides the first production evidence that a single generative model can outperform cascaded pipelines, but this remains early-stage

> The industrial recommendation landscape of 2026 is defined by a convergence: purpose-built generative architectures (HSTU, Wukong, OneTrans) that exhibit LLM-like scaling laws, and pre-trained LLMs (PLUM, HLLM, 360Brew) that bring world knowledge to the pipeline. The most sophisticated systems combine both.

With all these modeling advances, a natural question arises: how do we know if any of this is actually working? That brings us to evaluation — a topic with more depth and subtlety than most practitioners appreciate.

## Evaluation: Beyond Accuracy

Evaluating recommendation systems is deceptively difficult. The standard metrics are well-established, but they tell only part of the story, and optimizing for them in isolation can actively harm user experience.

### The Standard Metrics

Since recommendation is fundamentally a ranking problem, the standard metrics focus on the quality of the top-K items presented to the user.

**Precision@K** measures the fraction of recommended items in the top-K that are relevant:

$$\text{Precision@K} = \frac{|\text{relevant items in top-K}|}{K}$$

**Recall@K** measures the fraction of all relevant items that appear in the top-K recommendations:

$$\text{Recall@K} = \frac{|\text{relevant items in top-K}|}{|\text{all relevant items}|}$$

Precision and Recall are crude in that they treat all positions in the top-K equally. A relevant item at position 1 is just as valuable as one at position K. **Normalized Discounted Cumulative Gain (NDCG@K)** addresses this by applying a logarithmic discount based on position, rewarding systems that place relevant items earlier:

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

Here $rel_i$ is the relevance of the item at position $i$, and $\text{IDCG@K}$ is the DCG of the ideal ranking (all relevant items sorted by decreasing relevance at the top). An NDCG of 1.0 means the system produced the perfect ranking; lower values indicate relevant items appearing too late.

Two other common metrics are **Hit Rate@K** (did *at least one* relevant item appear in the top-K? — essentially a binary version of Recall) and **Mean Reciprocal Rank (MRR)**, which measures the reciprocal of the position of the first relevant item, averaged across users. MRR heavily rewards getting the *very first* recommendation right.

### Beyond Accuracy

Here is the uncomfortable truth: a system that maximizes NDCG@10 can be a terrible recommendation system.

> Optimizing purely for accuracy metrics leads to recommendation systems that exploit user biases rather than serving user interests — the algorithmic equivalent of a restaurant that only serves dessert because that is what gets the highest satisfaction ratings per dish.

**Popularity bias** is the most pervasive problem. Accuracy metrics are computed against held-out interactions, which skew heavily toward popular items. A model that recommends the most popular items to everyone can achieve surprisingly strong Precision and NDCG scores while providing zero personalization value.

**Diversity** measures how different the recommended items are from each other. A system recommending five nearly identical items may score well on relevance but poorly on user satisfaction. **Novelty** captures whether the system surfaces items the user would not have discovered on their own. **Serendipity** goes further: did the recommendation *surprise* the user in a positive way?

**Fairness** is increasingly critical. Does the system systematically under-recommend content from certain creators or about certain topics? Does it create **filter bubbles** where users are trapped in increasingly narrow content silos?

Perhaps the most sobering challenge is the **offline-online evaluation gap**. Offline metrics, computed on historical data, do not always correlate with online business metrics like engagement, retention, or revenue. A model that wins on offline NDCG might lose an A/B test. This happens because offline evaluation cannot capture feedback loops, presentation bias, or the novelty effect of showing users something they have never seen before. The gold standard remains online A/B testing, but it is expensive and slow.

With a clear understanding of both the models and how to evaluate them, let us translate this knowledge into practical guidance for anyone building a recommendation system today.

## Practical Takeaways

After covering the full landscape — from neighborhood methods through graph neural networks to LLM-augmented pipelines — the practical question remains: what should you actually use? The answer depends heavily on your scale, data, and constraints.

### A Decision Framework

**Starting out with a new project?** Begin with **matrix factorization**. Libraries like **Surprise** make this trivial to implement, and MF provides a remarkably strong baseline that many deep learning approaches struggle to meaningfully beat on smaller datasets. If you have implicit feedback (clicks, views, purchases rather than explicit ratings), use Hu et al.'s weighted matrix factorization via the **Implicit** library.

**Have rich side features (user demographics, item metadata)?** This is where **DeepFM** or **Wide & Deep** architectures shine. The ability to incorporate arbitrary features beyond just user-item interaction IDs gives these models a significant advantage when feature engineering is feasible.

**Sequential behavior is a strong signal?** If the order in which users interact with items matters — and it usually does in domains like e-commerce, music, and video — consider **SASRec** or **BERT4Rec**. SASRec tends to be the more practical choice: it is faster to train (causal attention versus masked prediction) and equally effective in most benchmarks.

**Explicit graph structure in your data?** When you have social connections, item co-purchase graphs, or user-item bipartite graphs with meaningful topology, **LightGCN** is the go-to. It outperforms NGCF while being simpler, and the neighborhood aggregation provides natural smoothing that helps with sparse users.

**Operating at scale with millions of items?** You need the full pipeline. A **two-tower model** for candidate generation (index item embeddings with **FAISS** or ScaNN for ANN retrieval), followed by a feature-rich ranker. **TensorFlow Recommenders (TFRS)** provides solid implementations of the two-tower architecture with built-in support for ANN indexing.

**Suffering from cold-start?** Combine content-based and collaborative approaches in a **hybrid system**. For new items with no interaction data, LLM-generated metadata enrichment can bootstrap the content-based component effectively. **LightFM** is specifically designed for hybrid recommendation and handles cold-start gracefully.

### Key Libraries and Tools

| Library | Best For | Language |
|---------|----------|----------|
| **Surprise** | Classical CF and MF baselines | Python |
| **Implicit** | Implicit feedback MF (ALS, BPR) | Python |
| **LightFM** | Hybrid content + collaborative | Python |
| **RecBole** | Reproducing research models (NCF, SASRec, LightGCN, etc.) | Python/PyTorch |
| **FAISS** | ANN search for candidate generation | Python/C++ |
| **TensorFlow Recommenders** | Production two-tower and ranking models | Python/TF |

### Benchmark Datasets

For development and benchmarking, the community has converged on a few standard datasets: **MovieLens** (available in 100K, 1M, 10M, and 25M variants — start with 1M for development, use 25M for serious benchmarking), **Amazon Product Reviews** (multiple categories, good for testing cross-domain transfer), and **Yelp** (rich side information including social graph and review text).

> If your deep learning model cannot beat a well-tuned matrix factorization baseline on your dataset, the problem is almost certainly with your model, your hyperparameters, or your evaluation — not with the dataset being "too easy."

With these tools in hand, let us step back and consider where the field is headed.

## Conclusion

The trajectory of recommendation systems over the past three decades tells a story of relentless abstraction. Goldberg's Tapestry system in 1992 required users to manually write filtering rules. Collaborative filtering automated the "finding similar users" step but still relied on hand-defined similarity metrics. Matrix factorization learned the representations themselves but in a fixed, linear space. Deep learning introduced nonlinear representation learning and the ability to incorporate arbitrary features. Transformers brought attention-based sequential modeling. Graph neural networks captured structural relationships. And now, LLMs are beginning to bring general world knowledge and natural language understanding into the mix.

Three themes have persisted throughout this evolution.

**First, the progression from hand-crafted features to learned representations.** Each generation of methods has pushed the boundary of what the model learns versus what the engineer specifies. Matrix factorization learned embeddings. Deep models learned feature interactions. Transformers learned attention patterns. This trend will continue: I suspect the next frontier is models that learn not just representations but the evaluation criteria themselves — systems that infer what "relevant" means for a given user in a given context, rather than optimizing a fixed proxy metric.

**Second, the tension between model expressiveness and serving efficiency.** The most expressive model is useless if it cannot serve predictions in 200 milliseconds. This tension gave us the multi-stage pipeline architecture, and it continues to shape how LLMs are being integrated. Every advance in model capability has been accompanied by engineering innovations to make it practical: ANN search for embedding models, knowledge distillation for LLMs, quantization and pruning for on-device inference.

**Third, the growing recognition that accuracy alone is not enough.** The field is slowly but meaningfully shifting from "predict the next click" toward "provide genuine value to the user." Diversity, novelty, fairness, and long-term user satisfaction are becoming first-class objectives rather than afterthoughts. This is both an ethical imperative and a business one — systems that trap users in filter bubbles may optimize short-term engagement but erode long-term trust.

Looking ahead, several frontiers are particularly promising. **On-device recommendation** is gaining momentum, driven by privacy concerns and the desire for low-latency personalization without sending user data to the cloud. **Causal reasoning** — moving from "users who watched X also watched Y" to "watching X *caused* interest in Y" — remains a largely unsolved challenge that could fundamentally improve recommendation quality. And the **evaluation paradigm** will continue evolving, with more emphasis on long-term user welfare and less on instantaneous click prediction.

The recommendation systems field sits at a fascinating intersection: it is one of the most commercially mature applications of machine learning, generating billions of dollars in value annually, while simultaneously being reshaped by the latest advances in foundation models. For practitioners and researchers alike, there has rarely been a more interesting time to work on this problem.

## References

- [Goldberg, D., Nichols, D., Oki, B. M., & Terry, D., "Using Collaborative Filtering to Weave an Information Tapestry," *Communications of the ACM*, 1992](https://doi.org/10.1145/138859.138867)
- [Resnick, P., Iacovou, N., Suchak, M., Bergstrom, P., & Riedl, J., "GroupLens: An Open Architecture for Collaborative Filtering of Netnews," *CSCW*, 1994](https://doi.org/10.1145/192844.192905)
- [Shardanand, U. & Maes, P., "Social Information Filtering: Algorithms for Automating 'Word of Mouth'," *CHI*, 1995](https://doi.org/10.1145/223904.223931)
- [Sarwar, B., Karypis, G., Konstan, J., & Riedl, J., "Item-Based Collaborative Filtering Recommendation Algorithms," *WWW*, 2001](https://doi.org/10.1145/371920.372071)
- [Linden, G., Smith, B., & York, J., "Amazon.com Recommendations: Item-to-Item Collaborative Filtering," *IEEE Internet Computing*, 2003](https://doi.org/10.1109/MIC.2003.1167344)
- [Hu, Y., Koren, Y., & Volinsky, C., "Collaborative Filtering for Implicit Feedback Datasets," *ICDM*, 2008](https://doi.org/10.1109/ICDM.2008.22)
- [Koren, Y., Bell, R., & Volinsky, C., "Matrix Factorization Techniques for Recommender Systems," *IEEE Computer*, 2009](https://doi.org/10.1109/MC.2009.263)
- [Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L., "BPR: Bayesian Personalized Ranking from Implicit Feedback," *UAI*, 2009](https://arxiv.org/abs/1205.2618)
- [Covington, P., Adams, J., & Sargin, E., "Deep Neural Networks for YouTube Recommendations," *RecSys*, 2016](https://doi.org/10.1145/2959100.2959190)
- [Cheng, H.-T., Koc, L., Harmsen, J., et al., "Wide & Deep Learning for Recommender Systems," *DLRS*, 2016](https://arxiv.org/abs/1606.07792)
- [He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S., "Neural Collaborative Filtering," *WWW*, 2017](https://arxiv.org/abs/1708.05031)
- [Guo, H., Tang, R., Ye, Y., Li, Z., & He, X., "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction," *IJCAI*, 2017](https://arxiv.org/abs/1703.04247)
- [Kang, W.-C. & McAuley, J., "Self-Attentive Sequential Recommendation," *ICDM*, 2018](https://arxiv.org/abs/1808.09781)
- [Sun, F., Liu, J., Wu, J., et al., "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers," *CIKM*, 2019](https://arxiv.org/abs/1904.06690)
- [Wang, X., He, X., Wang, M., Feng, F., & Chua, T.-S., "Neural Graph Collaborative Filtering," *SIGIR*, 2019](https://arxiv.org/abs/1905.08108)
- [He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation," *SIGIR*, 2020](https://arxiv.org/abs/2002.02126)
- [Rendle, S., Krichene, W., Zhang, L., & Anderson, J., "Neural Collaborative Filtering vs. Matrix Factorization Revisited," *RecSys*, 2020](https://arxiv.org/abs/2005.09683)
- [Zhang, B., et al., "Wukong: Towards a Scaling Law for Large-Scale Recommendation," *ICML*, 2024](https://arxiv.org/abs/2403.02545)
- [Zhai, J., et al., "Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations," *ICML*, 2024](https://arxiv.org/abs/2402.17152)
- [Chen, J., Chi, L., Peng, B., & Yuan, Z., "HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models," *ByteDance*, 2024](https://arxiv.org/abs/2409.12740)
- [Meta Engineering, "Sequence Learning: A Paradigm Shift for Personalized Ads Recommendations," *Meta Engineering Blog*, 2024](https://engineering.fb.com/2024/11/19/data-infrastructure/sequence-learning-personalized-ads-recommendations/)
- [Meta Engineering, "Meta Andromeda: Supercharging Advantage+ Automation," *Meta Engineering Blog*, 2024](https://engineering.fb.com/2024/12/02/production-engineering/meta-andromeda-advantage-automation-next-gen-personalized-ads-retrieval-engine/)
- [Wang, S., et al., "OneRec: Unifying Retrieve and Rank with Generative Recommender," *Kuaishou*, 2025](https://arxiv.org/abs/2502.18965)
- [Zhang, Z., et al., "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer," *ByteDance*, 2025](https://arxiv.org/abs/2510.26104)
- [Meta Engineering, "Meta's Generative Ads Model (GEM)," *Meta Engineering Blog*, 2025](https://engineering.fb.com/2025/11/10/ml-applications/metas-generative-ads-model-gem-the-central-brain-accelerating-ads-recommendation-ai-innovation/)
- [He, et al., "PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations," *Google/YouTube*, 2025](https://arxiv.org/abs/2510.07784)
- [Yan, E., "How LLMs Are (and Aren't) Changing RecSys," *eugeneyan.com*, 2025](https://eugeneyan.com/writing/llm-patterns/)
- [LinkedIn Engineering, "360Brew: LinkedIn's Unified Generative Ranking Model," *LinkedIn Engineering Blog*, 2025](https://www.linkedin.com/blog/engineering/)
- [Netflix, "UniCoRn: Unified Contextual Recommendation," *Netflix Tech Blog*, 2025](https://netflixtechblog.com/)
- [Meta, "Embedding LLMs in Recommendation Systems," *Meta AI Blog*, 2026](https://ai.meta.com/blog/)
- [Harper, F. M. & Konstan, J. A., "The MovieLens Datasets: History and Context," *ACM TiiS*, 2015](https://doi.org/10.1145/2827872)
- [Johnson, J., Douze, M., & Jégou, H., "Billion-Scale Similarity Search with GPUs," *IEEE Transactions on Big Data*, 2019](https://arxiv.org/abs/1702.08734)
