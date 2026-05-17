from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from concurrent.futures import ProcessPoolExecutor, as_completed

def cluster_prompts(prompts, num_clusters):
    if len(prompts) < num_clusters:
        print(f"[Warning] Only {len(prompts)} prompts, but {num_clusters} clusters requested. Skipping clustering.")
        return [0] * len(prompts)
    # if len(prompts) < num_clusters:
    #     print(f"[Warning] Only {len(prompts)} prompts, but {num_clusters} clusters requested. Assigning each prompt to its own cluster.")
    #     return list(range(len(prompts)))  # 每个 prompt 一个独立 cluster
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(prompts)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

# 贝叶斯更新概率
def update_probabilities(scores, labels, prior_probabilities):
    num_clusters = len(prior_probabilities)
    cluster_scores = [[] for _ in range(num_clusters)]
    for i, label in enumerate(labels):
        cluster_scores[label].append(scores[i])

    posterior_probabilities = []
    likelihoods = []
    for j in range(num_clusters):
        if not cluster_scores[j]:
            likelihoods.append(0)
        else:
            likelihoods.append(sum(cluster_scores[j]) / len(cluster_scores[j]))

    denominator = sum([likelihoods[k] * prior_probabilities[k] for k in range(num_clusters)])
    if denominator == 0:
        print("[Warning] All likelihoods are zero. Returning uniform posterior.")
        return [1.0 / num_clusters] * num_clusters

    for j in range(num_clusters):
        posterior_probability = (likelihoods[j] * prior_probabilities[j]) / denominator
        posterior_probabilities.append(posterior_probability)
    return posterior_probabilities