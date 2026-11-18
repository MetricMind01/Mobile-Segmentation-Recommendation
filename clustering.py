# clustering.py
# Step 3: Segment the phones into 4 groups using K-Means
#
# Run after preprocessing.py:  python clustering.py

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")

product = pd.read_csv("product_summary.csv")
print("Products:", product.shape)

# ------------------------------------------------------------
# 1. Choose the features for clustering
# ------------------------------------------------------------
# I tried 3 different feature sets and checked the silhouette score for each
# (higher silhouette = clusters are better separated).
feature_options = {
    "price only": ["avg_price"],
    "price + rating": ["avg_price", "avg_rating"],
    "all numeric features": ["avg_price", "avg_rating", "avg_battery", "avg_camera",
                             "avg_performance", "avg_design", "avg_display",
                             "avg_helpful_votes", "positive_pct"],
}

print("\nSilhouette score with 4 clusters for each feature set:")
for name, cols in feature_options.items():
    X_test = StandardScaler().fit_transform(product[cols])
    km_test = KMeans(n_clusters=4, random_state=42, n_init=10).fit(X_test)
    score = silhouette_score(X_test, km_test.labels_)
    print(f"  {name:22s} -> {score:.3f}")

# All the ratings are almost the same for every phone (about 3.05 to 3.15),
# so when I add them, K-Means just clusters the small random differences and the
# silhouette score drops. Price is the thing that really separates the phones,
# so I use price for the clustering and then use the ratings to ANALYSE each cluster.
features = ["avg_price"]

# ------------------------------------------------------------
# 2. Scaling
# ------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(product[features])

# ------------------------------------------------------------
# 3. Choose the number of clusters (elbow method + silhouette)
# ------------------------------------------------------------
k_values = range(2, 9)
inertia_list = []
silhouette_list = []

for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia_list.append(km.inertia_)
    silhouette_list.append(silhouette_score(X_scaled, km.labels_))

print("\nk  inertia  silhouette")
for k, i, s in zip(k_values, inertia_list, silhouette_list):
    print(f"{k}  {i:7.2f}  {s:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(list(k_values), inertia_list, marker="o")
axes[0].axvline(4, color="red", linestyle="--")
axes[0].set_title("Elbow method")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia")
axes[1].plot(list(k_values), silhouette_list, marker="o", color="green")
axes[1].axvline(4, color="red", linestyle="--")
axes[1].set_title("Silhouette score")
axes[1].set_xlabel("Number of clusters (k)")
plt.tight_layout()
plt.savefig("charts/13_elbow_silhouette.png")
plt.close()

# The elbow bends at k = 4 (inertia stops dropping fast after that) and
# 4 clusters give nice business groups: Budget, Mid-Range, Upper Mid-Range, Premium.
# After k = 4 the extra clusters just split one brand into smaller pieces.

# ------------------------------------------------------------
# 4. Final K-Means with 4 clusters
# ------------------------------------------------------------
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
product["cluster"] = kmeans.fit_predict(X_scaled)

# The cluster numbers (0,1,2,3) are random, so I name them by average price
cluster_order = product.groupby("cluster")["avg_price"].mean().sort_values().index
segment_names = ["Budget", "Mid-Range", "Upper Mid-Range", "Premium"]
name_map = {}
for i, cluster_number in enumerate(cluster_order):
    name_map[cluster_number] = segment_names[i]
product["segment"] = product["cluster"].map(name_map)

# ------------------------------------------------------------
# 5. Cluster analysis
# ------------------------------------------------------------
summary = product.groupby("segment").agg(
    products=("model", "count"),
    brands=("brand", lambda x: ", ".join(sorted(x.unique()))),
    avg_price=("avg_price", "mean"),
    avg_rating=("avg_rating", "mean"),
    avg_battery=("avg_battery", "mean"),
    avg_camera=("avg_camera", "mean"),
    avg_performance=("avg_performance", "mean"),
    avg_design=("avg_design", "mean"),
    avg_display=("avg_display", "mean"),
    positive_pct=("positive_pct", "mean"),
    total_reviews=("total_reviews", "sum"),
).round(2)
summary = summary.sort_values("avg_price").reset_index()

print("\n===== CLUSTER SUMMARY =====")
print(summary.to_string(index=False))

print("\nPhones in each segment:")
for seg in segment_names:
    models = product[product["segment"] == seg]["model"].tolist()
    print(f"  {seg}: {', '.join(models)}")

# ------------------------------------------------------------
# 6. Visualisations
# ------------------------------------------------------------
colors = {"Budget": "tab:green", "Mid-Range": "tab:blue",
          "Upper Mid-Range": "tab:orange", "Premium": "tab:red"}

# 6a. price of every phone coloured by segment
sorted_products = product.sort_values("avg_price")
plt.figure(figsize=(9, 7))
plt.barh(sorted_products["model"], sorted_products["avg_price"],
         color=[colors[s] for s in sorted_products["segment"]])
plt.title("Average price of each phone, coloured by segment")
plt.xlabel("Average price (USD)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
plt.legend(handles, colors.keys(), title="Segment")
plt.tight_layout()
plt.savefig("charts/14_segments_price_bar.png")
plt.close()

# 6b. price vs rating scatter coloured by segment
plt.figure(figsize=(9, 6))
sns.scatterplot(x="avg_price", y="avg_rating", hue="segment", hue_order=segment_names,
                palette=colors, data=product, s=150)
plt.ylim(1, 5)
plt.title("Segments: price vs rating (rating stays flat as price goes up)")
plt.xlabel("Average price (USD)")
plt.ylabel("Average rating")
plt.tight_layout()
plt.savefig("charts/15_segments_scatter.png")
plt.close()

# 6c. aspect ratings per segment
aspect_cols = ["avg_battery", "avg_camera", "avg_performance", "avg_design", "avg_display"]
plt.figure(figsize=(9, 5))
sns.heatmap(summary.set_index("segment")[aspect_cols], annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Average aspect rating in each segment")
plt.tight_layout()
plt.savefig("charts/16_segment_aspects.png")
plt.close()

# 6d. number of reviews per segment
plt.figure(figsize=(7, 5))
plt.bar(summary["segment"], summary["total_reviews"], color=[colors[s] for s in summary["segment"]])
plt.title("Total reviews in each segment")
plt.ylabel("Reviews")
plt.tight_layout()
plt.savefig("charts/17_segment_reviews.png")
plt.close()

# ------------------------------------------------------------
# 7. Business insights (written automatically from the numbers)
# ------------------------------------------------------------
lines = []
for _, row in summary.iterrows():
    lines.append(f"{row['segment']}: {row['products']} phones ({row['brands']}), "
                 f"average price ${row['avg_price']:.0f}, average rating {row['avg_rating']}, "
                 f"{row['positive_pct']:.1f}% positive reviews, {row['total_reviews']} reviews in total.")

premium = summary[summary["segment"] == "Premium"].iloc[0]
budget = summary[summary["segment"] == "Budget"].iloc[0]
price_ratio = premium["avg_price"] / budget["avg_price"]
rating_gap = premium["avg_rating"] - budget["avg_rating"]
biggest = summary.sort_values("total_reviews", ascending=False).iloc[0]

lines.append("")
lines.append(f"Premium phones cost about {price_ratio:.1f}x more than Budget phones, "
             f"but the rating difference is only {rating_gap:+.2f}. Higher price does not mean happier customers here.")
lines.append(f"The {biggest['segment']} segment has the most reviews ({biggest['total_reviews']}), "
             f"so it has the most customer activity.")
lines.append("Budget brands can compete on value, because customers rate them the same as premium brands.")
lines.append("Premium brands need to prove what the extra price gives (camera, design, performance) because "
             "the aspect ratings are not higher than the cheaper segments.")

print("\n===== INSIGHTS =====")
with open("cluster_report.txt", "w", encoding="utf-8") as f:
    f.write("CLUSTER ANALYSIS REPORT (K-Means, k = 4)\n========================================\n\n")
    for line in lines:
        print(line)
        f.write(line + "\n")

# ------------------------------------------------------------
# 8. Save everything
# ------------------------------------------------------------
product.to_csv("product_clusters.csv", index=False)
summary.to_csv("cluster_summary.csv", index=False)

with open("kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved: product_clusters.csv, cluster_summary.csv, cluster_report.txt, kmeans_model.pkl, scaler.pkl")
