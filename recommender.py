# recommender.py
# Step 4: Similarity-based recommendation system
#
# This file has the recommendation functions (the Streamlit app imports them).
# If you run it directly, it also tests the recommender:  python recommender.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# the product features used to compare two phones
FEATURES = [
    "avg_price", "avg_rating", "avg_battery", "avg_camera", "avg_performance",
    "avg_design", "avg_display", "avg_helpful_votes", "positive_pct",
]

# price is the most important thing when someone is choosing a phone,
# so I give it more weight after scaling
PRICE_WEIGHT = 3


def build_similarity(products):
    """Scale the features (0 to 1) and return a (phones x phones) cosine similarity table.
    I use MinMaxScaler here so all the values stay positive and the similarity is easy to read."""
    X = MinMaxScaler().fit_transform(products[FEATURES])
    X = pd.DataFrame(X, columns=FEATURES)
    X["avg_price"] = X["avg_price"] * PRICE_WEIGHT
    return cosine_similarity(X)


def recommend_similar(products, sim, model_name, n=5, same_segment_only=True):
    """Recommend phones that are similar to the selected phone.

    If same_segment_only is True, phones from the same K-Means segment come first
    (this is where clustering and recommendation are connected).
    """
    idx = products[products["model"] == model_name].index[0]
    my_segment = products.loc[idx, "segment"]

    result = products.copy()
    result["similarity"] = sim[idx]
    result["same_segment"] = result["segment"] == my_segment
    result = result[result["model"] != model_name]   # don't recommend the same phone

    if same_segment_only:
        result = result.sort_values(["same_segment", "similarity"], ascending=False)
    else:
        result = result.sort_values("similarity", ascending=False)

    result["similarity"] = result["similarity"].round(3)
    columns = ["model", "brand", "segment", "avg_price", "avg_rating", "similarity"]
    return result[columns].head(n)


def recommend_by_needs(products, max_price, min_rating, priority="avg_rating", brands=None, n=5):
    """Recommend phones from what the user types in: budget, minimum rating,
    the thing they care about most (camera, battery...) and optional brands."""
    result = products[(products["avg_price"] <= max_price) & (products["avg_rating"] >= min_rating)]
    if brands:
        result = result[result["brand"].isin(brands)]
    result = result.sort_values(priority, ascending=False)
    columns = ["model", "brand", "segment", "avg_price", "avg_rating", priority]
    columns = list(dict.fromkeys(columns))   # remove the repeat if priority is avg_rating
    return result[columns].head(n)


def basic_collaborative_filtering(reviews, model_name, n=3):
    """Very basic item-based collaborative filtering.
    Two phones are similar if different groups of customers (countries) rated them in a similar way."""
    matrix = reviews.pivot_table(index="model", columns="country", values="rating", aggfunc="mean")
    matrix = matrix.sub(matrix.mean(axis=1), axis=0)   # remove each phone's own average first
    sim = pd.DataFrame(cosine_similarity(matrix), index=matrix.index, columns=matrix.index)
    return sim[model_name].drop(model_name).sort_values(ascending=False).head(n).round(3)


# ------------------------------------------------------------
# Test the recommender when this file is run directly
# ------------------------------------------------------------
if __name__ == "__main__":
    products = pd.read_csv("product_clusters.csv")
    reviews = pd.read_csv("cleaned_mobile_reviews.csv")
    sim = build_similarity(products)

    # 1. recommend phones similar to a selected phone
    for phone in ["iPhone 15 Pro", "Redmi Note 13", "OnePlus 12"]:
        print(f"\nPhones similar to {phone}:")
        print(recommend_similar(products, sim, phone, n=3).to_string(index=False))

    # 2. recommend from user input
    print("\nUser input: budget $600, minimum rating 3.1, cares about camera most:")
    print(recommend_by_needs(products, 600, 3.1, "avg_camera", n=3).to_string(index=False))

    # 3. basic collaborative filtering
    print("\nBasic collaborative filtering - phones rated similarly to Pixel 8 across countries:")
    print(basic_collaborative_filtering(reviews, "Pixel 8"))

    # 4. validate the recommendations
    # Without forcing the segment, how many of the top 3 recommendations land in the
    # same price segment as the phone? Compare with picking 3 random phones.
    hits = 0
    total = 0
    price_diff_list = []
    for i in range(len(products)):
        phone = products.loc[i, "model"]
        top3 = recommend_similar(products, sim, phone, n=3, same_segment_only=False)
        my_segment = products.loc[i, "segment"]
        hits += (top3["segment"] == my_segment).sum()
        total += 3
        price_diff_list.append((top3["avg_price"] - products.loc[i, "avg_price"]).abs().mean())

    # chance of a random pick being in the same segment
    random_chance = 0
    for i in range(len(products)):
        same = (products["segment"] == products.loc[i, "segment"]).sum() - 1
        random_chance += same / (len(products) - 1)
    random_chance = random_chance / len(products)

    print("\n===== VALIDATION =====")
    print(f"Top-3 recommendations in the same segment: {hits / total * 100:.1f}%")
    print(f"Random picking would give only:             {random_chance * 100:.1f}%")
    # average price difference between a phone and ALL the other phones (for comparison)
    all_diff = []
    for i in range(len(products)):
        others = products.drop(i)
        all_diff.append((others["avg_price"] - products.loc[i, "avg_price"]).abs().mean())

    print(f"Average price gap to the recommended phones: ${sum(price_diff_list) / len(price_diff_list):.0f}")
    print(f"Average price gap to a random other phone:   ${sum(all_diff) / len(all_diff):.0f}")
