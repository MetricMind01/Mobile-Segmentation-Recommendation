# eda.py
# Step 2: Exploratory Data Analysis (EDA)
# All charts are saved in the "charts" folder and the key findings in eda_report.txt
#
# Run after preprocessing.py:  python eda.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("charts", exist_ok=True)
sns.set_theme(style="whitegrid")

df = pd.read_csv("cleaned_mobile_reviews.csv")
product = pd.read_csv("product_summary.csv")

findings = []  # I will collect the key findings here and save them at the end

# ------------------------------------------------------------
# 1. Statistical summary
# ------------------------------------------------------------
print("Shape:", df.shape)
print("\nNumbers summary:")
print(df[["age", "price_usd", "rating", "aspect_avg", "helpful_votes"]].describe().round(2))

# ------------------------------------------------------------
# 2. Product distribution across brands and countries
# ------------------------------------------------------------
brand_counts = df["brand"].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=brand_counts.index, y=brand_counts.values, color="steelblue")
plt.title("Number of reviews per brand")
plt.xlabel("Brand")
plt.ylabel("Reviews")
plt.tight_layout()
plt.savefig("charts/01_reviews_per_brand.png")
plt.close()

country_counts = df["country"].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=country_counts.index, y=country_counts.values, color="darkorange")
plt.title("Number of reviews per country")
plt.xlabel("Country")
plt.ylabel("Reviews")
plt.tight_layout()
plt.savefig("charts/02_reviews_per_country.png")
plt.close()

findings.append(f"Most reviewed brand: {brand_counts.index[0]} ({brand_counts.iloc[0]} reviews), "
                f"least reviewed: {brand_counts.index[-1]} ({brand_counts.iloc[-1]} reviews).")
findings.append(f"Most reviews come from {country_counts.index[0]} ({country_counts.iloc[0]}), "
                f"fewest from {country_counts.index[-1]} ({country_counts.iloc[-1]}). Countries are almost equally spread.")

# ------------------------------------------------------------
# 3. Rating and sentiment distribution
# ------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x="rating", data=df, color="seagreen", ax=axes[0])
axes[0].set_title("Rating distribution")
sentiment_counts = df["sentiment"].value_counts()
axes[1].pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%")
axes[1].set_title("Sentiment share")
plt.tight_layout()
plt.savefig("charts/03_rating_and_sentiment.png")
plt.close()

pos_pct = round(sentiment_counts["Positive"] / len(df) * 100, 1)
neg_pct = round(sentiment_counts["Negative"] / len(df) * 100, 1)
findings.append(f"Overall average rating is {df['rating'].mean():.2f} out of 5. "
                f"{pos_pct}% of the reviews are Positive and {neg_pct}% are Negative.")

# ------------------------------------------------------------
# 4. Price by brand and rating by brand
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
order = df.groupby("brand")["price_usd"].mean().sort_values(ascending=False).index
sns.boxplot(x="brand", y="price_usd", data=df, order=order, color="lightblue")
plt.title("Price (USD) range by brand")
plt.tight_layout()
plt.savefig("charts/04_price_by_brand.png")
plt.close()

brand_stats = df.groupby("brand").agg(avg_price=("price_usd", "mean"), avg_rating=("rating", "mean")).round(2)
brand_stats = brand_stats.sort_values("avg_price", ascending=False)
print("\nAverage price and rating by brand:")
print(brand_stats)

plt.figure(figsize=(9, 5))
sns.barplot(x=brand_stats.index, y=brand_stats["avg_rating"], color="mediumpurple")
plt.ylim(0, 5)
plt.title("Average rating by brand")
plt.ylabel("Average rating (out of 5)")
plt.tight_layout()
plt.savefig("charts/05_rating_by_brand.png")
plt.close()

findings.append(f"Most expensive brand is {brand_stats.index[0]} (avg ${brand_stats['avg_price'].iloc[0]:.0f}), "
                f"cheapest is {brand_stats.index[-1]} (avg ${brand_stats['avg_price'].iloc[-1]:.0f}).")
best_brand = brand_stats["avg_rating"].idxmax()
worst_brand = brand_stats["avg_rating"].idxmin()
findings.append(f"Best rated brand is {best_brand} ({brand_stats.loc[best_brand, 'avg_rating']}) and lowest is "
                f"{worst_brand} ({brand_stats.loc[worst_brand, 'avg_rating']}). The gap is very small.")

# ------------------------------------------------------------
# 5. Top rated and low rated products
# ------------------------------------------------------------
sorted_products = product.sort_values("avg_rating", ascending=False)
top5 = sorted_products.head(5)
low5 = sorted_products.tail(5)

print("\nTop 5 rated phones:")
print(top5[["brand", "model", "avg_price", "avg_rating"]])
print("\nLowest 5 rated phones:")
print(low5[["brand", "model", "avg_price", "avg_rating"]])

top_low = pd.concat([top5, low5])
plt.figure(figsize=(9, 6))
colors = ["green"] * 5 + ["red"] * 5
plt.barh(top_low["model"], top_low["avg_rating"], color=colors)
plt.xlim(0, 5)
plt.gca().invert_yaxis()
plt.title("Top 5 (green) and lowest 5 (red) rated phones")
plt.xlabel("Average rating")
plt.tight_layout()
plt.savefig("charts/06_top_and_low_rated.png")
plt.close()

findings.append(f"Top rated phone: {top5.iloc[0]['model']} ({top5.iloc[0]['avg_rating']}). "
                f"Lowest rated phone: {low5.iloc[-1]['model']} ({low5.iloc[-1]['avg_rating']}). "
                f"Difference between best and worst is only {top5.iloc[0]['avg_rating'] - low5.iloc[-1]['avg_rating']:.2f} points.")

# ------------------------------------------------------------
# 6. Price vs rating and specifications (correlation)
# ------------------------------------------------------------
corr_cols = ["price_usd", "rating", "battery_life_rating", "camera_rating",
             "performance_rating", "design_rating", "display_rating", "helpful_votes", "age"]
corr = df[corr_cols].corr().round(2)

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation heatmap")
plt.tight_layout()
plt.savefig("charts/07_correlation_heatmap.png")
plt.close()

price_rating_corr = corr.loc["price_usd", "rating"]
findings.append(f"Correlation between price and rating is {price_rating_corr} -> price has no effect on rating. "
                f"Expensive phones are NOT rated higher than cheap phones.")
findings.append(f"Overall rating is strongly linked with the 5 aspect ratings (about {corr.loc['rating', 'camera_rating']}) "
                f"and also with helpful votes ({corr.loc['rating', 'helpful_votes']}).")

plt.figure(figsize=(8, 6))
sns.scatterplot(x="avg_price", y="avg_rating", hue="brand", data=product, s=120)
plt.ylim(1, 5)
plt.title("Phone price vs average rating (each dot = one model)")
plt.xlabel("Average price (USD)")
plt.tight_layout()
plt.savefig("charts/08_price_vs_rating.png")
plt.close()

# ------------------------------------------------------------
# 7. Aspect ratings by brand (what customers like)
# ------------------------------------------------------------
aspect_cols = ["battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating"]
aspect_by_brand = df.groupby("brand")[aspect_cols].mean().round(2)

plt.figure(figsize=(9, 6))
sns.heatmap(aspect_by_brand, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title("Average aspect rating by brand")
plt.tight_layout()
plt.savefig("charts/09_aspects_by_brand.png")
plt.close()

# ------------------------------------------------------------
# 8. Sentiment by brand
# ------------------------------------------------------------
sent_by_brand = pd.crosstab(df["brand"], df["sentiment"], normalize="index") * 100
sent_by_brand = sent_by_brand[["Negative", "Neutral", "Positive"]]
sent_by_brand.plot(kind="bar", stacked=True, figsize=(9, 5), color=["tomato", "gold", "mediumseagreen"])
plt.title("Sentiment % by brand")
plt.ylabel("% of reviews")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("charts/10_sentiment_by_brand.png")
plt.close()

# ------------------------------------------------------------
# 9. Trend over time
# ------------------------------------------------------------
monthly = df.groupby("year_month").size()
plt.figure(figsize=(12, 4))
plt.plot(monthly.index, monthly.values, marker="o", markersize=3)
plt.xticks(monthly.index[::3], rotation=90)
plt.title("Number of reviews per month")
plt.tight_layout()
plt.savefig("charts/11_reviews_over_time.png")
plt.close()

# ------------------------------------------------------------
# 10. Customer preference patterns (who buys / reviews what)
# ------------------------------------------------------------
brand_country = pd.crosstab(df["country"], df["brand"], normalize="index") * 100
plt.figure(figsize=(10, 6))
sns.heatmap(brand_country.round(1), annot=True, cmap="Oranges", fmt=".1f")
plt.title("Brand share (%) inside each country")
plt.tight_layout()
plt.savefig("charts/12_brand_share_by_country.png")
plt.close()

brand_age = pd.crosstab(df["age_group"], df["brand"], normalize="index") * 100
platform_counts = df["source"].value_counts()
verified_pct = df["verified_purchase"].mean() * 100

top_brand_per_country = brand_country.idxmax(axis=1)
print("\nMost reviewed brand in each country:")
print(top_brand_per_country)

findings.append("Most reviewed brand per country: " +
                ", ".join([f"{c}: {b}" for c, b in top_brand_per_country.items()]) +
                ". The shares are close to equal, so there is no strong country preference.")
findings.append(f"Most reviews come from {platform_counts.index[0]} ({platform_counts.iloc[0]}), "
                f"{verified_pct:.1f}% of all reviews are verified purchases.")
findings.append(f"Most reviewers are in the {df['age_group'].value_counts().index[0]} age group, "
                f"average age is {df['age'].mean():.1f}.")

# ------------------------------------------------------------
# 11. Save the findings
# ------------------------------------------------------------
print("\n===== KEY FINDINGS =====")
with open("eda_report.txt", "w", encoding="utf-8") as f:
    f.write("EDA KEY FINDINGS\n================\n\n")
    for i, line in enumerate(findings, start=1):
        print(f"{i}. {line}")
        f.write(f"{i}. {line}\n")

print("\nCharts saved in the 'charts' folder, report saved as eda_report.txt")
