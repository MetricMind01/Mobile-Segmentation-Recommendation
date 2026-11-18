# preprocessing.py
# Step 1: Load the raw CSV, clean it, add some new columns, encode it
# and build a product-level table (one row per phone model).
#
# Run this first:  python preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------
# 1. Load the data
# ------------------------------------------------------------
df = pd.read_csv("Mobile_Reviews_Sentiment_null.csv")

print("Shape of the data:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())

# ------------------------------------------------------------
# 2. Check missing values and duplicates
# ------------------------------------------------------------
print("\nMissing values before cleaning:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# 3. Handle missing values
# ------------------------------------------------------------

# price_usd is missing in some rows, but price_local is there in most of them.
# price_local = price_usd * exchange_rate, so I can get the price back by dividing.
# price_local looks like "R$1754.35", so first I take only the number part.
df["local_amount"] = df["price_local"].str.extract(r"(\d+\.?\d*)$")[0].astype(float)

missing_price = df["price_usd"].isnull()
df.loc[missing_price, "price_usd"] = (
    df["local_amount"] / df["exchange_rate_to_usd"]
).round(2)

# a few rows have both price_usd and price_local missing -> use the median price of that model
df["price_usd"] = df["price_usd"].fillna(df.groupby("model")["price_usd"].transform("median"))

# rating is missing in some rows, but the 5 aspect ratings are always there
# (battery, camera, performance, design, display) so I use their average
aspect_cols = [
    "battery_life_rating",
    "camera_rating",
    "performance_rating",
    "design_rating",
    "display_rating",
]
df["aspect_avg"] = df[aspect_cols].mean(axis=1).round(2)
df["rating"] = df["rating"].fillna(df["aspect_avg"].round())
df["rating"] = df["rating"].clip(1, 5).astype(int)


# sentiment is missing in some rows -> decide it from the rating
def rating_to_sentiment(r):
    if r >= 4:
        return "Positive"
    elif r == 3:
        return "Neutral"
    else:
        return "Negative"


df["sentiment"] = df["sentiment"].fillna(df["rating"].apply(rating_to_sentiment))

# source (the website) is missing in some rows -> I can't guess it, so "Unknown"
df["source"] = df["source"].fillna("Unknown")

# ------------------------------------------------------------
# 4. Remove duplicates
# ------------------------------------------------------------
before = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset="review_id")
print("\nDuplicates removed:", before - len(df))

# ------------------------------------------------------------
# 5. Data type conversion
# ------------------------------------------------------------
df["review_date"] = pd.to_datetime(df["review_date"], format="%m/%d/%Y")
df["verified_purchase"] = df["verified_purchase"].astype(int)

# ------------------------------------------------------------
# 6. Feature engineering (new columns)
# ------------------------------------------------------------
df["review_year"] = df["review_date"].dt.year
df["year_month"] = df["review_date"].dt.strftime("%Y-%m")

df["age_group"] = pd.cut(
    df["age"],
    bins=[17, 24, 34, 44, 54, 100],
    labels=["18-24", "25-34", "35-44", "45-54", "55+"],
)

# ------------------------------------------------------------
# 7. Encoding the categorical columns
# ------------------------------------------------------------
# I keep the original text columns and add a *_code column next to them.
# (I will not feed brand_code into K-Means, because numbers like 0,1,2...
#  would make the model think there is an order between brands.)
le = LabelEncoder()
for col in ["brand", "model", "country", "source"]:
    df[col + "_code"] = le.fit_transform(df[col])

df["sentiment_code"] = df["sentiment"].map({"Negative": 0, "Neutral": 1, "Positive": 2})

# ------------------------------------------------------------
# 8. Drop the columns I don't need and save the clean data
# ------------------------------------------------------------
# customer_name = personal info and not useful, price_local/local_amount/exchange rate
# were only needed to fix the price
df = df.drop(columns=["customer_name", "price_local", "local_amount", "exchange_rate_to_usd"])

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "missing values left")

df.to_csv("cleaned_mobile_reviews.csv", index=False)
print("Saved: cleaned_mobile_reviews.csv", df.shape)

# ------------------------------------------------------------
# 9. Build the product table (one row per phone model)
# ------------------------------------------------------------
# The raw data has 50,000 reviews, but I want to segment and recommend *phones*,
# so I group the reviews by brand + model and take averages.
product = df.groupby(["brand", "model"]).agg(
    avg_price=("price_usd", "mean"),
    avg_rating=("rating", "mean"),
    avg_battery=("battery_life_rating", "mean"),
    avg_camera=("camera_rating", "mean"),
    avg_performance=("performance_rating", "mean"),
    avg_design=("design_rating", "mean"),
    avg_display=("display_rating", "mean"),
    avg_helpful_votes=("helpful_votes", "mean"),
    positive_pct=("sentiment", lambda s: (s == "Positive").mean() * 100),
    verified_pct=("verified_purchase", lambda s: s.mean() * 100),
    total_reviews=("review_id", "count"),
).reset_index()

product = product.round(2)
product.to_csv("product_summary.csv", index=False)

print("\nProduct table:", product.shape)
print(product)
print("\nSaved: product_summary.csv")
