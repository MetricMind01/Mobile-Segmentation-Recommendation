# app.py
# Step 5: Streamlit app for the Mobile Product Segmentation and Recommendation System
#
# Run with:  streamlit run app.py
# (run preprocessing.py and clustering.py once before this)

import os
import pandas as pd
import plotly.express as px
import streamlit as st

from recommender import build_similarity, recommend_similar, recommend_by_needs

st.set_page_config(page_title="Mobile Segmentation & Recommendation", page_icon="📱", layout="wide")

# ------------------------------------------------------------
# Load the data
# ------------------------------------------------------------
needed_files = ["cleaned_mobile_reviews.csv", "product_clusters.csv", "cluster_summary.csv"]
missing = [f for f in needed_files if not os.path.exists(f)]
if missing:
    st.error(f"These files are missing: {missing}. Please run preprocessing.py and clustering.py first.")
    st.stop()


@st.cache_data
def load_data():
    reviews = pd.read_csv("cleaned_mobile_reviews.csv")
    products = pd.read_csv("product_clusters.csv")
    summary = pd.read_csv("cluster_summary.csv")
    return reviews, products, summary


reviews, products, summary = load_data()
similarity = build_similarity(products)

segment_order = ["Budget", "Mid-Range", "Upper Mid-Range", "Premium"]
segment_colors = {"Budget": "#2ca02c", "Mid-Range": "#1f77b4",
                  "Upper Mid-Range": "#ff7f0e", "Premium": "#d62728"}

# ------------------------------------------------------------
# Title and tabs
# ------------------------------------------------------------
st.title("📱 Mobile Product Segmentation & Recommendation System")
st.write("Explore 50,000 global mobile reviews, see how phones group into market segments "
         "and get phone recommendations.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "EDA", "Segments (Clusters)", "Recommendations", "Insights"]
)

# ------------------------------------------------------------
# TAB 1: Overview
# ------------------------------------------------------------
with tab1:
    st.subheader("Dataset overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Reviews", f"{len(reviews):,}")
    col2.metric("Brands", reviews["brand"].nunique())
    col3.metric("Phone models", reviews["model"].nunique())
    col4.metric("Countries", reviews["country"].nunique())
    col5.metric("Average rating", f"{reviews['rating'].mean():.2f}")

    st.write("**Cleaned review data (first 100 rows):**")
    st.dataframe(reviews.head(100))

    st.write("**Product table (one row per phone model, with its segment):**")
    st.dataframe(products)

# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.subheader("Exploratory Data Analysis")

    # filters
    f1, f2 = st.columns(2)
    chosen_brands = f1.multiselect("Brand", sorted(reviews["brand"].unique()),
                                   default=sorted(reviews["brand"].unique()))
    chosen_countries = f2.multiselect("Country", sorted(reviews["country"].unique()),
                                      default=sorted(reviews["country"].unique()))

    data = reviews[reviews["brand"].isin(chosen_brands) & reviews["country"].isin(chosen_countries)]

    if len(data) == 0:
        st.warning("No data for this filter. Please select at least one brand and one country.")
    else:
        st.write(f"Showing **{len(data):,}** reviews")

        c1, c2 = st.columns(2)

        brand_counts = data["brand"].value_counts().reset_index()
        brand_counts.columns = ["brand", "reviews"]
        fig = px.bar(brand_counts, x="brand", y="reviews", title="Reviews per brand", color="brand")
        c1.plotly_chart(fig)

        rating_counts = data["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["rating", "reviews"]
        fig = px.bar(rating_counts, x="rating", y="reviews", title="Rating distribution")
        c2.plotly_chart(fig)

        c3, c4 = st.columns(2)

        fig = px.box(data, x="brand", y="price_usd", color="brand", title="Price (USD) by brand")
        c3.plotly_chart(fig)

        sentiment_brand = pd.crosstab(data["brand"], data["sentiment"], normalize="index") * 100
        sentiment_brand = sentiment_brand.reset_index().melt(id_vars="brand", var_name="sentiment", value_name="percent")
        fig = px.bar(sentiment_brand, x="brand", y="percent", color="sentiment",
                     title="Sentiment % by brand",
                     color_discrete_map={"Negative": "tomato", "Neutral": "gold", "Positive": "mediumseagreen"})
        c4.plotly_chart(fig)

        c5, c6 = st.columns(2)

        corr_cols = ["price_usd", "rating", "battery_life_rating", "camera_rating",
                     "performance_rating", "design_rating", "display_rating", "helpful_votes", "age"]
        fig = px.imshow(data[corr_cols].corr().round(2), text_auto=True,
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation heatmap")
        c5.plotly_chart(fig)

        monthly = data.groupby("year_month").size().reset_index(name="reviews")
        fig = px.line(monthly, x="year_month", y="reviews", title="Reviews per month")
        c6.plotly_chart(fig)

        st.write("**Top 5 and lowest 5 rated phones (all data):**")
        sorted_products = products.sort_values("avg_rating", ascending=False)
        t1, t2 = st.columns(2)
        t1.write("Top 5")
        t1.dataframe(sorted_products[["model", "brand", "avg_price", "avg_rating"]].head(5))
        t2.write("Lowest 5")
        t2.dataframe(sorted_products[["model", "brand", "avg_price", "avg_rating"]].tail(5))

# ------------------------------------------------------------
# TAB 3: Segments
# ------------------------------------------------------------
with tab3:
    st.subheader("Product segments (K-Means, 4 clusters)")
    st.write("The phones are grouped with K-Means using their average price. "
             "Then each group is described using the ratings and reviews.")

    st.dataframe(summary)

    s1, s2 = st.columns(2)

    fig = px.scatter(products, x="avg_price", y="avg_rating", color="segment", hover_name="model",
                     hover_data=["brand"], size="total_reviews", title="Price vs rating by segment",
                     category_orders={"segment": segment_order}, color_discrete_map=segment_colors)
    fig.update_yaxes(range=[1, 5])
    s1.plotly_chart(fig)

    sorted_by_price = products.sort_values("avg_price")
    fig = px.bar(sorted_by_price, x="avg_price", y="model", color="segment", orientation="h",
                 title="Average price of each phone", category_orders={"segment": segment_order},
                 color_discrete_map=segment_colors)
    s2.plotly_chart(fig)

    aspect_cols = ["avg_battery", "avg_camera", "avg_performance", "avg_design", "avg_display"]
    aspect_long = summary.melt(id_vars="segment", value_vars=aspect_cols, var_name="aspect", value_name="rating")
    fig = px.bar(aspect_long, x="aspect", y="rating", color="segment", barmode="group",
                 title="Average aspect rating in each segment",
                 category_orders={"segment": segment_order}, color_discrete_map=segment_colors)
    fig.update_yaxes(range=[0, 5])
    st.plotly_chart(fig)

    chosen_segment = st.selectbox("See the phones inside a segment", segment_order)
    st.dataframe(products[products["segment"] == chosen_segment][
        ["model", "brand", "avg_price", "avg_rating", "positive_pct", "total_reviews"]])

# ------------------------------------------------------------
# TAB 4: Recommendations
# ------------------------------------------------------------
with tab4:
    st.subheader("Phone recommendations")

    mode = st.radio("How do you want recommendations?",
                    ["Similar to a phone I like", "Based on my needs"], horizontal=True)

    if mode == "Similar to a phone I like":
        r1, r2, r3 = st.columns([2, 1, 1])
        phone = r1.selectbox("Select a phone", sorted(products["model"]))
        how_many = r2.slider("How many recommendations", 1, 10, 5)
        same_segment = r3.checkbox("Prefer the same segment", value=True)

        selected = products[products["model"] == phone].iloc[0]
        st.write(f"**{phone}** ({selected['brand']}) - segment: **{selected['segment']}**, "
                 f"average price: **${selected['avg_price']:.0f}**, average rating: **{selected['avg_rating']}**")

        result = recommend_similar(products, similarity, phone, n=how_many, same_segment_only=same_segment)
        st.write("**Recommended phones:**")
        st.dataframe(result)

        fig = px.bar(result, x="similarity", y="model", color="segment", orientation="h",
                     title=f"Similarity to {phone}", color_discrete_map=segment_colors)
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig)

    else:
        n1, n2 = st.columns(2)
        price_min = int(products["avg_price"].min())
        price_max = int(products["avg_price"].max()) + 1
        budget = n1.slider("Maximum budget (USD)", price_min, price_max, price_max)

        rating_min = float(products["avg_rating"].min())
        rating_max = float(products["avg_rating"].max())
        wanted_rating = n2.slider("Minimum rating", rating_min, rating_max, rating_min, step=0.01)

        n3, n4 = st.columns(2)
        priority_options = {"Overall rating": "avg_rating", "Battery": "avg_battery", "Camera": "avg_camera",
                            "Performance": "avg_performance", "Design": "avg_design", "Display": "avg_display"}
        priority_name = n3.selectbox("What matters most to you?", list(priority_options.keys()))
        wanted_brands = n4.multiselect("Brands (leave empty for all)", sorted(products["brand"].unique()))

        result = recommend_by_needs(products, budget, wanted_rating, priority_options[priority_name],
                                    brands=wanted_brands, n=5)
        if len(result) == 0:
            st.warning("No phone matches these filters. Try a higher budget or a lower minimum rating.")
        else:
            st.write("**Best matches for you:**")
            st.dataframe(result)

# ------------------------------------------------------------
# TAB 5: Insights
# ------------------------------------------------------------
with tab5:
    st.subheader("Key insights")

    corr_price_rating = reviews["price_usd"].corr(reviews["rating"])
    best = products.sort_values("avg_rating", ascending=False).iloc[0]
    worst = products.sort_values("avg_rating", ascending=True).iloc[0]
    premium_row = summary[summary["segment"] == "Premium"].iloc[0]
    budget_row = summary[summary["segment"] == "Budget"].iloc[0]
    biggest = summary.sort_values("total_reviews", ascending=False).iloc[0]

    st.markdown(f"""
**Product segmentation**
- The 22 phone models fall into 4 segments: Budget (${budget_row['avg_price']:.0f}), Mid-Range, Upper Mid-Range and Premium (${premium_row['avg_price']:.0f}).
- The **{biggest['segment']}** segment has the most reviews ({biggest['total_reviews']:,}).

**High and low performing products**
- Highest rated: **{best['model']}** ({best['avg_rating']}). Lowest rated: **{worst['model']}** ({worst['avg_rating']}).
- The gap between the best and worst phone is only {best['avg_rating'] - worst['avg_rating']:.2f} points.

**Price vs performance**
- Correlation between price and rating is **{corr_price_rating:.2f}**, so a higher price does not mean a higher rating.
- Premium phones cost about {premium_row['avg_price'] / budget_row['avg_price']:.1f}x more than Budget phones, but both segments have about the same rating ({premium_row['avg_rating']} vs {budget_row['avg_rating']}).

**Customer preference patterns**
- Reviews are spread almost equally between countries, brands and platforms, so no country strongly prefers one brand.
- {reviews['verified_purchase'].mean() * 100:.0f}% of the reviews are verified purchases.

**Business suggestions**
- Budget brands can use value for money in their marketing, because customers rate them the same as premium brands.
- Premium brands should show clearly what the extra price gives, because aspect ratings (camera, battery, design...) are not higher.
- Use the recommender on a website to show similar phones from the same price segment.
""")
