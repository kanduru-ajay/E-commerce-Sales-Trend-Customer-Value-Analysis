import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------
# 1. CREATE PROJECT DIRECTORIES
# ----------------------------------------------------------

os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("charts", exist_ok=True)

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(42)

# ----------------------------------------------------------
# 2. GENERATE SIMULATED E-COMMERCE DATA
# ----------------------------------------------------------

n = 2000

orders = pd.DataFrame({
    "order_id": range(1001, 1001 + n),

    "order_date": (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(
            rng.integers(0, 365, n), unit="D"
        )
    ),

    "customer_id": rng.integers(1, 301, n),

    "category": rng.choice(
        ["Electronics", "Fashion", "Home", "Books"],
        n,
        p=[0.30, 0.35, 0.20, 0.15]
    ),

    "amount": rng.gamma(2.0, 600, n).round(2)
})

orders.to_csv("data/ecommerce_orders.csv", index=False)

print("\nDATASET CREATED SUCCESSFULLY")
print(orders.head())

# ----------------------------------------------------------
# 3. DATA QUALITY CHECK
# ----------------------------------------------------------

print("\nDATASET INFORMATION")
print(orders.info())

print("\nMISSING VALUES")
print(orders.isnull().sum())

print("\nDUPLICATE ROWS")
print(orders.duplicated().sum())

print("\nDATASET SHAPE")
print(orders.shape)

# ----------------------------------------------------------
# 4. FEATURE ENGINEERING
# ----------------------------------------------------------

orders["order_date"] = pd.to_datetime(orders["order_date"])

orders["month"] = orders["order_date"].dt.to_period("M")
orders["month_name"] = orders["order_date"].dt.strftime("%b")
orders["weekday"] = orders["order_date"].dt.day_name()
orders["quarter"] = orders["order_date"].dt.quarter
orders["year"] = orders["order_date"].dt.year

# ----------------------------------------------------------
# 5. MONTHLY REVENUE ANALYSIS
# ----------------------------------------------------------

monthly = (
    orders.groupby("month")["amount"]
    .sum()
    .sort_index()
)

monthly_growth = monthly.pct_change() * 100

monthly_report = pd.DataFrame({
    "revenue": monthly,
    "growth_percentage": monthly_growth
})

monthly_report.to_csv(
    "outputs/monthly_revenue.csv"
)

print("\nMONTHLY REVENUE")
print(monthly_report.round(2))

best_month = monthly.idxmax()
worst_month = monthly.idxmin()

print("\nBEST MONTH:", best_month)
print("BEST MONTH REVENUE:", round(monthly.max(), 2))

print("\nWORST MONTH:", worst_month)
print("WORST MONTH REVENUE:", round(monthly.min(), 2))

# Monthly revenue chart

plt.figure(figsize=(12, 6))

plt.plot(
    monthly.index.astype(str),
    monthly.values,
    marker="o",
    linewidth=2,
    color="royalblue"
)

plt.title("Monthly E-commerce Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("charts/monthly_revenue.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 6. CATEGORY PERFORMANCE ANALYSIS
# ----------------------------------------------------------

category_report = (
    orders.groupby("category")["amount"]
    .agg(
        total_revenue="sum",
        average_order_value="mean",
        number_of_orders="count"
    )
    .sort_values("total_revenue", ascending=False)
)

category_report["revenue_share"] = (
    category_report["total_revenue"]
    / orders["amount"].sum() * 100
)

category_report.to_csv("outputs/category_analysis.csv")

print("\nCATEGORY PERFORMANCE")
print(category_report.round(2))

best_category = category_report["total_revenue"].idxmax()

print("\nBEST CATEGORY:", best_category)

# Category revenue chart

plt.figure(figsize=(10, 6))

sns.barplot(
    data=category_report.reset_index(),
    x="category",
    y="total_revenue",
    hue="category",
    palette="viridis",
    legend=False
)

plt.title("Revenue by Product Category")
plt.xlabel("Category")
plt.ylabel("Total Revenue")

plt.tight_layout()
plt.savefig("charts/category_revenue.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 7. MONTHLY CATEGORY PIVOT TABLE
# ----------------------------------------------------------

monthly_category = pd.pivot_table(
    orders,
    values="amount",
    index="month",
    columns="category",
    aggfunc="sum",
    fill_value=0
)

monthly_category.to_csv(
    "outputs/monthly_category_pivot.csv"
)

print("\nMONTHLY CATEGORY PIVOT TABLE")
print(monthly_category.round(2))

# ----------------------------------------------------------
# 8. CUSTOMER VALUE ANALYSIS
# ----------------------------------------------------------

customer_spend = (
    orders.groupby("customer_id")
    .agg(
        total_spend=("amount", "sum"),
        total_orders=("order_id", "count"),
        average_order_value=("amount", "mean"),
        last_purchase=("order_date", "max")
    )
    .sort_values("total_spend", ascending=False)
)

customer_spend.to_csv("outputs/customer_analysis.csv")

print("\nTOP 10 CUSTOMERS")
print(customer_spend.head(10).round(2))

# Top 10% customer revenue contribution

top_10_count = max(1, int(np.ceil(len(customer_spend) * 0.10)))

top_10_customers = customer_spend.head(top_10_count)

top_10_revenue_share = (
    top_10_customers["total_spend"].sum()
    / customer_spend["total_spend"].sum()
) * 100

print(
    f"\nTOP 10% CUSTOMERS GENERATE "
    f"{top_10_revenue_share:.2f}% OF TOTAL REVENUE"
)

# Customer spending chart

plt.figure(figsize=(10, 6))

customer_spend.head(10)["total_spend"].sort_values().plot(
    kind="barh",
    color="teal"
)

plt.title("Top 10 Most Valuable Customers")
plt.xlabel("Total Customer Spending")
plt.ylabel("Customer ID")

plt.tight_layout()
plt.savefig("charts/top_customers.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 9. CUSTOMER SEGMENTATION
# ----------------------------------------------------------

customer_spend["customer_segment"] = pd.qcut(
    customer_spend["total_spend"].rank(method="first"),
    q=4,
    labels=[
        "Low Value",
        "Regular",
        "High Value",
        "VIP"
    ]
)

segment_report = (
    customer_spend.groupby(
        "customer_segment",
        observed=False
    )
    .agg(
        customers=("total_spend", "count"),
        revenue=("total_spend", "sum"),
        average_spend=("total_spend", "mean")
    )
)

segment_report["revenue_share"] = (
    segment_report["revenue"]
    / customer_spend["total_spend"].sum()
    * 100
)

segment_report.to_csv("outputs/customer_segments.csv")

print("\nCUSTOMER SEGMENTATION")
print(segment_report.round(2))

# ----------------------------------------------------------
# 10. PARETO ANALYSIS
# ----------------------------------------------------------

pareto = customer_spend.sort_values(
    "total_spend",
    ascending=False
).copy()

pareto["cumulative_revenue"] = (
    pareto["total_spend"].cumsum()
)

pareto["cumulative_percentage"] = (
    pareto["cumulative_revenue"]
    / pareto["total_spend"].sum()
    * 100
)

pareto["customer_percentage"] = (
    np.arange(1, len(pareto) + 1)
    / len(pareto)
    * 100
)

pareto.to_csv("outputs/pareto_analysis.csv")

plt.figure(figsize=(10, 6))

plt.plot(
    pareto["customer_percentage"],
    pareto["cumulative_percentage"],
    color="darkorange",
    linewidth=2
)

plt.axhline(80, color="red", linestyle="--")

plt.title("Customer Revenue Pareto Analysis")
plt.xlabel("Cumulative Percentage of Customers")
plt.ylabel("Cumulative Percentage of Revenue")

plt.tight_layout()
plt.savefig("charts/pareto_analysis.png", dpi=300)
plt.close()

# ----------------------------------------------------------
# 11. BUSINESS RECOMMENDATIONS
# ----------------------------------------------------------

total_revenue = orders["amount"].sum()
total_orders = len(orders)
total_customers = orders["customer_id"].nunique()

average_order_value = orders["amount"].mean()

recommendations = [
    f"Focus on {best_category}, the highest revenue-generating category.",
    f"Investigate sales performance in {worst_month}, "
    "the lowest revenue month.",
    f"Create loyalty campaigns for the top 10% of customers, "
    f"who contribute {top_10_revenue_share:.1f}% of revenue.",
    "Develop targeted promotions for Low Value and Regular "
    "customers to encourage repeat purchases."
]

# ----------------------------------------------------------
# 12. GENERATE BUSINESS BRIEF
# ----------------------------------------------------------

with open("BUSINESS_BRIEF.md", "w", encoding="utf-8") as file:

    file.write("# E-commerce Sales & Customer Analysis\n\n")

    file.write("## Context\n")
    file.write(
        "This project analyses 2,000 simulated e-commerce "
        "orders from 2025 across 300 customers and four "
        "product categories.\n\n"
    )

    file.write("## Key Findings\n")

    file.write(
        f"- Total revenue: {total_revenue:,.2f}\n"
    )

    file.write(
        f"- Total orders: {total_orders:,}\n"
    )

    file.write(
        f"- Unique customers: {total_customers:,}\n"
    )

    file.write(
        f"- Average order value: {average_order_value:,.2f}\n"
    )

    file.write(
        f"- Best revenue month: {best_month}\n"
    )

    file.write(
        f"- Lowest revenue month: {worst_month}\n"
    )

    file.write(
        f"- Best category: {best_category}\n"
    )

    file.write(
        f"- Top 10% customer revenue contribution: "
        f"{top_10_revenue_share:.2f}%\n\n"
    )

    file.write("## Recommendations\n")

    for recommendation in recommendations:
        file.write(f"- {recommendation}\n")

    file.write("\n## Next Steps\n")
    file.write(
        "- Validate the findings with real, authorised sales data.\n"
        "- Monitor revenue and customer retention monthly.\n"
        "- Test loyalty offers and category-specific campaigns.\n"
        "- Track campaign results against a baseline.\n\n"
    )

    file.write(
        "**Data note:** This analysis uses generated practice "
        "data, not real customer transactions.\n"
    )

# ----------------------------------------------------------
# 13. EXPORT FINAL CLEAN DATASET
# ----------------------------------------------------------

orders.to_csv(
    "outputs/cleaned_ecommerce_data.csv",
    index=False
)

print("\n======================================")
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("======================================")

print("Generated files:")
print("- data/ecommerce_orders.csv")
print("- outputs/monthly_revenue.csv")
print("- outputs/category_analysis.csv")
print("- outputs/monthly_category_pivot.csv")
print("- outputs/customer_analysis.csv")
print("- outputs/customer_segments.csv")
print("- outputs/pareto_analysis.csv")
print("- outputs/cleaned_ecommerce_data.csv")
print("- BUSINESS_BRIEF.md")
print("- charts/monthly_revenue.png")
print("- charts/category_revenue.png")
print("- charts/top_customers.png")
print("- charts/pareto_analysis.png")