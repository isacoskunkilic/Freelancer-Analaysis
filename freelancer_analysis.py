import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyparsing import conditionAsParseAction  # Unused import, kept as-is per instruction

# Display full DataFrame output in console
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_colwidth", None)

# Freelancer profile data
freelancers_data = {
    "freelancer_id": ["F001", "F002", "F003", "F004", "F005", "F006", "F007", "F008", "F009", "F010"],
    "name": ["Lina Smith", "Jorge López", "Ayşe Kılıç", "Tom Becker", "Chen Li", "Sara Rossi", "Ali Can", "Emily Zhang", "Carlos Reyes", "Anna Ivanova"],
    "country": ["Germany", "Mexico", "Turkey", "USA", "China", "Italy", "Turkey", "China", "Mexico", "Russia"],
    "job_title": ["Data Scientist", "Web Developer", "AI Engineer", "Data Analyst", "ML Engineer", "UX Designer", "Data Scientist", "Web Developer", "Data Analyst", "AI Engineer"],
    "hourly_rate": [55, 30, 70, 45, 60, 40, 50, 35, 38, 65],
    "total_hours": [400, 300, 350, 420, 380, 310, 360, 290, 330, 370],
    "rating": [4.9, 4.5, 4.8, 4.2, 4.7, 4.3, 4.6, 4.4, 4.1, 4.9]
}

freelancers_df = pd.DataFrame(freelancers_data)

# Project performance and income data
projects_data = {
    "project_id": ["P101", "P102", "P103", "P104", "P105", "P106", "P107", "P108", "P109", "P110", "P111", "P112", "P113", "P114", "P115"],
    "freelancer_id": ["F001", "F002", "F001", "F003", "F004", "F005", "F006", "F007", "F008", "F009", "F010", "F003", "F005", "F002", "F009"],
    "project_type": ["Data Cleaning", "Web Redesign", "ML Deployment", "AI Automation", "Data Viz", "ML Training", "UX Audit", "Data Scraping", "Frontend Fix", "Reporting", "AI Audit", "Deep Learning", "Model Optimization", "Landing Page", "Backend Refactor"],
    "duration_days": [10, 15, 25, 30, 14, 20, 12, 16, 8, 10, 28, 18, 22, 14, 13],
    "client_country": ["USA", "UK", "France", "Germany", "USA", "Italy", "China", "Turkey", "China", "Mexico", "Russia", "UK", "France", "USA", "Brazil"],
    "income": [1200, 1500, 2500, 2800, 1100, 2000, 1300, 1600, 900, 950, 2700, 2400, 1900, 1450, 980]
}

projects_df = pd.DataFrame(projects_data)

# Sort projects by income in descending order
projects_df.sort_values("income", ascending=False, inplace=True)

# Get freelancer_ids of top 5 income earners
top_earner_ids = projects_df.nlargest(5, "income")["freelancer_id"].tolist()

# Filter freelancers who are in the top earning list
top_earners_df = freelancers_df[freelancers_df["freelancer_id"].isin(top_earner_ids)]
print("Top earning freelancers:")
print(top_earners_df)

# Assign first 10 incomes from project list to freelancers (mock assumption)
income_list = projects_df["income"][0:10].tolist()
freelancers_df["income"] = income_list

# Calculate country-based average income
country_mean_income = freelancers_df.groupby(["country"])["income"].mean().sort_values(ascending=False)
print("\nAverage income by country:")
print(country_mean_income)

# Calculate expected earnings based on rate and hours
freelancers_df["expected_earning"] = freelancers_df["hourly_rate"] / freelancers_df["total_hours"]
print("\nExpected earnings (hourly_rate / total_hours):")
print(freelancers_df[["freelancer_id", "expected_earning"]])

# Correlation between project duration and income
duration_income_corr = projects_df["duration_days"].corr(projects_df["income"])
print(f"\nCorrelation between duration and income: {duration_income_corr:.2f}")

# Count of each project type (popularity)
project_type_count = projects_df.groupby("project_type")["project_type"].count()
print("\nProject type popularity:")
print(project_type_count)

# Maximum income by project type
project_type_max_income = projects_df.groupby("project_type")["income"].max().sort_values(ascending=False)
print("\nTop earning project types:")
print(project_type_max_income)

# Correlation between expected earning and actual income
expected_actual_corr = freelancers_df["expected_earning"].corr(freelancers_df["income"])
print(f"\nCorrelation between expected earning and actual income: {expected_actual_corr:.2f}")

# Average income by job title and country
job_country_income = freelancers_df.groupby(["country", "job_title"])["income"].mean()
print("\nAverage income by country and job title:")
print(job_country_income)

# Filter high-rated freelancers
high_rated_df = freelancers_df.query("rating > 4.5")

# Count by country and job title for high-rated freelancers
high_rating_by_country = high_rated_df.groupby("country").size()
high_rating_by_job = high_rated_df.groupby("job_title").size()

print("\nHigh-rated freelancers per country:")
print(high_rating_by_country)

print("\nHigh-rated freelancers per job title:")
print(high_rating_by_job)

# Compare income to average
avg_income = freelancers_df["income"].mean()
conditions = [
    high_rated_df["income"] > avg_income,
    high_rated_df["income"] < avg_income
]
labels = ["Higher", "Lower"]

# Add a new column showing if freelancer earns more/less than average
high_rated_df["mean_income_diff"] = np.select(conditions, labels, default="NAN")
print("\nComparison to average income (Higher / Lower):")
print(high_rated_df[["freelancer_id", "income", "mean_income_diff"]])
import seaborn as sns
import matplotlib.pyplot as plt
country_income_df = country_mean_income.reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=country_income_df, x="country", y="income", estimator=np.mean, ci="sd", palette="viridis")
plt.title("Average Income per Country")
plt.xlabel("Country")
plt.ylabel("Average Income ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



# Convert Series to DataFrame
project_income_df = project_type_max_income.reset_index()

# Sort the DataFrame (ascending or descending)
project_income_df.sort_values(by="income", ascending=True, inplace=True)

plt.figure(figsize=(14, 6))
sns.lineplot(data=project_income_df, x="project_type", y="income", marker="o", linewidth=2.5, color="purple")
plt.title("Max Income by Project Type (Stylish Lineplot)", fontsize=14, weight="bold")
plt.xlabel("Project Type", fontsize=12)
plt.ylabel("Max Income ($)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

###########################################
# Pivot made again
pivot_df = freelancers_df.groupby(["country", "job_title"])["income"].mean().unstack()

# Plot
plt.figure(figsize=(10, 6))
sns.set_theme(style="white")

ax = sns.heatmap(
    pivot_df,
    annot=True,
    fmt=".0f",
    cmap=sns.diverging_palette(145, 300, s=85, l=40, as_cmap=True),  # Red-green diverging
    linewidths=0.8,
    linecolor="black",
    cbar_kws={"label": "Avg. Income ($)"},
    annot_kws={"fontsize": 10, "weight": "bold", "color": "black"}
)

plt.title("Average Income by Country and Job Title", fontsize=14, weight="bold", pad=15)
plt.xlabel("Job Title", fontsize=12)
plt.ylabel("Country", fontsize=12)
plt.xticks(rotation=30, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()
