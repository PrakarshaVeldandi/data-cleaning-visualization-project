import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load Dataset
housing = fetch_california_housing()

df = pd.DataFrame(
    housing.data,
    columns=housing.feature_names
)

df["HouseValue"] = housing.target

# Missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Outlier Removal
for column in df.columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[
        (df[column] >= lower) &
        (df[column] <= upper)
    ]

# Visualization 1
plt.figure(figsize=(8,5))
plt.hist(df["HouseValue"])
plt.title("House Value Distribution")
plt.savefig("../images/house_distribution.png")
plt.show()

# Visualization 2
plt.figure(figsize=(8,5))
plt.scatter(df["MedInc"],df["HouseValue"])
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.title("Income vs House Value")
plt.savefig("../images/income_vs_housevalue.png")
plt.show()

# Visualization 3
corr = df.corr()

plt.figure(figsize=(10,8))
plt.imshow(corr)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Heatmap")

plt.savefig("../images/correlation_heatmap.png")
plt.show()

print("Project Completed")
