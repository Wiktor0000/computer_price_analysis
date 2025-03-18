import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Wczytywanie df
df = pd.read_csv(r"C:\Users\wmusi\OneDrive\Pulpit\computer_price_data_set\laptop_price_dataset.csv")

# Sprawdzanie danych
df.info()
missing_values = df.isnull().sum()
print(f"Brakujące dane:\n{missing_values}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
print(df.head(3))
print(list(df.columns))

# Zmiana nazw kolumn
df.columns=["company", "product", "type_name", "inch", "screen_resolution", "cpu_company", "cpu_type",
            "cpu_frequency(ghz)", "ram(gb)", "memory", "gpu_company", "gpu_type", "system",
            "weight(kg)", "price(euro)"]
df["ram(gb)"]=df["ram(gb)"].astype(int)

# Test spearmana
spearman_corr=df[["inch", "cpu_frequency(ghz)", "ram(gb)", "weight(kg)",
                  "price(euro)"]].corr(method="spearman")
plt.figure(figsize=(10,6))
sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.yticks(rotation=45)
plt.show()