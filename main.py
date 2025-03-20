import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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

# Klasteryzacja kmeans
X=df[["cpu_frequency(ghz)", "ram(gb)"]].values
km=KMeans(n_clusters=3,
          init="random",
          n_init=10,
          max_iter=300,
          tol=1e-04,
          random_state=0)
y_km = km.fit_predict(X)

plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c="lightgreen",
            marker="s", edgecolor="black",
            label="cluster 1")
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c="orange",
            marker="o", edgecolor="black",
            label="cluster 2")
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c="lightblue",
            marker="v", edgecolor="black",
            label="cluster 3")
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker="*",
            c="red", edgecolor="black",
            label="centroids")
plt.legend(scatterpoints=1)
plt.grid()
plt.xlabel("Cpu Frequency (GHz")
plt.ylabel("Ram (GB)")
plt.show()

df["cluster"]=y_km+1
print(df.groupby("cluster")[["cpu_frequency(ghz)", "ram(gb)"]].mean())
print(df.groupby("cluster")[["cpu_frequency(ghz)", "ram(gb)"]].std())

#Rozkład cen
plt.figure(figsize=(10, 6))
plt.hist(df["price(euro)"], bins=30, color="skyblue", edgecolor="black")
plt.xlabel("Cena (euro)")
plt.ylabel("Liczba laptopów")
plt.title("Rozkład cen laptopów")
plt.grid()
plt.show()

#Cena a ilość RAM
plt.figure(figsize=(10,6))
plt.scatter(df["ram(gb)"], df["price(euro)"], color="skyblue", alpha=0.7)
plt.xlabel("Ram (GB)")
plt.ylabel("Cena (euro)")
plt.title("Cena vs RAM")
plt.grid()
plt.show()