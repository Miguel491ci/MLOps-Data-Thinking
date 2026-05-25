import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    N_CLUSTERS,
    RANDOM_STATE,
    SEGMENTED_DATA_PATH,
    FIGURES_DIR,
    METRICS_DIR,
)
from src.data import preprocess_data
from src.features import get_features, scale_features

def create_customer_segments():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = preprocess_data()
    X = get_features(df)
    X_scaled, scaler = scale_features(X)

    inertias = []
    silhouette_scores = []
    cluster_range = range(2, 9)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), inertias, marker="o")
    plt.title("Método del codo para seleccionar número de clusters")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elbow_method.png")
    plt.close()



    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), silhouette_scores, marker="o")
    plt.title("Silhouette score por número de clusters")
    plt.xlabel("Número de clusters")
    plt.ylabel("Silhouette score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "silhouette_scores.png")
    plt.close()

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df["customer_segment"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)
    df["pca_1"] = components[:, 0]
    df["pca_2"] = components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="pca_1",
        y="pca_2",
        hue="customer_segment",
        palette="Set2",
    )
    plt.title("Segmentos de clientes visualizados con PCA")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "customer_segments_pca.png")
    plt.close()

    segment_summary = df.groupby("customer_segment").agg({
        "age": "mean",
        "annual_income_k$": "mean",
        "spending_score_1_100": "mean",
        "gender": "mean",
    }).reset_index()

    segment_summary.to_csv(METRICS_DIR / "segment_summary.csv", index=False)

    SEGMENTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEGMENTED_DATA_PATH, index=False)

    print("Segmentos creados correctamente.")
    print(f"Dataset segmentado guardado en: {SEGMENTED_DATA_PATH}")
    print(f"Gráficas guardadas en: {FIGURES_DIR}")
    print(segment_summary)


if __name__ == "__main__":
    create_customer_segments()