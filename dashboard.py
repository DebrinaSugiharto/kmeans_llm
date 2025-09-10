import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq  # pastikan package ini sudah diinstall

# --- Load API Key ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Init LLM ---
llm = ChatGroq(
    temperature=0.7,
    model_name="llama-3.3-70b-versatile",
    groq_api_key=GROQ_API_KEY
)

# --- Upload CSV ---
def upload_csv():
    uploaded_file = st.file_uploader("📂 Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File berhasil diupload!")
            return df
        except Exception as e:
            st.error(f"❌ Gagal membaca file CSV: {e}")
            return None
    else:
        st.info("Silakan upload file CSV untuk mulai.")
        return None

# --- KMeans dengan UMAP ---
def kmeans_umap_clustering(df):
    st.subheader("⚡ KMeans Clustering dengan UMAP")

    num_cols = df.select_dtypes(include=["float", "int"]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Data harus punya minimal 2 kolom numerik untuk clustering.")
        return None, None, None

    # Input jumlah cluster
    k = st.slider("Pilih jumlah cluster (k)", min_value=2, max_value=10, value=3)

    # Reduksi dimensi dengan UMAP
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(df[num_cols])

    # Jalankan KMeans pada hasil UMAP
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding)

    # Tambahkan hasil ke dataframe
    df["cluster"] = cluster_labels
    df["umap_x"] = embedding[:, 0]
    df["umap_y"] = embedding[:, 1]

    # Hitung silhouette score
    score = silhouette_score(embedding, cluster_labels)

    # Tampilkan hasil clustering
    st.write("📋 Data dengan label cluster:")
    st.dataframe(df.head())

    st.success(f"📐 Silhouette Score: **{score:.3f}**")

    # Visualisasi UMAP scatter plot
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df, x="umap_x", y="umap_y",
        hue="cluster", palette="Set2", s=60, ax=ax
    )
    plt.title(f"UMAP + KMeans Clustering (k={k})")
    st.pyplot(fig)

    # Profiling cluster
    st.subheader("📑 Profiling Cluster")
    cluster_profile = df.groupby("cluster")[num_cols].agg(["mean", "std", "min", "max", "count"])
    st.dataframe(cluster_profile)

    return df, cluster_profile, score

# --- Insight dengan LLM Groq ---
def generate_insight_with_groq(cluster_profile, silhouette):
    prompt = f"""
    Berikut adalah hasil profiling cluster dari data yang diupload:

    {cluster_profile.to_string()}

    Silhouette Score clustering = {silhouette:.3f}

    Tolong jelaskan insight yang bisa diambil dari hasil clustering ini dalam bahasa Indonesia.
    - Apa karakteristik utama tiap cluster
    - Bagaimana kualitas cluster dilihat dari silhouette score
    """

    response = llm.invoke(prompt)
    return response.content

# --- Main App ---
if __name__ == "__main__":
    st.title("Dashboard Clustering (UMAP + KMeans + Profiling + Insight LLM) by Debrina SDS")

    df = upload_csv()
    if df is not None:
        df_clustered, cluster_profile, silhouette = kmeans_umap_clustering(df)

        if cluster_profile is not None:
            st.subheader("💡 Insight Otomatis dari LLM Groq")
            insight = generate_insight_with_groq(cluster_profile, silhouette)
            st.write(insight)
