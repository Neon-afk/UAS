import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Untuk memuat model yang disimpan

# Memuat model K-Means yang telah disimpan
model_path = 'kmeans_model.pkl'  # Lokasi file model
kmeans = joblib.load(model_path)

# Menyiapkan scaler yang digunakan pada saat pelatihan model
scaler_path = 'scaler.pkl'  # Lokasi file scaler (jika ada)
scaler = joblib.load(scaler_path)  # Memuat scaler jika disimpan terpisah

# Fungsi untuk memprediksi kluster berdasarkan input pengguna
def predict_cluster(age, bmi, glucose_level, family_history, smoker, kmeans, scaler):
    input_data = np.array([[age, bmi, glucose_level, family_history, smoker]])
    
    # Normalisasi input pengguna
    input_scaled = scaler.transform(input_data)
    
    # Prediksi kluster
    cluster = kmeans.predict(input_scaled)
    return cluster[0]

# Fungsi untuk memberikan penjelasan tentang kluster
def cluster_description(cluster_id):
    descriptions = {
        0: "Kluster 0: Individu dengan risiko rendah diabetes. Biasanya memiliki BMI rendah dan kadar glukosa normal.",
        1: "Kluster 1: Individu dengan risiko sedang. Memiliki BMI sedikit lebih tinggi dan glukosa yang sedikit lebih tinggi.",
        2: "Kluster 2: Individu dengan risiko tinggi. Memiliki BMI tinggi dan glukosa tinggi.",
        3: "Kluster 3: Individu yang sangat berisiko, kemungkinan besar memiliki riwayat keluarga diabetes dan pola makan yang kurang sehat.",
    }
    return descriptions.get(cluster_id, "Tidak ada deskripsi untuk kluster ini.")

# Streamlit interface
st.title('Diabetes Risk Clustering Using K-Means')
st.write("Aplikasi ini menggunakan K-Means untuk mengelompokkan individu berdasarkan faktor risiko diabetes.")

# Input data pengguna
st.subheader("Masukkan Data Anda untuk Prediksi Kluster")

age = st.number_input("Usia (Age)", min_value=18, max_value=100, value=30)
bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=50.0, value=22.0)
glucose_level = st.number_input("Tingkat Glukosa (Glucose Level)", min_value=50.0, max_value=250.0, value=100.0)
family_history = st.selectbox("Riwayat Keluarga (Family History)", options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')
smoker = st.selectbox("Merokok (Smoker)", options=[0, 1], format_func=lambda x: 'Tidak' if x == 0 else 'Ya')

# Ketika tombol "Prediksi Kluster" ditekan
if st.button("Prediksi Kluster"):
    cluster_predicted = predict_cluster(age, bmi, glucose_level, family_history, smoker, kmeans, scaler)
    st.write(f"Anda termasuk dalam **Kluster {cluster_predicted}**")

    # Menampilkan penjelasan tentang kluster
    cluster_info = cluster_description(cluster_predicted)
    st.write(f"**Keterangan Kluster {cluster_predicted}:**")
    st.write(cluster_info)

    # Menampilkan pusat kluster
    st.subheader("Pusat Kluster")
    cluster_centers = kmeans.cluster_centers_
    st.write(cluster_centers)

    # Menampilkan rata-rata fitur per kluster
    st.subheader("Rata-rata Fitur per Kluster")
    data_clean = pd.read_csv('data/diabetes_risk_dataset.csv')  # Sesuaikan path file dataset
    data_clean['family_history'] = data_clean['family_history'].astype(int)
    data_clean['smoker'] = data_clean['smoker'].astype(int)
    X = data_clean[['age', 'bmi', 'glucose_level', 'family_history', 'smoker']]

    # Normalisasi data untuk kluster
    X_scaled = scaler.transform(X)
    data_clean['cluster'] = kmeans.predict(X_scaled)
    st.write(data_clean.groupby('cluster')[['age', 'bmi', 'glucose_level', 'family_history', 'smoker']].mean())
