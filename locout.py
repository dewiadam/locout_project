import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from io import BytesIO

# ==============================
# Styling Custom
# ==============================
st.markdown(
    """
    <style>
    /* Header bar */
    .header-container {
        display: flex;
        align-items: center;
        background-color: #d32f2f;
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }
    .header-icon {
        width: 28px;
        height: 28px;
        margin-right: 10px;
        margin-top: -8px; /* naikkan ikon */
    }
    .header-text {
        display: flex;
        flex-direction: column;
        line-height: 1; /* rapatkan judul dan deskripsi */
    }
    .header-title {
        font-size: 28px;
        font-weight: bold;
        margin: 0;
        color: white;
    }
    .header-subtitle {
        font-size: 15px;
        margin: 2px 0 0 0; /* jarak kecil saja */
        color: #fce4ec;
    }
    
    /* Gaya umum */
    .stButton>button {
        background-color: #d32f2f;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #9a0007;
        color: #fff;
    }
    div[data-baseweb="select"] {
        max-width: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==============================
# Header
# ==============================
st.markdown(
    """
    <div class="header-container">
        <!-- Ikon pin lokasi merah -->
        <svg class="header-icon" xmlns="http://www.w3.org/2000/svg" fill="#ffffff" viewBox="0 0 24 24">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7zm0 
            9.5c-1.38 0-2.5-1.12-2.5-2.5S10.62 6.5 12 6.5s2.5 1.12 
            2.5 2.5S13.38 11.5 12 11.5z"/>
        </svg>
        <div class="header-text">
            <h1 class="header-title">LocOut</h1>
            <p class="header-subtitle">Deteksi Anomali Pemetaan Outlet</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Fungsi bantu
# ==============================
def to_radians(coords):
    return np.radians(coords)

def detect_anomaly(file, radius_meter):
    df = pd.read_excel(file, engine='openpyxl')
    coords = df[['LATITUDE', 'LONGITUDE']].values

    kms_per_radian = 6371.0088
    epsilon = (radius_meter / 1000) / kms_per_radian

    db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
    labels = db.fit_predict(to_radians(coords))

    df['cluster'] = labels
    clusters_suspicious = df[df['cluster'] != -1].groupby('cluster').filter(lambda x: len(x) >= 5)

    return df, clusters_suspicious

# ==============================
# UI Streamlit
# ==============================
uploaded_file = st.file_uploader("📂 Upload file Excel dataset outlet", type=["xlsx"])

radius_options = {
    "100 meter": 100,
    "50 meter": 50,
    "10 meter": 10
}

radius_label = st.selectbox("🎯 Pilih radius deteksi:", list(radius_options.keys()))
radius_value = radius_options[radius_label]

proses = st.button("🚀 Proses Deteksi", use_container_width=True)

# Jalankan deteksi
if uploaded_file and proses:
    full_df, result_df = detect_anomaly(uploaded_file, radius_value)
    st.success(f"✅ Deteksi selesai dengan radius {radius_label}.")

    total_points = len(result_df)
    total_clusters = result_df['cluster'].nunique()

    st.subheader("📊 Ringkasan Deteksi")
    st.write(f"Jumlah titik anomali terdeteksi: **{total_points}**")
    st.write(f"Jumlah cluster terdeteksi: **{total_clusters}**")

    if not result_df.empty:
        cluster_summary = (
            result_df.groupby('cluster')
            .agg(
                jumlah_titik=('cluster', 'size'),
                lat_rata2=('LATITUDE', 'mean'),
                lon_rata2=('LONGITUDE', 'mean')
            )
            .reset_index()
        )
        st.write("📍 Ringkasan cluster yang terdeteksi:")
        st.dataframe(cluster_summary)
    else:
        st.warning("Tidak ada cluster yang memenuhi syarat.")

    st.subheader("📑 Hasil Deteksi Detail")
    st.dataframe(result_df)

  
    # Simpan hasil ke Excel (tanpa ringkasan cluster)
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result_df.to_excel(writer, index=False, sheet_name='Hasil Deteksi')
    output.seek(0)

    st.download_button(
        label="💾 Download Hasil Deteksi",
        data=output,
        file_name=f"hasil_deteksi_anomali_outlet_{radius_label}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



