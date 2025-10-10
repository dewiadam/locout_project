import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from io import BytesIO

# =====================================
# STYLE UMUM (HEADER, SIDEBAR, BUTTON)
# =====================================
st.set_page_config(page_title="LocOut", layout="wide")
st.markdown("""
<style>
/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #fff;
}
div[data-testid="stSidebarNav"] ul {
    list-style: none;
    padding: 0;
}
div[data-testid="stSidebarNav"] li {
    background-color: #d32f2f;
    color: white;
    font-size: 22px;          /* diperbesar */
    font-weight: 700;
    padding: 10px 18px;
    border-radius: 10px;
    text-align: center;
    margin: 10px 0 20px 0;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
}
div[data-testid="stSidebarNav"] li:hover {
    background-color: #ffe6e6;
}
div[data-testid="stSidebarNav"] li.active {
    background-color: #d32f2f;
    color: white !important;
}

/* Ubah lebar sidebar */
section[data-testid="stSidebar"] {
    min-width: 300px !important;   /* default sekitar 250px */
    max-width: 320px !important;
}

/* Sesuaikan konten di dalamnya agar tetap rapi */
section[data-testid="stSidebar"] .css-1d391kg, 
section[data-testid="stSidebar"] .css-1lcbmhc {
    padding-left: 10px;
    padding-right: 10px;
}


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
    margin-top: -8px;
}
.header-text {
    display: flex;
    flex-direction: column;
    line-height: 1;
}
.header-title {
    font-size: 28px;
    font-weight: bold;
    margin: 0;
    color: white;
}
.header-subtitle {
    font-size: 15px;
    margin: 2px 0 0 0;
    color: #fce4ec;
}

/* Button */
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
}
</style>
""", unsafe_allow_html=True)


# =====================================
# SIDEBAR MENU
# =====================================
st.sidebar.markdown("""
    <div style="
        background-color: #d32f2f;
        color: white;
        text-align: center;
        font-weight: 700;
        font-size: 20px;
        padding: 12px 0;
        border-radius: 10px;
        margin-bottom: 25px;
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    ">
        LocOut Tools
    </div>
""", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "",
    ["🔎 Deteksi Anomali Outlet", "🛰️ Deteksi Coverage Outlet"],
    label_visibility="collapsed"
)


# =====================================
# HEADER (DINAMIS SESUAI MENU)
# =====================================
def header(subtitle):
    st.markdown(f"""
        <div class="header-container">
            <svg class="header-icon" xmlns="http://www.w3.org/2000/svg" fill="#ffffff" viewBox="0 0 24 24">
                <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 
                7-13c0-3.87-3.13-7-7-7zm0 
                9.5c-1.38 0-2.5-1.12-2.5-2.5S10.62 
                6.5 12 6.5s2.5 1.12 
                2.5 2.5S13.38 11.5 12 11.5z"/>
            </svg>
            <div class="header-text">
                <h1 class="header-title">LocOut</h1>
                <p class="header-subtitle">{subtitle}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


# =====================================
# MENU 1 - DETEKSI ANOMALI OUTLET
# =====================================
def page_anomali():
    header("Deteksi Anomali Pemetaan Outlet")

    def to_radians(coords):
        return np.radians(coords)

    def detect_anomaly(file, radius_meter):
        df = pd.read_excel(file, engine='openpyxl')
        coords = df[['lat_outlet', 'lon_outlet']].values

        kms_per_radian = 6371.0088
        epsilon = (radius_meter / 1000) / kms_per_radian

        db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine')
        labels = db.fit_predict(to_radians(coords))

        df['cluster'] = labels
        clusters_suspicious = df[df['cluster'] != -1].groupby('cluster').filter(lambda x: len(x) >= 5)

        return df, clusters_suspicious

    uploaded_file = st.file_uploader("📂 Upload file Excel dataset outlet", type=["xlsx"])
    radius_options = {"100 meter": 100, "50 meter": 50, "10 meter": 10}
    radius_label = st.selectbox("🎯 Pilih radius deteksi:", list(radius_options.keys()))
    radius_value = radius_options[radius_label]
    proses = st.button("🚀 Proses Deteksi", use_container_width=True)

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
                    lat_rata2=('lat_outlet', 'mean'),
                    lon_rata2=('lon_outlet', 'mean')
                )
                .reset_index()
            )
            st.write("📍 Ringkasan cluster yang terdeteksi:")
            st.dataframe(cluster_summary)
        else:
            st.warning("Tidak ada cluster yang memenuhi syarat.")

        st.subheader("📑 Hasil Deteksi Detail")
        st.dataframe(result_df)

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


# =====================================
# MENU 2 - DETEKSI COVERAGE OUTLET
# =====================================
def page_coverage():
    header("Deteksi Coverage Outlet")

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    site_file = st.file_uploader("📂 Upload file Site (Excel)", type=["xlsx"])
    outlet_file = st.file_uploader("📂 Upload file Outlet (Excel)", type=["xlsx"])
    radius_options = {"1 km": 1.0, "500 meter": 0.5}
    radius_label = st.selectbox("🎯 Pilih Radius Coverage:", list(radius_options.keys()))
    selected_radius = radius_options[radius_label]

    if st.button("🚀 Proses Deteksi", use_container_width=True):
        if site_file and outlet_file:
            sites = pd.read_excel(site_file)
            outlets = pd.read_excel(outlet_file)

            st.subheader("Preview Data Site")
            st.markdown(f"**Total row data site:** {len(sites)}")
            st.dataframe(sites.head())

            st.subheader("Preview Data Outlet")
            st.markdown(f"**Total row data outlet:** {len(outlets)}")
            st.dataframe(outlets.head())

            summary, detail_rows = [], []
            for _, site in sites.iterrows():
                covered_outlets = []
                for _, outlet in outlets.iterrows():
                    jarak = haversine(site["lat_site"], site["lon_site"],
                                      outlet["lat_outlet"], outlet["lon_outlet"])
                    if jarak <= selected_radius:
                        covered_outlets.append(outlet["id_outlet"])
                        detail_rows.append({
                            "id_site": site["id_site"],
                            "lat_site": site["lat_site"],
                            "lon_site": site["lon_site"],
                            "id_outlet": outlet["id_outlet"],
                            "lat_outlet": outlet["lat_outlet"],
                            "lon_outlet": outlet["lon_outlet"],
                            "jarak_km": jarak
                        })

                summary.append({
                    "id_site": site["id_site"],
                    "lat_site": site["lat_site"],
                    "lon_site": site["lon_site"],
                    "jumlah_outlet_tercover": len(covered_outlets),
                    "id_outlet_tercover": ', '.join(map(str, covered_outlets))
                })

            summary_df = pd.DataFrame(summary)
            detail_df = pd.DataFrame(detail_rows)

            st.subheader("📊 Summary Coverage")
            st.dataframe(summary_df)
            st.subheader("📋 Detail Site - Outlet Coverage")
            st.dataframe(detail_df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                summary_df.to_excel(writer, index=False, sheet_name="Summary")
                detail_df.to_excel(writer, index=False, sheet_name="Detail")

            st.download_button(
                "💾 Download Hasil (Summary + Detail)",
                data=output.getvalue(),
                file_name=f"coverage_result_{radius_label.replace(' ','_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# =====================================
# HALAMAN UTAMA SESUAI MENU
# =====================================
if "Anomali" in menu:
    page_anomali()
else:
    page_coverage()
