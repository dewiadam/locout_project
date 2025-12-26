import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
from io import BytesIO
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from math import radians, cos, sin, asin, sqrt
import folium
from streamlit_folium import st_folium
from k_means_constrained import KMeansConstrained
import io
import plotly.express as px
import math
from scipy.optimize import linear_sum_assignment # Library tambahan untuk pemetaan unik

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
    min-width: 270px !important;   /* default sekitar 250px */
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
    ["🔎 Deteksi Anomali Outlet", "🛰️ Deteksi Coverage Outlet", "🗺️ Mapping PJP", "🗺️ Optimasi Rute PJP"],
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

        try:
        # pastikan semua nilai bisa diubah ke float
            lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        except (ValueError, TypeError):
            # jika ada nilai aneh, kembalikan NaN supaya tidak error
            return np.nan
        
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
# MENU 3 - MAPPING PJP (OPTIMIZED)
# =====================================
def page_mapping():
    header("Sales Territory Mapping Optimizer")
    
    # Initialize session state for Mapping
    if "mapping_processed" not in st.session_state:
        st.session_state.mapping_processed = False

    uploaded_file = st.file_uploader(
        "📂 Upload Dataset Excel (id_outlet, nama_outlet, lon_outlet, lat_outlet, nama_sf)",
        type=["xlsx"], key="mapping_upload"
    )

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        required_cols = ["id_outlet", "nama_outlet", "lon_outlet", "lat_outlet", "nama_sf"]
        
        if not all(col in df.columns for col in required_cols):
            st.error(f"Kolom wajib: {', '.join(required_cols)}")
            return

        counts_all = df["nama_sf"].value_counts().reset_index()
        counts_all.columns = ["nama_sf", "Qty Awal"]
        sf_utama = counts_all[counts_all["Qty Awal"] >= 10]["nama_sf"].tolist()
        sf_minor = counts_all[counts_all["Qty Awal"] < 10]["nama_sf"].tolist()

        st.subheader("📊 Ringkasan Data Input")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Outlet", len(df))
        c2.metric("Total Sales Force", len(counts_all))
        c3.metric("Rata-rata Outlet/SF", round(len(df)/len(counts_all),1))

        # Parameter
        MIN_CAP, MAX_CAP = 75, 80
        num_clusters = len(sf_utama)
        actual_min = min(MIN_CAP, math.floor(len(df) / num_clusters)) if num_clusters > 0 else 0

        if st.button("⚡ Jalankan Optimasi & Mapping", use_container_width=True):
            with st.spinner("Sedang menghitung territory sales..."):
                try:
                    coords = df[["lat_outlet", "lon_outlet"]].values
                    model = KMeansConstrained(n_clusters=num_clusters, size_min=actual_min, size_max=MAX_CAP, random_state=42)
                    df["cluster_label"] = model.fit_predict(coords)

                    # Hungarian Assignment
                    cluster_labels = sorted(df["cluster_label"].unique())
                    cost_matrix = [[-df[df["cluster_label"] == label]["nama_sf"].value_counts().get(sf, 0) for sf in sf_utama] for label in cluster_labels]
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)
                    mapping = {cluster_labels[row_ind[i]]: sf_utama[col_ind[i]] for i in range(len(row_ind))}
                    
                    df["sf_baru"] = df["cluster_label"].map(mapping)
                    new_counts = df.groupby("sf_baru").size().reset_index(name="Qty Baru")
                    summary = counts_all.rename(columns={"nama_sf": "sf_baru"}).merge(new_counts, on="sf_baru", how="left").fillna(0)
                    summary["Qty Baru"] = summary["Qty Baru"].astype(int)

                    st.session_state.mapping_result = df.copy()
                    st.session_state.mapping_summary = summary.copy()
                    st.session_state.mapping_minor = sf_minor
                    st.session_state.mapping_processed = True

                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        df.drop(columns=["cluster_label"]).to_excel(writer, index=False, sheet_name="Detail Mapping")
                        summary.to_excel(writer, index=False, sheet_name="Summary SF")
                    st.session_state.mapping_excel = output.getvalue()
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")

        if st.session_state.mapping_processed:
            st.subheader("🗺️ Visualisasi Territory Baru")
            fig = px.scatter_mapbox(st.session_state.mapping_result, lat="lat_outlet", lon="lon_outlet", color="sf_baru", hover_name="nama_outlet", zoom=8, height=500)
            fig.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📊 Ringkasan Distribusi")
            st.dataframe(st.session_state.mapping_summary, use_container_width=True)

            if st.session_state.mapping_minor:
                st.warning(f"SF Minor dialihkan: {', '.join(st.session_state.mapping_minor)}")

            st.download_button("📩 Download Hasil Mapping", st.session_state.mapping_excel, file_name="territory_mapping_final.xlsx")

# =====================================
# MENU 4 - OPTIMASI RUTE PJP
# =====================================
def distance_to_kantor(lat, lon, kantor_coord):
    return geodesic((lat, lon), kantor_coord).km

def create_distance_matrix(coords):
    n = len(coords)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = geodesic(coords[i], coords[j]).km
    return matrix

def solve_tsp(distance_matrix):
    n = len(distance_matrix)
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = 5

    solution = routing.SolveWithParameters(search_params)

    if not solution:
        return list(range(n))

    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    return route

# ================= PAGE =================
def page_rute():
    st.header("Optimasi Rute Kunjungan SF")

    uploaded_file = st.file_uploader("📂 Upload file Outlet PJP", type=["xlsx", "csv"])

    with st.expander("🏢 Lokasi Kantor", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            kantor_lat = st.number_input("Latitude Kantor", value=-6.200000, format="%.6f")
        with col2:
            kantor_lon = st.number_input("Longitude Kantor", value=106.816666, format="%.6f")

    urutan_rute = st.radio(
        "🔁 Urutan Awal Kunjungan",
        ["Terdekat dari Kantor", "Terjauh dari Kantor"],
        horizontal=True
    )

    proses = st.button("🚀 Proses Data")

    if proses:
        if uploaded_file is None:
            st.warning("⚠️ Harap upload file outlet terlebih dahulu.")
            return

        df = (
            pd.read_csv(uploaded_file)
            if uploaded_file.name.endswith(".csv")
            else pd.read_excel(uploaded_file)
        )

        # Pastikan kolom sesuai dengan file yang diupload
        required_cols = {"lat_outlet", "lon_outlet", "nama_sf"}
        if not required_cols.issubset(df.columns):
            st.error(f"❌ File harus memiliki kolom: {', '.join(required_cols)}")
            return

        kantor_coord = (kantor_lat, kantor_lon)
        hasil_list = []
        sf_list = df["nama_sf"].unique()

        progress = st.progress(0)

        # Mapping Nama Hari
        list_hari = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]

        for i, sf in enumerate(sf_list):
            df_sf = df[df["nama_sf"] == sf].copy()

            # ================= HITUNG JARAK KE KANTOR =================
            df_sf["jarak_kantor_km"] = df_sf.apply(
                lambda r: distance_to_kantor(
                    r.lat_outlet, r.lon_outlet, kantor_coord
                ),
                axis=1
            )

            # ================= SORT SESUAI PILIHAN =================
            ascending = True if urutan_rute == "Terdekat dari Kantor" else False
            df_sf = df_sf.sort_values(
                "jarak_kantor_km", ascending=ascending
            ).reset_index(drop=True)

            # ================= TSP =================
            coords = [kantor_coord] + list(
                zip(df_sf["lat_outlet"], df_sf["lon_outlet"])
            )
            dist_matrix = create_distance_matrix(coords)
            route_idx = solve_tsp(dist_matrix)

            route_outlet = [idx - 1 for idx in route_idx if idx != 0]
            route_df = df_sf.iloc[route_outlet].copy()

            # ================= PENENTUAN HARI (TEKS) =================
            route_df["urutan_kunjungan"] = np.arange(1, len(route_df) + 1)
            
            # Hitung indeks hari (15 outlet per hari)
            angka_hari = ((route_df["urutan_kunjungan"] - 1) // 15) + 1
            
            # Mengubah angka hari menjadi Nama Hari (Senin, Selasa, dst)
            route_df["hari_ke"] = angka_hari.apply(lambda x: list_hari[(x-1) % 7])
            
            route_df["urutan_harian"] = (
                (route_df["urutan_kunjungan"] - 1) % 15
            ) + 1
            route_df["nama_sf"] = sf

            hasil_list.append(route_df)
            progress.progress((i + 1) / len(sf_list))

        # ================= FINAL OUTPUT =================
        df_hasil = pd.concat(hasil_list, ignore_index=True)

        st.subheader("📊 Hasil Rute Kunjungan")
        st.dataframe(df_hasil, use_container_width=True)

        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_hasil.to_excel(writer, index=False, sheet_name="Rute_SF_Optimized")

        st.download_button(
            "📥 Download Hasil (Excel)",
            data=output.getvalue(),
            file_name="pjp_rute_sf_optimized.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# =====================================
# HALAMAN UTAMA SESUAI MENU
# =====================================
if "Anomali" in menu:
    page_anomali()
elif "Coverage" in menu:
    page_coverage()
elif "Mapping" in menu:
    page_mapping()
else:
    page_rute()




