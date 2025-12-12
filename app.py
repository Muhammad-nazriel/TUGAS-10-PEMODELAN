import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from euler_model import logistic_euler

# Konfigurasi halaman
st.set_page_config(
    page_title="TA-09 Simulasi Pertumbuhan Internet",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
    <style>
        .header-title {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .header-subtitle {
            text-align: center;
            color: #555;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin: 10px 0;
        }
        .info-box {
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #1f77b4;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header-title">🌐 TA-09 Simulasi Pertumbuhan Pengguna Internet Global</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Model Logistik + Metode Euler (Pemodelan & Simulasi)</div>', unsafe_allow_html=True)

# Load dataset
df = pd.read_csv("dataset/number-of-internet-users.csv")
df = df[["Entity", "Year", "Number of Internet users"]]
df = df[df["Entity"] == "World"]

years = df["Year"].values
users = df["Number of Internet users"].values

U0 = users[0]

# Sidebar untuk parameter
st.sidebar.markdown("### ⚙️ Pengaturan Parameter Model")
st.sidebar.markdown("""
Sesuaikan parameter simulasi untuk melihat bagaimana model logistik 
berkembang dengan parameter berbeda.
""")

r = st.sidebar.slider("📈 Laju Pertumbuhan (r)", 0.01, 1.0, 0.1, step=0.01)
K = st.sidebar.slider("📊 Kapasitas Maksimum (K)", int(max(users)), int(max(users)*3), int(max(users)*1.2), step=int(max(users)*0.1))
h = st.sidebar.slider("⏱️ Step Size (h)", 0.01, 1.0, 0.1, step=0.01)

# Informasi parameter
st.sidebar.markdown('<div class="info-box"><strong>ℹ️ Informasi Parameter:</strong><br>' + 
                    f'• <strong>r:</strong> Laju pertumbuhan populasi<br>' +
                    f'• <strong>K:</strong> Kapasitas maksimum lingkungan<br>' +
                    f'• <strong>h:</strong> Ukuran langkah integrasi Euler</div>', 
                    unsafe_allow_html=True)

# Jalankan simulasi
t, U_sim = logistic_euler(U0, r, K, h, t_end=len(users))

# Hitung metrics
mse = np.mean((users - U_sim[:len(users)])**2)
mae = np.mean(np.abs(users - U_sim[:len(users)]))
rmse = np.sqrt(mse)

# Tampilkan metrics dalam kolom
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📉 MSE (Mean Squared Error)", f"{mse:.2e}")

with col2:
    st.metric("📊 MAE (Mean Absolute Error)", f"{mae:.2e}")

with col3:
    st.metric("√ RMSE (Root Mean Squared Error)", f"{rmse:.2e}")

# Section: Data Visualization
st.markdown("---")
st.markdown("### 📊 Visualisasi Data & Simulasi")

tab1, tab2 = st.tabs(["📈 Grafik Perbandingan", "📋 Data Table"])

with tab1:
    # Plot dengan styling yang lebih baik
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(years, users, marker='o', linewidth=2.5, markersize=6, 
            label="📍 Data Asli", color='#1f77b4')
    ax.plot(years, U_sim[:len(users)], linewidth=2.5, linestyle='--', 
            label="🔮 Simulasi Euler", color='#ff7f0e')
    
    ax.fill_between(years, users, U_sim[:len(users)], alpha=0.2, color='gray')
    
    ax.set_xlabel("Tahun", fontsize=12, fontweight='bold')
    ax.set_ylabel("Pengguna Internet", fontsize=12, fontweight='bold')
    ax.set_title("Perbandingan Data Asli vs Simulasi Euler\n(Model Logistik Pertumbuhan Internet Global)", 
                 fontsize=13, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')
    
    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.1f}B' if x >= 1e9 else f'{x/1e6:.0f}M'))
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    # Buat dataframe untuk ditampilkan
    comparison_df = pd.DataFrame({
        'Tahun': years,
        'Data Asli': users.astype(int),
        'Simulasi Euler': U_sim[:len(users)].astype(int),
        'Error Absolut': np.abs(users - U_sim[:len(users)]).astype(int)
    })
    
    st.dataframe(comparison_df, use_container_width=True)

# Section: Analisis
st.markdown("---")
st.markdown("### 📋 Analisis & Kesimpulan")

# Info box
st.markdown(f"""
<div class="info-box">
    <strong>📊 Hasil Simulasi:</strong><br>
    <ul>
        <li><strong>Nilai Awal (U₀):</strong> {U0:,.0f} pengguna</li>
        <li><strong>Kapasitas Maksimum (K):</strong> {K:,.0f} pengguna</li>
        <li><strong>Laju Pertumbuhan (r):</strong> {r}</li>
        <li><strong>Periode:</strong> {int(years[0])} - {int(years[-1])} ({len(years)} tahun)</li>
        <li><strong>Error (MSE):</strong> {mse:.4e}</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
**Kesimpulan:**
- Model logistik menggambarkan pertumbuhan populasi yang dibatasi oleh kapasitas lingkungan.
- Persamaan diferensial: **dU/dt = rU(1 - U/K)**
- Metode Euler memberikan aproksimasi numerik dari solusi analitik.
- Tuning parameter `r` dan `K` dapat meningkatkan akurasi simulasi terhadap data aktual.
- Error yang dipantau membantu mengevaluasi kesesuaian model dengan data observasi.
""")
