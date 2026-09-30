import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Dashboard PDS - Prediksi Mahasiswa Baru",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# STYLE - dibuat mirip dashboard contoh
# ============================================================
st.markdown("""
<style>
.stApp {
    background: #071326;
    color: #f8fafc;
}

.block-container {
    padding: 1.1rem 1.5rem 2rem 1.5rem;
    max-width: 1600px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06101f 0%, #0b1930 100%);
    border-right: 1px solid #203454;
}

[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}

.hero {
    background: linear-gradient(115deg, #0b1730 0%, #142b54 52%, #101f3e 100%);
    border: 1px solid #27456f;
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 18px;
    box-shadow: 0 12px 35px rgba(0,0,0,.25);
}

.hero h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 850;
}

.hero p {
    margin: 5px 0 0;
    color: #9fc4ff;
    font-size: 16px;
}

.hero small {
    display: block;
    color: #cbd5e1;
    margin-top: 10px;
}

.card {
    background: linear-gradient(145deg, #0d1c35, #0b1730);
    border: 1px solid #28456f;
    border-radius: 16px;
    padding: 17px 18px;
    min-height: 112px;
    box-shadow: 0 9px 24px rgba(0,0,0,.18);
}

.card-blue { border-top: 3px solid #3b82f6; }
.card-green { border-top: 3px solid #10b981; }
.card-purple { border-top: 3px solid #a855f7; }
.card-orange { border-top: 3px solid #f59e0b; }
.card-cyan { border-top: 3px solid #06b6d4; }

.label {
    color: #a9bad3;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .4px;
}

.value {
    color: #ffffff;
    font-size: 28px;
    font-weight: 850;
    margin: 7px 0 3px;
}

.sub {
    color: #91a4bf;
    font-size: 11px;
}

.panel {
    background: #0a1830;
    border: 1px solid #243f67;
    border-radius: 16px;
    padding: 14px 16px;
    margin-top: 14px;
}

.panel-title {
    color: #f8fafc;
    font-size: 17px;
    font-weight: 800;
    margin-bottom: 8px;
}

.prediction {
    background: linear-gradient(135deg, #182d72, #31206f);
    border: 1px solid #6554c0;
    border-radius: 16px;
    padding: 19px;
    min-height: 170px;
}

.prediction .year {
    color: #c4b5fd;
    font-size: 13px;
    font-weight: 750;
}

.prediction .number {
    color: white;
    font-size: 40px;
    font-weight: 900;
    margin: 3px 0;
}

.prediction .desc {
    color: #cbd5e1;
    font-size: 12px;
}

.eval {
    background: linear-gradient(135deg, #0c2631, #0a1e2c);
    border: 1px solid #145e68;
    border-radius: 16px;
    padding: 18px;
}

.eval h3 {
    color: #5eead4;
    margin: 0 0 12px;
    font-size: 17px;
}

.eval-item {
    background: #0d2139;
    border: 1px solid #24466e;
    border-radius: 11px;
    padding: 10px;
    text-align: center;
}

.eval-name {
    color: #9fb2ca;
    font-size: 11px;
    font-weight: 700;
}

.eval-value {
    color: white;
    font-size: 24px;
    font-weight: 850;
}

.info {
    background: #0b2631;
    border: 1px solid #14727a;
    color: #cbd5e1;
    border-radius: 11px;
    padding: 12px 14px;
    margin-top: 12px;
    font-size: 12px;
}

.footer {
    border-top: 1px solid #203654;
    margin-top: 22px;
    padding-top: 12px;
    text-align: center;
    color: #7186a4;
    font-size: 11px;
}

.stButton button {
    width: 100%;
    border-radius: 9px;
    border: 1px solid #315582;
    background: #112c55;
    color: white;
}

[data-testid="stFileUploader"] {
    border: 1px dashed #42658f;
    border-radius: 12px;
    padding: 8px;
    background: #0a1930;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="hero">
    <h1>🎓 Prediksi Jumlah Mahasiswa Baru</h1>
    <p>Menggunakan Metode Linear Regression</p>
    <small>
        Dashboard menampilkan analisis data, evaluasi model, dan prediksi
        jumlah mahasiswa baru berdasarkan data historis.
    </small>
</div>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🎓 Dashboard PDS")
    st.caption("Prediksi Mahasiswa Baru")
    st.markdown("---")

    menu = st.radio(
        "MENU",
        ["Beranda", "Data", "Model & Evaluasi", "Prediksi"],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")

    uploaded_file = st.file_uploader(
        "Upload Dataset PDS.xlsx",
        type=["xlsx", "xls"]
    )

    if uploaded_file:
        st.success("Dataset berhasil diupload!")

    st.markdown("---")
    st.caption("Model: Linear Regression")
    st.caption("Target: Jumlah Mahasiswa Baru")

# ============================================================
# BACA DATA
# ============================================================
if uploaded_file is None:
    st.warning("Upload **Dataset PDS.xlsx** terlebih dahulu pada sidebar.")
    st.stop()

try:
    df = pd.read_excel(
        uploaded_file,
        sheet_name="2022-2026",
        header=2
    )
    df = df.iloc[:, :8].copy()
    df.columns = [str(c).strip() for c in df.columns]
except Exception as e:
    st.error(f"Dataset gagal dibaca: {e}")
    st.stop()

# Konversi numerik
numeric_cols = [
    "Tahun",
    "Jumlah Pendaftar",
    "Daya Tampung",
    "Jumlah Diterima",
    "Jumlah Daftar Ulang",
    "Jumlah Mahasiswa Baru"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Tahun", "Jumlah Mahasiswa Baru"])
df["Tahun"] = df["Tahun"].astype(int)

# ============================================================
# REKAP TAHUNAN
# ============================================================
data_tahunan = (
    df.groupby("Tahun", as_index=False)["Jumlah Mahasiswa Baru"]
      .sum()
      .sort_values("Tahun")
)

# Data untuk model
data_train = data_tahunan[data_tahunan["Tahun"] < 2026]
data_test = data_tahunan[data_tahunan["Tahun"] == 2026]

model = None
mae = rmse = mape = None
pred_test = None
pred_2027 = None

if len(data_train) >= 2:
    model = LinearRegression()
    model.fit(
        data_train[["Tahun"]],
        data_train["Jumlah Mahasiswa Baru"]
    )

    if len(data_test) > 0:
        pred_test = model.predict(data_test[["Tahun"]])

        y_test = data_test["Jumlah Mahasiswa Baru"]
        mae = mean_absolute_error(y_test, pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_test))

        if np.all(y_test != 0):
            mape = np.mean(
                np.abs((y_test - pred_test) / y_test)
            ) * 100

    tahun_berikutnya = int(data_tahunan["Tahun"].max() + 1)
    pred_2027 = max(
        0,
        int(round(
            model.predict(
                pd.DataFrame({"Tahun": [tahun_berikutnya]})
            )[0]
        ))
    )

# ============================================================
# BERANDA
# ============================================================
if menu == "Beranda":

    total_pendaftar = int(df["Jumlah Pendaftar"].sum())
    total_diterima = int(df["Jumlah Diterima"].sum())
    total_daftar_ulang = int(df["Jumlah Daftar Ulang"].sum())
    total_mhs_baru = int(df["Jumlah Mahasiswa Baru"].sum())

    cards = [
        ("card-blue", "👥", "TOTAL PENDAFTAR", f"{total_pendaftar:,}", "2022–2026"),
        ("card-green", "✓", "TOTAL DITERIMA", f"{total_diterima:,}", "2022–2026"),
        ("card-purple", "👤", "TOTAL DAFTAR ULANG", f"{total_daftar_ulang:,}", "2022–2026"),
        ("card-orange", "🎓", "TOTAL MAHASISWA BARU", f"{total_mhs_baru:,}", "2022–2026"),
    ]

    cols = st.columns(4)

    for col, item in zip(cols, cards):
        css, icon, label, value, sub = item
        with col:
            st.markdown(f"""
            <div class="card {css}">
                <div class="label">{icon} &nbsp; {label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # Grafik trend
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data_tahunan["Tahun"],
        y=data_tahunan["Jumlah Mahasiswa Baru"],
        mode="lines+markers+text",
        text=data_tahunan["Jumlah Mahasiswa Baru"].round(0).astype(int),
        textposition="top center",
        name="Data Aktual",
        line=dict(color="#38bdf8", width=4),
        marker=dict(size=9, color="#38bdf8")
    ))

    if pred_2027 is not None:
        fig.add_trace(go.Scatter(
            x=[data_tahunan["Tahun"].max(), tahun_berikutnya],
            y=[
                data_tahunan["Jumlah Mahasiswa Baru"].iloc[-1],
                pred_2027
            ],
            mode="lines+markers+text",
            text=[
                int(data_tahunan["Jumlah Mahasiswa Baru"].iloc[-1]),
                pred_2027
            ],
            textposition="top center",
            name=f"Prediksi {tahun_berikutnya}",
            line=dict(color="#c084fc", width=3, dash="dash"),
            marker=dict(size=10, color="#c084fc")
        ))

    fig.update_layout(
        title="Tren Jumlah Mahasiswa Baru (2022–2026)",
        title_font=dict(color="white", size=18),
        paper_bgcolor="#0a1830",
        plot_bgcolor="#0a1830",
        font=dict(color="#cbd5e1"),
        height=410,
        margin=dict(l=20, r=20, t=55, b=20),
        xaxis=dict(
            gridcolor="#1d3557",
            zeroline=False
        ),
        yaxis=dict(
            gridcolor="#1d3557",
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            y=1.08
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.35, 1])

    # Grafik pendaftar prodi
    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">📊 Pendaftar Berdasarkan Program Studi</div>', unsafe_allow_html=True)

        if "Program Studi" in df.columns:
            prodi = (
                df.groupby("Program Studi", as_index=False)["Jumlah Pendaftar"]
                .sum()
                .sort_values("Jumlah Pendaftar", ascending=False)
                .head(8)
            )

            fig_prodi = go.Figure(go.Bar(
                x=prodi["Jumlah Pendaftar"],
                y=prodi["Program Studi"],
                orientation="h",
                text=prodi["Jumlah Pendaftar"],
                textposition="outside",
                marker_color="#6366f1"
            ))

            fig_prodi.update_layout(
                paper_bgcolor="#0a1830",
                plot_bgcolor="#0a1830",
                font=dict(color="#cbd5e1"),
                height=370,
                margin=dict(l=10, r=40, t=10, b=20),
                xaxis=dict(gridcolor="#1d3557"),
                yaxis=dict(gridcolor="#0a1830"),
                showlegend=False
            )

            st.plotly_chart(fig_prodi, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Prediksi
    with col2:
        if pred_2027 is not None:
            st.markdown(f"""
            <div class="prediction">
                <div class="year">🔮 PREDIKSI JUMLAH MAHASISWA BARU {tahun_berikutnya}</div>
                <div class="number">{pred_2027:,}</div>
                <div class="desc">
                    mahasiswa<br><br>
                    Hasil prediksi menggunakan model
                    <b>Linear Regression</b> berdasarkan data 2022–2026.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Evaluasi mini
        if mae is not None:
            st.markdown(f"""
            <div class="eval">
                <h3>🎯 Hasil Evaluasi Model</h3>
                <div style="display:flex;gap:8px;">
                    <div class="eval-item" style="flex:1;">
                        <div class="eval-name">MAE</div>
                        <div class="eval-value">{mae:.1f}</div>
                    </div>
                    <div class="eval-item" style="flex:1;">
                        <div class="eval-name">RMSE</div>
                        <div class="eval-value">{rmse:.1f}</div>
                    </div>
                    <div class="eval-item" style="flex:1;">
                        <div class="eval-name">MAPE</div>
                        <div class="eval-value">{mape:.2f}%</div>
                    </div>
                </div>
                <div class="info">
                    Model digunakan untuk memprediksi jumlah mahasiswa baru
                    berdasarkan pola tahun sebelumnya.
                </div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# SCATTER PLOT
# ============================================================
if menu == "Beranda":

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-title">🔵 Hubungan Jumlah Pendaftar dengan Mahasiswa Baru</div>',
        unsafe_allow_html=True
    )

    if "Jumlah Pendaftar" in df.columns:
        scatter_df = df[
            ["Jumlah Pendaftar", "Jumlah Mahasiswa Baru", "Tahun"]
        ].dropna().copy()

        # Setiap titik mewakili satu data program studi pada satu tahun.
        fig_scatter = go.Figure()

        fig_scatter.add_trace(go.Scatter(
            x=scatter_df["Jumlah Pendaftar"],
            y=scatter_df["Jumlah Mahasiswa Baru"],
            mode="markers",
            marker=dict(
                size=10,
                color="#38bdf8",
                opacity=0.78,
                line=dict(width=1, color="#dbeafe")
            ),
            text=[
                f"Tahun: {tahun}<br>"
                f"Pendaftar: {pendaftar:,}<br>"
                f"Mahasiswa Baru: {mhs:,}"
                for tahun, pendaftar, mhs in zip(
                    scatter_df["Tahun"],
                    scatter_df["Jumlah Pendaftar"],
                    scatter_df["Jumlah Mahasiswa Baru"]
                )
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Data"
        ))

        # Garis regresi untuk melihat kecenderungan hubungan
        if len(scatter_df) >= 2:
            x = scatter_df["Jumlah Pendaftar"].values
            y = scatter_df["Jumlah Mahasiswa Baru"].values

            koef = np.polyfit(x, y, 1)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = koef[0] * x_line + koef[1]

            fig_scatter.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(
                    color="#c084fc",
                    width=3,
                    dash="dash"
                ),
                name="Trend"
            ))

        fig_scatter.update_layout(
            paper_bgcolor="#0a1830",
            plot_bgcolor="#0a1830",
            font=dict(color="#cbd5e1"),
            height=400,
            margin=dict(l=20, r=20, t=25, b=20),
            xaxis=dict(
                title="Jumlah Pendaftar",
                gridcolor="#1d3557",
                zeroline=False
            ),
            yaxis=dict(
                title="Jumlah Mahasiswa Baru",
                gridcolor="#1d3557",
                zeroline=False
            ),
            legend=dict(
                orientation="h",
                y=1.08
            )
        )

        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("""
        <div class="info">
            <b>📌 Cara membaca scatter plot:</b><br>
            Setiap titik menunjukkan data jumlah pendaftar dan jumlah mahasiswa baru.
            Semakin menunjukkan pola naik dari kiri ke kanan, semakin terlihat
            kecenderungan bahwa jumlah mahasiswa baru meningkat ketika jumlah pendaftar meningkat.
            Garis putus-putus menunjukkan garis kecenderungan (trend).
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# DATA
# ============================================================
elif menu == "Data":
    st.markdown('<div class="panel-title">📋 Data Jumlah Mahasiswa Baru</div>', unsafe_allow_html=True)

    st.dataframe(
        df,
        use_container_width=True,
        height=560
    )

    st.download_button(
        "⬇️ Download Data CSV",
        df.to_csv(index=False).encode("utf-8"),
        "data_mahasiswa.csv",
        "text/csv"
    )

# ============================================================
# MODEL & EVALUASI
# ============================================================
elif menu == "Model & Evaluasi":

    st.markdown('<div class="panel-title">🎯 Model & Evaluasi</div>', unsafe_allow_html=True)

    if model is None or len(data_test) == 0:
        st.warning("Data training/testing belum cukup.")
        st.stop()

    actual = float(data_test["Jumlah Mahasiswa Baru"].iloc[0])
    predicted = float(pred_test[0])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card card-blue">
            <div class="label">MAE</div>
            <div class="value">{mae:.1f}</div>
            <div class="sub">Rata-rata kesalahan</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card card-purple">
            <div class="label">RMSE</div>
            <div class="value">{rmse:.1f}</div>
            <div class="sub">Akar kesalahan kuadrat</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card card-green">
            <div class="label">MAPE</div>
            <div class="value">{mape:.2f}%</div>
            <div class="sub">Persentase kesalahan</div>
        </div>
        """, unsafe_allow_html=True)

    eval_data = pd.DataFrame({
        "Kategori": ["Data Aktual", "Hasil Prediksi"],
        "Jumlah": [actual, predicted]
    })

    fig = go.Figure(go.Bar(
        x=eval_data["Kategori"],
        y=eval_data["Jumlah"],
        text=eval_data["Jumlah"].round(0).astype(int),
        textposition="outside",
        marker_color=["#38bdf8", "#c084fc"]
    ))

    fig.update_layout(
        title="Perbandingan Aktual vs Prediksi Tahun 2026",
        paper_bgcolor="#0a1830",
        plot_bgcolor="#0a1830",
        font=dict(color="#cbd5e1"),
        height=400,
        yaxis=dict(gridcolor="#1d3557")
    )

    st.plotly_chart(fig, use_container_width=True)

    error = abs(actual - predicted) / actual * 100 if actual != 0 else 0

    st.markdown(f"""
    <div class="info">
        <b>Kesimpulan Evaluasi:</b><br>
        Data aktual tahun 2026 adalah <b>{actual:,.0f}</b> mahasiswa,
        sedangkan hasil prediksi adalah <b>{predicted:,.2f}</b> mahasiswa.
        Persentase kesalahan sebesar <b>{error:.2f}%</b>.
        Semakin kecil MAE, RMSE, dan MAPE, semakin kecil kesalahan model.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PREDIKSI
# ============================================================
elif menu == "Prediksi":

    st.markdown('<div class="panel-title">🔮 Prediksi Jumlah Mahasiswa Baru</div>', unsafe_allow_html=True)

    if pred_2027 is None:
        st.warning("Model belum dapat dibuat.")
        st.stop()

    st.markdown(f"""
    <div class="prediction">
        <div class="year">PREDIKSI TAHUN {tahun_berikutnya}</div>
        <div class="number">{pred_2027:,}</div>
        <div class="desc">
            Perkiraan jumlah mahasiswa baru menggunakan Linear Regression.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabel aktual + prediksi
    pred_table = data_tahunan.copy()
    pred_table["Keterangan"] = "Aktual"

    next_row = pd.DataFrame({
        "Tahun": [tahun_berikutnya],
        "Jumlah Mahasiswa Baru": [pred_2027],
        "Keterangan": ["Prediksi"]
    })

    pred_table = pd.concat([pred_table, next_row], ignore_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(pred_table, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">
    PDS • Prediksi Jumlah Mahasiswa Baru &nbsp;|&nbsp;
    Linear Regression &nbsp;|&nbsp; Data 2022–2026
</div>
""", unsafe_allow_html=True)
