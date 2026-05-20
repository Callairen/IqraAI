import streamlit as st
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.perceptron import Perceptron
from src.train import train_model, prepare_data
from data.labels import LABELS, N_CLASSES
from visualization.plot_error import plot_error_per_epoch
from visualization.plot_characters import plot_prediction_result
from visualization.plot_weights import plot_weight_heatmap

# ─── Konfigurasi Halaman ───────────────────────────────────────
st.set_page_config(
    page_title="Pengenalan Huruf Arab — Perceptron",
    page_icon="🕌",
    layout="wide"
)

# ─── Judul ────────────────────────────────────────────────────
st.title("🕌 Pengenalan Huruf Arab dengan Perceptron")
st.markdown("**Jaringan Saraf Tiruan — FILKOM Universitas Brawijaya**")
st.divider()

# ─── Sidebar: Parameter Training ──────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameter Training")

    alpha = st.slider(
        "Learning Rate (α)",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Seberapa besar update bobot per iterasi"
    )

    theta = st.slider(
        "Threshold (θ)",
        min_value=0.1,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Batas fungsi aktivasi"
    )

    max_epoch = st.number_input(
        "Max Epoch",
        min_value=10,
        max_value=5000,
        value=1000,
        step=10
    )

    train_button = st.button(
        "🚀 Mulai Training",
        use_container_width=True,
        type="primary"
    )

    st.divider()
    st.markdown("**Kelompok [Nama Kelompok]**")
    st.markdown("FILKOM UB — Genap 2024/2025")

# ─── Training ─────────────────────────────────────────────────
if train_button or 'model' not in st.session_state:
    with st.spinner("Sedang training... sabar ya 😊"):
        model, history = train_model(
            alpha=alpha,
            theta=theta,
            max_epoch=max_epoch
        )
        st.session_state['model']   = model
        st.session_state['history'] = history

    n_epochs = len(history['error_per_epoch'])
    final_error = history['error_per_epoch'][-1]

    if final_error == 0:
        st.success(f"✅ Training konvergen di epoch ke-{n_epochs}!")
    else:
        st.warning(f"⚠️ Training berhenti di epoch ke-{n_epochs}, masih ada error.")

# ─── Tab Utama ────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🖊️ Uji Prediksi",
    "📈 Hasil Training",
    "📚 Tentang Proyek"
])

# ── Tab 1: Uji Prediksi ──────────────────────────────────────
with tab1:
    st.subheader("Gambar huruf Arab di grid 7×5")
    st.caption("Klik kotak untuk mengaktifkan pixel. Kotak hitam = 1, putih = -1")

    # Inisialisasi grid di session state
    if 'grid' not in st.session_state:
        st.session_state['grid'] = [[-1]*5 for _ in range(7)]

    col_reset, col_predict, col_spacer = st.columns([1, 1, 4])

    with col_reset:
        if st.button("🗑️ Reset Grid"):
            st.session_state['grid'] = [[-1]*5 for _ in range(7)]
            st.rerun()

    with col_predict:
        predict_btn = st.button("🔍 Prediksi", type="primary")

    # Tampilkan grid interaktif
    st.markdown("**Grid Input (klik untuk toggle):**")

    for i in range(7):
        cols = st.columns(5)
        for j in range(5):
            with cols[j]:
                val = st.session_state['grid'][i][j]
                emoji = "⬛" if val == 1 else "⬜"
                if st.button(emoji, key=f"cell_{i}_{j}"):
                    st.session_state['grid'][i][j] *= -1
                    st.rerun()

    # Prediksi
    if predict_btn and 'model' in st.session_state:
        model = st.session_state['model']
        grid_flat = np.array(st.session_state['grid']).flatten()

        pred_idx = model.predict_class(grid_flat)
        scores   = model.confidence_scores(grid_flat)

        st.divider()
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.markdown(f"### Prediksi: **{LABELS[pred_idx]}**")
            st.markdown(f"Confidence: **{scores[pred_idx]*100:.1f}%**")

        with col_r2:
            fig = plot_prediction_result(grid_flat, pred_idx, scores, LABELS)
            st.pyplot(fig)

# ── Tab 2: Hasil Training ─────────────────────────────────────
with tab2:
    if 'history' in st.session_state:
        history = st.session_state['history']
        model   = st.session_state['model']

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Total Epoch", len(history['error_per_epoch']))
        with col_m2:
            st.metric("Error Akhir", history['error_per_epoch'][-1])
        with col_m3:
            X, _ = prepare_data()
            correct = sum(
                model.predict_class(X[i]) == i
                for i in range(len(X))
            )
            st.metric("Akurasi", f"{correct/len(X)*100:.1f}%")

        st.subheader("Grafik Konvergensi")
        fig_error = plot_error_per_epoch(history)
        st.pyplot(fig_error)

        st.subheader("Heatmap Bobot Akhir")
        fig_w = plot_weight_heatmap(model, LABELS)
        st.pyplot(fig_w)
    else:
        st.info("Klik 'Mulai Training' di sidebar dulu ya!")

# ── Tab 3: Tentang Proyek ─────────────────────────────────────
with tab3:
    st.subheader("Tentang Proyek Ini")
    st.markdown("""
    Proyek ini mengimplementasikan algoritma **Perceptron** untuk
    mengenali huruf konsonan Arab menggunakan **Jaringan Saraf Tiruan**.

    **Dataset:** 14 huruf Arab dalam bentuk *isolated* (berdiri sendiri),
    direpresentasikan sebagai grid pixel 7×5 (35 input neuron).

    **Arsitektur:**
    - Input layer: 35 neuron
    - Output layer: 14 neuron (satu per huruf)
    - Fungsi aktivasi: bipolar threshold

    **Referensi:**
    - Fausett, L. (1994). *Fundamentals of Neural Networks*
    - Slide kuliah: Yuita Arum Sari — ANN: Perceptron, FILKOM UB 2025
    """)
