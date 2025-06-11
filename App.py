import streamlit as st
import pandas as pd
import joblib

# === Konfigurasi Tampilan ===
st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="centered")

# === Load model dan file pendukung ===
@st.cache_resource
def load_model():
    model = joblib.load("Model/dropout_model.joblib")
    scaler = joblib.load("Model/scaler.joblib")
    feature_order = joblib.load("Model/feature_order.joblib")
    label_encoder = joblib.load("Model/label_encoder.joblib")
    return model, scaler, feature_order, label_encoder

try:
    model, scaler, feature_order, label_encoder = load_model()
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

st.markdown("""
    <style>
    .big-title {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 0.5em;
    }
    .result-box {
        padding: 1.5em;
        border-radius: 10px;
        font-size: 1.2em;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-top: 1.5em;
    }
    .lulus { background-color: #2e7d32; }
    .aktif { background-color: #0277bd; }
    .dropout { background-color: #c62828; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">🎓 Prediksi Status Kelulusan Mahasiswa</div>', unsafe_allow_html=True)
st.markdown("Masukkan data mahasiswa untuk mengetahui apakah mereka akan **Lulus**, **Drop Out**, atau masih **Aktif**.")

# === Form Input ===
with st.form("form_prediksi"):
    st.subheader("📋 Input Data Mahasiswa")
    gender = st.selectbox("Jenis Kelamin", ['Laki-laki', 'Perempuan'])
    age = st.number_input("Umur saat mendaftar", min_value=15, max_value=100, value=18)
    high_school_type = st.selectbox("Asal Sekolah", ['SMA', 'SMK', 'MA', 'Lainnya'])
    total_credits = st.number_input("Total SKS diambil", min_value=0, max_value=200, value=20)
    avg_grade = st.number_input("Rata-rata Nilai (0.00 - 4.00)", min_value=0.00, max_value=4.00, step=0.01, value=2.50)
    scholarship = st.selectbox("Penerima Beasiswa", ['Ya', 'Tidak'])

    submit = st.form_submit_button("🔍 Prediksi")

# === Proses Prediksi ===
if submit:
    # Rule-based logic override
    if total_credits >= 150:
        pred_label = "Lulus"
    elif total_credits <= 30:
        pred_label = "Drop Out"
    elif 24 <= total_credits <= 150:
        pred_label = "Aktif"
    else:
        # === Proses prediksi dengan model ML ===
        gender_encoded = 1 if gender == 'Perempuan' else 0
        school_encoded = {'SMA': 0, 'SMK': 1, 'MA': 2, 'Lainnya': 3}[high_school_type]
        scholarship_encoded = 1 if scholarship == 'Ya' else 0

        input_dict = {
            'Gender': gender_encoded,
            'Age_at_enrollment': age,
            'High_School_Type': school_encoded,
            'Total_Credits': total_credits,
            'Average_Grade': avg_grade,
            'Scholarship': scholarship_encoded
        }

        df_input = pd.DataFrame([input_dict])
        df_input = df_input.reindex(columns=feature_order)
        df_scaled = scaler.transform(df_input)

        prediction = model.predict(df_scaled)
        pred_label = label_encoder.inverse_transform(prediction)[0]

    # === Tampilan hasil berwarna ===
    if pred_label == "Lulus":
        css_class = "lulus"
        emoji = "🎉"
    elif pred_label == "Aktif":
        css_class = "aktif"
        emoji = "📘"
    else:  # Drop Out
        css_class = "dropout"
        emoji = "⚠️"

    st.markdown(
        f'<div class="result-box {css_class}">{emoji} Hasil Prediksi: <span style="text-transform:uppercase">{pred_label}</span></div>',
        unsafe_allow_html=True
    )
