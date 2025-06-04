import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import hashlib
import json
import os
import base64

st.set_page_config(page_title="Fuzzy Crop Recommendation", layout="centered")

# ------------ BACKGROUND ------------ #

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_from_local(image_file):
    bin_str = get_base64_of_bin_file(image_file)
    page_bg_img = f'''
    <style>
    [data-testid="stApp"] {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background
set_bg_from_local("D:/prak scpk/data.jpg")


# ------------ USER LOGIN ------------ #

USER_FILE = "users.json"

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USER_FILE, "w") as f:
        json.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

users = load_users()

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'username' not in st.session_state:
    st.session_state.username = ''

def show_login():
    st.title("🚪 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Selamat datang, {username}!")
            st.rerun()
        else:
            st.error("Username atau password salah.")
    st.write("Belum punya akun?")
    if st.button("Daftar sekarang"):
        st.session_state.page = 'signup'
        st.rerun()

def show_signup():
    st.title("📝 Daftar Akun Baru")
    new_user = st.text_input("Username baru")
    new_pass = st.text_input("Password baru", type="password")
    if st.button("Daftar"):
        if new_user.strip() == "" or new_pass.strip() == "":
            st.warning("Username dan password tidak boleh kosong.")
        elif new_user in users:
            st.warning("Username sudah terdaftar.")
        else:
            users[new_user] = hash_password(new_pass)
            save_users(users)
            st.success("Akun berhasil dibuat! Silakan login.")
            st.session_state.page = 'login'
            st.rerun()
    if st.button("Kembali ke login"):
        st.session_state.page = 'login'
        st.rerun()

# ------------ DASHBOARD & FUZZY ------------ #

def show_dashboard():
    st.sidebar.write(f"👤 User: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = 'login'
        st.session_state.username = ''
        st.rerun()

    menu = st.sidebar.selectbox("📚 Menu", ["Lihat Dataset", "Input dan Hasil Fuzzy"])

    @st.cache_data
    def load_data():
        return pd.read_csv("dataset_fuzzy.csv")

    try:
        data = load_data()
        data_loaded = True
    except Exception as e:
        st.warning("⚠️ Gagal memuat dataset.")
        st.error(f"Detail error: {e}")
        data_loaded = False

    if menu == "Lihat Dataset":
        st.title("📋 Dataset Fuzzy Crop")
        if data_loaded:
            num_rows = st.sidebar.number_input("Jumlah baris", 5, len(data), 10)
            st.dataframe(data.head(num_rows))
            with st.expander("📊 Statistik"):
                st.write("Ukuran:", data.shape)
                st.write(data.describe())
                st.write("Missing Values:")
                st.write(data.isnull().sum())
    else:
        st.title("🌾 Input dan Rekomendasi Fuzzy")

        col1, col2 = st.columns(2)
        with col1:
            input_temp = st.slider("🌡️ Temperature (°C)", 0.0, 50.0, 25.0)
            input_ph = st.slider("🧪 pH", 3.0, 10.0, 6.5)
        with col2:
            input_hum = st.slider("💧 Humidity (%)", 0.0, 100.0, 70.0)
            input_rain = st.slider("☔ Rainfall (mm)", 0.0, 300.0, 150.0)

        # Fuzzy setup
        temperature = ctrl.Antecedent(np.arange(0, 50, 1), 'temperature')
        humidity = ctrl.Antecedent(np.arange(0, 100, 1), 'humidity')
        ph = ctrl.Antecedent(np.arange(3, 10, 0.1), 'ph')
        rainfall = ctrl.Antecedent(np.arange(0, 300, 1), 'rainfall')
        crop = ctrl.Consequent(np.arange(0, 100, 1), 'crop')

        temperature['low'] = fuzz.trimf(temperature.universe, [0, 0, 25])
        temperature['medium'] = fuzz.trapmf(temperature.universe, [20, 23, 27, 30])
        temperature['high'] = fuzz.trimf(temperature.universe, [25, 50, 50])

        humidity['low'] = fuzz.trimf(humidity.universe, [0, 0, 50])
        humidity['medium'] = fuzz.trapmf(humidity.universe, [30, 40, 60, 70])
        humidity['high'] = fuzz.trimf(humidity.universe, [50, 100, 100])

        ph['acidic'] = fuzz.trimf(ph.universe, [3, 3, 6])
        ph['neutral'] = fuzz.trapmf(ph.universe, [5.8, 6.2, 6.8, 7.2])
        ph['alkaline'] = fuzz.trimf(ph.universe, [7, 10, 10])

        rainfall['low'] = fuzz.trimf(rainfall.universe, [0, 0, 100])
        rainfall['medium'] = fuzz.trapmf(rainfall.universe, [70, 100, 200, 230])
        rainfall['high'] = fuzz.trimf(rainfall.universe, [200, 300, 300])

        crop['crop1'] = fuzz.trimf(crop.universe, [0, 0, 33])
        crop['crop2'] = fuzz.trimf(crop.universe, [33, 50, 66])
        crop['crop3'] = fuzz.trimf(crop.universe, [66, 100, 100])

        rules = [
            ctrl.Rule(temperature['low'] & humidity['high'] & ph['acidic'] & rainfall['medium'], crop['crop1']),
            ctrl.Rule(temperature['medium'] & humidity['medium'] & ph['neutral'] & rainfall['medium'], crop['crop2']),
            ctrl.Rule(temperature['high'] & humidity['low'] & ph['alkaline'] & rainfall['low'], crop['crop3']),
            ctrl.Rule(temperature['medium'] & humidity['high'] & ph['neutral'] & rainfall['high'], crop['crop1']),
            ctrl.Rule(temperature['low'] & humidity['low'] & ph['alkaline'] & rainfall['low'], crop['crop3']),
            ctrl.Rule(temperature['high'] & humidity['high'] & ph['acidic'] & rainfall['medium'], crop['crop2']),
            ctrl.Rule(temperature['low'] & humidity['medium'] & ph['neutral'] & rainfall['low'], crop['crop1']),
            ctrl.Rule(temperature['medium'] & humidity['low'] & ph['acidic'] & rainfall['medium'], crop['crop2']),
            ctrl.Rule(temperature['high'] & humidity['medium'] & ph['alkaline'] & rainfall['high'], crop['crop3']),
        ]

        crop_ctrl = ctrl.ControlSystem(rules)
        crop_sim = ctrl.ControlSystemSimulation(crop_ctrl)

        try:
            crop_sim.input['temperature'] = input_temp
            crop_sim.input['humidity'] = input_hum
            crop_sim.input['ph'] = input_ph
            crop_sim.input['rainfall'] = input_rain
            crop_sim.compute()

            output_crop = crop_sim.output['crop']
            if output_crop < 33:
                crop_name = "Crop 1"
            elif output_crop < 66:
                crop_name = "Crop 2"
            else:
                crop_name = "Crop 3"

            st.success(f"🌱 Rekomendasi tanaman terbaik: **{crop_name}**")
            st.caption(f"Nilai fuzzy output: {output_crop:.2f}")

            def plot_var(variable, label, nilai_vertikal=None):
                plt.figure(figsize=(8, 4))
                for term_name, term_obj in variable.terms.items():
                    plt.plot(variable.universe, term_obj.mf, label=term_name)
                if nilai_vertikal is not None:
                    plt.axvline(x=nilai_vertikal, color='r', linestyle='--', label=f"{label} input")
                plt.title(f"Fungsi Keanggotaan - {label}")
                plt.xlabel(label)
                plt.ylabel("Derajat Keanggotaan")
                plt.legend()
                plt.grid(True)
                st.pyplot(plt.gcf())
                plt.close()

            with st.expander("🔍 Keanggotaan Input"):
                tab1, tab2, tab3, tab4 = st.tabs(["Temperature", "Humidity", "pH", "Rainfall"])
                with tab1:
                    plot_var(temperature, "Temperature (°C)", input_temp)
                with tab2:
                    plot_var(humidity, "Humidity (%)", input_hum)
                with tab3:
                    plot_var(ph, "pH", input_ph)
                with tab4:
                    plot_var(rainfall, "Rainfall (mm)", input_rain)

            with st.expander("🔍 Keanggotaan Output"):
                plot_var(crop, "Crop", output_crop)

        except Exception as e:
            st.error(f"Error dalam perhitungan fuzzy: {e}")

# ------------ MAIN MENU ------------ #

if not st.session_state.logged_in:
    if st.session_state.page == 'login':
        show_login()
    else:
        show_signup()
else:
    show_dashboard()
