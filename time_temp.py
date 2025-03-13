import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.optimize import curve_fit

def newton_cooling(t, T_eq, T_0, tau):
    return T_eq + (T_0 - T_eq) * np.exp(-t / tau)

def fit_curve(time, temp):
    T_0 = temp[0]
    T_eq_guess = temp[-1]
    tau_guess = (time[-1] - time[0]) / 2
    
    popt, pcov = curve_fit(newton_cooling, time, temp, p0=[T_eq_guess, T_0, tau_guess])
    T_eq, T_0_fit, tau = popt
    
    # Generate more points for a smoother fitted curve
    time_fine = np.linspace(min(time), max(time), 500)
    fitted_temp_fine = newton_cooling(time_fine, *popt)
    
    # Calculate RMSE and R2
    fitted_temp = newton_cooling(np.array(time), *popt)
    residuals = np.array(temp) - fitted_temp
    rmse = np.sqrt(np.mean(residuals**2))
    ss_total = np.sum((np.array(temp) - np.mean(temp))**2)
    ss_residual = np.sum(residuals**2)
    r2 = 1 - (ss_residual / ss_total)
    
    return T_eq, tau, rmse, r2, time_fine, fitted_temp_fine

st.title("Temp vs Time")

# Reduce input box width using CSS
st.markdown(
    """
    <style>
    div[data-testid="stNumberInput"] {
        max-width: 110px !important; /* Ajusta el ancho de las cajas */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write("Measure equilibrium temp using Newton's Law of Cooling")

# Choose input method
input_method = st.radio("Select data input method:", ("Manual Entry", "Upload CSV"))

time = []
temp = []

if input_method == "Manual Entry":
    st.write("Enter time and temp values manually:")
    
    # Center the number input field
    col_main = st.columns([1, 2, 1])
    with col_main[1]:
        n = st.number_input("Number of measurements:", min_value=1, value=5, step=1, key="num_measurements")

    # Input fields for time and temperature
    for i in range(n):
        col1, col2 = st.columns([0.3, 0.3], gap='small')  # Reduce column spacing
        with col1:
            time.append(st.number_input(f"Time (s) {i+1}", key=f"time_{i}", step=1, format="%d"))
        with col2:
            temp.append(st.number_input(f"Temp {i+1}", key=f"temp_{i}"))

elif input_method == "Upload CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with time and temp columns", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "time" in df.columns and "temp" in df.columns:
            time = df["time"].values
            temp = df["temp"].values
        else:
            st.error("CSV must contain 'time' and 'temp' columns.")

# Button to execute the fitting
if st.button("Calculate") and len(time) > 0 and len(temp) > 0:
    time = np.array(time)
    temp = np.array(temp)
    
    T_eq, tau, rmse, r2, time_fine, fitted_temp_fine = fit_curve(time, temp)
    
    fig, ax = plt.subplots()
    ax.scatter(time, temp, color="red")
    ax.plot(time_fine, fitted_temp_fine, color="blue")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temp (°C)")
    
    # Add vertical line at tau (τ)
    ax.axvline(x=tau, color='gray', linestyle='dashed')
    
    # Adjust legend text with newlines
    legend_text = f"T_eq = {T_eq:.2f} °C\nτ = {tau:.2f} s\nRMSE = {rmse:.4f}\nR² = {r2:.4f}"
    ax.text(0.95, 0.05, legend_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
    
    st.pyplot(fig)
    
    st.write("At 3τ, 95% of T_eq is reached. At 5τ, 99% of T_eq is reached.")
