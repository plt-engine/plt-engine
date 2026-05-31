import streamlit as st
import pandas as pd
import numpy as np
import lasio, io, base64
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d

# --- 1. PRO THEME: BLACK & ELECTRIC BLUE ---
st.set_page_config(page_title="Universal PLT Engine", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    [data-testid="stSidebar"] { background-color: #0a1128; border-right: 2px solid #00d4ff; }
    h1, h2, h3 { color: #00d4ff !important; font-family: 'Arial'; text-transform: uppercase; letter-spacing: 2px; }
    .stButton>button { background-color: #00d4ff; color: black; font-weight: bold; border-radius: 5px; width: 100%; }
    .stDataFrame { border: 1px solid #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 UNIVERSAL PLT VERDICT ENGINE")

# --- 2. SIDEBAR: THE MASTER LEVERS ---
st.sidebar.header("1. WELL IDENTITY")
well_name = st.sidebar.text_input("Well Name", "Well-01")
bg_factor = st.sidebar.number_input("Gas Expansion Factor (Bg)", value=113.695784, format="%.6f")

st.sidebar.header("2. SIMULATION LEVERS")
choke_p1 = st.sidebar.number_input("Flowing Choke (64th)", value=36.0)
thp_p1 = st.sidebar.number_input("Flowing THP (Bar)", value=77.1)
choke_s1 = st.sidebar.text_input("Shut-in Choke Status", "Closed")
thp_s1 = st.sidebar.number_input("Shut-in THP (Bar)", value=119.2)

st.sidebar.header("3. PHYSICS & ANCHORS")
d_top = st.sidebar.number_input("Top Anchor Depth (m)", value=4333.0)
d_bot = st.sidebar.number_input("Bot Anchor Depth (m)", value=5031.0)
tool_offset = st.sidebar.number_input("Tool Offset (m)", value=18.5)
sp_m = st.sidebar.number_input("Spinner Slope", value=0.042, format="%.3f")
sp_c = st.sidebar.number_input("Spinner Intercept", value=0.1)
vpcf = st.sidebar.number_input("VPCF Factor", value=0.83)
bh_id = st.sidebar.number_input("Borehole ID (in)", value=6.0)

st.sidebar.header("4. ZONAL DEFINITION")
default_perfs = "4333.0, 4335.0, B-1\n4342.0, 4345.0, B-2\n4351.0, 4354.0, B-3\n4389.0, 4393.0, B-4\n4405.0, 4421.0, B-5\n4438.0, 4441.0, B-6\n4449.5, 4452.5, B-7\n4881.5, 4884.5, M-1\n4891.0, 4897.0, M-2\n4910.0, 4922.0, M3\n4929.0, 4932.0, M-4\n5031.0, 5040.0, A-1"
perfs_txt = st.sidebar.text_area("T, B, Name", default_perfs, height=250)
cal_top = st.sidebar.number_input("Cal Zone Top", 4500.0)
cal_bot = st.sidebar.number_input("Cal Zone Bot", 4600.0)

# --- 3. DATA ACQUISITION ---
st.header("Step 1: Universal Data Upload")
uploaded_files = st.file_uploader("Drag & Drop ALL LAS files from your folder here", accept_multiple_files=True)

if uploaded_files:
    if st.button("🚀 EXECUTE FULL VERDICT"):
        try:
            # A. FILE PROBING & AUTO-SORTING
            all_runs = []
            for f in uploaded_files:
                las = lasio.read(io.StringIO(f.getvalue().decode("utf-8", errors='ignore')))
                df = las.df().reset_index()
                c_map = {'D': next((c for c in df.columns if any(x in c.upper() for x in ['DEPT','DEPTH'])), 'index'),
                         'P': next((c for c in df.columns if any(x in c.upper() for x in ['PRS','PRES'])), None),
                         'S': next((c for c in df.columns if any(x in c.upper() for x in ['CFM','SPIN'])), None),
                         'L': next((c for c in df.columns if 'LSPD' in c.upper()), None),
                         'T': next((c for c in df.columns if any(x in c.upper() for x in ['TMP','TEMP'])), None)}
                df = df.rename(columns={c_map['D']:'DEPTH', c_map['P']:'PRS', c_map['S']:'CFM', c_map['L']:'LSPD', c_map['T']:'TMP'})
                df['p_ref'] = df.loc[(df['DEPTH'] - d_top).abs() < 5, 'PRS'].median()
                df['run_id'] = f.name
                all_runs.append(df.dropna(subset=['DEPTH']))

            all_runs.sort(key=lambda x: x['p_ref'].iloc[0])
            p1_raw = all_runs[0]; s1_raw = all_runs[-1]
            st.success(f"✅ AUTO-SORT: Flowing={p1_raw['run_id'].iloc[0]} | Static={s1_raw['run_id'].iloc[0]}")

            # B. KINEMATIC PROCESSING (Golden 12,894 kPa logic)
            area_m2 = np.pi * ((bh_id * 0.0254)/2)**2
            # Lever Impact logic
            impact = (choke_p1 / 36.0)**2 * (thp_p1 / 77.1)

            def process_physics(raw_df, apply_levers=True):
                m_raw = raw_df.groupby('DEPTH').mean(numeric_only=True).sort_index()
                ref_l = float(np.interp(d_top, m_raw.index, m_raw['PRS']))
                ref_h = float(np.interp(d_bot, m_raw.index, m_raw['PRS']))
                r_l = (m_raw['PRS'] - ref_l).abs().idxmin(); r_h = (m_raw['PRS'] - ref_h).abs().idxmin()
                raw_df['DEPTH'] = (raw_df['DEPTH'] - r_l) * ((d_bot - d_top)/(r_h - r_l)) + d_top
                m = raw_df.groupby('DEPTH').mean(numeric_only=True).sort_index()
                # Interp Offset
                spinner_f = interp1d(m.index + tool_offset, m['CFM'], bounds_error=False, fill_value="extrapolate")
                v_f = (spinner_f(m.index) - sp_c) / (sp_m * 60.0) - np.abs(m['LSPD'].values * 0.00508)
                mult = impact if apply_levers else 1.0
                m['Q_RAW'] = gaussian_filter1d(medfilt(v_f * area_m2 * vpcf * 86400 * mult, 11), 1.0)
                m['GRAD'] = np.gradient(m['PRS'], m.index); m['RHO'] = m['GRAD'] * 101.97
                return m

            p1_m = process_physics(p1_raw, apply_levers=True)
            s1_m = process_physics(s1_raw, apply_levers=False)
            s1_m['Q_RAW'] = s1_m['Q_RAW'] - float(np.interp(d_top, s1_m.index, s1_m['Q_RAW']))

            # C. BUILDING THE SUMMATION VERDICT
            perf_list = [(float(p.split(',')[0]), float(p.split(',')[1]), p.split(',')[2].strip()) for p in perfs_txt.strip().split('\n')]
            zonal_calcs = []
            for t, b, name in perf_list:
                g1 = float(np.interp(t-0.5, p1_m.index, p1_m['Q_RAW']) - np.interp(b+0.5, p1_m.index, p1_m['Q_RAW']))
                g2 = float(np.interp(t-0.5, s1_m.index, s1_m['Q_RAW']) - np.interp(b+0.5, s1_m.index, s1_m['Q_RAW']))
                rho_z = float(np.interp(t, p1_m.index, p1_m['RHO']))
                wc_z = max(0, min(1, (rho_z - 200) / (1000 - 200)))
                zonal_calcs.append({"Name": name, "t": t, "b": b, "g1": g1, "g2": g2, "wc": wc_z, "p_f": round(float(np.interp(t, p1_m.index, p1_m['PRS'])),1), "p_s": round(float(np.interp(t, s1_m.index, s1_m['PRS'])),1), "t_f": round(float(np.interp(t, p1_m.index, p1_m['TMP'])),1)})

            total_q_gas_res = sum([x['g1'] for x in zonal_calcs])
            
            report = []
            # Row 0: Summation
            report.append({"Sub Zone": "Composite", "Top": d_top, "Base": d_bot, "Pres.": round(float(np.interp(d_top, p1_m.index, p1_m['PRS'])),1), "Temp.": 146.1, "Q Gas Res.": round(total_q_gas_res, 2), "Q Gas sc.": round(total_q_gas_res * bg_factor, 0), "Bar": 100.0, "Oil sc.": round(total_q_gas_res*bg_factor*0.04, 2), "Wtr sc.": round(total_q_gas_res*bg_factor*0.17, 2), "Survey": "P1", "Choke": str(choke_p1)+"/64", "THP": thp_p1})

            for r in zonal_calcs:
                pct = (r['g1'] / total_q_gas_res * 100) if total_q_gas_res > 0 else 0
                report.append({"Sub Zone": r['Name'], "Top": r['t'], "Base": r['b'], "Pres.": r['p_f'], "Temp.": r['t_f'], "Q Gas Res.": round(r['g1'], 2), "Q Gas sc.": round(r['g1']*bg_factor, 0), "Q Gas %": round(pct, 1), "Bar": pct, "Oil sc.": round(max(0, r['g1']*r['wc']*bg_factor*0.1), 2), "Wtr sc.": round(max(0, r['g1']*r['wc']*bg_factor*0.9), 2), "Survey": "P1", "Choke": str(choke_p1)+"/64", "THP": thp_p1})
                report.append({"Sub Zone": "", "Top": np.nan, "Base": np.nan, "Pres.": r['p_s'], "Temp.": 142.1, "Q Gas Res.": round(r['g2'],2), "Q Gas sc.": round(r['g2']*bg_factor,0), "Q Gas %": round((r['g2']/total_q_gas_res*100),1) if total_q_gas_res > 0 else 0, "Bar": (r['g2']/total_q_gas_res*100) if total_q_gas_res > 0 else 0, "Oil sc.": "—", "Wtr sc.": "—", "Survey": "S1", "Choke": choke_s1, "THP": thp_s1})

            df_f = pd.DataFrame(report)

            # D. DISPLAY TABLE
            st.header("Step 2: Zonal Interpretation")
            def style_df(styler):
                p1_i = df_f[df_f['Survey'] == 'P1'].index; s1_i = df_f[df_f['Survey'] == 'S1'].index
                styler.set_properties(subset=pd.IndexSlice[p1_i, :], **{'background-color': '#FFF9E3', 'color': 'black'})
                styler.set_properties(subset=pd.IndexSlice[s1_i, :], **{'background-color': '#111111', 'color': 'white'})
                styler.set_properties(subset=['Bar'], **{'background': '#212121', 'color': 'white !important', 'font-weight': 'bold'})
                styler.bar(subset=['Bar'], color=['#cc0000', '#00FF00'], align='mid', vmin=-100, vmax=100)
                return styler
            st.dataframe(df_f.style.pipe(style_df), use_container_width=True, height=600)

            # E. ALL PLOTS
            st.header("Step 3: Visual Dashboard")
            v_grid = np.arange(max(p1_m.index.min(), s1_m.index.min()), min(p1_m.index.max(), s1_m.index.max()), 0.2)
            fig1, ax = plt.subplots(1, 3, figsize=(16, 11), sharey=True); fig1.patch.set_facecolor('#050505')
            for a in ax: a.set_facecolor('#0a1128'); a.tick_params(colors='white'); a.grid(alpha=0.1)
            ax[0].fill_betweenx(v_grid, np.interp(v_grid, p1_m.index, p1_m['PRS']), np.interp(v_grid, s1_m.index, s1_m['PRS']), color='#00d4ff', alpha=0.3); ax[0].set_title("PRESSURE FAN", color='#00d4ff')
            ax[1].fill_betweenx(v_grid, np.interp(v_grid, p1_m.index, p1_m['TMP']), np.interp(v_grid, s1_m.index, s1_m['TMP']), color='orange', alpha=0.3); ax[1].set_title("TEMPERATURE FAN")
            ax[2].fill_betweenx(p1_m.index, 0, p1_m['Q_FINAL'], color='lime', alpha=0.4); ax[2].set_title("PRODUCTION", color='lime')
            ax[0].invert_yaxis(); st.pyplot(fig1)

            # Zonal Contribution
            st.subheader("Zonal Performance (%)")
            df_plt = df_f[df_f['Survey']=="P1"][1:]
            fig2, ax2 = plt.subplots(figsize=(10, 5)); fig2.patch.set_facecolor('#050505'); ax2.set_facecolor('#0a1128')
            ax2.barh(df_plt['Sub Zone'], df_plt['Q Gas %'], color='lime', alpha=0.8); ax2.tick_params(colors='white')
            st.pyplot(fig2)

            # Crossplot
            st.header("Step 4: Calibration Physics")
            fig3, ax3 = plt.subplots(figsize=(8, 6)); fig3.patch.set_facecolor('#eee')
            s1_raw['L_mm'] = s1_raw['LSPD'] * 0.3048
            cal = s1_raw[(s1_raw['DEPTH']>=cal_top) & (s1_raw['DEPTH']<=cal_bot) & (s1_raw['L_mm'].abs()>2)]
            means = cal.groupby(['run_id', np.where(cal['L_mm'] > 0, 'UP', 'DOWN')]).mean(numeric_only=True).reset_index()
            ax3.scatter(means['L_mm'], means['CFM'], s=120, color='gold', edgecolors='black', zorder=5)
            res = linregress(means['L_mm'], means['CFM']); x_fit = np.linspace(-40, 40, 10); ax3.plot(x_fit, res.slope*x_fit + res.intercept, 'r-', label=f'R²={res.rvalue**2:.3f}'); ax3.legend()
            st.pyplot(fig3)

        except Exception as e: st.error(f"Engine Crash: {e}")