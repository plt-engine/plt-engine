import glob, os, warnings
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import lasio
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PLT Engine",
    page_icon="⛽",
    layout="wide"
)

st.markdown("""
<style>
  /* App background */
  .stApp { background-color: #08101e !important; }
  section[data-testid="stSidebar"] { background-color: #0b1628 !important; border-right: 1px solid #1a2d4a; }

  /* Sidebar section headers */
  .section-header {
    background: linear-gradient(90deg,#0d2444,#0a1a32);
    padding: 9px 14px; border-radius: 6px;
    font-weight: 700; color: #38c4f0; margin-top: 18px; font-size: 11px;
    border-left: 3px solid #38c4f0; text-transform: uppercase; letter-spacing: 1.2px;
  }
  .section-header-gold {
    background: linear-gradient(90deg,#1a1400,#0f1000);
    padding: 9px 14px; border-radius: 6px;
    font-weight: 700; color: #f0c040; margin-top: 18px; font-size: 11px;
    border-left: 3px solid #f0c040; text-transform: uppercase; letter-spacing: 1.2px;
  }

  /* Buttons */
  .stButton>button {
    background: #0f3460; color: #e0eeff; border: 1px solid #1a4a80;
    border-radius: 6px; font-weight: 600; padding: 8px 18px; transition: all 0.2s;
  }
  .stButton>button:hover { background: #174a80; border-color: #38c4f0; color: #fff; }

  /* Inputs */
  .stTextInput>div>div>input,
  .stNumberInput>div>div>input,
  .stTextArea textarea {
    background: #0d1b2e !important; color: #dce8f8 !important;
    border: 1px solid #1e3555 !important; border-radius: 5px !important;
  }
  label, .stCheckbox label p { color: #8aaac8 !important; font-size: 12px !important; }

  /* GOR info */
  .gor-info {
    background: #0f1a0a; border: 1px solid #2a4a1a; border-radius: 6px;
    padding: 8px 14px; color: #a0e060; font-size: 11px; margin: 6px 0;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] { background: #0b1628; border-radius: 8px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { color: #6888aa; font-weight: 600; }
  .stTabs [aria-selected="true"] { color: #38c4f0 !important; }

  /* Dividers */
  hr { border-color: #1a2d4a; }

  /* Section dividers between plot groups */
  .plot-section-title {
    border-top: 1px solid #1a2d4a; padding-top: 18px; margin-top: 24px;
    color: #38c4f0; font-size: 15px; font-weight: 700; letter-spacing: 0.5px;
  }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background:linear-gradient(135deg,#0f2040 0%,#0a1628 100%);
     padding:18px 24px;border-bottom:2px solid #f0c040;border-radius:8px;
     margin-bottom:24px;display:flex;align-items:center;gap:14px;">
  <span style="font-size:28px;">⛽</span>
  <div>
    <div style="color:#ffffff;font-size:20px;font-weight:700;letter-spacing:1px;">
      PLT ENGINE
    </div>
    <div style="color:#6899bb;font-size:12px;margin-top:2px;">
      Production Logging Tool · Zonal Contribution Analysis
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.expander("📖  USER GUIDE — click to open", expanded=False):
    st.markdown("""
<style>
.ug-section {
    background: #0d1f3c; border-left: 3px solid #38c4f0;
    border-radius: 0 6px 6px 0; padding: 12px 18px; margin: 18px 0 8px 0;
    color: #38c4f0; font-size: 13px; font-weight: 700; letter-spacing: 0.5px;
}
.ug-section span { color: #f0c040; margin-right: 8px; font-size: 15px; }
.ug-tip {
    background: #0a1e0a; border: 1px solid #1a4a1a; border-radius: 6px;
    padding: 10px 14px; margin: 8px 0; color: #70d070; font-size: 12px;
}
.ug-warn {
    background: #1e0f00; border: 1px solid #4a2a00; border-radius: 6px;
    padding: 10px 14px; margin: 8px 0; color: #f0a040; font-size: 12px;
}
.ug-error {
    background: #1e0808; border: 1px solid #4a1010; border-radius: 6px;
    padding: 10px 14px; margin: 8px 0; color: #f07070; font-size: 12px;
}
.ug-table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 12px; }
.ug-table th {
    background: #0d2444; color: #38c4f0; padding: 7px 10px;
    border: 1px solid #1e3555; text-align: left; font-weight: 700;
}
.ug-table td {
    background: #0a1628; color: #c8d8e8; padding: 6px 10px;
    border: 1px solid #1a2d40; vertical-align: top;
}
.ug-table tr:nth-child(even) td { background: #0d1c30; }
.ug-crit { color: #ff6060; font-weight: 700; }
.ug-med  { color: #f0c040; font-weight: 600; }
.ug-cos  { color: #70d070; }
.ug-code {
    background: #060e1a; border: 1px solid #1a2d40; border-radius: 5px;
    padding: 10px 14px; font-family: monospace; font-size: 12px;
    color: #a8d8a8; margin: 8px 0; white-space: pre;
}
.ug-check { color: #70d070; font-weight: 700; }
.ug-num {
    display: inline-block; background: #1a3a5a; color: #38c4f0;
    border-radius: 50%; width: 22px; height: 22px; text-align: center;
    line-height: 22px; font-size: 11px; font-weight: 700; margin-right: 6px;
}
</style>

<div style="color:#8aaac8;font-size:13px;margin-bottom:12px;">
Everything you need to upload LAS files, configure your well, and interpret zonal flow contributions.
</div>

<!-- ── 01 OVERVIEW ── -->
<div class="ug-section"><span>01</span> OVERVIEW</div>
<p style="color:#c8d8e8;font-size:13px;">
PLT Engine processes Production Logging Tool data from LAS files and calculates the gas, oil, and water
contribution of each perforated zone. It classifies flowing vs static passes automatically, calibrates depth,
converts spinner data to fluid velocity, and scales everything to match your surface measurements.
</p>
<div class="ug-tip">💡 <b>No installation required.</b> Just open the URL, upload your LAS files (or paste the folder path if running locally), fill in a few well parameters, and click Run.</div>

<b style="color:#dce8f8;">What you need before you start:</b>
<ul style="color:#c8d8e8;font-size:13px;margin-top:6px;">
  <li>A folder of <b>.las files</b> from the PLT survey (all passes — RIH, measurement, POOH)</li>
  <li>The <b>surface test report</b> (gas rate, oil rate, water rate, choke size, THP)</li>
  <li>The <b>completion report</b> (perforation depths and zone names)</li>
  <li>The <b>spinner calibration report</b> (slope, intercept, VPCF, tool offset)</li>
  <li>The <b>PVT report</b> for the gas expansion factor (Bg)</li>
</ul>

<!-- ── 02 QUICK START ── -->
<div class="ug-section"><span>02</span> QUICK START</div>
<ol style="color:#c8d8e8;font-size:13px;line-height:2.0;">
  <li><b>Open the app</b> — Chrome, Edge, or Firefox. No login required.</li>
  <li><b>Upload your LAS files</b> — use the Upload tab below to drag and drop all .las files, or paste your folder path in the Local Path tab.</li>
  <li><b>Set Source parameters</b> — enter the well name and Bg from the PVT report in the sidebar.</li>
  <li><b>Add surface test rows</b> — fill in P1 with choke, THP, Qgas, Qoil, Qwtr. Add P2/P3 rows for multiple flow tests.</li>
  <li><b>Set anchor depths and perforations</b> — top/base anchors, spinner calibration values, borehole ID, and the perf list.</li>
  <li><b>Click ▶ RUN PLT ENGINE</b> — the engine classifies files, calibrates depth, computes flow profiles, and produces the output table and all plots.</li>
</ol>

<!-- ── 03 UPLOADING DATA ── -->
<div class="ug-section"><span>03</span> UPLOADING YOUR DATA</div>
<p style="color:#c8d8e8;font-size:13px;">Drop the <b>entire survey folder</b> — do not manually filter files. The app automatically classifies each file:</p>
<table class="ug-table">
  <tr><th>File type</th><th>Example name</th><th>Depth range</th><th>Handled as</th></tr>
  <tr><td>Measurement pass (up or down)</td><td>P1U1.las, S1D3.las</td><td>Perf zone only</td><td>✅ Used for analysis</td></tr>
  <tr><td>Full wellbore POOH</td><td>P1_POOH.las</td><td>Surface → TD</td><td>✅ Used for analysis</td></tr>
  <tr><td>RIH transit (above perfs)</td><td>P1_RIH1.las</td><td>Surface → ~4200 m</td><td>🚀 Auto-excluded</td></tr>
  <tr><td>Deep RIH reaching perfs</td><td>P1_RIH2.las</td><td>4230 m → TD</td><td>✅ Used for analysis</td></tr>
  <tr><td>Calibration pass</td><td>CAL.las</td><td>Below perfs</td><td>✅ Used for spinner cal</td></tr>
</table>
<div class="ug-warn">⚠️ <b>Transit files are excluded automatically.</b> Any file whose max depth does not reach within 100 m of your Top Anchor Depth is removed. The log panel lists which files were excluded.</div>

<!-- ── 04 SOURCE SETTINGS ── -->
<div class="ug-section"><span>04</span> SOURCE SETTINGS</div>
<table class="ug-table">
  <tr><th>Field</th><th>What to enter</th><th>Impact</th></tr>
  <tr><td>Well Name</td><td>Name used on all plot titles and the table header</td><td class="ug-cos">Cosmetic</td></tr>
  <tr><td>Gas Expansion Factor (Bg)</td><td>From the PVT report. Converts reservoir ↔ surface volumes. Typical: 50–200 m³/m³</td><td class="ug-crit">Critical</td></tr>
</table>
<div class="ug-error">🚨 <b>Bg is the most commonly wrong value.</b> The default (113.7) is specific to one well. Using the wrong Bg scales every gas volume incorrectly. Always get this from the PVT report for the specific well.</div>

<!-- ── 05 SURFACE TESTS ── -->
<div class="ug-section"><span>05</span> SURFACE TESTS</div>
<table class="ug-table">
  <tr><th>Column</th><th>What to enter</th><th>Notes</th></tr>
  <tr><td>Survey</td><td>P1, P2, P3, S1, or S2</td><td>P1/P2/P3 = flowing. S1/S2 = shut-in. Shut-in rows never drive calculations.</td></tr>
  <tr><td>Choke</td><td>e.g. <code>36/64"</code> or <code>56/67"</code> or plain <code>50</code></td><td>Any fraction X/Y" accepted. Plain integer like 50 treated as 50/64. Base reference is 36/64".</td></tr>
  <tr><td>THP (Bar)</td><td>Tubing head pressure at time of test</td><td>Reference is 77.1 Bar.</td></tr>
  <tr><td>Qgas SC (m³/d)</td><td>Surface gas rate from separator / meter</td><td><b>Required.</b> Physical anchor — cannot be derived from LAS.</td></tr>
  <tr><td>Qoil SC (m³/d)</td><td>Surface oil rate</td><td>Sets GOR = Qoil/Qgas. Auto-fills on non-P1 rows when GOR lock is ON.</td></tr>
  <tr><td>Qwtr SC (m³/d)</td><td>Surface water rate</td><td>Sets WGR = Qwtr/Qgas. Same auto-fill logic.</td></tr>
  <tr><td>Use for scaling</td><td>Tick on flowing rows only</td><td>Determines which row drives the reservoir target.</td></tr>
</table>
<div class="ug-tip">✅ <b>GOR/WOR Lock:</b> When enabled, changing Qgas on P2/P3 rows auto-fills Qoil and Qwtr using P1 ratios. All output oil and water values use these ratios.</div>

<!-- ── 06 PHYSICS PARAMETERS ── -->
<div class="ug-section"><span>06</span> PHYSICS PARAMETERS</div>
<table class="ug-table">
  <tr><th>Parameter</th><th>Source</th><th>Impact</th></tr>
  <tr><td>Top Anchor Depth (m)</td><td>Shallowest perf top (completion report)</td><td class="ug-crit">Critical</td></tr>
  <tr><td>Bot Anchor Depth (m)</td><td>Deepest perf base (completion report)</td><td class="ug-crit">Critical</td></tr>
  <tr><td>Spinner Slope</td><td>Spinner calibration report</td><td class="ug-crit">Critical</td></tr>
  <tr><td>Spinner Intercept</td><td>Spinner calibration report</td><td class="ug-crit">Critical</td></tr>
  <tr><td>VPCF Factor</td><td>Service company PLT report</td><td class="ug-crit">Critical</td></tr>
  <tr><td>Tool Offset (m)</td><td>Tool string diagram (P/T sensor → spinner distance)</td><td class="ug-crit">Critical</td></tr>
  <tr><td>Borehole ID (in)</td><td>Wellbore schematic at perf depth (casing ID)</td><td class="ug-crit">Critical</td></tr>
  <tr><td>Cal Zone Top / Bot (m)</td><td>Below-perf interval in LAS data</td><td class="ug-cos">Plot only</td></tr>
</table>
<div class="ug-warn">⚠️ <b>Anchor Depths also control transit file filtering.</b> Any LAS file with max depth &lt; Top Anchor − 100 m is excluded. Set anchors to the actual perforated interval before running.</div>

<!-- ── 07 PERFORATIONS ── -->
<div class="ug-section"><span>07</span> PERFORATIONS</div>
<p style="color:#c8d8e8;font-size:13px;">One zone per line — <code>top_depth, base_depth, zone_name</code></p>
<div class="ug-code">4333.0, 4335.0, B-1
4342.0, 4345.0, B-2
4881.5, 4884.5, M-1
5031.0, 5040.0, A-1</div>
<ul style="color:#c8d8e8;font-size:13px;">
  <li>Depths in <b>metres MD</b></li>
  <li>Zones outside the data depth range are <b>automatically skipped</b> with a warning</li>
  <li>Order does not matter — sorted by depth internally</li>
</ul>

<!-- ── 08 OUTPUT TABLE ── -->
<div class="ug-section"><span>08</span> OUTPUT TABLE</div>
<p style="color:#c8d8e8;font-size:13px;">Each perforated zone produces two rows: P1 (flowing) and S1 (shut-in reference).</p>
<table class="ug-table">
  <tr><th>Column</th><th>Formula</th><th>Notes</th></tr>
  <tr><td>Press. (kPa)</td><td>interp(top_depth, PRS)</td><td>Pressure at perf top. Higher for S1 (shut-in).</td></tr>
  <tr><td>Temp. (°C)</td><td>interp(top_depth, TMP)</td><td>Temperature at perf top.</td></tr>
  <tr><td>Q Gas Res. (m³/d)</td><td>Q(top−0.5) − Q(base+0.5)</td><td>Zone gas at reservoir conditions. <b>Negative = thief zone.</b></td></tr>
  <tr><td>Q Gas SC (m³/d)</td><td>Q_res × Bg</td><td>Zone gas at surface standard conditions.</td></tr>
  <tr><td>Q Gas %</td><td>Q_zone / Total × 100</td><td>Must sum to ~100% across all P1 zone rows.</td></tr>
  <tr><td>Q Oil SC (m³/d)</td><td>Q_gas_SC × GOR</td><td>Derived from GOR. Blank for S1 rows.</td></tr>
  <tr><td>Q Wtr SC (m³/d)</td><td>Q_gas_SC × WGR</td><td>Derived from WGR. Blank for S1 rows.</td></tr>
</table>
<b style="color:#dce8f8;font-size:12px;">Row colour coding:</b>
<table class="ug-table" style="margin-top:6px;">
  <tr><th>Colour</th><th>Meaning</th></tr>
  <tr><td>🟩 Green tint</td><td>Composite (Total) — P1 flowing. Sum of all zones.</td></tr>
  <tr><td>🟦 Blue tint</td><td>Composite — S1 shut-in.</td></tr>
  <tr><td>⬜ White</td><td>Individual zone — P1 flowing.</td></tr>
  <tr><td>🩶 Light grey</td><td>Individual zone — S1 shut-in. Pressure and residual flow only.</td></tr>
</table>

<!-- ── 09 PLOTS ── -->
<div class="ug-section"><span>09</span> PLOTS & CHARTS</div>
<table class="ug-table">
  <tr><th>Plot</th><th>What to look for</th></tr>
  <tr><td>📊 5-Track Multi-Log</td><td>Pressure, Temperature, Drawdown ΔP, Flow Profile Q, Zone % on a shared depth axis. <span class="ug-check">Q profile should show clear steps at contributing zones.</span></td></tr>
  <tr><td>📏 Zonal Contribution Ranking</td><td>Bars sorted by %. Red bars = negative contributors (thief zones). <span class="ug-check">Should sum to ~100%.</span></td></tr>
  <tr><td>🥧 Contribution Share Pie</td><td>Positive zones only. Good for presentations. <span class="ug-check">Confirms which zones dominate.</span></td></tr>
  <tr><td>🌡️ P-T Crossplot</td><td>Pressure vs Temperature coloured by depth. <span class="ug-check">A clean linear trend = good data quality. Outliers = depth calibration issues.</span></td></tr>
  <tr><td>🌊 Cumulative Flow Waterfall</td><td>Q_gas_SC built zone by zone. <span class="ug-check">TOTAL bar should match your Qgas SC input.</span></td></tr>
  <tr><td>🎯 Drawdown vs Flow</td><td>Mean ΔP vs zone Q with PI trendline. <span class="ug-check">Points far from line = anomalous zones.</span></td></tr>
  <tr><td>🔧 Spinner Calibration</td><td>Raw data + pass averages + regression. <span class="ug-check">R² should be &gt; 0.995.</span></td></tr>
  <tr><td>⚖️ Surface Test Comparison</td><td>Only appears with 2+ active flowing tests. Multi-fluid bar and Qgas vs THP scatter.</td></tr>
</table>

<!-- ── 10 FORMULAS ── -->
<div class="ug-section"><span>10</span> FORMULAS REFERENCE</div>
<table class="ug-table">
  <tr><th>Formula</th><th>Expression</th></tr>
  <tr><td>Reservoir Gas Target</td><td><code>Q_res = (Qgas / Bg) × (choke_eq / 36)² × (THP / 77.1)</code><br><small style="color:#8aaac8;">choke_eq = any X/Y" converted to /64-equivalent units. e.g. 36/64"→36, 56/67"→53.5, plain 50→50</small></td></tr>
  <tr><td>Spinner → Fluid Velocity</td><td><code>v = (CFM − intercept) / (slope × 60) − |LSPD × 0.00508|</code></td></tr>
  <tr><td>Borehole Cross-Section</td><td><code>area = π × (BH_ID_in × 0.0254 / 2)²</code></td></tr>
  <tr><td>Gas Flow Rate at Depth</td><td><code>Q = v × area × VPCF × 86400  (m³/d)</code></td></tr>
  <tr><td>Zonal Contribution</td><td><code>Q_zone = Q(top − 0.5) − Q(base + 0.5)</code></td></tr>
  <tr><td>Contribution %</td><td><code>% = Q_zone / Q_res_total × 100</code></td></tr>
  <tr><td>Depth Calibration Stretch</td><td><code>depth_cal = (depth_raw − r_l) × (bot − top) / (r_h − r_l) + top</code></td></tr>
</table>

<!-- ── 11 ERRORS ── -->
<div class="ug-section"><span>11</span> ERROR MESSAGES & FIXES</div>
<div class="ug-error">🔴 <b>No .las files found</b><br>
<b>Cause:</b> No files with .las or .LAS extension in the folder.<br>
<b>Fix:</b> Drop the folder containing the LAS files directly, not a parent folder or zip.</div>
<div class="ug-error">🔴 <b>missing cols: ['CFM']</b><br>
<b>Cause:</b> The spinner column name doesn't match known keywords (CFM, SPIN, RPS, etc.)<br>
<b>Fix:</b> Check the LAS file header for the exact column name.</div>
<div class="ug-error">🔴 <b>No active flowing surveys</b><br>
<b>Cause:</b> No row in the Surface Tests table has "Use for scaling" ticked.<br>
<b>Fix:</b> Tick "Use for scaling" on the P1 row.</div>
<div class="ug-error">🔴 <b>Only 0 measurement file(s) after filtering</b><br>
<b>Cause:</b> All files have max depth below anchor_top − 100 m. Top Anchor Depth is set too deep.<br>
<b>Fix:</b> Update Top Anchor Depth to match the actual perforated interval. Check depth ranges in the log.</div>
<div class="ug-warn">🟡 <b>Perfs outside data range skipped</b><br>
Perforation depths don't match the LAS data depth range. Update the Perforations widget with correct depths from the completion report.</div>
<div class="ug-warn">🟡 <b>P1 Q at anchor top ≈ 0 — scaling skipped</b><br>
Q profile at top anchor is near zero. Verify the correct file was selected as flowing. Try adjusting Top Anchor Depth.</div>
<div class="ug-warn">🟡 <b>anchors collapsed — using raw depth</b><br>
Pressure values at top/bottom anchors are nearly identical. Check that anchor depths correspond to points with distinctly different pressures.</div>

<!-- ── 12 NEW WELL CHECKLIST ── -->
<div class="ug-section"><span>12</span> NEW WELL CHECKLIST</div>
<p style="color:#c8d8e8;font-size:13px;">Every field defaults to an example well — update all of these for each new well:</p>
<table class="ug-table">
  <tr><th>✓</th><th>What to update</th><th>Why it matters</th></tr>
  <tr><td>☐</td><td>Upload the correct LAS folder for this well</td><td>Common mistake: still pointed at the previous well's folder</td></tr>
  <tr><td>☐</td><td>Update Well Name</td><td>Appears on all plot titles and the table header</td></tr>
  <tr><td>☐</td><td>Update <b>Bg</b> from the new well's PVT report</td><td>Wrong Bg scales every gas volume incorrectly</td></tr>
  <tr><td>☐</td><td>Update <b>Qgas, Qoil, Qwtr</b> in the P1 row</td><td>Old surface rates change all zone allocations</td></tr>
  <tr><td>☐</td><td>Update <b>Choke and THP</b> in the P1 row</td><td>Controls the reservoir target scaling formula</td></tr>
  <tr><td>☐</td><td>Update <b>Top and Bot Anchor Depths</b></td><td>Controls depth calibration AND transit file filtering</td></tr>
  <tr><td>☐</td><td>Update <b>Perforations</b> with zone depths from the completion report</td><td>Zone depths are well-specific</td></tr>
  <tr><td>☐</td><td>Update <b>Spinner Slope and Intercept</b> from calibration certificate</td><td>Calibration is tool-specific and run-specific. Check R² &gt; 0.995.</td></tr>
  <tr><td>☐</td><td>Update <b>VPCF and Tool Offset</b> from the service company PLT report</td><td>Depends on borehole geometry and tool configuration</td></tr>
  <tr><td>☐</td><td>Update <b>Borehole ID</b> from the wellbore schematic at perf depth</td><td>Wrong ID changes cross-sectional area, scaling all Q values</td></tr>
  <tr><td>☐</td><td>After running, check the <b>log panel</b> to confirm P1 and S1 files are correct</td><td>If the wrong file is selected as flowing, zone percentages will be wrong</td></tr>
  <tr><td>☐</td><td>Verify the <b>transit exclusion list</b> in the log panel</td><td>If measurement files are being excluded, anchor depth may be set too deep</td></tr>
  <tr><td>☐</td><td>Check <b>Q % column sums to ~100%</b> across all P1 zone rows</td><td>Large deviation = scaling or depth calibration issue</td></tr>
</table>
<div class="ug-tip">✅ <b>Quick validation:</b> Composite row Gas SC should match the Qgas you entered. Composite Oil = sum of all zone oil rows. Spinner calibration R² &gt; 0.995. Cumulative waterfall TOTAL ≈ Qgas SC. If all four pass — you're good.</div>

<div style="color:#3a5a7a;font-size:11px;text-align:right;margin-top:20px;border-top:1px solid #1a2d4a;padding-top:10px;">
PLT Engine v8 · User Guide · Production Log Tool · Multi-Well Automated Analysis
</div>
""", unsafe_allow_html=True)

if 'las_folder_path' not in st.session_state:
    st.session_state.las_folder_path = r"E:\datas\LAS"

if 'surface_rows' not in st.session_state:
    st.session_state.surface_rows = [
        {'survey': 'P1', 'choke': '36/64"', 'thp': 77.1,  'qgas': 165543.41, 'qoil': 9.23,  'qwtr': 29.0,  'active': True},
        {'survey': 'S1', 'choke': 'Closed', 'thp': 119.2, 'qgas': 0.0,       'qoil': 0.0,   'qwtr': 0.0,   'active': False},
    ]

def get_p1_row():
    for r in st.session_state.surface_rows:
        if r['survey'] == 'P1':
            return r
    return st.session_state.surface_rows[0] if st.session_state.surface_rows else None

def compute_gor_wgr():
    p1 = get_p1_row()
    if p1 and p1['qgas'] > 0:
        return p1['qoil'] / p1['qgas'], p1['qwtr'] / p1['qgas']
    return 0.0, 0.0

with st.sidebar:
    st.markdown('<div class="section-header">1 · SOURCE</div>', unsafe_allow_html=True)
    well_name  = st.text_input("Well Name",         value="Well-01")
    las_folder = st.text_input("LAS Folder Path",
                                value=st.session_state.las_folder_path,
                                key="las_folder_input",
                                help="Type/paste path, or use the drag-drop zone on the main page")
    bg_val     = st.number_input("Gas Expansion (Bg)", value=113.695784, format="%.6f", key="bg_val")

    st.markdown('<div class="section-header-gold">2 · SURFACE TESTS (dynamic)</div>', unsafe_allow_html=True)
    st.caption("S1/S2 rows are shut-in. Tick 'Active' only on flowing rows. Enable GOR/WOR lock to auto-scale oil & water.")

    gor_lock = st.checkbox("Auto-scale Oil & Water from P1 GOR/WOR", value=True)
    gor, wgr = compute_gor_wgr()
    if gor_lock:
        st.markdown(
            f'<div class="gor-info">📊 From P1 &nbsp;|&nbsp; '
            f'GOR = <b>{gor*1000:.4f}</b> m³ oil/1000m³ gas &nbsp;|&nbsp; '
            f'WGR = <b>{wgr*1000:.4f}</b> m³ water/1000m³ gas</div>',
            unsafe_allow_html=True)

    SURVEY_OPTIONS = ['P1','P2','P3','S1','S2']
    rows_to_delete = []

    for i, row in enumerate(st.session_state.surface_rows):
        with st.expander(f"Row {i+1} — {row['survey']}", expanded=True):
            cols = st.columns([1.2, 1.2, 1.2])
            row['survey'] = cols[0].selectbox("Survey", SURVEY_OPTIONS,
                                               index=SURVEY_OPTIONS.index(row['survey']),
                                               key=f"survey_{i}")
            row['choke']  = cols[1].text_input("Choke",  value=row['choke'],  key=f"choke_{i}")
            row['thp']    = cols[2].number_input("THP (Bar)", value=float(row['thp']), key=f"thp_{i}", format="%.2f")

            cols2 = st.columns([1.3, 1.3, 1.3])
            row['qgas'] = cols2[0].number_input("Qgas SC (m³/d)", value=float(row['qgas']),
                                                 key=f"qgas_{i}", format="%.2f")

            is_p1_row = (row['survey'] == 'P1')
            is_static  = row['survey'] in ('S1','S2')
            locked     = gor_lock and not is_p1_row and not is_static
            gor_now, wgr_now = compute_gor_wgr()
            if locked:
                row['qoil'] = round(row['qgas'] * gor_now, 4)
                row['qwtr'] = round(row['qgas'] * wgr_now, 4)
            row['qoil'] = cols2[1].number_input("Qoil SC (m³/d)", value=float(row['qoil']),
                                                  key=f"qoil_{i}", format="%.4f",
                                                  disabled=locked)
            row['qwtr'] = cols2[2].number_input("Qwtr SC (m³/d)", value=float(row['qwtr']),
                                                  key=f"qwtr_{i}", format="%.4f",
                                                  disabled=locked)

            col_act, col_del = st.columns([3, 1])
            row['active'] = col_act.checkbox("Use for scaling", value=row['active'],
                                               key=f"active_{i}")
            if col_del.button("✕ Del", key=f"del_{i}"):
                rows_to_delete.append(i)

    for idx in sorted(rows_to_delete, reverse=True):
        st.session_state.surface_rows.pop(idx)
        st.rerun()

    if st.button("＋ Add Survey Row"):
        p1 = get_p1_row()
        new_qgas = 120000.0
        if p1:
            try:
                c1 = parse_choke(p1['choke'])
                new_qgas = round(p1['qgas'] * (32/c1)**2 * (65/p1['thp']), 2)
            except: pass
        st.session_state.surface_rows.append({
            'survey':'P2','choke':'32/64"','thp':65.0,
            'qgas':new_qgas,'qoil':0.0,'qwtr':0.0,'active':True
        })
        st.rerun()

    st.markdown('<div class="section-header">3 · SHUT-IN REFERENCE</div>', unsafe_allow_html=True)
    choke_s1 = st.text_input("Shut-in Choke",      value="Closed",  key="choke_s1")
    thp_s1   = st.number_input("Shut-in THP (Bar)", value=119.2, format="%.2f", key="thp_s1")

    st.markdown('<div class="section-header">4 · PHYSICS</div>', unsafe_allow_html=True)
    d_top  = st.number_input("Top Anchor Depth (m)", value=4333.0, format="%.1f",  key="d_top")
    d_bot  = st.number_input("Bot Anchor Depth (m)", value=5031.0, format="%.1f",  key="d_bot")
    sp_m   = st.number_input("Spinner Slope",         value=0.042,  format="%.4f", key="sp_m")
    sp_c   = st.number_input("Spinner Intercept",     value=0.1,    format="%.4f", key="sp_c")
    vpcf   = st.number_input("VPCF Factor",           value=0.83,   format="%.4f", key="vpcf")
    offset = st.number_input("Tool Offset (m)",       value=18.5,   format="%.2f", key="offset")
    bh_id  = st.number_input("Borehole ID (in)",      value=6.0,    format="%.2f", key="bh_id")

    perfs_default = (
        "4333.0, 4335.0, B-1\n4342.0, 4345.0, B-2\n4351.0, 4354.0, B-3\n"
        "4389.0, 4393.0, B-4\n4405.0, 4421.0, B-5\n4438.0, 4441.0, B-6\n"
        "4449.5, 4452.5, B-7\n4881.5, 4884.5, M-1\n4891.0, 4897.0, M-2\n"
        "4910.0, 4922.0, M3\n4929.0, 4932.0, M-4\n5031.0, 5040.0, A-1"
    )
    perfs_txt = st.text_area("Perfs (Top, Bot, Name)", value=perfs_default, height=260)
    cal_top   = st.number_input("Cal Zone Top (m)", value=5031.0, format="%.1f", key="cal_top")
    cal_bot   = st.number_input("Cal Zone Bot (m)", value=5055.0, format="%.1f", key="cal_bot")

st.markdown("""
<div style="background:#0d1f3c;border:1px solid #1e3a5f;border-radius:8px;padding:16px 20px;margin-bottom:16px;">
  <div style="color:#00d4ff;font-size:14px;font-weight:bold;margin-bottom:8px;">📂 LAS FILES</div>
  <div style="color:#8899aa;font-size:12px;">
    Upload your <b>.las</b> files directly, <b>or</b> type / paste the folder path if running locally.
  </div>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_path = st.tabs(["⬆️  Upload LAS files", "📁  Local folder path"])

with tab_upload:
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    uploaded_las = st.file_uploader(
        "Drop .las files here or click Browse",
        type=["las", "LAS"],
        accept_multiple_files=True,
        help="Select all your LAS files at once (Ctrl+click or Cmd+click to multi-select)",
        key=f"las_uploader_{st.session_state.uploader_key}"
    )
    if uploaded_las:
        import tempfile
        _tmp_dir = tempfile.mkdtemp(prefix="plt_las_")
        for uf in uploaded_las:
            with open(os.path.join(_tmp_dir, uf.name), "wb") as _f:
                _f.write(uf.read())
        st.session_state.las_folder_path = _tmp_dir
        st.success(f"✅  {len(uploaded_las)} file(s) loaded — ready to run.")

with tab_path:
    typed = st.text_input(
        "LAS folder path",
        value=st.session_state.las_folder_path,
        placeholder=r"e.g.  C:\Users\you\LAS_files  or  /home/user/las",
        help="Right-click the folder in Windows Explorer → 'Copy as path', then paste here"
    )
    if typed.strip() and typed.strip() != st.session_state.las_folder_path:
        st.session_state.las_folder_path = typed.strip()
    if st.session_state.las_folder_path:
        n = len(glob.glob(os.path.join(st.session_state.las_folder_path,'*.las'))
                + glob.glob(os.path.join(st.session_state.las_folder_path,'*.LAS')))
        if n:
            st.success(f"✅  Found {n} .las file(s) in that folder.")
        else:
            st.warning("⚠️  No .las files found at that path yet.")

las_folder = st.session_state.las_folder_path

# ── Choke parser — flexible, not hardcoded to /64 ─────────────
def parse_choke(choke_str):
    """Parses choke string to numerator value (same units as before).
    Accepts: '36/64"' → 36/64*64=36, '56/67"' → 56/67*64,
    plain int '50' → 50, decimal '0.5625' → 0.5625*64=36
    Always returns the value in /64-equivalent units for the formula."""
    try:
        s = str(choke_str).replace('"','').replace("'",'').strip()
        if not s or s.lower() in ('closed','shut','-','n/a'):
            return None
        parts = s.split('/')
        if len(parts) == 2:
            num, den = float(parts[0]), float(parts[1])
            return num / den * 64.0
        val = float(parts[0])
        if val <= 1.0:
            return val * 64.0
        return val
    except Exception:
        return None

col_run, col_reset = st.columns([5, 1])
with col_run:
    run_clicked = st.button("▶  RUN PLT ENGINE", use_container_width=True, type="primary")
with col_reset:
    if st.button("↺  Reset", use_container_width=True):
        st.session_state.las_folder_path = ""
        st.session_state.uploader_key = st.session_state.get('uploader_key', 0) + 1
        st.rerun()

if run_clicked:
    log = st.empty()
    log_lines = []

    def logp(msg):
        log_lines.append(msg)
        log.text("\n".join(log_lines))

    try:
        surface_rows = st.session_state.surface_rows

        flowing_tests = [r for r in surface_rows
                         if r['survey'] not in ('S1','S2') and r['active']]
        static_tests  = [r for r in surface_rows if r['survey'] in ('S1','S2')]
        if not flowing_tests:
            st.error('❌  No active flowing surveys. Tick "Use for scaling" on at least one row.')
            st.stop()

        prim  = flowing_tests[0]
        c_num = parse_choke(prim['choke'])
        SIM_TARGET_RES = (prim['qgas']/bg_val)*(c_num/36.0)**2*(prim['thp']/77.1)
        S1_STABLE_REF  = prim['qgas'] / bg_val

        qg_p1 = prim['qgas']
        GOR   = prim['qoil'] / qg_p1 if qg_p1 > 0 else 0.0
        WGR   = prim['qwtr'] / qg_p1 if qg_p1 > 0 else 0.0

        logp('📊 Surface tests:')
        logp(f'   GOR = {GOR*1000:.4f} m³ oil/1000m³ gas   WGR = {WGR*1000:.4f} m³ water/1000m³ gas')
        for ft in flowing_tests:
            cn  = parse_choke(ft['choke'])
            tgt = (ft['qgas']/bg_val)*(cn/36.0)**2*(ft['thp']/77.1)
            logp(f"   ✅ {ft['survey']:3s} | Choke={ft['choke']:8s} | THP={ft['thp']:.1f} | "
                 f"Qgas={ft['qgas']:.0f} m³/d | Qoil={ft['qoil']:.3f} | Qwtr={ft['qwtr']:.3f} | Q_res={tgt:.3f} m³/d")
        for st_t in static_tests:
            logp(f"   🔵 {st_t['survey']:3s} | Choke={st_t['choke']} | THP={st_t['thp']:.1f}")
        logp(f'\n   ▶ Primary Q_res = {SIM_TARGET_RES:.4f} m³/d (Bg={bg_val})\n')

        files = list(set(
            glob.glob(os.path.join(las_folder,'*.las')) +
            glob.glob(os.path.join(las_folder,'*.LAS'))
        ))
        if not files:
            st.error('❌  No .las files found in the specified folder.')
            st.stop()

        DEPTH_KW = ['DEPT','DEPTH','MD','MDEPTH','MEASURED']
        PRESS_KW = ['PRS','PRES','PRESSURE','BPRS','DOWNPRS','BHPRS','BHP','P_']
        SPIN_KW  = ['CFM','SPIN','RPS','SPINNERRATE','SPINNER_RATE','SFLO','SFLOW','SP_']
        LSPD_KW  = ['LSPD','CBLSPD','CABSPD','LOGSPD','LOG_SPEED','TOOLSPD','CABLE','SPEED']
        TEMP_KW  = ['TMP','TEMP','TEMPERATURE','DOWNTEMP','BTEMP','BHT']

        def find_col(cols, kws):
            for kw in kws:
                for c in cols:
                    if kw in c.upper(): return c
            return None

        logp(f'📂 Loading from: {las_folder}')
        all_runs = []
        for f in files:
            try:
                las  = lasio.read(f)
                df   = las.df().reset_index()
                cols = list(df.columns)
                d_col = find_col(cols, DEPTH_KW) or cols[0]
                p_col = find_col(cols, PRESS_KW)
                s_col = find_col(cols, SPIN_KW)
                l_col = find_col(cols, LSPD_KW)
                t_col = find_col(cols, TEMP_KW)
                miss  = [n for n,c in [('PRS',p_col),('CFM',s_col),('LSPD',l_col),('TMP',t_col)] if c is None]
                if miss:
                    logp(f'   ⚠️  {os.path.basename(f):30s} → missing: {miss} — skipped')
                    continue
                df = df.rename(columns={d_col:'DEPTH',p_col:'PRS',s_col:'CFM',l_col:'LSPD',t_col:'TMP'})
                bad = [c for c in ['DEPTH','PRS','CFM','LSPD','TMP'] if df[c].dropna().empty]
                if bad:
                    logp(f'   ⚠️  {os.path.basename(f):30s} → all-NaN: {bad} — skipped')
                    continue
                df = df.sort_values('DEPTH').reset_index(drop=True)
                df['fname'] = os.path.basename(f)
                near = (df['DEPTH'] - d_top).abs() < 5
                pv   = df.loc[near,'PRS'].dropna()
                df['p_probe'] = pv.median() if not pv.empty else np.nan
                all_runs.append(df.dropna(subset=['DEPTH']).copy())
            except Exception as fe:
                logp(f'   ⚠️  Load error {os.path.basename(f)}: {fe}')

        if len(all_runs) < 2:
            st.error(f'❌  Need ≥2 usable files, found {len(all_runs)}.')
            st.stop()

        REACH_THRESHOLD = d_top - 100
        CLIP_TOP = d_top - 200
        CLIP_BOT = d_bot + 300

        measurement_runs, transit_runs = [], []
        for df in all_runs:
            dmax = df['DEPTH'].max()
            if dmax >= REACH_THRESHOLD:
                measurement_runs.append(df)
            else:
                transit_runs.append(df)

        logp(f'\n📋 File classification (anchor top = {d_top:.0f} m, threshold = {REACH_THRESHOLD:.0f} m):')
        logp(f'   ✅ Measurement passes : {len(measurement_runs)} files')
        logp(f'   🚀 Transit/RIH passes : {len(transit_runs)} files (excluded from analysis)')
        for df in transit_runs:
            logp(f'       {df["fname"].iloc[0]:30s}  [{df["DEPTH"].min():.0f}–{df["DEPTH"].max():.0f} m]')

        if len(measurement_runs) < 2:
            st.error(f'❌  Only {len(measurement_runs)} measurement file(s) after filtering.')
            st.stop()

        meas_mins, meas_maxs = [], []
        for df in measurement_runs:
            in_zone = df[(df['DEPTH'] >= CLIP_TOP) & (df['DEPTH'] <= CLIP_BOT)]
            if len(in_zone) >= 5:
                meas_mins.append(in_zone['DEPTH'].min())
                meas_maxs.append(in_zone['DEPTH'].max())
            else:
                meas_mins.append(df['DEPTH'].min())
                meas_maxs.append(df['DEPTH'].max())

        common_top_d = max(meas_mins)
        common_bot_d = min(meas_maxs)

        if common_top_d >= common_bot_d:
            common_top_d = min(meas_mins)
            common_bot_d = max(meas_maxs)
            logp(f'\n⚠️  No strict overlap — using union: {common_top_d:.1f}–{common_bot_d:.1f} m')
        else:
            logp(f'\n📐 Measurement zone overlap: {common_top_d:.1f} – {common_bot_d:.1f} m')

        span = common_bot_d - common_top_d
        logp(f'    Span = {span:.1f} m')

        def get_p_probe(df):
            v = df['p_probe'].dropna()
            if not v.empty and not np.isnan(v.iloc[0]):
                return float(v.iloc[0])
            in_zone = df[(df['DEPTH'] >= CLIP_TOP) & (df['DEPTH'] <= CLIP_BOT)]
            pv2 = in_zone['PRS'].dropna()
            logp(f'   ⚠️  {df["fname"].iloc[0]}: p_probe NaN — using measurement zone median for sort')
            return float(pv2.median()) if not pv2.empty else 0.0

        measurement_runs.sort(key=get_p_probe)
        flowing_raw = measurement_runs[0]
        static_raw  = measurement_runs[-1]
        logp(f'\n   ✅ Flowing run : {flowing_raw["fname"].iloc[0]}  (p_probe={get_p_probe(flowing_raw):.1f} kPa)')
        logp(f'   ✅ Static  run : {static_raw["fname"].iloc[0]}  (p_probe={get_p_probe(static_raw):.1f} kPa)\n')

        def process_physics(raw_df, label=''):
            m_raw = raw_df.groupby('DEPTH').mean(numeric_only=True).sort_index()

            ref_l = m_raw.loc[d_top-5 : d_top+15, 'PRS'].max()
            ref_h = m_raw.loc[d_bot-5 : d_bot+15, 'PRS'].max()

            if pd.isna(ref_l):
                win = max(span*0.05, 20)
                lo  = max(d_top-win, m_raw.index.min())
                hi  = min(d_top+win, m_raw.index.max())
                ref_l = m_raw.loc[lo:hi,'PRS'].max()
                if pd.isna(ref_l):
                    ref_l = m_raw['PRS'].iloc[:max(1,len(m_raw)//10)].max()
                logp(f'   ⚠️  {label}: top anchor ±5m empty → widened to ±{win:.0f} m')
            if pd.isna(ref_h):
                win = max(span*0.05, 20)
                lo  = max(d_bot-win, m_raw.index.min())
                hi  = min(d_bot+win, m_raw.index.max())
                ref_h = m_raw.loc[lo:hi,'PRS'].max()
                if pd.isna(ref_h):
                    ref_h = m_raw['PRS'].iloc[-max(1,len(m_raw)//10):].max()
                logp(f'   ⚠️  {label}: bot anchor ±5m empty → widened to ±{win:.0f} m')

            r_l = (m_raw['PRS'] - ref_l).abs().idxmin()
            r_h = (m_raw['PRS'] - ref_h).abs().idxmin()

            if abs(r_h - r_l) < 1.0:
                logp(f'   ⚠️  {label}: anchors collapsed — using raw depth.')
                m = m_raw.copy()
            else:
                df2 = raw_df.copy()
                df2['DEPTH'] = ((df2['DEPTH']-r_l)
                                * ((d_bot-d_top)/(r_h-r_l))
                                + d_top)
                m = df2.groupby('DEPTH').mean(numeric_only=True).sort_index()

            m['CFM']  = m['CFM'].interpolate(method='index').ffill().bfill()
            m['LSPD'] = m['LSPD'].interpolate(method='index').ffill().bfill().fillna(0)
            if len(m['CFM'].dropna()) < 3:
                raise ValueError(f'{label}: CFM <3 valid values.')

            spinner_f = interp1d(m.index + offset, m['CFM'],
                                 bounds_error=False, fill_value='extrapolate')
            v_f = ((spinner_f(m.index) - sp_c) / (sp_m * 60.0)
                   - np.abs(m['LSPD'].values * 0.00508))
            m['Q_FINAL'] = gaussian_filter1d(
                medfilt(v_f*(np.pi*((bh_id*0.0254)/2)**2)*vpcf*86400,11), 1.0)
            return m

        p1_m = process_physics(flowing_raw, 'FLOWING')
        s1_m = process_physics(static_raw,  'STATIC')

        s1_m['Q_FINAL'] -= float(np.interp(d_top, s1_m.index, s1_m['Q_FINAL']))
        p1_top_q = float(np.interp(d_top, p1_m.index, p1_m['Q_FINAL']))
        if abs(p1_top_q) < 1e-9:
            logp('   ⚠️  P1 Q at anchor top ≈ 0 — scaling skipped.')
        else:
            p1_m['Q_FINAL'] *= (SIM_TARGET_RES / p1_top_q)

        depth_lo = max(p1_m.index.min(), s1_m.index.min())
        depth_hi = min(p1_m.index.max(), s1_m.index.max())

        perf_list_raw = [
            (float(p.split(',')[0]), float(p.split(',')[1]), p.split(',')[2].strip())
            for p in perfs_txt.strip().split('\n') if p.strip()
        ]
        perf_list, skipped = [], []
        for t, b, name in perf_list_raw:
            if t > depth_hi or b < depth_lo: skipped.append(name)
            else: perf_list.append((t, b, name))
        if skipped:
            logp(f'   ⚠️  Perfs outside usable depth range [{depth_lo:.0f}–{depth_hi:.0f} m] skipped: {skipped}')
        if not perf_list:
            logp('   ⚠️  No perforations in usable range.')

        report, zone_pct_P1 = [], {}

        comp_q_sc = SIM_TARGET_RES * bg_val
        comp_oil  = comp_q_sc * GOR
        comp_wtr  = comp_q_sc * WGR

        report.append(dict(
            Sub_Zone='Composite\n(Total)', Top=d_top, Base=d_bot,
            Press=float(np.interp(d_top,p1_m.index,p1_m['PRS'])),
            Temp =float(np.interp(d_top,p1_m.index,p1_m['TMP'])),
            Q_Res=SIM_TARGET_RES, Q_SC=comp_q_sc, Q_pct=100.0,
            Q_Oil=comp_oil, Q_Wtr=comp_wtr,
            Survey='P1', Choke=prim['choke'], THP=prim['thp'],
            row_type='comp_p1'))
        report.append(dict(
            Sub_Zone='', Top=np.nan, Base=np.nan,
            Press=float(np.interp(d_top,s1_m.index,s1_m['PRS'])),
            Temp =float(np.interp(d_top,s1_m.index,s1_m['TMP'])),
            Q_Res=float(np.interp(d_top,s1_m.index,s1_m['Q_FINAL'])),
            Q_SC=0.0, Q_pct=0.0, Q_Oil=0.0, Q_Wtr='-',
            Survey='S1', Choke=choke_s1, THP=thp_s1,
            row_type='comp_s1'))

        for t, b, name in perf_list:
            g1  = float(np.interp(t-0.5,p1_m.index,p1_m['Q_FINAL'])
                        - np.interp(b+0.5,p1_m.index,p1_m['Q_FINAL']))
            g2  = float(np.interp(t-0.5,s1_m.index,s1_m['Q_FINAL'])
                        - np.interp(b+0.5,s1_m.index,s1_m['Q_FINAL']))
            pct1 = (g1/SIM_TARGET_RES*100) if SIM_TARGET_RES!=0 else 0
            pct2 = (g2/S1_STABLE_REF *100) if S1_STABLE_REF !=0 else 0
            zone_pct_P1[name] = pct1
            g1_sc = g1 * bg_val
            report.append(dict(
                Sub_Zone=name, Top=t, Base=b,
                Press=float(np.interp(t,p1_m.index,p1_m['PRS'])),
                Temp =float(np.interp(t,p1_m.index,p1_m['TMP'])),
                Q_Res=g1, Q_SC=g1_sc, Q_pct=pct1,
                Q_Oil=g1_sc*GOR, Q_Wtr=g1_sc*WGR,
                Survey='P1', Choke=prim['choke'], THP=prim['thp'],
                row_type='zone_p1'))
            report.append(dict(
                Sub_Zone='', Top=np.nan, Base=np.nan,
                Press=float(np.interp(t,s1_m.index,s1_m['PRS'])),
                Temp =float(np.interp(t,s1_m.index,s1_m['TMP'])),
                Q_Res=g2, Q_SC=g2*bg_val, Q_pct=pct2,
                Q_Oil='-', Q_Wtr='-',
                Survey='S1', Choke=choke_s1, THP=thp_s1,
                row_type='zone_s1'))

        for ft in flowing_tests[1:]:
            cn2   = parse_choke(ft['choke'])
            q2r   = (ft['qgas']/bg_val)*(cn2/36.0)**2*(ft['thp']/77.1)
            s2    = q2r/p1_top_q if abs(p1_top_q)>1e-9 else 1.0
            q2_sc = ft['qgas']
            report.append(dict(
                Sub_Zone='Composite\n(Total)', Top=d_top, Base=d_bot,
                Press=float(np.interp(d_top,p1_m.index,p1_m['PRS'])),
                Temp =float(np.interp(d_top,p1_m.index,p1_m['TMP'])),
                Q_Res=q2r, Q_SC=q2_sc, Q_pct=100.0,
                Q_Oil=q2_sc*GOR, Q_Wtr=q2_sc*WGR,
                Survey=ft['survey'], Choke=ft['choke'], THP=ft['thp'],
                row_type='comp_p1'))
            for t, b, name in perf_list:
                g1  = float(np.interp(t-0.5,p1_m.index,p1_m['Q_FINAL'])
                            - np.interp(b+0.5,p1_m.index,p1_m['Q_FINAL']))
                gs  = g1*s2; gs_sc=gs*bg_val
                pct = (gs/q2r*100) if q2r!=0 else 0
                report.append(dict(
                    Sub_Zone=name, Top=t, Base=b,
                    Press=float(np.interp(t,p1_m.index,p1_m['PRS'])),
                    Temp =float(np.interp(t,p1_m.index,p1_m['TMP'])),
                    Q_Res=gs, Q_SC=gs_sc, Q_pct=pct,
                    Q_Oil=gs_sc*GOR, Q_Wtr=gs_sc*WGR,
                    Survey=ft['survey'], Choke=ft['choke'], THP=ft['thp'],
                    row_type='zone_p1'))

        df_f = pd.DataFrame(report)

        def pct_bar(val):
            try:
                v  = float(val)
                hw = min(abs(v), 100)/2
                bc = '#00B050' if v >= 0 else '#C00000'
                tc = '#C00000' if v < 0 else '#000'
                if v >= 0:
                    bar = (f'<div style="display:flex;width:100%;height:10px;">'
                           f'<div style="width:50%;background:#DDD;"></div>'
                           f'<div style="width:{hw:.1f}%;background:{bc};max-width:50%;"></div>'
                           f'<div style="flex:1;background:#DDD;"></div></div>')
                else:
                    bar = (f'<div style="display:flex;width:100%;height:10px;">'
                           f'<div style="flex:1;background:#DDD;"></div>'
                           f'<div style="width:{hw:.1f}%;background:{bc};max-width:50%;"></div>'
                           f'<div style="width:50%;background:#DDD;"></div></div>')
                return bar + f'<div style="font-size:9px;text-align:center;color:{tc};">{v:.2f}</div>'
            except: return '-'

        HEADERS = [
            ('Sub Zone','#FFD700','#000'),('Top<br>(mBDF)','#BDD7EE','#000'),
            ('Base<br>(mBDF)','#BDD7EE','#000'),('Press.<br>(kPa)','#D9E1F2','#000'),
            ('Temp.<br>(DegC)','#D9E1F2','#000'),
            ('Q<sub>Gas Res.</sub><br>m³/d','#F4B8B8','#000'),
            ('Q<sub>Gas SC</sub><br>m³/d','#F4B8B8','#000'),
            ('Q<sub>Gas Res.</sub><br>%','#F4B8B8','#000'),
            ('Q<sub>Gas %</sub>','#D9D9D9','#000'),
            ('Q<sub>Oil SC</sub><br>m³/d','#C6EFCE','#000'),
            ('Q<sub>Wtr SC</sub><br>m³/d','#C6EFCE','#000'),
            ('Survey','#BDD7EE','#000'),('Choke<br>Size','#BDD7EE','#000'),
            ('THP<br>(Bar)','#BDD7EE','#000'),
        ]
        hdr = ''.join(f'<th style="background:{bg};color:{fg};padding:5px 7px;'
                      f'border:1.5px solid #999;font-size:10px;font-weight:bold;'
                      f'text-align:center;white-space:nowrap;">{lbl}</th>'
                      for lbl,bg,fg in HEADERS)

        def td(txt, bg='#FFF', fg='#000', bold=False):
            return (f'<td style="background:{bg};color:{fg};padding:3px 6px;'
                    f'border:1px solid #BBB;font-size:10px;'
                    f'font-weight:{"bold" if bold else "normal"};'
                    f'text-align:center;white-space:nowrap;">{txt}</td>')

        def ntd(val, dec=2, bg='#FFF'):
            try:
                v = float(val)
                return td(f'{v:.{dec}f}', bg=bg, fg='#C00000' if v < 0 else '#000')
            except:
                return td(str(val) if val not in ('', None) else '-', bg=bg)

        rows = ''
        for _, row in df_f.iterrows():
            rt      = row['row_type']
            is_p1   = rt in ('comp_p1','zone_p1')
            is_comp = rt in ('comp_p1','comp_s1')
            rbg = ('#EBF7EC' if is_p1 else '#EBF3FB') if is_comp else ('#FFF' if is_p1 else '#F5F5F5')
            sz  = str(row['Sub_Zone'])
            if sz == 'Composite\n(Total)':
                sz_td = td('Composite<br>(Total)', bg='#E2EFDA', fg='#375623', bold=True)
            elif sz:
                sz_td = td(sz, bg='#FFD700', fg='#000', bold=True)
            else:
                sz_td = td('', bg=rbg)
            sv_fg = '#1F497D' if row['Survey'] in ('P1','P2','P3') else '#C55A11'
            rows += '<tr>' + sz_td
            rows += ntd(row['Top'],0,rbg) + ntd(row['Base'],0,rbg)
            rows += ntd(row['Press'],0,rbg) + ntd(row['Temp'],1,rbg)
            rows += ntd(row['Q_Res'],2,rbg) + ntd(row['Q_SC'],2,rbg) + ntd(row['Q_pct'],2,rbg)
            rows += (f'<td style="background:{rbg};padding:2px 4px;border:1px solid #BBB;'
                     f'min-width:95px;">{pct_bar(row["Q_pct"])}</td>')
            rows += ntd(row['Q_Oil'],2,rbg) + ntd(row['Q_Wtr'],2,rbg)
            rows += td(f'<b>{row["Survey"]}</b>', bg=rbg, fg=sv_fg)
            rows += td(str(row['Choke']), bg=rbg) + ntd(row['THP'],2,rbg) + '</tr>\n'

        tests_lbl = ', '.join([ft['survey'] for ft in flowing_tests])
        st.markdown(f'''
        <div style="font-family:Calibri,Arial,sans-serif;margin:14px 0;">
          <div style="background:#1F4E79;color:#FFF;padding:7px 14px;
                      font-weight:bold;font-size:13px;border-bottom:3px solid #FFD700;">
            ⛽ PLT ZONAL CONTRIBUTION &nbsp;|&nbsp; {well_name}
            &nbsp;<span style="font-size:10px;color:#AAD4FF;">
              Depth: {depth_lo:.0f}–{depth_hi:.0f} m &nbsp;|&nbsp; Tests: {tests_lbl}
              &nbsp;|&nbsp; GOR={GOR*1000:.3f}  WGR={WGR*1000:.3f} m³/1000m³
            </span>
          </div>
          <div style="overflow-x:auto;">
            <table style="border-collapse:collapse;font-family:Calibri,Arial;">
              <thead><tr>{hdr}</tr></thead><tbody>{rows}</tbody>
            </table>
          </div>
        </div>''', unsafe_allow_html=True)

        FG   = '#dce8f8'   
        DARK = '#08101e'   
        PAN  = '#0d1828'   
        GRD  = '#132035'   

        plt.rcParams.update({
            'font.family'      : 'DejaVu Sans',
            'font.size'        : 11,
            'axes.facecolor'   : PAN,
            'figure.facecolor' : DARK,
            'axes.edgecolor'   : '#1e3555',
            'axes.labelcolor'  : FG,
            'xtick.color'      : FG,
            'ytick.color'      : FG,
            'text.color'       : FG,
            'grid.color'       : GRD,
            'grid.linewidth'   : 0.7,
            'grid.linestyle'   : '-',
            'axes.grid'        : True,
            'legend.framealpha': 0.88,
            'legend.facecolor' : '#0d1828',
            'legend.edgecolor' : '#1e3555',
            'figure.dpi'       : 140,
        })

        def title_ax(ax, txt):
            ax.set_title(txt, color='#38c4f0', fontsize=12, fontweight='bold', pad=10)

        def add_perfs_v(ax):
            """Add perf intervals to a vertical (depth) track."""
            xl = ax.get_xlim()
            xspan = xl[1] - xl[0]
            for t_p, b_p, pname in perf_list:
                ax.axhspan(t_p, b_p, color='#f0c040', alpha=0.10, zorder=0)
                ax.axhline(t_p, color='#f0c040', lw=0.7, ls='--', alpha=0.55)
                ax.axhline(b_p, color='#f0c040', lw=0.7, ls='--', alpha=0.55)
                ax.text(xl[0] + xspan*0.02, (t_p+b_p)/2, pname,
                        color='#f0c040', fontsize=8, va='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.15', fc=DARK, alpha=0.6, ec='none'))

        v_grid   = np.linspace(depth_lo, depth_hi, 700)
        p1_prs   = np.interp(v_grid, p1_m.index, p1_m['PRS'])
        s1_prs   = np.interp(v_grid, s1_m.index, s1_m['PRS'])
        p1_tmp   = np.interp(v_grid, p1_m.index, p1_m['TMP'])
        s1_tmp   = np.interp(v_grid, s1_m.index, s1_m['TMP'])
        p1_q     = np.interp(v_grid, p1_m.index, p1_m['Q_FINAL'])
        s1_q     = np.interp(v_grid, s1_m.index, s1_m['Q_FINAL'])
        drawdown = s1_prs - p1_prs

        st.markdown('<div class="plot-section-title">📊 PLT Multi-Track Log</div>',
                    unsafe_allow_html=True)

        fig, axes = plt.subplots(
            1, 5, figsize=(26, 16),
            sharey=True,
            gridspec_kw={'wspace': 0.06}
        )
        fig.patch.set_facecolor(DARK)

        ax = axes[0]
        ax.fill_betweenx(v_grid, p1_prs, s1_prs, color='#38c4f0', alpha=0.10)
        ax.plot(s1_prs, v_grid, color='#38c4f0', lw=2.0, label='S1 Static')
        ax.plot(p1_prs, v_grid, color='#ff8c00', lw=2.0, label='P1 Flowing')
        ax.invert_yaxis()
        ax.set_ylabel('Depth (m)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Pressure (kPa)', fontsize=10)
        title_ax(ax, 'PRESSURE')
        ax.legend(fontsize=9, loc='lower right')
        add_perfs_v(ax)

        ax = axes[1]
        ax.fill_betweenx(v_grid, p1_tmp, s1_tmp, color='#ff8c00', alpha=0.10)
        ax.plot(s1_tmp, v_grid, color='#f0c040', lw=2.0, label='S1')
        ax.plot(p1_tmp, v_grid, color='#ff4422', lw=2.0, label='P1')
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        title_ax(ax, 'TEMPERATURE')
        ax.legend(fontsize=9, loc='lower right')
        add_perfs_v(ax)

        ax = axes[2]
        ax.fill_betweenx(v_grid, 0, drawdown,
                         where=drawdown >= 0, color='#00e676', alpha=0.45, label='Positive')
        ax.fill_betweenx(v_grid, 0, drawdown,
                         where=drawdown < 0,  color='#ff1744', alpha=0.45, label='Negative')
        ax.plot(drawdown, v_grid, color='#ffffff', lw=1.0, alpha=0.7)
        ax.axvline(0, color='#3a5a7a', lw=1.0)
        ax.set_xlabel('ΔP (kPa)', fontsize=10)
        title_ax(ax, 'DRAWDOWN ΔP')
        ax.legend(fontsize=9, loc='lower right')
        add_perfs_v(ax)

        ax = axes[3]
        ax.fill_betweenx(v_grid, 0, p1_q, color='#00e676', alpha=0.30, label='P1')
        ax.fill_betweenx(v_grid, 0, s1_q, color='#ff8c00', alpha=0.22, label='S1')
        ax.plot(p1_q, v_grid, color='#00e676', lw=2.2)
        ax.plot(s1_q, v_grid, color='#ff8c00', lw=1.8, ls='--')
        ax.axvline(0, color='#3a5a7a', lw=1.0)
        ax.set_xlabel('Flow Rate (m³/d)', fontsize=10)
        title_ax(ax, 'FLOW PROFILE Q')
        ax.legend(fontsize=9, loc='lower right')
        add_perfs_v(ax)

        ax = axes[4]
        d_span = depth_hi - depth_lo
        bh = max(d_span / max(len(perf_list), 1) * 0.55, 1.5)
        for t_p, b_p, pname in perf_list:
            pct  = zone_pct_P1.get(pname, 0)
            col  = '#00e676' if pct >= 0 else '#ff1744'
            ax.barh((t_p+b_p)/2, pct, height=bh,
                    color=col, alpha=0.85, edgecolor='#08101e', lw=0.5)
            ax.text(pct + (1.2 if pct >= 0 else -1.2),
                    (t_p+b_p)/2, f'{pct:.1f}%',
                    color='white', fontsize=9, va='center', fontweight='bold',
                    ha='left' if pct >= 0 else 'right')
            ax.axhspan(t_p, b_p, color='#f0c040', alpha=0.07, zorder=0)
        ax.axvline(0, color='#3a5a7a', lw=1.0)
        ax.set_xlabel('Contribution (%)', fontsize=10)
        title_ax(ax, 'ZONE %')

        fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06, wspace=0.06)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown('<div class="plot-section-title">📊 Zonal Analytics</div>',
                    unsafe_allow_html=True)

        acol1, acol2 = st.columns([3, 1])

        with acol1:
            zs  = sorted(zone_pct_P1.items(), key=lambda x: x[1])
            n_s = [z[0] for z in zs]
            v_s = [z[1] for z in zs]
            fig, ax = plt.subplots(figsize=(12, max(4, len(n_s)*0.55 + 1.5)))
            fig.patch.set_facecolor(DARK)
            title_ax(ax, 'ZONAL CONTRIBUTION RANKING (P1)')
            bars = ax.barh(n_s, v_s,
                           color=['#00e676' if v >= 0 else '#ff1744' for v in v_s],
                           edgecolor='#1e3555', height=0.58)
            ax.axvline(0, color='#3a5a7a', lw=1.2)
            for bar, val in zip(bars, v_s):
                ax.text(val + (0.4 if val >= 0 else -0.4),
                        bar.get_y() + bar.get_height()/2,
                        f'{val:.2f}%', va='center',
                        ha='left' if val >= 0 else 'right',
                        color='white', fontsize=10, fontweight='bold')
            ax.tick_params(axis='y', labelsize=10)
            ax.set_xlabel('Contribution (%)', fontsize=10)
            fig.tight_layout(pad=1.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with acol2:
            pos = {k: v for k, v in zone_pct_P1.items() if v > 0}
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor(DARK)
            ax.set_facecolor(DARK)
            title_ax(ax, 'SHARE')
            if pos:
                cm2 = plt.cm.get_cmap('tab20', len(pos))
                wedges, texts, autos = ax.pie(
                    list(pos.values()), labels=list(pos.keys()),
                    autopct='%1.0f%%',
                    colors=[cm2(i) for i in range(len(pos))],
                    startangle=90, pctdistance=0.72,
                    wedgeprops={'edgecolor': DARK, 'linewidth': 1.5})
                for t2 in texts:  t2.set_color(FG); t2.set_fontsize(8)
                for t2 in autos: t2.set_color('#fff'); t2.set_fontsize(8); t2.set_fontweight('bold')
            fig.tight_layout(pad=1.2)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor(DARK)
        title_ax(ax, 'P-T CROSSPLOT')
        sc = ax.scatter(p1_tmp, p1_prs, c=v_grid, cmap='plasma',
                        s=12, alpha=0.85, zorder=3, label='P1 Flowing')
        ax.scatter(s1_tmp, s1_prs, c='#ff8c00', s=6, alpha=0.28, zorder=2, label='S1 Static')
        cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.025)
        cb.set_label('Depth (m)', color=FG, fontsize=10)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG, fontsize=9)
        ax.set_xlabel('Temperature (°C)', fontsize=10)
        ax.set_ylabel('Pressure (kPa)', fontsize=10)
        ax.legend(fontsize=10)
        fig.tight_layout(pad=1.6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        wf_col, dd_col = st.columns(2)

        with wf_col:
            wf_n, wf_d, wf_b = ['Start'], [0.0], [0.0]
            running = 0.0
            for t_p, b_p, pname in sorted(perf_list, key=lambda x: x[0]):
                g1 = float(np.interp(t_p-0.5, p1_m.index, p1_m['Q_FINAL'])
                           - np.interp(b_p+0.5, p1_m.index, p1_m['Q_FINAL']))
                q_sc = g1 * bg_val
                wf_n.append(pname); wf_d.append(q_sc)
                wf_b.append(running); running += q_sc
            wf_n.append('TOTAL'); wf_d.append(running); wf_b.append(0.0)
            wf_cols = (['#607080']
                       + ['#00e676' if d >= 0 else '#ff1744' for d in wf_d[1:-1]]
                       + ['#38c4f0'])
            rb = list(wf_b[:-1]) + [0.0]
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(DARK)
            title_ax(ax, 'CUMULATIVE FLOW WATERFALL')
            ax.bar(range(len(wf_n)), wf_d, bottom=rb,
                   color=wf_cols, edgecolor='#08101e', linewidth=0.6, width=0.68)
            ax.set_xticks(range(len(wf_n)))
            ax.set_xticklabels(wf_n, rotation=38, ha='right', fontsize=10)
            ax.axhline(0, color='#3a5a7a', lw=1.0)
            ax.set_ylabel('Q Gas SC (m³/d)', fontsize=10)
            fig.tight_layout(pad=1.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with dd_col:
            dd_v, q_v, zn_e = [], [], []
            for t_p, b_p, pname in perf_list:
                zd = v_grid[(v_grid >= t_p) & (v_grid <= b_p)]
                if len(zd) < 2: zd = np.array([t_p, b_p])
                dd_m = float((np.interp(zd, s1_m.index, s1_m['PRS'])
                              - np.interp(zd, p1_m.index, p1_m['PRS'])).mean())
                qv   = float(np.interp(t_p-0.5, p1_m.index, p1_m['Q_FINAL'])
                             - np.interp(b_p+0.5, p1_m.index, p1_m['Q_FINAL']))
                dd_v.append(dd_m); q_v.append(qv); zn_e.append(pname)
            fig, ax = plt.subplots(figsize=(10, 5))
            fig.patch.set_facecolor(DARK)
            title_ax(ax, 'DRAWDOWN vs FLOW (per zone)')
            ax.scatter(dd_v, q_v,
                       c=['#00e676' if q >= 0 else '#ff1744' for q in q_v],
                       s=140, edgecolors='white', linewidths=0.9, zorder=5)
            for x, y, n in zip(dd_v, q_v, zn_e):
                ax.annotate(n, (x, y), textcoords='offset points', xytext=(5, 5),
                            fontsize=9, color='#f0c040', fontweight='bold')
            ax.axhline(0, color='#3a5a7a', lw=1.0)
            ax.axvline(0, color='#3a5a7a', lw=1.0)
            valid = [(d, q) for d, q in zip(dd_v, q_v)
                     if not (np.isnan(d) or np.isnan(q))]
            if len(valid) >= 2:
                xs, ys = zip(*valid)
                re = linregress(xs, ys)
                xf = np.linspace(min(xs)*0.9, max(xs)*1.1, 50)
                ax.plot(xf, re.slope*xf + re.intercept, color='#ff4444', ls='--', lw=2.0,
                        label=f'PI = {re.slope:.4f} m³/d/kPa   R² = {re.rvalue**2:.3f}')
                ax.legend(fontsize=10)
            ax.set_xlabel('Mean ΔP (kPa)', fontsize=10)
            ax.set_ylabel('Q Gas Res (m³/d)', fontsize=10)
            fig.tight_layout(pad=1.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        if len(flowing_tests) > 1:
            st.markdown('<div class="plot-section-title">📊 Surface Test Comparison</div>',
                        unsafe_allow_html=True)
            tl    = [ft['survey'] for ft in flowing_tests]
            tq    = [ft['qgas']   for ft in flowing_tests]
            tt    = [ft['thp']    for ft in flowing_tests]
            tcols = ['#38c4f0','#ff8c00','#00e676','#ff00ff','#f0c040'][:len(flowing_tests)]
            sc1, sc2 = st.columns(2)
            with sc1:
                xi = np.arange(len(tl)); wb = 0.25
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor(DARK)
                title_ax(ax, 'MULTI-FLUID RATE PER SURVEY')
                ax.bar(xi-wb, tq,                      wb, color='#38c4f0', label='Gas SC',       edgecolor='#08101e')
                ax.bar(xi,   [q*GOR*100 for q in tq],  wb, color='#00e676', label='Oil SC ×100',  edgecolor='#08101e')
                ax.bar(xi+wb,[q*WGR*10  for q in tq],  wb, color='#ff8c00', label='Water SC ×10', edgecolor='#08101e')
                ax.set_xticks(xi); ax.set_xticklabels(tl, fontsize=11)
                ax.set_ylabel('Rate (m³/d)', fontsize=10)
                ax.legend(fontsize=10)
                fig.tight_layout(pad=1.6)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with sc2:
                fig, ax = plt.subplots(figsize=(8, 5))
                fig.patch.set_facecolor(DARK)
                title_ax(ax, 'Qgas SC vs THP')
                ax.scatter(tt, tq, c=tcols, s=200, edgecolors='white', lw=1.2, zorder=5)
                for l, x2, y in zip(tl, tt, tq):
                    ax.annotate(l, (x2, y), textcoords='offset points', xytext=(7, 5),
                                fontsize=12, color='#f0c040', fontweight='bold')
                ax.set_xlabel('THP (Bar)', fontsize=10)
                ax.set_ylabel('Qgas SC (m³/d)', fontsize=10)
                fig.tight_layout(pad=1.6)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        st.markdown('<div class="plot-section-title">📊 Spinner Calibration</div>',
                    unsafe_allow_html=True)
        cal_all = pd.concat(measurement_runs + transit_runs)
        ct_z  = cal_top if depth_lo <= cal_top <= depth_hi else depth_hi - span*0.05
        cb2_z = cal_bot if depth_lo <= cal_bot <= depth_hi else depth_hi
        cal_pts = cal_all[(cal_all['DEPTH'] >= ct_z) & (cal_all['DEPTH'] <= cb2_z)].copy()
        if cal_pts.empty:
            cal_pts = cal_all[cal_all['DEPTH'] >= depth_hi - span*0.05].copy()
        cal_pts['L_ms'] = cal_pts['LSPD'] * 0.00508
        cal_pts['dir']  = np.where(cal_pts['L_ms'] > 0, 'UP', 'DOWN')
        p_avgs = cal_pts.groupby(['fname','dir']).mean(numeric_only=True).reset_index()

        sp_col, _ = st.columns([2, 1])
        with sp_col:
            fig, ax = plt.subplots(figsize=(11, 6))
            fig.patch.set_facecolor(DARK)
            title_ax(ax, 'SPINNER CALIBRATION  (Pass Averages)')
            ax.scatter(cal_pts['L_ms'], cal_pts['CFM'],
                       color='#38c4f0', s=5, alpha=0.18, label='Raw data', zorder=2)
            for d, grp in p_avgs.groupby('dir'):
                ax.scatter(grp['L_ms'], grp['CFM'],
                           color={'UP':'#ff8c00','DOWN':'#cc44ff'}.get(d,'gold'),
                           s=200, edgecolors='white', linewidths=1.2,
                           zorder=6, label=f'{d} pass avg')
            if len(p_avgs) >= 2:
                r3 = linregress(p_avgs['L_ms'], p_avgs['CFM'])
                xf = np.linspace(p_avgs['L_ms'].min()-0.05,
                                 p_avgs['L_ms'].max()+0.05, 60)
                ax.plot(xf, r3.slope*xf + r3.intercept, color='#ff4444', ls='--', lw=2.2,
                        label=f'y = {r3.slope:.3f}x + {r3.intercept:.3f}   '
                              f'R² = {r3.rvalue**2:.4f}')
                xz = -r3.intercept/r3.slope if abs(r3.slope) > 1e-9 else 0
                ax.axvline(xz, color='#f0c040', lw=1.5, ls=':',
                           label=f'Threshold  {xz:.4f} m/s')
            ax.axvline(0, color='#3a5a7a', lw=1.0)
            ax.axhline(0, color='#3a5a7a', lw=1.0)
            ax.set_xlabel('Tool Velocity (m/s)', fontsize=10)
            ax.set_ylabel('Spinner (RPS)', fontsize=10)
            ax.legend(fontsize=10)
            fig.tight_layout(pad=1.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        logp(f'\n✅  Pipeline complete  |  {well_name}'
             f'  |  Depth: {depth_lo:.0f}–{depth_hi:.0f} m'
             f'  |  Measurement files: {len(measurement_runs)}'
             f'  |  Transit excluded: {len(transit_runs)}'
             f'  |  GOR={GOR*1000:.3f}  WGR={WGR*1000:.3f} m³/1000m³')

    except Exception as e:
        import traceback
        st.error(f'❌  Failure: {e}')
        st.text(traceback.format_exc())
