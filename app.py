import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import html as html_lib
import warnings

# ─────────────────────────────────────────────
#   PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PathwayNet · Pharmacogenomics",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#   GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* ── BASE ───────────────────────────────── */
html, body, .stApp {
    background: #f5f3f0 !important;
    color: #2c2c2c;
    font-family: 'Inter', sans-serif;
}
section[data-testid="stSidebar"] {
    background: #eeece8 !important;
    border-right: 1px solid #dddad5;
}
hr { border-color: #dddad5 !important; }

/* ── TYPOGRAPHY ─────────────────────────── */
h1,h2,h3,h4,h5,h6 {
    color: #3d5a80 !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.3px;
}

/* ── SIDEBAR ELEMENTS ───────────────────── */
.sidebar-brand {
    padding: 18px 0 22px 0;
    text-align: center;
    border-bottom: 1px solid #dddad5;
    margin-bottom: 22px;
}
.brand-icon { font-size: 2em; margin-bottom: 6px; }
.brand-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3em;
    font-weight: 700;
    color: #3d5a80;
    letter-spacing: 1px;
}
.brand-version {
    font-size: 0.62em;
    color: #9a9390;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    margin-top: 3px;
}
.sidebar-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65em;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 8px;
}
.drug-badge {
    background: rgba(61,90,128,0.06);
    border: 1px solid rgba(61,90,128,0.18);
    border-left: 3px solid #3d5a80;
    border-radius: 8px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78em;
    color: #3d5a80;
    word-break: break-word;
    margin: 8px 0 0 0;
    line-height: 1.5;
}

/* ── BUTTONS ────────────────────────────── */
div[data-testid="stButton"] > button {
    background: #3d5a80;
    color: #f5f3f0;
    border: 1px solid #2e4460;
    border-radius: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: 2px;
    width: 100%;
    padding: 14px 10px;
    transition: all 0.2s ease;
    text-transform: uppercase;
    font-size: 0.82em;
}
div[data-testid="stButton"] > button:hover {
    background: #2e4460;
    box-shadow: 0 2px 12px rgba(61,90,128,0.18);
    border-color: #3d5a80;
}

/* ── METRICS ────────────────────────────── */
div[data-testid="stMetricValue"] {
    color: #4a8c6f !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.85em !important;
    font-weight: 700 !important;
}
div[data-testid="stMetricLabel"] {
    color: #9a9390 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68em !important;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* ── PROGRESS ───────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #3d5a80, #7ba7bc, #4a8c6f) !important;
}

/* ── PAGE HEADER ────────────────────────── */
.page-online-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(74,140,111,0.07);
    border: 1px solid rgba(74,140,111,0.22);
    color: #4a8c6f;
    border-radius: 30px;
    padding: 5px 16px;
    font-size: 0.68em;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.pulse-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4a8c6f;
    box-shadow: 0 0 6px rgba(74,140,111,0.5);
    animation: pulseAnim 2s ease-in-out infinite;
    flex-shrink: 0;
}
@keyframes pulseAnim {
    0%,100% { opacity:1; }
    50% { opacity:0.3; }
}
.page-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2em;
    font-weight: 700;
    color: #3d5a80;
    letter-spacing: 0.5px;
    margin: 0 0 6px 0;
    line-height: 1.15;
}
.page-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68em;
    color: #9a9390;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    margin-bottom: 28px;
}

/* ── SECTION HEADERS ────────────────────── */
.section-hdr {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 52px 0 22px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid #dddad5;
}
.section-hdr-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #dddad5, transparent);
}
.section-hdr-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    white-space: nowrap;
}
.section-hdr-icon { font-size: 0.9em; }

/* ── COMPOUND CARD ──────────────────────── */
.compound-card-wrap {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-left: 4px solid #3d5a80;
    border-radius: 14px;
    padding: 28px 30px 22px 30px;
    margin-bottom: 28px;
}
.compound-card-header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 14px;
    flex-wrap: wrap;
    gap: 10px;
}
.compound-tag {
    display: inline-block;
    background: rgba(61,90,128,0.08);
    border: 1px solid rgba(61,90,128,0.2);
    color: #3d5a80;
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 0.67em;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.compound-no-data-tag {
    background: rgba(154,147,144,0.08);
    border-color: rgba(154,147,144,0.2);
    color: #9a9390;
}
.compound-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.22em;
    font-weight: 700;
    color: #2c2c2c;
    margin: 0 0 12px 0;
    line-height: 1.3;
}
.compound-description {
    font-family: 'Inter', sans-serif;
    font-size: 0.9em;
    line-height: 1.85;
    color: #555555;
    margin-bottom: 0;
}
.compound-description-missing {
    color: #9a9390;
    font-style: italic;
}
.compound-divider {
    border: none;
    border-top: 1px solid #dddad5;
    margin: 18px 0;
}
.cancer-types-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.63em;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 10px;
}
.cancer-pill {
    display: inline-block;
    background: rgba(123,167,188,0.1);
    border: 1px solid rgba(123,167,188,0.25);
    color: #4a7a96;
    border-radius: 6px;
    padding: 4px 11px;
    font-size: 0.72em;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    margin: 3px 4px 3px 0;
}
.meta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}
.meta-pill {
    background: rgba(61,90,128,0.05);
    border: 1px solid rgba(61,90,128,0.12);
    border-radius: 7px;
    padding: 6px 14px;
    font-size: 0.72em;
    font-family: 'JetBrains Mono', monospace;
    color: #555;
}
.meta-pill span { color: #3d5a80; }

/* ── KPI CARDS ──────────────────────────── */
.kpi-outer {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 12px;
    padding: 20px 16px 16px;
    text-align: center;
    height: 100%;
}
.kpi-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62em;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 10px;
}
.kpi-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2em;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.kpi-sub {
    font-size: 0.7em;
    color: #9a9390;
    font-family: 'Inter', sans-serif;
    line-height: 1.4;
}

/* ── VALIDITY BADGE ─────────────────────── */
.validity-badge {
    border-radius: 8px;
    padding: 12px 20px;
    margin: 18px 0 0 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8em;
    letter-spacing: 0.5px;
}

/* ── PERTURBATION VERDICT ───────────────── */
.verdict-wrap {
    border-radius: 14px;
    padding: 28px 30px;
    background: #faf9f7;
    margin-top: 20px;
}
.verdict-target-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 6px;
}
.verdict-target-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05em;
    font-weight: 700;
    margin-bottom: 18px;
}
.verdict-conclusion {
    font-family: 'Inter', sans-serif;
    font-size: 0.88em;
    line-height: 1.85;
    color: #444;
}
.verdict-footer {
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid #dddad5;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67em;
    color: #9a9390;
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
}
.verdict-footer span b { color: #555; }

/* ── AWAITING CARD ──────────────────────── */
.await-card {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-left: 4px solid #3d5a80;
    border-radius: 14px;
    padding: 32px 30px;
}
.await-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1em;
    color: #3d5a80;
    font-weight: 700;
    margin-bottom: 14px;
    letter-spacing: 0.5px;
}
.await-body {
    color: #555;
    font-size: 0.87em;
    line-height: 1.9;
    margin-bottom: 22px;
}
.feature-chips {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.feature-chip {
    background: rgba(61,90,128,0.06);
    border: 1px solid rgba(61,90,128,0.14);
    border-radius: 8px;
    padding: 9px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.71em;
    color: #3d5a80;
    letter-spacing: 0.3px;
}

/* Plotly chart spacing */
.js-plotly-plot { margin-bottom: 10px; }

/* selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background: #faf9f7 !important;
    border: 1px solid #dddad5 !important;
    border-radius: 8px !important;
    color: #2c2c2c !important;
}

/* ── SIDEBAR NAV ─────────────────────────── */
.nav-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 8px;
    margin-top: 4px;
}
div[data-testid="stRadio"] > div {
    gap: 6px !important;
}
div[data-testid="stRadio"] label {
    background: rgba(61,90,128,0.04) !important;
    border: 1px solid #dddad5 !important;
    border-radius: 8px !important;
    padding: 9px 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78em !important;
    color: #555 !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    width: 100% !important;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(61,90,128,0.08) !important;
    border-color: #3d5a80 !important;
    color: #3d5a80 !important;
}
div[data-testid="stRadio"] label[data-checked="true"],
div[data-testid="stRadio"] label[aria-checked="true"] {
    background: rgba(61,90,128,0.1) !important;
    border-color: #3d5a80 !important;
    color: #3d5a80 !important;
}

/* ── FOOTER ─────────────────────────────── */
.site-footer {
    margin-top: 70px;
    padding: 26px 0 14px 0;
    border-top: 1px solid #dddad5;
    text-align: center;
}
.footer-disclaimer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68em;
    letter-spacing: 1px;
    color: #9a9390;
    margin-bottom: 8px;
}
.footer-sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.72em;
    color: #c0bdb9;
}

/* ── LEARN PAGE ─────────────────────────── */
.learn-hero {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 16px;
    padding: 48px 44px 40px 44px;
    margin-bottom: 36px;
    position: relative;
    overflow: hidden;
}
.learn-hero::after {
    content: '🧬';
    position: absolute;
    right: 36px; top: 28px;
    font-size: 6em;
    opacity: 0.05;
    pointer-events: none;
}
.learn-hero-tag {
    display: inline-block;
    background: rgba(74,140,111,0.08);
    border: 1px solid rgba(74,140,111,0.22);
    color: #4a8c6f;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.68em;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.learn-hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9em;
    font-weight: 700;
    color: #2c2c2c;
    line-height: 1.25;
    margin-bottom: 18px;
}
.learn-hero-body {
    font-family: 'Inter', sans-serif;
    font-size: 1.0em;
    line-height: 1.9;
    color: #555;
    max-width: 700px;
}
.learn-toc {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 40px;
}
.learn-toc-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 16px;
}
.learn-toc-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid #e8e5e1;
    font-family: 'Inter', sans-serif;
    font-size: 0.88em;
    color: #555;
}
.learn-toc-item:last-child { border-bottom: none; }
.learn-toc-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75em;
    color: #9a9390;
    min-width: 24px;
}
.learn-section-card {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 14px;
    padding: 32px 34px;
    margin-bottom: 24px;
}
.learn-section-icon {
    font-size: 2em;
    margin-bottom: 14px;
    display: block;
}
.learn-section-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 6px;
}
.learn-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1em;
    font-weight: 700;
    color: #3d5a80;
    margin-bottom: 16px;
    line-height: 1.3;
}
.learn-body {
    font-family: 'Inter', sans-serif;
    font-size: 0.9em;
    line-height: 1.9;
    color: #444;
}
.learn-body strong { color: #2c2c2c; }
.learn-body em { color: #3d5a80; font-style: normal; font-weight: 500; }
.learn-analogy {
    background: rgba(61,90,128,0.04);
    border: 1px solid rgba(61,90,128,0.12);
    border-left: 3px solid #3d5a80;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 16px 0 0 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.85em;
    color: #555;
    line-height: 1.75;
}
.learn-analogy-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7em;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3d5a80;
    margin-bottom: 6px;
}
.learn-glossary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 10px;
}
.learn-glossary-item {
    background: #f5f3f0;
    border: 1px solid #dddad5;
    border-radius: 10px;
    padding: 16px 18px;
}
.learn-glossary-term {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78em;
    font-weight: 700;
    color: #3d5a80;
    margin-bottom: 6px;
}
.learn-glossary-def {
    font-family: 'Inter', sans-serif;
    font-size: 0.82em;
    line-height: 1.7;
    color: #555;
}
.learn-step {
    display: flex;
    gap: 18px;
    align-items: flex-start;
    padding: 16px 0;
    border-bottom: 1px solid #e8e5e1;
}
.learn-step:last-child { border-bottom: none; padding-bottom: 0; }
.learn-step-num {
    width: 34px; height: 34px;
    border-radius: 50%;
    background: rgba(61,90,128,0.08);
    border: 1px solid rgba(61,90,128,0.2);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8em;
    font-weight: 700;
    color: #3d5a80;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
}
.learn-step-content-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    font-weight: 700;
    color: #2c2c2c;
    margin-bottom: 5px;
}
.learn-step-content-body {
    font-family: 'Inter', sans-serif;
    font-size: 0.85em;
    line-height: 1.75;
    color: #555;
}
.learn-callout {
    background: rgba(74,140,111,0.05);
    border: 1px solid rgba(74,140,111,0.18);
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 16px;
    font-family: 'Inter', sans-serif;
    font-size: 0.88em;
    color: #3a7057;
    line-height: 1.75;
}
.learn-callout strong { color: #2d5c45; }
.learn-warn {
    background: rgba(180,120,40,0.05);
    border: 1px solid rgba(180,120,40,0.18);
    border-radius: 10px;
    padding: 16px 20px;
    font-family: 'Inter', sans-serif;
    font-size: 0.86em;
    color: #8a5e1a;
    line-height: 1.75;
    margin-top: 16px;
}

/* ── CREDITS PAGE ───────────────────────── */
.credits-hero {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 16px;
    padding: 52px 48px 44px 48px;
    margin-bottom: 36px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.credits-hero::before {
    content: '';
    position: absolute;
    top: -60px; left: 50%;
    transform: translateX(-50%);
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(61,90,128,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.credits-institute-badge {
    display: inline-block;
    background: rgba(74,140,111,0.07);
    border: 1px solid rgba(74,140,111,0.2);
    color: #4a8c6f;
    border-radius: 20px;
    padding: 5px 18px;
    font-size: 0.68em;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-bottom: 22px;
}
.credits-project-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.6em;
    font-weight: 700;
    color: #2c2c2c;
    line-height: 1.3;
    margin-bottom: 10px;
}
.credits-dept {
    font-family: 'Inter', sans-serif;
    font-size: 0.92em;
    color: #555;
    margin-bottom: 6px;
}
.credits-institute {
    font-family: 'Inter', sans-serif;
    font-size: 0.9em;
    color: #9a9390;
    margin-bottom: 0;
}
.credits-team-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-bottom: 24px;
}
.credits-member-card {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 14px;
    padding: 28px 22px;
    text-align: center;
}
.credits-member-avatar {
    width: 58px; height: 58px;
    border-radius: 50%;
    background: rgba(61,90,128,0.1);
    border: 2px solid rgba(61,90,128,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4em;
    margin: 0 auto 16px auto;
}
.credits-member-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9em;
    font-weight: 700;
    color: #2c2c2c;
    margin-bottom: 6px;
}
.credits-member-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68em;
    color: #9a9390;
    letter-spacing: 1.5px;
    margin-bottom: 10px;
}
.credits-member-role {
    display: inline-block;
    background: rgba(61,90,128,0.07);
    border: 1px solid rgba(61,90,128,0.15);
    color: #3d5a80;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.68em;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 1px;
}
.credits-guide-card {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-left: 4px solid #3d5a80;
    border-radius: 14px;
    padding: 30px 32px;
    margin-bottom: 18px;
}
.credits-guide-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 8px;
}
.credits-guide-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05em;
    font-weight: 700;
    color: #3d5a80;
    margin-bottom: 4px;
}
.credits-guide-quals {
    font-family: 'Inter', sans-serif;
    font-size: 0.82em;
    color: #9a9390;
}
.credits-hod-card {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-left: 4px solid #4a8c6f;
    border-radius: 14px;
    padding: 30px 32px;
    margin-bottom: 18px;
}
.credits-hod-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05em;
    font-weight: 700;
    color: #4a8c6f;
    margin-bottom: 4px;
}
.credits-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    margin: 32px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #dddad5;
}
.credits-data-row {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 8px;
}
.credits-data-pill {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 8px;
    padding: 10px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74em;
    color: #555;
}
.credits-data-pill span { color: #3d5a80; display:block; font-size:0.88em; margin-top:3px; }

/* ── RESEARCHER PAGE ────────────────────── */
.res-hero {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-top: 3px solid #3d5a80;
    border-radius: 16px;
    padding: 48px 44px 42px 44px;
    margin-bottom: 36px;
}
.res-tag {
    display: inline-block;
    background: rgba(61,90,128,0.07);
    border: 1px solid rgba(61,90,128,0.18);
    color: #3d5a80;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.67em;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 18px;
}
.res-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.75em;
    font-weight: 700;
    color: #2c2c2c;
    line-height: 1.25;
    margin-bottom: 16px;
}
.res-abstract {
    font-family: 'Inter', sans-serif;
    font-size: 0.92em;
    line-height: 1.95;
    color: #555;
    max-width: 800px;
    border-left: 3px solid rgba(61,90,128,0.3);
    padding-left: 20px;
    margin-top: 16px;
}
.res-card {
    background: #faf9f7;
    border: 1px solid #dddad5;
    border-radius: 14px;
    padding: 32px 34px;
    margin-bottom: 22px;
}
.res-card-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6em;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #9a9390;
    margin-bottom: 5px;
}
.res-card-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05em;
    font-weight: 700;
    color: #3d5a80;
    margin-bottom: 18px;
    line-height: 1.3;
}
.res-body {
    font-family: 'Inter', sans-serif;
    font-size: 0.895em;
    line-height: 2.0;
    color: #444;
}
.res-body strong { color: #2c2c2c; }
.res-body em { color: #3d5a80; font-style: normal; font-weight: 600; }
.res-body ul { padding-left: 20px; margin: 10px 0; }
.res-body ul li { margin-bottom: 6px; }
.res-highlight {
    background: rgba(61,90,128,0.04);
    border: 1px solid rgba(61,90,128,0.12);
    border-left: 3px solid #3d5a80;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 18px 0 0 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.86em;
    color: #555;
    line-height: 1.8;
}
.res-highlight strong { color: #2c2c2c; }
.res-formula {
    background: #f0ede9;
    border: 1px solid #dddad5;
    border-radius: 8px;
    padding: 14px 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82em;
    color: #3a7057;
    margin: 14px 0;
    line-height: 1.9;
}
.res-metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-top: 18px;
}
.res-metric-box {
    background: #f5f3f0;
    border: 1px solid #dddad5;
    border-radius: 10px;
    padding: 16px 14px;
    text-align: center;
}
.res-metric-name {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68em;
    color: #9a9390;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.res-metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1em;
    font-weight: 700;
    color: #3d5a80;
}
.res-metric-desc {
    font-size: 0.72em;
    font-family: 'Inter', sans-serif;
    color: #9a9390;
    margin-top: 4px;
    line-height: 1.5;
}
.res-ref-item {
    padding: 12px 0;
    border-bottom: 1px solid #e8e5e1;
    font-family: 'Inter', sans-serif;
    font-size: 0.83em;
    color: #555;
    line-height: 1.7;
}
.res-ref-item:last-child { border-bottom: none; }
.res-ref-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75em;
    color: #3d5a80;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#   CONSTANTS
# ─────────────────────────────────────────────
FILE_PATHS = {
    'mutation':     '/Users/roycer/Documents/data/mutations_summary_20250318.csv',
    'drug':         '/Users/roycer/Documents/data/GDSC1_fitted_dose_response_27Oct23.xlsx',
    'gmt_hallmark': '/Users/roycer/Documents/data/h.all.v7.x.symbols.gmt.txt',
    'gmt_kegg':     '/Users/roycer/Documents/data/c2.cp.kegg.v7.x.symbols.gmt.txt',
    'descriptions': '/Users/roycer/Documents/data/description_separated.xlsx',
}

TCGA_NAMES = {
    'SKCM': 'Skin Cutaneous Melanoma',
    'GBM':  'Glioblastoma Multiforme',
    'LUAD': 'Lung Adenocarcinoma',
    'SCLC': 'Small Cell Lung Cancer',
    'NB':   'Neuroblastoma',
    'ESCA': 'Esophageal Carcinoma',
    'BRCA': 'Breast Invasive Carcinoma',
    'HNSC': 'Head & Neck Squamous Cell Carcinoma',
    'LAML': 'Acute Myeloid Leukemia',
    'BLCA': 'Bladder Urothelial Carcinoma',
    'KIRC': 'Kidney Clear Cell Carcinoma',
    'MM':   'Multiple Myeloma',
    'PAAD': 'Pancreatic Adenocarcinoma',
    'LCML': 'Chronic Myelogenous Leukemia',
    'LUSC': 'Lung Squamous Cell Carcinoma',
    'ALL':  'Acute Lymphoblastic Leukemia',
    'COREAD': 'Colorectal Adenocarcinoma',
    'LGG':  'Lower Grade Glioma',
    'OV':   'Ovarian Serous Cystadenocarcinoma',
    'DLBC': 'Diffuse Large B-cell Lymphoma',
    'MB':   'Medulloblastoma',
    'CLL':  'Chronic Lymphocytic Leukemia',
    'STAD': 'Stomach Adenocarcinoma',
    'MESO': 'Mesothelioma',
    'PRAD': 'Prostate Adenocarcinoma',
    'THCA': 'Thyroid Carcinoma',
    'UCEC': 'Uterine Corpus Endometrial Carcinoma',
    'CESC': 'Cervical Squamous Cell Carcinoma',
    'LIHC': 'Liver Hepatocellular Carcinoma',
    'UNCLASSIFIED': 'Unclassified',
}

# These 3 don't appear in descriptions file at all → manual entries only
EXTRA_DESCRIPTIONS = {
    "eEF2K Inhibitor, A-484954": (
        "A-484954 is a potent and highly selective small-molecule inhibitor of eukaryotic elongation factor 2 kinase "
        "(eEF2K). eEF2K is an atypical calcium/calmodulin-dependent kinase that regulates the elongation phase of "
        "protein synthesis. By inhibiting eEF2K, A-484954 prevents the phosphorylation of eEF2, maintaining it in "
        "an active state. Used in cancer research to study cellular stress responses, autophagy, and translational "
        "control, as eEF2K is implicated in cancer cell survival under nutrient deprivation."
    ),
    "kb NB 142-70": (
        "kb NB 142-70 is a potent, cell-permeable, and highly selective inhibitor of Protein Kinase D (PKD). The PKD "
        "family of serine/threonine kinases plays roles in Golgi complex function, cell proliferation, apoptosis, and "
        "immune regulation. In oncology research, kb NB 142-70 is used to study PKD signaling and its role in tumor "
        "progression, angiogenesis, and metastasis."
    ),
    "rTRAIL": (
        "Recombinant Tumor Necrosis Factor (TNF)-Related Apoptosis-Inducing Ligand. A biological therapeutic designed "
        "to selectively induce apoptosis in malignant cells while sparing normal cells. rTRAIL binds to death receptors "
        "(DR4/TRAIL-R1 and DR5/TRAIL-R2), triggering the death-inducing signaling complex (DISC) and activating "
        "the extrinsic caspase cascade."
    ),
}

# ─────────────────────────────────────────────
#   NAME NORMALISATION MAP
#   GDSC name  →  Description file key
# ─────────────────────────────────────────────
GDSC_TO_DESC = {
    "(5Z)-7-Oxozeaenol":                     "(5Z)-7-Oxozeanol",
    "BAY-MPS-combo 2 (paclitaxel 1 uM)":     "BAY-MPS-combo 2",
    "BAY-MPS-combo-1 (paclitaxel 5 uM)":     "BAY-MPS-combo-1",
    "Brivanib, BMS-540215":                   "Brivanib (BMS-540215)",
    "CAP-232, TT-232, TLN-232":              "CAP-232 (TT-232)",
    "MetAP2 Inhibitor, A832234":             "MetAP2 Inhibitor A832234",
    "Venotoclax":                             "Venetoclax",
    "eEF2K Inhibitor, A-484954":             "eEF2K Inhibitor, A-484954",  # manual
    "kb NB 142-70":                           "kb NB 142-70",              # manual
    "rTRAIL":                                 "rTRAIL",                    # manual
    "torin2":                                 "Torin 2",
}

# ─────────────────────────────────────────────
#   DATA LOADING
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    try:
        mut_df = pd.read_csv(FILE_PATHS['mutation'])
        mut_matrix = mut_df.pivot_table(
            index='model_id', columns='gene_symbol', values='vaf', fill_value=0
        )
        mut_matrix = (mut_matrix > 0).astype(float)

        try:
            dr_df = pd.read_excel(FILE_PATHS['drug'])
        except Exception:
            dr_df = pd.read_csv(FILE_PATHS['drug'])

        pathways = {}
        for p_path in [FILE_PATHS['gmt_hallmark'], FILE_PATHS['gmt_kegg']]:
            with open(p_path, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) > 2:
                        pathways[parts[0]] = parts[2:]

        # Load descriptions from Excel ONLY
        try:
            desc_df = pd.read_excel(FILE_PATHS['descriptions'])
            desc_dict = dict(zip(
                desc_df['DRUG NAME'].str.strip(),
                desc_df['DRUG DESCRIPTION'].str.strip()
            ))
        except Exception:
            desc_dict = {}

        # Inject manual-only entries (not in Excel)
        desc_dict.update(EXTRA_DESCRIPTIONS)

        return mut_matrix, dr_df, pathways, desc_dict

    except Exception as e:
        return None, None, str(e), {}


# ─────────────────────────────────────────────
#   DESCRIPTION LOOKUP  (safe, no HTML bleed)
# ─────────────────────────────────────────────
def lookup_description(gdsc_drug_name: str, desc_dict: dict) -> str | None:
    """Returns plain text description or None. Never returns HTML."""
    # 1. Normalisation map
    lookup_key = GDSC_TO_DESC.get(gdsc_drug_name, gdsc_drug_name)
    if lookup_key in desc_dict:
        return desc_dict[lookup_key]
    # 2. Exact
    if gdsc_drug_name in desc_dict:
        return desc_dict[gdsc_drug_name]
    # 3. Case-insensitive
    dl = gdsc_drug_name.lower()
    for k, v in desc_dict.items():
        if k.lower() == dl:
            return v
    return None


# ─────────────────────────────────────────────
#   MODEL
# ─────────────────────────────────────────────
class ResearchPathwayNet(nn.Module):
    def __init__(self, mask_matrix):
        super().__init__()
        self.mask   = nn.Parameter(torch.tensor(mask_matrix, dtype=torch.float32), requires_grad=False)
        self.w_bio  = nn.Parameter(torch.randn(mask_matrix.shape) * 0.02)
        self.b_bio  = nn.Parameter(torch.zeros(mask_matrix.shape[1]))
        self.hidden = nn.Linear(mask_matrix.shape[1], 32)
        self.output = nn.Linear(32, 1)
        self.act    = nn.ELU()

    def forward(self, x):
        pa = self.act(torch.mm(x, self.w_bio * self.mask) + self.b_bio)
        h  = self.act(self.hidden(pa))
        return self.output(h), pa


def train_model(mask, X_tr, y_tr, progress_callback=None):
    model = ResearchPathwayNet(mask)
    opt   = torch.optim.Adam(model.parameters(), lr=0.005)
    crit  = nn.MSELoss()
    model.train()
    for i in range(200):
        opt.zero_grad()
        p, _ = model(X_tr)
        loss = crit(p, y_tr); loss.backward(); opt.step()
        if progress_callback and i % 20 == 0:
            progress_callback(i / 200)
    return model


# ─────────────────────────────────────────────
#   COMPOUND CARD  (renders only safe HTML)
# ─────────────────────────────────────────────
def render_compound_card(drug_name: str, desc_dict: dict, dr_df):
    description = lookup_description(drug_name, desc_dict)
    has_desc    = description is not None

    safe_name = html_lib.escape(drug_name)
    safe_desc = html_lib.escape(description) if has_desc else None

    # ── GDSC metadata ────────────────────────────────────────────────────
    meta_parts   = []
    cancer_parts = []
    has_gdsc     = (dr_df is not None) and (drug_name in dr_df['DRUG_NAME'].values)

    if has_gdsc:
        sub     = dr_df[dr_df['DRUG_NAME'] == drug_name]
        target  = html_lib.escape(str(sub['PUTATIVE_TARGET'].iloc[0])) if 'PUTATIVE_TARGET' in sub.columns else ''
        pathway = html_lib.escape(str(sub['PATHWAY_NAME'].iloc[0]))    if 'PATHWAY_NAME'    in sub.columns else ''
        n_lines = int(sub['CELL_LINE_NAME'].nunique())
        auc_val = float(sub['AUC'].mean())

        MP = (
            'background:rgba(27,55,110,0.2);border:1px solid rgba(59,109,232,0.14);'
            'border-radius:7px;padding:6px 14px;font-size:0.72em;'
            'font-family:"JetBrains Mono",monospace;color:#4a7ac4;'
            'display:inline-block;'
        )
        VS = 'color:#7fb3fa;'

        if target  and target  != 'nan':
            meta_parts.append(f'<span style="{MP}">🎯 Target &nbsp;<span style="{VS}">{target}</span></span>')
        if pathway and pathway != 'nan':
            meta_parts.append(f'<span style="{MP}">↗ Pathway &nbsp;<span style="{VS}">{pathway}</span></span>')
        meta_parts.append(f'<span style="{MP}">🔬 Cell Lines &nbsp;<span style="{VS}">{n_lines}</span></span>')
        meta_parts.append(f'<span style="{MP}">📈 Mean AUC &nbsp;<span style="{VS}">{auc_val:.3f}</span></span>')

        cancer_counts = sub[sub['TCGA_DESC'].notna()]['TCGA_DESC'].value_counts()
        top_cancers   = [c for c in cancer_counts.index if c != 'UNCLASSIFIED'][:7]
        CP = (
            'display:inline-block;background:rgba(167,139,250,0.07);'
            'border:1px solid rgba(167,139,250,0.18);color:#a78bfa;border-radius:6px;'
            'padding:4px 11px;font-size:0.72em;font-family:"Inter",sans-serif;'
            'font-weight:500;margin:3px 4px 3px 0;'
        )
        for c in top_cancers:
            cancer_parts.append(
                f'<span style="{CP}">{html_lib.escape(TCGA_NAMES.get(c, c))}</span>'
            )

    # ── Inline styles (avoids Streamlit CSS class lookup issues) ─────────
    CARD_BORDER = "#3d5a80" if has_desc else "#b0aaa5"

    CARD_S = (
        f'background:#faf9f7;'
        f'border:1px solid #dddad5;border-left:4px solid {CARD_BORDER};'
        f'border-radius:14px;padding:28px 30px 24px 30px;margin-bottom:28px;'
    )
    TAG_S = (
        'display:inline-block;'
        + ('background:rgba(61,90,128,0.08);border:1px solid rgba(61,90,128,0.2);color:#3d5a80;'
           if has_desc else
           'background:rgba(154,147,144,0.08);border:1px solid rgba(154,147,144,0.2);color:#9a9390;')
        + 'border-radius:6px;padding:4px 12px;font-size:0.67em;'
          'font-family:"JetBrains Mono",monospace;letter-spacing:2px;text-transform:uppercase;'
    )
    NAME_S = (
        'font-family:"JetBrains Mono",monospace;font-size:1.22em;font-weight:700;'
        'color:#2c2c2c;margin:12px 0 14px 0;line-height:1.3;'
    )
    DESC_S = (
        'font-family:"Inter",sans-serif;font-size:0.9em;line-height:1.85;margin:0;'
        + ('color:#555;' if has_desc else 'color:#9a9390;font-style:italic;')
    )
    DIV_S  = 'height:1px;background:#dddad5;margin:20px 0;'
    CL_S   = (
        'font-family:"JetBrains Mono",monospace;font-size:0.63em;letter-spacing:2.5px;'
        'text-transform:uppercase;color:#9a9390;margin-bottom:10px;'
    )
    MR_S   = 'display:flex;flex-wrap:wrap;gap:10px;margin-top:18px;'

    desc_text = safe_desc if has_desc else (
        "No description available in the database for this compound. "
        "It may be an experimental identifier pending annotation."
    )

    # ── Build HTML as a list of strings, then join ────────────────────────
    parts = [
        f'<div style="{CARD_S}">',
        f'  <span style="{TAG_S}">🔬 &nbsp;Compound Profile</span>',
        f'  <div style="{NAME_S}">{safe_name}</div>',
        f'  <div style="{DESC_S}">{desc_text}</div>',
    ]

    if cancer_parts:
        parts.append(f'  <div style="{DIV_S}"></div>')
        parts.append(f'  <div style="{CL_S}">🏥 &nbsp;Studied Cancer Types</div>')
        parts.append(f'  <div>{"".join(cancer_parts)}</div>')

    if meta_parts:
        parts.append(f'  <div style="{MR_S}">{"".join(meta_parts)}</div>')

    parts.append('</div>')

    st.markdown("\n".join(parts), unsafe_allow_html=True)


# ─────────────────────────────────────────────
#   HELPERS
# ─────────────────────────────────────────────
def section_header(icon: str, label: str):
    st.markdown(f"""
<div class="section-hdr">
  <span class="section-hdr-icon">{icon}</span>
  <span class="section-hdr-text">{label}</span>
  <div class="section-hdr-line"></div>
</div>
""", unsafe_allow_html=True)


PLOT_CFG = dict(
    template="plotly_white",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(245,243,240,0.85)',
    font=dict(family='JetBrains Mono', color='#555555', size=11),
)


# ─────────────────────────────────────────────
#   FOOTER
# ─────────────────────────────────────────────
def render_footer():
    st.markdown("""
<div class="site-footer">
  <div class="footer-disclaimer">
    ⚠️ &nbsp; For academic and research purposes only. Not intended for clinical or medical use.
  </div>
  <div class="footer-sub">
    PathwayNet v2.1 &nbsp;·&nbsp; GDSC1 Dataset &nbsp;·&nbsp; KEGG &amp; MSigDB Hallmark Pathways
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#   CREDITS PAGE
# ─────────────────────────────────────────────
def render_credits_page():
    st.markdown("""
<div class="credits-hero">
  <div class="credits-institute-badge">Team 9 &nbsp;·&nbsp; Academic Project</div>
  <div class="credits-project-title">PathwayNet: Bio-Computational Pharmacogenomics Platform</div>
  <div style="height:14px;"></div>
  <div class="credits-dept">Department of Biotechnology</div>
  <div class="credits-dept">SRM Institute of Science and Technology, Ramapuram</div>
  <div style="height:8px;"></div>
  <div class="credits-institute">Academic Year 2025 &ndash; 2026</div>
</div>
""", unsafe_allow_html=True)

    # Guide + HOD
    st.markdown('<div class="credits-section-title">Faculty Guidance</div>', unsafe_allow_html=True)
    col_g, col_h = st.columns(2, gap="large")
    with col_g:
        st.markdown("""
<div class="credits-guide-card">
  <div class="credits-guide-label">Internal Guide</div>
  <div class="credits-guide-name">Dr. P. Kowsalya</div>
  <div class="credits-guide-quals">M.Tech., Ph.D.</div>
  <div style="height:12px;"></div>
  <div style="font-family:'Inter',sans-serif;font-size:0.82em;color:#555;line-height:1.75;">
    Department of Biotechnology<br>
    SRM Institute of Science and Technology, Ramapuram
  </div>
</div>
""", unsafe_allow_html=True)

    with col_h:
        st.markdown("""
<div class="credits-hod-card">
  <div class="credits-guide-label">Special Thanks &nbsp; &mdash; &nbsp; Head of Department</div>
  <div class="credits-hod-name">Dr. R.V. Hemavathy</div>
  <div style="font-family:'Inter',sans-serif;font-size:0.82em;color:#4a8c6f;margin-bottom:12px;">
    B.Tech., M.Tech., Ph.D. &nbsp;|&nbsp; Associate Professor &amp; HOD
  </div>
  <div style="font-family:'Inter',sans-serif;font-size:0.82em;color:#4a8c6f88;line-height:1.75;">
    Department of Biotechnology<br>
    SRM Institute of Science and Technology, Ramapuram
  </div>
</div>
""", unsafe_allow_html=True)

    # Team members
    st.markdown('<div class="credits-section-title">Project Team</div>', unsafe_allow_html=True)
    members = [
        ("🧬", "Deepshika", "RA2211009020063", "Research Lead"),
        ("💻", "Tony Royce R", "RA2211009020016", "Lead Developer"),
        ("🔬", "Kamesh", "RA2211009020044", "Quality Assurance"),
    ]
    cols = st.columns(3, gap="large")
    for col, (icon, name, reg, role) in zip(cols, members):
        with col:
            st.markdown(f"""
<div class="credits-member-card">
  <div class="credits-member-avatar">{icon}</div>
  <div class="credits-member-name">{name}</div>
  <div class="credits-member-id">{reg}</div>
  <div class="credits-member-role">{role}</div>
</div>
""", unsafe_allow_html=True)

    # Data and tools
    st.markdown('<div class="credits-section-title">Datasets and Tools Used</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="credits-data-row">
  <div class="credits-data-pill">Dataset<span>GDSC1 (Genomics of Drug Sensitivity in Cancer)</span></div>
  <div class="credits-data-pill">Pathways<span>KEGG Canonical Pathways</span></div>
  <div class="credits-data-pill">Pathways<span>MSigDB Hallmark Gene Sets v7</span></div>
  <div class="credits-data-pill">Framework<span>PyTorch (Neural Network)</span></div>
  <div class="credits-data-pill">Framework<span>Streamlit (Web Application)</span></div>
  <div class="credits-data-pill">Analysis<span>scikit-learn, SciPy, NumPy, Pandas</span></div>
  <div class="credits-data-pill">Visualisation<span>Plotly (Interactive Charts)</span></div>
  <div class="credits-data-pill">Language<span>Python 3.11</span></div>
</div>
""", unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
<div style="margin-top:44px; background:#f5f3f0; border:1px solid #dddad5;
     border-radius:12px; padding:24px 28px; text-align:center;">
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.65em;letter-spacing:2.5px;
              text-transform:uppercase;color:#9a9390;margin-bottom:10px;">Academic Disclaimer</div>
  <div style="font-family:'Inter',sans-serif;font-size:0.85em;color:#555;line-height:1.9;">
    This project was developed as an academic submission by undergraduate students of the Department
    of Biotechnology, SRM Institute of Science and Technology, Ramapuram. The platform is intended
    solely for educational and research exploration. It does not constitute a clinical tool, a
    validated diagnostic instrument, or a medical device of any description. All predictions
    generated by this system must be treated as computational hypotheses requiring laboratory
    validation before any scientific conclusions may be drawn.
  </div>
</div>
""", unsafe_allow_html=True)
    render_footer()


# ─────────────────────────────────────────────
#   RESEARCHER / TEACHER PAGE
# ─────────────────────────────────────────────
def render_researcher_page():
    st.markdown("""
<div class="res-hero">
  <div class="res-tag">📄 Technical Documentation</div>
  <div class="res-title">PathwayNet: A Biologically-Constrained<br>Neural Network for Pharmacogenomics</div>
  <div style="font-family:'JetBrains Mono',monospace;font-size:0.68em;letter-spacing:2.5px;
              text-transform:uppercase;color:#9a9390;margin:10px 0 0 0;">
    Team 9 &nbsp;|&nbsp; Dept. of Biotechnology &nbsp;|&nbsp; SRM IST Ramapuram
  </div>
  <div class="res-abstract">
    PathwayNet is an in-silico pharmacogenomics platform that integrates somatic mutation profiles
    from cancer cell lines with drug sensitivity data from the GDSC1 database to train a
    biologically-constrained sparse neural network. The network architecture is defined by
    gene-pathway membership drawn from KEGG canonical pathways and MSigDB Hallmark gene sets.
    The system identifies pathway-level resistance mechanisms, benchmarks biological structure
    against a randomised null model, performs cancer-type-stratified IC50 distribution analysis,
    and executes counterfactual in-silico perturbation to evaluate candidate therapeutic targets.
    This document provides detailed technical and conceptual documentation intended for educators,
    reviewers, and researchers evaluating the project.
  </div>
</div>
""", unsafe_allow_html=True)

    # 1. Research Motivation
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 01</div>
  <div class="res-card-title">Research Motivation and Problem Statement</div>
  <div class="res-body">
    <strong>The Clinical Problem:</strong> Intrinsic and acquired drug resistance remains one of
    the foremost challenges in oncology. Despite the availability of molecularly targeted therapies,
    a significant proportion of patients either fail to respond at baseline or develop resistance
    after an initial period of response. Understanding which genomic features determine drug
    sensitivity is therefore a research priority of substantial clinical importance.
    <br><br>
    <strong>The Computational Gap:</strong> High-throughput pharmacogenomics screening programmes
    such as GDSC1 and CCLE have generated large-scale datasets pairing cancer cell line genomic
    profiles with drug sensitivity measurements. However, most existing machine learning approaches
    applied to these datasets treat genes as independent features, ignoring the rich functional
    relationships encoded within biological pathways. This results in models that, even when
    predictively accurate, offer limited mechanistic interpretability.
    <br><br>
    <strong>The Hypothesis Driving This Project:</strong> Incorporating prior biological knowledge,
    specifically pathway membership, directly into the neural network architecture will produce a
    model that is both more interpretable and at least as predictive as a fully connected baseline.
    If the biologically structured model outperforms a pathway-randomised control, this provides
    empirical evidence that pathway structure genuinely encodes information relevant to drug
    response prediction.
    <br><br>
    <strong>Scope and Boundaries:</strong> This platform is designed as a research hypothesis
    generator and educational demonstration. All analyses are performed on in-vitro cell line data.
    No claims are made regarding in-vivo efficacy, patient-level prediction, or clinical
    applicability of any output produced by this system.
  </div>
</div>
""", unsafe_allow_html=True)

    # 2. Data Sources
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 02</div>
  <div class="res-card-title">Data Sources and Preprocessing</div>
  <div class="res-body">
    <strong>GDSC1 Drug Sensitivity Dataset:</strong> The Genomics of Drug Sensitivity in Cancer
    (GDSC1) database, published and maintained by the Wellcome Sanger Institute, provides
    pharmacological profiles for over 1,000 cancer cell lines tested with more than 400 anti-cancer
    compounds. The primary sensitivity metric used in this platform is the natural logarithm of
    the IC50 value (ln IC50), where IC50 represents the drug concentration required to inhibit
    cell viability by 50 percent. Lower ln IC50 values indicate greater drug sensitivity; higher
    values indicate resistance. The dataset also provides area under the dose-response curve (AUC),
    Z-scores, RMSE of curve fitting, and TCGA cancer type classification for each cell line.
    <br><br>
    <strong>Mutation Data:</strong> Somatic mutation profiles are derived from whole-exome or
    whole-genome sequencing of cancer cell lines. The variant allele frequency (VAF) for each
    gene in each cell line is binarised: any VAF greater than zero is treated as a mutation
    present (value of 1.0); zero VAF is treated as wild-type (value of 0.0). This produces a
    binary mutation matrix of dimensions N x G, where N is the number of cell lines and G is
    the number of unique genes observed in the dataset.
    <br><br>
    <strong>Pathway Gene Sets:</strong> Two curated pathway databases are used to define the
    gene-pathway membership structure:
    <ul>
      <li><strong>KEGG Canonical Pathways</strong> (c2.cp.kegg gene set collection from MSigDB
      v7): biochemically curated pathways covering metabolism, signalling, and disease.</li>
      <li><strong>MSigDB Hallmark Gene Sets</strong> (h.all.v7): 50 coherent biological
      processes derived by aggregating canonical pathway data to reduce redundancy.</li>
    </ul>
    <strong>Alignment and Filtering:</strong> Cell lines present in both the mutation matrix and
    the drug sensitivity data are retained for analysis. Only cell lines with a minimum of 40
    overlapping samples are included for a given drug. Pathways with at least one gene
    represented in the mutation matrix are used to build the masking structure.
  </div>
  <div class="res-highlight">
    <strong>Key preprocessing note:</strong> No gene expression, copy number variation, or
    methylation data is incorporated at this stage. The model operates exclusively on binary
    somatic mutation profiles, which represents a deliberate simplification to isolate the
    contribution of mutational structure.
  </div>
</div>
""", unsafe_allow_html=True)

    # 3. Model Architecture
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 03</div>
  <div class="res-card-title">Model Architecture: Biologically-Constrained Sparse Neural Network</div>
  <div class="res-body">
    <strong>Core Design Principle:</strong> The first layer of the neural network is not fully
    connected. Each neuron in the first layer corresponds to one biological pathway, and a neuron
    receives input only from genes that belong to that pathway according to KEGG or Hallmark
    definitions. This is implemented via a binary mask matrix M of dimensions G x P, where G is
    the number of genes and P is the number of active pathways. The element M[g,p] equals 1 if
    gene g belongs to pathway p, and 0 otherwise.
    <br><br>
    <strong>Forward Pass:</strong> During the forward pass, the effective weight matrix is computed
    as the element-wise product of the learnable weight matrix W and the fixed mask M. This ensures
    that gradients do not flow through zero-masked connections, enforcing sparsity without requiring
    specialised sparse operations.
  </div>
  <div class="res-formula">
    Input layer:&nbsp;&nbsp;&nbsp;&nbsp; x &isin; {0,1}&sup;G &nbsp;&nbsp;&nbsp;(binary mutation vector)<br>
    Pathway layer: &nbsp;&nbsp; h&#8321; = ELU( x &middot; (W &odot; M) + b&#8321; ) &nbsp;&nbsp;&nbsp;shape: (P,)<br>
    Hidden layer: &nbsp;&nbsp;&nbsp; h&#8322; = ELU( W&#8322; h&#8321; + b&#8322; ) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shape: (32,)<br>
    Output: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; y&#770; = W&#8323; h&#8322; + b&#8323; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;shape: (1,)
  </div>
  <div class="res-body">
    <strong>Activation Function:</strong> The Exponential Linear Unit (ELU) is used in place of
    ReLU throughout the network. ELU produces non-zero outputs for negative inputs, which avoids
    the dying neuron problem and produces smoother gradient flow, which is beneficial when working
    with sparse binary input matrices.
    <br><br>
    <strong>Training Configuration:</strong> The model is trained with the Adam optimiser at a
    learning rate of 0.005, using mean squared error (MSE) loss against standardised IC50 values
    (StandardScaler applied to training targets only). Training runs for 200 epochs. A 75/25
    train-test split is applied with a fixed random seed (42) for reproducibility.
    <br><br>
    <strong>Randomised Control Model:</strong> An identical architecture is trained in parallel,
    but with the mask matrix column-shuffled independently for each pathway. This preserves the
    overall sparsity structure (each pathway receives the same number of gene inputs) while
    destroying the biological meaning of the connections. Performance comparison between the
    biological model and this null model constitutes the structure validity test.
  </div>
</div>
""", unsafe_allow_html=True)

    # 4. Evaluation Metrics
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 04</div>
  <div class="res-card-title">Evaluation Metrics and Their Interpretation</div>
  <div class="res-body">
    The following metrics are computed on held-out test data to characterise model performance.
    All metrics compare predicted ln IC50 values against experimentally measured values.
  </div>
  <div class="res-metric-row">
    <div class="res-metric-box">
      <div class="res-metric-name">Pearson R</div>
      <div class="res-metric-val">r</div>
      <div class="res-metric-desc">Linear correlation between predicted and actual values. Range -1 to 1. Threshold of 0.3 is commonly cited as meaningful in GDSC literature.</div>
    </div>
    <div class="res-metric-box">
      <div class="res-metric-name">Spearman rho</div>
      <div class="res-metric-val">&rho;</div>
      <div class="res-metric-desc">Rank-order correlation. More robust to outliers than Pearson R. Appropriate for evaluating relative ranking of drug sensitivity.</div>
    </div>
    <div class="res-metric-box">
      <div class="res-metric-name">R squared</div>
      <div class="res-metric-val">R&sup2;</div>
      <div class="res-metric-desc">Coefficient of determination. Proportion of variance in IC50 explained by the model. Negative values indicate the model performs worse than predicting the mean.</div>
    </div>
    <div class="res-metric-box">
      <div class="res-metric-name">MAE / RMSE</div>
      <div class="res-metric-val">ln</div>
      <div class="res-metric-desc">Mean absolute error and root mean squared error in ln IC50 units. RMSE penalises large individual errors more heavily than MAE.</div>
    </div>
  </div>
  <div class="res-body" style="margin-top:18px;">
    <strong>Structure Validity Test:</strong> The delta Pearson R (Bio model R minus Randomised
    control R) is the primary indicator of whether biological pathway structure contributes
    meaningful signal. A positive delta provides evidence in support of the biological structure
    hypothesis. A delta below zero would indicate the randomised model performs equivalently or
    better, suggesting either data insufficiency or that the mutation-only feature space is too
    sparse to benefit from pathway constraint.
  </div>
</div>
""", unsafe_allow_html=True)

    # 5. Resistance driver analysis
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 05</div>
  <div class="res-card-title">Pathway-Level Resistance Mechanism Analysis</div>
  <div class="res-body">
    <strong>Method:</strong> Following model training and evaluation, the pathway activation
    values (outputs of the first network layer, h1) are extracted for all test set samples.
    The test samples are ranked by predicted IC50 and divided into two groups: the top 15 most
    resistant samples and the bottom 15 most sensitive samples.
    <br><br>
    For each pathway p, the differential activation score is computed as:
  </div>
  <div class="res-formula">
    delta&#8346; = mean( h&#8321;&#8346;[resistant samples] ) &minus; mean( h&#8321;&#8346;[sensitive samples] )
  </div>
  <div class="res-body">
    Pathways with a large positive delta are interpreted as being disproportionately active in
    resistant cells. Pathways with a large negative delta are more active in sensitive cells.
    The top 18 pathways by absolute delta value are visualised in the resistance driver bar chart.
    <br><br>
    <strong>Interpretive Caution:</strong> Differential pathway activation in the neural network
    reflects learned associations from the training data. It does not establish causal mechanisms.
    Over-representation of specific pathways in resistant cells may reflect passenger mutations
    rather than driver events, and must be validated by independent biological experiments.
  </div>
</div>
""", unsafe_allow_html=True)

    # 6. Virtual CRISPR
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 06</div>
  <div class="res-card-title">Counterfactual Perturbation Analysis (In-Silico CRISPR)</div>
  <div class="res-body">
    <strong>Conceptual Framework:</strong> Counterfactual analysis in this context asks: if the
    mutational status of all genes belonging to the top resistance pathway were set to wild-type
    (zero), how would the model's predicted IC50 change for the most resistant cell line? This
    mimics, computationally, the effect of a complete CRISPR-mediated loss-of-function knockdown
    of every mutated gene in that pathway.
    <br><br>
    <strong>Procedure:</strong> The test cell line with the highest predicted IC50 is selected.
    Its binary mutation vector is copied. All positions corresponding to genes in the target
    pathway that are present in the mutation matrix are set to zero. The modified vector is passed
    through the trained biological model to obtain a counterfactual predicted IC50.
  </div>
  <div class="res-formula">
    Sensitivity Delta = Baseline Predicted IC50 &minus; Counterfactual Predicted IC50<br><br>
    Positive delta &nbsp;&rarr;&nbsp; Pathway knockout reduces predicted resistance (valid target hypothesis)<br>
    Negative delta &nbsp;&rarr;&nbsp; Pathway knockout increases or does not reduce resistance (escape redundancy)
  </div>
  <div class="res-body">
    <strong>Limitations of the Perturbation Approach:</strong> This analysis assumes that the
    trained model generalises correctly to out-of-distribution inputs, which cannot be guaranteed.
    The model was not trained on CRISPR-perturbed data. The result represents the model's learned
    extrapolation rather than a biologically validated outcome. Furthermore, epistasis, compensatory
    pathway activation, and off-target effects are not modelled.
  </div>
  <div class="res-highlight">
    <strong>Appropriate Use:</strong> The perturbation result should be treated as a ranked
    hypothesis for wet-lab follow-up, not as a validated biological finding. It is most useful
    when combined with literature evidence and independent experimental validation.
  </div>
</div>
""", unsafe_allow_html=True)

    # 7. Cancer type analysis
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 07</div>
  <div class="res-card-title">Cancer-Type-Stratified IC50 Distribution</div>
  <div class="res-body">
    <strong>Purpose:</strong> The GDSC1 dataset assigns each cell line a TCGA cancer type
    classification. Stratifying the IC50 distribution by cancer type reveals which tumour lineages
    are inherently more or less sensitive to a given compound, independent of any model prediction.
    This analysis is derived directly from the raw experimental data and does not depend on the
    neural network.
    <br><br>
    <strong>Visualisation:</strong> Horizontal box plots are rendered for each cancer type,
    ordered from lowest median ln IC50 (most sensitive) to highest (most resistant). Boxes
    represent the interquartile range (25th to 75th percentile). The internal line marks the
    median. The diamond marker represents the mean. Colour is mapped to median IC50 value using
    a green-to-red gradient scale.
    <br><br>
    <strong>Interpretive Note:</strong> Differences in IC50 distribution between cancer types
    reflect the aggregate pharmacological response of the cell lines within each lineage. They
    are influenced by both the genetic background of the cell lines and the biological mechanism
    of the drug. A cell line classified as SKCM (melanoma), for example, may respond differently
    from primary melanoma tumours in a clinical setting due to the absence of a tumour
    microenvironment, immune infiltration, and stromal interactions in the in-vitro model.
  </div>
</div>
""", unsafe_allow_html=True)

    # 8. References
    st.markdown("""
<div class="res-card">
  <div class="res-card-num">Section 08</div>
  <div class="res-card-title">Key References and Data Sources</div>
  <div class="res-ref-item">
    <span class="res-ref-num">[1]</span>
    Yang, W. et al. (2013). Genomics of Drug Sensitivity in Cancer (GDSC): a resource for
    therapeutic biomarker discovery in cancer cells. <em>Nucleic Acids Research</em>, 41(D1),
    D955-D961.
  </div>
  <div class="res-ref-item">
    <span class="res-ref-num">[2]</span>
    Kanehisa, M. and Goto, S. (2000). KEGG: Kyoto Encyclopedia of Genes and Genomes.
    <em>Nucleic Acids Research</em>, 28(1), 27-30.
  </div>
  <div class="res-ref-item">
    <span class="res-ref-num">[3]</span>
    Liberzon, A. et al. (2015). The Molecular Signatures Database (MSigDB) Hallmark Gene Set
    Collection. <em>Cell Systems</em>, 1(6), 417-425.
  </div>
  <div class="res-ref-item">
    <span class="res-ref-num">[4]</span>
    Elmarakeby, H.A. et al. (2021). Biologically informed deep neural network for prostate
    cancer discovery. <em>Nature</em>, 598(7880), 348-352. (Conceptual basis for pathway-masked
    network architecture.)
  </div>
  <div class="res-ref-item">
    <span class="res-ref-num">[5]</span>
    Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning
    Library. <em>Advances in Neural Information Processing Systems</em>, 32.
  </div>
  <div class="res-ref-item">
    <span class="res-ref-num">[6]</span>
    Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
    <em>Journal of Machine Learning Research</em>, 12, 2825-2830.
  </div>
</div>
""", unsafe_allow_html=True)

    render_footer()


# ─────────────────────────────────────────────
#   LEARN PAGE
# ─────────────────────────────────────────────
def render_learn_page():
    st.markdown("""
<div class="learn-hero">
  <div class="learn-hero-tag">Educational Guide &nbsp;|&nbsp; Accessible Science</div>
  <div class="learn-hero-title">Understanding PathwayNet:<br>From Biology to Computation</div>
  <div class="learn-hero-body">
    This guide is designed for students, educators, and individuals without a specialised background
    in bioinformatics or computational biology. Each section builds progressively on the previous,
    introducing concepts with clarity and precision. Where technical terms are unavoidable, they
    are defined immediately upon introduction.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-toc">
  <div class="learn-toc-title">Contents of This Guide</div>
  <div class="learn-toc-item"><span class="learn-toc-num">01</span> The Biology of Cancer and Drug Resistance</div>
  <div class="learn-toc-item"><span class="learn-toc-num">02</span> Genes, Mutations, and Biological Pathways</div>
  <div class="learn-toc-item"><span class="learn-toc-num">03</span> What is PathwayNet and What Question Does It Answer?</div>
  <div class="learn-toc-item"><span class="learn-toc-num">04</span> The Data Behind the Tool</div>
  <div class="learn-toc-item"><span class="learn-toc-num">05</span> How the Neural Network is Built and Trained</div>
  <div class="learn-toc-item"><span class="learn-toc-num">06</span> What is CRISPR and What Does the Virtual Experiment Simulate?</div>
  <div class="learn-toc-item"><span class="learn-toc-num">07</span> How to Interpret the Charts and Output Metrics</div>
  <div class="learn-toc-item"><span class="learn-toc-num">08</span> Cancer Types Covered by This Platform</div>
  <div class="learn-toc-item"><span class="learn-toc-num">09</span> Glossary of Key Terms</div>
  <div class="learn-toc-item"><span class="learn-toc-num">10</span> Limitations and Responsible Interpretation</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x1F9A0;</span>
  <div class="learn-section-num">Section 01</div>
  <div class="learn-section-title">The Biology of Cancer and Drug Resistance</div>
  <div class="learn-body">
    <strong>What is cancer?</strong> The human body is composed of trillions of cells, each
    governed by a precise set of genetic instructions encoded within deoxyribonucleic acid (DNA).
    Under normal conditions, cells divide in a regulated manner: they grow, perform their
    designated function, and eventually undergo programmed death to make way for new cells.
    Cancer arises when this regulation breaks down. Certain cells begin dividing uncontrollably,
    ignore signals that should halt their growth, and may invade surrounding tissues or migrate
    to distant organs through a process called metastasis.
    <br><br>
    <strong>Why do cancers differ from one another?</strong> Cancer is not a single disease. It
    is a broad category encompassing over 200 distinct conditions, each characterised by different
    cell types of origin, different genetic alterations, and different clinical behaviours. Lung
    cancer, breast cancer, and leukaemia arise in entirely different cell types and respond to
    different treatments. Even within a single cancer type, individual tumours may carry markedly
    different genetic mutations, which explains why two patients with the same diagnosis may
    respond very differently to the same drug.
    <br><br>
    <strong>What is drug resistance?</strong> When an anti-cancer drug is administered, it may
    initially be effective at suppressing tumour cells. However, within a tumour, not all cells
    are genetically identical. Some cells may carry mutations that confer a survival advantage in
    the presence of the drug. These cells survive treatment, continue to proliferate, and give
    rise to a tumour population that no longer responds to the original therapy. This process is
    called acquired drug resistance, and it represents one of the most significant obstacles in
    cancer treatment today.
    <br><br>
    <strong>Why is predicting resistance important?</strong> If researchers can identify, prior
    to treatment, which genetic features are associated with resistance to a particular drug, they
    can make more informed decisions about which therapies to use, which combination approaches
    might be effective, and which alternative biological targets are worth pursuing.
    Computational tools such as PathwayNet are designed to assist in this prediction process.
  </div>
  <div class="learn-analogy">
    <div class="learn-analogy-label">Conceptual Illustration</div>
    Consider a population of bacteria treated with an antibiotic. Most bacteria are eliminated,
    but a small subset carrying a resistance-conferring mutation survive and reproduce. Subsequent
    generations are predominantly resistant. The situation in cancer is analogous: the drug applies
    selection pressure, and genetically resistant subpopulations outcompete sensitive ones over time.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x1F9EC;</span>
  <div class="learn-section-num">Section 02</div>
  <div class="learn-section-title">Genes, Mutations, and Biological Pathways</div>
  <div class="learn-body">
    <strong>What is a gene?</strong> A gene is a discrete segment of DNA that carries the
    instructions for producing a specific protein. Proteins are the molecular machines that carry
    out virtually every function within a cell: they catalyse chemical reactions, transmit signals,
    provide structural support, and regulate the activity of other genes. The human genome contains
    approximately 20,000 protein-coding genes.
    <br><br>
    <strong>What is a mutation?</strong> A mutation is an alteration in the DNA sequence of a
    gene. Mutations may arise spontaneously during DNA replication, as a result of environmental
    exposures such as ultraviolet radiation or chemical carcinogens, or through inherited
    predispositions. Not all mutations are harmful; many are silent and do not affect protein
    function. However, mutations in genes that regulate cell growth, division, or survival can
    disrupt the normal balance and contribute to cancer. In this platform, the focus is on
    somatic mutations, meaning mutations that arise within a cell during an individual's lifetime
    rather than being inherited.
    <br><br>
    <strong>What is a biological pathway?</strong> Genes and their protein products rarely act
    in isolation. They function within structured networks of interacting molecules called
    biological pathways. A pathway is a series of molecular events, each step triggered or
    modulated by the product of the previous step, that together accomplish a defined cellular
    function. For example, the PI3K-AKT-mTOR signalling pathway governs cell survival and growth;
    the DNA damage response pathway detects and repairs errors in the genome; the apoptosis pathway
    orchestrates programmed cell death. Many cancer-associated mutations affect genes within these
    critical pathways.
    <br><br>
    <strong>Why pathways matter for drug response:</strong> Because genes within a pathway work
    in concert, a mutation in any one member may alter the overall activity of the entire pathway.
    Furthermore, if a drug targets a specific pathway, mutations occurring upstream or downstream
    of the drug target may either sensitise or desensitise the cell to the drug. This is why
    pathway-level analysis provides more biologically meaningful insight than examining individual
    genes in isolation.
  </div>
  <div class="learn-analogy">
    <div class="learn-analogy-label">Conceptual Illustration</div>
    A biological pathway can be understood as a relay race. Each runner (protein) receives a
    baton (signal) from the previous runner and passes it forward. If one runner is incapacitated
    due to a mutation, the baton may be dropped, delayed, or redirected in an abnormal manner.
    A drug that targets a specific runner in the relay may be rendered ineffective if a mutation
    elsewhere in the relay has already altered the outcome of the race.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x1F52C;</span>
  <div class="learn-section-num">Section 03</div>
  <div class="learn-section-title">What is PathwayNet and What Question Does It Answer?</div>
  <div class="learn-body">
    <strong>The central research question:</strong> Given the somatic mutation profile of a cancer
    cell line, can a computational model accurately predict how sensitive or resistant that cell
    line will be to a specific anti-cancer compound? And can the architecture of that model be
    designed such that it reveals which biological pathways are most strongly associated with
    resistance?
    <br><br>
    <strong>The limitation of standard approaches:</strong> Most machine learning models applied
    to genomic data treat each gene as an independent input feature, ignoring the functional
    relationships between genes. A model that has no awareness of which genes share a biological
    pathway cannot leverage this organisational information to produce more informed predictions
    or biologically interpretable outputs.
    <br><br>
    <strong>The PathwayNet approach:</strong> PathwayNet addresses this by embedding biological
    pathway knowledge directly into the architecture of the neural network. The connections between
    the input layer (individual genes) and the first hidden layer (biological pathways) are
    restricted to reflect actual gene-pathway membership, as defined by the KEGG and Hallmark
    databases. Genes can only connect to the pathways they are known to participate in. This
    creates a model whose internal computations correspond directly to identifiable biological
    entities, making its outputs interpretable in a biologically meaningful context.
    <br><br>
    <strong>What the tool produces:</strong> For any selected drug in the GDSC1 dataset,
    PathwayNet generates a drug sensitivity prediction model, a benchmarking comparison against a
    randomised structural control, a ranked list of pathways associated with drug resistance, a
    cancer-type-stratified distribution of IC50 values, and a counterfactual simulation of pathway
    inhibition via in-silico perturbation.
  </div>
  <div class="learn-callout">
    <strong>Core principle:</strong> By constraining the model architecture to reflect known
    biology, PathwayNet produces results that are not only predictive but also mechanistically
    interpretable, providing experimentally testable hypotheses to guide downstream laboratory
    investigation.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x1F5C4;&#xFE0F;</span>
  <div class="learn-section-num">Section 04</div>
  <div class="learn-section-title">The Data Behind the Tool</div>
  <div class="learn-body">
    <strong>GDSC1 (Genomics of Drug Sensitivity in Cancer):</strong> Produced by the Wellcome
    Sanger Institute in the United Kingdom, GDSC1 is one of the largest publicly available
    pharmacogenomics datasets. It reports IC50 values for over 1,000 cancer cell lines tested
    against hundreds of anti-cancer compounds under standardised laboratory conditions. The natural
    logarithm of the IC50 (ln IC50) is used as the response variable in this model, as the
    logarithmic transformation normalises the highly skewed distribution of raw concentration
    values and makes the data more suitable for regression analysis.
    <br><br>
    <strong>Cancer cell lines:</strong> A cell line is a population of cancer cells derived from
    a patient tumour sample that is grown continuously in laboratory conditions. Cell lines allow
    researchers to test drugs under controlled, reproducible conditions on a scale that would be
    impossible directly in patients. However, they represent a significant simplification of real
    tumours; they lack the three-dimensional architecture, vascular supply, immune cell infiltration,
    and stromal interactions present in living tissue.
    <br><br>
    <strong>Mutation data:</strong> Each cell line has been genomically characterised through DNA
    sequencing. The mutation data records which genes carry somatic mutations in each cell line.
    This is encoded as a binary matrix: a value of 1 indicates that at least one mutation was
    detected in a given gene in a given cell line; a value of 0 indicates no detected mutation.
    <br><br>
    <strong>Pathway databases:</strong> KEGG (Kyoto Encyclopedia of Genes and Genomes) and the
    MSigDB Hallmark collection define the gene-pathway membership that structures the neural
    network. KEGG provides detailed, biochemically curated pathway maps covering metabolism,
    signalling, and disease. The Hallmark collection provides 50 well-defined, minimally
    redundant gene sets representing major biological states and processes.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x2699;&#xFE0F;</span>
  <div class="learn-section-num">Section 05</div>
  <div class="learn-section-title">How the Neural Network is Built and Trained</div>
  <div class="learn-body">
    <strong>What is a neural network?</strong> A neural network is a computational system
    composed of layers of mathematical units called neurons, connected by weighted edges. It
    learns by systematically adjusting these weights during training to minimise the difference
    between its predictions and the actual measured values in the training data. Neural networks
    are particularly well-suited for learning complex, non-linear relationships within large
    biological datasets.
    <br><br>
    <strong>The pathway constraint:</strong> In a standard neural network, every neuron in one
    layer is connected to every neuron in the next layer. PathwayNet replaces this fully connected
    first layer with a sparse layer governed by a binary mask matrix. This matrix is constructed
    from the pathway database and contains a value of 1 only where a gene is known to belong to
    a specific biological pathway. The network weights in the first layer are multiplied by this
    fixed mask before each calculation, ensuring that only biologically valid gene-to-pathway
    connections contribute to the model's predictions.
    <br><br>
    <strong>Training procedure:</strong> The model is trained on 75 percent of the available
    cell line data and evaluated on the remaining 25 percent. During training, the network is
    presented with binary mutation profiles paired with drug sensitivity values. It iteratively
    adjusts its internal weights to produce predictions that more closely match the measured
    values. The Adam optimiser governs the weight update process over 200 training iterations,
    known as epochs.
    <br><br>
    <strong>The randomised control:</strong> An identical architecture is simultaneously trained
    with a shuffled version of the pathway mask, in which each pathway still receives the same
    number of gene inputs but those genes are randomly selected rather than drawn from known
    biological members. This control establishes the performance level achievable without
    biological structure. If the biologically-constrained model substantially outperforms the
    randomised control, this constitutes evidence that the pathway structure encodes genuine,
    informative biological organisation.
  </div>
  <div class="learn-analogy">
    <div class="learn-analogy-label">Conceptual Illustration</div>
    Training a neural network is comparable to calibrating a sensitive instrument. Initially,
    the instrument produces inaccurate readings. With each adjustment cycle, it is fine-tuned
    until its outputs most closely correspond to the known reference values. The pathway
    constraint is equivalent to pre-configuring the instrument to only register signals from
    frequency bands where meaningful data is known to exist, rather than scanning across all
    possible frequencies indiscriminately.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x2702;&#xFE0F;</span>
  <div class="learn-section-num">Section 06</div>
  <div class="learn-section-title">What is CRISPR and What Does the Virtual Experiment Simulate?</div>
  <div class="learn-body">
    <strong>What is CRISPR-Cas9?</strong> CRISPR-Cas9 is a molecular tool that enables
    researchers to edit DNA within living cells with high precision. It functions by directing
    a protein called Cas9 to a specific location in the genome, where it introduces a targeted
    cut in the DNA strand. Depending on how the cell repairs this cut, the targeted gene can
    be disrupted, corrected, or replaced. In cancer research, CRISPR knockout experiments are
    used to investigate what happens to a cell when a specific gene is no longer functional.
    <br><br>
    <strong>What does a knockout experiment tell researchers?</strong> If a gene suspected of
    driving drug resistance is knocked out and the cancer cells subsequently become more sensitive
    to the drug, this provides strong evidence that the gene (or its associated pathway) is
    contributing to resistance and represents a potential therapeutic target for further
    investigation.
    <br><br>
    <strong>What does PathwayNet simulate?</strong> Because real CRISPR experiments are
    time-consuming and costly, PathwayNet provides a computational approximation of this
    experimental approach. The tool identifies the most resistant cell line in the test set and
    determines which pathway is most strongly associated with its resistance. It then creates a
    modified version of that cell line's mutation profile in which all mutated genes within the
    target pathway are set to zero, computationally simulating a state in which those mutations
    are absent. This modified profile is passed through the trained neural network to generate
    a counterfactual predicted IC50 value.
    <br><br>
    <strong>Interpreting the result:</strong> If the counterfactual IC50 is lower than the
    baseline (that is, if the sensitivity delta is positive), the model predicts that eliminating
    mutational activity in that pathway would increase the cell's responsiveness to the drug.
    This is presented as a hypothesis warranting experimental validation, not as a confirmed
    biological finding.
  </div>
  <div class="learn-warn">
    This analysis constitutes a mathematical simulation conducted on a trained statistical model.
    It represents a computational hypothesis and must not be interpreted as experimental evidence.
    Independent laboratory validation is required before any scientific conclusion may be drawn.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x1F4CA;</span>
  <div class="learn-section-num">Section 07</div>
  <div class="learn-section-title">How to Interpret the Charts and Output Metrics</div>
  <div class="learn-body">
    <strong>IC50 and ln IC50:</strong> The IC50 is the drug concentration required to reduce
    cell viability by 50 percent. A lower IC50 indicates high sensitivity (a small amount of
    drug produces a strong effect). A higher IC50 indicates resistance. Because IC50 values span
    many orders of magnitude, their natural logarithm (ln IC50) is used for modelling. Lower
    ln IC50 values consistently denote greater sensitivity throughout this platform.
    <br><br>
    <strong>Pearson R</strong> measures the linear correlation between the model's predicted
    ln IC50 values and the actual measured values on the held-out test set. Values range from
    negative 1 to positive 1. In the pharmacogenomics literature, a Pearson R exceeding 0.3
    is generally considered indicative of meaningful predictive performance.
    <br><br>
    <strong>Spearman rho</strong> measures rank-order correlation. It assesses whether the model
    correctly ranks cell lines from most sensitive to most resistant, independently of the
    absolute magnitude of the values. This metric is more robust to extreme outliers than Pearson R.
    <br><br>
    <strong>R squared</strong> represents the proportion of variation in measured IC50 values
    that the model can account for. A value of 0.25 indicates that 25 percent of observed
    variability is explained by the model; the remainder reflects biological factors not captured
    in the mutation data.
    <br><br>
    <strong>MAE and RMSE</strong> are both expressed in ln IC50 units and quantify average
    prediction error. RMSE gives greater weight to large individual errors due to its squared
    formulation, making it more sensitive to outliers than MAE.
    <br><br>
    <strong>Structure Validity Chart:</strong> Compares Pearson R of the biologically-constrained
    model against the randomised control. A clearly taller biological model bar indicates that
    pathway structure contributes genuine predictive signal beyond what random gene assignment
    achieves.
    <br><br>
    <strong>Actual vs Predicted Scatter Plot:</strong> Each point is a cancer cell line from
    the test set, coloured by absolute residual (green = small error, red = large error). The
    dotted diagonal represents perfect prediction; the solid line is the fitted regression.
    <br><br>
    <strong>Resistance Drivers Chart:</strong> Bars extending rightward (red) indicate pathways
    disproportionately active in resistant cell lines. Bars extending leftward (blue) indicate
    pathways more active in sensitive cells. Bar length reflects magnitude of the differential.
    <br><br>
    <strong>Cancer-Type IC50 Box Plots:</strong> Each horizontal box plot represents one cancer
    type, ordered from most sensitive (left) to most resistant (right). The box spans the
    interquartile range; the central line marks the median; the diamond symbol marks the mean.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x1F3E5;</span>
  <div class="learn-section-num">Section 08</div>
  <div class="learn-section-title">Cancer Types Covered by This Platform</div>
  <div class="learn-body">
    The GDSC1 dataset classifies each cancer cell line using TCGA abbreviations. The following
    cancer types may appear in the IC50 distribution visualisation:
  </div>
  <br>
  <div class="learn-glossary-grid">
    <div class="learn-glossary-item"><div class="learn-glossary-term">BRCA</div><div class="learn-glossary-def">Breast Invasive Carcinoma. Cancer originating in breast tissue, encompassing several molecular subtypes with differing clinical behaviour and treatment responses.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">LUAD / LUSC</div><div class="learn-glossary-def">Lung Adenocarcinoma and Lung Squamous Cell Carcinoma. Two major subtypes of non-small cell lung cancer with distinct mutational profiles.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">SKCM</div><div class="learn-glossary-def">Skin Cutaneous Melanoma. Arises from pigment-producing melanocytes and is characterised by a very high mutation burden due to ultraviolet radiation exposure.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">GBM</div><div class="learn-glossary-def">Glioblastoma Multiforme. A highly aggressive primary brain tumour with very poor prognosis and limited treatment options.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">COREAD</div><div class="learn-glossary-def">Colorectal Adenocarcinoma. Cancer of the colon or rectum; among the most common cancers worldwide, with well-characterised genetic subtypes.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">LAML</div><div class="learn-glossary-def">Acute Myeloid Leukemia. A rapidly progressing cancer of the bone marrow and blood involving accumulation of immature myeloid cells.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">DLBC</div><div class="learn-glossary-def">Diffuse Large B-cell Lymphoma. The most common form of non-Hodgkin lymphoma in adults, arising from mature B lymphocytes.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">OV</div><div class="learn-glossary-def">Ovarian Serous Cystadenocarcinoma. High-grade ovarian cancer, frequently characterised by BRCA1 and BRCA2 mutations, often diagnosed at an advanced stage.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">PAAD</div><div class="learn-glossary-def">Pancreatic Adenocarcinoma. One of the most treatment-resistant cancers, typically diagnosed late and with poor five-year survival rates.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">SCLC</div><div class="learn-glossary-def">Small Cell Lung Cancer. A neuroendocrine tumour defined by rapid growth, early metastasis, and high initial chemosensitivity followed by frequent relapse.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">MM</div><div class="learn-glossary-def">Multiple Myeloma. A malignancy of plasma cells in the bone marrow, leading to abnormal immunoglobulin production and skeletal destruction.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">STAD / ESCA</div><div class="learn-glossary-def">Stomach Adenocarcinoma and Esophageal Carcinoma. Upper gastrointestinal tract cancers sharing several genomic alterations and therapeutic vulnerabilities.</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x1F4D6;</span>
  <div class="learn-section-num">Section 09</div>
  <div class="learn-section-title">Glossary of Key Terms</div>
  <div class="learn-glossary-grid">
    <div class="learn-glossary-item"><div class="learn-glossary-term">IC50</div><div class="learn-glossary-def">Half-maximal inhibitory concentration. The drug concentration required to suppress cell viability by 50 percent. A lower IC50 indicates greater drug sensitivity.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">ln IC50</div><div class="learn-glossary-def">The natural logarithm of IC50. Used to normalise the highly skewed raw distribution of drug concentration values for statistical modelling purposes.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Somatic Mutation</div><div class="learn-glossary-def">A DNA alteration arising within a cell during an individual's lifetime, as opposed to an inherited germline mutation. Somatic mutations in oncogenes and tumour suppressors drive cancer development.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Biological Pathway</div><div class="learn-glossary-def">A structured sequence of molecular interactions between genes and proteins that together accomplish a defined cellular function, such as growth regulation or apoptosis.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Drug Resistance</div><div class="learn-glossary-def">A state in which cancer cells exhibit reduced sensitivity to a therapeutic compound, commonly arising through mutation-driven circumvention of the drug's mechanism of action.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Neural Network</div><div class="learn-glossary-def">A computational model composed of interconnected weighted units that learns predictive mappings from input data by iteratively minimising prediction error during training.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Sparse Masking</div><div class="learn-glossary-def">The technique used in PathwayNet to restrict gene-to-pathway connections within the network based on known biological pathway membership, enforcing biological structure in the model architecture.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">CRISPR-Cas9</div><div class="learn-glossary-def">A molecular gene-editing system capable of introducing targeted disruptions, corrections, or replacements in specific DNA sequences within living cells.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Counterfactual Perturbation</div><div class="learn-glossary-def">A computational simulation in which the model input is modified to represent a hypothetical condition (such as gene knockout) and the resultant predicted output is compared to the unmodified baseline.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">GDSC1</div><div class="learn-glossary-def">Genomics of Drug Sensitivity in Cancer, First Screen. A large public pharmacogenomics database providing drug sensitivity profiles for over 1,000 cancer cell lines tested with hundreds of anti-cancer compounds.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">KEGG</div><div class="learn-glossary-def">Kyoto Encyclopedia of Genes and Genomes. A comprehensive database providing curated biochemical pathway maps, molecular interaction networks, and genomic annotations.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">MSigDB Hallmark</div><div class="learn-glossary-def">Molecular Signatures Database Hallmark Gene Sets. A collection of 50 carefully curated gene sets representing coherent biological states or processes, designed to minimise inter-set redundancy.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Pearson R</div><div class="learn-glossary-def">A statistical measure of linear correlation between two continuous variables, ranging from negative 1 to positive 1. Used to assess agreement between predicted and measured drug sensitivity values.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">Cell Line</div><div class="learn-glossary-def">A population of cancer cells derived from a patient tumour and maintained in continuous laboratory culture, enabling reproducible pharmacological testing under controlled conditions.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">VAF</div><div class="learn-glossary-def">Variant Allele Frequency. The proportion of sequencing reads supporting a given variant at a specific genomic position. In this platform, any VAF greater than zero is treated as a binary mutation-present signal.</div></div>
    <div class="learn-glossary-item"><div class="learn-glossary-term">AUC</div><div class="learn-glossary-def">Area Under the dose-response Curve. A complementary drug sensitivity measure that integrates cellular response across a range of drug concentrations rather than at a single concentration point.</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="learn-section-card">
  <span class="learn-section-icon">&#x26A0;&#xFE0F;</span>
  <div class="learn-section-num">Section 10</div>
  <div class="learn-section-title">Limitations and Responsible Interpretation</div>
  <div class="learn-body">
    <strong>1. In-vitro data does not fully represent in-vivo biology.</strong> All experimental
    data in GDSC1 was generated from cancer cell lines maintained in two-dimensional culture
    conditions. Real tumours exist within a complex tissue microenvironment involving vascular
    supply, immune cell infiltration, extracellular matrix interactions, and paracrine signalling.
    These factors substantially influence drug sensitivity in living organisms and are not
    represented in cell line data. Predictions made by this model must not be extrapolated to
    patient tumours.
    <br><br>
    <strong>2. The model relies exclusively on binary somatic mutation data.</strong> Drug
    sensitivity is influenced by numerous molecular features beyond somatic mutations, including
    gene expression levels, copy number variations, DNA methylation patterns, protein abundance,
    and post-translational modifications. The binary mutation-only feature representation used
    here is a deliberate simplification that limits the model's explanatory capacity relative
    to multi-omic approaches.
    <br><br>
    <strong>3. Predictive accuracy is partial, not complete.</strong> Pearson R values in the
    range of 0.3 to 0.6 are typical for pharmacogenomics prediction models, reflecting genuine
    biological noise and the inherent difficulty of predicting drug sensitivity from genomic
    data alone. A low Pearson R for a given drug does not necessarily indicate model failure;
    it may reflect limitations of the feature space itself.
    <br><br>
    <strong>4. The counterfactual perturbation is a mathematical extrapolation only.</strong>
    The virtual CRISPR simulation extrapolates the trained model to an input configuration it
    was not trained on. The model cannot account for compensatory pathway activation, synthetic
    lethality interactions, or off-target biological effects. All perturbation results must be
    treated as hypothesis-generating and not hypothesis-confirming.
    <br><br>
    <strong>5. Correlation does not imply causation.</strong> Pathways identified as resistance
    drivers reflect associations learned from the training data. These associations may involve
    passenger mutations that co-occur with true driver events rather than reflecting direct causal
    mechanisms. Biochemical validation is required before any pathway is designated as a genuine
    resistance mechanism.
  </div>
  <div class="learn-warn">
    This platform is provided for academic and educational purposes only. It is not a validated
    clinical tool, diagnostic instrument, or medical device of any description. No output generated
    by this system should be used to inform decisions regarding patient care, drug selection, or
    medical treatment. All computational results require independent laboratory validation.
  </div>
</div>
""", unsafe_allow_html=True)

    render_footer()



# ─────────────────────────────────────────────
#   MAIN
# ─────────────────────────────────────────────
def main():
    # ── Sidebar brand ────────────────────────────
    st.sidebar.markdown("""
<div class="sidebar-brand">
  <div class="brand-icon">🧬</div>
  <div class="brand-name">PathwayNet</div>
  <div class="brand-version">Pharmacogenomics v2.1</div>
</div>
""", unsafe_allow_html=True)

    # ── Page navigation ──────────────────────────
    st.sidebar.markdown('<div class="nav-label">Navigate</div>', unsafe_allow_html=True)
    page = st.sidebar.radio(
        "",
        ["🔬  Analysis", "📖  Learn", "🏛️  For Researchers", "🎓  Credits"],
        label_visibility="collapsed"
    )
    st.sidebar.markdown('<hr style="border-color:#dddad5; margin:18px 0;">', unsafe_allow_html=True)

    if page == "📖  Learn":
        st.markdown("""
<div class="page-online-badge"><span class="pulse-dot"></span>Educational Guide</div>
<div class="page-title">UNDERSTANDING PATHWAYNET</div>
<div class="page-subtitle">A formal, accessible guide for students and educators</div>
""", unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#dddad5; margin-bottom:32px;">', unsafe_allow_html=True)
        render_learn_page()
        return

    if page == "🏛️  For Researchers":
        st.markdown("""
<div class="page-online-badge"><span class="pulse-dot"></span>Technical Documentation</div>
<div class="page-title">RESEARCHER DOCUMENTATION</div>
<div class="page-subtitle">Full technical and methodological reference for educators and reviewers</div>
""", unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#dddad5; margin-bottom:32px;">', unsafe_allow_html=True)
        render_researcher_page()
        return

    if page == "🎓  Credits":
        st.markdown("""
<div class="page-online-badge"><span class="pulse-dot"></span>Team 9</div>
<div class="page-title">PROJECT CREDITS</div>
<div class="page-subtitle">SRM Institute of Science and Technology, Ramapuram &nbsp;|&nbsp; Dept. of Biotechnology</div>
""", unsafe_allow_html=True)
        st.markdown('<hr style="border-color:#dddad5; margin-bottom:32px;">', unsafe_allow_html=True)
        render_credits_page()
        return

    # ── Analysis page ────────────────────────────
    with st.spinner("Initialising data streams…"):
        mut_matrix, dr_df, pathways_or_err, desc_dict = load_data()

    if mut_matrix is None:
        st.error(f"❌ DATA ERROR: {pathways_or_err}")
        return

    gdsc_drugs  = sorted(dr_df['DRUG_NAME'].dropna().unique().tolist())
    extra_only  = [d for d in EXTRA_DESCRIPTIONS if d not in gdsc_drugs]
    valid_drugs = sorted(gdsc_drugs + extra_only)

    st.sidebar.markdown('<div class="nav-label">Target Compound</div>', unsafe_allow_html=True)
    target_drug = st.sidebar.selectbox("", valid_drugs, index=0, label_visibility="collapsed")
    st.sidebar.markdown(f'<div class="drug-badge">⚗ &nbsp;{html_lib.escape(target_drug)}</div>',
                        unsafe_allow_html=True)
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.sidebar.button("⚡  INITIATE ANALYSIS", type="primary")

    # ── Page header ───────────────────────────
    st.markdown("""
<div class="page-online-badge"><span class="pulse-dot"></span>System Online</div>
<div class="page-title">IN-SILICO PATHWAY PERTURBATION</div>
<div class="page-subtitle">GDSC1 · KEGG · MSigDB Hallmark · Bio-Constrained Neural Network</div>
""", unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#dddad5; margin-bottom:28px;">', unsafe_allow_html=True)

    # ── Compound card (always visible) ────────
    render_compound_card(target_drug, desc_dict, dr_df)

    if run_btn:
        perform_analysis(target_drug, mut_matrix, dr_df, pathways_or_err, desc_dict)
        render_footer()
    else:
        st.markdown("""
<div class="await-card">
  <div class="await-title">▶ &nbsp;AWAITING ANALYSIS COMMAND</div>
  <div class="await-body">
    Compound profile loaded above. Click <strong style="color:#2c2c2c;">⚡ INITIATE ANALYSIS</strong>
    in the sidebar to build a sparse biologically-constrained neural network using KEGG &amp;
    MSigDB Hallmark pathway definitions, then run counterfactual CRISPR-like perturbation modelling.
  </div>
  <div class="feature-chips">
    <div class="feature-chip">🔗 Pathway-Constrained Architecture</div>
    <div class="feature-chip">🎲 Randomised Control Benchmark</div>
    <div class="feature-chip">🧪 Counterfactual CRISPR Simulation</div>
    <div class="feature-chip">📊 Cancer-Stratified IC50 Analysis</div>
    <div class="feature-chip">📐 Spearman ρ · R² · MAE · RMSE</div>
  </div>
</div>
""", unsafe_allow_html=True)
        render_footer()


# ─────────────────────────────────────────────
#   ANALYSIS ENGINE
# ─────────────────────────────────────────────
def perform_analysis(target_drug, mut_matrix, dr_df, pathways, desc_dict):
    drug_data = dr_df[dr_df['DRUG_NAME'] == target_drug]
    if drug_data.empty:
        st.warning(f"⚠️ **'{target_drug}'** has no GDSC1 data. Compound profile shown above for reference.")
        return

    y_raw  = drug_data.groupby('SANGER_MODEL_ID')['LN_IC50'].mean()
    common = mut_matrix.index.intersection(y_raw.index)
    if len(common) < 40:
        st.error(f"❌ Insufficient data (N={len(common)}). Minimum 40 samples required.")
        return

    X_all = mut_matrix.loc[common]
    y_all = y_raw.loc[common]

    gene_list = X_all.columns.tolist()
    gene_map  = {g: i for i, g in enumerate(gene_list)}

    bio_masks, active_pws = [], []
    for pw, genes in pathways.items():
        idxs = [gene_map[g] for g in genes if g in gene_map]
        if idxs:
            col = np.zeros(len(gene_list)); col[idxs] = 1
            bio_masks.append(col); active_pws.append(pw)

    if not bio_masks:
        st.error("❌ No pathway overlap found."); return

    mask_bio  = np.array(bio_masks).T
    mask_rand = np.copy(mask_bio)
    for j in range(mask_rand.shape[1]):
        np.random.shuffle(mask_rand[:, j])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all.values, y_all.values, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    y_tr_s = torch.FloatTensor(scaler.fit_transform(y_tr.reshape(-1, 1)))
    X_tr_t = torch.FloatTensor(X_tr)
    X_te_t = torch.FloatTensor(X_te)

    progress = st.progress(0, text="⚙️  Compiling neural architectures…")

    def upd_bio(p):
        progress.progress(int(p * 50), text=f"🔗  Training biological network — {int(p*100)}%")
    def upd_rand(p):
        progress.progress(50 + int(p * 50), text=f"🎲  Training randomised control — {int(p*100)}%")

    model_bio  = train_model(mask_bio,  X_tr_t, y_tr_s, upd_bio)
    model_rand = train_model(mask_rand, X_tr_t, y_tr_s, upd_rand)
    progress.empty()

    model_bio.eval(); model_rand.eval()
    with torch.no_grad():
        p_bio_s, acts_bio = model_bio(X_te_t)
        p_rand_s, _       = model_rand(X_te_t)
        preds_bio  = scaler.inverse_transform(p_bio_s.numpy()).flatten()
        preds_rand = scaler.inverse_transform(p_rand_s.numpy()).flatten()

    pcc_bio,  _ = pearsonr(y_te, preds_bio)
    pcc_rand, _ = pearsonr(y_te, preds_rand)
    scc_bio,  _ = spearmanr(y_te, preds_bio)
    r2_bio      = r2_score(y_te, preds_bio)
    mae_bio     = mean_absolute_error(y_te, preds_bio)
    rmse_bio    = np.sqrt(np.mean((y_te - preds_bio) ** 2))
    validity    = pcc_bio - pcc_rand

    # ════════════════════════════════════════
    #   SECTION 1 — KPIs
    # ════════════════════════════════════════
    section_header("◈", "MODEL PERFORMANCE OVERVIEW")

    cols = st.columns(6, gap="medium")
    kpis = [
        ("Pearson R  (Bio)", f"{pcc_bio:.3f}",
         f"Δ {validity:+.3f} vs control",
         "#4a8c6f" if pcc_bio > 0.3 else "#b07d2a" if pcc_bio > 0.15 else "#c0392b"),
        ("Pearson R  (Control)", f"{pcc_rand:.3f}",
         "Randomised baseline",
         "#888"),
        ("Spearman ρ", f"{scc_bio:.3f}",
         "Rank-order correlation",
         "#4a8c6f" if scc_bio > 0.3 else "#b07d2a"),
        ("R² Score", f"{r2_bio:.3f}",
         "Variance explained",
         "#4a8c6f" if r2_bio > 0.2 else "#b07d2a" if r2_bio > 0.05 else "#c0392b"),
        ("MAE", f"{mae_bio:.3f}",
         "ln IC50 units",
         "#c0392b" if mae_bio > 1.5 else "#b07d2a" if mae_bio > 1.0 else "#4a8c6f"),
        ("RMSE", f"{rmse_bio:.3f}",
         "ln IC50 units",
         "#c0392b" if rmse_bio > 1.8 else "#b07d2a" if rmse_bio > 1.2 else "#4a8c6f"),
    ]
    for col, (label, val, sub, color) in zip(cols, kpis):
        with col:
            st.markdown(f"""
<div class="kpi-outer">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color};">{val}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""", unsafe_allow_html=True)

    # Validity badge
    if validity > 0.05:
        vbc, vbt = "#4a8c6f", "✅ &nbsp;BIOLOGICAL STRUCTURE VALIDATED"
        vbg = "rgba(74,140,111,0.05)"
    elif validity > 0:
        vbc, vbt = "#b07d2a", "⚠️ &nbsp;MARGINAL BIOLOGICAL SIGNAL"
        vbg = "rgba(176,125,42,0.05)"
    else:
        vbc, vbt = "#c0392b", "❌ &nbsp;NO BIOLOGICAL ADVANTAGE DETECTED"
        vbg = "rgba(192,57,43,0.05)"

    st.markdown(f"""
<div class="validity-badge" style="background:{vbg}; border:1px solid {vbc}25;
     border-left:4px solid {vbc}; color:{vbc};">
  {vbt} &nbsp;·&nbsp; Δ Pearson R = {validity:+.4f}
</div>
""", unsafe_allow_html=True)

    # ════════════════════════════════════════
    #   SECTION 2 — STRUCTURE VALIDITY + SCATTER
    # ════════════════════════════════════════
    section_header("◉", "BENCHMARKING &amp; PREDICTION ACCURACY")

    col_l, gap_col, col_r = st.columns([1, 0.06, 1.7])

    with col_l:
        # ── Bar: Bio vs Random ────────────────
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=['Random Control', 'Biological\nPathway Net'],
            y=[pcc_rand, pcc_bio],
            marker=dict(
                color=['#c5d4e8', '#3d5a80'],
                line=dict(color=['#9ab0cc', '#4a8c6f'], width=2),
                opacity=[0.85, 1.0],
            ),
            text=[f"R = {pcc_rand:.3f}", f"R = {pcc_bio:.3f}"],
            textposition='outside',
            textfont=dict(size=13, color='#2c2c2c'),
            width=[0.4, 0.4],
        ))
        fig_bar.add_hline(
            y=0.3, line_dash="dot", line_color="#b07d2a", line_width=1.2,
            annotation_text="R = 0.3", annotation_position="top right",
            annotation_font=dict(color="#b07d2a", size=9, family="JetBrains Mono")
        )
        fig_bar.update_layout(
            **PLOT_CFG,
            title=dict(text="<b>Structure Validity</b>",
                       font=dict(size=12, color='#555')),
            height=380,
            margin=dict(l=20, r=20, t=55, b=20),
            yaxis=dict(range=[0, max(pcc_rand, pcc_bio) * 1.35],
                       gridcolor='#dddad5', title='Pearson R',
                       title_font=dict(size=11)),
            xaxis=dict(showgrid=False),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_r:
        # ── Scatter: Actual vs Predicted ──────
        residuals = preds_bio - y_te
        abs_res   = np.abs(residuals)

        fig_scat = go.Figure()
        fig_scat.add_trace(go.Scatter(
            x=y_te, y=preds_bio, mode='markers',
            marker=dict(
                color=abs_res,
                colorscale=[[0, '#4a8c6f'], [0.4, '#b07d2a'], [1.0, '#c0392b']],
                size=9, opacity=0.82,
                line=dict(width=0.5, color='rgba(0,0,0,0.08)'),
                colorbar=dict(
                    title=dict(text="│Residual│",
                               font=dict(size=9, family='JetBrains Mono')),
                    thickness=11, len=0.7,
                    tickfont=dict(size=8, family='JetBrains Mono'), x=1.02
                ),
                cmin=0, cmax=abs_res.max()
            ),
            hovertemplate="Actual: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
        ))
        lo = min(y_te.min(), preds_bio.min())
        hi = max(y_te.max(), preds_bio.max())
        fig_scat.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode='lines',
            line=dict(color='rgba(0,0,0,0.1)', width=1.5, dash='dot'),
            hoverinfo='skip', showlegend=False
        ))
        m, b   = np.polyfit(y_te, preds_bio, 1)
        x_fit  = np.linspace(y_te.min(), y_te.max(), 120)
        fig_scat.add_trace(go.Scatter(
            x=x_fit, y=m * x_fit + b, mode='lines',
            line=dict(color='#3d5a80', width=2.5),
            name=f"Fit  R={pcc_bio:.3f}",
            hoverinfo='skip'
        ))
        fig_scat.update_layout(
            **PLOT_CFG,
            title=dict(
                text=f"<b>Actual vs Predicted ln IC50</b>"
                     f"  ·  R={pcc_bio:.3f}  ρ={scc_bio:.3f}  R²={r2_bio:.3f}",
                font=dict(size=11, color='#555')
            ),
            height=380,
            margin=dict(l=30, r=70, t=55, b=30),
            xaxis=dict(title='Actual ln IC50',   gridcolor='#dddad5', zeroline=False),
            yaxis=dict(title='Predicted ln IC50', gridcolor='#dddad5', zeroline=False),
            legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)'),
            showlegend=True,
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    # ════════════════════════════════════════
    #   SECTION 3 — CANCER IC50 BOX PLOTS
    # ════════════════════════════════════════
    section_header("◉", "CANCER-TYPE IC50 DISTRIBUTION")

    drug_sub = dr_df[dr_df['DRUG_NAME'] == target_drug].copy()
    drug_sub = drug_sub[drug_sub['TCGA_DESC'].notna() & (drug_sub['TCGA_DESC'] != 'UNCLASSIFIED')]
    drug_sub['CancerFull'] = drug_sub['TCGA_DESC'].map(TCGA_NAMES).fillna(drug_sub['TCGA_DESC'])

    if not drug_sub.empty:
        order = drug_sub.groupby('CancerFull')['LN_IC50'].median().sort_values().index.tolist()
        medians = drug_sub.groupby('CancerFull')['LN_IC50'].median()
        med_lo, med_hi = medians.min(), medians.max()

        def grad_color(cancer):
            v  = (medians[cancer] - med_lo) / (med_hi - med_lo + 1e-9)
            r  = int(52  + v * (239 - 52))
            g  = int(211 + v * (68  - 211))
            b  = int(153 + v * (68  - 153))
            return f'rgb({r},{g},{b})'

        fig_box = go.Figure()
        for cancer in order:
            vals = drug_sub[drug_sub['CancerFull'] == cancer]['LN_IC50'].values
            clr  = grad_color(cancer)
            fig_box.add_trace(go.Box(
                x=vals, name=cancer, orientation='h',
                marker_color=clr,
                line=dict(width=1.5, color=clr),
                fillcolor=clr.replace('rgb(', 'rgba(').replace(')', ',0.15)'),
                boxmean=True,
                hovertemplate=f"<b>{cancer}</b><br>ln IC50: %{{x:.2f}}<br>N = {len(vals)}<extra></extra>"
            ))

        fig_box.update_layout(
            **PLOT_CFG,
            height=max(400, len(order) * 32 + 80),
            margin=dict(l=20, r=30, t=40, b=50),
            xaxis=dict(
                title='ln IC50  ( ← more sensitive  |  more resistant → )',
                gridcolor='#dddad5', zeroline=False,
                title_font=dict(size=11)
            ),
            yaxis=dict(showgrid=False, tickfont=dict(size=10.5)),
            showlegend=False,
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("No cancer-type breakdown available for this compound.")

    # ════════════════════════════════════════
    #   SECTION 4 — RESISTANCE DRIVERS
    # ════════════════════════════════════════
    section_header("◉", "PATHWAY-LEVEL RESISTANCE DRIVERS")

    idx_sorted  = np.argsort(preds_bio)
    res_idx, sen_idx = idx_sorted[-15:], idx_sorted[:15]
    diff    = acts_bio.numpy()[res_idx].mean(0) - acts_bio.numpy()[sen_idx].mean(0)
    top_idx = np.argsort(np.abs(diff))[-18:][::-1]

    top_pws    = [active_pws[i]  for i in top_idx]
    top_scores = diff[top_idx]
    pw_labels  = [
        pw.replace('KEGG_', '').replace('HALLMARK_', '').replace('_', ' ')
        for pw in top_pws
    ]
    bar_clr = ['#c0392b' if v > 0 else '#3d5a80' for v in top_scores]

    fig_resist = go.Figure()
    fig_resist.add_trace(go.Bar(
        x=top_scores[::-1],
        y=pw_labels[::-1],
        orientation='h',
        marker=dict(color=bar_clr[::-1], opacity=0.88,
                    line=dict(color='rgba(0,0,0,0.05)', width=1)),
        text=[f"{v:+.3f}" for v in top_scores[::-1]],
        textposition='outside',
        textfont=dict(size=10, color='#555'),
        hovertemplate="%{y}<br>Δ Activation: %{x:.4f}<extra></extra>"
    ))
    fig_resist.add_vline(x=0, line_width=1.5, line_color='rgba(0,0,0,0.12)')
    fig_resist.update_layout(
        **PLOT_CFG,
        height=600,
        margin=dict(l=20, r=100, t=30, b=60),
        xaxis=dict(
            title='Activation Delta  (Resistant − Sensitive)',
            gridcolor='#dddad5', zeroline=False,
            title_font=dict(size=11)
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=10.5)),
        showlegend=False,
        annotations=[
            dict(x=max(top_scores) * 0.55, y=-0.065, xref='x', yref='paper',
                 text="▶  MORE RESISTANT", showarrow=False,
                 font=dict(color='#c0392b', size=9, family='JetBrains Mono')),
            dict(x=min(top_scores) * 0.55, y=-0.065, xref='x', yref='paper',
                 text="◀  MORE SENSITIVE", showarrow=False,
                 font=dict(color='#3d5a80', size=9, family='JetBrains Mono')),
        ]
    )
    st.plotly_chart(fig_resist, use_container_width=True)

    # ════════════════════════════════════════
    #   SECTION 5 — PERTURBATION
    # ════════════════════════════════════════
    section_header("◉", "COUNTERFACTUAL CRISPR PERTURBATION")

    top_resist_idx = top_idx[np.argmax(top_scores)]
    target_pw      = active_pws[top_resist_idx]
    worst_idx      = np.argmax(preds_bio)
    baseline_pred  = preds_bio[worst_idx]

    pw_genes       = pathways[target_pw]
    gml            = {g: i for i, g in enumerate(X_all.columns.tolist())}
    target_indices = [gml[g] for g in pw_genes if g in gml]

    cf_input = torch.FloatTensor(X_te[worst_idx]).clone()
    cf_input[target_indices] = 0.0
    with torch.no_grad():
        cf_pred_s, _ = model_bio(cf_input.unsqueeze(0))
        cf_pred = scaler.inverse_transform(cf_pred_s.numpy())[0][0]

    delta        = baseline_pred - cf_pred
    c_border     = "#4a8c6f" if delta > 0 else "#c0392b"
    status_icon  = "✅" if delta > 0 else "⚠️"
    status_label = "VALID THERAPEUTIC TARGET" if delta > 0 else "NON-VIABLE — ESCAPE REDUNDANCY"
    pw_display   = target_pw.replace('KEGG_', '').replace('HALLMARK_', '').replace('_', ' ')
    conclusion   = (
        f"In silico knockout of <strong>{len(target_indices)}</strong> mutated genes within "
        f"<strong>{html_lib.escape(pw_display)}</strong> shifts the most resistant cell line toward "
        f"increased sensitivity, validating it as a synergistic co-target for "
        f"<strong>{html_lib.escape(target_drug)}</strong>."
    ) if delta > 0 else (
        f"Perturbation of <strong>{html_lib.escape(pw_display)}</strong> does not reduce predicted "
        f"resistance. Alternative escape pathways likely dominate. Consider combinatorial knockouts "
        f"or upstream regulator targeting."
    )

    p1, p2, p3 = st.columns(3, gap="medium")
    for col, label, val, sub, color in [
        (p1, "Baseline ln IC50",    f"{baseline_pred:.3f}", "Pre-perturbation (most resistant cell)", "#2c2c2c"),
        (p2, "Post-CRISPR ln IC50", f"{cf_pred:.3f}",       "After pathway knockout",                 "#2c2c2c"),
        (p3, "Sensitivity Δ",       f"{delta:+.4f}",        "Positive = sensitised to drug",          c_border),
    ]:
        with col:
            st.markdown(f"""
<div class="kpi-outer" style="{'border:1px solid ' + c_border + '22;' if label == 'Sensitivity Δ' else ''}">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value" style="color:{color};">{val}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="verdict-wrap"
     style="border:1px solid {c_border}20; border-left:4px solid {c_border}; margin-top:20px;">
  <div style="display:flex; align-items:flex-start; gap:16px;">
    <span style="font-size:1.5em; line-height:1; padding-top:2px;">{status_icon}</span>
    <div style="flex:1;">
      <div class="verdict-target-label">Top Resistance Pathway · Knockout Target</div>
      <div class="verdict-target-name" style="color:{c_border};">{html_lib.escape(pw_display)}</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.62em; letter-spacing:3px;
                  text-transform:uppercase; color:#9a9390; margin-bottom:6px;">Verdict</div>
      <div class="verdict-conclusion">
        <strong style="color:{c_border};">{status_label}.</strong><br>
        {conclusion}
      </div>
      <div class="verdict-footer">
        <span>Genes knocked out: <b>{len(target_indices)}</b></span>
        <span>Total pathway members: <b>{len(pw_genes)}</b></span>
        <span>Dataset coverage: <b>{len(target_indices)/max(len(pw_genes),1)*100:.0f}%</b></span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
