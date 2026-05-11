import streamlit as st
import json, os
import pandas as pd
import numpy as np
from collections import Counter

RESULTS_FILE = '/content/drive/MyDrive/atsr_results.json'

st.set_page_config(page_title='ATSR Dashboard', layout='wide', page_icon='🧠')

@st.cache_data(ttl=30)
def load_results():
    if not os.path.exists(RESULTS_FILE):
        return {}
    with open(RESULTS_FILE) as f:
        return json.load(f)

results = load_results()

PAGE = st.sidebar.radio('Page', [
    '📊 Overview',
    '🤖 Per-Model Deep Dive',
    '🔍 Per-Sample Inspector',
    '🧪 Ablation: Gate Conditions',
])

# ────────────────────────────────────────────────────────────────────────────
if PAGE == '📊 Overview':
    st.title('🧠 ATSR Hallucination Detection — Benchmark Overview')
    if not results:
        st.warning('No results found. Run the benchmark notebook first.')
        st.stop()
    rows = []
    for model, ds_dict in results.items():
        for ds, r in ds_dict.items():
            rows.append({
                'Model':      model,
                'Dataset':    ds,
                'F1 Before':  round(r['f1_before'], 4),
                'F1 After':   round(r['f1_after'],  4),
                'Δ F1':       round(r['improvement'], 4),
                'Trigger%':   round(r['trigger_rate'] * 100, 1),
                'Accept%':    round(r['accept_rate']  * 100, 1),
                'AUROC':      round(r['auroc'], 3)       if r.get('auroc')       else None,
                'TPR@5FPR':   round(r['tpr_at_5fpr'], 3) if r.get('tpr_at_5fpr') else None,
                'Runtime(m)': round(r['runtime_min'], 1),
            })
    df = pd.DataFrame(rows)

    def _color_delta(v):
        if v > 0:  return 'background-color:#1a4a2e'
        if v < 0:  return 'background-color:#4a1a1a'
        return ''

    st.dataframe(
        df.style.applymap(_color_delta, subset=['Δ F1']),
        use_container_width=True
    )
    st.subheader('Mean Δ F1 by model (across all datasets)')
    avg = df.groupby('Model')['Δ F1'].mean().reset_index()
    st.bar_chart(avg.set_index('Model'))
    st.subheader('Trigger rate vs Accept rate')
    ta = df[['Model', 'Dataset', 'Trigger%', 'Accept%']].set_index(['Model', 'Dataset'])
    st.bar_chart(ta)

# ────────────────────────────────────────────────────────────────────────────
elif PAGE == '🤖 Per-Model Deep Dive':
    st.title('🤖 Per-Model Deep Dive')
    if not results:
        st.warning('No results yet.'); st.stop()
    model   = st.selectbox('Select model', list(results.keys()))
    ds_dict = results[model]

    cols = st.columns(min(len(ds_dict), 5))
    for i, (ds, r) in enumerate(ds_dict.items()):
        with cols[i % len(cols)]:
            delta = r['improvement']
            icon  = '🟢' if delta > 0 else ('🔴' if delta < 0 else '⚪')
            st.metric(f'{icon} {ds}', f"{r['f1_after']:.4f}", delta=f"{delta:+.4f}")
            st.caption(f"Trigger {r['trigger_rate']:.1%} | Accept {r['accept_rate']:.1%}")
            if r.get('auroc'):
                st.caption(f"AUROC {r['auroc']:.3f} | TPR@5FPR {r.get('tpr_at_5fpr','N/A')}")

    st.subheader('F1 Before vs After per dataset')
    df2 = pd.DataFrame([
        {'Dataset': ds, 'Before': r['f1_before'], 'After': r['f1_after']}
        for ds, r in ds_dict.items()
    ])
    st.bar_chart(df2.set_index('Dataset'))

    st.subheader('Gate accept rate per dataset')
    df3 = pd.DataFrame([
        {'Dataset': ds, 'Trigger%': r['trigger_rate']*100, 'Accept%': r['accept_rate']*100}
        for ds, r in ds_dict.items()
    ])
    st.bar_chart(df3.set_index('Dataset'))

# ────────────────────────────────────────────────────────────────────────────
elif PAGE == '🔍 Per-Sample Inspector':
    st.title('🔍 Per-Sample Inspector')
    if not results:
        st.warning('No results yet.'); st.stop()
    model   = st.selectbox('Model',   list(results.keys()))
    dataset = st.selectbox('Dataset', list(results[model].keys()))
    samples = results[model][dataset].get('per_sample', [])
    valid   = [s for s in samples if 'error' not in s]
    if not valid:
        st.warning('No valid samples in this dataset.'); st.stop()

    sort_by = st.radio('Sort by', [
        'Improvement ↓', 'Regression ↓', 'Triggered only', 'Accepted only'
    ], horizontal=True)
    if sort_by == 'Improvement ↓':
        valid = sorted(valid, key=lambda r: r['f1_final'] - r['f1_baseline'], reverse=True)
    elif sort_by == 'Regression ↓':
        valid = sorted(valid, key=lambda r: r['f1_final'] - r['f1_baseline'])
    elif sort_by == 'Triggered only':
        valid = [r for r in valid if r.get('triggered')]
    elif sort_by == 'Accepted only':
        valid = [r for r in valid if r.get('accepted')]

    if not valid:
        st.warning('No samples match that filter.'); st.stop()

    idx = st.slider('Sample', 0, len(valid)-1, 0)
    r   = valid[idx]

    st.markdown(f"**Question:** {r['question']}")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('**Baseline answer**')
        st.info(r.get('baseline', ''))
        st.metric('F1 baseline', f"{r['f1_baseline']:.4f}")
    with c2:
        lbl = '**Final (gate ACCEPTED correction)**' if r.get('accepted') \
              else '**Final = Baseline (gate rejected)**'
        st.markdown(lbl)
        if r.get('accepted'):
            st.success(r.get('final', ''))
        else:
            st.warning(r.get('final', r.get('baseline', '')))
        delta = r['f1_final'] - r['f1_baseline']
        st.metric('F1 final', f"{r['f1_final']:.4f}", delta=f"{delta:+.4f}")

    st.caption(
        f"Triggered={r.get('triggered')} | Accepted={r.get('accepted')} "
        f"| Gate: {r.get('gate_reason','')} | Signals fired: {r.get('n_fired','-')}/3"
    )
    if r.get('signals'):
        with st.expander('Detection signals'):
            st.json(r['signals'])
    if r.get('flagged'):
        st.caption(f"Flagged tokens: {r['flagged']}")

# ────────────────────────────────────────────────────────────────────────────
elif PAGE == '🧪 Ablation: Gate Conditions':
    st.title('🧪 Ablation: Gate Condition Analysis')
    if not results:
        st.warning('No results yet.'); st.stop()
    model   = st.selectbox('Model',   list(results.keys()))
    dataset = st.selectbox('Dataset', list(results[model].keys()))
    samples = [
        s for s in results[model][dataset].get('per_sample', [])
        if 'error' not in s and s.get('triggered')
    ]
    if not samples:
        st.warning('No triggered samples for this model/dataset.'); st.stop()

    reasons = [s.get('gate_reason', 'unknown') for s in samples]
    counts  = Counter(reasons)
    df_gate = pd.DataFrame(counts.items(), columns=['Gate Reason', 'Count'])
    df_gate = df_gate.sort_values('Count', ascending=False)

    st.subheader('Correction disposition (why accepted or rejected)')
    st.bar_chart(df_gate.set_index('Gate Reason'))
    st.dataframe(df_gate, use_container_width=True)

    accepted = [s for s in samples if s.get('accepted')]
    rejected = [s for s in samples if not s.get('accepted')]

    col1, col2, col3 = st.columns(3)
    col1.metric('Triggered', len(samples))
    col2.metric('Accepted',  len(accepted))
    col3.metric('Rejected',  len(rejected))

    st.subheader('F1 impact: accepted vs gate-rejected')
    c1, c2 = st.columns(2)
    if accepted:
        acc_d = [s['f1_final'] - s['f1_baseline'] for s in accepted]
        c1.metric('Accepted — mean Δ F1',
                  f'{np.mean(acc_d):+.4f}',
                  delta=f'{len(accepted)} corrections applied')
        c1.caption('Positive = gate correctly accepted improvements')
    if rejected:
        rej_d = [s.get('f1_corrected', s['f1_baseline']) - s['f1_baseline']
                 for s in rejected if 'f1_corrected' in s]
        if rej_d:
            c2.metric('Rejected — mean Δ F1 *if* applied',
                      f'{np.mean(rej_d):+.4f}',
                      delta=f'{len(rejected)} corrections blocked')
            c2.caption('Negative = gate correctly blocked harmful corrections')

    st.subheader('Signal strength histogram (triggered samples)')
    entropies = [s['signals']['entropy'] for s in samples if s.get('signals')]
    if entropies:
        hist_df = pd.DataFrame({'Entropy': entropies})
        st.bar_chart(hist_df['Entropy'].value_counts(bins=20).sort_index())
