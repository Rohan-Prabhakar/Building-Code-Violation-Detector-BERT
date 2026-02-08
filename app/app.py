import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import io
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="ViolationBERT", page_icon="🏗️", layout="wide")

HF_REPO = "Rohan1103/ViolationBERT"

@st.cache_resource
def load_model():
    class ViolationClassifier(nn.Module):
        def __init__(self, model_name, num_categories, num_severities, dropout=0.3):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            h = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.category_head = nn.Sequential(
                nn.Linear(h, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_categories))
            self.severity_head = nn.Sequential(
                nn.Linear(h, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, num_severities))
        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = self.dropout(out.last_hidden_state[:, 0, :])
            return self.category_head(cls), self.severity_head(cls)

    model_path = hf_hub_download(repo_id=HF_REPO, filename="final_model.pt")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    cfg = ckpt['model_config']
    lm = ckpt['label_maps']

    model = ViolationClassifier(cfg['model_name'], cfg['num_categories'], cfg['num_severities'], cfg['dropout'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    id2cat = {int(k): v for k, v in lm['id2cat'].items()}
    id2sev = {int(k): v for k, v in lm['id2sev'].items()}

    return model, tokenizer, id2cat, id2sev

@torch.no_grad()
def predict_single(text, model, tokenizer, id2cat, id2sev):
    enc = tokenizer(text.upper(), max_length=256, padding='max_length', truncation=True, return_tensors='pt')
    c_log, s_log = model(enc['input_ids'], enc['attention_mask'])
    c_probs = torch.softmax(c_log, 1).numpy()[0]
    s_probs = torch.softmax(s_log, 1).numpy()[0]
    return {
        'category': id2cat[c_probs.argmax()],
        'cat_conf': float(c_probs.max()),
        'cat_all': {id2cat[i]: float(p) for i, p in enumerate(c_probs)},
        'severity': id2sev[s_probs.argmax()],
        'sev_conf': float(s_probs.max()),
        'sev_all': {id2sev[i]: float(p) for i, p in enumerate(s_probs)},
    }

@torch.no_grad()
def predict_batch(texts, model, tokenizer, id2cat, id2sev, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        bt = texts[i:i+batch_size]
        enc = tokenizer([t.upper() for t in bt], max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        c_log, s_log = model(enc['input_ids'], enc['attention_mask'])
        c_probs = torch.softmax(c_log, 1).detach().numpy()
        s_probs = torch.softmax(s_log, 1).detach().numpy()
        for j in range(len(bt)):
            results.append({
                'Description': bt[j][:120],
                'Category': id2cat[c_probs[j].argmax()],
                'Cat Confidence': round(float(c_probs[j].max()) * 100, 1),
                'Severity': id2sev[s_probs[j].argmax()],
                'Sev Confidence': round(float(s_probs[j].max()) * 100, 1),
            })
    return results

model, tokenizer, id2cat, id2sev = load_model()

# Sidebar
st.sidebar.title("🏗️ ViolationBERT")
mode = st.sidebar.radio("Mode", ["Single Prediction", "Batch Analysis"])

st.sidebar.markdown("##")
st.sidebar.metric("Category F1", "0.898")
st.sidebar.metric("Severity F1", "0.864")
st.sidebar.metric("Training Data", "189K")

st.sidebar.markdown("##")
with st.sidebar.expander("About"):
    st.markdown("""
    **ViolationBERT** is a fine-tuned RoBERTa-base model trained on 189,000+ NYC DOB violation records.
    
    **8 categories**: Construction, Elevators, Mechanical, Plumbing, Quality of Life, Regulatory, Site Safety, Zoning
    
    **3 severity levels**: HIGH, MEDIUM, LOW
    
    [GitHub](https://github.com/Rohan-Prabhakar/Code-Violation-Detector-BERT-) | [HuggingFace](https://huggingface.co/Rohan1103/ViolationBERT)
    """)


# ============================================================
# SINGLE PREDICTION
# ============================================================
if mode == "Single Prediction":
    st.title("🏗️ ViolationBERT")
    st.subheader("Building Code Violation Classifier")
    st.caption("Fine-tuned RoBERTa classifying NYC building violations by category and severity")

    st.markdown("##")
    col1, col2 = st.columns([1.2, 1])

    with col1:
        text_input = st.text_area(
            "Enter a violation description:",
            height=150,
            placeholder="e.g. FAILURE TO MAINTAIN BUILDING WALL NOTED BRICKS FALLING FROM FACADE"
        )

        examples = [
            "FAILURE TO MAINTAIN BUILDING WALL NOTED BRICKS FALLING FROM FACADE POSING DANGER TO PEDESTRIANS",
            "WORK WITHOUT A PERMIT CONTRACTOR PERFORMING ELECTRICAL WORK ON 3RD FLOOR WITHOUT DOB APPROVAL",
            "ELEVATOR INSPECTION OVERDUE CERTIFICATE EXPIRED LAST YEAR BUILDING HAS 6 PASSENGER ELEVATORS",
            "FENCE EXCEEDS PERMITTED HEIGHT IN FRONT YARD SETBACK AREA ZONING VIOLATION",
            "FAILURE TO PROVIDE SITE SAFETY MANAGER DURING ACTIVE DEMOLITION OF 5 STORY BUILDING",
            "BOILER FAILED ANNUAL INSPECTION DUE TO CRACKED HEAT EXCHANGER AND GAS LEAK DETECTED",
            "ILLEGAL CONVERSION OF COMMERCIAL SPACE TO RESIDENTIAL USE WITHOUT CERTIFICATE OF OCCUPANCY",
            "EXIT DOOR NOT SELF CLOSING ON 2ND FLOOR OF PUBLIC ASSEMBLY SPACE CAPACITY 300 PERSONS",
        ]

        st.markdown("**Try an example:**")
        selected = st.selectbox("Select example", ["(type your own)"] + examples, label_visibility="collapsed")
        if selected != "(type your own)":
            text_input = selected

        predict_btn = st.button("Classify Violation", type="primary", use_container_width=True)

    with col2:
        if predict_btn and text_input.strip():
            result = predict_single(text_input, model, tokenizer, id2cat, id2sev)

            sev_colors = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}

            st.markdown("### Results")
            st.markdown(f"**Category:** `{result['category']}`  ({result['cat_conf']*100:.1f}% confidence)")
            st.markdown(f"**Severity:** {sev_colors.get(result['severity'], '')} `{result['severity']}`  ({result['sev_conf']*100:.1f}% confidence)")

            if result['sev_conf'] < 0.6:
                st.warning("Low confidence on severity. Flag for human review.")
            if result['severity'] == 'HIGH' and result['sev_conf'] > 0.8:
                st.error("HAZARDOUS: This violation requires immediate attention.")

            st.markdown("####")
            st.markdown("**Category Probabilities:**")
            for cat, prob in sorted(result['cat_all'].items(), key=lambda x: x[1], reverse=True):
                st.progress(prob, text=f"{cat}: {prob*100:.1f}%")

            st.markdown("**Severity Probabilities:**")
            for sev, prob in sorted(result['sev_all'].items(), key=lambda x: x[1], reverse=True):
                st.progress(prob, text=f"{sev}: {prob*100:.1f}%")

        elif predict_btn:
            st.warning("Please enter a violation description.")


# ============================================================
# BATCH ANALYSIS
# ============================================================
elif mode == "Batch Analysis":
    st.title("📊 Batch Violation Analysis")
    st.subheader("Upload a file with violation descriptions to classify in bulk")

    st.markdown("##")

    upload_tab, paste_tab = st.tabs(["Upload File", "Paste Text"])

    with upload_tab:
        st.markdown("Upload a **CSV** or **TXT** file containing violation descriptions.")
        st.caption("For CSV: must have a column named 'description', 'violation_description', or 'text'")

        uploaded = st.file_uploader("Choose file", type=['csv', 'txt'])

        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
                text_col = None
                for col in ['description', 'violation_description', 'text', 'Description', 'Violation_Description']:
                    if col in df.columns:
                        text_col = col
                        break
                if text_col is None:
                    text_col = df.columns[0]
                    st.info(f"No standard column found. Using first column: '{text_col}'")
                texts = df[text_col].dropna().astype(str).tolist()
            else:
                content = uploaded.read().decode('utf-8')
                texts = [line.strip() for line in content.split('\n') if line.strip()]

            st.success(f"Loaded {len(texts)} violations")

            if st.button("Classify All", type="primary", use_container_width=True):
                with st.spinner(f"Classifying {len(texts)} violations..."):
                    results = predict_batch(texts, model, tokenizer, id2cat, id2sev)

                results_df = pd.DataFrame(results)

                # Summary metrics
                st.markdown("### Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Violations", len(results_df))
                with col2:
                    high_count = (results_df['Severity'] == 'HIGH').sum()
                    st.metric("HIGH Severity", high_count)
                with col3:
                    avg_conf = results_df['Cat Confidence'].mean()
                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")

                # Category breakdown
                st.markdown("### Category Breakdown")
                cat_counts = results_df['Category'].value_counts()
                st.bar_chart(cat_counts)

                # Severity breakdown
                st.markdown("### Severity Breakdown")
                sev_counts = results_df['Severity'].value_counts()
                st.bar_chart(sev_counts)

                # Full results table
                st.markdown("### Detailed Results")

                def color_severity(val):
                    if val == 'HIGH':
                        return 'background-color: #ffcccc'
                    elif val == 'MEDIUM':
                        return 'background-color: #fff3cd'
                    else:
                        return 'background-color: #d4edda'

                styled = results_df.style.applymap(color_severity, subset=['Severity'])
                st.dataframe(styled, use_container_width=True, height=400)

                # Download button
                csv_out = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results as CSV",
                    csv_out,
                    "violation_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

    with paste_tab:
        st.markdown("Paste multiple violation descriptions, one per line:")
        pasted = st.text_area("Paste violations here:", height=300,
                              placeholder="FAILURE TO MAINTAIN BUILDING WALL...\nWORK WITHOUT A PERMIT...\nELEVATOR INSPECTION OVERDUE...")

        if st.button("Classify Pasted", type="primary", use_container_width=True, key="paste_btn"):
            lines = [l.strip() for l in pasted.split('\n') if l.strip()]
            if lines:
                with st.spinner(f"Classifying {len(lines)} violations..."):
                    results = predict_batch(lines, model, tokenizer, id2cat, id2sev)

                results_df = pd.DataFrame(results)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total", len(results_df))
                with col2:
                    st.metric("HIGH", (results_df['Severity'] == 'HIGH').sum())
                with col3:
                    st.metric("Avg Conf", f"{results_df['Cat Confidence'].mean():.1f}%")

                def color_severity(val):
                    if val == 'HIGH':
                        return 'background-color: #ffcccc'
                    elif val == 'MEDIUM':
                        return 'background-color: #fff3cd'
                    else:
                        return 'background-color: #d4edda'

                styled = results_df.style.applymap(color_severity, subset=['Severity'])
                st.dataframe(styled, use_container_width=True, height=400)

                csv_out = results_df.to_csv(index=False)
                st.download_button("Download Results as CSV", csv_out, "violation_predictions.csv", "text/csv", use_container_width=True)
            else:
                st.warning("Please paste at least one violation description.")