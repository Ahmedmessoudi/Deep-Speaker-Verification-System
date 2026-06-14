import os

import streamlit as st
import requests
import plotly.graph_objects as go

st.set_page_config(page_title="Voice Biometrics UI", layout="wide")

st.title("Voice Biometrics - Demo UI")

api_url = os.getenv("API_URL", "http://localhost:8000")

# Simple sidebar menu
page = st.sidebar.selectbox("Menu", ("Compare & Verify", "Augment & Test"))

st.sidebar.markdown("---")
st.sidebar.markdown("### Settings")
threshold = st.sidebar.slider("Confidence Threshold (Taux de confiance)", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

if "aug_results" not in st.session_state:
    st.session_state["aug_results"] = []

if page == "Compare & Verify":
    uploaded_1 = st.file_uploader("Upload first audio (reference)", type=["wav", "mp3", "flac"], key="f1")
    uploaded_2 = st.file_uploader("Upload second audio (test)", type=["wav", "mp3", "flac"], key="f2")

    # Choice of output format for model comparison: default is Tables
    output_choice = st.radio("Comparison output format:", ("Tables", "Plots"), index=0, key="comp_format")

    if st.button("Verify"):
        if not uploaded_1 or not uploaded_2:
            st.warning("Please upload two audio files")
        else:
            files = {
                'file1': (uploaded_1.name, uploaded_1.getvalue()),
                'file2': (uploaded_2.name, uploaded_2.getvalue())
            }

            with st.spinner("Calling verification API..."):
                try:
                    r = requests.post(f"{api_url}/verify", files=files, params={"threshold": threshold})
                    if r.status_code == 200:
                        res = r.json()
                        msg = f"Similarity: {res['similarity_score']:.4f} — Same speaker: {res['is_same_speaker']} (Threshold: {threshold})"
                        if res['is_same_speaker']:
                            st.success(msg)
                        else:
                            st.error(msg)
                    else:
                        st.error(f"API error: {r.status_code} {r.text}")
                except Exception as e:
                    st.error(f"Request failed: {e}")

    if uploaded_1 and uploaded_2 and st.button("Compare with Noise (ECAPA-TDNN)"):
        try:
            files = {
                'file1': (uploaded_1.name, uploaded_1.getvalue()),
                'file2': (uploaded_2.name, uploaded_2.getvalue())
            }

            with st.spinner("Generating comparison..."):
                r = requests.post(f"{api_url}/compare-models", files=files)

            if r.status_code != 200:
                st.error(f"API error: {r.status_code} {r.text}")
            else:
                result = r.json()

                models = result["models"]
                ecapa = models["ecapa_tdnn"]

                if output_choice == "Plots":
                    # Plot ECAPA-TDNN Similarity vs SNR
                    snr_fig = go.Figure()
                    snr_fig.add_trace(go.Scatter(
                        x=[item["snr_db"] for item in ecapa["snr_curve"]],
                        y=[item["similarity"] for item in ecapa["snr_curve"]],
                        mode="lines+markers",
                        name="ECAPA-TDNN",
                        line=dict(color="#636EFA", width=2),
                        marker=dict(size=8),
                    ))
                    snr_fig.update_layout(
                        title="ECAPA-TDNN Similarity vs SNR (dB)",
                        xaxis_title="SNR (dB)",
                        yaxis_title="Cosine Similarity",
                        template="plotly_white",
                        height=450,
                    )
                    st.plotly_chart(snr_fig, use_container_width=True)
                else:
                    # Tables view (default)
                    st.subheader("Original Similarity (ECAPA-TDNN)")
                    st.table([
                        {
                            "Model": "ECAPA-TDNN", 
                            "Original Similarity": f"{ecapa['original_similarity']:.4f}",
                            "Same Speaker (Match)": ecapa['original_similarity'] >= threshold
                        },
                    ])

                    # Build ECAPA-only SNR table
                    snr_rows = []
                    for item in ecapa["snr_curve"]:
                        snr_db = item["snr_db"]
                        ecapa_sim = item["similarity"]
                        snr_rows.append({
                            "SNR (dB)": snr_db,
                            "ECAPA-TDNN Similarity": f"{ecapa_sim:.4f}",
                            "Match": ecapa_sim >= threshold,
                        })

                    st.subheader("ECAPA-TDNN Similarity vs SNR (table)")
                    st.table(snr_rows)
        except Exception as e:
            st.error(f"Comparison request failed: {e}")

elif page == "Augment & Test":
    st.header("Augment & Test")
    ref = st.file_uploader("Upload your voice (reference)", type=["wav", "mp3", "flac"], key="ref_voice")
    if not ref:
        st.info("Upload a reference audio to start")
    else:
        # Fetch augmentation categories and samples
        try:
            r = requests.get(f"{api_url}/augmentation-categories")
            if r.status_code != 200:
                st.error(f"Could not list augmentation categories: {r.status_code}")
            else:
                meta = r.json()
                cats = meta.get('categories', [])
                samples = meta.get('samples', {})

                # Prefer 'noise' category if available
                default_cat = 'noise' if 'noise' in cats else (cats[0] if cats else None)
                category = st.selectbox("Category", options=cats, index=cats.index(default_cat) if default_cat in cats else 0) if cats else None

                # Show available sample noise files for category
                available = samples.get(category, []) if category else []
                if not available:
                    st.warning("No sample noise files found for selected category")
                else:
                    # Let user pick up to 5 files (default first 5)
                    default_selection = available[:5]
                    chosen = st.multiselect("Select up to 5 noise files to apply", options=available, default=default_selection, max_selections=5)

                    snr_db = st.slider("SNR (dB) for augmentation", min_value=0, max_value=30, value=10)

                    if st.button("Generate augmentations"):
                        aug_results = []
                        for nf in chosen:
                            try:
                                files = {'file': (ref.name, ref.getvalue())}
                                data = {'noise_file': nf, 'snr_db': str(snr_db)}
                                with st.spinner(f"Augmenting with {nf} ..."):
                                    rr = requests.post(f"{api_url}/augment-with-file", files=files, data=data)

                                if rr.status_code != 200:
                                    st.warning(f"Augmentation failed for {nf}: {rr.status_code}")
                                    continue

                                audio_bytes = rr.content
                                aug_results.append((nf, audio_bytes))
                            except Exception as e:
                                st.warning(f"Error augmenting {nf}: {e}")

                        st.session_state["aug_results"] = aug_results

                    # Display the last generated augmentations with test buttons
                    if st.session_state["aug_results"]:
                        st.divider()
                        st.subheader("Generated augmentations")
                        for idx, (nf, audio_bytes) in enumerate(st.session_state["aug_results"]):
                            st.subheader(f"Augmentation {idx+1}: {nf}")
                            st.audio(audio_bytes, format='audio/wav')
                            button_key = f"test_{idx}_{nf.replace('/', '_').replace(' ', '_')}"
                            if st.button(f"Test against original #{idx+1}", key=button_key):
                                try:
                                    files = {
                                        'file1': (ref.name, ref.getvalue()),
                                        'file2': (f"aug_{nf}", audio_bytes)
                                    }
                                    with st.spinner("Calling ECAPA-TDNN verify API..."):
                                        r_ecapa = requests.post(f"{api_url}/verify", files=files, params={"threshold": threshold, "model_type": "ecapa_tdnn"})
                                    
                                    st.markdown("##### Verification Result (ECAPA-TDNN):")
                                    
                                    if r_ecapa.status_code == 200:
                                        j_ecapa = r_ecapa.json()
                                        msg = f"**ECAPA-TDNN**: Similarity: {j_ecapa['similarity_score']:.4f} — Same speaker: {j_ecapa['is_same_speaker']} (Threshold: {threshold})"
                                        if j_ecapa['is_same_speaker']:
                                            st.success(msg)
                                        else:
                                            st.error(msg)
                                    else:
                                        st.error(f"ECAPA-TDNN verify API error: {r_ecapa.status_code}")
                                        
                                except Exception as e:
                                    st.error(f"Verification failed: {e}")

        except Exception as e:
            st.error(f"Could not contact API: {e}")

st.markdown("---")
st.header("Quick embedding extraction")
upload_e = st.file_uploader("Upload audio to extract embedding", type=["wav", "mp3", "flac"], key="embed")
if upload_e and st.button("Get Embedding"):
    try:
        files = {'file': (upload_e.name, upload_e.getvalue())}
        r = requests.post(f"{api_url}/embed", files=files)
        if r.status_code == 200:
            res = r.json()
            st.write("Speaker ID:", res.get('speaker_id'))
            st.write("Embedding length:", len(res.get('embedding', [])))
        else:
            st.error(f"API error: {r.status_code} {r.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
