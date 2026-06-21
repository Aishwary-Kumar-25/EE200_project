import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import pickle

# Import backend functions
from q3A_backend import get_spectrogram, extract_peaks_dynamic, generate_paired_hashes, AudioFingerprintDB

# --- Page Configuration ---
# --- Page Configuration ---
st.set_page_config(page_title="EE200 Audio Fingerprinting", layout="wide")

# --- Title Section ---
st.title("EE200: Course Project")
st.subheader("Question 3: Sonic Signatures & Signals to Software")

# Display group details
col1, col2 = st.columns(2)
with col1:
    st.write("**Group Members:**")
    st.write("- Aishwary Kumar (250078)")
    st.write("- Atharv Upadhyay (250235)")
with col2:
    st.write("**Instructor:** Dr. Tushar Sandhan")

# --- Helper: Load Database ---
@st.cache_resource
def load_database(filepath="fingerprint_database.pkl"):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

db = load_database()

if not db:
    st.warning("No database found! Please run your backend script to generate 'fingerprint_database.pkl'.")
    st.stop()

# Calculate statistics for the library
song_stats = {}
for hash_key, entries in db.paired_db.items():
    for song_name, _ in entries:
        song_stats[song_name] = song_stats.get(song_name, 0) + 1

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Library Overview", "Identify Single Clip", "Batch Processing"])

if page == "Library Overview":
    st.title("Library Overview")
    st.write("View the contents of your currently indexed audio database.")
    
    st.metric(label="Total Unique Hashes", value=f"{len(db.paired_db):,}")
    
    st.subheader("Hashes per Track")
    if song_stats:
        chart_data = pd.DataFrame(list(song_stats.items()), columns=['Song', 'Hashes']).set_index('Song')
        st.bar_chart(chart_data)
    
    with st.expander("View Raw Database Details"):
        st.dataframe(chart_data)

elif page == "Identify Single Clip":
    st.title("Identify Audio Clip")
    
    uploaded_file = st.file_uploader("Upload an audio segment (MP3/WAV)", type=['mp3', 'wav'])
    
    if uploaded_file:
        st.audio(uploaded_file)
        
        with st.spinner("Processing audio and searching database..."):
            audio, sr = librosa.load(uploaded_file, sr=44100, mono=True)
            
            # 1. Match
            song, votes = db.identify(audio, fs=sr, mode='paired')
            
            # 2. Extract features for visuals
            freqs, times, spec_db = get_spectrogram(audio, fs=sr, nperseg=2048)
            peaks = extract_peaks_dynamic(spec_db)
            peak_times = [times[t] for f, t in peaks]
            peak_freqs = [freqs[f] for f, t in peaks]
            
            # 3. Calculate offsets to find exactly where it matched
            query_hashes = generate_paired_hashes(peaks)
            offsets = []
            for hash_key, q_t_anchors in query_hashes.items():
                if hash_key in db.paired_db:
                    for db_song_name, db_t_anchor in db.paired_db[hash_key]:
                        if db_song_name == song:
                            for q_t_anchor in q_t_anchors:
                                offsets.append(db_t_anchor - q_t_anchor)
                                
            peak_offset = 0
            if offsets:
                min_off, max_off = min(offsets), max(offsets)
                counts, bin_edges = np.histogram(offsets, bins=np.arange(min_off - 0.5, max_off + 1.5, 1))
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                max_idx = np.argmax(counts)
                peak_offset = bin_centers[max_idx]
            
        st.success(f"### Prediction: {song}")
        st.info(f"Confidence: {votes} aligned hashes")

        # --- Visualizations ---
        with st.expander("1. Feature Extraction (Spectrogram & Peaks)", expanded=True):
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            mesh = ax1.pcolormesh(times, freqs, spec_db, shading='gouraud', cmap='viridis')
            ax1.scatter(peak_times, peak_freqs, color='red', s=15, alpha=0.7, label='Constellation Peaks')
            
            ax1.set_xlim(0, min(30.0, times[-1])) 
            ax1.set_ylim(0, 4000) 
            ax1.set_xlabel("Time (Seconds)")
            ax1.set_ylabel("Frequency (Hz)")
            fig1.colorbar(mesh, ax=ax1, label="Magnitude (dB)")
            ax1.legend(loc="upper right")
            
            st.pyplot(fig1)
            plt.close(fig1)

        with st.expander("2. Database Search (Where in the song?)", expanded=True):
            if song != "Unknown" and offsets:
                # Extract full constellation for the matched song directly from db
                matched_song_t = []
                matched_song_f = []
                for f_idx, entries in db.single_db.items():
                    for db_song, t_idx in entries:
                        if db_song == song:
                            matched_song_t.append(t_idx)
                            matched_song_f.append(f_idx)
                
                fig_search, ax_search = plt.subplots(figsize=(10, 3))
                # Plot the full track's peaks
                ax_search.scatter(matched_song_t, matched_song_f, color='teal', s=3, alpha=0.5)
                
                # Highlight the matched window
                query_duration_frames = len(times)
                ax_search.axvspan(peak_offset, peak_offset + query_duration_frames, 
                                  color='orange', alpha=0.4, label='Query Match Location')
                
                ax_search.set_xlabel("Time (Frames)")
                ax_search.set_ylabel("Frequency Bin")
                ax_search.legend(loc="upper right")
                
                st.pyplot(fig_search)
                plt.close(fig_search)
            else:
                st.warning("Could not locate song constellation in database.")

        with st.expander("3. The Proof (Alignment Spike)", expanded=True):
            if offsets:
                fig3, ax3 = plt.subplots(figsize=(10, 3))
                ax3.plot(bin_centers, counts, color='steelblue', linewidth=1.5)
                
                # Highlight the spike
                ax3.axvline(peak_offset, color='red', linestyle='--', label=f'Spike at {counts[max_idx]} hashes')
                
                ax3.set_xlabel("Time Offset (Database frame - Query frame)")
                ax3.set_ylabel("Number of Aligned Hashes")
                ax3.legend()
                
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.warning("Not enough matches to generate an alignment spike.")

elif page == "Batch Processing":
    st.title("Batch Identification")
    st.write("Upload multiple clips to identify them all at once.")
    
    uploaded_files = st.file_uploader("Upload multiple audio files", accept_multiple_files=True, type=['mp3', 'wav'])
    
    if uploaded_files and st.button("Process All"):
        results = []
        progress_bar = st.progress(0)
        
        for i, file in enumerate(uploaded_files):
            audio, sr = librosa.load(file, sr=44100, mono=True)
            song, votes = db.identify(audio, fs=sr)
            
            # Assignment specific format: must output 'none' if no match is found
            pred_label = "none" if song == "Unknown" else song
            
            results.append({
                "filename": file.name,
                "prediction": pred_label
            })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        df = pd.DataFrame(results)
        st.success("Batch processing complete!")
        st.dataframe(df, use_container_width=True)
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download results.csv",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )