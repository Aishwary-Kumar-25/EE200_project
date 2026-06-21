import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
import librosa
import pickle

def get_spectrogram(audio_signal, fs=44100, nperseg=2048):
    freqs, times, spec = spectrogram(audio_signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    spec_db = 20 * np.log10(np.abs(spec) + 1e-10)
    return freqs, times, spec_db

def extract_peaks_dynamic(spec_db, neighborhood_size=25, top_percentage=0.015):
    threshold = np.quantile(spec_db, 1.0 - top_percentage)
    local_max = (maximum_filter(spec_db, size=neighborhood_size) == spec_db)
    background = (spec_db >= threshold)
    peaks_mask = local_max & background
    f_indices, t_indices = np.where(peaks_mask)
    return list(zip(f_indices, t_indices))

def generate_paired_hashes(peaks, fan_out=5, max_time_gap=20):
    hashes = {}
    peaks = sorted(peaks, key=lambda x: x[1])
    num_peaks = len(peaks)
    
    for i in range(num_peaks):
        for j in range(i + 1, min(i + 1 + fan_out, num_peaks)):
            f_anchor, t_anchor = peaks[i]
            f_target, t_target = peaks[j]
            dt = t_target - t_anchor
            if 0 < dt <= max_time_gap:
                hash_key = (f_anchor, f_target, dt)
                hashes.setdefault(hash_key, []).append(t_anchor)
    return hashes

class AudioFingerprintDB:
    def __init__(self):
        self.paired_db = {}
        self.single_db = {}
        
    def index_song(self, song_name, audio_signal, fs=44100):
        _, _, spec_db = get_spectrogram(audio_signal, fs, nperseg=2048)
        peaks = extract_peaks_dynamic(spec_db)
        
        paired_hashes = generate_paired_hashes(peaks)
        for hash_key, t_anchors in paired_hashes.items():
            for t_idx in t_anchors:
                self.paired_db.setdefault(hash_key, []).append((song_name, t_idx))
                
        for f_idx, t_idx in peaks:
            self.single_db.setdefault(f_idx, []).append((song_name, t_idx))

    def identify(self, query_signal, fs=44100, mode='paired'):
        _, _, spec_db = get_spectrogram(query_signal, fs, nperseg=2048)
        query_peaks = extract_peaks_dynamic(spec_db)
        matches = {}
        
        if mode == 'paired':
            query_hashes = generate_paired_hashes(query_peaks)
            for hash_key, q_t_anchors in query_hashes.items():
                if hash_key in self.paired_db:
                    for song_name, db_t_anchor in self.paired_db[hash_key]:
                        for q_t_anchor in q_t_anchors:
                            matches.setdefault(song_name, []).append(db_t_anchor - q_t_anchor)
        else:
            for q_f, q_t in query_peaks:
                if q_f in self.single_db:
                    for song_name, db_t in self.single_db[q_f]:
                        matches.setdefault(song_name, []).append(db_t - q_t)
                        
        best_song, max_votes = "Unknown", 0
        for song_name, offsets in matches.items():
            if len(offsets) == 0: continue
            counts, _ = np.histogram(offsets, bins=np.arange(min(offsets)-0.5, max(offsets)+1.5, 1))
            highest_vote = np.max(counts)
            if highest_vote > max_votes:
                max_votes = highest_vote
                best_song = song_name
        return best_song, max_votes


if __name__ == "__main__":
    library_path = "Songs"
    db_filepath = "fingerprint_database.pkl"
    sampling_rate = 44100
    
    # 1. Defined mp3_files first, so the tests at the bottom can always find a song.
    mp3_files = glob.glob(os.path.join(library_path, "*.mp3"))
    
    # 2. Database Loading/Building
    if os.path.exists(db_filepath):
        print(f"Loading existing database from {db_filepath}...")
        with open(db_filepath, 'rb') as f:
            fingerprint_system = pickle.load(f)
        print(f"Success! Loaded {len(fingerprint_system.paired_db)} paired hashes.")
        
    else:
        print("No existing database found. Building Audio Structural Database from MP3 Files...")
        fingerprint_system = AudioFingerprintDB()
        
        if len(mp3_files) == 0:
            print(f"Warning: No .mp3 files found in '{library_path}'.")
        else:
            for idx, filepath in enumerate(mp3_files, start=1):
                song_label = os.path.splitext(os.path.basename(filepath))[0]
                raw_audio, fs = librosa.load(filepath, sr=sampling_rate, mono=True)
                fingerprint_system.index_song(song_label, raw_audio, fs=sampling_rate)
                print(f"[{idx}/{len(mp3_files)}] Fingerprinted: {song_label}")
            
            with open(db_filepath, 'wb') as f:
                pickle.dump(fingerprint_system, f)
            print(f"\nDatabase fully indexed and saved to {db_filepath}!")

    if len(mp3_files) > 0:
        print("\n Running Mandatory Diagnostic Experiments")
        
        test_song_path = mp3_files[0]
        print(test_song_path)
        song_true_label = os.path.splitext(os.path.basename(test_song_path))[0]
        sample_signal, _ = librosa.load(test_song_path, sr=sampling_rate, mono=True)
        
        # 1. Window Length Experiments
        f_short, t_short, spec_short = get_spectrogram(sample_signal, nperseg=256)
        f_long, t_long, spec_long = get_spectrogram(sample_signal, nperseg=8192)

        plt.figure(figsize=(14, 6))
        plt.suptitle(f" Song: {song_true_label}\nWindow Length Experiments", fontsize=16)
        
        plt.subplot(1, 2, 1)
        mesh_short = plt.pcolormesh(t_short[:100], f_short, spec_short[:, :100], shading='gouraud', cmap='magma')
        plt.title("Short Window (nperseg=256)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(mesh_short, label="Magnitude (dB)")
        
        plt.subplot(1, 2, 2)
        mesh_long = plt.pcolormesh(t_long[:10], f_long, spec_long[:, :10], shading='gouraud', cmap='magma')
        plt.title("Long Window (nperseg=8192)")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(mesh_long, label="Magnitude (dB)")

        # 2. Degradation Tests
        query_clip = sample_signal[0 : 5 * sampling_rate]
        noise = np.random.normal(0, 0.02, query_clip.shape)
        noisy_query = query_clip + noise
        
        match_song, match_votes = fingerprint_system.identify(noisy_query, fs=sampling_rate, mode='paired')
        print(f"[Noise Test] Paired Hash resolved: '{match_song}' with {match_votes} votes.")

        match_song_s, match_votes_s = fingerprint_system.identify(noisy_query, fs=sampling_rate, mode='single')
        print(f"[Single Peak Test] Solo peak resolved: '{match_song_s}' with {match_votes_s} entries.")

        stretched_query = np.interp(np.arange(0, len(query_clip), 1.05), np.arange(len(query_clip)), query_clip)
        match_song_stretch, match_votes_stretch = fingerprint_system.identify(stretched_query, fs=sampling_rate, mode='paired')
        print(f"[Pitch Shift Test] Resolved: '{match_song_stretch}' with {match_votes_stretch} votes.")

        start_sec = 0
        end_sec = 30
        slice_signal = sample_signal[int(start_sec * sampling_rate) : int(end_sec * sampling_rate)]
        freqs, times, spec_db = get_spectrogram(slice_signal, fs=sampling_rate)
        peaks = extract_peaks_dynamic(spec_db, neighborhood_size=15, top_percentage=0.01)
        
        peak_times = [times[t_idx] for f_idx, t_idx in peaks]
        peak_freqs = [freqs[f_idx] for f_idx, t_idx in peaks]

        plt.figure(figsize=(11, 6))
        mesh_const = plt.pcolormesh(times, freqs, spec_db, shading='gouraud', cmap='magma')
        plt.colorbar(mesh_const, label="Magnitude (dB)")
        plt.scatter(peak_times, peak_freqs, edgecolors='green', facecolors='none', s=80, linewidths=2.0)
        plt.ylim(0, 4000)
        
        
        plt.title(f"Song: {song_true_label}\nSpectrogram Slice with Peak Identifiers")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.show()