import numpy as np
import librosa
import mido
from tkinter import Tk, Label, Button, filedialog, StringVar
import tkinter as tk
from tkinter import ttk
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class DrumBeatExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("Drum Beat Extractor for Maschine")
        self.root.geometry("800x700")
        # New color scheme
        self.bg_color = "#3C3F41"
        self.widget_bg = "#4B4F52"
        self.text_color = "#E8E8E8"
        self.accent_color = "#4A88C7"
        self.button_color = "#5E6063"
        self.plot_bg = "#313335"
        self.waveform_color = "#7FB0DF"
        self.kick_color = "#FF6B6B"
        self.snare_color = "#4ECDC4"
        self.hihat_color = "#FFE66D"

        self.root.configure(bg=self.bg_color)
        self.is_analyzing = False
        
        self.setup_ui()
        
        # Analysis parameters
        self.audio_file = None
        self.y = None
        self.sr = None
        self.tempo = 120
        self.onset_frames = None
        self.drum_types = None
        
    def setup_ui(self):
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam') # Use a theme that allows easier color config
        self.style.configure("TButton", font=("Arial", 12), background=self.button_color, foreground=self.text_color)
        self.style.map("TButton", background=[('active', self.accent_color)])
        self.style.configure("TLabel", font=("Arial", 12), background=self.bg_color, foreground=self.text_color)
        self.style.configure("Header.TLabel", font=("Arial", 16, "bold"), background=self.bg_color, foreground=self.text_color)
        self.style.configure("TEntry", fieldbackground=self.widget_bg, foreground=self.text_color, insertcolor=self.text_color)
        self.style.configure("TScale", background=self.widget_bg)
        
        # Title
        title_label = ttk.Label(self.root, text="Drum Beat Extractor for Maschine", style="Header.TLabel")
        title_label.pack(pady=20)
        
        # File selection
        file_frame = tk.Frame(self.root, bg=self.bg_color)
        file_frame.pack(fill="x", padx=20, pady=10)
        
        self.file_path = StringVar()
        self.file_path.set("No file selected")
        
        file_label = ttk.Label(file_frame, text="Audio File:", style="TLabel")
        file_label.pack(side="left", padx=5)
        
        path_label = ttk.Label(file_frame, textvariable=self.file_path, style="TLabel")
        path_label.pack(side="left", padx=5, fill="x", expand=True)
        
        self.select_btn = ttk.Button(file_frame, text="Select File", command=self.select_file)
        self.select_btn.pack(side="right", padx=5)
        
        # Tempo selection
        tempo_frame = tk.Frame(self.root, bg=self.bg_color)
        tempo_frame.pack(fill="x", padx=20, pady=10)
        
        tempo_label = ttk.Label(tempo_frame, text="Tempo (BPM):", style="TLabel")
        tempo_label.pack(side="left", padx=5)
        
        self.tempo_var = tk.StringVar(value="120")
        self.tempo_entry = ttk.Entry(tempo_frame, textvariable=self.tempo_var, width=6)
        self.tempo_entry.pack(side="left", padx=5)
        
        self.detect_tempo_btn = ttk.Button(tempo_frame, text="Detect Tempo", command=self.start_tempo_detection_thread)
        self.detect_tempo_btn.pack(side="left", padx=5)
        
        # Sensitivity slider
        sensitivity_frame = tk.Frame(self.root, bg=self.bg_color)
        sensitivity_frame.pack(fill="x", padx=20, pady=10)
        
        sensitivity_label = ttk.Label(sensitivity_frame, text="Detection Sensitivity:", style="TLabel")
        sensitivity_label.pack(side="left", padx=5)
        
        self.sensitivity = tk.DoubleVar(value=0.5)
        sensitivity_slider = ttk.Scale(sensitivity_frame, from_=0.1, to=1.0, orient="horizontal", 
                                    variable=self.sensitivity, length=200)
        sensitivity_slider.pack(side="left", padx=5, fill="x", expand=True)
        
        # Analysis and export button (Combined)
        btn_frame = tk.Frame(self.root, bg=self.bg_color)
        btn_frame.pack(fill="x", padx=20, pady=10)
        
        # Add combined button
        self.analyze_export_btn = ttk.Button(btn_frame, text="Analyze & Export MIDI", command=self.start_analysis_and_export_thread)
        self.analyze_export_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        # Status display
        self.status_var = StringVar()
        self.status_var.set("Ready")
        status_label = ttk.Label(self.root, textvariable=self.status_var, style="TLabel")
        status_label.pack(pady=10)
        
        # Create a frame for the visualization
        self.viz_frame = tk.Frame(self.root, bg=self.bg_color)
        self.viz_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Create initial empty figure
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.patch.set_facecolor(self.plot_bg) # Use new color
        self.ax.set_facecolor(self.plot_bg) # Use new color
        self.ax.tick_params(colors=self.text_color)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.text_color)
        self.ax.set_title("Waveform & Detected Drum Hits", color=self.text_color)
        self.ax.set_xlabel("Time (s)", color=self.text_color)
        self.ax.set_ylabel("Amplitude", color=self.text_color)
        
        # Add the plot to the UI
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def select_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac")]
        )
        if file_path:
            self.audio_file = file_path
            self.file_path.set(os.path.basename(file_path))
            self.status_var.set(f"File loaded: {os.path.basename(file_path)}")
    
    def start_tempo_detection_thread(self):
        if not self.audio_file:
            self.status_var.set("Please select an audio file first")
            return

        self.status_var.set("Detecting tempo...")
        self.detect_tempo_btn.config(state=tk.DISABLED)
        self.analyze_export_btn.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._detect_tempo_task)
        thread.start()

    def _update_status(self, message):
        self.status_var.set(message)
        if "Detecting tempo..." not in message and not self.is_analyzing:
             self.detect_tempo_btn.config(state=tk.NORMAL)
             if hasattr(self, 'analyze_export_btn'):
                 self.analyze_export_btn.config(state=tk.NORMAL)
        elif "Analyzing audio..." not in message:
            self.detect_tempo_btn.config(state=tk.NORMAL)
            self.analyze_export_btn.config(state=tk.NORMAL)
            self.select_btn.config(state=tk.NORMAL)
            self.is_analyzing = False

    def _detect_tempo_task(self):
        """The actual tempo detection logic to run in a thread"""
        try:
            y, sr = librosa.load(self.audio_file)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            # Schedule UI update in main thread
            # Access tempo[0] to get the scalar value before int conversion
            self.root.after(0, self.tempo_var.set, f"{int(tempo[0])}")
            self.root.after(0, self._update_status, f"Detected tempo: {int(tempo[0])} BPM")
        except Exception as e:
            # Schedule error message update in main thread
            self.root.after(0, self._update_status, f"Error detecting tempo: {str(e)}")
            # Ensure buttons are re-enabled on error
            self.root.after(10, self._re_enable_buttons_after_error)
    
    def start_analysis_and_export_thread(self):
        if self.is_analyzing:
            self.status_var.set("Analysis already in progress...")
            return
        if not self.audio_file:
            self.status_var.set("Please select an audio file first")
            return

        self.status_var.set("Analyzing audio...")
        self.is_analyzing = True
        # Disable buttons
        self.select_btn.config(state=tk.DISABLED)
        self.detect_tempo_btn.config(state=tk.DISABLED)
        self.analyze_export_btn.config(state=tk.DISABLED)

        # Run analysis in a separate thread
        thread = threading.Thread(target=self._analyze_audio_task)
        thread.start()

    def _analyze_audio_task(self):
        """The actual audio analysis logic to run in a thread"""
        try:
            # Load audio
            self.y, self.sr = librosa.load(self.audio_file)

            # Get tempo
            try:
                self.tempo = float(self.tempo_var.get())
            except ValueError:
                self.tempo = 120 # Default if entry is invalid
                self.root.after(0, self.tempo_var.set, "120") # Update UI

            # Detect onsets
            onset_env = librosa.onset.onset_strength(
                y=self.y,
                sr=self.sr,
                hop_length=512,
                aggregate=np.median
            )

            # Adjust threshold with sensitivity slider
            # Lower threshold means more sensitivity (detects quieter onsets)
            # We invert the slider value (0.1 to 1.0) so higher slider means more sensitive
            wait_time = 0.04 # Corresponds to roughly 1/16th note at 120bpm, prevents double triggers
            delta_time = (1.0 - self.sensitivity.get()) * 0.1 # Adjust sensitivity range
            pre_avg_time = 0.1
            # post_avg_time = 0.0 # Use default = 1 frame
            pre_max_time = 0.03
            # post_max_time = 0.0 # Use default = 1 frame

            # Ensure frame counts are non-negative integers
            wait_frames = max(0, int(wait_time * self.sr / 512))
            pre_avg_frames = max(0, int(pre_avg_time * self.sr / 512))
            pre_max_frames = max(0, int(pre_max_time * self.sr / 512))

            self.onset_frames = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=self.sr,
                hop_length=512,
                backtrack=True,
                units='frames',
                wait=wait_frames,
                delta=delta_time,
                pre_avg=pre_avg_frames,
                post_avg=1, # Default positive value
                pre_max=pre_max_frames,
                post_max=1 # Default positive value
            )

            onset_times = librosa.frames_to_time(self.onset_frames, sr=self.sr, hop_length=512)

            # Enhanced drum classification
            self.drum_types = []
            for i, frame in enumerate(self.onset_frames):
                start_sample = frame * 512
                # Use a slightly longer segment for better feature extraction
                end_sample = min(len(self.y), start_sample + int(0.1 * self.sr)) # 100ms segment
                if start_sample >= end_sample:
                    continue

                segment = self.y[start_sample:end_sample]
                if len(segment) == 0:
                    continue

                # Calculate features
                spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sr)[0].mean()
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.sr)[0].mean()
                rms = np.sqrt(np.mean(segment**2))
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y=segment)[0].mean()

                # Refined rules for drum classification (still basic, adjust as needed)
                if rms > 0.04 and spectral_centroid < 1500 and spectral_bandwidth < 2000:
                    drum_type = 36  # Kick (C1)
                elif zero_crossing_rate > 0.15 and spectral_centroid > 2500:
                    drum_type = 42  # Closed Hi-hat (F#1)
                elif rms > 0.03 and spectral_bandwidth > 2500:
                     drum_type = 38 # Snare (D1)
                # Add more rules? Open Hi-hat (46), Crash (49), Ride (51), Toms (48, 47, 45, 43, 41)?
                # Example for Open Hi-hat (might be tricky to distinguish from closed)
                # elif zero_crossing_rate > 0.1 and spectral_centroid > 3000 and rms < 0.08:
                #    drum_type = 46 # Open Hi-hat (A#1)
                else:
                    drum_type = 38  # Default to Snare if unsure

                self.drum_types.append(drum_type)

            # Schedule visualization update AND export prompt in main thread
            self.root.after(0, self.update_visualization, onset_times)
            self.root.after(10, self._prompt_and_export_midi) # Schedule export prompt shortly after viz update

        except Exception as e:
             # Schedule error message update in main thread
             self.root.after(0, self._update_status, f"Error analyzing audio: {str(e)}")
             # Ensure buttons are re-enabled on error
             self.root.after(10, self._re_enable_buttons_after_error)
    
    def update_visualization(self, onset_times):
        # Clear previous plot
        self.ax.clear()
        
        # Set background color and text properties
        self.ax.set_facecolor(self.plot_bg)
        self.ax.tick_params(colors=self.text_color)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.text_color)
            
        # Plot waveform
        # Optimization: Plot fewer points for long files to potentially speed up rendering
        plot_decimation = 1
        max_points = 50000 # Limit points to potentially avoid slow rendering
        if len(self.y) > max_points:
            plot_decimation = int(len(self.y) / max_points)

        time = np.arange(0, len(self.y), plot_decimation) / self.sr
        waveform_data = self.y[::plot_decimation]

        self.ax.plot(time, waveform_data, color=self.waveform_color, alpha=0.7)
        
        # Plot onset markers with new colors
        colors = {36: self.kick_color, 38: self.snare_color, 42: self.hihat_color}  # Kick, Snare, Hi-hat
        labels = {36: 'Kick', 38: 'Snare', 42: 'Hi-hat'}
        legend_elements = []
        
        used_labels = set()
        for i, onset_time in enumerate(onset_times):
            if i < len(self.drum_types):
                drum_type = self.drum_types[i]
                color = colors.get(drum_type, 'white')
                label = labels.get(drum_type, 'Other')
                
                self.ax.axvline(x=onset_time, color=color, alpha=0.8)
                
                if label not in used_labels:
                    legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, label=label))
                    used_labels.add(label)
        
        self.ax.set_title("Waveform & Detected Drum Hits", color=self.text_color)
        self.ax.set_xlabel("Time (s)", color=self.text_color)
        self.ax.set_ylabel("Amplitude", color=self.text_color)
        # Make legend text readable
        legend = self.ax.legend(handles=legend_elements)
        for text in legend.get_texts():
            text.set_color(self.text_color)
        
        # Update the canvas
        try:
             self.canvas.draw_idle() # Use draw_idle for potentially better responsiveness
        except Exception as e:
             print(f"Error drawing canvas: {e}") # Add print for debugging canvas errors
    
    def _re_enable_buttons_after_error(self):
        """Re-enables buttons if an error occurred during analysis/tempo detection."""
        self.is_analyzing = False
        self.select_btn.config(state=tk.NORMAL)
        self.detect_tempo_btn.config(state=tk.NORMAL)
        if hasattr(self, 'analyze_export_btn'):
            self.analyze_export_btn.config(state=tk.NORMAL)

    def _prompt_and_export_midi(self):
        if self.onset_frames is None or not self.drum_types:
            self._update_status("Analysis data missing, cannot export MIDI.")
            self._re_enable_buttons_after_error() # Re-enable buttons
            return

        self.status_var.set("Analysis complete. Please choose MIDI export location...")

        file_path = filedialog.asksaveasfilename(
            defaultextension=".mid",
            filetypes=[("MIDI Files", "*.mid")]
        )

        if not file_path:
            self._update_status("MIDI export cancelled.")
            self._re_enable_buttons_after_error() # Re-enable buttons
            return

        try:
            # Create a MIDI file
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            # Set tempo
            tempo_in_microseconds = mido.bpm2tempo(self.tempo)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo_in_microseconds))
            
            # Set time signature (assuming 4/4)
            track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
            
            # Convert onset frames to ticks
            ticks_per_beat = mid.ticks_per_beat
            seconds_per_tick = 60.0 / (self.tempo * ticks_per_beat)
            
            onset_times = librosa.frames_to_time(self.onset_frames, sr=self.sr, hop_length=512)
            
            # Translate each onset to a MIDI note
            last_tick = 0
            for i, onset_time in enumerate(onset_times):
                tick = int(onset_time / seconds_per_tick)
                delta_time = tick - last_tick
                last_tick = tick
                
                if i < len(self.drum_types):
                    note = self.drum_types[i]
                    
                    # Note on
                    track.append(mido.Message('note_on', note=note, velocity=100, time=delta_time))
                    # Note off (very short duration)
                    track.append(mido.Message('note_off', note=note, velocity=0, time=10))
            
            # Save the MIDI file
            mid.save(file_path)
            self._update_status(f"MIDI exported to {os.path.basename(file_path)}")
            
        except Exception as e:
            self._update_status(f"Error exporting MIDI: {str(e)}")
        finally:
            # Ensure buttons are re-enabled regardless of export success/failure/cancel
            self._re_enable_buttons_after_error()


if __name__ == "__main__":
    root = Tk()
    app = DrumBeatExtractor(root)
    root.mainloop()