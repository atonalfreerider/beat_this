import sys
import os
import json
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.generators import Sine
from pathlib import Path
from beat_this.inference import File2Beats
import argparse

def calculate_bpm(start_time, end_time):
    measure_duration = end_time - start_time
    return 240 / measure_duration  # 4 beats per measure * 60 seconds per minute

def split_measures(beats, beat_types, min_bpm=120, max_bpm=190):
    split_beats = []
    split_beat_types = []
    i = 0
    while i < len(beats) - 1:
        start_time = beats[i]
        end_index = next((j for j in range(i+1, len(beats)) if beat_types[j] == 1), len(beats))
        end_time = beats[min(end_index, len(beats) - 1)]
        
        measure_bpm = calculate_bpm(start_time, end_time)
        
        if measure_bpm < min_bpm:
            mid_time = (start_time + end_time) / 2
            half_measure_bpm = calculate_bpm(start_time, mid_time)
            
            if half_measure_bpm <= max_bpm:
                # Split the measure
                split_beats.extend([start_time, mid_time])
                split_beat_types.extend([1, 1])
                for j in range(i+1, end_index):
                    if beats[j] < mid_time:
                        split_beats.append(beats[j])
                        split_beat_types.append(beat_types[j])
                    else:
                        split_beats.append(beats[j])
                        split_beat_types.append(beat_types[j])
            else:
                # Keep original measure
                split_beats.extend(beats[i:end_index])
                split_beat_types.extend(beat_types[i:end_index])
        else:
            # Keep original measure
            split_beats.extend(beats[i:end_index])
            split_beat_types.extend(beat_types[i:end_index])
        
        i = end_index
    
    return np.array(split_beats), np.array(split_beat_types)

def calculate_zouk_time(beats, beat_types, max_bpm=190):
    zouk_time_beats = []
    
    i = 0
    while i < len(beats) - 1:
        start_time = beats[i]
        end_index = next((j for j in range(i+1, len(beats)) if beat_types[j] == 1), len(beats))
        end_time = beats[min(end_index, len(beats) - 1)]
        
        measure_bpm = calculate_bpm(start_time, end_time)
        print(f"Measure BPM: {measure_bpm:.2f}")
        
        if measure_bpm > max_bpm:
            # Check if the next measure is also above max_bpm
            if end_index < len(beats) - 1:
                next_end_index = next((j for j in range(end_index+1, len(beats)) if beat_types[j] == 1), len(beats))
                next_end_time = beats[min(next_end_index, len(beats) - 1)]
                next_measure_bpm = calculate_bpm(end_time, next_end_time)
                
                if next_measure_bpm > max_bpm:
                    # Combine the two measures
                    combined_duration = next_end_time - start_time
                    combined_bpm = 480 / combined_duration  # 8 beats * 60 seconds
                    adjusted_bpm = combined_bpm / 4
                    
                    # Generate 8 Zouk beats for the combined measure
                    for j in range(8):
                        time = start_time + (combined_duration * j / 8)
                        beat_type = 1 if j in [0, 4] else (2 if j in [3, 6] else 3)
                        zouk_time_beats.append([time, beat_type])
                    
                    i = next_end_index
                    continue
        
        # Generate 8 Zouk beats for this measure
        measure_duration = end_time - start_time
        for j in range(8):
            time = start_time + (measure_duration * j / 8)
            beat_type = 1 if j in [0, 4] else (2 if j in [3, 6] else 3)
            zouk_time_beats.append([time, beat_type])
        
        i = end_index
    
    print(f"Generated {len(zouk_time_beats)} Zouk time beats")
    return np.array(zouk_time_beats)

def run_inference(audio_file):
    print(f"Running inference on {audio_file}")
    f2b = File2Beats()
    audio_path = Path(audio_file)
    beat, downbeat = f2b(audio_path)
    print(f"Inference complete. Found {len(beat)} beats and {len(downbeat)} downbeats.")
    return beat, downbeat

def process_audio(audio_file, debug=False):
    file_dir = os.path.dirname(os.path.abspath(audio_file))
    file_name = os.path.splitext(os.path.basename(audio_file))[0]

    print(f"Loading audio file: {audio_file}")
    y, sr = librosa.load(audio_file)
    print(f"Audio duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")

    # Run inference
    inferred_beats, inferred_downbeats = run_inference(audio_file)

    # Convert inferred beats to the format expected by the rest of the script
    beats = inferred_beats
    beat_types = np.ones_like(beats)
    beat_types[np.isin(beats, inferred_downbeats)] = 1
    beat_types[~np.isin(beats, inferred_downbeats)] = 2

    # Continue with the existing processing
    beats, beat_types = split_measures(beats, beat_types)
    zouk_time_beats = calculate_zouk_time(beats, beat_types)

    if len(zouk_time_beats) == 0:
        print("No Zouk time beats generated. Exiting.")
        return

    if debug:
        plt.figure(figsize=(20, 8))
        librosa.display.waveshow(y, sr=sr, alpha=0.5)
        for beat in zouk_time_beats:
            time, beat_type = beat
            if beat_type == 1:
                plt.axvline(x=time, color='r', linewidth=2, alpha=0.8)
            elif beat_type == 2:
                plt.axvline(x=time, color='g', linewidth=1.5, alpha=0.6)
        
        # Add BPM information
        audio_duration = librosa.get_duration(y=y, sr=sr)
        for i in range(0, int(audio_duration), 30):
            section_beats = [beat for beat in zouk_time_beats if i <= beat[0] < i + 30]
            if len(section_beats) > 1:
                section_times = [beat[0] for beat in section_beats]
                section_bpm = calculate_bpm(section_times[0], section_times[-1]) * 2  # Multiply by 2 because we're using half-measures
                plt.text(i / audio_duration, 1.05, f"{section_bpm:.0f} BPM", transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

        plt.title('Audio Waveform with Zouk Time Beats')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust top margin to accommodate BPM text
        plt.savefig(os.path.join(file_dir, f"{file_name}_zouk-time-analysis.png"))
        plt.close()

        print(f"Saved waveform plot: {file_name}_zouk-time-analysis.png")

        audio = AudioSegment.from_file(audio_file)
        downbeat_tick = Sine(1000).to_audio_segment(duration=50).fade_out(25).apply_gain(-3)
        upbeat_tick = Sine(1200).to_audio_segment(duration=50).fade_out(25).apply_gain(-6)

        for beat in zouk_time_beats:
            time, beat_type = beat
            position_ms = int(time * 1000)
            if beat_type == 1:
                audio = audio.overlay(downbeat_tick, position=position_ms)
            elif beat_type == 2:
                audio = audio.overlay(upbeat_tick, position=position_ms)

        audio.export(os.path.join(file_dir, f"{file_name}_zouk-time-analysis.wav"), format="wav")
        print(f"Saved audio with beats: {file_name}_zouk-time-analysis.wav")

    with open(os.path.join(file_dir, f"{file_name}_zouk-time-analysis.json"), 'w') as json_file:
        json.dump(zouk_time_beats.tolist(), json_file, indent=4)
    print(f"Saved beat data: {file_name}_zouk-time-analysis.json")

    print(f"Zouk Time analysis complete for {audio_file}")
    print(f"Outputs saved in {file_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze audio file for Zouk timing")
    parser.add_argument("audio_file", help="Path to the audio file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (generate WAV and graph outputs)")
    args = parser.parse_args()

    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)
    
    process_audio(args.audio_file, debug=args.debug)
