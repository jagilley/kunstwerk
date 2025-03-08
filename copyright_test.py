import os
import sys
from moviepy.editor import AudioFileClip, concatenate_audioclips
from typing import List
from config_parser import parse_opera_config

def create_copyright_test_audio(
    config_path: str,
    output_filename: str = None
):
    """
    Creates a simple audio file combining all audio clips without any visuals,
    for the purpose of testing copyright issues on YouTube.
    
    Args:
        config_path: Path to the opera configuration YAML file
        output_filename: Path to save the output file (default: opera_name_copyright_test.mp4)
    """
    # Parse configuration
    config = parse_opera_config(config_path)
    
    if not output_filename:
        output_filename = f'output/{config.file_prefix}_copyright_test.mp4'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    print(f"Creating copyright test audio for {config.title}...")
    
    # Load audio files
    audio_files = [f"audio/{config.file_prefix}/{str(i).zfill(2)}.m4a" 
                   for i in range(config.start_idx, config.end_idx)]
    
    # Verify audio files exist
    for audio_file in audio_files:
        if not os.path.exists(audio_file):
            print(f"Warning: Audio file not found: {audio_file}")
    
    # Load audio clips
    audio_clips: List[AudioFileClip] = []
    for audio_path in audio_files:
        if os.path.exists(audio_path):
            clip = AudioFileClip(audio_path)
            audio_clips.append(clip)
            print(f"Added audio clip: {audio_path} (duration: {clip.duration:.2f}s)")
    
    if not audio_clips:
        print("Error: No valid audio clips found")
        return
    
    # Combine audio clips
    print("Combining audio clips...")
    combined_audio = concatenate_audioclips(audio_clips)
    total_duration = combined_audio.duration
    
    # Create a blank video with the combined audio
    print(f"Creating blank video with combined audio (duration: {total_duration:.2f}s)...")
    combined_audio.write_audiofile("temp_audio.mp3")
    
    # Use ffmpeg to create a blank video with the audio
    os.system(f"ffmpeg -y -f lavfi -i color=c=black:s=1280x720:r=1 -i temp_audio.mp3 -c:v libx264 -tune stillimage -c:a aac -b:a 192k -shortest {output_filename}")
    
    # Clean up temporary audio file
    os.remove("temp_audio.mp3")
    
    # Close all audio clips to free memory
    for clip in audio_clips:
        clip.close()
    combined_audio.close()
    
    print(f"Copyright test video saved to: {output_filename}")
    print("Use this file to test for copyright issues before creating the full video.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python copyright_test.py <config.yaml>")
        sys.exit(1)
    
    create_copyright_test_audio(sys.argv[1])