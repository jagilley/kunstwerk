import json
import os
import sys
import librosa
from glob import glob

def convert_transcription(input_file, output_file=None, audio_file=None):
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get audio duration if audio file is provided
    duration = None
    if audio_file and os.path.exists(audio_file):
        try:
            duration = librosa.get_duration(path=audio_file)
        except Exception as e:
            print(f"Error getting duration from {audio_file}: {e}")
    
    # If no duration obtained, use the last word's end time as an estimate
    if duration is None and data.get("words"):
        words = data.get("words", [])
        if words:
            duration = max((word.get("end", 0.0) for word in words if word.get("type") != "spacing"), default=0.0)
    
    # Initialize the new format
    new_format = {
        "duration": duration or 0.0,  # Use calculated duration or 0 if not available
        "language": {
            "ita": "italian",
            "deu": "german", 
            "eng": "english",
            "fra": "french",
            "spa": "spanish"
        }.get(data.get("language_code", ""), data.get("language_code", "")),
        "text": data.get("text", ""),
        "segments": None,
        "words": []
    }
    
    # Convert the words, skipping spaces
    for word in data.get("words", []):
        if word.get("type") == "spacing":
            continue
            
        new_format["words"].append({
            "start": word.get("start", 0.0),
            "end": word.get("end", 0.0),
            "word": word.get("text", "")
        })
    
    # Determine output filename if not provided
    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = f"{base}.json"
    
    # Write the converted data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_format, f, indent=2, ensure_ascii=False)
    
    return output_file

def convert_directory(directory_pattern, audio_dir=None):
    files = glob(directory_pattern)
    for file in files:
        if file.endswith('.json'):
            # Try to find corresponding audio file
            audio_file = None
            if audio_dir:
                # Extract the base filename without extension
                base_name = os.path.splitext(os.path.basename(file))[0]
                # Look for audio files with matching name in audio_dir
                for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                    possible_audio = os.path.join(audio_dir, f"{base_name}{ext}")
                    if os.path.exists(possible_audio):
                        audio_file = possible_audio
                        break
            
            output = convert_transcription(file, audio_file=audio_file)
            print(f"Converted {file} to {output}" + (f" with duration from {audio_file}" if audio_file else ""))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert transcription files to a new format')
    parser.add_argument('files', nargs='+', help='JSON file(s) to convert')
    parser.add_argument('--audio-dir', '-a', help='Directory containing audio files matching the JSON filenames')
    
    args = parser.parse_args()
    
    for pattern in args.files:
        files = glob(pattern)
        for file in files:
            if os.path.isfile(file) and file.endswith('.json'):
                # Try to find corresponding audio file
                audio_file = None
                if args.audio_dir:
                    # Extract the base filename without extension
                    base_name = os.path.splitext(os.path.basename(file))[0]
                    # Look for audio files with matching name in audio_dir
                    for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
                        possible_audio = os.path.join(args.audio_dir, f"{base_name}{ext}")
                        if os.path.exists(possible_audio):
                            audio_file = possible_audio
                            break
                
                output = convert_transcription(file, audio_file=audio_file)
                print(f"Converted {file} to {output}" + (f" with duration from {audio_file}" if audio_file else ""))
            else:
                print(f"Skipping {file} - not a JSON file or doesn't exist")