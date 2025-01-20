# Kunstwerk

A Python-based tool for generating parallel subtitle videos for opera performances, with synchronized original language and translated text.

This is the repo behind the [YouTube channel of the same name](https://www.youtube.com/@kunstwerk-opera)!

## Features

- Automatic audio transcription using OpenAI's Whisper model
- Vocal separation using Demucs
- Text alignment between transcribed audio and libretto
- Parallel subtitle video generation with original and translated text
- Support for multiple languages
- Configurable video output settings

## Prerequisites

- Python 3.8+
- FFmpeg
- yt-dlp
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kunstwerk
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key'
```

## Usage

Example output: [Tristan und Isolde with parallel subtitles](https://www.youtube.com/watch?v=2R6lTcdJoCk)

1. Create a YAML configuration file for your opera (see example configs):

```yaml
title: TRISTAN UND ISOLDE
file_prefix: tristan
language: de
start_idx: 1
end_idx: 33
overture_indices: [1]
secondary_color: Silver
video_width: 3840
video_height: 2160
font_size: 96
res_divisor: 1
playlist_url: https://www.youtube.com/playlist?list=EXAMPLE

characters:
  - Tristan
  - Isolde
  # Add other characters...
```

2. Process the opera:
```bash
python kunstwerk.py configs/your_config.yaml
```

You can also skip certain steps if you've already completed them:
```bash
# Skip download/separation if you already have the audio files:
python kunstwerk.py configs/your_config.yaml --skip-download

# Skip transcription if you already have the transcriptions:
python kunstwerk.py configs/your_config.yaml --skip-transcribe

# Skip both download and transcription:
python kunstwerk.py configs/your_config.yaml --skip-download --skip-transcribe
```

## Configuration Options

- `title`: Opera title displayed in the video
- `file_prefix`: Prefix for generated files
- `language`: Source language code (e.g., 'de' for German)
- `start_idx`/`end_idx`: Range of scenes to process
- `overture_indices`: List of instrumental sections to skip
- `secondary_color`: Color for translated text
- `video_width`/`video_height`: Output video dimensions
- `font_size`: Base font size
- `res_divisor`: Resolution scaling factor
- `playlist_url`: YouTube playlist URL for downloading
- `characters`: List of character names for formatting

## Project Structure

- `separate.sh`: Downloads and separates audio
- `transcribe.py`: Handles audio transcription
- `make_video.py`: Generates the final video
- `align.py`: Aligns transcribed text with libretto
- `config_parser.py`: Parses YAML configuration
- `video_gen/`: Video generation modules
  - `config/`: Configuration classes
  - `frame/`: Frame generation
  - `text/`: Text formatting
  - `video/`: Video creation
