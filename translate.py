#!/usr/bin/env python3
from typing import List, Tuple, Iterator
import anthropic
from pathlib import Path
import os
import argparse
from tqdm import tqdm
from config_parser import parse_opera_config, OperaConfig

def chunk_text(text: str, chunk_size: int = 12) -> Iterator[Tuple[List[str], int]]:
    """Split text into chunks of roughly chunk_size double newlines each"""
    lines = text.split("\n\n")
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        yield chunk, i

def create_translation_prompt(text: str, source_lang: str, target_lang: str, opera_title: str) -> str:
    """Creates a prompt for Claude to translate opera text"""
    return f"""You are an expert translator of opera libretti from {source_lang} to {target_lang}.
You are currently translating "{opera_title}". Please translate the following section of text, maintaining the poetic and dramatic qualities while ensuring accuracy. It is imperative that you preserve all line breaks and formatting exactly as in the original. There should be a 1:1 correspondence between each line in the original and each line in the new language.

Here is the text to translate:

{text}

Provide only the translation with the same line break structure, nothing else."""

def translate_chunk(
    client: anthropic.Anthropic, 
    chunk: List[str], 
    source_lang: str, 
    target_lang: str,
    opera_title: str
) -> str:
    """Translates a chunk of text using Claude"""
    chunk_text = "\n\n".join(chunk)
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        messages=[
            {"role": "user", "content": create_translation_prompt(
                chunk_text, source_lang, target_lang, opera_title
            )}
        ]
    )
    return message.content[0].text

def translate_libretto(config: OperaConfig, target_lang: str, force: bool = False) -> List[Tuple[str, str]]:
    """Translates entire libretto and returns original/translation pairs"""
    # Check if translation already exists
    target_path = Path(f"libretti/{config.file_prefix}_{target_lang}.txt")
    if target_path.exists() and not force:
        print(f"Translation to {target_lang} already exists at {target_path}. Use --force to overwrite.")
        return []

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    
    # Read original libretto
    source_path = Path(f"libretti/{config.file_prefix}_{config.language}.txt")
    with open(source_path, "r", encoding="utf-8") as f:
        source_text = f.read()
    
    translated_chunks = []
    
    # Translate each chunk
    for chunk, start_idx in tqdm(list(chunk_text(source_text)), desc="Translating chunks"):
        if any(line.strip() for line in chunk):  # Skip chunks that are all empty
            translated = translate_chunk(
                client, chunk, config.language, target_lang, config.title
            )
            translated_chunks.append((chunk, translated.split("\n\n")))
    
    # Reconstruct full translation
    source_lines = []
    translated_lines = []
    
    for original_chunk, translated_chunk in translated_chunks:
        source_lines.extend(original_chunk)
        translated_lines.extend(translated_chunk)
    
    # Save translations
    output_path = Path(f"libretti/{config.file_prefix}_{target_lang}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(translated_lines))
        
    return list(zip(source_lines, translated_lines))

def main():
    parser = argparse.ArgumentParser(description="Translate opera libretti using Claude")
    parser.add_argument("config", help="Path to the opera configuration YAML file")
    parser.add_argument("target_lang", help="Target language code (e.g., 'en' for English)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing translation file")
    
    args = parser.parse_args()
    
    try:
        config = parse_opera_config(args.config)
        translate_libretto(config, args.target_lang, args.force)
        print("\nTranslation completed successfully.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
