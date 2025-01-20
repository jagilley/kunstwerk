from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import re

@dataclass
class StageDirection:
    text: str

@dataclass
class Lyric:
    text: str
    character: str
    start_time: Optional[float]
    end_time: Optional[float]

@dataclass
class Line:
    language: str
    content: Union[StageDirection, Lyric]

@dataclass
class Libretto:
    lines: List[Tuple[Line]]

def parse_libretto(html_content: str) -> Libretto:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all table rows
    rows = soup.find_all('tr')
    
    all_lines = []
    
    for row in rows:
        # Get the two columns (German and English)
        columns = row.find_all('td')
        if len(columns) != 2:
            continue
            
        german_text = columns[0].get_text(strip=True)
        english_text = columns[1].get_text(strip=True)
        
        if not german_text or not english_text:
            continue
            
        # Process the content of each column
        german_lines = []
        english_lines = []
        
        # Process each element in the German column
        for element in columns[0].children:
            if element.name == 'i':
                # Stage direction
                german_lines.append(Line(
                    language="de",
                    content=StageDirection(text=element.get_text().strip())
                ))
            elif element.name == 'b':
                # Character name - skip it as it will be part of the next line
                continue
            elif isinstance(element, str) and element.strip():
                # Check if this is a character's line
                if columns[0].find_previous('b') is not None:
                    character = columns[0].find_previous('b').get_text().strip()
                    german_lines.append(Line(
                        language="de",
                        content=Lyric(
                            text=element.strip(),
                            character=character,
                            start_time=None,
                            end_time=None
                        )
                    ))
                    
        # Process each element in the English column
        for element in columns[1].children:
            if element.name == 'i':
                # Stage direction
                english_lines.append(Line(
                    language="en",
                    content=StageDirection(text=element.get_text().strip())
                ))
            elif element.name == 'b':
                # Character name - skip it
                continue
            elif isinstance(element, str) and element.strip():
                # Check if this is a character's line
                if columns[1].find_previous('b') is not None:
                    character = columns[1].find_previous('b').get_text().strip()
                    english_lines.append(Line(
                        language="en",
                        content=Lyric(
                            text=element.strip(),
                            character=character,
                            start_time=None,
                            end_time=None
                        )
                    ))
        
        # Pair up the German and English lines
        for de_line, en_line in zip(german_lines, english_lines):
            all_lines.append((de_line, en_line))
    
    return Libretto(lines=all_lines)

# Example usage:
with open('libretti/rheingold_aligned.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

libretto = parse_libretto(html_content)

# Print some results to verify
for line_pair in libretto.lines[:5]:  # First 5 pairs
    de_line, en_line = line_pair
    if isinstance(de_line.content, StageDirection):
        print(f"Stage Direction DE: {de_line.content.text}")
        print(f"Stage Direction EN: {en_line.content.text}")
    else:
        print(f"Lyric DE: {de_line.content.character}: {de_line.content.text}")
        print(f"Lyric EN: {en_line.content.character}: {en_line.content.text}")
    print("---")