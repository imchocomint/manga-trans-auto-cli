#!/usr/bin/env python3

"""
Batch Image Translation with Advanced Sentence Grouping
Extracts text with coordinates, translates via Google Translate API,
then groups words into sentences using IMPROVED algorithm that handles:
  - Speech bubbles (vertical stacking)
  - Multi-line dialog boxes
  - Proximity-based grouping (horizontal gaps)
  - Column-based grouping (same x-position)

Features:
- Bulk process indexed images (01.webp, 02.webp, etc.)
- Extracts text with exact coordinates (x, y, width, height)
- Translates directly with Google Translate (no DeepL)
- Advanced grouping: handles speech bubbles, dialogs, columns
- Saves JSON files with sentences + word coordinates
- Resume capability (skip existing)
- Range processing (--start 1 --end 50)
- Configurable grouping parameters

Setup:
1. Set up Google Cloud Vision (see GOOGLE_VISION_SETUP.md)
2. export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
3. pip install google-cloud-vision google-cloud-translate

Usage:
    # Process all images, translate to English
    python batch_translate_google.py --input-dir ./images --output-dir ./json_data
    
    # Adjust grouping thresholds
    python batch_translate_google.py --input-dir ./images --output-dir ./json_data --vertical-gap 25
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

# Import backend classes
try:
    from google_vision_coordinates_deepl import GoogleVisionOCR
    from google.cloud import translate_v2
except ImportError as e:
    if 'google_vision_coordinates_deepl' in str(e):
        print("[ERROR] Cannot import from google_vision_coordinates_deepl.py")
        print("Make sure google_vision_coordinates_deepl.py is in the same directory")
    else:
        print("[ERROR] Cannot import google.cloud.translate_v2")
        print("Install with: pip install google-cloud-translate")
    sys.exit(1)


class GoogleTranslator:
    """Simple wrapper for Google Translate API."""

    def __init__(self, target_language: str = 'en'):
        """Initialize translator."""
        try:
            from google.cloud import translate_v2 as translate
            self.client = translate.Client()
            self.target_language = target_language
        except Exception as e:
            print(f"[ERROR] Failed to initialize Google Translate: {str(e)}")
            sys.exit(1)

    def translate(self, text: str) -> Optional[str]:
        """Translate text to target language."""
        try:
            result = self.client.translate(
                text,
                target_language=self.target_language,
                format_="text",  # optional but safe
            )
            return result["translatedText"]
        except Exception as e:
            print(f"[WARN] Translation error: {str(e)}")
            return None

class AdvancedSentenceGrouper:
    """
    Advanced sentence grouper.
    Modes:
      - "lines": current behavior (line + gap based)
      - "bubble": group words into spatial regions (bubbles/boxes)
    """

    def __init__(
        self,
        horizontal_gap_threshold: int = 30,
        vertical_gap_threshold: int = 25,
        column_x_threshold: int = 20,
        mode: str = "lines",
        region_margin: int = 15,
    ):
        self.horizontal_gap_threshold = horizontal_gap_threshold
        self.vertical_gap_threshold = vertical_gap_threshold
        self.column_x_threshold = column_x_threshold
        self.mode = mode
        self.region_margin = region_margin
    
    def _get_word_center_x(self, word: Dict) -> float:
        """Get horizontal center of word."""
        return word['x'] + word['width'] / 2
    
    def _get_word_center_y(self, word: Dict) -> float:
        """Get vertical center of word."""
        return word['y'] + word['height'] / 2
    
    def _group_by_lines(self, words: List[Dict]) -> List[List[Dict]]:
        """
        Group words into horizontal lines based on y-coordinate proximity.
        Words within vertical_gap_threshold pixels are on same line.
        """
        if not words:
            return []
        
        # Sort by y position (top to bottom)
        sorted_words = sorted(words, key=lambda w: w['y'])
        
        lines = []
        current_line = [sorted_words[0]]
        current_y = sorted_words[0]['y']
        
        for word in sorted_words[1:]:
            # Check if word is on same line (within vertical threshold)
            if abs(word['y'] - current_y) <= self.vertical_gap_threshold:
                current_line.append(word)
            else:
                # New line
                lines.append(current_line)
                current_line = [word]
                current_y = word['y']
        
        # Add last line
        if current_line:
            lines.append(current_line)
        
        # Sort words within each line by x position (left to right)
        for line in lines:
            line.sort(key=lambda w: w['x'])
        
        return lines
    
    def _group_words_in_line(self, line_words: List[Dict]) -> List[List[Dict]]:
        """
        Group words within a line into sentence chunks based on horizontal gaps.
        Large gaps (> threshold) indicate separate sentences.
        """
        if not line_words:
            return []
        
        if len(line_words) == 1:
            return [line_words]
        
        chunks = []
        current_chunk = [line_words[0]]
        
        for i in range(1, len(line_words)):
            prev_word = line_words[i - 1]
            curr_word = line_words[i]
            
            # Calculate gap between words
            gap = curr_word['x'] - (prev_word['x'] + prev_word['width'])
            
            if gap <= self.horizontal_gap_threshold:
                # Same sentence chunk
                current_chunk.append(curr_word)
            else:
                # New sentence chunk
                chunks.append(current_chunk)
                current_chunk = [curr_word]
        
        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def group_words(self, words: List[Dict]) -> List[Dict]:
        """Group words into sentences."""
        if not words:
            return []

        # NEW: bubble/box mode
        if self.mode == "bubble":
            regions = self._group_by_regions(words)
            sentences = []
            for region in regions:
                # Sort words inside a bubble in reading order
                region_sorted = sorted(region, key=lambda w: (w["y"], w["x"]))
                sentences.append(self._create_sentence(region_sorted))
            return sentences

        # OLD behavior: line + gap based
        lines = self._group_by_lines(words)
        sentences = []
        for line in lines:
            chunks = self._group_words_in_line(line)
            for chunk in chunks:
                sentence = self._create_sentence(chunk)
                sentences.append(sentence)
        return sentences
    
    def _create_sentence(self, words: List[Dict]) -> Dict:
        """Create sentence dict from word list (no translation yet)."""
        sentence = {
            "original_words": [w["text"] for w in words],
            "full_original": " ".join(w["text"] for w in words),
            "word_count": len(words),
            "coordinates": {
                "x": min(w["x"] for w in words),
                "y": min(w["y"] for w in words),
                "x_end": max(w["x"] + w["width"] for w in words),
                "y_end": max(w["y"] + w["height"] for w in words),
            },
        }

        return sentence
    def _group_by_regions(self, words: List[Dict]) -> List[List[Dict]]:

        if not words:
            return []

        # Precompute expanded boxes
        boxes = []
        m = self.region_margin
        for w in words:
            x1 = w["x"] - m
            y1 = w["y"] - m
            x2 = w["x"] + w["width"] + m
            y2 = w["y"] + w["height"] + m
            boxes.append((x1, y1, x2, y2))

        n = len(words)
        visited = [False] * n
        regions: List[List[Dict]] = []

        for i in range(n):
            if visited[i]:
                continue
            stack = [i]
            visited[i] = True
            group = []

            while stack:
                j = stack.pop()
                group.append(words[j])
                x1j, y1j, x2j, y2j = boxes[j]

                for k in range(n):
                    if visited[k]:
                        continue
                    x1k, y1k, x2k, y2k = boxes[k]
                    # Overlap test
                    if not (x2j < x1k or x2k < x1j or y2j < y1k or y2k < y1j):
                        visited[k] = True
                        stack.append(k)

            regions.append(group)

        return regions


class BatchImageTranslatorGoogle:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_language: str = "en",
        image_format: str = "webp",
        padding: int = 2,
        horizontal_gap: int = 30,
        vertical_gap: int = 25,
        column_threshold: int = 20,
        group_mode: str = "lines",
        region_margin: int = 15,
    ):
        ...
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_language = target_language
        self.image_format = image_format.lstrip('.')
        self.padding = padding
        
        # Validate input directory
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend classes
        try:
            self.ocr = GoogleVisionOCR()
            self.translator = GoogleTranslator(target_language=target_language)
            self.grouper = AdvancedSentenceGrouper(
                horizontal_gap_threshold=horizontal_gap,
                vertical_gap_threshold=vertical_gap,
                column_x_threshold=column_threshold,
                mode=group_mode,
                region_margin=region_margin,
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize: {str(e)}")
            sys.exit(1)
    
    def find_images(self, start: int = None, end: int = None) -> list:
        """Find all indexed images in directory."""
        images = []
        
        pattern = f'*.{self.image_format}'
        files = sorted(self.input_dir.glob(pattern))
        
        for filepath in files:
            try:
                index = int(filepath.stem)
                images.append((index, filepath))
            except ValueError:
                pass
        
        if start is not None:
            images = [(idx, path) for idx, path in images if idx >= start]
        if end is not None:
            images = [(idx, path) for idx, path in images if idx <= end]
        
        return images
    
    def get_output_path(self, index: int) -> Path:
        """Generate output JSON path."""
        padded_index = str(index).zfill(self.padding)
        return self.output_dir / f'{padded_index}.json'
    
    def process_images(
        self,
        start: int = None,
        end: int = None,
        skip_existing: bool = False,
        verbose: bool = True
    ) -> dict:
        """
        Process all images in range.
        """
        images = self.find_images(start, end)
        
        if not images:
            print(f"[ERROR] No images found in {self.input_dir}")
            return {'success': False, 'processed': 0, 'failed': 0}
        
        # Language display name
        lang_names = {
            'en': 'English', 'id': 'Indonesian', 'vi': 'Vietnamese',
            'zh': 'Chinese (Simplified)', 'ja': 'Japanese', 'ko': 'Korean',
            'fr': 'French', 'es': 'Spanish', 'de': 'German',
            'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian',
            'ar': 'Arabic', 'th': 'Thai',
        }
        lang_display = lang_names.get(self.target_language, self.target_language.upper())
        
        print(f"\n{'='*70}")
        print(f"Batch Google Vision OCR + Google Translate (Advanced Grouping)")
        print(f"{'='*70}")
        print(f"Input directory:      {self.input_dir}")
        print(f"Output directory:     {self.output_dir}")
        print(f"Target Language:      {lang_display} ({self.target_language})")
        print(f"Images found:         {len(images)}")
        print(f"Range:                {start or images[0][0]} to {end or images[-1][0]}")
        print(f"{'='*70}\n")
        
        results = {
            'success': True,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        for idx, (index, input_path) in enumerate(images, 1):
            output_path = self.get_output_path(index)
            
            if skip_existing and output_path.exists():
                if verbose:
                    print(f"[{idx:3d}/{len(images)}] ⊘ {input_path.name} (exists)")
                results['skipped'] += 1
                continue
            
            try:
                if verbose:
                    print(f"[{idx:3d}/{len(images)}] → {input_path.name}...", end=' ', flush=True)
                
                # Extract text with Google Vision
                ocr_result = self.ocr.extract_text_with_coordinates(str(input_path))
                
                if not ocr_result.get('text'):
                    if verbose:
                        print(f"✗ No text extracted")
                    results['failed'] += 1
                    results['errors'].append({
                        'file': input_path.name,
                        'error': 'No text extracted from image'
                    })
                    continue
                
                # Use raw OCR words (no per-word translation)
                words = ocr_result["words"]

                # 1) Group words into sentences/bubbles
                sentences = self.grouper.group_words(words)

                if not sentences:
                    if verbose:
                        print("✗ No sentences formed")
                    results["failed"] += 1
                    results["errors"].append({
                        "file": input_path.name,
                        "error": "No sentences formed from words",
                    })
                    continue

                # 2) Translate each sentence as a whole
                for sentence in sentences:
                    translated = self.translator.translate(sentence["full_original"])
                    sentence["full_translation"] = translated or ""
                    # Optional: rough token split
                    sentence["translated_words"] = (translated or "").split()

                # 3) (Optional) mark that words have no direct translation
                for w in words:
                    w["translation"] = None  # or delete this key entirely

                # 4) Save JSON
                data = {
                    "version": "1.1",
                    "word_count": len(words),
                    "sentence_count": len(sentences),
                    "grouping_params": {
                        "horizontal_gap_threshold": self.grouper.horizontal_gap_threshold,
                        "vertical_gap_threshold": self.grouper.vertical_gap_threshold,
                        "column_x_threshold": self.grouper.column_x_threshold,
                    },
                    "words": words,
                    "sentences": sentences,
                }
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                if verbose:
                    print(f"✓ ({len(words)} words, {len(sentences)} sentences)")
                results['processed'] += 1
                
            except Exception as e:
                if verbose:
                    print(f"✗ {str(e)}")
                results['failed'] += 1
                results['errors'].append({
                    'file': input_path.name,
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Summary")
        print(f"{'='*70}")
        print(f"✓ Processed: {results['processed']}")
        print(f"✗ Failed:    {results['failed']}")
        print(f"⊘ Skipped:   {results['skipped']}")
        print(f"{'='*70}\n")
        
        if results['errors']:
            print("Errors:")
            for error in results['errors']:
                print(f"  - {error['file']}: {error['error']}")
            print()
        
        results['success'] = results['failed'] == 0
        return results


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Batch Image Translation with Advanced Sentence Grouping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Uses Google Cloud Vision for OCR + Google Translate for translation.
Advanced grouping handles speech bubbles, dialogs, and multi-line text.

Examples:
  # Default settings
  python batch_translate_google.py --input-dir ./images --output-dir ./json_data
  
  # Adjust for tightly spaced text (speech bubbles)
  python batch_translate_google.py \\
    --input-dir ./images --output-dir ./json_data \\
    --horizontal-gap 20 --vertical-gap 15
  
  # Adjust for loosely spaced text
  python batch_translate_google.py \\
    --input-dir ./images --output-dir ./json_data \\
    --horizontal-gap 50 --vertical-gap 40
  
  # Different language
  python batch_translate_google.py --input-dir ./images --output-dir ./json_data --lang id
  
  # Process range
  python batch_translate_google.py --input-dir ./images --output-dir ./json_data --start 1 --end 50
  
  # Resume
  python batch_translate_google.py --input-dir ./images --output-dir ./json_data --skip-existing

Grouping Parameters:
  --horizontal-gap (default: 30px)
    Gap between words on same line to split sentences
    Smaller = more sentences | Larger = fewer sentences
  
  --vertical-gap (default: 25px)
    Gap between lines to consider same text block
    Smaller = stricter line grouping | Larger = looser grouping
  
  For speech bubbles: use smaller gaps (15-25px)
  For regular documents: use standard gaps (25-35px)
  For sparse text: use larger gaps (35-50px)
        """
    )
    
    parser.add_argument('--input-dir', required=True, help='Input directory with images')
    parser.add_argument('--output-dir', required=True, help='Output directory for JSON files')
    parser.add_argument('--lang', default='en', help='Target language code (default: en)')
    parser.add_argument('--format', default='webp', help='Image format (default: webp)')
    parser.add_argument('--padding', type=int, default=2, help='Zero-padding (default: 2)')
    parser.add_argument('--horizontal-gap', type=int, default=30, help='Horizontal gap threshold (default: 30)')
    parser.add_argument('--vertical-gap', type=int, default=25, help='Vertical gap threshold (default: 25)')
    parser.add_argument('--column-threshold', type=int, default=20, help='Column x-diff threshold (default: 20)')
    parser.add_argument('--start', type=int, help='Start index')
    parser.add_argument('--end', type=int, help='End index')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing JSON files')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    parser.add_argument(
        "--group-mode",
        choices=["lines", "bubble"],
        default="lines",
        help='Grouping strategy: "lines" (default) or "bubble" (1 sentence per region/bubble)',
    )
    parser.add_argument(
        "--region-margin",
        type=int,
        default=15,
        help="Pixel margin used when merging words into regions in bubble mode",
    )
    args = parser.parse_args()
    
    try:
        # Create translator
        translator = BatchImageTranslatorGoogle(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_language=args.lang,
            image_format=args.format,
            padding=args.padding,
            horizontal_gap=args.horizontal_gap,
            vertical_gap=args.vertical_gap,
            column_threshold=args.column_threshold,
            group_mode=args.group_mode,
            region_margin=args.region_margin,
        )
        
        # Process images
        results = translator.process_images(
            start=args.start,
            end=args.end,
            skip_existing=args.skip_existing,
            verbose=not args.quiet
        )
        
        sys.exit(0 if results['success'] else 1)
        
    except Exception as e:
        print(f"\n[FATAL ERROR] {str(e)}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
