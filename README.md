# Extract text + translate sentences from manga images
A product of Perplexity. I'm not affliated with them in any way.
## Workflow
Source image -> Google Cloud OCR -> Cloud Translation API -> JSON file(s) containing source and translated text

Human input: final touching -> edit into speech bubbles -> profit
## Install and use
Get a Google Cloud API which have access to both GCOCR and CTAPI.

Create a virtual environment if possible. Run: ` pip install -r requirements.txt `

Run the Python file.

## Usage
### Input Cloud Credentials
- Windows (PS): $env:GOOGLE_APPLICATION_CREDENTIALS = "D:\path\to\your-key.json"
- mac/Linux: export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"

### Command options
Basic use: `python main.py --input-dir INPUT --output-dir OUTPUT [options]`

Options:
- `--lang`
Target language for translation (default: en).

- `--format`
Image format extension (default: webp). Only files matching *.FORMAT are processed.

- `--padding`
Zero‑padding for output filenames (default: 2; so 1 → 01.json).

- `--horizontal-gap`
Max horizontal gap (pixels) between words on the same line before they are split into different chunks. Default: 30.

- `--vertical-gap`
Max vertical distance (pixels) to consider words on the same line. Default: 25.

- `--column-threshold`
Max horizontal difference (pixels) to be considered the same column (used in line grouping). Default: 20.

- `--group-mode`
Grouping strategy (default: lines).

  lines – traditional line + gap based grouping; good for documents or when bubbles are clearly separated.

  bubble – words are grouped into connected spatial regions (speech bubbles/boxes). Each region becomes one sentence.

- `--region-margin`
Pixel margin when merging words into regions in bubble mode (default: 15). Higher values merge more distant words into the same bubble.

- `--start`
First index to process (e.g. --start 5 → start at 05.webp).

- `--end`
Last index to process (inclusive). Useful for testing a subset.

- `--skip-existing`
If set, skip images whose corresponding JSON already exists in --output-dir.

- `--quiet` / `-q`
Minimal logging output.

## Development
- Chunky code. If anyone is willing to split to multiple files, please do.
- Need some kind of optimizations; I don't know which.
- Configure the script to take the API key from CLI instead of manually exporting.
