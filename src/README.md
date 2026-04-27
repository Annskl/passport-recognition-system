# Source code

Source code of the desktop application for Russian passport recognition.

## Structure

| File | Purpose |
|---|---|
| `main_app.py` | Entry point: dependency and model checks, GUI launch |
| `passport_gui.py` | PyQt5 interface (image loading, results table, export) |
| `passport_processor.py` | Pipeline: YOLOv8 → orientation → field detection → OCR → validation |
| `passport_utils.py` | Config, export to Excel/CSV/JSON/SQLite, database comparison |
| `PassportProcessor.spec` | PyInstaller config for building the `.exe` |
| `config.json` | Model paths (relative) |
| `requirements.txt` | Python dependencies |

## Running from source

Requirements: Python 3.11, [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
