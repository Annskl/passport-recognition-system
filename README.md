#  Passport Recognition System

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-green.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

##  Overview
Automated system for Russian passport recognition and data extraction using computer vision and deep learning techniques. Developed as a Bachelor's thesis project.

## Features

- Automatically detects and crops Russian passport pages from scans or photos  
- Extracts key fields: full name, date of birth, place of birth, issue date, authority code, issuing authority  
- Crops and saves the passport holder’s photo  
- Corrects orientation and perspective distortions of the document  
- Recognizes text using a hybrid OCR approach (Tesseract + EasyOCR) with post-processing and fallback  
- Supports batch processing with a graphical interface (progress bar and detailed view)  
- Validates and auto-corrects recognized data to reduce errors  
- **Exports results** to multiple formats: Excel (`.xlsx`), CSV, JSON, and SQLite  
- Works fully offline on Windows


##  Technologies Used
- **Deep Learning:** YOLOv8, PyTorch
- **OCR:** Tesseract, EasyOCR (with custom fine-tuning)
- **Data Annotation:** Roboflow
- **GUI:** PyQt5
- **Language:** Python 3.11

##  System Output Examples

### Detection and Recognition Results
![result program](images/result_program.png)
![result program detailed view](images/result_program_detailed_view.png)


##  Research Contributions
1. Developed a hybrid approach combining classical CV and deep learning
2. Created custom dataset with 500+ annotated passport images
3. Implemented fallback mechanisms for robust performance
4. Achieved 72% field extraction even on low-quality images

##  Documentation
Full thesis available: [Bachelor_Thesis_Passport_Processing.pdf](docs/Bachelor_Thesis_Passport_Processing.pdf)
Presentation of project: [Bachelor_Presentation_Passport_Processing.pdf](docs/Bachelor_Presentation_Passport_Processing.pdf)


---
*This project was developed for educational purposes. All passport images used are synthetic or properly anonymized.*
