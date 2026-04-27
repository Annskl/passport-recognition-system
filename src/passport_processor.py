"""Распознавание паспортов РФ: детекция страниц/полей и OCR."""

import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import easyocr
import pytesseract
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from math import sin, cos, radians, fabs

def _find_tesseract_path() -> str:
    """Ищет исполняемый файл Tesseract в стандартных местах установки"""
    candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return "tesseract"

pytesseract.pytesseract.tesseract_cmd = _find_tesseract_path()


@dataclass
class PassportData:
    """Структура для хранения данных паспорта"""
    surname: str = ""
    name: str = ""
    patronymic: str = ""
    birth_date: str = ""
    birth_place: str = ""
    passport_number: str = ""
    face_image: Optional[np.ndarray] = None
    page2_image: Optional[np.ndarray] = None
    original_image: Optional[np.ndarray] = None
    confidence_scores: Dict[str, float] = None
    processing_errors: List[str] = None
    file_name: str = ""

    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.processing_errors is None:
            self.processing_errors = []


class PassportProcessor:
    """Главный класс для обработки паспортов"""

    def __init__(self, models_config: Dict[str, str]):
        """
        Инициализация процессора

        Args:
            models_config: Словарь с путями к моделям
        """
        print("Инициализация процессора паспортов...")

        # Загрузка YOLO моделей
        self.page2_model = YOLO(models_config['page2_seg_model'])
        self.fields_model = YOLO(models_config['fields_detect_model'])
        self.number_model = YOLO(models_config['passport_number_model'])

        # Инициализация OCR
        self.easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=False)

        # Инициализация дообученного EasyOCR для fallback
        if 'finetuned_easyocr_model' in models_config:
            self.finetuned_easyocr = easyocr.Reader(
                ['ru', 'en'],
                gpu=False,
                model_storage_directory=models_config['finetuned_easyocr_model']
            )
        else:
            self.finetuned_easyocr = self.easyocr_reader  # Используем обычный если нет дообученного

        # Маппинг классов для модели полей
        self.field_class_mapping = {
            0: 'birth_date',
            1: 'birth_place',
            2: 'face',
            3: 'name',
            4: 'patronymic',
            5: 'surname'
        }

        # Регулярные выражения и валидаторы
        self._init_validators()

        # Загрузка словарей для улучшения OCR
        if 'ocr_training_data' in models_config:
            self._load_ocr_vocabularies(models_config['ocr_training_data'])

        print("Процессор готов к работе")

    def _init_validators(self):
        """Инициализация валидаторов и регулярных выражений"""

        # Гласные и согласные
        self.vowels = 'АЕЁИОУЫЭЮЯ'
        self.consonants = 'БВГДЖЗЙКЛМНПРСТФХЦЧШЩ'

        # Допустимые сочетания из 3 согласных
        self.allowed_triple_consonants = ['НКТ', 'НДР', 'ДСЛ', 'РКГ', 'ДЖД', 'СТК', 'ВСТ', 'МРК']

        # Паттерны для валидации
        self.validators = {
            'surname': {
                'pattern': r'^[А-ЯЁ][А-ЯЁа-яё\-]{1,19}$',
                'forbidden': ['ФАМИЛИЯ', 'ИМЯ', 'ОТЧЕСТВО', 'МУЖ', 'ЖЕН'],
                'forbidden_starts': ['Ь', 'Ъ', 'Ы'],
                'min_length': 2,
                'max_length': 16
            },
            'name': {
                'pattern': r'^[А-ЯЁ][А-ЯЁа-яё\-]{1,19}$',
                'forbidden': ['ИМЯ', 'ФАМИЛИЯ', 'ОТЧЕСТВО', 'МУЖ', 'ЖЕН', 'ПОЛ', 'РФ'],
                'forbidden_starts': ['Ь', 'Ъ', 'Ы'],
                'forbidden_endings': ['ИЧ', 'ВНА', 'ОВИЧ', 'ЕВИЧ', 'ОВНА', 'ЕВНА'],
                'min_length': 2,
                'max_length': 16
            },
            'patronymic': {
                'pattern': r'^[А-ЯЁ][А-ЯЁа-яё]{1,18}(ИЧ|ОВИЧ|ЕВИЧ|ЬЕВИЧ|ВНА|ОВНА|ЕВНА|ЬЕВНА|ИЧНА|ИНИЧНА)$',
                'required_endings': ['ИЧ', 'ОВИЧ', 'ЕВИЧ', 'ЬЕВИЧ', 'ВНА', 'ОВНА', 'ЕВНА', 'ЬЕВНА', 'ИЧНА', 'ИНИЧНА'],
                'forbidden': ['ОТЧЕСТВО', 'ФАМИЛИЯ', 'ИМЯ'],
                'forbidden_starts': ['Ь', 'Ъ', 'Ы'],
                'min_length': 2,
                'max_length': 16
            },
            'birth_date': {
                'pattern': r'^\d{2}\.\d{2}\.\d{4}$',
                'validator': self._validate_date
            },
            'birth_place': {
                'pattern': r'^[А-ЯЁГ][А-ЯЁа-яё\s\-\.\,\d]*$',
                'forbidden': ['МЕСТО', 'РОЖДЕНИЯ', 'ДАТА', 'РОССИЯ', 'РАЙОНА', 'ЛЕНИНСКОГО', 'СОВЕТСКОГО',
                            'ОТЧЕСТВО', 'ОБЛ', 'ДАГА', 'РОР'],
                'prefixes': ['Г', 'ГОР', 'ГОРОД', 'П', 'ПОС', 'ПОСЕЛОК', 'ПОСЁЛОК',
                           'С', 'СЕЛО', 'СТ', 'СТАНИЦА', 'Х', 'ХУТ', 'ХУТОР',
                           'Д', 'ДЕР', 'ДЕРЕВНЯ', 'РЕСП', 'ОБЛ', 'КР', 'КРАЙ',
                           'Р-Н', 'РАЙОН', 'АО', 'АОБЛ'],
                'min_length': 3,
                'max_length': 30,  # Увеличил для длинных названий
                'max_spaces': 3
            },
            'passport_number': {
                'pattern': r'^\d{10}$',
                'alt_pattern': r'^\d{2}\s?\d{2}\s?\d{6}$'
            }
        }

        # Замены латиницы, визуально похожей на кириллицу
        self.char_replacements = {
            'A': 'А', 'E': 'Е', 'O': 'О', 'P': 'Р', 'C': 'С', 'Y': 'У', 'X': 'Х',
            'H': 'Н', 'K': 'К', 'M': 'М', 'T': 'Т', 'B': 'В'
        }

    def _validate_date(self, date_str: str) -> bool:
        """Валидация даты"""
        try:
            parts = date_str.split('.')
            if len(parts) != 3:
                return False

            day, month, year = map(int, parts)

            # Базовые проверки
            if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= datetime.now().year):
                return False

            # Проверка корректности даты
            datetime(year, month, day)
            return True
        except:
            return False

    def _load_ocr_vocabularies(self, training_path: str):
        """Загрузка словарей из обучающих данных для улучшения OCR"""
        self.ocr_vocabularies = {}
        training_path = Path(training_path)

        field_mapping = {
            'surname': 'surname',
            'name': 'name',
            'patronymic': 'patronymic',
            'birth_date': 'birth_date',
            'birth_place': 'birth_place'
        }

        for field_type, folder_name in field_mapping.items():
            labels_folder = training_path / folder_name / 'labels'
            vocabulary = set()

            if labels_folder.exists():
                for txt_file in labels_folder.glob('*.txt'):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:
                            vocabulary.add(text)

            self.ocr_vocabularies[field_type] = vocabulary

    def process_image(self, image_path: str) -> PassportData:
        """
        Главный метод обработки изображения паспорта

        Args:
            image_path: Путь к изображению

        Returns:
            PassportData с извлеченными данными
        """
        result = PassportData()
        result.file_name = Path(image_path).name

        try:
            # Загружаем изображение
            image = cv2.imread(image_path)
            if image is None:
                result.processing_errors.append(f"Не удалось загрузить изображение: {image_path}")
                return result

            result.original_image = image.copy()

            # 1. Находим и извлекаем 2-ю страницу паспорта
            page2_cropped = self._extract_page2(image)
            if page2_cropped is None:
                result.processing_errors.append("Не найдена 2-я страница паспорта")
                return result

            # 2. Определяем правильную ориентацию
            page2_oriented = self._ensure_correct_orientation(page2_cropped)
            result.page2_image = page2_oriented.copy()

            # 3. Находим поля на странице
            fields_data = self._detect_and_extract_fields(page2_oriented)

            # 4. Обрабатываем текстовые поля
            for field_name, field_info in fields_data.items():
                if field_name == 'face' and field_info['image'] is not None:
                    result.face_image = field_info['image']
                else:
                    text = field_info.get('text', '')
                    setattr(result, field_name, text)
                    result.confidence_scores[field_name] = field_info.get('confidence', 0.0)

            # 5. Находим номер паспорта
            rotated_90 = cv2.rotate(page2_oriented, cv2.ROTATE_90_COUNTERCLOCKWISE)
            passport_number = self._extract_passport_number(rotated_90)
            result.passport_number = passport_number['text']
            result.confidence_scores['passport_number'] = passport_number['confidence']

            # 6. Проверяем дубликаты в ФИО
            self._check_and_fix_duplicates(result, fields_data)

            # 7. Постобработка даты
            if result.birth_date:
                result.birth_date = self._postprocess_date(result.birth_date)

            # 8. Fallback OCR для пустых полей с ограничением по областям
            self._smart_ocr_fallback_limited(page2_oriented, result, fields_data)

            # 9. Проверка места рождения на совпадение с ФИО
            self._validate_birth_place(result, fields_data)

            # 10. Финальная валидация места рождения
            self._final_validate_birth_place(result)

        except Exception as e:
            result.processing_errors.append(f"Ошибка обработки: {str(e)}")

        return result

    def _extract_page2(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Извлечение 2-й страницы паспорта с выравниванием"""

        # Детекция маски страницы
        seg = self.page2_model.predict(image, imgsz=640, conf=0.35, verbose=False)[0]
        if not seg.masks:
            return None

        H, W = image.shape[:2]

        # Получаем маску
        mask = (seg.masks.data.sum(dim=0).cpu().numpy() > 0).astype(np.uint8) * 255
        mask = cv2.resize(mask, (W, H), cv2.INTER_NEAREST)

        # Находим контур
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        angle = rect[-1]

        # Приведение к горизонтали
        if rect[1][0] < rect[1][1]:
            angle += 90

        # Поворачиваем изображение и маску
        img_rot, M = self._rotate_bound(image, angle)
        mask_rot, _ = self._rotate_bound(mask, angle)

        # Находим новый контур на повернутой маске
        cnts, _ = cv2.findContours(mask_rot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None

        cnt_rot = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt_rot)

        # Обрезаем
        crop = img_rot[y:y+h, x:x+w]

        return crop

    def _rotate_bound(self, image: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Поворот изображения без обрезки"""
        (h, w) = image.shape[:2]
        angle_rad = radians(angle)

        # Вычисляем новые размеры
        nw = int(abs(sin(angle_rad)) * h + abs(cos(angle_rad)) * w)
        nh = int(abs(sin(angle_rad)) * w + abs(cos(angle_rad)) * h)

        # Матрица поворота с центром в середине изображения
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)

        # Корректируем смещение
        M[0, 2] += (nw - w) / 2
        M[1, 2] += (nh - h) / 2

        # Применяем поворот
        rotated = cv2.warpAffine(image, M, (nw, nh),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

        return rotated, M

    def _ensure_correct_orientation(self, image: np.ndarray) -> np.ndarray:
        """Определение правильной ориентации страницы"""

        print("\n=== Проверка ориентации ===")

        # Инициализируем OpenCV детектор лица
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        orientations = [
            (image, "0°"),
            (cv2.rotate(image, cv2.ROTATE_180), "180°")
        ]

        best_orientation = None
        best_score = -1
        best_image = None
        best_fields_count = 0

        for img, angle in orientations:
            # YOLO детекция с conf=0.4
            results = self.fields_model.predict(img, imgsz=512, conf=0.4, verbose=False)[0]
            boxes = results.boxes

            if boxes is not None and len(boxes) > 0:
                num_fields = len(boxes)
                avg_conf = boxes.conf.mean().item()

                # Проверка наличия лица через YOLO
                face_detected_yolo = False
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
                    if cls == 2:  # класс 2 - это лицо
                        face_detected_yolo = True
                        break

                # Проверка лица через OpenCV
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                face_detected_cv = len(faces) > 0

                # Комплексная оценка
                score = num_fields * 10 + avg_conf * 100
                if face_detected_yolo:
                    score += 50
                if face_detected_cv:
                    score += 30

                # Штраф за отсутствие лица
                if not face_detected_yolo and not face_detected_cv:
                    score -= 50

                print(f"{angle}: полей={num_fields}, ср.conf={avg_conf:.3f}, "
                      f"YOLO лицо={face_detected_yolo}, OpenCV лицо={face_detected_cv}, score={score:.1f}")

                if score > best_score:
                    best_score = score
                    best_orientation = angle
                    best_image = img
                    best_fields_count = num_fields

        if best_image is not None:
            print(f"\n✓ Выбрана ориентация {best_orientation}")
            return best_image
        else:
            print("\n✓ Оставляем исходную ориентацию")
            return image

    def _detect_and_extract_fields(self, image: np.ndarray) -> Dict[str, Dict]:
        """Детекция и извлечение полей со страницы"""
        fields_data = {}

        # Детекция полей YOLO с conf=0.4
        results = self.fields_model.predict(image, imgsz=640, conf=0.4, verbose=False)

        if not results or results[0].boxes is None:
            return fields_data

        boxes = results[0].boxes

        # Группируем детекции по классам
        detections_by_class = {}

        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item()) if hasattr(boxes.cls[i], 'item') else int(boxes.cls[i])
            conf = boxes.conf[i].item() if hasattr(boxes.conf[i], 'item') else float(boxes.conf[i])
            box = boxes.xyxy[i].cpu().numpy()

            field_type = self.field_class_mapping.get(cls)
            if not field_type:
                continue

            if field_type not in detections_by_class:
                detections_by_class[field_type] = []

            detections_by_class[field_type].append({
                'box': box,
                'conf': conf
            })

        # Обрабатываем детекции для каждого класса
        for field_type, detections in detections_by_class.items():
            # Сортируем по confidence и берем лучшую
            detections.sort(key=lambda x: x['conf'], reverse=True)
            best_detection = detections[0]

            x1, y1, x2, y2 = map(int, best_detection['box'])

            # Добавляем небольшой отступ для OCR
            padding = 5
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)

            crop = image[y1:y2, x1:x2]

            if field_type == 'face':
                fields_data[field_type] = {
                    'image': crop,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(best_detection['conf'])
                }
            else:
                # Распознаем текст
                text_result = self._recognize_field_text(crop, field_type)

                fields_data[field_type] = {
                    'text': text_result['text'],
                    'bbox': (x1, y1, x2, y2),
                    'confidence': text_result['confidence'],
                    'ocr_method': text_result['method'],
                    'alternatives': [det for det in detections[1:] if det['conf'] > 0.4]  # Сохраняем альтернативы
                }

        return fields_data

    def _recognize_field_text(self, image: np.ndarray, field_type: str) -> Dict[str, any]:
        """Распознавание текста с приоритетом простого Tesseract без излишней обработки"""

        # Конвертируем в градации серого если нужно
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Сначала пробуем без предобработки — Tesseract часто лучше работает на оригинале
        tesseract_config = self._get_tesseract_config_simple(field_type)
        tesseract_text_raw = ""

        try:
            # Пробуем без обработки
            tesseract_text_raw = pytesseract.image_to_string(gray, config=tesseract_config).strip()

            # Очистка текста БЕЗ замены символов для первой попытки
            tesseract_text_clean = self._clean_text_simple(tesseract_text_raw, field_type)

            # Проверяем валидность
            if self._is_text_valid_simple(tesseract_text_clean, field_type):
                return {
                    'text': tesseract_text_clean,
                    'confidence': 0.95,
                    'method': 'tesseract_raw'
                }
        except:
            pass

        # Если простой метод не сработал, пробуем с обработкой
        processed = self._preprocess_for_ocr_minimal(gray, field_type)

        # Tesseract с обработкой
        tesseract_text = ""
        tesseract_valid = False

        try:
            tesseract_text = pytesseract.image_to_string(processed, config=tesseract_config).strip()
            tesseract_text = self._clean_text(tesseract_text, field_type)
            tesseract_valid = self._is_text_valid_for_ocr_switch(tesseract_text, field_type)
        except:
            pass

        # EasyOCR как запасной вариант
        easyocr_text = ""
        easyocr_conf = 0.0
        easyocr_valid = False

        try:
            if len(processed.shape) == 2:
                processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = processed

            easyocr_results = self.finetuned_easyocr.readtext(processed_rgb)

            if easyocr_results:
                # Берем результат с наибольшей confidence
                best_result = max(easyocr_results, key=lambda x: x[2])
                easyocr_text = best_result[1]
                easyocr_conf = best_result[2]

                # Очистка текста
                easyocr_text = self._clean_text(easyocr_text, field_type)
                easyocr_valid = self._is_text_valid_for_ocr_switch(easyocr_text, field_type)
        except:
            pass

        # Выбираем лучший результат
        # Приоритет: валидный сырой Tesseract > валидный обработанный Tesseract > валидный EasyOCR
        if tesseract_text_raw and self._is_text_valid_simple(tesseract_text_raw, field_type):
            return {
                'text': tesseract_text_clean,
                'confidence': 0.95,
                'method': 'tesseract_raw'
            }
        elif tesseract_valid and len(tesseract_text) > 3:
            return {
                'text': tesseract_text,
                'confidence': 0.9,
                'method': 'tesseract'
            }
        elif easyocr_valid and easyocr_conf > 0.5:
            return {
                'text': easyocr_text,
                'confidence': easyocr_conf,
                'method': 'finetuned_easyocr'
            }
        # Если все невалидны, берем наиболее длинный
        elif tesseract_text and len(tesseract_text) >= len(easyocr_text):
            return {
                'text': tesseract_text,
                'confidence': 0.7,
                'method': 'tesseract'
            }
        elif easyocr_text:
            return {
                'text': easyocr_text,
                'confidence': easyocr_conf,
                'method': 'finetuned_easyocr'
            }

        return {'text': '', 'confidence': 0.0, 'method': 'none'}

    def _preprocess_for_ocr_minimal(self, image: np.ndarray, field_type: str) -> np.ndarray:
        """Минимальная предобработка изображения для OCR"""

        # Только масштабирование, БЕЗ CLAHE!
        scale_factor = 2
        height, width = image.shape[:2]
        scaled = cv2.resize(image, (width * scale_factor, height * scale_factor),
                          interpolation=cv2.INTER_CUBIC)

        return scaled

    def _get_tesseract_config(self, field_type: str) -> str:
        """Получение конфигурации Tesseract для типа поля"""
        configs = {
            'surname': '-l rus --psm 8 -c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
            'name': '-l rus --psm 8 -c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
            'patronymic': '-l rus --psm 8 -c tessedit_char_whitelist=АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ',
            'birth_date': '--psm 8 -c tessedit_char_whitelist=0123456789.',
            'birth_place': '-l rus --psm 6',
            'passport_number': '--psm 8 -c tessedit_char_whitelist=0123456789'
        }
        return configs.get(field_type, '-l rus --psm 6')

    def _get_tesseract_config_simple(self, field_type: str) -> str:
        """Упрощенная конфигурация Tesseract без whitelist"""
        configs = {
            'surname': '-l rus --psm 8',
            'name': '-l rus --psm 8',
            'patronymic': '-l rus --psm 8',
            'birth_date': '--psm 8',
            'birth_place': '-l rus --psm 6',
            'passport_number': '--psm 8'
        }
        return configs.get(field_type, '-l rus --psm 6')

    def _clean_text_simple(self, text: str, field_type: str) -> str:
        """Простая очистка текста без агрессивных замен"""
        if not text:
            return ""

        cleaned = text.strip()

        if field_type == 'birth_date':
            # Для дат оставляем только цифры и точки
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            cleaned = cleaned.strip('.')
            cleaned = re.sub(r'\.{2,}', '.', cleaned)

        elif field_type in ['surname', 'name', 'patronymic']:
            # Убираем только явно не русские символы, сохраняя регистр
            # Сначала проверяем, есть ли русские буквы вообще
            if re.search(r'[А-ЯЁа-яё]', cleaned):
                # Убираем только не-буквы, сохраняя регистр исходного текста
                cleaned = re.sub(r'[^А-ЯЁа-яё\-]', '', cleaned)
                # Приводим к верхнему регистру
                cleaned = cleaned.upper()
                # Убираем дефисы в начале и конце
                cleaned = cleaned.strip('-')

        elif field_type == 'birth_place':
            # Русские буквы, цифры, дефисы, точки, пробелы
            cleaned = re.sub(r'[^А-ЯЁа-яё0-9\s\-\.\,]', '', cleaned).upper()
            cleaned = re.sub(r'^[\.\-\,]+', '', cleaned).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if cleaned.endswith('.'):
                cleaned = cleaned[:-1].strip()

        elif field_type == 'passport_number':
            # Только цифры
            cleaned = re.sub(r'[^\d]', '', cleaned)

        return cleaned.strip()

    def _replace_similar_chars(self, text: str) -> str:
        """Замена похожих символов - используется ТОЛЬКО при необходимости"""
        if not text:
            return text

        # Применяем замены только если в тексте есть латинские буквы
        if re.search(r'[A-Za-z]', text):
            for old, new in self.char_replacements.items():
                text = text.replace(old, new)

        return text

    def _clean_text(self, text: str, field_type: str) -> str:
        """Очистка распознанного текста с заменой символов"""
        if not text:
            return ""

        # Сначала заменяем похожие символы
        if field_type in ['surname', 'name', 'patronymic', 'birth_place']:
            text = self._replace_similar_chars(text)

        cleaned = text.strip()

        if field_type == 'birth_date':
            # Для дат оставляем только цифры и точки
            cleaned = re.sub(r'[^\d.]', '', cleaned)
            cleaned = cleaned.strip('.')
            cleaned = re.sub(r'\.{2,}', '.', cleaned)

        elif field_type in ['surname', 'name', 'patronymic']:
            # Только русские буквы и дефисы
            cleaned = re.sub(r'[^А-ЯЁа-яё\-]', '', cleaned).upper()
            # Убираем дефисы в начале и конце
            cleaned = cleaned.strip('-')

        elif field_type == 'birth_place':
            # Русские буквы, цифры, дефисы, точки, пробелы
            cleaned = re.sub(r'[^А-ЯЁа-яё0-9\s\-\.\,]', '', cleaned).upper()
            cleaned = re.sub(r'^[\.\-\,]+', '', cleaned).strip()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            if cleaned.endswith('.'):
                cleaned = cleaned[:-1].strip()

            # Обработка цифр - разрешаем только паттерн типа "ГОРОД-46"
            if any(c.isdigit() for c in cleaned):
                pattern = r'([А-ЯЁ]+)-(\d{1,2})$'  # Только 1-2 цифры после дефиса в конце
                if not re.search(pattern, cleaned):
                    # Если цифры есть, но не в правильном формате - удаляем их
                    cleaned = re.sub(r'\d', '', cleaned).strip()
                    cleaned = re.sub(r'\s+', ' ', cleaned)

        elif field_type == 'passport_number':
            # Только цифры
            cleaned = re.sub(r'[^\d]', '', cleaned)

        return cleaned.strip()

    def _is_text_valid_simple(self, text: str, field_type: str) -> bool:
        """Простая проверка валидности текста"""
        if not text or len(text) < 2:
            return False

        # Базовые проверки для ФИО
        if field_type in ['surname', 'name', 'patronymic']:
            # Должны быть только русские буквы
            if not re.match(r'^[А-ЯЁ][А-ЯЁ\-]*$', text):
                return False

            # Проверка длины
            if len(text) < 2 or len(text) > 16:
                return False

            # Проверка на запрещенные слова
            validator = self.validators.get(field_type, {})
            if text in validator.get('forbidden', []):
                return False

        elif field_type == 'birth_place':
            # Проверка на минимальную длину
            if len(text) < 3:
                return False

            # Проверка на запрещенные слова
            validator = self.validators.get(field_type, {})
            if text in validator.get('forbidden', []):
                return False

            # Не может быть только цифры
            if text.isdigit():
                return False

            # Не может быть только точка или точки
            if re.match(r'^\.+$', text):
                return False

        return True

    def _is_text_valid_for_ocr_switch(self, text: str, field_type: str) -> bool:
        """Проверка нужно ли переключаться на другой OCR"""
        if not text:
            return False

        # Если Tesseract дал <= 3 символа - нужен другой OCR
        if len(text) <= 3:
            return False

        # Проверка на 3 гласные подряд
        if field_type in ['surname', 'name', 'patronymic', 'birth_place']:
            for i in range(len(text) - 2):
                if all(c in self.vowels for c in text[i:i+3]):
                    return False

        # Проверка на 3 согласные подряд в имени
        if field_type == 'name':
            for i in range(len(text) - 2):
                triple = text[i:i+3]
                if all(c in self.consonants for c in triple) and triple not in self.allowed_triple_consonants:
                    return False

        # Проверка длины
        if field_type in ['surname', 'name', 'patronymic']:
            if len(text) > 16:
                return False
        elif field_type == 'birth_place':
            if len(text) > 30:  # Для места рождения разрешаем больше
                return False

        # Проверка на запрещенные значения
        validator = self.validators.get(field_type, {})
        if text in validator.get('forbidden', []):
            return False

        # Проверка для места рождения
        if field_type == 'birth_place':
            # Не может быть только префикс
            if text in self.validators['birth_place']['prefixes']:
                return False
            # Не может быть датой
            if re.match(r'^\d{2}\.\d{2}\.\d{4}$', text):
                return False
            # Проверка на количество пробелов
            space_count = text.count(' ')
            if space_count > 3:
                return False

        return True

    def _check_and_fix_duplicates(self, result: PassportData, fields_data: Dict[str, Dict]):
        """Проверка и исправление дубликатов в ФИО"""

        # Проверяем дубликаты
        fields = ['surname', 'name', 'patronymic']
        values = {}

        for field in fields:
            value = getattr(result, field, '')
            if value:
                if value in values:
                    # Найден дубликат!
                    other_field = values[value]

                    # Сравниваем confidence
                    conf1 = result.confidence_scores.get(field, 0)
                    conf2 = result.confidence_scores.get(other_field, 0)

                    if conf1 < conf2:
                        # Текущее поле имеет меньший confidence - очищаем его
                        setattr(result, field, '')
                        result.confidence_scores[field] = 0

                        # Пытаемся найти альтернативу из других детекций
                        if field in fields_data and 'alternatives' in fields_data[field]:
                            for alt in fields_data[field]['alternatives']:
                                # Распознаем текст в альтернативном боксе
                                x1, y1, x2, y2 = map(int, alt['box'])
                                crop = result.page2_image[y1:y2, x1:x2]

                                alt_text_result = self._recognize_field_text(crop, field)
                                alt_text = alt_text_result['text']

                                if alt_text and alt_text != value and alt_text not in values.values():
                                    setattr(result, field, alt_text)
                                    result.confidence_scores[field] = alt_text_result['confidence']
                                    break
                    else:
                        # Другое поле имеет меньший confidence - очищаем его
                        setattr(result, other_field, '')
                        result.confidence_scores[other_field] = 0
                else:
                    values[value] = field

    def _validate_birth_place(self, result: PassportData, fields_data: Dict[str, Dict]):
        """Проверка и исправление места рождения"""

        birth_place = result.birth_place

        if not birth_place:
            return

        # Проверка на совпадение с ФИО
        fio_fields = ['surname', 'name', 'patronymic']
        for field in fio_fields:
            field_value = getattr(result, field, '')
            if field_value and birth_place == field_value:
                print(f"Место рождения совпадает с {field}: {birth_place}")

                # Очищаем место рождения
                result.birth_place = ''
                result.confidence_scores['birth_place'] = 0

                # Пытаемся найти альтернативу
                if 'birth_place' in fields_data and 'alternatives' in fields_data['birth_place']:
                    for alt in fields_data['birth_place']['alternatives']:
                        # Распознаем текст в альтернативном боксе
                        x1, y1, x2, y2 = map(int, alt['box'])
                        crop = result.page2_image[y1:y2, x1:x2]

                        alt_text_result = self._recognize_field_text(crop, 'birth_place')
                        alt_text = alt_text_result['text']

                        # Проверяем что альтернатива не совпадает с ФИО
                        if alt_text and alt_text not in [getattr(result, f, '') for f in fio_fields]:
                            # Проверяем на запрещенные слова
                            if alt_text not in self.validators['birth_place']['forbidden']:
                                result.birth_place = alt_text
                                result.confidence_scores['birth_place'] = alt_text_result['confidence']
                                print(f"Найдена альтернатива для места рождения: {alt_text}")
                                break

                return

    def _final_validate_birth_place(self, result: PassportData):
        """Финальная валидация места рождения"""

        if not result.birth_place:
            return

        birth_place = result.birth_place

        # Проверка на дату (не должно быть датой рождения)
        if re.match(r'^\d{2}\.\d{2}\.\d{4}$', birth_place):
            result.birth_place = ''
            result.confidence_scores['birth_place'] = 0
            return

        # Проверка на слишком короткое значение
        if len(birth_place) < 3:
            result.birth_place = ''
            result.confidence_scores['birth_place'] = 0
            return

        # Проверка на запрещенные слова
        if birth_place in self.validators['birth_place']['forbidden']:
            result.birth_place = ''
            result.confidence_scores['birth_place'] = 0
            return

        # Проверка на только цифры
        if birth_place.isdigit():
            result.birth_place = ''
            result.confidence_scores['birth_place'] = 0
            return

        # Исправление распространенных ошибок
        # РОР. -> ГОР.
        if birth_place.startswith('РОР'):
            birth_place = birth_place.replace('РОР', 'ГОР', 1)
            result.birth_place = birth_place

        # Убираем двойные точки
        birth_place = re.sub(r'\.{2,}', '.', birth_place)
        result.birth_place = birth_place

    def _postprocess_date(self, date_str: str) -> str:
        """Постобработка даты - приведение к формату XX.XX.XXXX"""
        if not date_str:
            return ""

        # Убираем все кроме цифр
        digits = re.sub(r'\D', '', date_str)

        # Если 8 цифр - форматируем
        if len(digits) == 8:
            return f"{digits[:2]}.{digits[2:4]}.{digits[4:]}"

        # Если уже в правильном формате
        if re.match(r'^\d{2}\.\d{2}\.\d{4}$', date_str):
            return date_str

        # Пытаемся исправить формат с пробелами
        date_with_spaces = re.sub(r'\s+', '.', date_str)
        date_with_spaces = re.sub(r'\.+', '.', date_with_spaces)
        date_with_spaces = date_with_spaces.strip('.')

        if re.match(r'^\d{2}\.\d{2}\.\d{4}$', date_with_spaces):
            return date_with_spaces

        return ""

    def _extract_passport_number(self, image: np.ndarray) -> Dict[str, any]:
        """Извлечение номера паспорта"""

        # Детекция области с номером
        results = self.number_model.predict(image, conf=0.35, verbose=False)

        if not results or not results[0].boxes:
            # Fallback - сканируем весь документ
            return self._scan_for_passport_number(image)

        # Берем первый найденный бокс
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Расширяем область для лучшего распознавания
        padding = 10
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = image[y1:y2, x1:x2]

        # Распознаем номер
        number_text = self._recognize_passport_number(crop)

        return {
            'text': number_text or '',
            'confidence': 0.9 if number_text else 0.0
        }

    def _recognize_passport_number(self, image: np.ndarray) -> Optional[str]:
        """Распознавание номера паспорта в области"""

        # Минимальная предобработка
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Пробуем сначала без обработки
        try:
            text = pytesseract.image_to_string(gray, config='--psm 8').strip()
            number = self._find_passport_number_in_text(text)
            if number:
                return number
        except:
            pass

        # Если не получилось - с масштабированием
        scaled = cv2.resize(gray, (gray.shape[1] * 2, gray.shape[0] * 2), interpolation=cv2.INTER_CUBIC)

        try:
            text = pytesseract.image_to_string(scaled, config='--psm 8').strip()
            number = self._find_passport_number_in_text(text)
            if number:
                return number
        except:
            pass

        # Если не получилось - EasyOCR
        try:
            rgb = cv2.cvtColor(scaled, cv2.COLOR_GRAY2RGB)
            results = self.finetuned_easyocr.readtext(rgb)
            if results:
                text = ' '.join([r[1] for r in results])
                return self._find_passport_number_in_text(text)
        except:
            pass

        return None

    def _find_passport_number_in_text(self, text: str) -> Optional[str]:
        """Поиск номера паспорта в тексте"""

        # Убираем все кроме цифр и пробелов
        digits_only = re.sub(r'[^\d\s]', ' ', text)

        # Ищем последовательность из 10 цифр
        patterns = [
            r'(\d{4})\s*(\d{6})',  # 4 + 6 цифр
            r'(\d{2})\s*(\d{2})\s*(\d{6})',  # 2 + 2 + 6 цифр
            r'(\d{10})',  # 10 цифр подряд
        ]

        for pattern in patterns:
            match = re.search(pattern, digits_only)
            if match:
                # Собираем все группы
                number = ''.join(match.groups())
                if len(number) == 10:
                    return number

        # Если не нашли, пробуем найти любые 10 цифр подряд
        all_digits = re.sub(r'\D', '', text)
        if len(all_digits) >= 10:
            return all_digits[:10]

        return None

    def _scan_for_passport_number(self, image: np.ndarray) -> Dict[str, any]:
        """Сканирование всего изображения в поисках номера паспорта"""

        try:
            # Конвертируем в RGB для EasyOCR
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Полное сканирование EasyOCR
            results = self.finetuned_easyocr.readtext(image_rgb)
            full_text = ' '.join([text for _, text, _ in results])

            number = self._find_passport_number_in_text(full_text)

            return {
                'text': number or '',
                'confidence': 0.7 if number else 0.0
            }
        except:
            return {'text': '', 'confidence': 0.0}

    def _smart_ocr_fallback_limited(self, image: np.ndarray, result: PassportData, fields_data: Dict[str, Dict]):
        """Умный fallback для пустых полей С ОГРАНИЧЕНИЕМ ПО ОБЛАСТЯМ"""

        print("\n=== Smart OCR Fallback (Limited) ===")

        # Список полей для проверки
        text_fields = ['surname', 'name', 'patronymic', 'birth_date', 'birth_place']

        # Проверяем какие поля пустые или некорректные
        empty_fields = []
        for field in text_fields:
            value = getattr(result, field, '')
            if not value or not self._is_text_valid_for_ocr_switch(value, field):
                empty_fields.append(field)
            # Специальная проверка для места рождения
            elif field == 'birth_place':
                # Проверяем на совпадение с ФИО
                if value in [result.surname, result.name, result.patronymic]:
                    empty_fields.append(field)
                # Проверяем на запрещенные слова
                elif value in self.validators['birth_place']['forbidden']:
                    empty_fields.append(field)

        if not empty_fields:
            print("Все поля заполнены корректно")
            return

        print(f"Пустые/некорректные поля: {empty_fields}")

        # Для каждого пустого поля ищем в ОГРАНИЧЕННОЙ области
        for field in empty_fields:
            if field not in fields_data:
                continue

            # Получаем область поля
            field_bbox = fields_data[field].get('bbox')
            if not field_bbox:
                continue

            x1, y1, x2, y2 = field_bbox

            # Расширяем область поиска на 20%
            width = x2 - x1
            height = y2 - y1
            expand_x = int(width * 0.2)
            expand_y = int(height * 0.2)

            x1 = max(0, x1 - expand_x)
            y1 = max(0, y1 - expand_y)
            x2 = min(image.shape[1], x2 + expand_x)
            y2 = min(image.shape[0], y2 + expand_y)

            # Вырезаем расширенную область
            search_area = image[y1:y2, x1:x2]

            try:
                if len(search_area.shape) == 2:
                    search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_GRAY2RGB)
                else:
                    search_area_rgb = cv2.cvtColor(search_area, cv2.COLOR_BGR2RGB)

                # OCR только в этой области
                ocr_results = self.finetuned_easyocr.readtext(search_area_rgb)

                best_candidate = None
                best_score = 0

                for bbox, text, conf in ocr_results:
                    if conf < 0.3:
                        continue

                    cleaned_text = self._clean_text_simple(text, field)

                    if not cleaned_text:
                        continue

                    # Проверяем валидность для поля
                    if not self._is_text_valid_simple(cleaned_text, field):
                        continue

                    # Проверяем что текст не является запрещенным словом
                    validator = self.validators.get(field, {})
                    if cleaned_text in validator.get('forbidden', []):
                        continue

                    # Специальные проверки для места рождения
                    if field == 'birth_place':
                        # Не должно совпадать с ФИО
                        if cleaned_text in [result.surname, result.name, result.patronymic]:
                            continue
                        # Не должно быть датой
                        if re.match(r'^\d{2}\.\d{2}\.\d{4}$', cleaned_text):
                            continue

                    # Оценка кандидата
                    score = conf

                    # Дополнительные проверки для конкретных полей
                    if field == 'patronymic' and 'required_endings' in validator:
                        if any(cleaned_text.endswith(ending) for ending in validator['required_endings']):
                            score *= 1.5

                    if score > best_score:
                        best_score = score
                        best_candidate = cleaned_text

                # Применяем найденное значение
                if best_candidate and best_score > 0.5:
                    print(f"✓ Найдено значение для {field}: {best_candidate} (score: {best_score:.2f})")
                    setattr(result, field, best_candidate)
                    result.confidence_scores[field] = best_score
                else:
                    print(f"✗ Не найдено подходящее значение для {field}")

            except Exception as e:
                print(f"Ошибка в Smart OCR Fallback для {field}: {str(e)}")
                result.processing_errors.append(f"Smart OCR Fallback error for {field}: {str(e)}")