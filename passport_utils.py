"""Утилиты и конфигурация для обработки паспортов."""

import os
import sys
import json
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
from datetime import datetime


class Config:
    """Конфигурация приложения"""

    DEFAULT_MODELS = {
        'page2_seg_model': 'models/page2_seg_model/best.pt',
        'fields_detect_model': 'models/fields_detect_model/best.pt',
        'passport_number_model': 'models/passport_number_model/best.pt',
        'ocr_training_data': 'models/ocr_training'
    }

    # Поддерживаемые форматы изображений
    SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']

    # Настройки экспорта
    EXPORT_FORMATS = {
        'excel': {'extension': '.xlsx', 'name': 'Excel'},
        'sqlite': {'extension': '.db', 'name': 'SQLite Database'},
        'json': {'extension': '.json', 'name': 'JSON'},
        'csv': {'extension': '.csv', 'name': 'CSV'}
    }

    # Поля для отображения в таблице
    TABLE_FIELDS = {
        'original_image': 'Исходное фото',
        'page2_image': 'Фото 2 страницы',
        'face_image': 'Фото лица',
        'surname': 'Фамилия',
        'name': 'Имя',
        'patronymic': 'Отчество',
        'birth_date': 'Дата рождения',
        'birth_place': 'Место рождения',
        'passport_number': 'Номер паспорта'
    }

    @classmethod
    def get_base_dir(cls) -> str:
        """Возвращает базовую директорию: папка с данными EXE или папка скрипта.
        В PyInstaller 6.x файлы данных кладутся в _internal/, на которую
        указывает sys._MEIPASS, поэтому используем её, а не папку exe."""
        if getattr(sys, 'frozen', False):
            return sys._MEIPASS
        return os.path.dirname(os.path.abspath(__file__))

    @classmethod
    def load_config(cls, base_dir: str = None, config_path: str = "config.json") -> Dict:
        """Загрузка конфигурации из файла с разрешением путей относительно base_dir"""
        if base_dir is None:
            base_dir = cls.get_base_dir()

        full_config_path = os.path.join(base_dir, config_path)

        if os.path.exists(full_config_path):
            try:
                with open(full_config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                for key in cls.DEFAULT_MODELS:
                    if key not in loaded_config:
                        loaded_config[key] = cls.DEFAULT_MODELS[key]
            except Exception as e:
                print(f"Ошибка загрузки конфигурации: {e}")
                loaded_config = dict(cls.DEFAULT_MODELS)
        else:
            loaded_config = dict(cls.DEFAULT_MODELS)

        model_keys = ['page2_seg_model', 'fields_detect_model', 'passport_number_model', 'ocr_training_data']
        for key in model_keys:
            if key in loaded_config and not os.path.isabs(loaded_config[key]):
                loaded_config[key] = os.path.join(base_dir, loaded_config[key])

        return loaded_config

    @classmethod
    def save_config(cls, config: Dict, config_path: str = "config.json"):
        """Сохранение конфигурации в файл"""
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Ошибка сохранения конфигурации: {e}")


class DataExporter:
    """Класс для экспорта данных в различные форматы"""

    @staticmethod
    def export_to_excel(data: List[Dict], output_path: str, selected_fields: List[str]):
        """Экспорт в Excel"""
        # Подготавливаем данные для DataFrame
        df_data = []

        for item in data:
            row = {}
            for field in selected_fields:
                if field == 'file_name':
                    # Для поля file_name используем прямое название
                    row['Файл'] = item.get(field, '')
                elif field in ['original_image', 'page2_image', 'face_image']:
                    # Для изображений сохраняем путь или пометку
                    row[Config.TABLE_FIELDS[field]] = 'Изображение' if field in item and item[field] is not None else ''
                else:
                    row[Config.TABLE_FIELDS[field]] = item.get(field, '')
            df_data.append(row)

        # Создаем DataFrame и сохраняем
        df = pd.DataFrame(df_data)
        df.to_excel(output_path, index=False, engine='openpyxl')

    @staticmethod
    def export_to_sqlite(data: List[Dict], output_path: str, selected_fields: List[str]):
        """Экспорт в SQLite"""
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()

        # Создаем таблицу
        fields_sql = []
        for field in selected_fields:
            if field in ['original_image', 'page2_image', 'face_image']:
                fields_sql.append(f"{field} BLOB")
            else:
                fields_sql.append(f"{field} TEXT")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS passports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            {', '.join(fields_sql)},
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_sql)

        # Вставляем данные
        for item in data:
            values = []
            for field in selected_fields:
                if field in ['original_image', 'page2_image', 'face_image']:
                    # Конвертируем изображение в байты
                    if field in item and item[field] is not None:
                        _, buffer = cv2.imencode('.png', item[field])
                        values.append(buffer.tobytes())
                    else:
                        values.append(None)
                else:
                    values.append(item.get(field, ''))

            placeholders = ', '.join(['?' for _ in selected_fields])
            insert_sql = f"INSERT INTO passports ({', '.join(selected_fields)}) VALUES ({placeholders})"
            cursor.execute(insert_sql, values)

        conn.commit()
        conn.close()

    @staticmethod
    def export_to_json(data: List[Dict], output_path: str, selected_fields: List[str]):
        """Экспорт в JSON"""
        export_data = []

        for item in data:
            row = {}
            for field in selected_fields:
                if field in ['original_image', 'page2_image', 'face_image']:
                    # Для изображений можно сохранить base64 или пропустить
                    row[field] = None  # Или конвертировать в base64 если нужно
                else:
                    row[field] = item.get(field, '')
            export_data.append(row)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_to_csv(data: List[Dict], output_path: str, selected_fields: List[str]):
        """Экспорт в CSV"""
        # Аналогично Excel, но в CSV
        df_data = []

        for item in data:
            row = {}
            for field in selected_fields:
                if field == 'file_name':
                    # Для поля file_name используем прямое название
                    row['Файл'] = item.get(field, '')
                elif field in ['original_image', 'page2_image', 'face_image']:
                    row[Config.TABLE_FIELDS[field]] = 'Изображение' if field in item and item[field] is not None else ''
                else:
                    row[Config.TABLE_FIELDS[field]] = item.get(field, '')
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    @staticmethod
    def export_faces(data: List[Dict], output_folder: str):
        """Экспорт фотографий лиц в отдельную папку"""
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, item in enumerate(data):
            if 'face_image' in item and item['face_image'] is not None:
                # Генерируем имя файла
                surname = item.get('surname', '')
                name = item.get('name', '')
                patronymic = item.get('patronymic', '')
                
                if surname or name:
                    filename = f"{surname}_{name}_{patronymic}_{i+1}.jpg"
                else:
                    filename = f"face_{i+1}.jpg"
                
                # Очищаем имя файла от недопустимых символов
                filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-', '.'))
                
                # Сохраняем изображение
                cv2.imwrite(str(output_path / filename), item['face_image'])


class DataComparator:
    """Класс для сравнения данных с базой"""
    
    @staticmethod
    def load_comparison_data(file_path: str) -> List[Dict]:
        """Загрузка данных для сравнения"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.xlsx', '.xls']:
            # Загрузка из Excel
            df = pd.read_excel(file_path)
            return df.to_dict('records')
        
        elif file_ext == '.db':
            # Загрузка из SQLite
            conn = sqlite3.connect(file_path)
            # Получаем имя первой таблицы
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            if tables:
                table_name = tables[0][0]
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                conn.close()
                return df.to_dict('records')
            conn.close()
            return []
        
        elif file_ext == '.csv':
            # Загрузка из CSV
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            return df.to_dict('records')
        
        elif file_ext == '.json':
            # Загрузка из JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_ext}")
    
    @staticmethod
    def compare_data(passport_data: Dict, comparison_data: List[Dict]) -> Dict:
        """
        Сравнение данных паспорта с базой
        
        Returns:
            Dict с результатами сравнения:
            - 'match_found': bool - найдено ли совпадение
            - 'matched_record': Dict - запись из базы
            - 'match_score': float - степень совпадения (0-1)
            - 'differences': List[str] - список различий
        """
        best_match = None
        best_score = 0
        
        for record in comparison_data:
            score = 0
            matches = 0
            total_fields = 0
            
            # Сравниваем основные поля
            fields_to_compare = ['surname', 'name', 'patronymic', 'birth_date', 'passport_number']
            
            for field in fields_to_compare:
                if field in passport_data and field in record:
                    total_fields += 1
                    passport_value = str(passport_data.get(field, '')).strip().upper()
                    record_value = str(record.get(field, '')).strip().upper()
                    
                    if passport_value and record_value:
                        if passport_value == record_value:
                            matches += 1
                            score += 1
                        else:
                            # Частичное совпадение для ФИО
                            if field in ['surname', 'name', 'patronymic']:
                                # Проверка схожести строк
                                similarity = DataComparator._string_similarity(passport_value, record_value)
                                if similarity > 0.8:
                                    matches += 0.5
                                    score += similarity
            
            if total_fields > 0:
                match_score = score / total_fields
                if match_score > best_score:
                    best_score = match_score
                    best_match = record
        
        # Анализируем результаты
        if best_match and best_score > 0.7:  # Порог совпадения 70%
            differences = []
            
            for field in fields_to_compare:
                passport_value = str(passport_data.get(field, '')).strip()
                record_value = str(best_match.get(field, '')).strip()
                
                if passport_value != record_value:
                    differences.append(f"{field}: '{passport_value}' != '{record_value}'")
            
            return {
                'match_found': True,
                'matched_record': best_match,
                'match_score': best_score,
                'differences': differences
            }
        
        return {
            'match_found': False,
            'matched_record': None,
            'match_score': 0,
            'differences': []
        }
    
    @staticmethod
    def _string_similarity(s1: str, s2: str) -> float:
        """Вычисление схожести строк (простой алгоритм)"""
        if not s1 or not s2:
            return 0.0
        
        # Простое сравнение по совпадающим символам
        matches = sum(c1 == c2 for c1, c2 in zip(s1, s2))
        return matches / max(len(s1), len(s2))


class ImageUtils:
    """Утилиты для работы с изображениями"""
    
    @staticmethod
    def resize_image_for_display(image: np.ndarray, max_width: int = 200, max_height: int = 200) -> np.ndarray:
        """Изменение размера изображения для отображения в интерфейсе"""
        if image is None:
            return None
        
        h, w = image.shape[:2]
        
        # Вычисляем коэффициент масштабирования
        scale = min(max_width / w, max_height / h)
        
        if scale < 1:
            new_width = int(w * scale)
            new_height = int(h * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return image
    
    @staticmethod
    def numpy_to_pixmap(image: np.ndarray):
        """Конвертация numpy array в QPixmap для отображения в Qt"""
        from PyQt5.QtGui import QImage, QPixmap
        
        if image is None:
            return None
        
        if len(image.shape) == 2:
            # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        else:
            # Color
            height, width, channel = image.shape
            bytes_per_line = 3 * width
            # OpenCV использует BGR, Qt использует RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        return QPixmap.fromImage(q_image)
    
    @staticmethod
    def is_valid_image_file(file_path: str) -> bool:
        """Проверка, является ли файл допустимым изображением"""
        if not os.path.exists(file_path):
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in Config.SUPPORTED_IMAGE_FORMATS


class ValidationUtils:
    """Утилиты для дополнительной валидации данных"""
    
    @staticmethod
    def validate_passport_data(data: Dict) -> List[str]:
        """
        Комплексная валидация данных паспорта
        
        Returns:
            Список предупреждений/ошибок
        """
        warnings = []
        
        # Проверка ФИО
        if data.get('surname') and len(data['surname']) < 2:
            warnings.append("Фамилия слишком короткая")
        
        if data.get('name') and len(data['name']) < 2:
            warnings.append("Имя слишком короткое")
        
        # Проверка даты рождения
        if data.get('birth_date'):
            try:
                day, month, year = map(int, data['birth_date'].split('.'))
                birth_datetime = datetime(year, month, day)
                
                # Проверка возраста
                age = (datetime.now() - birth_datetime).days / 365.25
                if age < 14:
                    warnings.append("Возраст менее 14 лет")
                elif age > 100:
                    warnings.append("Возраст более 100 лет")
                
            except:
                warnings.append("Некорректный формат даты рождения")
        
        # Проверка номера паспорта
        if data.get('passport_number'):
            if len(data['passport_number']) != 10:
                warnings.append(f"Номер паспорта должен содержать 10 цифр (найдено: {len(data['passport_number'])})")
            elif not data['passport_number'].isdigit():
                warnings.append("Номер паспорта должен содержать только цифры")
        
        # Проверка места рождения
        if data.get('birth_place') and len(data['birth_place']) < 3:
            warnings.append("Место рождения слишком короткое")
        
        return warnings
    
    @staticmethod
    def format_passport_number(number: str) -> str:
        """Форматирование номера паспорта для отображения"""
        if len(number) == 10 and number.isdigit():
            return f"{number[:4]} {number[4:]}"
        return number