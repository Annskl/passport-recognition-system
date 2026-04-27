"""Главный файл приложения обработки паспортов РФ."""

import sys
import os

# Определяем базовую директорию для правильной работы путей.
# В PyInstaller 6.x данные лежат в _internal/ (sys._MEIPASS), а не рядом с exe.
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Добавляем базовую директорию в sys.path
sys.path.insert(0, BASE_DIR)

# Устанавливаем рабочую директорию
os.chdir(BASE_DIR)

# Проверка и установка переменных окружения для лучшей производительности
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

if getattr(sys, 'frozen', False):
    import warnings
    warnings.filterwarnings('ignore')

from passport_gui import main


def check_dependencies():
    """Проверка критических зависимостей"""
    missing_deps = []

    dependencies = {
        'cv2': 'opencv-python',
        'PyQt5': 'PyQt5',
        'ultralytics': 'ultralytics',
        'easyocr': 'easyocr',
        'pandas': 'pandas'
    }

    for module, package in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(package)

    if missing_deps:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)

        msg = f"Не удалось загрузить необходимые библиотеки:\n\n"
        msg += "\n".join(missing_deps)
        msg += "\n\nПрограмма не может быть запущена."

        QMessageBox.critical(None, "Ошибка зависимостей", msg)
        sys.exit(1)

    # Проверка Tesseract (опционально)
    try:
        import pytesseract
        # Пробуем найти Tesseract
        tesseract_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]

        tesseract_found = False
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                tesseract_found = True
                break

        if not tesseract_found:
            print("Предупреждение: Tesseract не найден. Будет использоваться только EasyOCR.")

    except ImportError:
        print("Предупреждение: pytesseract не установлен. Будет использоваться только EasyOCR.")


def check_models():
    """Проверка наличия моделей"""
    models_to_check = [
        "models/page2_seg_model/best.pt",
        "models/fields_detect_model/best.pt",
        "models/passport_number_model/best.pt"
    ]

    missing_models = []
    for model_path in models_to_check:
        full_path = os.path.join(BASE_DIR, model_path)
        if not os.path.exists(full_path):
            missing_models.append(model_path)

    if missing_models:
        from PyQt5.QtWidgets import QApplication, QMessageBox

        if 'qApp' not in globals():
            app = QApplication(sys.argv)

        msg = "Не найдены файлы моделей:\n\n"
        msg += "\n".join(missing_models)
        msg += "\n\nУбедитесь, что папка 'models' находится рядом с программой."

        QMessageBox.critical(None, "Ошибка моделей", msg)
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Проверяем зависимости
        check_dependencies()

        # Проверяем модели
        check_models()

        # Запускаем приложение
        main()

    except ImportError as e:
        from PyQt5.QtWidgets import QApplication, QMessageBox

        app = QApplication(sys.argv)

        error_msg = f"Ошибка импорта модулей:\n{str(e)}\n\n"
        error_msg += "Возможные причины:\n"
        error_msg += "1. Не все зависимости включены в сборку\n"
        error_msg += "2. Конфликт версий библиотек\n"
        error_msg += "3. Повреждены файлы программы"

        QMessageBox.critical(None, "Критическая ошибка", error_msg)
        sys.exit(1)

    except Exception as e:
        # Обработка других критических ошибок
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is None:
                app = QApplication(sys.argv)

            error_msg = f"Критическая ошибка при запуске:\n\n{str(e)}\n\n"
            error_msg += "Попробуйте:\n"
            error_msg += "1. Перезапустить программу\n"
            error_msg += "2. Запустить от имени администратора\n"
            error_msg += "3. Проверить антивирус"

            QMessageBox.critical(None, "Критическая ошибка", error_msg)

        except:
            # Если даже Qt не работает, выводим в консоль
            print(f"\nКритическая ошибка: {e}")
            if not getattr(sys, 'frozen', False):
                import traceback
                traceback.print_exc()

        sys.exit(1)