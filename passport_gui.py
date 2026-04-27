"""GUI приложения распознавания паспортов на PyQt5."""

import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Optional

from passport_processor import PassportProcessor, PassportData
from passport_utils import Config, ImageUtils, DataExporter, DataComparator as DataComparatorUtils


class ImageDelegate(QStyledItemDelegate):
    """Делегат для отображения изображений в таблице с масштабированием"""
    def paint(self, painter, option, index):
        if index.data(Qt.UserRole) is not None:
            # Получаем изображение
            image = index.data(Qt.UserRole)

            # Создаем pixmap
            pixmap = ImageUtils.numpy_to_pixmap(image)

            # Масштабируем под размер ячейки с отступом
            rect = option.rect
            rect.adjust(2, 2, -2, -2)  # Отступ 2 пикселя

            scaled_pixmap = pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

            # Центрируем изображение
            x_offset = (rect.width() - scaled_pixmap.width()) // 2
            y_offset = (rect.height() - scaled_pixmap.height()) // 2

            painter.drawPixmap(rect.x() + x_offset, rect.y() + y_offset, scaled_pixmap)
        else:
            # Если нет изображения, рисуем "-"
            painter.drawText(option.rect, Qt.AlignCenter, "-")


class ProcessingThread(QThread):
    """Поток для обработки изображений"""
    progress = pyqtSignal(int)
    result_ready = pyqtSignal(object)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, processor, images):
        super().__init__()
        self.processor = processor
        self.images = images
        self.is_running = True

    def run(self):
        """Запуск обработки"""
        total = len(self.images)

        for i, image_path in enumerate(self.images):
            if not self.is_running:
                break

            try:
                result = self.processor.process_image(image_path)
                self.result_ready.emit(result)
            except Exception as e:
                self.error.emit(f"Ошибка при обработке {image_path}: {str(e)}")

            progress = int((i + 1) / total * 100)
            self.progress.emit(progress)

        self.finished.emit()

    def stop(self):
        """Остановка обработки"""
        self.is_running = False


class HelpDialog(QDialog):
    """Диалог помощи с подсказками по использованию"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Помощь")
        self.setModal(True)
        self.resize(700, 600)

        layout = QVBoxLayout()

        # Создаем область прокрутки
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)

        # Заголовок
        title = QLabel("Справка по использованию приложения")
        title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        scroll_layout.addWidget(title)

        # Секции помощи
        sections = [
            ("Начало работы", [
                "1. Загрузите изображения паспортов через меню 'Файл' → 'Загрузить изображения' или загрузите целую папку",
                "2. Нажмите 'Начать обработку' (F5) для запуска распознавания",
                "3. Дождитесь завершения обработки",
                "4. Просмотрите и отредактируйте результаты в таблице"
            ]),

            ("Цветовая индикация в таблице", [
                "• Желтый фон — низкая уверенность распознавания (менее 50%)",
                "• Светло-красный фон — поле не распознано (пустое)",
                "• Обычный фон — успешное распознавание"
            ]),

            ("Редактирование данных", [
                "• Двойной клик на текстовое поле — редактирование прямо в таблице",
                "• Двойной клик на изображение — просмотр в полном размере",
                "• Двойной клик на имя файла — детальный просмотр записи",
                "• В детальном просмотре нажмите 'Редактировать' для изменения всех полей"
            ]),

            ("Сравнение с базой данных", [
                "• База данных должна содержать поля с такими же названиями:",
                "  - surname (фамилия)",
                "  - name (имя)",
                "  - patronymic (отчество)",
                "  - birth_date (дата рождения)",
                "  - passport_number (номер паспорта)",
                "• Поддерживаемые форматы баз: SQLite (.db), Excel (.xlsx), CSV, JSON",
                "• Сравнение происходит по процентному совпадению полей (порог 70%)"
            ]),

            ("Экспорт данных", [
                "• Доступные форматы: Excel (.xlsx), CSV, JSON, SQLite (.db)",
                "• Можно выбрать какие поля экспортировать",
                "• Изображения не экспортируются в Excel/CSV/JSON",
                "• Для экспорта фото лиц используйте отдельную функцию"
            ]),

            ("Горячие клавиши", [
                "• Ctrl+O — Загрузить изображения",
                "• Ctrl+Shift+O — Загрузить папку",
                "• F5 — Начать обработку",
                "• Esc — Остановить обработку",
                "• F2 — Детальный просмотр",
                "• Ctrl+S — Экспортировать данные",
                "• Ctrl+Q — Выход из программы"
            ])
        ]

        # Добавляем секции
        for section_title, items in sections:
            # Заголовок секции
            section_label = QLabel(section_title)
            section_label.setStyleSheet("font-size: 14px; font-weight: bold; padding-top: 15px;")
            scroll_layout.addWidget(section_label)

            # Элементы секции
            for item in items:
                item_label = QLabel(item)
                item_label.setWordWrap(True)
                item_label.setStyleSheet("padding-left: 20px; padding-bottom: 5px;")
                scroll_layout.addWidget(item_label)

        # Растягивающийся элемент
        scroll_layout.addStretch()

        # Настраиваем прокрутку
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        # Кнопка закрытия
        close_button = QPushButton("Закрыть")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)


class DetailViewDialog(QDialog):
    """Диалог детального просмотра записи с возможностью листания и редактирования"""
    def __init__(self, all_data, current_index, parent=None):
        super().__init__(parent)
        self.all_data = all_data
        self.current_index = current_index
        self.parent_window = parent
        self.edit_mode = False
        self.edit_widgets = {}

        # ИСПРАВЛЕНИЕ: Убираем вопросительный знак из заголовка окна
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self.init_ui()
        self.show_record(self.current_index)

    def init_ui(self):
        self.setWindowTitle("Детальный просмотр")
        self.setModal(True)
        screen = QApplication.desktop().screenGeometry()
        self.resize(screen.width() - 100, screen.height() - 100)

        # Главный layout
        main_layout = QVBoxLayout()

        # Панель навигации
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("← Предыдущий")
        self.prev_button.clicked.connect(self.show_previous)
        nav_layout.addWidget(self.prev_button)

        self.record_label = QLabel()
        self.record_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.record_label, 1)

        # Кнопка редактирования
        self.edit_button = QPushButton("Редактировать")
        self.edit_button.clicked.connect(self.toggle_edit_mode)
        nav_layout.addWidget(self.edit_button)

        self.next_button = QPushButton("Следующий →")
        self.next_button.clicked.connect(self.show_next)
        nav_layout.addWidget(self.next_button)

        main_layout.addLayout(nav_layout)

        # Прокручиваемая область
        self.scroll = QScrollArea()
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)

        self.scroll.setWidget(self.scroll_widget)
        self.scroll.setWidgetResizable(True)

        main_layout.addWidget(self.scroll)

        # Кнопка закрытия
        close_button = QPushButton("Закрыть (Esc)")
        close_button.clicked.connect(self.accept)
        main_layout.addWidget(close_button, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)

        # Горячие клавиши
        self.prev_button.setShortcut(Qt.Key_Left)
        self.next_button.setShortcut(Qt.Key_Right)
        close_button.setShortcut(Qt.Key_Escape)

    def toggle_edit_mode(self):
        """Переключение режима редактирования"""
        self.edit_mode = not self.edit_mode

        if self.edit_mode:
            self.edit_button.setText("Завершить редактирование")
            self.edit_button.setStyleSheet("background-color: #4CAF50; color: white;")
            # Включаем поля редактирования
            for field, widget in self.edit_widgets.items():
                if isinstance(widget, QLineEdit):
                    widget.setReadOnly(False)
                    widget.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        else:
            self.edit_button.setText("Редактировать")
            self.edit_button.setStyleSheet("")
            # Сохраняем изменения
            self.save_changes()
            # Отключаем поля редактирования
            for field, widget in self.edit_widgets.items():
                if isinstance(widget, QLineEdit):
                    widget.setReadOnly(True)
                    widget.setStyleSheet("background-color: #f0f0f0; border: none;")

    def save_changes(self):
        """Сохранение изменений"""
        data = self.all_data[self.current_index]

        for field, widget in self.edit_widgets.items():
            if isinstance(widget, QLineEdit):
                new_value = widget.text()
                data[field] = new_value

        # Обновляем таблицу в главном окне
        if self.parent_window:
            self.parent_window.update_table_row(self.current_index)

    def clear_layout(self, layout):
        """Полная очистка layout от всех виджетов"""
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()
                else:
                    sublayout = item.layout()
                    if sublayout is not None:
                        self.clear_layout(sublayout)
                        sublayout.deleteLater()

    def show_record(self, index):
        """Показать запись с указанным индексом"""
        if 0 <= index < len(self.all_data):
            self.current_index = index
            self.data = self.all_data[index]
            self.edit_widgets.clear()

            # Очищаем предыдущее содержимое
            self.clear_layout(self.scroll_layout)

            # Обновляем навигацию
            self.record_label.setText(f"Запись {index + 1} из {len(self.all_data)}")
            self.prev_button.setEnabled(index > 0)
            self.next_button.setEnabled(index < len(self.all_data) - 1)

            # Заголовок с именем файла
            title = QLabel(f"Файл: {self.data.get('file_name', 'Неизвестно')}")
            title.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
            self.scroll_layout.addWidget(title)

            # Горизонтальный layout для изображений
            images_layout = QHBoxLayout()

            # Показываем изображения
            image_fields = [
                ('original_image', 'Исходное изображение'),
                ('page2_image', '2-я страница паспорта'),
                ('face_image', 'Фото лица')
            ]

            screen = QApplication.desktop().screenGeometry()

            for field, title_text in image_fields:
                if self.data.get(field) is not None:
                    # Создаем виджет для изображения
                    img_widget = QWidget()
                    img_layout = QVBoxLayout(img_widget)

                    # Заголовок
                    label = QLabel(title_text)
                    label.setAlignment(Qt.AlignCenter)
                    label.setStyleSheet("font-weight: bold; padding: 5px;")
                    img_layout.addWidget(label)

                    # Изображение
                    pixmap = ImageUtils.numpy_to_pixmap(self.data[field])
                    max_height = int(screen.height() * 0.5)
                    pixmap = pixmap.scaledToHeight(max_height, Qt.SmoothTransformation)

                    img_label = QLabel()
                    img_label.setPixmap(pixmap)
                    img_label.setAlignment(Qt.AlignCenter)
                    img_layout.addWidget(img_label)

                    images_layout.addWidget(img_widget)

            # Добавляем layout с изображениями
            images_widget = QWidget()
            images_widget.setLayout(images_layout)
            self.scroll_layout.addWidget(images_widget)

            # Разделитель
            line = QLabel()
            line.setFrameStyle(QLabel.HLine | QLabel.Sunken)
            self.scroll_layout.addWidget(line)

            # Текстовые поля
            text_fields_widget = QWidget()
            text_fields_layout = QGridLayout(text_fields_widget)
            text_fields_layout.setSpacing(10)

            text_fields = [
                ('surname', 'Фамилия'),
                ('name', 'Имя'),
                ('patronymic', 'Отчество'),
                ('birth_date', 'Дата рождения'),
                ('birth_place', 'Место рождения'),
                ('passport_number', 'Номер паспорта')
            ]

            for i, (field, title_text) in enumerate(text_fields):
                # Название поля
                label = QLabel(f"{title_text}:")
                label.setStyleSheet("font-weight: bold;")
                text_fields_layout.addWidget(label, i // 3, (i % 3) * 2)

                # Значение поля
                value = self.data.get(field, '-')
                confidence = self.data.get('confidence_scores', {}).get(field, 0)

                # Создаем редактируемое поле
                value_edit = QLineEdit(str(value) if value else '-')
                value_edit.setReadOnly(True)

                if confidence > 0:
                    if confidence < 0.5:
                        value_edit.setStyleSheet("background-color: yellow; padding: 5px; border: none;")
                    else:
                        value_edit.setStyleSheet("background-color: lightgreen; padding: 5px; border: none;")
                else:
                    value_edit.setStyleSheet("background-color: #f0f0f0; padding: 5px; border: none;")

                self.edit_widgets[field] = value_edit
                text_fields_layout.addWidget(value_edit, i // 3, (i % 3) * 2 + 1)

            self.scroll_layout.addWidget(text_fields_widget)

            # Растягивающийся элемент внизу
            spacer = QWidget()
            spacer.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
            self.scroll_layout.addWidget(spacer)

            # Прокручиваем вверх
            self.scroll.verticalScrollBar().setValue(0)

            # Принудительное обновление
            QApplication.processEvents()

    def show_previous(self):
        """Показать предыдущую запись"""
        if self.edit_mode:
            self.toggle_edit_mode()  # Сохраняем изменения
        if self.current_index > 0:
            self.show_record(self.current_index - 1)

    def show_next(self):
        """Показать следующую запись"""
        if self.edit_mode:
            self.toggle_edit_mode()  # Сохраняем изменения
        if self.current_index < len(self.all_data) - 1:
            self.show_record(self.current_index + 1)


class PassportApp(QMainWindow):
    """Главное окно приложения"""

    def __init__(self):
        super().__init__()
        self.processor = None
        self.processing_thread = None
        self.images_to_process = []
        self.results_data = []
        self.comparison_db = None
        self.image_delegate = ImageDelegate()

        self.init_ui()
        self.load_processor()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Обработка паспортов РФ")
        self.setWindowIcon(QIcon("icon.png"))

        # Устанавливаем размер окна
        screen = QApplication.desktop().screenGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        # Центрируем окно
        self.move(int(screen.width() * 0.1), int(screen.height() * 0.1))

        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Главный layout
        main_layout = QVBoxLayout(central_widget)

        # Создаем меню
        self.create_menu()

        # Создаем панель инструментов
        self.create_toolbar()

        # Создаем основную область
        self.create_main_area(main_layout)

        # Создаем статусбар
        self.create_statusbar()

    def create_menu(self):
        """Создание главного меню"""
        menubar = self.menuBar()

        # Меню Файл
        file_menu = menubar.addMenu('Файл')

        self.action_load_images = QAction('Загрузить изображения', self)
        self.action_load_images.setShortcut('Ctrl+O')
        self.action_load_images.triggered.connect(self.load_images)
        file_menu.addAction(self.action_load_images)

        self.action_load_folder = QAction('Загрузить папку', self)
        self.action_load_folder.setShortcut('Ctrl+Shift+O')
        self.action_load_folder.triggered.connect(self.load_folder)
        file_menu.addAction(self.action_load_folder)

        file_menu.addSeparator()

        exit_action = QAction('Выход', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Меню Обработка
        process_menu = menubar.addMenu('Обработка')

        self.action_process = QAction('Начать обработку', self)
        self.action_process.setShortcut('F5')
        self.action_process.triggered.connect(self.start_processing)
        self.action_process.setEnabled(False)
        process_menu.addAction(self.action_process)

        self.action_stop = QAction('Остановить обработку', self)
        self.action_stop.setShortcut('Esc')
        self.action_stop.triggered.connect(self.stop_processing)
        self.action_stop.setEnabled(False)
        process_menu.addAction(self.action_stop)

        # Меню Вид
        view_menu = menubar.addMenu('Вид')

        # Подменю для отображаемых полей
        self.fields_menu = view_menu.addMenu('Отображаемые поля')

        self.field_actions = {}
        for field, title in Config.TABLE_FIELDS.items():
            action = QAction(title, self, checkable=True)
            action.setChecked(True)
            if field in ['original_image', 'page2_image', 'face_image']:
                action.setChecked(False)  # По умолчанию изображения скрыты
            action.triggered.connect(lambda checked, f=field: self.toggle_field_visibility(f, checked))
            self.fields_menu.addAction(action)
            self.field_actions[field] = action

        view_menu.addSeparator()

        self.action_detail_view = QAction('Детальный просмотр', self)
        self.action_detail_view.setShortcut('F2')
        self.action_detail_view.triggered.connect(self.show_detail_view)
        view_menu.addAction(self.action_detail_view)

        # Меню Сравнение
        compare_menu = menubar.addMenu('Сравнение')

        self.action_load_db = QAction('Загрузить базу данных', self)
        self.action_load_db.triggered.connect(self.load_comparison_db)
        compare_menu.addAction(self.action_load_db)

        self.action_compare = QAction('Сравнить с базой', self)
        self.action_compare.triggered.connect(self.compare_data)
        self.action_compare.setEnabled(False)
        compare_menu.addAction(self.action_compare)

        # Меню Экспорт
        export_menu = menubar.addMenu('Экспорт')

        self.action_export = QAction('Экспортировать данные', self)
        self.action_export.setShortcut('Ctrl+S')
        self.action_export.triggered.connect(self.export_data)
        export_menu.addAction(self.action_export)

        self.action_export_faces = QAction('Экспортировать фото лиц', self)
        self.action_export_faces.triggered.connect(self.export_faces)
        export_menu.addAction(self.action_export_faces)

        # ДОБАВЛЕНИЕ: Меню Помощь
        help_menu = menubar.addMenu('Помощь')

        help_action = QAction('Справка', self)
        help_action.setShortcut('F1')
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)

    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = self.addToolBar('Главная')
        toolbar.setMovable(False)

        # Кнопки загрузки
        toolbar.addAction(self.action_load_images)
        toolbar.addAction(self.action_load_folder)
        toolbar.addSeparator()

        # Кнопки обработки
        toolbar.addAction(self.action_process)
        toolbar.addAction(self.action_stop)
        toolbar.addSeparator()

        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        toolbar.addWidget(self.progress_bar)

        # ДОБАВЛЕНИЕ: Кнопка помощи в тулбар
        toolbar.addSeparator()
        help_button = QAction(QIcon.fromTheme("help-contents"), "Помощь", self)
        help_button.setToolTip("Показать справку (F1)")
        help_button.triggered.connect(self.show_help)
        toolbar.addAction(help_button)

    def create_main_area(self, parent_layout):
        """Создание основной области с таблицей"""
        # Только таблица результатов, без панели полей
        self.table = QTableWidget()
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.table.itemChanged.connect(self.on_item_changed)

        self.setup_table()
        parent_layout.addWidget(self.table)

    def setup_table(self):
        """Настройка таблицы"""
        # Устанавливаем заголовки
        headers = ['Файл'] + [Config.TABLE_FIELDS[field] for field in Config.TABLE_FIELDS.keys()]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        # Настройка внешнего вида
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        # ИСПРАВЛЕНИЕ: Настройка прокрутки для предотвращения ухода столбцов за границы
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Настройка заголовка таблицы
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)  # НЕ растягиваем последний столбец

        # Устанавливаем минимальную ширину для колонок
        for i in range(self.table.columnCount()):
            header.setMinimumSectionSize(50)

        # Устанавливаем режим изменения размера
        header.setSectionResizeMode(QHeaderView.Interactive)

        # Устанавливаем предпочтительную ширину для колонок с текстом
        self.table.setColumnWidth(0, 150)  # Файл

        # Индексы колонок для текстовых полей
        text_field_indices = {
            'surname': 4,
            'name': 5,
            'patronymic': 6,
            'birth_date': 7,
            'birth_place': 8,
            'passport_number': 9
        }

        for field, index in text_field_indices.items():
            if field in ['birth_place']:
                self.table.setColumnWidth(index, 200)  # Место рождения шире
            elif field in ['passport_number']:
                self.table.setColumnWidth(index, 120)
            else:
                self.table.setColumnWidth(index, 100)

        # Устанавливаем делегаты для столбцов с изображениями
        for i, field in enumerate(Config.TABLE_FIELDS.keys()):
            if field in ['original_image', 'page2_image', 'face_image']:
                col_index = i + 1
                self.table.setItemDelegateForColumn(col_index, self.image_delegate)

        # Скрываем столбцы с изображениями по умолчанию
        for i, field in enumerate(['original_image', 'page2_image', 'face_image']):
            if field in Config.TABLE_FIELDS:
                col_index = list(Config.TABLE_FIELDS.keys()).index(field) + 1
                self.table.setColumnHidden(col_index, True)

        # Устанавливаем высоту строк по умолчанию
        self.table.verticalHeader().setDefaultSectionSize(30)

        # ИСПРАВЛЕНИЕ: Обеспечиваем правильную работу горизонтальной прокрутки
        self.table.setSizeAdjustPolicy(QTableWidget.AdjustToContents)

    def create_statusbar(self):
        """Создание статусбара"""
        self.status_bar = self.statusBar()
        self.status_label = QLabel("Готов к работе")
        self.status_bar.addWidget(self.status_label)

        # Счетчик записей
        self.count_label = QLabel("Записей: 0")
        self.status_bar.addPermanentWidget(self.count_label)

    def load_processor(self):
        """Загрузка процессора"""
        try:
            base_dir = Config.get_base_dir()
            config = Config.load_config(base_dir=base_dir)
            self.processor = PassportProcessor(config)
            self.status_label.setText("Процессор загружен")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить процессор:\n{str(e)}")
            self.close()

    def show_help(self):
        """Показать диалог помощи"""
        help_dialog = HelpDialog(self)
        help_dialog.exec_()

    def load_images(self):
        """Загрузка изображений"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите изображения",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if files:
            self.images_to_process = files
            self.status_label.setText(f"Загружено изображений: {len(files)}")
            self.action_process.setEnabled(True)

    def load_folder(self):
        """Загрузка папки с изображениями"""
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку")

        if folder:
            folder_path = Path(folder)
            images = []

            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                images.extend(folder_path.glob(ext))

            if images:
                self.images_to_process = [str(img) for img in images]
                self.status_label.setText(f"Загружено изображений: {len(images)}")
                self.action_process.setEnabled(True)
            else:
                QMessageBox.warning(self, "Предупреждение", "В папке не найдено изображений")

    def start_processing(self):
        """Запуск обработки"""
        if not self.images_to_process:
            return

        # Очищаем предыдущие результаты
        self.results_data.clear()
        self.table.setRowCount(0)

        # Создаем и запускаем поток обработки
        self.processing_thread = ProcessingThread(self.processor, self.images_to_process)
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.result_ready.connect(self.add_result)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.show_error)

        # Обновляем UI
        self.action_process.setEnabled(False)
        self.action_stop.setEnabled(True)
        self.status_label.setText("Обработка...")

        self.processing_thread.start()

    def stop_processing(self):
        """Остановка обработки"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.status_label.setText("Остановка обработки...")

    def update_progress(self, value):
        """Обновление прогресса"""
        self.progress_bar.setValue(value)

    def add_result(self, result: PassportData):
        """Добавление результата в таблицу"""
        # Конвертируем в словарь для удобства
        data = {
            'file_name': result.file_name,
            'surname': result.surname,
            'name': result.name,
            'patronymic': result.patronymic,
            'birth_date': result.birth_date,
            'birth_place': result.birth_place,
            'passport_number': result.passport_number,
            'original_image': result.original_image,
            'page2_image': result.page2_image,
            'face_image': result.face_image,
            'confidence_scores': result.confidence_scores,
            'processing_errors': result.processing_errors
        }

        self.results_data.append(data)

        # Добавляем строку в таблицу
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)

        # Временно отключаем сигналы чтобы избежать рекурсии
        self.table.blockSignals(True)

        # Заполняем ячейки
        # Имя файла
        file_item = QTableWidgetItem(data['file_name'])
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)  # Запрещаем редактирование
        self.table.setItem(row_position, 0, file_item)

        col_index = 1
        for field in Config.TABLE_FIELDS.keys():
            if field in ['original_image', 'page2_image', 'face_image']:
                # Для изображений создаем элемент с данными
                if data.get(field) is not None:
                    item = QTableWidgetItem()
                    item.setData(Qt.UserRole, data[field])  # Сохраняем изображение
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # Запрещаем редактирование
                    self.table.setItem(row_position, col_index, item)
                else:
                    item = QTableWidgetItem("-")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row_position, col_index, item)
            else:
                # Для текстовых полей
                value = data.get(field, '')
                display_text = str(value) if value else ''

                item = QTableWidgetItem(display_text)

                # Центрируем текст для лучшей видимости
                item.setTextAlignment(Qt.AlignCenter)

                # Подсветка ошибок
                if field in data.get('confidence_scores', {}) and data['confidence_scores'][field] < 0.5:
                    item.setBackground(QColor(255, 255, 0))  # Желтый фон для низкой уверенности
                elif not display_text:  # Если поле пустое
                    item.setBackground(QColor(255, 200, 200))  # Светло-красный фон для пустых полей

                self.table.setItem(row_position, col_index, item)
            col_index += 1

        # Включаем сигналы обратно
        self.table.blockSignals(False)

        # Обновляем счетчик
        self.count_label.setText(f"Записей: {len(self.results_data)}")

        # Прокручиваем к новой записи
        self.table.scrollToItem(self.table.item(row_position, 0))

        # ИСПРАВЛЕНИЕ: Обновляем размеры столбцов после добавления данных
        self.table.resizeColumnsToContents()

    def processing_finished(self):
        """Обработка завершена"""
        self.action_process.setEnabled(True)
        self.action_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.status_label.setText("Обработка завершена")

        if self.results_data:
            self.action_compare.setEnabled(True)

        QMessageBox.information(self, "Готово", f"Обработка завершена.\nОбработано записей: {len(self.results_data)}")

    def show_error(self, error_msg):
        """Показать ошибку"""
        QMessageBox.warning(self, "Ошибка", error_msg)

    def show_context_menu(self, pos):
        """Показать контекстное меню"""
        if self.table.selectedIndexes():
            menu = QMenu()

            action_view = menu.addAction("Детальный просмотр")
            action_view.triggered.connect(self.show_detail_view)

            action_delete = menu.addAction("Удалить")
            action_delete.triggered.connect(self.delete_selected)

            menu.exec_(self.table.mapToGlobal(pos))

    def show_detail_view(self):
        """Показать детальный просмотр"""
        current_row = self.table.currentRow()
        if current_row >= 0:
            self.show_detail_view_for_row(current_row)

    def show_detail_view_for_row(self, row):
        """Показать детальный просмотр для конкретной строки"""
        if 0 <= row < len(self.results_data):
            dialog = DetailViewDialog(self.results_data, row, self)
            dialog.exec_()

    def delete_selected(self):
        """Удалить выбранные записи"""
        selected_rows = set()
        for index in self.table.selectedIndexes():
            selected_rows.add(index.row())

        if selected_rows:
            reply = QMessageBox.question(
                self,
                "Подтверждение",
                f"Удалить {len(selected_rows)} записей?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                # Удаляем в обратном порядке
                for row in sorted(selected_rows, reverse=True):
                    self.table.removeRow(row)
                    del self.results_data[row]

                self.count_label.setText(f"Записей: {len(self.results_data)}")

    def on_item_double_clicked(self, item):
        """Обработка двойного клика по элементу таблицы"""
        row = item.row()
        col = item.column()

        if col > 0:  # Пропускаем столбец с именем файла
            field = list(Config.TABLE_FIELDS.keys())[col - 1]

            # Для изображений - только просмотр
            if field in ['original_image', 'page2_image', 'face_image']:
                self.view_image(row, field)
        else:
            # При клике на имя файла - детальный просмотр
            self.show_detail_view_for_row(row)

    def on_item_changed(self, item):
        """Обработка изменения элемента таблицы"""
        if item is None:
            return

        row = item.row()
        col = item.column()

        # Пропускаем столбец с именем файла и столбцы с изображениями
        if col > 0 and row < len(self.results_data):
            field = list(Config.TABLE_FIELDS.keys())[col - 1]

            if field not in ['original_image', 'page2_image', 'face_image']:
                # Обновляем данные
                new_value = item.text()
                self.results_data[row][field] = new_value

    def view_image(self, row, field):
        """Просмотр изображения"""
        if 0 <= row < len(self.results_data):
            image = self.results_data[row].get(field)
            if image is not None:
                # Создаем диалог для показа изображения
                dialog = QDialog(self)
                dialog.setWindowTitle(Config.TABLE_FIELDS[field])
                dialog.setModal(True)

                layout = QVBoxLayout()

                # Изображение
                label = QLabel()
                pixmap = ImageUtils.numpy_to_pixmap(image)

                # Масштабируем под размер экрана
                screen = QApplication.desktop().screenGeometry()
                max_size = QSize(int(screen.width() * 0.8), int(screen.height() * 0.8))
                pixmap = pixmap.scaled(max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                label.setPixmap(pixmap)
                layout.addWidget(label)

                # Кнопка закрытия
                close_button = QPushButton("Закрыть")
                close_button.clicked.connect(dialog.accept)
                layout.addWidget(close_button, alignment=Qt.AlignCenter)

                dialog.setLayout(layout)
                dialog.exec_()

    def update_table_row(self, row_index):
        """Обновление строки таблицы после редактирования"""
        if 0 <= row_index < len(self.results_data):
            data = self.results_data[row_index]

            # Временно отключаем сигналы
            self.table.blockSignals(True)

            # Обновляем только текстовые поля
            col_index = 1
            for field in Config.TABLE_FIELDS.keys():
                if field not in ['original_image', 'page2_image', 'face_image']:
                    value = data.get(field, '')
                    item = self.table.item(row_index, col_index)
                    if item:
                        item.setText(str(value) if value else '')
                col_index += 1

            # Включаем сигналы обратно
            self.table.blockSignals(False)

    def toggle_field_visibility(self, field, visible):
        """Переключение видимости поля в таблице"""
        col_index = list(Config.TABLE_FIELDS.keys()).index(field) + 1
        self.table.setColumnHidden(col_index, not visible)

        # Если показываем столбец с изображением, увеличиваем высоту строк
        if visible and field in ['original_image', 'page2_image', 'face_image']:
            for row in range(self.table.rowCount()):
                if any(self.results_data[row].get(f) is not None
                       for f in ['original_image', 'page2_image', 'face_image']):
                    self.table.setRowHeight(row, 80)
        else:
            # Возвращаем стандартную высоту
            for row in range(self.table.rowCount()):
                self.table.setRowHeight(row, 30)

    def load_comparison_db(self):
        """Загрузка базы для сравнения"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл базы данных",
            "",
            "All Supported (*.json *.xlsx *.xls *.csv *.db);;JSON Files (*.json);;Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;SQLite Database (*.db)"
        )

        if file_path:
            try:
                self.comparison_db = DataComparatorUtils.load_comparison_data(file_path)
                QMessageBox.information(self, "Успешно", f"База данных загружена.\nЗаписей: {len(self.comparison_db)}")
                self.action_compare.setEnabled(bool(self.results_data))
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить базу:\n{str(e)}")

    def compare_data(self):
        """Сравнение с базой"""
        if not self.comparison_db:
            QMessageBox.warning(self, "Предупреждение", "Сначала загрузите базу данных для сравнения")
            return

        # Диалог прогресса
        progress = QProgressDialog("Сравнение данных...", "Отмена", 0, len(self.results_data), self)
        progress.setWindowModality(Qt.WindowModal)

        matches_found = []

        for i, passport_data in enumerate(self.results_data):
            if progress.wasCanceled():
                break

            progress.setValue(i)

            result = DataComparatorUtils.compare_data(passport_data, self.comparison_db)
            if result['match_found']:
                matches_found.append({
                    'passport_data': passport_data,
                    'matched_record': result['matched_record'],
                    'match_score': result['match_score'],
                    'differences': result['differences']
                })

        progress.setValue(len(self.results_data))

        # Показываем результаты
        if matches_found:
            msg = f"Найдено совпадений: {len(matches_found)}\n\n"

            for i, match in enumerate(matches_found[:10]):  # Показываем первые 10
                msg += f"{i+1}. {match['passport_data']['file_name']}\n"
                msg += f"   Совпадение: {match['match_score']:.1%}\n"
                if match['differences']:
                    msg += f"   Различия: {'; '.join(match['differences'][:2])}\n"
                msg += "\n"

            if len(matches_found) > 10:
                msg += f"...и еще {len(matches_found) - 10} совпадений"

            QMessageBox.information(self, "Результаты сравнения", msg)
        else:
            QMessageBox.information(self, "Результаты сравнения", "Совпадений не найдено")

    def export_data(self):
        """Экспорт данных"""
        if not self.results_data:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта")
            return

        # Диалог выбора полей для экспорта
        fields_dialog = QDialog(self)
        fields_dialog.setWindowTitle("Выберите поля для экспорта")
        fields_dialog.setModal(True)

        layout = QVBoxLayout()

        # ИСПРАВЛЕНИЕ: Добавляем поле "Файл" в список
        checkboxes = {}

        # Сначала добавляем чекбокс для файла
        file_checkbox = QCheckBox("Файл")
        file_checkbox.setChecked(False)  # По умолчанию не выбран
        checkboxes['file_name'] = file_checkbox
        layout.addWidget(file_checkbox)

        # Затем добавляем только текстовые поля (без изображений)
        for field, title in Config.TABLE_FIELDS.items():
            # ИСПРАВЛЕНИЕ: Пропускаем поля с изображениями
            if field not in ['original_image', 'page2_image', 'face_image']:
                checkbox = QCheckBox(title)
                checkbox.setChecked(True)  # Текстовые поля выбраны по умолчанию
                checkboxes[field] = checkbox
                layout.addWidget(checkbox)

        # Кнопки
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Отмена")

        ok_button.clicked.connect(fields_dialog.accept)
        cancel_button.clicked.connect(fields_dialog.reject)

        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        fields_dialog.setLayout(layout)

        if fields_dialog.exec_() == QDialog.Accepted:
            # Получаем выбранные поля
            selected_fields = [field for field, checkbox in checkboxes.items() if checkbox.isChecked()]

            if not selected_fields:
                QMessageBox.warning(self, "Предупреждение", "Не выбрано ни одного поля")
                return

            # Выбор файла для сохранения
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self,
                "Сохранить данные",
                f"passport_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "Excel Files (*.xlsx);;JSON Files (*.json);;CSV Files (*.csv);;SQLite Database (*.db)"
            )

            if file_path:
                try:
                    # Определяем формат по расширению или фильтру
                    if file_path.endswith('.xlsx') or 'Excel' in selected_filter:
                        if not file_path.endswith('.xlsx'):
                            file_path += '.xlsx'
                        DataExporter.export_to_excel(self.results_data, file_path, selected_fields)
                    elif file_path.endswith('.json') or 'JSON' in selected_filter:
                        if not file_path.endswith('.json'):
                            file_path += '.json'
                        DataExporter.export_to_json(self.results_data, file_path, selected_fields)
                    elif file_path.endswith('.csv') or 'CSV' in selected_filter:
                        if not file_path.endswith('.csv'):
                            file_path += '.csv'
                        DataExporter.export_to_csv(self.results_data, file_path, selected_fields)
                    elif file_path.endswith('.db') or 'SQLite' in selected_filter:
                        if not file_path.endswith('.db'):
                            file_path += '.db'
                        DataExporter.export_to_sqlite(self.results_data, file_path, selected_fields)

                    QMessageBox.information(self, "Успешно", f"Данные экспортированы в:\n{file_path}")

                except Exception as e:
                    QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте:\n{str(e)}")

    def export_faces(self):
        """Экспорт фото лиц"""
        if not self.results_data:
            QMessageBox.warning(self, "Предупреждение", "Нет данных для экспорта")
            return

        # Проверяем наличие фото лиц
        faces_count = sum(1 for data in self.results_data if data.get('face_image') is not None)

        if faces_count == 0:
            QMessageBox.warning(self, "Предупреждение", "Не найдено ни одного фото лица")
            return

        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения фото лиц")

        if folder:
            try:
                DataExporter.export_faces(self.results_data, folder)
                QMessageBox.information(self, "Успешно", f"Экспортировано фото лиц: {faces_count}\nВ папку: {folder}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка при экспорте фото:\n{str(e)}")


def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Устанавливаем иконку приложения
    app.setWindowIcon(QIcon("icon.png"))

    # Создаем и показываем главное окно
    window = PassportApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()