import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from datetime import datetime
import json
import csv
import tensorflow as tf
from tensorflow import keras

class MNISTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Classifier")
        self.root.geometry("1200x800")
        
        # Инициализация всех атрибутов перед их использованием
        self.current_theme = "light"
        self.users_file = "users.json"  # Добавлено
        self.settings_file = "app_settings.json"
        self.db_file = "mnist_data.csv"
        self.current_user = None
        self.history = []
        
        # Теперь можно загружать пользователей
        self.users = self.load_users()
        
        # Загрузка модели MNIST
        self.model = self.load_model()
        
        # Инициализация базы данных
        self.init_database()
        
        # Создание интерфейса
        self.create_widgets()
        self.apply_theme()
        self.load_settings()
        
        # Авторизация
        self.show_login_dialog()
    
    def load_users(self):
        """Загрузка пользователей из файла"""
        default_users = {
            "admin": {"password": "admin123", "role": "admin"},
            "1": {"password": "1", "role": "admin"},
            "user1": {"password": "user1123", "role": "user"},
            "user2": {"password": "user2123", "role": "user"}
        }
        
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, "r") as f:
                    return json.load(f)
            except:
                return default_users
        return default_users
    
    def save_users(self):
        """Сохранение пользователей в файл"""
        try:
            with open(self.users_file, "w") as f:
                json.dump(self.users, f)
        except Exception as e:
            print(f"Ошибка сохранения пользователей: {str(e)}")
    

    # Попробуйте явно создать модель с правильной архитектурой:
    def load_model(self):
        """Загрузка модели MNIST"""
        try:
            # Вариант 1: Загрузка всей модели в новом формате
            # model = keras.models.load_model('mnist_model.keras')
            
            # Вариант 2: Или создание модели + загрузка весов
            model = keras.Sequential([
                keras.layers.Flatten(input_shape=(28, 28)),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(10, activation='softmax')
            ])
            model.load_weights('mnist_weights.weights.h5')  # Новый формат
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            
            return model
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")
            return None
        
    def predict(self):
        """Предсказание цифры на загруженном изображении с полной диагностикой"""
        # Проверка наличия модели
        if not self.model:
            messagebox.showerror("Ошибка", "Модель не загружена")
            self.status_bar.config(text="Ошибка: модель не загружена")
            return
            
        # Проверка авторизации пользователя
        if not self.current_user:
            messagebox.showwarning("Предупреждение", "Сначала авторизуйтесь")
            self.status_bar.config(text="Требуется авторизация")
            return
                    
        # Проверка загруженного изображения
        if not hasattr(self, 'image'):
            messagebox.showwarning("Предупреждение", "Сначала загрузите изображение")
            self.status_bar.config(text="Нет загруженного изображения")
            return

        try:
            # ===== 1. Подготовка изображения =====
            # Конвертируем в массив numpy и инвертируем цвета (MNIST использует белые цифры на черном фоне)
            img_array = 255 - np.array(self.image.resize((28, 28)))
            
            # Нормализация (приводим к диапазону 0-1)
            img_array = img_array.astype("float32") / 255.0
            
            # Добавляем размерность батча (1, 28, 28)
            img_array = img_array.reshape(1, 28, 28)
            
            # ===== 2. Диагностический вывод =====
            print("\n=== Диагностика перед предсказанием ===")
            print("Форма массива:", img_array.shape)
            print("Диапазон значений:", np.min(img_array), "-", np.max(img_array))
            print("Среднее значение пикселей:", np.mean(img_array))
            
            # Визуализация того, что "видит" модель
            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img_array[0], cmap='gray')
            plt.title("Что видит модель")
            plt.axis('off')
            
            # ===== 3. Выполнение предсказания =====
            predictions = self.model.predict(img_array, verbose=0)[0]  # Берем первый (и единственный) элемент
            digit = np.argmax(predictions)
            confidence = np.max(predictions)
            
            # ===== 4. Анализ результатов =====
            print("\nРезультаты предсказания:")
            print("Все вероятности:", [f"{p:.4f}" for p in predictions])
            print(f"Предсказанная цифра: {digit} (уверенность: {confidence:.2%})")
            
            # Визуализация вероятностей
            plt.subplot(1, 2, 2)
            bars = plt.bar(range(10), predictions, color='skyblue')
            bars[digit].set_color('orange')
            plt.title("Вероятности цифр")
            plt.xlabel("Цифра")
            plt.ylabel("Вероятность")
            plt.xticks(range(10))
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()
            
            # ===== 5. Обновление интерфейса =====
            self.result_label.config(text=f"Результат: {digit} (уверенность: {confidence:.1%})")
            self.update_probability_plot(predictions)  # Передаем все 10 вероятностей
            
            # Сохранение в историю
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history.append({
                "image": self.current_image_path,
                "predicted": digit,
                "confidence": confidence,
                "timestamp": timestamp,
                "user": self.current_user
            })
            self.update_history()
            
            self.status_bar.config(text=f"Распознано: цифра {digit}")
            
            # ===== 6. Проверка аномальных предсказаний =====
            if confidence < 0.5:
                messagebox.showwarning("Низкая уверенность", 
                                    f"Модель не уверена в результате (уверенность: {confidence:.1%})")
            elif np.sum(predictions > 0.1) > 2:  # Если несколько цифр с высокой вероятностью
                messagebox.showinfo("Возможные альтернативы", 
                                "Есть другие вероятные варианты")

        except Exception as e:
            error_msg = f"Ошибка при распознавании: {str(e)}"
            messagebox.showerror("Ошибка", error_msg)
            self.status_bar.config(text=error_msg)
            print(f"\n!!! Ошибка: {e}")

    
    
    def init_database(self):
        """Инициализация базы данных"""
        if not os.path.exists(self.db_file):
            columns = ["id", "image_path", "predicted", "confidence", "timestamp", "user"]
            with open(self.db_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)
        
        # Загрузка данных
        self.df = pd.read_csv(self.db_file) if os.path.exists(self.db_file) else pd.DataFrame()
    
    def create_widgets(self):
        """Создание всех элементов интерфейса"""
        # Меню
        self.create_menu()
        
        # Панель инструментов
        self.create_toolbar()
        
        # Основные фреймы
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Левая панель (ввод данных)
        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Правая панель (результаты и данные)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Элементы левой панели
        self.create_input_panel(left_panel)
        
        # Элементы правой панели
        self.create_results_panel(right_panel)
        self.create_data_table(right_panel)
        
        # Статус бар
        self.status_bar = ttk.Label(self.root, text="Готов к работе", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=5)
    
    def create_menu(self):
        """Создание меню приложения"""
        menubar = tk.Menu(self.root)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Экспорт в CSV", command=self.export_to_csv)
        file_menu.add_command(label="Экспорт в Excel", command=self.export_to_excel)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)
        
        # Меню Настройки
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Сменить пользователя", command=self.show_login_dialog)
        settings_menu.add_command(label="Тема: Светлая", command=lambda: self.set_theme("light"))
        settings_menu.add_command(label="Тема: Темная", command=lambda: self.set_theme("dark"))
        
        # Админские функции
        if self.current_user and self.users.get(self.current_user, {}).get("role") == "admin":
            settings_menu.add_separator()
            settings_menu.add_command(label="Управление пользователями", command=self.manage_users)
        
        menubar.add_cascade(label="Настройки", menu=settings_menu)
        
        # Меню Помощь
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="О программе", command=self.show_about)
        menubar.add_cascade(label="Помощь", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(fill=tk.X)
        
        # Кнопки панели инструментов
        ttk.Button(toolbar, text="Загрузить изображение", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Сделать прогноз", command=self.predict).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Добавить в БД", command=self.add_to_database).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Графики", command=self.show_graphs_dialog).pack(side=tk.LEFT, padx=2)
        
        # Отображение текущего пользователя
        self.user_label = ttk.Label(toolbar, text=f"Пользователь: {self.current_user if self.current_user else 'Не авторизован'}")
        self.user_label.pack(side=tk.RIGHT, padx=10)
    
    def create_input_panel(self, parent):
        """Создание панели ввода данных"""
        input_frame = ttk.LabelFrame(parent, text="Ввод данных", padding=10)
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        # Холст для отображения изображения
        self.canvas = tk.Canvas(input_frame, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)
        
        # Кнопка загрузки изображения
        ttk.Button(input_frame, text="Выбрать изображение", command=self.load_image).pack(fill=tk.X)
        
        # Параметры для прогноза
        params_frame = ttk.LabelFrame(input_frame, text="Параметры", padding=10)
        params_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(params_frame, text="Размер изображения:").grid(row=0, column=0, sticky=tk.W)
        self.size_slider = ttk.Scale(params_frame, from_=20, to=40, value=28)
        self.size_slider.grid(row=0, column=1, sticky=tk.EW)
        
        ttk.Label(params_frame, text="Яркость:").grid(row=1, column=0, sticky=tk.W)
        self.brightness_slider = ttk.Scale(params_frame, from_=0, to=2, value=1)
        self.brightness_slider.grid(row=1, column=1, sticky=tk.EW)
        
        # Кнопка прогноза
        ttk.Button(input_frame, text="Распознать цифру", command=self.predict).pack(fill=tk.X, pady=5)
    
    def create_results_panel(self, parent):
        """Создание панели результатов"""
        results_frame = ttk.LabelFrame(parent, text="Результаты", padding=10)
        results_frame.pack(fill=tk.X, pady=5)
        
        # Отображение предсказания
        self.result_label = ttk.Label(results_frame, text="Результат: -", font=("Arial", 14))
        self.result_label.pack()
        
        # График вероятностей
        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=results_frame)
        self.canvas_fig.get_tk_widget().pack(fill=tk.X)
        
        # История предсказаний
        history_frame = ttk.LabelFrame(results_frame, text="История", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.history_text = ScrolledText(history_frame, height=5)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        self.history_text.config(state=tk.DISABLED)
    
    def create_data_table(self, parent):
        """Создание таблицы с данными"""
        table_frame = ttk.LabelFrame(parent, text="База данных", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Создание Treeview
        self.tree = ttk.Treeview(table_frame, columns=("ID", "Image", "Predicted", "Confidence", "Date", "User"), show="headings")
        
        # Настройка колонок
        self.tree.heading("ID", text="ID")
        self.tree.heading("Image", text="Изображение")
        self.tree.heading("Predicted", text="Предсказание")
        self.tree.heading("Confidence", text="Уверенность")
        self.tree.heading("Date", text="Дата")
        self.tree.heading("User", text="Пользователь")
        
        self.tree.column("ID", width=50)
        self.tree.column("Image", width=150)
        self.tree.column("Predicted", width=80)
        self.tree.column("Confidence", width=80)
        self.tree.column("Date", width=120)
        self.tree.column("User", width=100)
        
        # Полоса прокрутки
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Заполнение таблицы данными
        self.update_table()
    
    def load_image(self):
        """Загрузка изображения"""
        file_path = filedialog.askopenfilename(filetypes=[("Изображения", "*.png;*.jpg;*.jpeg")])
        if file_path:
            try:
                self.current_image_path = file_path
                self.image = Image.open(file_path).convert('L')
                
                # Применяем параметры
                size = int(self.size_slider.get())
                brightness = float(self.brightness_slider.get())
                
                # Изменяем размер и яркость
                img = self.image.resize((size, size))
                img = img.point(lambda p: p * brightness)
                
                # Отображаем на холсте
                img_tk = ImageTk.PhotoImage(img.resize((280, 280)))
                
                self.canvas.delete("all")
                self.canvas.create_image(140, 140, image=img_tk)
                self.canvas.image = img_tk
                
                self.status_bar.config(text=f"Загружено: {file_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {str(e)}")
                self.status_bar.config(text="Ошибка загрузки изображения")
    
    
    
    def update_probability_plot(self, probabilities):
        """Обновление графика вероятностей для всех цифр 0-9"""
        self.ax.clear()
        
        # Создаем список цветов - оранжевый для максимальной вероятности, голубой для остальных
        colors = ['skyblue'] * 10
        predicted_digit = np.argmax(probabilities)
        colors[predicted_digit] = 'orange'
        
        # Создаем столбчатую диаграмму
        bars = self.ax.bar(range(10), probabilities, color=colors, edgecolor='white', linewidth=1)
        
        # Добавляем значения вероятностей над столбцами
        for i, bar in enumerate(bars):
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1%}',
                        ha='center', va='bottom', fontsize=9)
        
        # Настраиваем внешний вид графика
        self.ax.set_title("Вероятности цифр", fontsize=12, pad=10)
        self.ax.set_xlabel("Цифра", fontsize=10)
        self.ax.set_ylabel("Вероятность", fontsize=10)
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 1.1)  # Оставляем место для текста над столбцами
        
        # Добавляем сетку для лучшей читаемости
        self.ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        
        # Убираем верхнюю и правую границы
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        
        # Обновляем холст
        self.canvas_fig.draw()
    
    def update_history(self):
        """Обновление истории предсказаний"""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.delete(1.0, tk.END)
        
        for item in reversed(self.history[-5:]):  # Показываем последние 5 записей
            self.history_text.insert(tk.END, 
                                   f"{item['timestamp']}: {item['predicted']} (уверенность: {item['confidence']:.1%})\n"
                                   f"Файл: {item['image']}\nПользователь: {item['user']}\n\n")
        
        self.history_text.config(state=tk.DISABLED)
    
    def add_to_database(self):
        """Добавление записи в базу данных"""
        if not self.current_user:
            messagebox.showwarning("Предупреждение", "Сначала авторизуйтесь")
            return
            
        if hasattr(self, 'image') and hasattr(self, 'history') and self.history:
            last_pred = self.history[-1]
            
            new_id = len(self.df) + 1 if not self.df.empty else 1
            
            new_row = {
                "id": new_id,
                "image_path": last_pred['image'],
                "predicted": last_pred['predicted'],
                "confidence": last_pred['confidence'],
                "timestamp": last_pred['timestamp'],
                "user": self.current_user
            }
            
            # Добавляем в DataFrame
            self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
            
            # Сохраняем в CSV
            self.df.to_csv(self.db_file, index=False)
            
            # Обновляем таблицу
            self.update_table()
            
            messagebox.showinfo("Успех", "Запись добавлена в базу данных")
            self.status_bar.config(text="Запись добавлена в базу данных")
        else:
            messagebox.showwarning("Предупреждение", "Нет данных для добавления в базу")
    
    def update_table(self):
        """Обновление таблицы данными"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for _, row in self.df.iterrows():
            self.tree.insert("", tk.END, values=(
                row['id'],
                os.path.basename(row['image_path']),
                row['predicted'],
                f"{row['confidence']:.1%}",
                row['timestamp'],
                row['user']
            ))
    
    def export_to_csv(self):
        """Экспорт данных в CSV"""
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.df.to_csv(file_path, index=False)
            self.status_bar.config(text=f"Данные экспортированы в {file_path}")
    
    def export_to_excel(self):
        """Экспорт данных в Excel"""
        try:
            import openpyxl
            file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
            if file_path:
                self.df.to_excel(file_path, index=False)
                self.status_bar.config(text=f"Данные экспортированы в {file_path}")
        except ImportError:
            messagebox.showerror("Ошибка", "Для экспорта в Excel требуется библиотека openpyxl")
    
    def show_graphs_dialog(self):
        """Диалог выбора графиков"""
        if not self.df.empty:
            dialog = tk.Toplevel(self.root)
            dialog.title("Построение графиков")
            dialog.geometry("400x300")
            
            ttk.Label(dialog, text="Выберите тип графика:").pack(pady=10)
            
            graph_types = [
                "Распределение предсказанных цифр",
                "Зависимость уверенности от пользователя",
                "Количество распознаваний по дням"
            ]
            
            for gtype in graph_types:
                ttk.Button(dialog, text=gtype, command=lambda t=gtype: self.show_graph(t)).pack(fill=tk.X, padx=20, pady=5)
        else:
            messagebox.showwarning("Предупреждение", "Нет данных для построения графиков")
    
    def show_graph(self, graph_type):
        """Отображение выбранного графика"""
        fig, ax = plt.subplots(figsize=(8, 5))
        
        if graph_type == "Распределение предсказанных цифр":
            sns.countplot(data=self.df, x='predicted', ax=ax)
            ax.set_title("Распределение предсказанных цифр")
            ax.set_xlabel("Цифра")
            ax.set_ylabel("Количество")
        
        elif graph_type == "Зависимость уверенности от пользователя":
            sns.boxplot(data=self.df, x='user', y='confidence', ax=ax)
            ax.set_title("Уверенность предсказаний по пользователям")
            ax.set_xlabel("Пользователь")
            ax.set_ylabel("Уверенность")
        
        elif graph_type == "Количество распознаваний по дням":
            self.df['date'] = pd.to_datetime(self.df['timestamp']).dt.date
            daily_counts = self.df.groupby('date').size()
            daily_counts.plot(kind='bar', ax=ax)
            ax.set_title("Количество распознаваний по дням")
            ax.set_xlabel("Дата")
            ax.set_ylabel("Количество")
        
        plt.tight_layout()
        
        # Отображение в отдельном окне
        graph_window = tk.Toplevel(self.root)
        graph_window.title(graph_type)
        
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(graph_window, text="Сохранить как PNG", 
                  command=lambda: self.save_figure(fig)).pack(pady=5)
    
    def save_figure(self, fig):
        """Сохранение графика в файл"""
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            fig.savefig(file_path)
            self.status_bar.config(text=f"График сохранен в {file_path}")
    
    def show_login_dialog(self):
        """Диалог авторизации"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Авторизация")
        dialog.geometry("300x300")
        dialog.resizable(False, False)
        
        ttk.Label(dialog, text="Логин:").pack(pady=5)
        username_entry = ttk.Entry(dialog)
        username_entry.pack(pady=5)
        
        ttk.Label(dialog, text="Пароль:").pack(pady=5)
        password_entry = ttk.Entry(dialog, show="*")
        password_entry.pack(pady=5)
        
        def try_login():
            username = username_entry.get()
            password = password_entry.get()
            
            if username in self.users and self.users[username]["password"] == password:
                self.current_user = username
                self.user_label.config(text=f"Пользователь: {username}")
                self.create_menu()  # Обновляем меню с учетом прав
                dialog.destroy()
                messagebox.showinfo("Успех", f"Добро пожаловать, {username}!")
            else:
                messagebox.showerror("Ошибка", "Неверный логин или пароль")
        
        ttk.Button(dialog, text="Войти", command=try_login).pack(pady=10)
        
        # Кнопка регистрации (только если не авторизован)
        if not self.current_user:
            ttk.Button(dialog, text="Зарегистрироваться", 
                      command=lambda: self.show_register_dialog(dialog)).pack(pady=5)
        
        dialog.transient(self.root)
        dialog.grab_set()
        self.root.wait_window(dialog)
    
    def show_register_dialog(self, parent):
        """Диалог регистрации нового пользователя"""
        dialog = tk.Toplevel(parent)
        dialog.title("Регистрация")
        dialog.geometry("300x250")
        
        ttk.Label(dialog, text="Логин:").pack(pady=5)
        username_entry = ttk.Entry(dialog)
        username_entry.pack(pady=5)
        
        ttk.Label(dialog, text="Пароль:").pack(pady=5)
        password_entry = ttk.Entry(dialog, show="*")
        password_entry.pack(pady=5)
        
        ttk.Label(dialog, text="Подтвердите пароль:").pack(pady=5)
        confirm_entry = ttk.Entry(dialog, show="*")
        confirm_entry.pack(pady=5)
        
        def register():
            username = username_entry.get()
            password = password_entry.get()
            confirm = confirm_entry.get()
            
            if not username or not password:
                messagebox.showerror("Ошибка", "Логин и пароль не могут быть пустыми")
                return
            
            if password != confirm:
                messagebox.showerror("Ошибка", "Пароли не совпадают")
                return
            
            if username in self.users:
                messagebox.showerror("Ошибка", "Пользователь уже существует")
                return
            
            self.users[username] = {"password": password, "role": "user"}
            self.save_users()
            messagebox.showinfo("Успех", "Регистрация прошла успешно!")
            dialog.destroy()
        
        ttk.Button(dialog, text="Зарегистрироваться", command=register).pack(pady=10)
    
    def manage_users(self):
        """Управление пользователями (для администратора)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Управление пользователями")
        dialog.geometry("500x400")
        
        tree = ttk.Treeview(dialog, columns=("username", "role"), show="headings")
        tree.heading("username", text="Логин")
        tree.heading("role", text="Роль")
        
        for username, data in self.users.items():
            tree.insert("", tk.END, values=(username, data["role"]))
        
        tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Функции управления
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(frame, text="Удалить", 
                  command=lambda: self.delete_user(tree)).pack(side=tk.LEFT)
        
        ttk.Button(frame, text="Изменить роль", 
                  command=lambda: self.change_user_role(tree)).pack(side=tk.LEFT, padx=5)
    
    def delete_user(self, tree):
        """Удаление пользователя"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите пользователя")
            return
        
        username = tree.item(selected[0])["values"][0]
        if username == self.current_user:
            messagebox.showerror("Ошибка", "Нельзя удалить текущего пользователя")
            return
        
        if username == "admin":
            messagebox.showerror("Ошибка", "Нельзя удалить администратора")
            return
        
        if messagebox.askyesno("Подтверждение", f"Удалить пользователя {username}?"):
            del self.users[username]
            self.save_users()
            tree.delete(selected[0])
    
    def change_user_role(self, tree):
        """Изменение роли пользователя"""
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите пользователя")
            return
        
        username = tree.item(selected[0])["values"][0]
        if username == "admin":
            messagebox.showerror("Ошибка", "Нельзя изменить роль администратора")
            return
        
        current_role = self.users[username]["role"]
        new_role = "admin" if current_role == "user" else "user"
        
        self.users[username]["role"] = new_role
        self.save_users()
        tree.item(selected[0], values=(username, new_role))
        messagebox.showinfo("Успех", f"Роль пользователя {username} изменена на {new_role}")
    
    def set_theme(self, theme):
        """Установка цветовой темы"""
        self.current_theme = theme
        self.apply_theme()
        self.save_settings()
    
    def apply_theme(self):
        """Применение выбранной темы"""
        try:
            style = ttk.Style()
            
            if self.current_theme == "dark":
                # Цвета для темной темы
                bg = "#2d2d2d"
                fg = "#ffffff"
                widget_bg = "#3d3d3d"
                widget_fg = "#ffffff"
                select_bg = "#4a6987"
                select_fg = "#ffffff"
                hover_bg = "#505050"
                pressed_bg = "#404040"
                border_color = "#5a5a5a"
                style.theme_use('alt')
                
                # Настройка стилей для кнопок
                style.configure('TButton', 
                            background=widget_bg,
                            foreground=fg,
                            bordercolor=border_color,
                            borderwidth=1,
                            relief="raised",
                            padding=6,
                            font=('Arial', 10))
                
                style.map('TButton',
                        background=[('active', hover_bg), ('pressed', pressed_bg)],
                        relief=[('pressed', 'sunken'), ('!pressed', 'raised')],
                        bordercolor=[('active', '#707070')],
                        lightcolor=[('pressed', pressed_bg), ('!pressed', hover_bg)],
                        darkcolor=[('pressed', pressed_bg), ('!pressed', hover_bg)])
            else:
                bg = "#f0f0f0"
                fg = "#000000"
                widget_bg = "#ffffff"
                widget_fg = "#000000"
                style.theme_use('clam')  # Светлая тема
            
            # Настройка стилей для ttk-виджетов
            style.configure('.', background=bg, foreground=fg)
            style.configure('TFrame', background=bg)
            style.configure('TLabel', background=bg, foreground=fg)
            style.configure('TButton', background=widget_bg, foreground=fg)
            
            style.configure('TEntry', fieldbackground=widget_bg, foreground=fg)
            style.configure('TCombobox', fieldbackground=widget_bg, foreground=fg)
            style.configure('Treeview', background=widget_bg, foreground=fg, fieldbackground=widget_bg)
            style.map('Treeview', background=[('selected', '#347083')])
            
            # Применяем цвета к основным виджетам
            self.root.config(bg=bg)
            
            # Рекурсивно применяем тему ко всем виджетам
            for widget in self.root.winfo_children():
                self.apply_theme_to_widget(widget, bg, fg, widget_bg, widget_fg)
                
        except Exception as e:
            print(f"Ошибка применения темы: {str(e)}")
    
    def apply_theme_to_widget(self, widget, bg, fg, widget_bg, widget_fg):
        """Рекурсивное применение темы к виджетам"""
        try:
            # Пропускаем ttk-виджеты (они используют стили)
            if isinstance(widget, ttk.Widget):
                return
                
            # Обрабатываем стандартные tkinter-виджеты
            if isinstance(widget, (tk.Menu, tk.Toplevel)):
                widget.config(bg=bg, fg=fg)
                
            elif isinstance(widget, (tk.Frame, tk.LabelFrame, tk.Canvas)):
                widget.config(bg=bg)
                
            elif isinstance(widget, (tk.Label, tk.Button, tk.Entry)):
                widget.config(bg=widget_bg, fg=fg)
                
            elif isinstance(widget, (tk.Text, ScrolledText)):
                widget.config(bg=widget_bg, fg=fg, insertbackground=fg)
                
            elif isinstance(widget, tk.Scrollbar):
                widget.config(bg=widget_bg, troughcolor=bg)
                
            # Применяем к дочерним виджетам
            for child in widget.winfo_children():
                self.apply_theme_to_widget(child, bg, fg, widget_bg, widget_fg)
                
        except Exception as e:
            print(f"Ошибка применения темы к {widget}: {str(e)}")
    
    def load_settings(self):
        """Загрузка настроек из файла"""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    self.current_theme = settings.get("theme", "light")
            except:
                pass
    
    def save_settings(self):
        """Сохранение настроек в файл"""
        try:
            settings = {
                "theme": self.current_theme
            }
            
            with open(self.settings_file, "w") as f:
                json.dump(settings, f)
        except Exception as e:
            print(f"Ошибка сохранения настроек: {str(e)}")
    
    def show_about(self):
        """Окно 'О программе'"""
        about_text = """
        MNIST Classifier
        
        Версия 1.0
        Разработано для распознавания рукописных цифр
        
        Используемые технологии:
        - Python 3.6+
        - TensorFlow/Keras для модели распознавания
        - Pandas для работы с данными
        - Matplotlib для визуализации
        
        Для работы требуется файл модели mnist_model.h5
        """
        
        messagebox.showinfo("О программе", about_text)
    
    def __del__(self):
        """Деструктор - сохраняем настройки при выходе"""
        try:
            if hasattr(self, 'settings_file') and hasattr(self, 'current_theme'):
                self.save_settings()
            if hasattr(self, 'users_file') and hasattr(self, 'users'):
                self.save_users()
        except Exception as e:
            print(f"Ошибка при сохранении данных: {str(e)}")
            self.save_users()

if __name__ == "__main__":
    root = tk.Tk()
    app = MNISTApp(root)
    root.mainloop()