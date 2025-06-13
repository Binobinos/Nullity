import csv
import json
import math
import os
import threading
from tkinter import ttk, filedialog, Tk, Canvas, Frame, LEFT, Label, X, GROOVE, HORIZONTAL
from typing import List, Tuple, Union, Dict

import numpy as np
from PIL import Image

from Nullity.NullityNetworkMini import Neuralnetwork


def pressed_array(data: List[List[int]]) -> List[float]:
    """
    Преобразует 32x32 мерный массив в 1024 мерный массив
    :param data: 32x32 мерный массив чисел
    :return: 1024 мерный массив чисел
    """
    new_list = []
    for dt in data:
        new_list.extend(dt)

    def pressed_int(num: int):
        return int(num) / 256

    return list(map(pressed_int, new_list))


def one_hot_vector_number(number: int) -> List[int]:
    """
    Возвращает one hot вектор числа
    :param number: Число
    :return: one hot вектор
    """
    return [0 if number - 1 != num else 1 for num in range(33)]


class RussianNetwork:
    def __init__(self, layers: List[Tuple[int, str]],
                 PATH_TO_DATASET: str,
                 epoch: int = 1,
                 learning_rate: float = 0.1) -> None:
        """
        Инициализация
        :param layers: Список слоёв в формате List[Tuple[количество нейронов в слое, функция активации]]
        :param PATH_TO_DATASET: путь к датасету чисел
        :param epoch: Количество эпох в обучении модели
        :param learning_rate: Скорость обучения модели
        """
        self.data = []
        self.path_dataset = PATH_TO_DATASET
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.network = Neuralnetwork()
        for layer in layers:
            self.network.add_layer(layer[0], layer[1])

    def preset_data(self):
        """
        Подготавливает данные для обучения модели
        :return:
        """
        self.data = []
        print(f"Загружаем датасет")
        with (open(os.path.join(self.path_dataset, "all_letters_info.csv"), "r", encoding="utf-8") as file):
            csv_file = csv.reader(file)
            for data in csv_file:
                image = Image.open(os.path.join(self.path_dataset, r"all_letters_image\all_letters_image", data[2]))
                image = image.convert("L")
                data_list = pressed_array(np.array(image).tolist())
                self.data.append((data_list, one_hot_vector_number(int(data[1]))))

    def train(self, is_all_update=True) -> None:
        """
        Запускает обучение модели
        :return:
        """
        self.preset_data()
        if is_all_update:
            self.network.calculating_weights()
        self.network.train(data=self.data, epoch=self.epoch, learning_rate=self.learning_rate)

    def predict(self, data: List[int]) -> List[Union[int, float]]:
        return self.network.predict(data)

    def save_weights(self, path_to_save: str) -> None:
        with open(path_to_save, "w") as file:
            file.writelines(json.dumps(self.network.return_weights(), indent=4))

    def load_weights(self, path_to_save: str) -> None:
        with open(path_to_save, "r") as file:
            self.network.weights, self.network.biases = json.loads("\n".join(file.readlines()))


class GuiNetworkProgram:
    def __init__(self, path: str):
        self.root = Tk()
        self.root.title("Распознавание рукописных Русских букв")
        self.root.geometry("1200x700")  # Увеличили размер окна

        # Нейросеть
        self.network = RussianNetwork(
            [(1025, "SIGMOID"), (512, "SIGMOID"), (128, "SIGMOID"), (33, "SIGMOID")],
            path
        )

        # Переменные для рисования
        self.rectangles = []
        self.rectangles_color = [[255 for _ in range(32)] for _ in range(32)]
        self.color_step = 25  # Увеличили шаг изменения цвета
        self.drawing = False
        self.breaking = False
        self.brush_radius = 1
        self.cursor_radius_indicator = None
        self.epochs_ = ttk.Entry(self.root)
        self.epochs_.place(x=770, y=620)
        # Буквы русского алфавита
        self.letters = [
            'А', 'Б', 'В', 'Г', 'Д', 'Е', 'Ё', 'Ж', 'З', 'И', 'Й',
            'К', 'Л', 'М', 'Н', 'О', 'П', 'Р', 'С', 'Т', 'У', 'Ф',
            'Х', 'Ц', 'Ч', 'Ш', 'Щ', 'Ъ', 'Ы', 'Ь', 'Э', 'Ю', 'Я'
        ]
        self.letter_progress = {letter: 0.0 for letter in self.letters}
        self.letter_widgets = {}  # Будет хранить виджеты букв

        self.setup_ui()

    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Основные кнопки управления
        ttk.Button(self.root, text="Сохранить модель", command=self.save_model).place(x=20, y=620)
        ttk.Button(self.root, text="Загрузить модель", command=self.load_model).place(x=180, y=620)
        ttk.Button(self.root, text="Обучить модель", command=self.train_model).place(x=340, y=620)
        ttk.Button(self.root, text="Распознать", command=self.predict_letter).place(x=500, y=620)
        ttk.Button(self.root, text="Очистить", command=self.clear_canvas).place(x=660, y=620)
        # Холст для рисования
        self.canvas = Canvas(self.root, bg="white", width=512, height=512)
        self.canvas.place(x=20, y=20)
        self.setup_canvas()

        # Панель с буквами
        self.setup_letters_panel()

    def setup_canvas(self):
        """Настройка холста для рисования"""
        for y in range(32):
            self.rectangles.append([])
            for x in range(32):
                rect = self.canvas.create_rectangle(
                    x * 16, y * 16, (x + 1) * 16, (y + 1) * 16,
                    fill="#ffffff", outline="#e0e0e0", width=1,
                    tags=[f"{x}{y}"]
                )
                self.rectangles[y].append(rect)

        # Привязка событий мыши
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", lambda e: setattr(self, "drawing", True))
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "drawing", False))
        self.canvas.bind("<Button-3>", lambda e: setattr(self, "breaking", True))
        self.canvas.bind("<ButtonRelease-3>", lambda e: setattr(self, "breaking", False))
        self.canvas.bind("<MouseWheel>", self.change_brush_radius)

    def setup_letters_panel(self):
        """Настройка панели с буквами"""
        letters_frame = Frame(self.root, bg="#f0f0f0", bd=2, relief=GROOVE)
        letters_frame.place(x=560, y=20, width=600, height=580)

        Label(letters_frame,
              text="Результаты распознавания",
              font=("Arial", 14, "bold"),
              bg="#f0f0f0").pack(pady=10)

        # Создаем фрейм для прокрутки
        canvas = Canvas(letters_frame, bg="#f0f0f0", highlightthickness=0)
        scrollbar = ttk.Scrollbar(letters_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = Frame(canvas, bg="#f0f0f0")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Создаем виджеты для каждой буквы
        for letter in self.letters:
            frame = Frame(scrollable_frame, bg="#f0f0f0")
            frame.pack(fill=X, padx=10, pady=2)

            # Метка с буквой
            lbl = Label(frame, text=letter, font=("Arial", 14), width=3, bg="#f0f0f0")
            lbl.pack(side=LEFT)

            # Прогресс-бар
            progress = ttk.Progressbar(frame, orient=HORIZONTAL, length=300)
            progress.pack(side=LEFT, padx=5, expand=True, fill=X)

            # Метка с процентом
            percent = Label(frame, text="0%", font=("Arial", 10), width=4, bg="#f0f0f0")
            percent.pack(side=LEFT)

            # Сохраняем виджеты
            self.letter_widgets[letter] = {
                "label": lbl,
                "progress": progress,
                "percent": percent
            }

    def update_letter_progress(self, new_progress: Dict[str, float]):
        """Обновляет прогресс букв по словарю"""
        for letter, progress in new_progress.items():
            if letter in self.letter_progress:
                # Ограничиваем значение от 0 до 1
                progress = max(0.0, min(1.0, progress))
                self.letter_progress[letter] = progress

                # Обновляем прогресс-бар
                self.letter_widgets[letter]["progress"]["value"] = progress * 100

                # Обновляем процент
                self.letter_widgets[letter]["percent"]["text"] = f"{int(progress * 100)}%"

                # Меняем цвет текста в зависимости от уверенности
                color = "#%02x%02x%02x" % (
                    int(255 * (1 - progress)),
                    int(255 * progress),
                    100
                )
                self.letter_widgets[letter]["label"]["fg"] = color

    def color_to_str(self, color):
        """Преобразует значение цвета в HEX-формат"""
        color = max(0, min(255, color))
        return f"#{hex(color)[2:].zfill(2) * 3}"

    def draw_with_radius(self, x, y, is_drawing):
        """Обрабатывает рисование с учетом радиуса кисти"""
        for dy in range(-self.brush_radius, self.brush_radius + 1):
            for dx in range(-self.brush_radius, self.brush_radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < 32 and 0 <= ny < 32:
                    if math.sqrt(dx ** 2 + dy ** 2) <= self.brush_radius:
                        if is_drawing:
                            self.rectangles_color[ny][nx] = max(0, self.rectangles_color[ny][nx] - self.color_step)
                        else:
                            self.rectangles_color[ny][nx] = min(255, self.rectangles_color[ny][nx] + self.color_step)
                        self.canvas.itemconfig(
                            self.rectangles[ny][nx],
                            fill=self.color_to_str(self.rectangles_color[ny][nx])
                        )

    def update_cursor_indicator(self, x, y):
        """Обновляет индикатор радиуса кисти"""
        if self.cursor_radius_indicator:
            self.canvas.delete(self.cursor_radius_indicator)

        if 0 <= x < 32 and 0 <= y < 32:
            self.cursor_radius_indicator = self.canvas.create_oval(
                (x - self.brush_radius) * 16,
                (y - self.brush_radius) * 16,
                (x + self.brush_radius + 1) * 16,
                (y + self.brush_radius + 1) * 16,
                outline="red", dash=(2, 2), width=1
            )

    def change_brush_radius(self, event):
        """Изменяет радиус кисти колесиком мыши"""
        if event.delta > 0:
            self.brush_radius = min(5, self.brush_radius + 1)
        else:
            self.brush_radius = max(1, self.brush_radius - 1)
        self.update_cursor_indicator(event.x // 16, event.y // 16)

    def on_mouse_move(self, event):
        """Обработчик движения мыши"""
        x, y = event.x // 16, event.y // 16
        self.update_cursor_indicator(x, y)
        if self.drawing or self.breaking:
            self.draw_with_radius(x, y, self.drawing)

    def predict_letter(self):
        """Предсказывает букву и обновляет индикаторы"""
        # Преобразуем холст в массив данных
        input_data = []
        for y in range(32):
            for x in range(32):
                # Нормализуем значения (0-255 -> 0.0-1.0)
                input_data.append((255 - self.rectangles_color[y][x]) / 256)

        # Получаем предсказания от нейросети
        predictions = self.network.predict(input_data)

        # Создаем словарь для обновления
        progress_dict = {}
        for i, letter in enumerate(self.letters):
            progress_dict[letter] = predictions[i]

        # Обновляем интерфейс
        self.update_letter_progress(progress_dict)

    def clear_canvas(self):
        """Очищает холст"""
        for y in range(32):
            for x in range(32):
                self.rectangles_color[y][x] = 255
                self.canvas.itemconfig(
                    self.rectangles[y][x],
                    fill="#ffffff"
                )
        # Сбрасываем прогресс букв
        self.update_letter_progress({letter: 0.0 for letter in self.letters})

    def save_model(self):
        """Сохраняет веса модели"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            self.network.save_weights(filename)

    def load_model(self):
        """Загружает веса модели"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")]
        )
        if filename:
            self.network.load_weights(filename)

    def train_model(self):
        """Обучает модель"""
        if self.epochs_:
            print(f"Начинаем обучение")
            self.network.epoch = int(self.epochs_.get())
            job = threading.Thread(target=self.network.train)
            job.start()

    def main(self):
        """Запускает приложение"""
        self.root.mainloop()


if __name__ == "__main__":
    app = GuiNetworkProgram(
        r'C:\Users\Misha\.cache\kagglehub\datasets\tatianasnwrt\russian-handwritten-letters\versions\18')
    app.main()
