import numpy as np
import random
from PIL import Image, ImageTk
from math import floor, ceil
import tkinter as tk
from numba import njit

# Функция для вычисления барицентрических координат
@njit(cache=True)
def barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2):
    denumerator = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2) # Удвоенная площадь треугольника (полигона целиком)
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denumerator # Удвоенная плозадь треугольника, образованного данной точкой и вершинами 1 и 2
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denumerator # Удвоенная площадь треугольника, образованного данной точкой и вершинами 0 и 2
    lambda2 = 1.0 - lambda0 - lambda1 # Так как барицентрические координаты в сумме дают 1, вычисляем лямбда 2
    return lambda0, lambda1, lambda2

# Функция отрисовки треугольника
@njit(cache=True)
def triangle(x0, y0, x1, y1, x2, y2, z0, z1, z2, img_mat, brightness, z_buffer, color, brightness_koef): # x0, y0, x1, y1, x2, y2 - экранные коорлинаты, z0, z1, z2 - глубина вершины

    H, W = img_mat.shape[0], img_mat.shape[1] # Определяем ширину и высоту по матрице

    # Ограничивающий прямоугольник (описан вокруг треугольника, чтобы не перебирать все пиксели), определяем границы с округлением, чтобы захватить все пиксели с учетом дробных частей
    xmin = floor(min(min(x0, x1), x2))
    xmax = ceil(max(max(x0, x1), x2))
    ymin = floor(min(min(y0, y1), y2))
    ymax = ceil(max(max(y0, y1), y2))

    # Не рисуем треугольник, если он целиком вне экрана
    if xmax < 0 or xmin >= W or ymax < 0 or ymin >= H:
        return

    # Частичное отсечение, если только часть ограничивающего прямоугольника вне экрана
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax >= W: xmax = W - 1
    if ymax >= H: ymax = H - 1

    # Построчно проходим по ограничивающему прямоугольнику включая ymax и xmax
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            l0, l1, l2 = barycentric_coordinates(x + 0.5, y + 0.5, x0, y0, x1, y1, x2, y2) # Cчитаем барицентрические координаты для центра текущего пикселя
            if l0 >= 0 and l1 >= 0 and l2 >= 0: # Если точка внутри треугольника
                z = l0 * z0 + l1 * z1 + l2 * z2 # Линейная интерполяция z по треугольнику (считаем z текущей точки по z вершин треугольника)
                # Отрисовываем то, что ближе к нам по z и обновляем z-буфер
                if z < z_buffer[y, x]: # Определение глубины для текущего пикселя и сравнение с z-буфером
                    # Считаем конечную яркость
                    final_brightness = brightness * brightness_koef

                    # Считаем цветовые каналы
                    r = int(color[0] * final_brightness)
                    g = int(color[1] * final_brightness)
                    b = int(color[2] * final_brightness)

                    # Проверяем ограничения
                    if r > 255: r = 255
                    if g > 255: g = 255
                    if b > 255: b = 255
                    if r < 0: r = 0
                    if g < 0: g = 0
                    if b < 0: b = 0

                    # Заполняем цветовые каналы на итоговой картинке
                    img_mat[y, x, 0] = r
                    img_mat[y, x, 1] = g
                    img_mat[y, x, 2] = b     
                    z_buffer[y, x] = z # Обновляем z-буфер ближайшей глубиной         

# Функция вычисления нормали к треугольнику
@njit(cache=True)
def normal(v0, v1, v2):

    # Вектор из v1 в v2
    ax = v1[0] - v2[0]
    ay = v1[1] - v2[1]
    az = v1[2] - v2[2]

    # Вектор из v1 в v0
    bx = v1[0] - v0[0]
    by = v1[1] - v0[1]
    bz = v1[2] - v0[2]

    # Векторы a и b образуют плоскость в которой лежит треугольник, значит нормаль к этой плоскости равна нормали к треугольнику (вычисляем по векторному произведению)
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    
    length = np.sqrt(nx * nx + ny * ny + nz * nz) # Длина вектора нормали
    return nx / length, ny / length, nz / length # Возвращаем нормированные координаты

# Функция рендера модели (основной цикл отрисовки)
@njit(cache=True)
def render_model(vertices, faces, img_mat, z_buffer, cx, cy, scale, center_x, center_y, center_z, color, brightness_koef, brightness_add):
    # Сбрасываем изображение(заполняем матрицу чёрным, заполняем z-buffer бесконечностями)
    img_mat[...] = 0
    z_buffer[...] = np.inf

    # Цикл отрисовки модели
    for i in range(len(faces)): # Проходимся по всем полигонам
        # Находим индексы вершин в списке полигонов
        i0 = faces[i, 0]
        i1 = faces[i, 1]
        i2 = faces[i, 2]
        # Находим вершины полигона по найденным индексам вершин
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]

        nx, ny, nz = normal(v0, v1, v2) # Считаем нормаль к полигону

        #if nz >= 0: continue

        # Считаем яркость
        brightness = int(-255 * nz + brightness_add) 
        if brightness < 0: brightness = 0
        if brightness > 255: brightness = 255 

        # Проекция трехмерных координат на двумерные
        px0 = -(v0[0] - center_x) * scale + cx
        py0 = -(v0[2] - center_z) * scale + cy
        
        px1 = -(v1[0] - center_x) * scale + cx
        py1 = -(v1[2] - center_z) * scale + cy
        
        px2 = -(v2[0] - center_x) * scale + cx
        py2 = -(v2[2] - center_z) * scale + cy

        # Отрисовка полигона
        triangle(px0, py0, px1, py1, px2, py2, v0[1], v1[1], v2[1], img_mat, brightness, z_buffer, color, brightness_koef)

# Собираем сцену (класс для входных данных из файла)
class Scene:
    # Инициализация сцены (собираем вершины и полигоны, считаем габариты модели)
    def __init__(self, obj_path):
        self.vertices = [] # Список вершин, каждый элемент - координаты вершины x,y,z
        self.faces = [] # Список полигонов, каждый эелмент хранит индексы вершит (какая с какой соединяется, для образования полигона)
        self.load_obj(obj_path)

        # Превращаем списки в numpy массивы
        self.vertices = np.array(self.vertices, dtype=np.float64)
        self.faces = np.array(self.faces, dtype=np.int32)

        # Вычисление габаритов модели (ограничивающий параллелепипед, в который вписана модель)
        min_vals = self.vertices.min(axis=0) # Минимальные значение по осям
        max_vals = self.vertices.max(axis=0) # Максимальные значения по осям

        # Находим центры по осям
        self.center_x = (min_vals[0] + max_vals[0]) / 2.0
        self.center_y = (min_vals[1] + max_vals[1]) / 2.0
        self.center_z = (min_vals[2] + max_vals[2]) / 2.0

        # Масштаб берем по максимальной стороне x, y или z (считаем размеры сторон и определяем самую большую)
        self.size_x = max_vals[0] - min_vals[0]
        self.size_y = max_vals[1] - min_vals[1]
        self.size_z = max_vals[2] - min_vals[2]
        self.max_size = max(self.size_x, self.size_y, self.size_z)

        if self.max_size == 0: self.max_size = 1 
        self.scale = 900.0 / self.max_size # Вычисляем масштабный коэффициент, максимальная сторона будет ровно 900 пикселей

        # Задаём цвет по умолчанию
        self.colors = [
            np.array([1.0, 1.0, 1.0]), # Белый
            np.array([0.8, 0.2, 0.2]), # Красный
            np.array([0.2, 0.8, 0.2]), # Зеленый
            np.array([0.2, 0.2, 0.8]), # Синий
            np.array([0.8, 0.8, 0.2]), # Желтый
            np.array([0.8, 0.2, 0.8]), # Фиолетовый
        ]
        self.color_index = 0
        self.color = self.colors[0]

        # Коэфициенты для яркости
        self.brightness_koef = 1.0 
        self.brightness_add = 100.0

    # Метод для парсинга файла
    def load_obj(self, filename):
        with open(filename) as file: # Открытие файла модели
            for s in file: # Парсим файл
                s1 = s.split() # Разбиваем строку по пробельным символам и записываем куски как элементы списка строк
                if not s1:  # если список пустой (строка была пустой), пропускаем её
                    continue
                if(s1[0] == 'v'): # Если первый элемент списка "v", значит в строке координаты вершины
                    self.vertices.append([float(s1[1]), float(s1[2]), float(s1[3])]) # Добавляем вершину в конец списка (элемент списка вершина в виде её координат)
                if(s1[0] == 'f'): # Если первый элемент списка "f", значит в строке полигон (записан в виде a/b/c d/e/f g/h/i где первые числа - индексы вершин, начиная с 1)
                    i0 = int(s1[1].split('/')[0]) - 1 # Разбиваем первую тройку по /, берём только первые числа разбиение и вычитаем 1 сразу для корректировки индексации
                    i1 = int(s1[2].split('/')[0]) - 1 # Также разбиваем вторую тройку по /
                    i2 = int(s1[3].split('/')[0]) - 1 # Также разюиваем третью тройку по /
                    self.faces.append([i0, i1, i2]) # Добавляем в список полигона

# Класс для отрисовки
class Renderer:
    # Метод инициализации рендера (принимаем данные по размеру изображения и находим центр переданного холста)
    def __init__(self, H, W):
        # Присваиваем длину и ширину в объект рендера
        self.W = W 
        self.H = H

        self.img_mat = np.zeros((H, W, 3), dtype=np.uint8) # Трёхмерный массив, матрица чёрных пикселей 1000 на 1000, начальная матрица для отрисовки
        self.z_buffer = np.full((H, W), np.inf) # z-буфер, двумерный массив 1000 на 1000, заполненный бесконечностями

        # Находим центр холста
        self.cx = W / 2
        self.cy = H / 2

    # Метод рендера, вызываем функцию и передаём в нее все переменные для отрисовки, который вне класса для обёртки в njit
    def render(self, scene):
        render_model(scene.vertices, scene.faces, self.img_mat, self.z_buffer, self.cx, self.cy, scene.scale, scene.center_x, scene.center_y, scene.center_z, scene.color, scene.brightness_koef, scene.brightness_add)

# Класс интерфейса
class App:
    def __init__(self, scene, renderer, win_w = 900, win_h = 900):
        self.scene = scene
        self.renderer = renderer

        # Оставляем как есть
        # Окно верхнего уровня
        self.root = tk.Tk()
        # Размеры окна
        self.root.geometry(f"{win_w}x{win_h}")
        # Область для рисования внутри окна
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.focus_set()
        # Обновление изображения при изменении размеров окна
        self.canvas.bind('<Configure>', lambda e: self._refit())
        # Привязка обработчика клавиш
        self.root.bind('<Key>', self.on_key)

        # Изображение внутри окна
        self.photo = None
        self.full_image = None

        self.render()

    # Бинды для изменения масштаба, яркости, цвета и сохранения
    def on_key(self, event):
        k = event.keysym

        if k == 'g':
            self.scene.scale *= 1.1
        elif k == 'l':
            self.scene.scale /= 1.1
        elif k == 'c':
            if(self.scene.color_index < len(self.scene.colors) - 1):
                self.scene.color_index += 1
            else:
                self.scene.color_index = 0
            self.scene.color = self.scene.colors[self.scene.color_index]
        elif k == 'u':
            self.scene.brightness_koef *= 1.1
        elif k == 'd':
            self.scene.brightness_koef /= 1.1
        elif k == 'space':
            Image.fromarray(self.renderer.img_mat).save('img.png')
            return 'break'
        else: 
            return

        self.render()
        # Не обрабатываем дальше – return 'brake'
        return 'break'

    def render(self):
        self.renderer.render(self.scene)
        self.full_image = Image.fromarray(self.renderer.img_mat)
        self._refit()

    # скопировать как есть
    def _refit(self):
        # Если ещё не отрендерилось, пропускаем
        if self.full_image is None: return
        # Ширина и высота области для рисования
        ww = self.canvas.winfo_width()
        wh = self.canvas.winfo_height()
        # Если вдруг сделали область слишком маленькой
        if ww < 10 or wh < 10: return
        # Вписать с сохранением пропорций
        s = min(ww / self.renderer.W, wh / self.renderer.H)
        dw = max(1, int(self.renderer.W * s))
        dh = max(1, int(self.renderer.H * s))
        disp = self.full_image.resize((dw, dh), Image.BILINEAR)
        # Очень важно сохранить ImageTk.PhotoImage(disp) в поле класса
        # Если оно будет локальным, то удалится после выхода сборщиком мусора
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(ww // 2, wh // 2, image=self.photo)   

    # скопировать как есть
    def run(self):
        self.root.mainloop()

# Основной блок кода, передаём имя файла размеры изображения, создаём объекты класса
if __name__ == '__main__':
    scene = Scene('Borderlands cosplay-obj.obj')
    renderer = Renderer(1000, 1000)
    renderer.render(scene)
    App(scene, renderer).run()
              

