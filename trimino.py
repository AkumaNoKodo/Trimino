import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.backends.backend_pdf import PdfPages


class Coordinates:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.xy = [x, y]

    def __str__(self):
        return f"[{self.x}, {self.y}]"


class Text:
    def __init__(self, a: Coordinates, b: Coordinates, rotation: str):
        self.a = a
        self.b = b
        self.rotation = rotation

        self._offset = -10
        self.text = ""

        self.edge_center = self.get_edge_center()
        self.angle_rad = self.get_angle_rad()
        self.angle_deg = np.rad2deg(self.angle_rad if rotation == "up" else self.angle_rad + np.pi)

        self.text_center = self.get_text_center()

    def get_text_center(self):
        center = [self._offset * np.cos(self.angle_rad - np.pi / 2), self._offset * np.sin(self.angle_rad - np.pi / 2)]
        center = np.round(center, decimals=2)

        return center if self.rotation == "up" else [-x for x in center]

    def get_edge_center(self):
        x = (self.a.x + self.b.x) / 2
        y = (self.a.y + self.b.y) / 2
        return Coordinates(x, y)

    def get_angle_rad(self):
        return np.arctan2(self.b.y - self.a.y, self.b.x - self.a.x)

    @staticmethod
    def calculate_text_size(text):
        length = len(text)
        max_text_size = 17
        min_text_size = 12

        min_length = 1
        max_length = 16

        slope = (min_text_size - max_text_size) / (max_length - min_length)

        intercept = max_text_size - slope * min_length

        text_size = slope * length + intercept

        text_size = round(text_size)
        text_size = max(min_text_size, min(text_size, max_text_size))

        return text_size

    def plot_text(self, ax):
        ax.annotate(self.text, xy=(self.edge_center.x, self.edge_center.y), xytext=self.text_center,
                    textcoords="offset points", ha='center', va='center',
                    rotation=self.angle_deg, rotation_mode='anchor', fontsize=self.calculate_text_size(self.text))


class Triangle:
    def __init__(self, center: Coordinates, rotation: str):
        self.center = center
        self.rotation = rotation
        self.peaks = self.get_peaks()

        if rotation == "up":
            self.edges_text = \
                {"left": Text(self.peaks[2], self.peaks[0], self.rotation),
                 "right": Text(self.peaks[1], self.peaks[2], self.rotation),
                 "bottom": Text(self.peaks[0], self.peaks[1], self.rotation)}
        elif rotation == "down":
            self.edges_text = \
                {"left": Text(self.peaks[2], self.peaks[0], self.rotation),
                 "right": Text(self.peaks[1], self.peaks[2], self.rotation),
                 "top": Text(self.peaks[0], self.peaks[1], self.rotation)}
        else:
            raise ValueError("Rotation must be either 'up' or 'down'.")

    def get_peaks(self):
        y_shift = 1 if self.rotation == "up" else -1
        return [Coordinates(self.center.x - 1, self.center.y - 1 * y_shift),
                Coordinates(self.center.x + 1, self.center.y - 1 * y_shift),
                Coordinates(self.center.x, self.center.y + 1 * y_shift)]

    def add_text(self, text: str, position: str):
        if position in self.edges_text:
            self.edges_text[position].text = text
        else:
            print(f"Invalid position on [{text}]")

    def plot(self, ax):
        triangle = plt.Polygon((self.peaks[0].xy, self.peaks[1].xy, self.peaks[2].xy), closed=True,
                               edgecolor='black', facecolor='none', linewidth=2)
        [self.edges_text[text].plot_text(ax) for text in self.edges_text]
        ax.add_patch(triangle)


class CombinationGenerator:
    def __init__(self, file_path: str, list_name: str):
        self.file_path = file_path
        self.pairs = self._get_pairs(list_name)

    def _get_pairs(self, list_name: str):
        shapes_dict = {}
        current_section = None
        with open(self.file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    current_section = line[1:].strip()

                if current_section == list_name:
                    line = line.split(', ')
                    shapes_dict[line[0]] = line[1]

        return shapes_dict

    def generate_random_pairs(self):
        keys = list(self.pairs.keys())
        selected_key = random.choice(keys)
        selected_value = self.pairs[selected_key]
        self.pairs.pop(selected_key)

        return [selected_key, selected_value]


class Grid:
    def __init__(self, pair_list_path: str, list_name: str):
        self.triangles_list = []
        self.text_pairs = CombinationGenerator(pair_list_path, list_name)
        self.ax = None
        self.figure = None

    def add_triangle(self, triangle: Triangle):
        self.triangles_list.append(triangle)

    def plot(self):
        width_in_inches = 210 / 25.4
        height_in_inches = 297 / 25.4

        fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches))
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])

        self._add_text_to_triangles()
        [triangle.plot(ax) for triangle in self.triangles_list]

        ax.autoscale()
        ax.set_axis_off()

        self.ax = ax
        self.figure = fig

    def show(self):
        plt.show(self.ax)

    def save_pdf(self, out_file_name: str):
        with PdfPages(f'{out_file_name}.pdf') as pdf:
            pdf.savefig(self.figure)

    def _add_text_to_triangles(self):
        def assign_text(pos_a, pos_b, direction_a, direction_b):
            if self.triangles_list[pos_a].edges_text[direction_a].text:
                return

            pair = self.text_pairs.generate_random_pairs()
            self.triangles_list[pos_a].add_text(pair[0], direction_a)
            self.triangles_list[pos_b].add_text(pair[1], direction_b)

        for position_a, triangle_a in enumerate(self.triangles_list):
            for position_b, triangle_b in enumerate(self.triangles_list):
                if triangle_b.center.y == triangle_a.center.y and triangle_b.center.x - 1 == triangle_a.center.x:
                    assign_text(position_a, position_b, "right", "left")

                if (triangle_a.rotation == "up"
                        and triangle_b.center.y + 2 == triangle_a.center.y
                        and triangle_b.center.x == triangle_a.center.x):
                    assign_text(position_a, position_b, "bottom", "top")


class Shape:
    def __init__(self, shape_list_file: str, shape_type: str):
        shapes_dict = self.get_shapes_data(shape_list_file)
        if type in shapes_dict:
            self.shape_map = shapes_dict[shape_type]
        else:
            raise ValueError(f"Shape with name [{shape_type}] not exist.")

    @staticmethod
    def get_shapes_data(shapes_data_path):
        shapes_dict = {}
        current_section = None
        with open(shapes_data_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    current_section = line[1:].strip()
                    shapes_dict[current_section] = []
                elif line:
                    shapes_dict[current_section].append(line.split(', '))
                    shapes_dict[current_section][-1][0] = float(shapes_dict[current_section][-1][0])
                    shapes_dict[current_section][-1][1] = float(shapes_dict[current_section][-1][1])
                    shapes_dict[current_section][-1][2] = shapes_dict[current_section][-1][2].strip()
        return shapes_dict


class Image:
    def __init__(self, shape_list_file: str, shape_type: str, pair_list_file: str):
        self.shape: Shape = Shape(shape_list_file, shape_type)
        self.grid: Grid = Grid(pair_list_file)

    def create_image(self, out_file_name: str = "out", plot_image: bool = 0):
        for triangle_data in self.shape.shape_map:
            self.grid.add_triangle(Triangle(Coordinates(triangle_data[0], triangle_data[1]), triangle_data[2]))

        self.grid.plot()

        if plot_image:
            self.grid.show()

        self.grid.save_pdf(out_file_name)
