# -*- coding: utf-8 -*-
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio  # pip install segyio

from util.My_tool1 import *

INPUT_PATH = "inputs/noised.segy"
MODEL_PATH = "trained_model/model.pth"
OUTPUT_PATH = "results/denoised.segy"
PARAMS_PATH = "inputs/inputParameters.json"


def report_progress(perc: float):
    """Выводит прогресс выполнения в stdout в формате <PROGRESS: N %>."""
    if perc <= 100:
        print(f"<PROGRESS: {int(perc)} %>", flush=True)


def load_parameters(path: str) -> dict:
    """Загружает параметры из JSON-файла. Значения Date и Timestamp хранятся как long (мс с 1970-01-01)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл параметров не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    report_progress(5)
    return params


def read_segy(file_path: str) -> np.ndarray:
    """
    Чтение SEGY 2D и возврат в виде numpy массива (traces x samples)
    """
    with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
        n_traces = segyfile.tracecount
        n_samples = segyfile.samples.size
        data = np.zeros((n_traces, n_samples), dtype=np.float32)
        for i in range(n_traces):
            data[i, :] = segyfile.trace[i]
    return data

def write_segy(file_path: str, data: np.ndarray, template_path: str):
    """
    Запись 2D numpy массива в SEGY, используя исходный файл как шаблон для заголовков.
    """
    n_traces, n_samples = data.shape
    with segyio.open(template_path, "r", ignore_geometry=True) as template:
        spec = segyio.spec()
        spec.format = template.format
        spec.samples = template.samples[:n_samples]  # подрезаем, если нужно
        spec.ilines = [0]  # фиктивная линия
        spec.xlines = list(range(n_traces))

    with segyio.create(file_path, spec) as segy_out:
        for i in range(n_traces):
            segy_out.trace[i] = data[i, :]
        segy_out.flush()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем параметры
    params = load_parameters(PARAMS_PATH)
    print("Загруженные параметры:", params)
    report_progress(10)

    # Загружаем модель
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()
    report_progress(30)

    # Загружаем SEGY данные
    y = read_segy(INPUT_PATH)
    report_progress(50)

    # Переводим данные на device
    y_tensor = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    y_tensor = y_tensor.to(device, dtype=torch.float32)

    # Инференс
    start_time = time.time()
    x_tensor = model(y_tensor)
    elapsed_time = time.time() - start_time
    report_progress(80)

    # Преобразуем результат в numpy
    x_ = (
        x_tensor.view(y.shape[0], y.shape[1])
        .cpu()
        .detach()
        .numpy()
        .astype(np.float64)
    )
    report_progress(90)

    print(f"Inference time: {elapsed_time:.3f} seconds")

    # Визуализация
    plt.imshow(y, cmap="gray", aspect="auto", vmin=-1, vmax=1)
    plt.title("Прореженные данные")
    plt.xlabel("Расстояние от источника, м")
    plt.ylabel("Время свободного пробега, мс")
    plt.show()

    plt.imshow(x_, cmap="gray", aspect="auto", vmin=-1, vmax=1)
    plt.title("Результат работы нейросети")
    plt.xlabel("Расстояние от источника, м")
    plt.ylabel("Время свободного пробега, мс")
    plt.show()

    # Сохраняем результат в SEGY, используя исходный файл как шаблон для заголовков
    write_segy(OUTPUT_PATH, x_, INPUT_PATH)
    report_progress(100)
    print(f"Результат сохранён в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
