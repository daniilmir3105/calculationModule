# -*- coding: utf-8 -*-
import os
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from util.My_tool1 import *

INPUT_PATH = "inputs/noised.npy"
MODEL_PATH = "trained_model/model.pth"
OUTPUT_PATH = "results/denoised.npy"
PARAMS_PATH = "inputs/inputParameters.json"


def report_progress(perc: float):
    """
    Выводит прогресс выполнения в stdout в формате <PROGRESS: N %>.
    :param perc: процент выполнения, игнорируются значения > 100
    """
    if perc <= 100:
        print(f"<PROGRESS: {int(perc)} %>", flush=True)


def load_parameters(path: str) -> dict:
    """
    Загружает параметры из JSON-файла.
    Значения Date и Timestamp хранятся как long (мс с 1970-01-01).
    :param path: путь к JSON
    :return: словарь параметров
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл параметров не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    report_progress(5)  # Загрузка параметров завершена
    return params


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
    report_progress(30)  # Модель загружена

    # Загружаем данные
    y = np.load(INPUT_PATH)
    report_progress(50)  # Данные загружены

    # Переводим данные на device
    y_tensor = torch.from_numpy(y).view(1, -1, y.shape[0], y.shape[1])
    y_tensor = y_tensor.to(device, dtype=torch.float32)

    # Инференс
    start_time = time.time()
    x_tensor = model(y_tensor)
    elapsed_time = time.time() - start_time
    report_progress(80)  # Инференс завершен

    # Преобразуем результат в numpy
    x_ = (
        x_tensor.view(y.shape[0], y.shape[1])
        .cpu()
        .detach()
        .numpy()
        .astype(np.float64)
    )
    report_progress(90)  # Результат подготовлен

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

    # Сохраняем результат
    np.save(OUTPUT_PATH, x_)
    report_progress(100)  # Сохранение завершено
    print(f"Результат сохранён в {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
