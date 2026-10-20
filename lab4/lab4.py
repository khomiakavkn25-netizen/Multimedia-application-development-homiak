"""Лабораторна робота 4. Предиктори PNG (None, Sub, Up) + адаптивний вибір."""
import os
import numpy as np
from collections import Counter

W, H = 512, 384


# ---------- Ентропія (з лабораторної 3) ----------
def entropy(img):
    counts = np.bincount(img.ravel(), minlength=256)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


# ---------- Тестові зображення ----------
def make_gradient():
    row = (np.arange(W) // 2).astype(np.uint8)          # 0,0,1,1,2,2,... -> 256 значень
    return np.tile(row, (H, 1))


def make_noise():
    return np.random.default_rng(1).integers(0, 256, (H, W), dtype=np.uint8)


def load_gray(name):
    """Зчитує зображення з папки скрипта й переводить у відтінки сірого."""
    from PIL import Image
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def make_screenshot():
    return load_gray("images.jfif")


def make_photo():
    return load_gray("photo_2026-10-15_19-00-45.jpg")


# ---------- Завдання 1: фільтр одного рядка ----------
def filter_row(cur, prev, f):
    a = np.concatenate(([0], cur[:-1])).astype(np.uint8)  # лівий сусід
    b = prev                                              # верхній сусід
    if f == 0:
        pred = np.zeros_like(cur)
    elif f == 1:
        pred = a
    else:
        pred = b
    return ((cur.astype(int) - pred.astype(int)) % 256).astype(np.uint8)


# ---------- Завдання 2: ціле зображення ----------
def filter_image(img, f):
    out = np.zeros_like(img)
    prev = np.zeros(img.shape[1], dtype=np.uint8)
    for y in range(img.shape[0]):
        out[y] = filter_row(img[y], prev, f)
        prev = img[y]                                     # вихідний рядок, не out[y]
    return out


# ---------- Завдання 3: зворотне перетворення ----------
def unfilter_row(enc, prev, f):
    n = len(enc)
    out = np.zeros(n, dtype=np.uint8)
    for x in range(n):
        a = out[x - 1] if x > 0 else 0                    # уже відновлений лівий
        b = prev[x]                                       # уже відновлений верхній
        pred = 0 if f == 0 else (int(a) if f == 1 else int(b))
        out[x] = (int(enc[x]) + pred) % 256
    return out


def unfilter_image(enc, f):
    out = np.zeros_like(enc)
    prev = np.zeros(enc.shape[1], dtype=np.uint8)
    for y in range(enc.shape[0]):
        out[y] = unfilter_row(enc[y], prev, f)
        prev = out[y]                                     # тут саме відновлений рядок
    return out


# ---------- Завдання 4: адаптивний вибір ----------
def sum_abs(row):
    r = row.astype(int)
    signed = np.where(r > 127, r - 256, r)
    return int(np.abs(signed).sum())


def filter_adaptive(img):
    out = np.zeros_like(img)
    chosen = []
    prev = np.zeros(img.shape[1], dtype=np.uint8)
    for y in range(img.shape[0]):
        best_f, best_row, best_s = None, None, None
        for f in (0, 1, 2):
            r = filter_row(img[y], prev, f)
            s = sum_abs(r)
            if best_s is None or s < best_s:
                best_f, best_row, best_s = f, r, s
        out[y] = best_row
        chosen.append(best_f)
        prev = img[y]
    return out, chosen


def unfilter_adaptive(enc, chosen):
    out = np.zeros_like(enc)
    prev = np.zeros(enc.shape[1], dtype=np.uint8)
    for y in range(enc.shape[0]):
        out[y] = unfilter_row(enc[y], prev, chosen[y])
        prev = out[y]
    return out


if __name__ == "__main__":
    # Перевірка завдання 1
    row = np.array([40, 42, 44, 46, 48, 50, 52, 54], dtype=np.uint8)
    zeros = np.zeros(8, dtype=np.uint8)
    print("Завдання 1:")
    print(filter_row(row, zeros, 1))   # [40 2 2 2 2 2 2 2]
    print(filter_row(row, zeros, 0))   # [40 42 ... 54]
    print(filter_row(row, row, 2))     # [0 0 0 0 0 0 0 0]

    images = {
        "Градієнт": make_gradient(),
        "Білий шум": make_noise(),
        "Скріншот": make_screenshot(),
        "Фотографія": make_photo(),
    }
    names = ["None", "Sub", "Up"]

    # Завдання 2
    print("\nТаблиця 1. Ентропія, біт на піксель")
    print(f"{'Зображення':<12}{'До':>7}{'None':>7}{'Sub':>7}{'Up':>7}")
    for name, img in images.items():
        vals = [entropy(filter_image(img, f)) for f in (0, 1, 2)]
        print(f"{name:<12}{entropy(img):7.2f}" + "".join(f"{v:7.2f}" for v in vals))

    # Завдання 3
    print("\nЗавдання 3: відновлення байт у байт")
    for name, img in images.items():
        res = [np.array_equal(img, unfilter_image(filter_image(img, f), f)) for f in (0, 1, 2)]
        print(f"{name:<12}", res)

    # Завдання 4 + 5
    print("\nЗавдання 4: адаптивний вибір")
    summary = []
    for name, img in images.items():
        enc, chosen = filter_adaptive(img)
        ok = np.array_equal(img, unfilter_adaptive(enc, chosen))
        cnt = Counter(chosen)
        dist = {names[k]: cnt.get(k, 0) for k in (0, 1, 2)}
        e0, e1 = entropy(img), entropy(enc)
        best_single = min(range(3), key=lambda f: entropy(filter_image(img, f)))
        print(f"{name:<12} відновлення={ok}  вибір={dist}  ентропія={e1:.2f}")
        summary.append((name, dist, e0, e1, 100 * (e0 - e1) / e0, img.shape[0]))

    print("\nТаблиця 2. Підсумок")
    print(f"{'Тип':<12}{'Оптимальний фільтр':<28}{'До / після':>14}{'Виграш, %':>12}")
    for name, dist, e0, e1, g, h in summary:
        top = max(dist, key=dist.get)
        print(f"{name:<12}{top + ' (' + str(dist[top]) + ' з ' + str(h) + ' рядків)':<28}"
              f"{e0:6.2f} / {e1:4.2f}{g:12.1f}")

    # Запитання 7: накладні витрати на номер фільтра
    print(f"\nНомери фільтрів: по 1 байту на рядок, фото {images['Фотографія'].shape[0]} байт, {100/images['Фотографія'].shape[1]:.2f}% від розміру")