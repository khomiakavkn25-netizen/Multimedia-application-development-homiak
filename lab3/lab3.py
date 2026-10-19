import io
import numpy as np
from PIL import Image

# ---------- 1. Зображення ----------
grad = np.tile(np.linspace(0, 255, 512).astype(np.uint8), (384, 1))
rng = np.random.default_rng(1)
noise = rng.integers(0, 256, (384, 512)).astype(np.uint8)

def load(path):
    try:
        return np.array(Image.open(path).convert("L").resize((512, 384)))
    except FileNotFoundError:
        print(f"[!] Немає файлу {path} - поклади його поруч зі скриптом")
        return None

scr = load("photo_2026-10-16_22-17-19.jpg")
photo = load("photo_2026-10-15_18-31-45.jpg")

images = {"градієнт": grad, "білий шум": noise}
if scr is not None:   images["скріншот"] = scr
if photo is not None: images["фотографія"] = photo

ORIG = 512 * 384  # 196608 байтів

# ---------- 2. Ентропія ----------
def entropy(img):
    values, counts = np.unique(img, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())

# ---------- 3. RLE (власна реалізація) ----------
def rle_size(img):
    data = img.ravel()
    n = len(data)
    i = 0
    out = 0
    while i < n:
        j = i
        # серія не довша за 255 і не виходить за масив
        while j + 1 < n and data[j + 1] == data[i] and (j + 1 - i) < 255:
            j += 1
        out += 2          # одна пара: лічильник + значення
        i = j + 1         # на кінець серії
    return out

# ---------- 4. Хаффман (заготовка) ----------
def huffman_codes(freq):
    if len(freq) == 1:
        return {list(freq)[0]: "0"}
    nodes = [[n, {v: ""}] for v, n in freq.items()]
    while len(nodes) > 1:
        nodes.sort(key=lambda node: node[0])
        a = nodes.pop(0)
        b = nodes.pop(0)
        for v in a[1]:  a[1][v] = "0" + a[1][v]
        for v in b[1]:  b[1][v] = "1" + b[1][v]
        nodes.append([a[0] + b[0], {**a[1], **b[1]}])
    return nodes[0][1]

def huffman_stats(img):
    values, counts = np.unique(img, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(values, counts)}
    codes = huffman_codes(freq)
    total = sum(freq.values())
    avg = sum(freq[v] * len(codes[v]) for v in freq) / total
    lengths = [len(c) for c in codes.values()]
    return avg, min(lengths), max(lengths)

# ---------- 5. PNG ----------
def png_size(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", optimize=True)
    return len(buf.getvalue())

# ---------- Перевірка RLE на прикладі з рисунка 2 ----------
# (15 байтів -> 6; тут свій приклад: 3 серії = 6 байтів)
assert rle_size(np.array([5,5,5,7,7,9,9,9,9], dtype=np.uint8)) == 6
assert rle_size(np.array([1]*300, dtype=np.uint8)) == 4  # 255 + 45

# ---------- Таблиця ----------
print(f"{'Зображення':<12}{'Ентропія':>10}{'Хаффман':>10}{'коди':>10}{'RLE, %':>10}{'PNG, %':>10}")
rows = []
for name, im in images.items():
    e = entropy(im)
    avg, lo, hi = huffman_stats(im)
    rle = rle_size(im) / ORIG * 100
    png = png_size(im) / ORIG * 100
    print(f"{name:<12}{e:>10.2f}{avg:>10.2f}{f'{lo}-{hi}':>10}{rle:>10.1f}{png:>10.2f}")