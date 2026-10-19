import numpy as np
from PIL import Image


def to_light(code):
    c = np.asarray(code, dtype=np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_code(light):
    l = np.asarray(light, dtype=np.float64)
    s = np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)
    return s * 255.0


def test_gamma_functions():
    print("--- Перевірка Завдання 1 ---")
    mid = (to_light(0) + to_light(255)) / 2
    print("to_code(mid):", round(float(to_code(mid))))
    print("to_light(128):", round(float(to_light(128)) * 100, 1))
    print("to_light(64):", round(float(to_light(64)) * 100, 1))
    print("to_light(192):", round(float(to_light(192)) * 100, 1))
    print()


def make_board(size=512, cell=8):
    yy, xx = np.indices((size, size))
    return (((yy // cell + xx // cell) % 2) * 255).astype(np.uint8)


def box_average(img, k):
    h, w = img.shape
    return img.reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def run_checkerboard_experiment():
    print("--- Перевірка Завдання 2 ---")
    board = make_board(512, 8)

    naive = box_average(board.astype(np.float64), 16)

    board_light = to_light(board)
    avg_light = box_average(board_light, 16)
    correct = to_code(avg_light)

    print("Наївно (середній код):   ", round(float(naive.mean()), 1))
    print("Правильно (середній код):", round(float(correct.mean()), 1))

    orig_light_mean = to_light(board).mean() * 100
    naive_light_mean = to_light(naive).mean() * 100
    correct_light_mean = to_light(correct).mean() * 100

    print(f"Середне світло оригіналу: {orig_light_mean:.1f}%")
    print(f"Середне світло наївно:    {naive_light_mean:.1f}%")
    print(f"Середне світло правильно: {correct_light_mean:.1f}%")

    Image.fromarray(board).save("board_orig.png")
    Image.fromarray(np.clip(naive, 0, 255).astype(np.uint8)).save("board_naive.png")
    Image.fromarray(np.clip(correct, 0, 255).astype(np.uint8)).save("board_correct.png")
    print("Зображення шахівниці збережені.")
    print()


def rgb_to_ycbcr(img):
    x = img.astype(np.float64)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.402 * (cr - 128)
    g = y - 0.344136 * (cb - 128) - 0.714136 * (cr - 128)
    b = y + 1.772 * (cb - 128)

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def run_ycbcr_experiment(img_path):
    print("--- Перевірка Завдання 3 ---")
    img = np.array(Image.open(img_path).convert("RGB"))

    y, cb, cr = rgb_to_ycbcr(img)
    back = ycbcr_to_rgb(y, cb, cr)

    max_diff = np.abs(back.astype(int) - img.astype(int)).max()
    print("Максимальна різниця після перетворення туди й назад:", max_diff)

    gray_y = np.clip(y, 0, 255).astype(np.uint8)
    gray_naive = img.mean(axis=-1).astype(np.uint8)

    Image.fromarray(gray_y).save("gray_y.png")
    Image.fromarray(gray_naive).save("gray_naive.png")
    print("Зображення YCbCr збережені.")
    print()


def rgb_to_hsv(img):
    x = img.astype(np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = x.max(-1); mn = x.min(-1); d = mx - mn

    h = np.zeros_like(mx)
    m = (d != 0) & (mx == r); h[m] = (60 * ((g - b)[m] / d[m])) % 360
    m = (d != 0) & (mx == g); h[m] = 60 * ((b - r)[m] / d[m]) + 120
    m = (d != 0) & (mx == b); h[m] = 60 * ((r - g)[m] / d[m]) + 240

    s = np.where(mx == 0, 0, d / np.where(mx == 0, 1, mx))
    return h, s, mx


def hsv_to_rgb(h, s, v):
    c = v * s
    h_prime = (h % 360) / 60.0
    x = c * (1.0 - np.abs(h_prime % 2 - 1.0))
    m = v - c

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    sec = np.floor(h_prime).astype(int) % 6

    r[sec == 0], g[sec == 0], b[sec == 0] = c[sec == 0], x[sec == 0], 0
    r[sec == 1], g[sec == 1], b[sec == 1] = x[sec == 1], c[sec == 1], 0
    r[sec == 2], g[sec == 2], b[sec == 2] = 0, c[sec == 2], x[sec == 2]
    r[sec == 3], g[sec == 3], b[sec == 3] = 0, x[sec == 3], c[sec == 3]
    r[sec == 4], g[sec == 4], b[sec == 4] = x[sec == 4], 0, c[sec == 4]
    r[sec == 5], g[sec == 5], b[sec == 5] = c[sec == 5], 0, x[sec == 5]

    rgb = np.stack([r + m, g + m, b + m], axis=-1) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def run_hsv_experiments(img_path):
    print("--- Перевірка Завдання 4 ---")
    img = np.array(Image.open(img_path).convert("RGB"))
    h, s, v = rgb_to_hsv(img)

    back = hsv_to_rgb(h, s, v)
    print("HSV roundtrip max diff:", np.abs(back.astype(int) - img.astype(int)).max())

    exp1 = hsv_to_rgb(h, s * 0.3, v)
    Image.fromarray(exp1).save("hsv_sat_low.png")

    exp2 = hsv_to_rgb(h, np.clip(s * 2.0, 0, 1.0), v)
    Image.fromarray(exp2).save("hsv_sat_high.png")

    exp3 = hsv_to_rgb((h + 60) % 360, s, v)
    Image.fromarray(exp3).save("hsv_hue_shift.png")

    print("Результати досліджень HSV збережені.")
    print()


if __name__ == "__main__":
    test_gamma_functions()
    run_checkerboard_experiment()

    sample_img = np.zeros((200, 200, 3), dtype=np.uint8)
    sample_img[:100, :100] = [255, 0, 0]
    sample_img[:100, 100:] = [0, 255, 0]
    sample_img[100:, :100] = [0, 0, 255]
    sample_img[100:, 100:] = [255, 255, 0]
    Image.fromarray(sample_img).save("sample.png")

    run_ycbcr_experiment("sample.png")
    run_hsv_experiments("sample.png")