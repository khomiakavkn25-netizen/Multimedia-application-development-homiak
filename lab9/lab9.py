import numpy as np
from PIL import Image
from skimage import data


def to565(c):
    c = np.asarray(c, dtype=float)
    r = np.round(c[..., 0] / 255 * 31) / 31 * 255
    g = np.round(c[..., 1] / 255 * 63) / 63 * 255
    b = np.round(c[..., 2] / 255 * 31) / 31 * 255
    return np.stack([r, g, b], axis=-1)


def blocks_of(img):
    h, w, _ = img.shape
    return (img.reshape(h // 4, 4, w // 4, 4, 3)
               .transpose(0, 2, 1, 3, 4)
               .reshape(-1, 16, 3).astype(float))


def unblocks(bl, h, w):
    return (bl.reshape(h // 4, w // 4, 4, 4, 3)
              .transpose(0, 2, 1, 3, 4)
              .reshape(h, w, 3))


def stats(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    rmse = np.sqrt((d ** 2).mean())
    psnr = 20 * np.log10(255 / rmse) if rmse > 0 else float("inf")
    return d.max(), d.mean(), rmse, psnr


def angle_error(a, b):
    def unit(x):
        v = np.asarray(x, float) / 255 * 2 - 1
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    cos = (unit(a) * unit(b)).sum(-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def save(img, name):
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(name)
    print("   збережено:", name)


def photo():
    return data.astronaut()

def gradient(N=256):
    x = np.linspace(0, 255, N)
    g = np.zeros((N, N, 3))
    for c in range(3):
        g[..., c] = x[None, :]
    return g.astype(np.uint8)

def normalmap(N=256, freq=14, amp=0.75):
    y, x = np.indices((N, N)).astype(float) / N * 2 * np.pi
    hx = amp * np.cos(freq * x) * np.cos(freq * 0.7 * y)
    hy = amp * np.cos(freq * 0.7 * x) * np.cos(freq * y)
    n = np.stack([hx, hy, np.ones_like(hx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255).round().astype(np.uint8)

def stripes(N=256, wd=37):
    im = np.zeros((N, N, 3), np.uint8)
    cols = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
            (0,255,255), (255,0,255), (255,255,255)]
    for i in range(0, N, wd):
        im[:, i:i+wd] = cols[(i // wd) % len(cols)]
    return im

TESTS = {"фотографія": photo(), "градієнт": gradient(),
         "нормал-мапа": normalmap(), "різкі межі": stripes()}


def make_palette(c0, c1):
    c0 = np.asarray(c0, float)
    c1 = np.asarray(c1, float)
    return np.stack([c0, c1, (2 * c0 + c1) / 3, (c0 + 2 * c1) / 3])


def pick_endpoints(blk, mode="bbox"):
    if mode == "bbox":
        return blk.max(axis=0), blk.min(axis=0)
    else:
        L = blk @ np.array([0.299, 0.587, 0.114])
        return blk[L.argmax()], blk[L.argmin()]


def encode_block(blk, mode="bbox"):
    c0, c1 = pick_endpoints(blk, mode)
    pal = make_palette(to565(c0), to565(c1))
    d = ((blk[:, None, :] - pal[None, :, :]) ** 2).sum(-1)
    return pal, d.argmin(axis=1)


def encode_best(blk):
    best = None
    for m in ("bbox", "pixel"):
        pal, idx = encode_block(blk, m)
        err = ((pal[idx] - blk) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, pal, idx)
    return best[1], best[2]



def bc1(img, mode="bbox"):
    h, w, _ = img.shape
    out = np.zeros((h // 4 * w // 4, 16, 3))
    for i, blk in enumerate(blocks_of(img)):
        if mode == "best":
            pal, idx = encode_best(blk)
        else:
            pal, idx = encode_block(blk, mode)
        out[i] = pal[idx]
    return unblocks(out, h, w)



def check(label, ok, detail=""):
    print(f"   [{'OK ' if ok else 'НІ '}] {label}" + ("" if ok or not detail else f"   — {detail}"))
    if not ok:
        print("\n   Зупиняюся. Виправте це місце й запустіть знову.")
        raise SystemExit(1)


def self_test():
    print("\n1. Палітра з чотирьох кольорів")
    pal = make_palette(np.array([255., 255., 255.]), np.array([0., 0., 0.]))
    check("форма (4, 3)", np.shape(pal) == (4, 3), str(np.shape(pal)))
    check("колір 0 — це перший опорний", np.allclose(pal[0], 255))
    check("колір 1 — це другий опорний", np.allclose(pal[1], 0))
    check("колір 2 на третині шляху", np.allclose(pal[2], 170, atol=1), f"у вас {pal[2][0]:.0f}")
    check("колір 3 на двох третинах", np.allclose(pal[3], 85, atol=1), f"у вас {pal[3][0]:.0f}")

    print("\n2. Вибір опорних кольорів")
    blk = np.zeros((16, 3))
    blk[:8] = [255, 0, 0]
    blk[8:] = [0, 255, 0]
    c0, c1 = pick_endpoints(blk, "bbox")
    check("коробка дає жовтий і чорний",
          np.allclose(sorted([c0.sum(), c1.sum()]), [0, 510]),
          f"{c0.round(0)} і {c1.round(0)}")
    c0, c1 = pick_endpoints(blk, "pixel")
    got = sorted([tuple(c0), tuple(c1)])
    check("реальні пікселі дають червоний і зелений",
          got == [(0., 255., 0.), (255., 0., 0.)], f"{c0.round(0)} і {c1.round(0)}")

    print("\n3. Кодування одного блоку")
    pal, idx = encode_block(blk, "pixel")
    check("індексів рівно 16", len(idx) == 16, str(len(idx)))
    check("усі індекси в діапазоні 0…3", idx.min() >= 0 and idx.max() <= 3)
    err = np.abs(pal[idx] - blk).max()
    check("два чисті кольори відтворено точно", err < 1, f"похибка {err:.0f}")
    pal, idx = encode_block(blk, "bbox")
    err = np.abs(pal[idx] - blk).max()
    check("а коробка на тому самому блоці помиляється", err > 100, f"похибка {err:.0f}")

    print("\n4. Ціле зображення")
    out = bc1(TESTS["фотографія"])
    check("розмір збігся", out.shape == TESTS["фотографія"].shape, str(out.shape))
    mx, mn, r, p = stats(TESTS["фотографія"], out)
    check("RMSE на фотографії близько 5,7", abs(r - 5.71) < 0.5, f"у вас {r:.2f}")
    check("PSNR близько 33", abs(p - 33.0) < 0.8, f"у вас {p:.1f}")

    print("\n   Усі тести пройдено.\n")


def run_all():
    print(f"   {'зображення':<14}{'макс':>7}{'середня':>10}{'RMSE':>8}{'PSNR':>8}")
    for name, im in TESTS.items():
        out = bc1(im)
        mx, mn, r, p = stats(im, out)
        print(f"   {name:<14}{mx:>7.0f}{mn:>10.2f}{r:>8.2f}{p:>8.1f}")
        save(out, f"bc1_{name}.png")
    ang = angle_error(TESTS["нормал-мапа"], bc1(TESTS["нормал-мапа"]))
    print(f"\n   нормал-мапа: кутова похибка {ang.mean():.2f}° у середньому, "
          f"{ang.max():.2f}° максимум")
    print(f"   пікселів із похибкою більше за 2°: {(ang > 2).mean() * 100:.1f}%")


def task3():
    print("\nЗавдання 3. Порівняння двох способів (RMSE)")
    print(f"   {'зображення':<14}{'коробка':>9}{'пікселі':>9}{'найкращий з двох':>20}")
    rows = {}
    for name, im in TESTS.items():
        a = bc1(im, "bbox")
        b = bc1(im, "pixel")
        c = bc1(im, "best")
        rows[name] = (stats(im, a)[2], stats(im, b)[2], stats(im, c)[2])
        print(f"   {name:<14}{rows[name][0]:>9.2f}{rows[name][1]:>9.2f}{rows[name][2]:>20.2f}")
    nm = TESTS["нормал-мапа"]
    for mode in ("bbox", "pixel", "best"):
        ang = angle_error(nm, bc1(nm, mode))
        print(f"   нормал-мапа [{mode}]: середня {ang.mean():.2f}°, "
              f"максимум {ang.max():.2f}°, >2°: {(ang > 2).mean() * 100:.1f}%")
    return rows


def own_image():
    N = 128
    im = np.full((N, N, 3), 255, np.uint8)
    im[10:57, 10:57] = (255, 0, 0)
    im[10:57, 71:118] = (0, 200, 0)
    im[71:118, 10:57] = (0, 0, 255)
    im[71:118, 71:118] = (255, 220, 0)
    yy, xx = np.indices((N, N))
    ring = (yy - 64) ** 2 + (xx - 64) ** 2
    im[(ring < 19 ** 2)] = (0, 0, 0)
    im[(ring < 11 ** 2)] = (255, 0, 255)
    cols = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255)])
    for x in range(N):
        im[60:68, x] = cols[x % 3] if (x < 10 or x > 117) else im[60:68, x]
    im[0:8, :] = cols[np.arange(N) % 3][None, :, :]
    im[N - 8:, :] = cols[np.arange(N) % 3][None, :, :]
    return im


def task4_own():
    print("\nЗавдання 4. Власне зображення")
    im = own_image()
    save(im, "own_original.png")
    for mode in ("bbox", "pixel"):
        out = bc1(im, mode)
        mx, mn, r, p = stats(im, out)
        print(f"   [{mode}] макс {mx:.0f}, середня {mn:.2f}, RMSE {r:.2f}, PSNR {p:.1f}")
        save(out, f"bc1_власне_{mode}.png")
        diff = np.abs(im.astype(float) - out).max(axis=-1)
        save(np.clip(diff * 3, 0, 255)[..., None].repeat(3, -1), f"diff_власне_{mode}.png")


def diff_maps():
    for name, im in TESTS.items():
        out = bc1(im)
        diff = np.abs(im.astype(float) - out).max(axis=-1)
        save(np.clip(diff * 3, 0, 255)[..., None].repeat(3, -1), f"diff_{name}.png")


if __name__ == "__main__":
    try:
        self_test()
    except NotImplementedError as e:
        print(f"\n   Місце {e} ще не зроблено.\n")
        raise SystemExit(1)
    print("Завдання 4. Усі чотири зображення")
    run_all()
    task3()
    diff_maps()
    task4_own()