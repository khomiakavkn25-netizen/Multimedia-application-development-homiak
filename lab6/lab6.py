import io
import sys
import zlib

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity

Q50 = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

QUALITIES = [95, 90, 75, 50, 30, 15, 5]


def load_gray(path=None):
    if path is None:
        from skimage import data
        arr = np.asarray(data.camera(), dtype=np.uint8)
    else:
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    h, w = arr.shape
    return arr[: h - h % 8, : w - w % 8]


def blockify(img):
    h, w = img.shape
    return (img.reshape(h // 8, 8, w // 8, 8)
               .swapaxes(1, 2)
               .reshape(-1, 8, 8))


def unblockify(blocks, h, w):
    return (blocks.reshape(h // 8, w // 8, 8, 8)
                  .swapaxes(1, 2)
                  .reshape(h, w))


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    return float(structural_similarity(a, b, data_range=255))


def size_bytes(coeffs):
    return len(zlib.compress(coeffs.tobytes(), 9))


def jpeg_size(img, q):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=q)
    return buf.tell()


def dct_matrix(N=8):
    D = np.zeros((N, N))
    for k in range(N):
        a = np.sqrt(1 / N) if k == 0 else np.sqrt(2 / N)
        for n in range(N):
            D[k, n] = a * np.cos(np.pi * (2 * n + 1) * k / (2 * N))
    return D


D = dct_matrix()


def dct2(block):
    return D @ block @ D.T


def idct2(coeffs):
    return D.T @ coeffs @ D


def qtable(q):
    q = max(1, min(99, q))
    S = 5000 / q if q < 50 else 200 - 2 * q
    return np.clip(np.floor((S * Q50 + 50) / 100), 1, 255)


def encode(img, q):
    Q = qtable(q)
    blocks = blockify(img.astype(np.float64) - 128)
    coeffs = np.array([np.round(dct2(b) / Q) for b in blocks])
    return coeffs.astype(np.int16), Q


def decode(coeffs, Q, h, w):
    blocks = np.array([idct2(c * Q) for c in coeffs])
    img = unblockify(blocks, h, w) + 128
    return np.clip(img, 0, 255).round().astype(np.uint8)


def check(name, ok, info=""):
    print(f"{'ТАК' if ok else 'НІ '}  {name}  {info}")
    return ok


def run_tests(img):
    h, w = img.shape
    ok = True
    ok &= check("1. рядки D ортонормовані", np.allclose(D @ D.T, np.eye(8)))
    ok &= check("1. рядок 0 сталий", np.allclose(D[0], D[0, 0]))

    blk = blockify(img.astype(np.float64) - 128)[1000 % len(blockify(img))]
    err = np.abs(idct2(dct2(blk)) - blk).max()
    ok &= check("2. dct2 -> idct2 без втрат", err < 1e-9, f"похибка {err:.1e}")

    ok &= check("3. qtable(50) == Q50", np.array_equal(qtable(50), Q50))
    ok &= check("3. середні qtable (90/50/10)", True,
                f"{qtable(90).mean():.0f} / {qtable(50).mean():.0f} / {qtable(10).mean():.0f}")

    coeffs, Q = encode(img, 50)
    nz = np.count_nonzero(coeffs.reshape(len(coeffs), -1), axis=1).mean()
    rec = decode(coeffs, Q, h, w)
    ok &= check("4. кодер дає int16 і форму (N,8,8)",
                coeffs.dtype == np.int16 and coeffs.shape[1:] == (8, 8))
    ok &= check("5. декодер дає uint8 потрібної форми",
                rec.dtype == np.uint8 and rec.shape == (h, w))
    ok &= check("5. q=50: PSNR і SSIM", True,
                f"у середньому виживає {nz:.1f} з 64, "
                f"PSNR {psnr(img, rec):.1f} дБ, SSIM {ssim(img, rec):.3f}")
    return bool(ok)


def quality_curve(img, outdir="."):
    h, w = img.shape
    raw = img.size
    rows = []
    for q in QUALITIES:
        coeffs, Q = encode(img, q)
        rec = decode(coeffs, Q, h, w)
        Image.fromarray(rec).save(f"{outdir}/q{q:02d}.png")
        nbytes = size_bytes(coeffs)
        rows.append(dict(
            q=q, bytes=nbytes, ratio=raw / nbytes,
            zeros=100 * (coeffs == 0).mean(),
            nz=np.count_nonzero(coeffs.reshape(len(coeffs), -1), axis=1).mean(),
            psnr=psnr(img, rec), ssim=ssim(img, rec), jpeg=jpeg_size(img, q)))
    return rows


def print_table(rows, raw):
    print(f"\nСирий розмір {raw} байт")
    print(f"{'q':>3} {'байт':>8} {'стиснення':>10} {'нулів':>7} "
          f"{'PSNR':>6} {'SSIM':>6} {'JPEG, байт':>11}")
    for r in rows:
        print(f"{r['q']:>3} {r['bytes']:>8} {r['ratio']:>9.1f}x {r['zeros']:>6.1f}% "
              f"{r['psnr']:>6.1f} {r['ssim']:>6.3f} {r['jpeg']:>11}")


def plot_curve(rows, path="curve.png"):
    q = [r["q"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].plot(q, [r["bytes"] / 1000 for r in rows], "o-", label="наш (zlib)")
    ax[0].plot(q, [r["jpeg"] / 1000 for r in rows], "s--", label="справжній JPEG")
    ax[0].set_ylabel("розмір, КБ"); ax[0].legend()
    ax[1].plot(q, [r["psnr"] for r in rows], "o-", color="tab:green")
    ax[1].set_ylabel("PSNR, дБ")
    ax[2].plot(q, [r["ssim"] for r in rows], "o-", color="tab:red")
    ax[2].set_ylabel("SSIM")
    for a in ax:
        a.set_xlabel("якість q"); a.grid(alpha=.3); a.invert_xaxis()
    fig.suptitle("Крива «розмір / PSNR / SSIM»")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    img = load_gray(sys.argv[1] if len(sys.argv) > 1 else None)
    print("Зображення:", img.shape)
    if not run_tests(img):
        print("\nЄ «НІ» — виправте й запустіть знову.")
        sys.exit(1)
    rows = quality_curve(img)
    print_table(rows, img.size)
    plot_curve(rows)
    Image.fromarray(img).save("original.png")
    print("\nЗбережено curve.png, original.png, q95.png ... q05.png")