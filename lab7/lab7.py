import time

import cv2
import numpy as np
from PIL import Image
from skimage import data


def conv2(img, K):
    kh, kw = K.shape
    ph, pw = kh // 2, kw // 2
    pad = np.pad(img, ((ph, ph), (pw, pw)), mode='edge')
    out = np.zeros_like(img)
    for i in range(kh):
        for j in range(kw):
            out += K[i, j] * pad[i:i + img.shape[0], j:j + img.shape[1]]
    return out


def gauss1d(sigma, n):
    x = np.arange(n) - n // 2
    k = np.exp(-x ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def conv_sep(img, k):
    n = len(k) // 2
    pad = np.pad(img, ((0, 0), (n, n)), mode='edge')
    tmp = np.zeros_like(img)
    for j, v in enumerate(k):
        tmp += v * pad[:, j:j + img.shape[1]]
    pad = np.pad(tmp, ((n, n), (0, 0)), mode='edge')
    out = np.zeros_like(img)
    for i, v in enumerate(k):
        out += v * pad[i:i + img.shape[0], :]
    return out


def best(f, r=5):
    ts = []
    for _ in range(r):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts) * 1000


def save(arr, name):
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(name)


def main():
    gray = np.array(Image.fromarray(data.astronaut()).convert('L')).astype(float)
    save(gray, 'original.png')


    avg = conv2(gray, np.ones((3, 3)) / 9)
    print(f'Середня яскравість: оригінал {gray.mean():.3f}, після 3x3 {avg.mean():.3f}')


    k = gauss1d(1.6, 7)
    BLUR = np.outer(k, k)
    blur = conv2(gray, BLUR)
    save(blur, 'blur.png')

    SX = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], float)
    SY = SX.T
    gx, gy = conv2(gray, SX), conv2(gray, SY)
    edges = np.hypot(gx, gy)
    edges = edges / edges.max() * 255
    save(edges, 'sobel.png')

    detail = gray - blur
    sharp = np.clip(gray + 1.2 * detail, 0, 255)
    save(sharp, 'sharp.png')


    print('Середнє по периметру blur / gray: '
          f'{np.mean(np.r_[blur[0], blur[-1], blur[:, 0], blur[:, -1]]):.2f} / '
          f'{np.mean(np.r_[gray[0], gray[-1], gray[:, 0], gray[:, -1]]):.2f}')


    K15 = np.outer(gauss1d(3.0, 15), gauss1d(3.0, 15))
    print('Розбіжність conv2 / conv_sep (15x15):',
          np.abs(conv2(gray, K15) - conv_sep(gray, gauss1d(3.0, 15))).max())

    print('\nТаблиця замірів (мс), зображення 512x512')
    print('Ядро     Звичайне  Роздільне  Виграш   OpenCV')
    for n in (3, 5, 9, 15, 31):
        k = gauss1d(max(n / 5, 0.6), n)
        K = np.outer(k, k)
        diff = np.abs(conv2(gray, K) - conv_sep(gray, k)).max()
        ta = best(lambda: conv2(gray, K))
        tb = best(lambda: conv_sep(gray, k))
        tc = best(lambda: cv2.GaussianBlur(gray, (n, n), max(n / 5, 0.6),
                                           borderType=cv2.BORDER_REPLICATE))
        print(f'{n:>2}x{n:<2} {ta:10.1f} {tb:10.1f} {ta / tb:7.1f}x {tc:8.1f}'
              f'   (розбіжність {diff:.1e})')


    n, sigma = 15, 3.0
    k = gauss1d(sigma, n)
    mine = conv_sep(gray, k)
    lib = cv2.GaussianBlur(gray, (n, n), sigma, borderType=cv2.BORDER_REPLICATE)
    print('\nрозбіжність з OpenCV:', np.abs(mine - lib).max())
    print('ваш час: ', best(lambda: conv_sep(gray, k)))
    print('OpenCV:  ', best(lambda: cv2.GaussianBlur(
        gray, (n, n), sigma, borderType=cv2.BORDER_REPLICATE)))


    EMBOSS = np.array([[-2, -1, 0],
                       [-1, 1, 1],
                       [0, 1, 2]], float)
    MOTION = np.eye(9) / 9
    save(np.clip(conv2(gray, EMBOSS) + 128, 0, 255), 'emboss.png')
    save(conv2(gray, MOTION), 'motion.png')


if __name__ == '__main__':
    main()