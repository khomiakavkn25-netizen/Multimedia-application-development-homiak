import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import data


def rgb2ycbcr(img):
    a = img.astype(np.float64)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    return Y, Cb, Cr


def ycbcr2rgb(Y, Cb, Cr):
    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255).round().astype(np.uint8)


def down422(C):
    h, w = C.shape
    return C.reshape(h, w // 2, 2).mean(axis=2)


def up422(C, w):
    return np.repeat(C, 2, axis=1)[:, :w]


def down420(C):
    h, w = C.shape
    return C.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def up420(C, h, w):
    return np.repeat(np.repeat(C, 2, axis=0), 2, axis=1)[:h, :w]


def scheme(img, mode):
    Y, Cb, Cr = rgb2ycbcr(img)
    h, w = Y.shape
    if mode == '4:4:4':
        pass
    elif mode == '4:2:2':
        Cb = up422(down422(Cb), w)
        Cr = up422(down422(Cr), w)
    elif mode == '4:2:0':
        Cb = up420(down420(Cb), h, w)
        Cr = up420(down420(Cr), h, w)
    return ycbcr2rgb(Y, Cb, Cr)


def redtext(W=320, H=96):
    im = Image.new('RGB', (W, H), (0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 48)
    d.text((12, 22), 'ЧЕРВОНИЙ', font=f, fill=(255, 0, 0))
    return np.array(im)


def lines(W=320, H=160):
    im = np.zeros((H, W, 3), np.uint8)
    cols = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i, c in enumerate(cols):
        x = 24 + i * 48
        im[12:70, x:x + 2] = c
        im[12:70, x + 22:x + 23] = c
    for i, c in enumerate(cols[:4]):
        y = 90 + i * 16
        im[y:y + 2, 24:296] = c
        im[y + 7:y + 8, 24:296] = c
    return im


def diffmap(orig, out):
    d = np.abs(out.astype(int) - orig.astype(int))
    return d.max(axis=2)


def save_diff(orig, fname, title):
    dm = diffmap(orig, scheme(orig, '4:2:0'))
    plt.figure()
    plt.imshow(np.clip(dm * 2, 0, 255), cmap='inferno', vmin=0, vmax=255)
    plt.title(title + ' (підсилення ×2)')
    plt.axis('off')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()


def measure(orig, mode):
    out = scheme(orig, mode)
    d = np.abs(out.astype(int) - orig.astype(int))
    mx = d.max(axis=2)
    return d.max(), d.mean(), (mx > 0).mean() * 100, (mx > 30).mean() * 100


if __name__ == '__main__':
    photo = data.astronaut()

    Y, Cb, Cr = rgb2ycbcr(photo)
    back = ycbcr2rgb(Y, Cb, Cr)
    print('Завдання 1, макс похибка:',
          np.abs(back.astype(int) - photo.astype(int)).max())

    txt = redtext()
    out = scheme(txt, '4:2:2')
    Image.fromarray(np.concatenate([txt, out], axis=0)).save('text422.png')
    print('4:2:2 текст, макс похибка:',
          np.abs(out.astype(int) - txt.astype(int)).max())

    out = scheme(txt, '4:2:0')
    Image.fromarray(np.concatenate([txt, out], axis=0)).save('text420.png')
    d = np.abs(out.astype(int) - txt.astype(int))
    print('4:2:0 текст, макс похибка:', d.max(),
          'по каналах R,G,B:', [int(d[..., k].max()) for k in range(3)])

    save_diff(txt, 'diff_text.png', 'Червоний текст, 4:2:0')
    save_diff(photo, 'diff_photo.png', 'Фотографія, 4:2:0')

    tests = {'червоний текст': txt, 'тонкі лінії': lines(), 'фотографія': photo}
    for name, im in tests.items():
        for m in ('4:4:4', '4:2:2', '4:2:0'):
            print(name, m, '%3d %5.2f %5.1f%% %5.2f%%' % measure(im, m))

    lin = lines()
    trio = [scheme(lin, m) for m in ('4:4:4', '4:2:2', '4:2:0')]
    Image.fromarray(np.concatenate(trio, axis=1)).save('lines420.png')

    # Додатково: Pillow JPEG
    Image.fromarray(txt).save('q92_444.jpg', quality=92, subsampling=0)
    Image.fromarray(txt).save('q92_420.jpg', quality=92, subsampling=2)