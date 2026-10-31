import numpy as np
from PIL import Image
from skimage import data


def to_light(c):
    c = np.asarray(c, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92,
                    ((c + 0.055) / 1.055) ** 2.4)


def to_code(L):
    L = np.asarray(L, dtype=float)
    c = np.where(L <= 0.0031308, L * 12.92,
                 1.055 * np.power(np.maximum(L, 0), 1 / 2.4) - 0.055)
    return np.clip(c * 255, 0, 255)


def half_naive(img):
    h, w = img.shape
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def half_linear(img):
    L = to_light(img)
    h, w = L.shape
    return to_code(L.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)))


def checker(N=512, cell=4, ang=18):
    y, x = np.indices((N, N)).astype(float) - N / 2
    a = np.deg2rad(ang)
    xr = x * np.cos(a) - y * np.sin(a)
    yr = x * np.sin(a) + y * np.cos(a)
    return (((np.floor(xr / cell) + np.floor(yr / cell)) % 2) * 255)


if __name__ == '__main__':
    gray = np.array(Image.fromarray(data.astronaut()).convert('L'))

    print('=== Завдання 1. Збільшення ===')
    small = gray[340:356, 372:388]
    im = Image.fromarray(small)
    im.resize((320, 320), Image.NEAREST).save('up_nearest.png')
    im.resize((320, 320), Image.BILINEAR).save('up_bilinear.png')
    im.resize((320, 320), Image.LANCZOS).save('up_lanczos.png')
    a = np.array(Image.open('up_nearest.png')).astype(int)
    b = np.array(Image.open('up_bilinear.png')).astype(int)
    c = np.array(Image.open('up_lanczos.png')).astype(int)
    print('унікальних значень: nearest %d, bilinear %d, lanczos %d'
          % (len(np.unique(a)), len(np.unique(b)), len(np.unique(c))))
    print('середня |nearest-bilinear| = %.2f, |bilinear-lanczos| = %.2f'
          % (np.abs(a - b).mean(), np.abs(b - c).mean()))
    blk = a.reshape(16, 20, 16, 20)
    print('nearest складається з однорідних блоків 20x20:',
          bool((blk == blk[:, :1, :, :1]).all()))

    print('\n=== Завдання 2. Зменшення й аліасинг ===')
    ch = checker()
    k = 4
    nearest = ch[::k, ::k]
    area = ch.reshape(512 // k, k, 512 // k, k).mean(axis=(1, 3))
    for name, im2 in (('ds_nearest', nearest), ('ds_area', area)):
        Image.fromarray(im2.astype(np.uint8)).save(name + '.png')
        print(name, 'середнє %.1f розкид %.1f' % (im2.mean(), im2.std()))
    # додатково: оригінал для порівняння
    Image.fromarray(ch.astype(np.uint8)).save('checker_original.png')

    print('\n=== Завдання 3. Ланцюжок міпмап ===')
    g = gray.astype(float)
    level = 0
    while g.shape[0] >= 2:
        Image.fromarray(g.astype(np.uint8)).save(f'mip_{level}.png')
        print(level, g.shape[0], 'середня яскравість %.1f' % g.mean())
        g = half_naive(g)
        level += 1
    print('усього рівнів:', level)

    print('\n=== Завдання 4. Скільки коштує гама ===')
    cb = np.indices((256, 256)).sum(axis=0) % 2 * 255.0
    print('наївно: %.1f' % half_naive(cb).mean())
    print('правильно: %.1f' % half_linear(cb).mean())

    gn = gray.astype(float)
    gl = gray.astype(float)
    print(0, gn.shape[0], '%.1f %.1f різниця %+.1f'
          % (gn.mean(), gl.mean(), gl.mean() - gn.mean()))
    for lv in range(1, 6):
        gn, gl = half_naive(gn), half_linear(gl)
        print(lv, gn.shape[0], '%.1f %.1f різниця %+.1f'
              % (gn.mean(), gl.mean(), gl.mean() - gn.mean()))

    print('\n=== Додатково: Pillow ===')
    imc = Image.fromarray(ch.astype(np.uint8))
    imc.resize((128, 128), Image.NEAREST).save('lib_nearest.png')
    imc.resize((128, 128), Image.BOX).save('lib_box.png')
    imc.resize((128, 128), Image.LANCZOS).save('lib_lanczos.png')
    box = np.array(Image.open('lib_box.png')).astype(float)
    print('макс. розбіжність lib_box і ds_area: %.1f'
          % np.abs(box - area.astype(np.uint8).astype(float)).max())
    cb_img = Image.fromarray(cb.astype(np.uint8))
    r = np.array(cb_img.resize((128, 128), Image.BOX))
    print('BOX на шахівниці 1 піксель: середнє %.1f' % r.mean())