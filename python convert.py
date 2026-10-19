import sys
import struct
import numpy as np

def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    
    pos = 0
    fields = []
    
    while len(fields) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
            
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] not in (b"\n", b""):
                pos += 1
            continue
            
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
            
        fields.append(data[start:pos])
        
    if fields[0] != b"P6":
        raise ValueError("Це не PPM формату P6")
        
    width = int(fields[1])
    height = int(fields[2])
    maxval = int(fields[3])
    
    if maxval != 255:
        raise ValueError("Підтримується лише maxval = 255")
        
    pos += 1
    need = width * height * 3
    if len(data) - pos < need:
        raise ValueError("Файл обірваний: пікселів менше, ніж треба")
        
    pixels = np.frombuffer(data, dtype=np.uint8, count=need, offset=pos)
    return pixels.reshape(height, width, 3)

def write_bmp24(path, img):
    height, width, _ = img.shape
    stride = ((width * 24 + 31) // 32) * 4
    padding = stride - width * 3
    offbits = 54
    filesize = offbits + stride * height
    
    out = bytearray()
    out += struct.pack("<2sIHHI", b"BM", filesize, 0, 0, offbits)
    out += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, stride * height, 2835, 2835, 0, 0)
    
    for y in range(height - 1, -1, -1):
        for x in range(width):
            r, g, b = img[y, x]
            out += bytes((b, g, r))
        out += b"\x00" * padding
        
    with open(path, "wb") as f:
        f.write(out)

def read_bmp24(path):
    with open(path, "rb") as f:
        data = f.read()
        
    offbits = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    height = struct.unpack_from("<i", data, 22)[0]
    bits = struct.unpack_from("<H", data, 28)[0]
    
    stride = ((width * bits + 31) // 32) * 4
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        start = offbits + y * stride
        for x in range(width):
            b = data[start + x * 3 + 0]
            g = data[start + x * 3 + 1]
            r = data[start + x * 3 + 2]
            img[height - 1 - y, x] = (r, g, b)
            
    return img

def write_bmp8(path, img):
    height, width, _ = img.shape
    stride = ((width * 8 + 31) // 32) * 4
    padding = stride - width * 1
    offbits = 1078
    filesize = offbits + stride * height
    
    palette = bytearray()
    for i in range(256):
        if i < 216:
            r = (i // 36) * 51
            g = ((i // 6) % 6) * 51
            b = (i % 6) * 51
        else:
            r = g = b = 0
        palette += bytes((b, g, r, 0))
        
    out = bytearray()
    out += struct.pack("<2sIHHI", b"BM", filesize, 0, 0, offbits)
    out += struct.pack("<IiiHHIIiiII", 40, width, height, 1, 8, 0, stride * height, 2835, 2835, 256, 0)
    out += palette
    
    def color_index(r, g, b):
        return 36 * (int(r) // 51) + 6 * (int(g) // 51) + (int(b) // 51)

    for y in range(height - 1, -1, -1):
        for x in range(width):
            r, g, b = img[y, x]
            idx = color_index(r, g, b)
            out += bytes((idx,))
        out += b"\x00" * padding
        
    with open(path, "wb") as f:
        f.write(out)

def run_tests():
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    img[0:2, 0:3] = (255, 0, 0)
    img[0:2, 3:6] = (0, 255, 0)
    img[2:4, 0:3] = (0, 0, 255)
    img[2:4, 3:6] = (255, 255, 255)
    
    with open("test.ppm", "wb") as f:
        f.write(b"P6\n6 4\n255\n")
        f.write(img.tobytes())
        
    print("Збережено тестовий файл test.ppm")
    
    a = read_ppm("test.ppm")
    
    write_bmp24("test.bmp", a)
    print("Збережено 24-бітний файл test.bmp")
    
    b = read_bmp24("test.bmp")
    is_equal = np.array_equal(a, b)
    print(f"Чи збігаються масиви (Завдання 3): {is_equal}")
    
    write_bmp8("test_palette.bmp", a)
    print("Збережено 8-бітний палітровий файл test_palette.bmp")

def main():
    if len(sys.argv) == 1:
        run_tests()
    elif len(sys.argv) == 3:
        src = sys.argv[1]
        dst = sys.argv[2]
        if src.endswith(".ppm") and dst.endswith(".bmp"):
            write_bmp24(dst, read_ppm(src))
            print(f"Успішно конвертовано: {src} -> {dst}")
        elif src.endswith(".bmp") and dst.endswith(".ppm"):
            img = read_bmp24(src)
            height, width, _ = img.shape
            with open(dst, "wb") as f:
                f.write(f"P6\n{width} {height}\n255\n".encode('ascii'))
                f.write(img.tobytes())
            print(f"Успішно конвертовано: {src} -> {dst}")
        else:
            print("Не знаю, як перетворити ці формати. Підтримуються .ppm -> .bmp та .bmp -> .ppm")
    else:
        print("Використання як конвертер: python convert.py <вхідний_файл> <вихідний_файл>")
        print("Або запустіть без аргументів: python convert.py (для базової перевірки та генерації файлів)")

if __name__ == "__main__":
    main()