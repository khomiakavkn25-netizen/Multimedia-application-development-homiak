import wave
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 44100


def save_wav(path, x, sr=SR):
    y = np.clip(np.round(np.asarray(x) * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(y.tobytes())


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return x.astype(float) / 32767, sr


def snr_db(x, q):
    noise = q - x
    return 10 * np.log10(np.sum(x ** 2) / np.sum(noise ** 2))


def spectrum_db(x, sr=SR):
    w = np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x * w)) / (np.sum(w) / 2)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return f, 20 * np.log10(mag + 1e-12)


def peak_freq(x, sr=SR):
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return float(f[np.argmax(mag)])


def plot_spectrogram(x, path, sr=SR):
    plt.figure(figsize=(8, 4))
    plt.specgram(x, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
    plt.xlabel("Час, с")
    plt.ylabel("Частота, Гц")
    plt.title("Спектрограма свіпу (sr = %d Гц)" % sr)
    plt.colorbar(label="дБ")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_spectrum(items, path, sr=SR):
    fig, axes = plt.subplots(1, len(items), figsize=(11, 4), sharey=True)
    for ax, (x, title) in zip(np.atleast_1d(axes), items):
        f, d = spectrum_db(x, sr)
        ax.plot(f, d, lw=0.8)
        ax.set_xlim(0, 6000)
        ax.set_ylim(-120, 5)
        ax.set_title(title)
        ax.set_xlabel("Частота, Гц")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("Рівень, дБ")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def tone(freq, dur=2.0, amp=0.5, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


def sweep(f0, f1, dur, amp=0.5, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    ph = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2)
    return amp * np.sin(ph)


def quantize(x, bits):
    L = 2 ** bits
    k = L / 2 - 1
    return np.round(x * k) / k


def dither(x, bits, seed=0):
    rng = np.random.default_rng(seed)
    step = 2 / 2 ** bits
    return quantize(x + rng.uniform(-step / 2, step / 2, len(x)), bits)


def check(name, ok):
    print(("ТАК " if ok else "НІ  ") + name)
    return ok


def window_freqs(x, sr, n=5):
    m = len(x) // n
    return [peak_freq(x[i * m:(i + 1) * m], sr) for i in range(n)]


def max_harmonic_db(x, f0, sr=SR, nh=12):
    f, d = spectrum_db(x, sr)
    ref = d[np.argmin(np.abs(f - f0))]
    vals = []
    for k in range(2, nh + 1):
        if k * f0 < sr / 2:
            i = np.argmin(np.abs(f - k * f0))
            vals.append(d[max(i - 3, 0):i + 4].max() - ref)
    return max(vals)


def main():
    for f in (220, 440, 880):
        save_wav(f"tone_{f}.wav", tone(f, 2.0))
    check("Завд.1: тон 440 Гц має пік 440 Гц", abs(peak_freq(tone(440, 2.0)) - 440) < 2)
    check("Завд.1: тривалість файлу 2 с", len(tone(440, 2.0)) == 2 * SR)

    s0 = sweep(200, 800, 4.0, sr=SR)
    fr = window_freqs(s0, SR)
    print("   свіп 200→800 Гц, 5 вікон:", [round(v) for v in fr])
    check("Завд.2: свіп росте рівномірно (260,380,500,620,740)",
          all(abs(a - b) < 15 for a, b in zip(fr, [260, 380, 500, 620, 740])))
    sr = 8000
    s = sweep(0, sr, 6.0, sr=sr)
    save_wav("sweep.wav", s, sr=sr)
    plot_spectrogram(s, "sweep.png", sr=sr)
    res = {}
    for f in (3000, 5000, 7000):
        x = tone(f, 1.0, sr=sr)
        save_wav(f"alias_{f}.wav", x, sr=sr)
        res[f] = round(peak_freq(x, sr))
        print("  ", f, "→", res[f])
    check("Завд.2: аліасинг 3000→3000, 5000→3000, 7000→1000",
          res == {3000: 3000, 5000: 3000, 7000: 1000})

    t = tone(440, 2.0, amp=0.9)
    oks = True
    for b in (16, 8, 4):
        q = quantize(t, b)
        save_wav(f"quant_{b}bit.wav", q)
        meas, theory = snr_db(t, q), 6.02 * b + 1.76
        print(f"   {b} біт: SNR {meas:.1f} дБ (теорія {theory:.1f})")
        oks &= abs(meas - theory) < 2.0
    check("Завд.3: SNR близький до теорії (допуск 2 дБ)", oks)

    q = tone(440, 3.0, amp=0.25)
    a, b = quantize(q, 4), dither(q, 4)
    save_wav("dither_off.wav", a)
    save_wav("dither_on.wav", b)
    plot_spectrum([(a, "без дизеру"), (b, "з дизером")], "dither.png")
    s_off, s_on = snr_db(q, a), snr_db(q, b)
    h_off, h_on = max_harmonic_db(a, 440), max_harmonic_db(b, 440)
    print(f"   SNR без: {s_off:.1f} дБ, з: {s_on:.1f} дБ")
    print(f"   найгучніша гармоніка без: {h_off:.1f} дБ, з: {h_on:.1f} дБ")
    check("Завд.4: дизер знижує гармоніки", h_on < h_off - 10)
    check("Завд.4: SNR з дизером гірший", s_on < s_off)


if __name__ == "__main__":
    main()