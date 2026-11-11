import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SR = 44100


def music(dur=3.0, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    x = np.zeros_like(t)
    for f0 in (220.0, 277.18, 329.63):
        for k, a in enumerate([1, .5, .3, .18, .1, .06], start=1):
            x += a * np.sin(2 * np.pi * f0 * k * t + k)
    x *= np.minimum(1, t * 8) * np.exp(-0.35 * t)
    return 0.85 * x / np.abs(x).max()


def save_wav(path, x, sr=SR):
    x = np.asarray(x)
    x = x / max(1.0, np.abs(x).max())
    wavfile.write(path, sr, np.round(x * 32767).astype(np.int16))


def eq(x, bands, sr=SR):
    sp = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    g = np.ones(len(sp))
    for lo, hi, db in bands:
        g[(fr >= lo) & (fr < hi)] = 10 ** (db / 20)
    return np.fft.irfft(sp * g, len(x))


def band_energy(x, lo, hi, sr=SR):
    sp = np.abs(np.fft.rfft(x))
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    return 10 * np.log10((sp[(fr >= lo) & (fr < hi)] ** 2).sum() + 1e-20)


def main():
    x = music()
    wavfile.write("source.wav", SR, (x * 32767).astype(np.int16))
    n = 8192
    sp = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    db = 20 * np.log10(sp / sp.max() + 1e-12)
    fr = np.fft.rfftfreq(n, 1 / SR)
    plt.figure(figsize=(8, 4))
    plt.plot(fr[fr < 2500], db[fr < 2500], lw=0.8)
    plt.xlabel("Частота, Гц"); plt.ylabel("Рівень, дБ"); plt.ylim(-100, 5)
    plt.title("Спектр фрагмента (вікно Ганна, n = 8192)"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("spectrum.png", dpi=120); plt.close()

    f, t, S = spectrogram(x, SR, nperseg=2048, noverlap=1792)
    plt.figure(figsize=(8, 4))
    plt.pcolormesh(t, f, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma")
    plt.ylim(0, 3000); plt.xlabel("Час, с"); plt.ylabel("Частота, Гц")
    plt.title("Спектрограма (nperseg = 2048)"); plt.colorbar(label="дБ")
    plt.tight_layout(); plt.savefig("spectrogram.png", dpi=120); plt.close()
    top = np.sort(fr[np.argsort(sp)[-30:]])
    print("Піки близько 220/277/330:", [round(v) for v in top if v < 360][:12])

    n = 1024
    wins = (("прямокутне", np.ones(n)), ("Ганна", np.hanning(n)), ("Блекмана", np.blackman(n)))
    table = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)
    for r, (k, label) in enumerate(((100.0, "рівно на біні"), (100.5, "між бінами"))):
        f0 = k * SR / n
        tt = np.arange(n) / SR
        xx = np.sin(2 * np.pi * f0 * tt)
        for c, (name, w) in enumerate(wins):
            s = np.abs(np.fft.rfft(xx * w))
            d = 20 * np.log10(s / s.max() + 1e-15)
            far = np.concatenate([d[:95], d[106:]])
            table[(label, name)] = far.max()
            print(label, name, "найвищий сторонній пік %.1f дБ" % far.max())
            axes[r, c].plot(d, lw=0.7); axes[r, c].set_xlim(50, 160)
            axes[r, c].set_ylim(-140, 5); axes[r, c].grid(alpha=.3)
            axes[r, c].set_title(f"{name}, {label}")
    for ax in axes[1]: ax.set_xlabel("Номер біна")
    for ax in axes[:, 0]: ax.set_ylabel("дБ")
    plt.tight_layout(); plt.savefig("windows.png", dpi=110); plt.close()

    dur = 1.4
    t = np.arange(int(dur * SR)) / SR
    z = np.zeros_like(t)
    seg = t < 0.6
    z[seg] = 0.5 * np.sin(2 * np.pi * 440 * t[seg]) + 0.5 * np.sin(2 * np.pi * 480 * t[seg])
    for c in (0.80, 0.92, 1.04, 1.16):
        i = int(c * SR)
        z[i:i + 40] += 1.0 * np.hanning(40)
    for nn in (128, 4096):
        f, tt, S = spectrogram(z, SR, nperseg=nn, noverlap=int(nn * 0.9))
        plt.figure(figsize=(8, 4))
        plt.pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading="auto", cmap="magma")
        plt.ylim(0, 1500); plt.xlabel("Час, с"); plt.ylabel("Частота, Гц")
        plt.title(f"Спектрограма, вікно {nn}"); plt.colorbar(label="дБ")
        plt.tight_layout(); plt.savefig(f"spec_{nn}.png", dpi=120); plt.close()

    y = eq(x, [(0, 250, 12)])
    bands = ((0, 250), (250, 1000), (1000, 4000), (4000, SR // 2))
    rows = []
    for lo, hi in bands:
        a, b = band_energy(x, lo, hi), band_energy(y, lo, hi)
        rows.append((lo, hi, a, b))
        print(lo, hi, "%.1f → %.1f  (Δ %.2f)" % (a, b, b - a))
    fx = np.fft.rfftfreq(len(x), 1 / SR)
    plt.figure(figsize=(8, 4))
    for sig, lab in ((x, "до"), (y, "після (+12 дБ, 0–250 Гц)")):
        s = np.abs(np.fft.rfft(sig)); plt.plot(fx, 20 * np.log10(s / s.max() + 1e-12), lw=.7, label=lab)
    plt.xlim(0, 1500); plt.ylim(-100, 5); plt.legend(); plt.grid(alpha=.3)
    plt.xlabel("Частота, Гц"); plt.ylabel("дБ"); plt.title("Спектр до і після еквалайзера")
    plt.tight_layout(); plt.savefig("eq_spectrum.png", dpi=120); plt.close()

    presets = {
        "preset_bass.wav": [(0, 250, 12)],
        "preset_voice.wav": [(0, 150, -12), (1000, 4000, 6)],
        "preset_phone.wav": [(0, 300, -60), (3000, SR // 2, -60)],
    }
    for name, bnd in presets.items():
        save_wav(name, eq(x, bnd))

    ok = all(abs((b - a) - (12 if lo == 0 else 0)) < 0.01 for lo, hi, a, b in rows)
    print(("ТАК " if ok else "НІ  ") + "еквалайзер: +12,0 дБ у першій смузі, решта без змін")
    return table, rows


if __name__ == "__main__":
    main()