import numpy as np
import matplotlib.pyplot as plt

from e2e.environment.sionna_iterator import SionnaMunichIterator



N_RX_X = 32
N_RX_Y = 32
N_TX = 1
N_FREQS = 1000
freqs = np.linspace(28.5e9, 31.5e9, N_FREQS)

sionna_iter = SionnaMunichIterator()

def time_gate(s_pars):
    return s_pars
    cir = np.fft.ifft(s_pars, axis=-1)
    window = np.zeros_like(cir)
    window[:, :, 100:-100] = 1.
    cir_gated = cir * window
    s_pars_gated = np.fft.fft(cir_gated, axis=-1)
    return s_pars_gated

s_pars = sionna_iter.all_s_pars[0]
s_pars = s_pars.reshape(N_RX_X, N_RX_Y, N_FREQS)
s_pars = time_gate(s_pars)
cir = np.fft.ifft(s_pars, axis=-1)

s_pars = sionna_iter.all_s_pars[50]
s_pars = s_pars.reshape(N_RX_X, N_RX_Y, N_FREQS)
s_pars = time_gate(s_pars)
cir2 = np.fft.ifft(s_pars, axis=-1)


plt.figure()
plt.plot(10 * np.log10(np.abs(cir[0, 0])))
plt.plot(10 * np.log10(np.abs(cir2[0, 0])))
plt.figure()
plt.subplot(121)
plt.plot(10 * np.log10(np.abs(s_pars[0, 0])))
plt.subplot(122)
plt.plot(np.unwrap(np.angle(s_pars[0, 0])))
plt.show()

plt.figure()
for i, s_pars in enumerate(sionna_iter):
    if i % 10 != 0:
        continue
    print(i, s_pars.shape, np.mean(s_pars))
    s_pars = s_pars.reshape(N_RX_X, N_RX_Y, N_FREQS)
    s_pars = time_gate(s_pars)

    s_pars_fft = np.fft.fft(np.fft.fft(s_pars, 256, 0), 256, 1)
    s_pars_fft = np.fft.fftshift(np.fft.fftshift(s_pars_fft, 0), 1)
    s_pars_fft_energy = np.sum(np.abs(s_pars_fft) ** 2, axis=2)
    clim = [np.max(10 * np.log10(s_pars_fft_energy)) - 40, np.max(10 * np.log10(s_pars_fft_energy))]

    plt.imshow(10 * np.log10(s_pars_fft_energy))
    if i == 0:
        plt.colorbar()
    plt.clim(clim)
    plt.show()
