import torch


class NarrowbandSignal:
    def __init__(self, metadata: dict):
        self.metadata = metadata
    
    def generate(self, t):
        fc = self.metadata['fc']
        signal = torch.ones_like(t)
        # carrier = torch.exp(2j * torch.pi * fc * t)
        carrier = 1.0  # no modulation, keep baseband
        signal = signal * carrier
        return signal

class RandomWidebandSignal:
    def __init__(self, metadata: dict):
        self.metadata = metadata
    
    def generate(self, t):
        sample_rate = 1 / (t[1] - t[0])
        n_samples = t.shape[0]
        f = torch.fft.fftfreq(n_samples, 1 / sample_rate)
        fc = self.metadata['fc']
        bw = self.metadata['bw']

        # randn bandlimited signal
        signal = torch.randn(t.shape)
        signal[(f < -bw/2) | (f > bw/2)] = 0
        # modulate up to fc
        signal = torch.fft.ifft(signal)
        # carrier = torch.exp(2j * torch.pi * fc * t)
        carrier = 1.0  # no modulation, keep baseband
        signal = signal * carrier

        return signal

class FMCWSignal:
    def __init__(self, metadata: dict):
        self.metadata = metadata
    
    def generate(self, t):
        fc = self.metadata['fc']
        bw = self.metadata['bw']
        chirp_duration = self.metadata['chirp_duration']

        # compute chirp constant
        k = bw / chirp_duration

        # compute baseband chirp
        signal = torch.exp(2j * torch.pi * (k * t**2 / 2))
        signal *= torch.exp(-2j * torch.pi * bw/2 * t)
        # modulate up to fc
        # carrier = torch.exp(2j * torch.pi * fc * t)
        carrier = 1.0  # no modulation, keep baseband
        signal = signal * carrier

        return signal


if __name__ == "__main__":
    # try narrowband, wideband, and FMCW
    metadata = {
        'fc': 100e9,
        'bw': 10e9,
        'sample_rate': 20e9,
        'chirp_duration': 100e-6,
    }
    narrowband_signal = NarrowbandSignal(metadata)
    wideband_signal = RandomWidebandSignal(metadata)
    fmcw_signal = FMCWSignal(metadata)

    n_samples = int(metadata['chirp_duration'] * metadata['sample_rate'])
    print('n_samples:', n_samples)
    t = torch.arange(n_samples) / metadata['sample_rate']

    y_narrowband = narrowband_signal.generate(t)
    y_wideband = wideband_signal.generate(t)
    y_fmcw = fmcw_signal.generate(t)

    y_narrowband_fft = torch.fft.fft(y_narrowband)
    y_wideband_fft = torch.fft.fft(y_wideband)
    y_fmcw_fft = torch.fft.fft(y_fmcw)

    f = torch.fft.fftfreq(n_samples, 1 / metadata['sample_rate'])

    import matplotlib.pyplot as plt
    plt.figure()
    plt.suptitle('Time Domain')
    plt.subplot(1, 3, 1)
    plt.plot(t, torch.real(y_narrowband))
    plt.subplot(1, 3, 2)
    plt.plot(t, torch.real(y_wideband))
    plt.subplot(1, 3, 3)
    plt.plot(t, torch.real(y_fmcw))

    plt.figure()
    plt.suptitle('Frequency Domain')
    plt.subplot(1, 3, 1)
    plt.plot(f, torch.abs(y_narrowband_fft))
    plt.subplot(1, 3, 2)
    plt.plot(f, torch.abs(y_wideband_fft))
    plt.subplot(1, 3, 3)
    plt.plot(f, torch.abs(y_fmcw_fft))
    plt.show()
