import sys
import librosa
import librosa.display
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QTabWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import scipy


class WaveformSpectrogramPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 1600, 1000)
        self.setWindowTitle('Music Information Retrieval Softwware')
        self.setWindowIcon(QIcon('icon.png'))

        # Create button
        self.btn = QPushButton('Select WAV file', self)
        self.btn.setToolTip('Click to select a WAV file')
        self.btn.clicked.connect(self.selectWAV)
        self.btn.setStyleSheet(
            'background-color: #000000; color: gold; font-weight: bold; padding: 10px; border-radius: 5px;')

        self.fig1 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas1 = FigureCanvas(self.fig1)
        self.ax1 = self.fig1.add_subplot(111)  # 1st subplot for waveform
        self.ax1.set_xlabel('Time (seconds)')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.set_title(
            'Waveform Plot', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax1.set_facecolor("black")

        self.fig2 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas2 = FigureCanvas(self.fig2)
        # 2nd subplot for Log Spectogram
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.set_ylabel('Frequency')
        self.ax2.set_title('Logarithmic Frequency Spectogram',
                           fontname="Times New Roman", size=28, fontweight="bold")
        self.ax2.set_facecolor("black")

        self.fig5 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas5 = FigureCanvas(self.fig5)
        # 5th subplot for Lin Spectogram
        self.ax5 = self.fig5.add_subplot(111)
        self.ax5.set_xlabel('Time (seconds)')
        self.ax5.set_ylabel('Frequency')
        self.ax5.set_title('Logarithmic Frequency Spectogram',
                           fontname="Times New Roman", size=28, fontweight="bold")
        self.ax5.set_facecolor("black")

        self.fig3 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas3 = FigureCanvas(self.fig3)
        # 3rd subplot for mel frequency spectrogram
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_xlabel('Time (seconds)')
        self.ax3.set_ylabel('Mel Frequency')
        self.ax3.set_title(
            'Mel Spectrogram', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax3.set_facecolor("black")

        self.fig4 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas4 = FigureCanvas(self.fig4)
        # 4th subplot for mel frequency cepctral coefficients
        self.ax4 = self.fig4.add_subplot(111)
        self.ax4.set_xlabel('Time (seconds)')
        self.ax4.set_ylabel('MFCC')
        self.ax4.set_title('MFCC', fontname="Times New Roman",
                           size=28, fontweight="bold")
        self.ax4.set_facecolor("black")

        # Chromagram STFT
        self.fig6 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas6 = FigureCanvas(self.fig6)
        self.ax6 = self.fig6.add_subplot(111)
        self.ax6.set_title(
            'Chromagram STFT', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax6.set_xlabel('Time')
        self.ax6.set_ylabel('Pitch Class')

        # Chromagram CQT
        self.fig7 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas7 = FigureCanvas(self.fig7)
        self.ax7 = self.fig7.add_subplot(111)
        self.ax7.set_title(
            'Chromagram CQT', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax7.set_xlabel('Time')
        self.ax7.set_ylabel('Pitch Class')

        # Chromagram CQT
        self.fig8 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas8 = FigureCanvas(self.fig8)
        self.ax8 = self.fig8.add_subplot(111)
        self.ax8.set_title(
            'Chromagram CENS', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax8.set_xlabel('Time')
        self.ax8.set_ylabel('Pitch Class')

        # Onset Detection
        self.fig9 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas9 = FigureCanvas(self.fig9)
        self.ax9 = self.fig9.add_subplot(111)
        self.ax9.set_title(
            'Onset Detection', fontname="Times New Roman", size=28, fontweight="bold")
        # self.ax9.set_xlabel('Time')
        # self.ax9.set_ylabel('Pitch Class')

        # RMS
        self.fig10 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas10 = FigureCanvas(self.fig10)
        self.ax10 = self.fig10.add_subplot(111)
        self.ax10.set_title('RMS', fontname="Times New Roman",
                            size=28, fontweight="bold")
        # self.ax10.set_xlabel('Time')
        # self.ax10.set_ylabel('Pitch Class')

        # New
        self.fig11 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas11 = FigureCanvas(self.fig11)
        self.ax11 = self.fig11.add_subplot(111)
        self.ax11.set_title(
            'Zero Crossing Rate', fontname="Times New Roman", size=28, fontweight="bold")
        # self.ax11.set_xlabel('Time')
        # self.ax11.set_ylabel('Pitch Class')

        # RMS
        self.fig12 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas12 = FigureCanvas(self.fig12)
        self.ax12 = self.fig12.add_subplot(111)
        self.ax12.set_title(
            'Spectral Centroid', fontname="Times New Roman", size=28, fontweight="bold")
        # self.ax12.set_xlabel('Time')
        # self.ax12.set_ylabel('Pitch Class')

        # RMS
        self.fig13 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas13 = FigureCanvas(self.fig13)
        self.ax13 = self.fig13.add_subplot(111)
        self.ax13.set_title(
            'Spectral Bandwidth', fontname="Times New Roman", size=28, fontweight="bold")
        # self.ax13.set_xlabel('Time')
        # self.ax13.set_ylabel('Pitch Class')

        # RMS
        self.fig14 = Figure(figsize=(12, 10), facecolor="gold")
        self.canvas14 = FigureCanvas(self.fig14)
        self.ax14 = self.fig14.add_subplot(111)
        self.ax14.set_title(
            'Spectral Body', fontname="Times New Roman", size=28, fontweight="bold")
        # self.ax14.set_xlabel('Time')
        # self.ax14.set_ylabel('Pitch Class')

        # Add figures to tabs
        self.tabs = QTabWidget()
        self.tabs2 = QTabWidget()
        self.tabs.setStyleSheet(
            'selection-background-color: black; background-color: black; color: gold; font-weight: bold; padding: 10px; border-radius: 5px;')
        self.tabs.addTab(self.canvas1, "Waveform Plot")
        self.tabs.addTab(self.canvas2, "Logarithmic Frequency Spectogram")
        self.tabs.addTab(self.canvas3, "Mel Spectogram")
        self.tabs.addTab(self.canvas4, "MFCC")
        self.tabs.addTab(self.canvas5, "Linear Frequency Spectogram")
        self.tabs.addTab(self.canvas6, "Chromagram STFT")
        self.tabs.addTab(self.canvas7, "Chromagram CQT")
        self.tabs.addTab(self.canvas8, "Chromagram CENS")
        self.tabs.addTab(self.canvas9, "Onset Detection")
        self.tabs.addTab(self.canvas10, "RMS")
        self.tabs.addTab(self.canvas11, "Zero Crossing Rate")
        self.tabs.addTab(self.canvas12, "Spectral Centroid")
        self.tabs.addTab(self.canvas13, "Spectral Bandwidth")
        self.tabs.addTab(self.canvas14, "Spectral Weight")

        # Create layouts
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.btn)
        self.layout.addWidget(self.tabs,)
        self.setLayout(self.layout)
        self.show()

    def selectWAV(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self, 'Select WAV file', '', 'WAV files (*.wav)')
        if filePath:
            self.plotWaveformSpectrogram(filePath)

    def plotWaveformSpectrogram(self, filePath):
        y, sr = librosa.load(filePath)

        # plot waveform
        t = np.linspace(0, len(y) / sr, len(y))
        self.ax1.clear()
        self.ax1.plot(t, y, linewidth=0.5, color="gold")
        self.ax1.set_title(
            "Waveform", fontname="Times New Roman", size=28, fontweight="bold")
        self.ax1.set_facecolor('xkcd:black')
        self.ax1.set_xlabel('Time (seconds)')
        self.ax1.set_ylabel('Amplitude')

        # Chromagram STFT
        self.ax6.clear()
        S_db3 = np.abs(librosa.stft(y, n_fft=4096))**2
        chroma1 = librosa.feature.chroma_stft(
            S=S_db3, sr=sr, n_chroma=12)
        librosa.display.specshow(
            chroma1, x_axis='time', y_axis='chroma', cmap='coolwarm', ax=self.ax6)
        self.ax6.set_title(
            'Chromagram STFT', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax6.set_xlabel('Time')
        self.ax6.set_ylabel('Pitch Class)')

        # Chromagram CQT
        self.ax7.clear()
        chroma2 = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12)
        librosa.display.specshow(
            chroma2, x_axis='time', y_axis='chroma', cmap='coolwarm', ax=self.ax7)
        self.ax7.set_title(
            'Chromagram CQT', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax7.set_xlabel('Time')
        self.ax7.set_ylabel('Pitch Class')

        # Chromagram CENS
        self.ax8.clear()
        chroma3 = librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=12)
        librosa.display.specshow(
            chroma3, x_axis='time', y_axis='chroma', cmap='coolwarm', ax=self.ax8)
        self.ax8.set_title(
            'Chromagram CENS', fontname="Times New Roman", size=28, fontweight="bold")
        self.ax8.set_xlabel('Time')
        self.ax8.set_ylabel('Pitch Class')

        # plot log spectrogram
        self.ax2.clear()
        S = librosa.stft(y)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        librosa.display.specshow(
            S_db, x_axis='time', y_axis='log', sr=sr, ax=self.ax2)
        self.ax2.set_title('Logarithmic Frequency Spectogram',
                           fontname="Times New Roman", size=28, fontweight="bold")

        # plot lin spectrogram
        self.ax5.clear()
        S2 = librosa.stft(y)
        S_db2 = librosa.amplitude_to_db(np.abs(S2), ref=np.max)
        librosa.display.specshow(
            S_db2, x_axis='time', y_axis='linear', sr=sr, ax=self.ax5)
        self.ax5.set_title('Linear Frequency Spectogram',
                           fontname="Times New Roman", size=28, fontweight="bold")

        # plot mel spectrogram
        self.ax3.clear()
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(
            mel_spec_db, x_axis='time', y_axis='mel', sr=sr, ax=self.ax3)
        self.ax3.set_title('Mel Spectogram',
                           fontname="Times New Roman", size=28, fontweight="bold")

        # MFCC
        self.ax4.clear()
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_fft=2048, hop_length=512, n_mfcc=20)
        mfcc_db = librosa.amplitude_to_db(mfcc, ref=np.max)
        librosa.display.specshow(
            mfcc_db, x_axis='time', sr=sr, ax=self.ax4)
        self.ax4.set_title('MFCC',
                           fontname="Times New Roman", size=28, fontweight="bold")

        # Onset Detection
        self.ax9.clear()
        o_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(o_env, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        self.ax9.plot(times, o_env, label='Onset strength')
        self.ax9.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
                        linestyle='--', label='Onsets')
        self.ax9.set_title(
            "Onset Detection", fontname="Times New Roman", size=28, fontweight="bold")

        # new
        self.ax10.clear()
        S = librosa.magphase(librosa.stft(y, window=np.ones, center=False))[0]
        rms = librosa.feature.rms(S=S)
        times = librosa.times_like(rms)
        self.ax10.semilogy(times, rms[0], label='RMS Energy')
        # self.ax10.set(xticks=[])
        self.ax10.set_xlabel('Time')
        self.ax10.set_ylabel('RMS Log')
        self.ax10.set_title("RMS", fontname="Times New Roman",
                            size=28, fontweight="bold")

        # Zero Crossing Rate
        self.ax11.clear()
        FL = 1024
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=FL)
        frames = range(0, len(zcr[0]))
        t = librosa.frames_to_time(frames)
        self.ax11.plot(t, zcr[0] * FL, label="Zero Crossing Rate")
        self.ax11.set_xlabel("Time")
        self.ax11.set_ylabel("Rate")
        self.ax11.set_title(
            "Zero Crossing Rate", fontname="Times New Roman", size=28, fontweight="bold")

        # Spectral Centroid
        self.ax12.clear()
        specC = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        frames = range(0, len(specC))
        t = librosa.frames_to_time(frames)
        self.ax12.plot(t, specC, label="Zero Crossing Rate")
        self.ax12.set_xlabel("Time")
        self.ax12.set_ylabel("Spectral Centroid")
        self.ax12.set_title(
            "Spectral Centroid", fontname="Times New Roman", size=28, fontweight="bold")

        # Spectral Bandwidth
        self.ax13.clear()
        specB = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        frames = range(0, len(specC))
        t = librosa.frames_to_time(frames)
        self.ax13.plot(t, specB, label="Zero Crossing Rate")
        self.ax13.set_xlabel("Time")
        self.ax13.set_ylabel("Spectral Bandwidth")
        self.ax13.set_title(
            "Spectral Bandwidth", fontname="Times New Roman", size=28, fontweight="bold")

        # Spectral Centroid and BandWidth on Spectogram
        self.ax14.clear()
        S, phase = librosa.magphase(librosa.stft(y=y))
        frames = range(0, len(specC))
        t = librosa.frames_to_time(frames)
        specB = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                 y_axis='log', x_axis='time', ax=self.ax14)
        self.ax14.set_title('Spectral Body',
                            fontname="Times New Roman", size=28, fontweight="bold")
        self.ax14.set(xlim=[t.min(), t.max()])
        self.ax14.fill_between(t, np.minimum(specC[0] + specB[0], sr/2), np.maximum(0, specC[0] - specB[0]),

                               alpha=0.3, label='Centroid +- bandwidth', color="white")
        self.ax14.plot(t, specC, label='Spectral centroid', color='w')

        # update plot
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        self.canvas4.draw()
        self.canvas5.draw()
        self.canvas6.draw()
        self.canvas7.draw()
        self.canvas8.draw()
        self.canvas9.draw()
        self.canvas10.draw()
        self.canvas11.draw()
        self.canvas12.draw()
        self.canvas13.draw()
        self.canvas14.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    waveformSpectrogramPlot = WaveformSpectrogramPlot()
    sys.exit(app.exec_())
