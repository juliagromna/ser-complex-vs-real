"""
Wspólna konfiguracja eksperymentu.

Moduł gromadzi wszystkie parametry współdzielone przez notatniki: ścieżki,
parametry analizy czasowo-częstotliwościowej, wymiary reprezentacji,
hiperparametry procesu uczenia oraz definicje klas emocjonalnych.

Notatniki zawierają własne komórki konfiguracyjne, dzięki czemu mogą być
uruchamiane niezależnie w środowisku Kaggle Notebooks. Niniejszy plik pełni rolę
zbiorczej dokumentacji przyjętych ustawień oraz punktu odniesienia przy
odtwarzaniu eksperymentu.
"""

from pathlib import Path

# --- srodowisko i sciezki -------------------------------------------------

ON_KAGGLE = Path("/kaggle/input").exists()

if ON_KAGGLE:
    RAW_DIR = Path("/kaggle/input/cremad/AudioWAV")
    DATA = Path("/kaggle/input/ser-processed")
    WORK = Path("/kaggle/working")
else:
    RAW_DIR = Path("data/raw/CREMA-D/AudioWAV")
    DATA = Path("data/processed")
    WORK = Path(".")

RESULTS = WORK / "results"
FIGURES = WORK / "figures"

# --- parametry sygnalu ----------------------------------------------------

SR = 16000                # czestotliwosc probkowania [Hz]
TRIM_TOP_DB = 30          # prog przycinania ciszy [dB ponizej szczytu]
SEG_SEC = 1.0             # dlugosc segmentu [s]
SEG_HOP_SEC = 0.5         # przesuniecie okna segmentacji [s]
MIN_TAIL_RATIO = 0.6      # minimalny udzial dlugosci ostatniego segmentu
PEAK_TARGET = 0.95        # docelowa wartosc szczytowa po normalizacji

SEG_LEN = int(SEG_SEC * SR)
SEG_HOP = int(SEG_HOP_SEC * SR)

# --- transformata czasowo-czestotliwosciowa -------------------------------

N_FFT = 512               # rozmiar transformaty
WIN_LENGTH = 400          # dlugosc okna analizy (25 ms)
HOP_LENGTH = 160          # przesuniecie ramki (10 ms)
WINDOW = "hann"
CENTER = True

N_FRAMES = SEG_LEN // HOP_LENGTH + 1     # 101 ramek czasowych
N_BINS = N_FFT // 2 + 1                  # 257 prazkow czestotliwosciowych

# --- reprezentacje --------------------------------------------------------

N_MELS = 64               # liczba filtrow melowych
N_MFCC = 40               # zachowywane wspolczynniki cepstralne
FMIN, FMAX = 20, 8000     # zakres czestotliwosci banku filtrow [Hz]
POWER_COMPRESSION = 0.3   # wykladnik kompresji potegowej modulu

# --- podzial danych -------------------------------------------------------

SEED = 42                 # ziarno podzialu na mowcow
N_TEST_ACTORS = 14
N_VAL_ACTORS = 14         # pozostalych 63 aktorow tworzy zbior treningowy

# --- uczenie --------------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]     # ziarna inicjalizacji, wspolne dla modeli
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 15
GRAD_CLIP = 5.0

BATCH_SIZE = 128          # modele rzeczywiste
BATCH_SIZE_COMPLEX = 64   # model zespolony, ograniczenie pamieci akceleratora

# --- konfiguracja stabilizowana modelu zespolonego ------------------------

STABLE_LABEL_SMOOTHING = 0.05
STABLE_GRAD_CLIP = 1.0
STABLE_MAX_EPOCHS = 60
STABLE_PATIENCE = 25
STABLE_PCT_START = 0.05   # udzial fazy narastania w harmonogramie OneCycleLR

# --- architektury ---------------------------------------------------------

# liczby filtrow w kolejnych blokach splotowych
CH_REAL = (32, 64, 128, 128)     # modele M1, M2, M3
CH_COMPLEX = (23, 45, 90, 90)    # model M4, redukcja o czynnik sqrt(2)

RNN_HIDDEN_REAL = 128
RNN_HIDDEN_COMPLEX = 90

# stopnie redukcji w warstwach redukujacych, (czestotliwosc, czas)
POOLS_MFCC = ((2, 2), (2, 2), (2, 1), (2, 1))     # wejscie 40 prazkow
POOLS_SPEC = ((4, 2), (4, 2), (4, 1), (2, 1))     # wejscie 257 prazkow

# --- klasy emocjonalne ----------------------------------------------------

EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
EMOTION_PL = {
    "ANG": "złość",
    "DIS": "wstręt",
    "FEA": "strach",
    "HAP": "radość",
    "NEU": "neutralny",
    "SAD": "smutek",
}
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}
N_CLASSES = len(EMOTIONS)

LABELS_PL = [EMOTION_PL[e] for e in EMOTIONS]

# --- pliki wynikowe -------------------------------------------------------

PLIKI_WYNIKOW = {
    "M1 (MFCC)":      "m1_mfcc_5seeds.json",
    "M2 (|STFT|)":    "m2_magnitude_5seeds.json",
    "M3 (Re/Im)":     "m3_reim_5seeds.json",
    "M4 (zespolony)": "m4_complex_5seeds.json",
}
PLIK_STABILIZOWANY = "m4_stable_5seeds.json"

# paleta o zroznicowanej jasnosci, czytelna rowniez w skali szarosci
KOLORY = {
    "M1 (MFCC)":      "#16305c",
    "M2 (|STFT|)":    "#4a8fb5",
    "M3 (Re/Im)":     "#d99b3a",
    "M4 (zespolony)": "#8c2f39",
}


if __name__ == "__main__":
    print(f"środowisko          : {'Kaggle' if ON_KAGGLE else 'lokalne'}")
    print(f"dane surowe         : {RAW_DIR}  (istnieje: {RAW_DIR.exists()})")
    print(f"dane przetworzone   : {DATA}  (istnieje: {DATA.exists()})")
    print(f"segment             : {SEG_SEC} s = {SEG_LEN} próbek")
    print(f"kształt STFT        : {N_BINS} × {N_FRAMES}")
    print(f"ziarna              : {SEEDS}")
    print(f"klasy               : {', '.join(LABELS_PL)}")
