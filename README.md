# Rzeczywiste i zespolone reprezentacje sygnału mowy w rozpoznawaniu emocji

Kod źródłowy do pracy magisterskiej *Analiza porównawcza rzeczywistych i zespolonych
reprezentacji sygnału mowy w dedykowanych modelach głębokiego uczenia*
(Politechnika Krakowska, Wydział Informatyki i Telekomunikacji, 2026).

Praca porównuje cztery potoki klasyfikacji emocji z mowy, tworzące drabinę ablacyjną,
w której każdy kolejny krok zmienia jeden czynnik: sposób redukcji widma, obecność
informacji fazowej oraz typ arytmetyki.

## Badane modele

| model | reprezentacja wejściowa | faza | arytmetyka | parametry |
|---|---|---|---|---|
| M1 | MFCC + Δ + ΔΔ, `3 × 40 × 101` | pominięta | ℝ | 554 727 |
| M2 | \|STFT\|^0,3, `1 × 257 × 101` | pominięta | ℝ | 554 151 |
| M3 | Re/Im STFT^0,3, `2 × 257 × 101` | zachowana | ℝ | 554 439 |
| M4 | STFT ∈ ℂ, `1 × 257 × 101` | zachowana | ℂ | 543 819 |

Wszystkie modele mają tę samą topologię: cztery bloki splotowe, dwukierunkową
warstwę GRU, agregację z mechanizmem uwagi oraz moduł decyzyjny. Liczbę parametrów
zrównano z dokładnością do dwóch procent.

## Wyniki

UAR na poziomie nagrań, średnia z pięciu przebiegów różniących się ziarnem
inicjalizacji:

| model | UAR | macro-F1 | czas epoki | latencja |
|---|---|---|---|---|
| M1 | 0,6117 ± 0,0149 | 0,6043 | 4,94 s | 1,17 ms |
| M2 | **0,6295 ± 0,0095** | **0,6222** | 15,39 s | 1,14 ms |
| M3 | 0,6001 ± 0,0053 | 0,5924 | 15,74 s | 1,39 ms |
| M4 | 0,4715 ± 0,0229 | 0,4305 | 151,92 s | 54,72 ms |

Różnice pomiędzy kolejnymi ogniwami drabiny, wraz z przedziałami ufności
wyznaczonymi metodą bootstrapu blokowego po mówcach:

| krok | badany czynnik | ΔUAR | 95% CI |
|---|---|---|---|
| M1 → M2 | parametryzacja mel-cepstralna | +1,78 p.p. | [−0,71; +4,25] |
| M2 → M3 | udostępnienie informacji fazowej | −2,93 p.p. | [−4,48; −1,56] |
| M3 → M4 | arytmetyka zespolona | −12,86 p.p. | [−15,04; −10,79] |

Wyniki obowiązują dla zbioru CREMA-D przy zastosowanym podziale na mówców.

## Struktura repozytorium

```
├── config.py                         zbiorcza konfiguracja eksperymentu
├── notebooks/
│   ├── 01_preprocessing.ipynb        przygotowanie danych, podział, segmentacja
│   ├── 02_model_m1_mfcc.ipynb        model referencyjny
│   ├── 03_model_m2_magnitude.ipynb   model kontrolny, moduł STFT
│   ├── 04_model_m3_reim.ipynb        model kontrolny, część rzeczywista i urojona
│   ├── 05_model_m4_complex.ipynb     model zespolony
│   ├── 06_model_m4_stable.ipynb      model zespolony, konfiguracja stabilizowana
│   ├── 07_testy_jednostkowe.ipynb    weryfikacja warstw zespolonych
│   ├── 08_rysunki.ipynb              generowanie wykresów
│   └── 09_analiza_statystyczna.ipynb bootstrap blokowy po mówcach
├── src/
│   └── complex_layers.py             warstwy zespolone
├── results/
│   ├── m1_mfcc_5seeds.json
│   ├── m2_magnitude_5seeds.json
│   ├── m3_reim_5seeds.json
│   ├── m4_complex_5seeds.json
│   ├── m4_stable_5seeds.json
│   └── split_actors.json             przydział mówców do podzbiorów
├── figures/                          wykresy użyte w pracy
├── requirements.txt
└── README.md
```

Notatniki zawierają własne komórki konfiguracyjne, dzięki czemu każdy z nich
uruchamia się niezależnie w środowisku Kaggle Notebooks. Plik `config.py` gromadzi
te ustawienia w jednym miejscu i stanowi punkt odniesienia przy odtwarzaniu
eksperymentu. Warstwy zespolone, powtarzane w trzech notatnikach, wydzielono do
`src/complex_layers.py` wraz z opisem przyjętych rozwiązań.

## Dane

Wykorzystano zbiór **CREMA-D** (Cao i in., 2014), zawierający 7442 nagrania mowy
emocjonalnej wykonane przez 91 aktorów, obejmujące sześć klas emocjonalnych.
Zbiór nie jest częścią repozytorium ze względu na rozmiar. Można go pobrać
z repozytorium projektu: https://github.com/CheyneyComputerScience/CREMA-D

Wykorzystano etykiety odpowiadające emocji zamierzonej przez aktora, zakodowane
w nazwie pliku.

### Przetwarzanie wstępne

Sygnał sprowadzano do postaci monofonicznej o częstotliwości próbkowania 16 kHz,
normalizowano szczytowo do poziomu 0,95, usuwano ciszę z początku i końca nagrania
przy progu 30 dB poniżej wartości szczytowej, a następnie dzielono na segmenty
o długości jednej sekundy z przesunięciem 0,5 s. Wynikiem jest 30 568 segmentów
wyprowadzonych z 7441 nagrań; jedno nagranie odrzucono jako pozbawione treści
akustycznej po usunięciu ciszy.

Podziału dokonano na poziomie mówców, z ziarnem generatora liczb losowych równym 42:
63 aktorów w zbiorze treningowym oraz po 14 w walidacyjnym i testowym. Identyfikatory
aktorów przypisanych do poszczególnych podzbiorów zapisano w `results/split_actors.json`.

## Uruchomienie

### Środowisko

Eksperymenty przeprowadzono w środowisku Kaggle Notebooks na akceleratorze
NVIDIA Tesla T4, przy następujących wersjach oprogramowania:

- Python 3.12.13
- PyTorch 2.10.0 (CUDA 12.8)
- librosa 0.11.0

Notatniki wykrywają środowisko automatycznie i dostosowują ścieżki, dzięki czemu
mogą być uruchamiane zarówno na Kaggle, jak i lokalnie.

### Lokalnie

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Instrukcja instalacji PyTorch dla konkretnej konfiguracji sprzętowej znajduje się
pod adresem https://pytorch.org/get-started/locally

Następnie umieść zbiór CREMA-D w katalogu `data/raw/CREMA-D/AudioWAV` i uruchom
notatniki w kolejności numerów.

### Czas obliczeń

Wartości orientacyjne dla pojedynczego akceleratora T4, przy pięciu przebiegach
na konfigurację:

| etap | czas |
|---|---|
| przetwarzanie wstępne | ~2 min |
| model M1 | ~20 min |
| modele M2 i M3 | ~1 h każdy |
| model M4 | ~7 h |
| testy jednostkowe | ~20 s |

## Warstwy zespolone

Operacje zespolone nie są dostępne w standardowej bibliotece PyTorch i zostały
zaimplementowane samodzielnie w `src/complex_layers.py`:

| operacja | realizacja |
|---|---|
| splot | cztery sploty rzeczywiste zgodnie z regułą mnożenia zespolonego |
| normalizacja | wybielanie macierzy kowariancji 2×2, postać zamknięta pierwiastka |
| aktywacja | ℂReLU, niezależnie dla części rzeczywistej i urojonej |
| redukcja | selekcja według modułu, zwracana pełna wartość zespolona |
| jednostka rekurencyjna | bramki wyznaczane z modułu, stan ukryty w ℂ |
| inicjalizacja | moduł z rozkładu Rayleigha, faza z rozkładu jednostajnego |

Poprawność implementacji zweryfikowano w `notebooks/07_testy_jednostkowe.ipynb`.
Testy obejmują porównanie wyniku każdej warstwy z bezpośrednim obliczeniem na
częściach rzeczywistej i urojonej, weryfikację własności statystycznych
inicjalizacji oraz kontrolę gradientów metodą różnic skończonych w arytmetyce
podwójnej precyzji. Wszystkie 38 testów kończy się powodzeniem.

## Analiza statystyczna

Zbiór testowy obejmuje 1142 nagrania pochodzące od 14 mówców, wobec czego nagrania
tej samej osoby nie stanowią obserwacji niezależnych. Zastosowano bootstrap blokowy
na poziomie mówców: w każdej z 10 000 replikacji losowano ze zwracaniem
czternastoelementową próbę mówców, a miarę wyznaczano niezależnie dla wszystkich
przebiegów danego modelu. Porównaniu podlegała różnica wartości uśrednionych.

Przy czternastu klastrach precyzja wyznaczonych przedziałów pozostaje ograniczona,
a formułowane na ich podstawie wnioski wymagają ostrożności.

## Ograniczenia

- badania przeprowadzono na jednym zbiorze danych, obejmującym mowę aktorską
  w języku angielskim rejestrowaną w warunkach studyjnych,
- podziału na mówców dokonano jednokrotnie, wobec czego powtórzenia opisują
  zmienność procesu optymalizacji, nie zaś niepewność wynikającą z doboru mówców,
- pomiary czasu odnoszą się do konkretnej implementacji; warstwa rekurencyjna
  modelu zespolonego zrealizowana jest jako jawna pętla w języku Python, podczas
  gdy modele rzeczywiste wykorzystują zoptymalizowaną implementację cuDNN,
- perturbację fazy wprowadzano wyłącznie na etapie wnioskowania, co mierzy
  wrażliwość wytrenowanego modelu na zaburzenie wejścia, nie zaś niezbędność
  informacji fazowej dla zadania.

## Bibliografia

Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., Verma, R. (2014).
*CREMA-D: Crowd-sourced emotional multimodal actors dataset.*
IEEE Transactions on Affective Computing, 5(4), 377–390.

Trabelsi, C., Bilaniuk, O., Zhang, Y., Serdyuk, D., Subramanian, S., Santos, J. F.,
Mehri, S., Rostamzadeh, N., Bengio, Y., Pal, C. J. (2018). *Deep complex networks.*
International Conference on Learning Representations.

## Licencja

Kod udostępniony na licencji MIT. Zbiór CREMA-D podlega odrębnym warunkom
licencyjnym określonym przez jego autorów.
