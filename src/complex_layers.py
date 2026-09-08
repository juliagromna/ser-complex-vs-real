"""
Warstwy operujące na liczbach zespolonych.

Moduł zawiera implementacje operacji niedostępnych w standardowej bibliotece
PyTorch: splotu zespolonego, normalizacji wsadowej z wybielaniem macierzy
kowariancji, aktywacji CReLU, warstwy redukującej z selekcją według modułu oraz
zespolonej jednostki rekurencyjnej GRU.

Tensory zespolone przechowywane są jako typ complex64. Automatyczne
różniczkowanie realizowane jest przez mechanizm biblioteki PyTorch, zgodny
z rachunkiem Wirtingera, wobec czego gradienty nie wymagają odrębnej
implementacji.

Poprawność wszystkich warstw weryfikowana jest w notatniku
notebooks/07_testy_jednostkowe.ipynb.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def crelu(z: torch.Tensor) -> torch.Tensor:
    """Aktywacja CReLU: funkcja ReLU aplikowana niezależnie do obu składowych.

    Progowanie składowych ortogonalnych modyfikuje kąt fazowy wyniku, co stanowi
    właściwość tej funkcji, a nie efekt uboczny implementacji.
    """
    return torch.complex(F.relu(z.real), F.relu(z.imag))


class ComplexConv2d(nn.Module):
    """Zespolony splot dwuwymiarowy.

    Wyjście wyznaczane jest zgodnie z regułą mnożenia liczb zespolonych:

        (W_R + jW_I) * (X_R + jX_I)
            = (W_R * X_R - W_I * X_I) + j(W_I * X_R + W_R * X_I)

    co wymaga czterech splotów rzeczywistych. Część rzeczywista i urojona jądra
    przechowywane są w dwóch osobnych warstwach nn.Conv2d.

    Wagi inicjalizowane są w reprezentacji biegunowej: moduł z rozkładu
    Rayleigha o parametrze skali sigma = 1 / sqrt(2 * fan_in), kąt fazowy
    z rozkładu jednostajnego na przedziale (-pi, pi]. Odpowiada to warunkowi
    E[|W|^2] = 1 / fan_in.
    """

    def __init__(self, cin: int, cout: int, k: int = 3, padding: int = 1):
        super().__init__()
        self.cr = nn.Conv2d(cin, cout, k, padding=padding, bias=False)
        self.ci = nn.Conv2d(cin, cout, k, padding=padding, bias=False)
        self._init_complex(cin * k * k)

    def _init_complex(self, fan_in: int) -> None:
        sigma = 1.0 / math.sqrt(2 * fan_in)
        with torch.no_grad():
            mag = torch.from_numpy(
                np.random.rayleigh(sigma, size=tuple(self.cr.weight.shape))
            ).float()
            phase = torch.empty_like(self.cr.weight).uniform_(-math.pi, math.pi)
            self.cr.weight.copy_(mag * torch.cos(phase))
            self.ci.weight.copy_(mag * torch.sin(phase))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zr, zi = z.real, z.imag
        return torch.complex(self.cr(zr) - self.ci(zi),
                             self.cr(zi) + self.ci(zr))


class ComplexLinear(nn.Module):
    """Zespolona warstwa w pełni połączona.

    Realizuje transformację afiniczną w dziedzinie liczb zespolonych, analogicznie
    do ComplexConv2d. Inicjalizacja wag przebiega w reprezentacji biegunowej.
    """

    def __init__(self, cin, cout, bias=True):
            super().__init__()
            self.lr = nn.Linear(cin, cout, bias=False)
            self.li = nn.Linear(cin, cout, bias=False)
            self.use_bias = bias
            if bias:
                self.b_r = nn.Parameter(torch.zeros(cout))
                self.b_i = nn.Parameter(torch.zeros(cout))
            sigma = 1.0 / math.sqrt(2 * cin)
            with torch.no_grad():
                mag = torch.from_numpy(
                    np.random.rayleigh(sigma, size=tuple(self.lr.weight.shape))).float()
                phase = torch.empty_like(self.lr.weight).uniform_(-math.pi, math.pi)
                self.lr.weight.copy_(mag * torch.cos(phase))
                self.li.weight.copy_(mag * torch.sin(phase))
    
    def forward(self, z):
        zr, zi = z.real, z.imag
        out_r = self.lr(zr) - self.li(zi)
        out_i = self.lr(zi) + self.li(zr)
        if self.use_bias:
            out_r = out_r + self.b_r
            out_i = out_i + self.b_i
        return torch.complex(out_r, out_i)


class ComplexBatchNorm2d(nn.Module):
    """Zespolona normalizacja wsadowa z wybielaniem macierzy kowariancji.

    Niezależna standaryzacja części rzeczywistej i urojonej prowadziłaby do
    niezależnego skalowania obu osi, a w konsekwencji do zniekształcenia rozkładu
    modułu i kątów fazowych. Warstwa estymuje zatem pełną macierz kowariancji
    2x2 pomiędzy składowymi i mnoży wycentrowane dane przez jej symetryczny
    odwrotny pierwiastek, wyznaczany w postaci zamkniętej:

        V^(-1/2) = 1/(s*t) * [[V_ii + s, -V_ri], [-V_ri, V_rr + s]]
        s = sqrt(det V),  t = sqrt(tr V + 2s)

    Rozkłady dające czynnik trójkątny, takie jak rozkład Choleskiego, również
    wybielają dane, wprowadzają jednak dodatkowy obrót płaszczyzny zespolonej.

    Parametry uczone: symetryczna macierz skalująca gamma (trzy wartości na kanał)
    oraz wektor przesunięcia beta (dwie wartości na kanał).

    Źródło: Trabelsi i in., Deep Complex Networks, ICLR 2018.
    """

    def __init__(self, c: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum

        self.g_rr = nn.Parameter(torch.full((c,), 1 / math.sqrt(2)))
        self.g_ii = nn.Parameter(torch.full((c,), 1 / math.sqrt(2)))
        self.g_ri = nn.Parameter(torch.zeros(c))
        self.b_r = nn.Parameter(torch.zeros(c))
        self.b_i = nn.Parameter(torch.zeros(c))

        for name, value in [("rm_r", torch.zeros(c)),
                            ("rm_i", torch.zeros(c)),
                            ("rv_rr", torch.full((c,), 1 / math.sqrt(2))),
                            ("rv_ii", torch.full((c,), 1 / math.sqrt(2))),
                            ("rv_ri", torch.zeros(c))]:
            self.register_buffer(name, value)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        zr, zi = z.real, z.imag
        dims = (0, 2, 3)

        if self.training:
            mr, mi = zr.mean(dims), zi.mean(dims)
            with torch.no_grad():
                self.rm_r.mul_(1 - self.momentum).add_(self.momentum * mr)
                self.rm_i.mul_(1 - self.momentum).add_(self.momentum * mi)
        else:
            mr, mi = self.rm_r, self.rm_i

        v = lambda t: t.view(1, -1, 1, 1)
        xr, xi = zr - v(mr), zi - v(mi)

        if self.training:
            vrr = (xr * xr).mean(dims) + self.eps
            vii = (xi * xi).mean(dims) + self.eps
            vri = (xr * xi).mean(dims)
            with torch.no_grad():
                self.rv_rr.mul_(1 - self.momentum).add_(self.momentum * vrr)
                self.rv_ii.mul_(1 - self.momentum).add_(self.momentum * vii)
                self.rv_ri.mul_(1 - self.momentum).add_(self.momentum * vri)
        else:
            vrr, vii, vri = self.rv_rr, self.rv_ii, self.rv_ri

        det = (vrr * vii - vri * vri).clamp_min(self.eps)
        s = torch.sqrt(det)
        t = torch.sqrt(vrr + vii + 2 * s).clamp_min(self.eps)
        inv = 1.0 / (s * t)
        wrr, wii, wri = (vii + s) * inv, (vrr + s) * inv, -vri * inv

        nr = v(wrr) * xr + v(wri) * xi
        ni = v(wri) * xr + v(wii) * xi

        outr = v(self.g_rr) * nr + v(self.g_ri) * ni + v(self.b_r)
        outi = v(self.g_ri) * nr + v(self.g_ii) * ni + v(self.b_i)
        return torch.complex(outr, outi)


class ComplexConvBlock(nn.Module):
    """Blok splotowy: splot zespolony, normalizacja, aktywacja i redukcja.

    Redukcja rozmiaru przestrzennego wymaga rozdzielenia kryterium selekcji od
    zwracanej wartości, ponieważ zbiór liczb zespolonych nie posiada
    uporządkowania zgodnego z działaniami ciała. Wyboru dokonuje się na podstawie
    modułu, natomiast na wyjście przekazywana jest pełna wartość zespolona
    wskazanego elementu wraz z jego kątem fazowym.
    """

    def __init__(self, cin: int, cout: int, pool: tuple):
        super().__init__()
        self.conv = ComplexConv2d(cin, cout)
        self.bn = ComplexBatchNorm2d(cout)
        self.pool = pool

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = crelu(self.bn(self.conv(z)))
        pf, pt = self.pool

        _, idx = F.max_pool2d(z.abs(), (pf, pt), return_indices=True)
        B, C, _, _ = z.shape
        flat_r = z.real.reshape(B, C, -1).gather(2, idx.reshape(B, C, -1))
        flat_i = z.imag.reshape(B, C, -1).gather(2, idx.reshape(B, C, -1))
        return torch.complex(flat_r.reshape(idx.shape), flat_i.reshape(idx.shape))


class ComplexGRUCell(nn.Module):
    """Komórka zespolonej jednostki GRU.

    Bezpośrednie zastosowanie zespolonej funkcji logistycznej nie jest możliwe:
    zgodnie z twierdzeniem Liouville'a funkcja całkowita i jednocześnie
    ograniczona musi być stała. Wartości bramek wyznaczane są zatem przez
    zastosowanie rzeczywistej funkcji logistycznej do modułu odpowiedniej
    transformacji afinicznej:

        z_t = sigma(|W_z x_t + U_z h_{t-1}|)
        r_t = sigma(|W_r x_t + U_r h_{t-1}|)
        n_t = Ctanh(W_h x_t + U_h (r_t * h_{t-1}))
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}

    Rzeczywisty współczynnik z przedziału (0,1) skaluje daną składową stanu bez
    wprowadzania dodatkowego obrotu fazy.

    Rozwiązanie stanowi adaptację jednostki GRU do dziedziny zespolonej,
    wykorzystującą zasadę rzutowania bramek zastosowaną przez Woltera i Yao
    (Complex gated recurrent neural networks, NeurIPS 2018) dla wariantu
    opartego na innej postaci funkcji bramkującej.
    """

    def __init__(self, cin: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.x2h = ComplexLinear(cin, 3 * hidden)
        self.h2h = ComplexLinear(hidden, 3 * hidden, bias=False)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        gx, gh = self.x2h(x), self.h2h(h)
        H = self.hidden

        z = torch.sigmoid((gx[:, :H] + gh[:, :H]).abs())            # aktualizacja
        r = torch.sigmoid((gx[:, H:2 * H] + gh[:, H:2 * H]).abs())  # reset

        n = gx[:, 2 * H:] + r.to(gh.dtype) * gh[:, 2 * H:]
        n = torch.complex(torch.tanh(n.real), torch.tanh(n.imag))

        zc = z.to(h.dtype)
        return (1 - zc) * n + zc * h


class ComplexBiGRU(nn.Module):
    """Dwukierunkowa zespolona warstwa rekurencyjna.

    Sekwencja przetwarzana jest dwukrotnie, w kierunku zgodnym i przeciwnym do
    upływu czasu, a wynikowe stany łączone są przez konkatenację, wskutek czego
    wymiar wyjścia ulega podwojeniu.

    W odróżnieniu od warstwy nn.GRU, wykorzystującej zoptymalizowaną
    implementację cuDNN, kroki czasowe realizowane są jawną pętlą, co ma istotne
    konsekwencje dla czasu wykonania.
    """

    def __init__(self, cin: int, hidden: int):
        super().__init__()
        self.fwd = ComplexGRUCell(cin, hidden)
        self.bwd = ComplexGRUCell(cin, hidden)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        h = torch.zeros(B, self.hidden, dtype=x.dtype, device=x.device)
        out_f = []
        for t in range(T):
            h = self.fwd(x[:, t], h)
            out_f.append(h)

        h = torch.zeros(B, self.hidden, dtype=x.dtype, device=x.device)
        out_b = []
        for t in reversed(range(T)):
            h = self.bwd(x[:, t], h)
            out_b.append(h)
        out_b.reverse()

        return torch.cat([torch.stack(out_f, 1), torch.stack(out_b, 1)], dim=-1)


def power_compression(z: torch.Tensor, p: float = 0.3,
                      eps: float = 1e-8) -> torch.Tensor:
    """Kompresja potęgowa modułu przy zachowaniu kąta fazowego.

        T_p(z) = |z|^p * exp(j*theta) = |z|^(p-1) * z,   p w (0,1)

    Logarytm liczby zespolonej jest funkcją wieloznaczną i nieokreśloną w zerze,
    wobec czego skala logarytmiczna stosowana dla reprezentacji rzeczywistych nie
    ma bezpośredniego odpowiednika w dziedzinie zespolonej. Kompresja potęgowa
    ogranicza rozpiętość dynamiczną w sposób zbliżony, pozostając funkcją ciągłą.

    Wartość zerowa obsługiwana jest przez ograniczenie modułu od dołu stałą eps.
    """
    mag = z.abs().clamp_min(eps)
    return z * (mag ** (p - 1.0))
