import pandas as pd
import numpy as np


def generiraj_newton(primeri, sum=0):
    """
    Generira podatke za F = m a
    :param primeri:
    :return:
    """
    np.random.seed(1234)
    podatki = {"F": [], "m": [], "a": []}
    for i in range(primeri):
        m, a = np.random.rand(2)
        f = m * a * (1 + np.random.normal(0,sum))
        podatki["F"].append(f)
        podatki["m"].append(m)
        podatki["a"].append(a)
    return pd.DataFrame(podatki)

def generiraj_stefan(primeri, sum=0):
    """
    Generira podatke za j = sigma T^4
    :param primeri:
    :return:
    """
    np.random.seed(1234)
    podatki = {"j": [], "T": []}
    sigma = 5.67 * 10 ** -8
    for i in range(primeri):
        t = 100 * np.random.rand(1)[0] + 100
        j = sigma * t ** 4 * (1 + np.random.normal(0,sum))
        podatki["j"].append(j)
        podatki["T"].append(t)
    return pd.DataFrame(podatki)


def generiraj_lorenz(primeri, sum=0):
    """
    Generira podatke za gama = sqrt(1 - (v / c)^2)
    :param primeri:
    :return:
    """
    np.random.seed(1234)
    podatki = {"gama": [], "v": []}
    c = 3 * 10 ** 5  # [km / s]
    for i in range(primeri):
        v = np.random.rand(1)[0] * c
        gama = np.sqrt(1 - (v / c) ** 2) * (1 + np.random.normal(0,sum))
        podatki["gama"].append(gama)
        podatki["v"].append(v)
    return pd.DataFrame(podatki)


def generiraj_energijski_zakon_const(primeri, sum=0):
    """
    Generira podatke za m g h + 0.5 m v^2 = c.
    Predpostavili bomo, da imajo vsa telesa na zacetku isto enerigjo (isti c)
    :param primeri:
    :return:
    """
    np.random.seed(1234)
    podatki = {"m": [], "h": [], "v": []}
    c = 100  # [J]
    g = 9.81  # [m/s^2]
    for i in range(primeri):
        m, h = np.random.rand(2)
        h = h * 10  # med 0 in 10 metri --> 0 <= Wp = m g h < 1 * 10 * 10 = 100
        v = np.sqrt((c - m * g * h) * 2 / m) * (1 + np.random.normal(0,sum))
        podatki["m"].append(m)
        podatki["v"].append(v)
        podatki["h"].append(h)
    return pd.DataFrame(podatki)

def generiraj_energijski_zakon(primeri, sum=0):
    """
    Generira podatke za m g h + 0.5 m v^2 = c.
    Predpostavili bomo, da imajo vsa telesa na zacetku isto enerigjo (isti c)
    :param primeri:
    :return:
    """
    np.random.seed(1234)
    podatki = {"E": [], "m": [], "h": [], "v": []}
    g = 9.81  # [m/s^2]
    for i in range(primeri):
        m, h, v = np.random.rand(3)
        #h = h * 10  # med 0 in 10 metri 
        #v = v * 10 # med 0 in 10 metri na sekundo
        podatki["m"].append(m)
        podatki["v"].append(v)
        podatki["h"].append(h)
        podatki["E"].append((m*g*h + 0.5*m*v**2)*(1 + np.random.normal(0,sum)))
    return pd.DataFrame(podatki)


def generiraj_ploscina(primeri, sum=0):
    """
    Generira podatke za p = n a ** 2 / (4 tan (pi / n))
    :param primeri:
    :return:
    """
    np.random.seed(1234)
    podatki = {"n": [], "a": [], "p": []}
    for n in range(3, primeri + 3):
        a = np.random.rand(1)[0]
        p = n * a ** 2 / (4 * np.tan(np.pi / n)) * (1 + np.random.normal(0,sum))
        podatki["n"].append(n)
        podatki["a"].append(a)
        podatki["p"].append(p)
    return pd.DataFrame(podatki)




def bacon(datoteka, max_iter=20):
    df = pd.read_csv(datoteka)
    stolpci = list(df.columns)
    for i in range(max_iter):
        df_array = np.array(df)
        # ocenimo, ali obstaja konstanta
        sig = np.std(df_array, axis=0)
        # izracunajmo vrstne rede, najdemo korelacije med njimi
        vrstni_redi = df.rank(axis=0)
        korelacije = np.corrcoef(df_array, df_array)  # korelacije[i, j]
        # oglej si np.max(korelacije) in np.min(korelacije)

        # pamento poimenuj novi stolpec, da bomo lahko iz njega razbrali enacbo


