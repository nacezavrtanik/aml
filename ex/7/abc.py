"""Class 7, Exercises A, B, C: Machine Learning on Complex Data Structures, Part 1

This exercise class was based on a code template by Matej Petković, written in Slovene. My own task consisted of writing
the functions `razsiri_z_hours` and `razsiri_z_attributes` according to given specifications, as well as filling in the
missing code in the final for-loop of this module.
"""

import logging
import pandas as pd
import numpy as np
import itertools
from typing import List, Union
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def create_logger(name, stopnja=logging.INFO):
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(filename)s:%(funcName)s:%(lineno)d]:  %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.addHandler(ch)
    logger.setLevel(stopnja)
    logger.propagate = False
    return logger


def preberi_vse(tabele):
    """
    Naloži vse podatke. Ugibanje tipa stolpca (str/int/...) prepustimo pandam.

    :return: slovar {ime tabele: pripadajoč pd.DataFrame}
    """
    return {tabela: pd.read_csv(f"yelp_{tabela}.txt", sep="\t") for tabela in tabele}


def pretvori_vse_one_hot(df: pd.DataFrame, stolpci: List[str], drop: Union[str, List[str]] = "first"):
    """
    Pretvori vsak kategoričen (= nominalen) atribut v nekaj 0/1-atributov (0/1-kodiranje).
    Uporabi OneHotEncoder iz sklearn.

    :param df: pandas tabela podatkov
    :param stolpci: seznam imen vhodnih spremenljivk
    :param drop: taktika odmeta za enega od ustvarjenih 0/1-atributov (1 je odveč ...)
                 Po navadi 'first', lahko pa mu podamo tudi seznam [vrednost za odmet] dolžine 1.
    :return: pretvorjen df
    """
    bloki = []
    for stolpec in stolpci:
        x = np.array(df[stolpec]).reshape((-1, 1))
        tip = x.dtype.name
        if "float" not in tip and "int" not in tip:
            encoder = OneHotEncoder(drop=drop, sparse=False)
            x_v_numericni_obliki = encoder.fit_transform(x)
        else:
            x_v_numericni_obliki = x
        bloki.append(x_v_numericni_obliki)
    return np.block(bloki)


def preizkusi_podatke(xs: np.ndarray, y: np.ndarray):
    """
    Razdeli tabelo xs in vektor y na učni (75%) in testni (25%) del ter preveri, kako dobro se da napovedati
    vrednosti ciljne spremenljivke z naključnim gozdom.

    Za primerjavo naredi enako še z modelom, ki vedno napove večinski razred.

    Izpiše izmerjeni točnosti.

    :param xs: 2D tabela vrednosti vhodnih spremenljivk
    :param y: 1D tabela vrednosti ciljne spremenljivke
    :return: None
    """
    x_ucna, x_testna, y_ucna, y_testna = train_test_split(xs, y, test_size=0.25, random_state=1234, stratify=y)
    modeli = [
        ("Naključni gozd", RandomForestClassifier(random_state=1234)),
        ("Bučman", DummyClassifier())
    ]
    print("MODEL          | TOČNOST")
    print("-------------------------")
    for ime, model in modeli:
        model.fit(x_ucna, y_ucna)
        napovedi = model.predict(x_testna)
        print(f"{ime: <14} | {accuracy_score(y_testna, napovedi):.3f}")


########################################################################################################################
#
# Spodaj so definirani testni podatki, s katerimi bomo spoznali nekatere uporabne metode
# za spreminjaje pd.DataFrame.
#
########################################################################################################################
LOGGER = create_logger("")
testni_podatki1 = pd.DataFrame(
    data={
        "x": [10, 20, 30, 10],
        "y": [11, 12, "13", 14],
        "z": ["-1", "-2", "-3", "-4"],
        "w": ["a", "b", "4", "č"]
    }
)
testni_podatki2 = pd.DataFrame(
    index=["id1", "id2", "id3"],
    columns=["x", "y"],
    data=-1
)

########################################################################################################################
#
# Dostopanje do trenutnih vrednosti
#
########################################################################################################################

# Tako dostopamo do vrednosti v stolpcu v prvem primeru (index je kar 0, 1, ...):
stolpec1 = testni_podatki1["x"]
x1_i_ta = stolpec1[1]
# ali kar testni_podatki1["x"][1]
# ali kar testni_podatki1.at[1, "x"]

# Tako pa v drugem:
x2_i_ta = testni_podatki2["x"]["id2"]  # == testni_podatki2.at["id2", "x"]
LOGGER.info(f"testni_podatki1/2[x][i] prvič : {x1_i_ta} {x2_i_ta}")

# Lahko pa tudi tako:
x1_i_ta_prek_vrstice = testni_podatki1.iloc[1]["x"]
x2_i_ta_prek_vrstice = testni_podatki2.iloc[1]["x"]
LOGGER.info(f"testni_podatki1/2[x][i] drugič: {x1_i_ta_prek_vrstice} {x2_i_ta_prek_vrstice}")

# Tako iteriramo po vrsticah:
for testni_podatki in [testni_podatki1, testni_podatki2]:
    for i_vrstice, vrsta in testni_podatki.iterrows():
        if i_vrstice in [1, "id2"]:
            LOGGER.info(f"Celotna vrsta {i_vrstice}:\n{vrsta}")
            LOGGER.info(f"vrsta[x] = {vrsta['x']}")
        else:
            LOGGER.info(f"Preskočim vrsto {i_vrstice} ...")

########################################################################################################################
#
# Izračun zaloge vrednosti stolpca
#
########################################################################################################################

LOGGER.info(f"Zaloga vrednosti za testni podatki 1 (stolpec x): {testni_podatki1['x'].unique()}")
LOGGER.info(f"Zaloga vrednosti za testni podatki 2 (stolpec x): {testni_podatki2['x'].unique()}")


########################################################################################################################
#
# Spreminjanje podatkov: posamično
#
########################################################################################################################

testni_podatki1.at[1, "x"] = -20
LOGGER.info(f"Spremenjeni podatki:\n{testni_podatki1}")

########################################################################################################################
#
# Spreminjanje podatkov: masovno
#
# Bolj učinkovito (zaradi pandasovih optimizacij) je namesto for zanke in gornjega pristopa narediti takole:
#
########################################################################################################################

# nadomesti vrednosti v stolpcu x z njihovimi kvadrati
testni_podatki1["x"] = testni_podatki1["x"].apply(lambda t: t ** 2)
# nadomesti ...                 y            dvakratniki
testni_podatki1["y"] = testni_podatki1["y"].apply(lambda t: 2 * t)  # pozor: "13" --> "1313"
LOGGER.info(f"Spremenjeni podatki:\n{testni_podatki1}")

########################################################################################################################
#
# Spreminjanje podatkov: str --> numerično
#
# Imeti 22, 24 in "1313" je tečno ali nevarno (namesto 26 smo dobili "1313") Raje bi imeli 22, 24, 1313.
# Prav tako namesto "-1", "-2" ... raje -1, -2 ...
#
# Poskusimo v numerično obliko pretvoriti vse stolpce, ki jih lahko:
########################################################################################################################

for stolpec in testni_podatki1.columns:
    # argument errors je prednastavljen na raise (sproži se izjema, če pretvorba ne uspe)
    numericne_vrednosti = pd.to_numeric(testni_podatki1[stolpec], errors="coerce")
    LOGGER.info(f"Numerične pretvorbe za {stolpec}: {list(numericne_vrednosti)}")
    # nadomestimo stare z novimi samo, če smo uspeli pretvoriti vse
    if all(numericne_vrednosti.notnull()):
        LOGGER.info(f"    Posodabljam {stolpec} v numerično obliko.")
        testni_podatki1[stolpec] = numericne_vrednosti
    else:
        LOGGER.info(f"    Stolpec {stolpec} puščam nedotaknjen.")


########################################################################################################################
#
# numpy in pandas se dobro razumeta
#
########################################################################################################################

numericni_podatki_kot_array1 = np.array(testni_podatki1[["x", "y", "z"]])
numericni_podatki_kot_array2 = np.array(testni_podatki2)

LOGGER.info(f"np.array za test1:\n{numericni_podatki_kot_array1}")
LOGGER.info(f"np.array za test2:\n{numericni_podatki_kot_array2}")

aTa1 = numericni_podatki_kot_array1.T @ numericni_podatki_kot_array1

LOGGER.info(f"array1^T array1:\n{aTa1}")

########################################################################################################################
#
# Ker bomo lepili različne bloke, pokažimo še funkcijo np.block
#
########################################################################################################################

levi = np.array([[1, 2, 3], [-1, -2, -3]])
desni = np.array([[4, 5, 6, 7], [-4, -5, -6, -7]])
oboji = np.block([levi, desni])

LOGGER.info(f"Zlepek levih in desnih:\n{oboji}")

########################################################################################################################
# Iz tabele business smo izluščili stolpce, ki ustrezajo (smiselnim) vhodnim spremenljivkam in ciljni spremenljivki.
########################################################################################################################

TABELE = [
    "attributes",
    "business",
    "hours",
    "review",
    "users"
]
PODATKI = preberi_vse(TABELE)

business = PODATKI["business"]
stolpci = list(business.columns)
print("Zaloga vrednosti ciljne spremenljivke:", list(business[stolpci[-1]].unique()))
stolpci_x = stolpci[3:-1]  # odstrani id, ime in y
xs_business = pretvori_vse_one_hot(business, stolpci_x)
y_business = np.array(business[stolpci[-1]])

preizkusi_podatke(xs_business, y_business)


# 1 Complete `razsiri_z_hours`
########################################################################################################################
# Zapišite funkcijo razsiri_z_hours(business_ids), ki sprejme zaporedje id-jev biznisov in izvede naslednje korake:
#  1) izlušči iz stolpca "day" v tabeli PODATKI["hours"] vse vrednosti tega stolpca (pričakujemo jih 7)
#  2) naredi seznam novih 14 stolpcev z imeni oblike dan_open in dan_close (npr. "Monday_open"), ki bodo povedali,
#     kdaj se na dani dan poslovni obrat odpre/zapre
#  3) spremeni vrednosti stolpcev "open time" in "close time" v tabeli PODATKI["hours"] v številske
#     (npr. "11:00:00" --> 11, "21:30:00" -> 21 oz. 21.5 oz. 22)
#  4) pripravi pd.DataFrame z zgoraj omenjenimi 14 stolpci, indeksom business_ids in primerno prednastavljeno
#     vrednostjo povsod tabeli (ta vrednost bo uporabljena, če prava vrednost ni dostopna)
#  5) napolni novo tabelo z dejanskimi vrednostmi iz tabele PODATKI["hours"]
#  6) pretvori novo tabelo v (numeričen) np.ndarray (velikosti len(business_ids) x 14) in ga vrne#
########################################################################################################################


def razsiri_z_hours(business_ids):

    hours = PODATKI.get('hours')
    actions = ['open', 'close']

    # 1) Extract unique day values
    days = pd.unique(hours['day'])

    # 2) Create list of column names of form 'day_open', 'day_close'
    new_columns = [f'{day}_{action}' for day in days for action in actions]

    # 3) Convert values in columns 'open time' and 'close time' to numeric values
    def time_string_to_float(x):
        h, m, _ = x.split(':')
        h = int(h)
        m = int(m) / 60
        return h + m
    hours[['open time', 'close time']] = hours[['open time', 'close time']].applymap(time_string_to_float)

    # 4) Create dataframe with dummy values, with new columns, and indexed by `business_ids`
    new_df = pd.DataFrame(index=business_ids, columns=new_columns, data=np.nan)

    # 5) Fill dataframe with actual values
    for _, row in hours.iterrows():
        business_id = row['business_id']
        day = row['day']
        for action in actions:
            new_df.loc[business_id, f'{day}_{action}'] = row[f'{action} time']

    # 6) Convert pandas.DataFrame to numpy.ndarray
    new_df = np.array(new_df)

    return new_df


# 2 Complete `razsiri_z_attributes`
########################################################################################################################
# Zapišite funkcijo razsiri_z_attributes(business_ids), ki sprejme zaporedje id-jev biznisov in izvede naslednje korake:
#  1) izlušči iz stolpca "name" v tabeli PODATKI["attributes"] vse vrednosti tega stolpca
#  2) naredi seznam novih stolpcev z imeni, ki so vrednosti stolpca "name"
#  3) naredi novo tabelo z indeksom business_ids in stolpci iz prejšnje točke; prednastavljena vrednost za
#     tabelo naj bo ustrezno izbrana in naj upošteva, da so nekateri stolpci kategorični (npr. Attire ("casual", ...)
#     in garage ("true"/"false")), nekateri pa numerični (npr. Price Range (1, 2, ...)).
#  4) napolni novo tabelo z dejanskimi vrednostmi iz tabele PODATKI["attributes"]
#  5) pretvori stolpce, ki pripadajo numeričnim atributov, v numerične
#  6) s pomočjo funkcije pretvori_vse_one_hot pretvori novo tabelo v numeričen np.ndarray in ga vrne.
#     OneHotEncoder naj odvrže stolpec, ki pripada prednastavljeni vrednosti - to dosežemo s klicem
#     pretvori_vse_one_hot(..., drop=[prednastavljena]).
########################################################################################################################


def razsiri_z_attributes(business_ids):

    attributes = PODATKI.get('attributes')

    # 1) Extract unique name values
    names = pd.unique(attributes['name'])

    # 2) Create list of column names from `names`
    new_columns = list(names)

    # 3) Create dataframe with dummy values, with new columns, and indexed by `business_ids`
    new_df = pd.DataFrame(index=business_ids, columns=new_columns, data=np.nan)

    # 4) Fill dataframe with actual values
    for _, row in attributes.iterrows():
        business_id = row['business_id']
        name = row['name']
        value = row['value']
        new_df.loc[business_id, name] = value

    # 5) Transform columns with numeric attributes into numeric columns
    for column in new_df.columns:
        new_df[column] = pd.to_numeric(new_df[column], errors='ignore')

    # 6) Convert pandas.DataFrame to numpy.ndarray
    new_df = pretvori_vse_one_hot(new_df, new_df.columns)

    return new_df


########################################################################################################################
#  Preizkusimo kodo tako, da pogledamo, kako se obnesejo modeli pri vseh možnih kombinacijah blokov
#  vhodnih spremenljivk:
#  - imamo tri bloke: ["business", "hours", "attributes"]
#  - za vsako neprazen podseznam se naučimo model in si ogledamo njegovo točnost
#
#  V pomoč naj bo spodnja funkcija in definirane spremenljivke.
#
########################################################################################################################
def generiraj_vse_neprazne_podsezname(n):
    """
    Generira vse neprazne podsezname seznama z n elementi. Namesto dejanskih elementov le indekse.
    :param n:
    :return:
    """
    podseznami = []
    for velikost in range(1, n + 1):
        for moznost in itertools.combinations(range(n), velikost):
            podseznami.append(moznost)
    return podseznami


biznisi = list(business["business_id"])
xs_hours = razsiri_z_hours(biznisi)
xs_attributes = razsiri_z_attributes(biznisi)
vsi_bloki = [xs_business, xs_hours, xs_attributes]
imena_blokov = ["business", "hours", "attributes"]
n_blokov = len(imena_blokov)
for podseznam in generiraj_vse_neprazne_podsezname(n_blokov):
    # TODO
    print(f"Rezultati za bloke nekako naračunane iz {podseznam}:")
