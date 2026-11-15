"""Preprocesare comuna pentru CNN si U-Net."""

from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

DIMENSIUNE = (256, 256)
FORMA_PERECHE = (256, 256, 1)
URL_DATE = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset"
)


def pregateste_fisiere(tabel, folder_date):
    cai = pd.concat([tabel["imagine"], tabel["masca"]])
    if all((folder_date / cale).is_file() for cale in cai):
        return

    folder_date.parent.mkdir(parents=True, exist_ok=True)
    arhiva = folder_date.with_suffix(".zip")
    if not arhiva.is_file():
        print("Descarc arhiva Kaggle...")
        temporar = arhiva.with_suffix(".part")
        urlretrieve(URL_DATE, temporar)
        temporar.replace(arhiva)

    print("Extrag imaginile si mastile...")
    with ZipFile(arhiva) as zip_date:
        zip_date.extractall(folder_date)

    lipsa = [cale for cale in cai if not (folder_date / cale).is_file()]
    if lipsa:
        raise FileNotFoundError(f"{len(lipsa)} fisiere lipsa. Primul: {lipsa[0]}")


def incarca_pereche(cale_imagine, cale_masca):
    with Image.open(cale_imagine) as fisier_ct:
        ct = fisier_ct.convert("L").resize(DIMENSIUNE, Image.Resampling.BILINEAR)
        imagine = np.asarray(ct, dtype=np.float32) / 255.0

    with Image.open(cale_masca) as fisier_masca:
        masca = fisier_masca.convert("L").resize(DIMENSIUNE, Image.Resampling.NEAREST)
        etichete = (np.asarray(masca) > 0).astype(np.float32)

    return imagine[..., np.newaxis], etichete[..., np.newaxis]


def incarca_pentru_tensorflow(cale_ct, cale_masca):
    def citeste(ct, masca):
        return incarca_pereche(ct.decode("utf-8"), masca.decode("utf-8"))

    ct, masca = tf.numpy_function(
        citeste, [cale_ct, cale_masca], [tf.float32, tf.float32]
    )
    ct.set_shape(FORMA_PERECHE)
    masca.set_shape(FORMA_PERECHE)
    return ct, masca


def augmentare_aleatorie(ct, masca):
    # Imaginile si mastile primesc aceeasi transformare.
    pereche = tf.image.random_flip_left_right(tf.concat([ct, masca], axis=-1))
    return pereche[..., :1], pereche[..., 1:]


def creeaza_set(randuri, folder_date, marime_lot, seed, antrenare):
    imagini = [str(folder_date / cale) for cale in randuri["imagine"]]
    masti = [str(folder_date / cale) for cale in randuri["masca"]]
    set_date = tf.data.Dataset.from_tensor_slices((imagini, masti))

    if antrenare:
        set_date = set_date.shuffle(len(randuri), seed=seed)

    set_date = set_date.map(incarca_pentru_tensorflow)
    if antrenare:
        set_date = set_date.map(augmentare_aleatorie)

    return set_date.batch(marime_lot).prefetch(tf.data.AUTOTUNE)


def pregateste_seturi(cale_csv, folder_date, marime_lot=8, seed=42):
    """Returneaza seturile de antrenare, validare si test din CSV."""
    tabel = pd.read_csv(cale_csv)
    folder_date = Path(folder_date)
    pregateste_fisiere(tabel, folder_date)

    seturi = {}
    for nume in ("Antrenare", "Validare", "Test"):
        randuri = tabel[tabel["partitie"] == nume]
        if randuri.empty:
            raise ValueError(f"Partitia {nume} nu contine perechi.")
        seturi[nume] = creeaza_set(
            randuri, folder_date, marime_lot, seed, antrenare=nume == "Antrenare"
        )
    return seturi
