import os
import shutil
import scipy.io
import numpy as np

# Carica le annotazioni dal file .mat
annotations = scipy.io.loadmat('stanford-cars-dataset/cars_annos.mat')

# Estrai i nomi delle classi (marca e modello)
class_names = annotations['class_names'][0]

# Crea le cartelle per ogni classe
train_dir = './train/cars_train'
test_dir = './test/cars_test'

# Crea le sottocartelle per ogni classe in train e test (se non esistono)
for class_name in class_names:
    # Assicurati che class_name sia una stringa
    class_name_str = class_name[0] if isinstance(class_name, np.ndarray) else str(class_name)
    os.makedirs(os.path.join(train_dir, class_name_str), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name_str), exist_ok=True)

# Funzione per spostare le immagini nella cartella corretta
def move_images(images, class_name, target_dir):
    for img_name in images:
        # Assicurati che img_name sia una stringa
        img_name_str = img_name[0] if isinstance(img_name, np.ndarray) else str(img_name)
        if isinstance(img_name_str, np.ndarray):  # Gestire eventuali ndarray all'interno di img_name
            img_name_str = img_name_str.item()  # Estrai il valore scalare
        # Stampa per debugging (opzionale)
        print(f"Moving image: {img_name_str}")
        src = os.path.join(target_dir, img_name_str)
        dest = os.path.join(target_dir, str(class_name), img_name_str)  # Forza class_name come stringa
        shutil.move(src, dest)

# Sposta le immagini nelle cartelle di addestramento e test
for idx, class_name in enumerate(class_names):
    train_images = annotations['annotations'][0][idx][0][0]  # Immagini di addestramento
    test_images = annotations['annotations'][0][idx][1][0]  # Immagini di test

    move_images(train_images, class_name, train_dir)
    move_images(test_images, class_name, test_dir)

print("Riorganizzazione completata!")