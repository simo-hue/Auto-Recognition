import os
import shutil
import scipy.io

def carica_annotazioni(annotations_file):
    try:
        # Carica il file .mat con le annotazioni
        annotations = scipy.io.loadmat(annotations_file)
        print("Chiavi nel file .mat:", annotations.keys())  # Visualizza le chiavi per il debug

        # Supponiamo che le annotazioni siano nella chiave 'annotations'
        ann = annotations.get('annotations', None)
        
        if ann is None:
            raise ValueError(f"Le annotazioni non sono state trovate nel file {annotations_file}.")
        
        return ann

    except Exception as e:
        print(f"Errore durante il caricamento del file .mat: {e}")
        return None

def riorganizza_dataset(source_dir, target_dir, annotations):
    # Verifica se la cartella di origine esiste
    if not os.path.exists(source_dir):
        print(f"Errore: la cartella {source_dir} non esiste!")
        return

    # Estrai le informazioni dalle annotazioni
    for anno in annotations[0]:
        immagine_name = anno[0][0]  # Nome immagine
        classe_id = anno[5][0][0]  # ID della classe (classe numerica)

        # Debug per capire come viene mostrato il nome dell'immagine
        print(f"Nome immagine: {immagine_name}, ID classe: {classe_id}")

        # Verifica se c'è un prefisso da rimuovere dal nome dell'immagine
        if immagine_name.startswith("car_ims/"):
            immagine_name = immagine_name[len("car_ims/"):]

        # Crea il percorso completo per l'immagine nella cartella di origine
        immagine_path = os.path.join(source_dir, immagine_name)
        if os.path.isfile(immagine_path):
            # Crea la cartella di destinazione per la classe se non esiste
            target_class_path = os.path.join(target_dir, str(classe_id))
            if not os.path.exists(target_class_path):
                os.makedirs(target_class_path)
                print(f"Creata la cartella per la classe {classe_id} a {target_class_path}")

            # Sposta l'immagine nella cartella di destinazione
            target_image_path = os.path.join(target_class_path, immagine_name)

            try:
                shutil.move(immagine_path, target_image_path)
                print(f"Immagine {immagine_name} spostata in {target_image_path}")
            except Exception as e:
                print(f"Errore durante lo spostamento dell'immagine {immagine_name}: {e}")
        else:
            print(f"Immagine {immagine_name} non trovata nel percorso {immagine_path}.")

def main():
    # Percorso delle annotazioni
    train_annotations_file = 'stanford-cars-dataset/cars_annos.mat'
    
    # Carica le annotazioni
    annotations = carica_annotazioni(train_annotations_file)

    if annotations is None:
        print("Errore nel caricamento delle annotazioni. Terminando il programma.")
        return

    # Percorsi per il training
    train_source_dir = 'stanford-cars-dataset/train/cars_train'
    train_target_dir = 'organized_dataset/train'

    # Verifica se la cartella di destinazione esiste, se no la crea
    if not os.path.exists(train_target_dir):
        os.makedirs(train_target_dir)

    # Riorganizza il dataset per train
    riorganizza_dataset(train_source_dir, train_target_dir, annotations)

if __name__ == "__main__":
    main()