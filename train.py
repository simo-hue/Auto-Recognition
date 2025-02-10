import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Parametri di configurazione
img_height = 224  # Altezza delle immagini di input (per ResNet50)
img_width = 224   # Larghezza delle immagini di input (per ResNet50)
batch_size = 32   # Numero di immagini da processare per ogni batch

# Percorso al dataset scaricato (assumendo che il dataset sia nella stessa cartella di train.py)
dataset_dir = './'  # La cartella corrente in cui si trova train.py

# Creazione dei generatori di dati per il preprocessing delle immagini
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizzazione dei pixel tra 0 e 1
    rotation_range=20,  # Rotazione casuale dell'immagine fino a 20 gradi
    width_shift_range=0.2,  # Spostamento orizzontale casuale delle immagini
    height_shift_range=0.2,  # Spostamento verticale casuale delle immagini
    shear_range=0.2,  # Trasformazioni di taglio casuali
    zoom_range=0.2,  # Zoom casuale dell'immagine
    horizontal_flip=True)  # Abilita il flip orizzontale per aumentare la variabilità dei dati

# Preprocessing per i dati di test (solo normalizzazione)
test_datagen = ImageDataGenerator(rescale=1./255)

# Creazione dei generatori per caricare le immagini di addestramento e di test
train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'train/cars_train'),  # Cartella delle immagini di addestramento
    target_size=(img_height, img_width),  # Ridimensionamento delle immagini
    batch_size=batch_size,  # Numero di immagini per batch
    class_mode='categorical')  # Etichette in formato categoriale (multi-classe)

test_generator = test_datagen.flow_from_directory(
    os.path.join(dataset_dir, 'test/cars_test'),  # Cartella delle immagini di test
    target_size=(img_height, img_width),  # Ridimensionamento delle immagini
    batch_size=batch_size,  # Numero di immagini per batch
    class_mode='categorical')  # Etichette in formato categoriale (multi-classe)

# Creazione del modello con ResNet50 pre-addestrato come base
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Congela i pesi del modello pre-addestrato per evitare che vengano modificati durante l'allenamento

# Creazione del modello finale con un classificatore aggiuntivo sopra ResNet50
model = Sequential([
    base_model,  # Aggiunge la rete pre-addestrata come base
    GlobalAveragePooling2D(),  # Pooing globale per ridurre la dimensione dell'output
    Dense(1024, activation='relu'),  # Strato completamente connesso con 1024 neuroni
    Dense(train_generator.num_classes, activation='softmax')  # Strato di output per classificazione multi-classe
])

# Compilazione del modello
model.compile(optimizer=Adam(learning_rate=0.0001),  # Ottimizzatore Adam con un learning rate basso
              loss='categorical_crossentropy',  # Funzione di perdita per problemi multi-classe
              metrics=['accuracy'])  # Metri per valutare la performance (accuratezza)

# Allenamento del modello
history = model.fit(
    train_generator,  # Dati di allenamento
    epochs=10,  # Numero di epoche per allenare il modello
    validation_data=test_generator)  # Dati di test per validazione

# Salvataggio del modello allenato
model.save('car_recognition_model.h5')  # Salva il modello in un file H5

# Stampa messaggio di completamento
print("Modello allenato e salvato come car_recognition_model.h5")