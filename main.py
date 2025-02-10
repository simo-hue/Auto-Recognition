import kagglehub

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download latest version
path = kagglehub.dataset_download("jessicali9530/stanford-cars-dataset")

print("Path to dataset files:", path)

# Parametri di configurazione
img_height = 224
img_width = 224
batch_size = 32

# Creazione dei generatori per il training e il test
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizza i pixel (portandoli nell'intervallo [0, 1])
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)  # Solo normalizzazione per i dati di test

# Creazione dei generatori
train_generator = train_datagen.flow_from_directory(
    'path_to_stanford_cars/train',  # Sostituisci con il percorso corretto del tuo dataset
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')  # Etichette categoriali per il multi-class classification

test_generator = test_datagen.flow_from_directory(
    'path_to_stanford_cars/test',  # Sostituisci con il percorso corretto del tuo dataset
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')