# Auto-Recognition: Riconoscimento di Auto Basato sul Dataset Stanford Cars

Questo progetto si concentra sulla creazione di un sistema di riconoscimento delle auto che può identificare il marchio e il modello di un'auto data un'immagine. Il progetto utilizza il dataset Stanford Cars per addestrare un modello di rete neurale basato su **ResNet-50** (un tipo di rete neurale convoluzionale pre-addestrata) per l'analisi delle immagini.

## Obiettivo del Progetto

L'obiettivo è creare un modello di machine learning che:
1. Identifichi correttamente il marchio e il modello di un'auto in un'immagine.
2. Utilizzi il dataset Stanford Cars, che contiene immagini di 196 classi di auto, per addestrare il modello.
3. Consenta di applicare il modello per inferenze su nuove immagini, restituendo il nome della marca e del modello dell'auto.

## Struttura del Progetto

La struttura del progetto è la seguente:
Auto-Recognition/
│
├── cars_annos.mat               # File delle annotazioni del dataset Stanford Cars
├── train/                       # Cartella contenente le immagini di addestramento organizzate per classe
│   └── cars_train/
│
├── test/                        # Cartella contenente le immagini di test organizzate per classe
│   └── cars_test/
│
├── models/                      # Cartella contenente i modelli addestrati
│
├── data_preprocessing.py        # Script per pre-elaborare e riorganizzare i dati
├── model_training.py            # Script per addestrare il modello
├── predict.py                   # Script per fare previsioni sulle nuove immagini
└── requirements.txt             # File contenente le dipendenze Python

## Dataset

Il dataset **Stanford Cars** è un ampio set di dati che contiene immagini di 196 classi di auto, con le etichette corrispondenti ai modelli e ai marchi. Il dataset è disponibile su Kaggle e include due sottoinsiemi:
- **Train set**: contiene le immagini per l'addestramento del modello.
- **Test set**: contiene le immagini per testare il modello.

### Come ottenere il dataset:
1. Vai su [Stanford Cars Dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) su Kaggle.
2. Scarica il file ZIP contenente il dataset.
3. Estrai il contenuto del file ZIP nella stessa directory del progetto.

## Installazione

### Requisiti

Prima di eseguire il progetto, assicurati di avere installato Python 3.6+ e di aver configurato un ambiente virtuale. Puoi installare le dipendenze utilizzando il file `requirements.txt`.

1. Crea un ambiente virtuale (opzionale, ma consigliato):

    ```bash
    python3 -m venv venv
    ```

2. Attiva l'ambiente virtuale:

    - Su macOS/Linux:
    
      ```bash
      source venv/bin/activate
      ```

    - Su Windows:
    
      ```bash
      venv\Scripts\activate
      ```

3. Installa le dipendenze:

    ```bash
    pip install -r requirements.txt
    ```

## Come eseguire il progetto

### Fase 1: Pre-elaborazione dei dati

1. Esegui il file `data_preprocessing.py` per riorganizzare le immagini del dataset in cartelle separate per ciascun marchio e modello di auto.

    ```bash
    python data_preprocessing.py
    ```

   Questo script esegue:
   - Carica il file `cars_annos.mat` per ottenere le annotazioni delle immagini.
   - Riorganizza le immagini in cartelle basate sul marchio e sul modello dell'auto.
   
2. Dopo aver eseguito questo script, troverai le cartelle riorganizzate in `train/cars_train` e `test/cars_test`.

### Fase 2: Addestramento del modello

Una volta che i dati sono stati pre-elaborati, puoi addestrare il modello.

1. Esegui il file `model_training.py` per addestrare il modello di riconoscimento delle auto utilizzando ResNet-50 come base.

    ```bash
    python model_training.py
    ```

2. Durante l'addestramento, verrà utilizzato il modello pre-addestrato ResNet-50 senza la parte finale (head), a cui verrà aggiunta una nuova testa per la classificazione delle auto. Questo approccio permette di sfruttare le capacità del modello pre-addestrato per migliorare le performance con un numero minore di epoche.

3. Una volta completato l'addestramento, il modello verrà salvato nella cartella `models/`.

### Fase 3: Previsioni

1. Dopo aver addestrato il modello, puoi utilizzare il file `predict.py` per fare previsioni su nuove immagini.

    ```bash
    python predict.py --image_path "/path/to/your/image.jpg"
    ```

2. Il modello restituirà il nome del marchio e del modello dell'auto presente nell'immagine.

## Dipendenze

Le seguenti dipendenze Python sono necessarie per eseguire il progetto:

- `tensorflow`: per costruire e addestrare il modello di deep learning.
- `keras`: per l'implementazione del modello.
- `numpy`: per la manipolazione dei dati.
- `scipy`: per caricare i dati dal file `.mat` del dataset.
- `matplotlib`: per la visualizzazione delle immagini (opzionale, ma utile per il debugging).
- `pillow`: per l'elaborazione delle immagini.
- `tqdm`: per monitorare il progresso durante l'addestramento.

Queste dipendenze sono già elencate nel file `requirements.txt`.

## Contribuire

Se desideri contribuire a questo progetto, sentiti libero di fare un fork della repository, apportare modifiche e inviare una pull request. Assicurati di seguire le linee guida per il codice e di testare accuratamente tutte le modifiche.

## Licenza

Questo progetto è distribuito sotto la licenza MIT. Vedi il file `LICENSE` per ulteriori dettagli.

## Contatti

Per domande o suggerimenti, sentiti libero di aprire una **issue** nella repository o di contattarmi direttamente via email.