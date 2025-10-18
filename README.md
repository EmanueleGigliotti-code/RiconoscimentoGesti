# ✋ Riconoscimento Gesti

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-VideoCapture-informational)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-success)
![scikit--learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)

---

## 🧭 Breve introduzione
Riconoscimento in tempo reale delle **lettere dell’alfabeto** con **MediaPipe Hands** e un **Random Forest** scikit-learn.

**Il progetto include:**
1. **Raccolta dati** via webcam (estrazione feature normalizzate dai 21 landmarks *x,y,z* della mano)  
2. **Addestramento** (pipeline `StandardScaler + RandomForestClassifier`)  
3. **Predizione live** con **smoothing** (finestra mobile)

---

## ▶️ Come utilizzare il programma

> Prima di tutto “insegno” al modello i gesti della mano che deve imparare.

### 1) 📸 Raccolta dati
Interfaccia da terminale tramite `argparse`. In base alla tua dir e al nome del file, avvia la collect, ad esempio per **A**:

    python rilevatore_gesti_alfabeto.py collect --label A

Con la lettera S salvo i vari frame in base a luci e angolazioni. Una volta fatto con la lettera Q esco e chiude la webcam.

### 2) 🧠 Addestramento del modello

Una volta raccolti i dati vado ad addestrare il modello: 

    python rilevatore_gesti_alfabeto.py train --in data.csv 
    
Il model non l'ho specificato e ho lasciato quello di defeault

Output: accuratezza test e classification_report.

### 3) 🚀 Modalità Live
Una volta salvato il modello procedo con la modalità live e faccio predizioni in tempo reale 

    python rilevatore_gesti_alfabeto.py live
