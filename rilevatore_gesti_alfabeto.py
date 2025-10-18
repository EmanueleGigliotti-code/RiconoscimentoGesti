import argparse
import csv
import os
from collections import deque
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils

ALFABETO = [
    'A','B','C','D','E','F','G','H','I','K',
    'L','M','N','O','P','Q','R','S','T','U',
    'V','W','X','Y'
]

def estrai_feature_da_landmarks(hand_landmarks, frame_width, frame_height):
    if hand_landmarks is None:
        return None
    pts = []
    for lm in hand_landmarks.landmark:
        pts.append([lm.x * frame_width, lm.y * frame_height, lm.z * frame_width])
    pts = np.array(pts)

    origin = pts[0]
    pts_rel = pts - origin

    max_dist = np.max(np.linalg.norm(pts_rel, axis=1))
    if max_dist == 0:
        max_dist = 1.0
    pts_rel /= max_dist

    return pts_rel.flatten()

def modalità_collect(args):
    out_path = args.out
    label = args.label.upper()

    if label not in ALFABETO:
        print(f"Attenzione: la lettera '{label}' non è nella lista supportata. Le lettere supportate sono:\n{ALFABETO}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossibile aprire la webcam")
        return

    mp_hands = MP_HANDS.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # crea file se non esiste e scrive header
    header_written = os.path.exists(out_path)
    with open(out_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not header_written:
            header = [f"f{i}" for i in range(21 * 3)] + ['label']
            writer.writerow(header)

        print("Premi 's' per salvare il frame corrente con la label, 'q' per uscire.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    MP_DRAW.draw_landmarks(frame, hand_landmarks, MP_HANDS.HAND_CONNECTIONS)

            cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Collect - Webcam', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('s'):
                if results.multi_hand_landmarks:
                    feature = estrai_feature_da_landmarks(results.multi_hand_landmarks[0], frame.shape[1], frame.shape[0])
                    if feature is not None:
                        row = list(feature) + [label]
                        writer.writerow(row)
                        print(f"Salvato: {label} (dimensione feature {len(feature)})")
                    else:
                        print("Feature non valida")
                else:
                    print("Nessuna mano rilevata, non salvato")

    cap.release()
    cv2.destroyAllWindows()


#----- Addestramento
def modalità_train(args):
    in_path = args.infile
    model_path = args.model

    if not os.path.exists(in_path):
        print(f"File dati non trovato: {in_path}")
        return

    df = pd.read_csv(in_path)
    print(f"Dati caricati: {df.shape[0]} righe")

    X = df.drop(columns=['label']).values
    y = df['label'].values

    # filtro: mantieni solo le classi che sono nella lista ALFABETO
    mask = [lab in ALFABETO for lab in y]
    X = X[mask]
    y = y[mask]

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # pipeline: scaler + classificatore
    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42))
    print("Inizio addestramento...")
    pipeline.fit(X_train, y_train)
    print("Addestramento completato")

    # valutazione
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuratezza test: {acc:.3f}")
    print("Report di classificazione:")
    print(classification_report(y_test, y_pred))

    # salva modello
    joblib.dump(pipeline, model_path)
    print(f"Modello salvato in: {model_path}")

#----- Live prediction
def modalità_live(args):
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Modello non trovato: {model_path}. Usa la modalità 'train' per crearne uno.")
        return

    pipeline = joblib.load(model_path)
    print("Modello caricato. Avvio webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Impossibile aprire la webcam")
        return

    mp_hands = MP_HANDS.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    history = deque(maxlen=7)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)

        pred_letter = None
        pred_score = 0.0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            MP_DRAW.draw_landmarks(frame, hand_landmarks, MP_HANDS.HAND_CONNECTIONS)

            feat = estrai_feature_da_landmarks(hand_landmarks, frame.shape[1], frame.shape[0])
            if feat is not None:
                probs = pipeline.predict_proba([feat])[0]
                idx = np.argmax(probs)
                letter = pipeline.classes_[idx]
                score = probs[idx]

                history.append((letter, score))

                # prendo la lettera più frequente
                letters = [h[0] for h in history]
                pred_letter = max(set(letters), key=letters.count)
                # prendo la media dei punteggi per la lettera
                scores_for_letter = [h[1] for h in history if h[0] == pred_letter]
                pred_score = float(np.mean(scores_for_letter)) if scores_for_letter else 0.0

        # UI overlay
        text = f"Pred: {pred_letter if pred_letter else '-'} ({pred_score:.2f})"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        if 'J' in ALFABETO or 'Z' in ALFABETO:
            cv2.putText(frame, "Nota: J e Z sono dinamiche e potrebbero non essere riconosciute.", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        cv2.imshow('Live - Riconoscimento alfabeto', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# ------- MAIN
def main():
    parser = argparse.ArgumentParser(description='Rilevatore di gesti alfabeto (MediaPipe + scikit-learn)')
    sub = parser.add_subparsers(dest='mode', required=True)

    p_collect = sub.add_parser('collect', help='Raccogli dati da webcam e salva feature in CSV')
    p_collect.add_argument('--label', required=True, help='Lettera da associare ai frame (A-Z)')
    p_collect.add_argument('--out', default='data.csv', help='File CSV di output (default: data.csv)')

    p_train = sub.add_parser('train', help='Addestra il modello dai dati CSV')
    p_train.add_argument('--in', dest='infile', required=True, help='File CSV di input')
    p_train.add_argument('--model', default='model.joblib', help='Percorso per salvare il modello')

    p_live = sub.add_parser('live', help='Esegui riconoscimento in tempo reale')
    p_live.add_argument('--model', default='model.joblib', help='Modello serializzato da usare')

    args = parser.parse_args()

    if args.mode == 'collect':
        modalità_collect(args)
    elif args.mode == 'train':
        modalità_train(args)
    elif args.mode == 'live':
        modalità_live(args)

if __name__ == '__main__':
    main()
