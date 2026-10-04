import json
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import mediapipe as mp
from collections import deque

from model import VSLModel
from sentence_builder import AUTO_SENTENCE_WORDS, SentenceComposer, WordBuffer


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_MAP_PATH = os.path.join(ROOT_DIR, 'label_map.json')
MODEL_PATH = os.path.join(ROOT_DIR, 'best_model.pth')
SCALER_PATH = os.path.join(ROOT_DIR, 'scaler.npz')
SEQ_LENGTH = 60
CONFIDENCE_THRESHOLD = 0.2
SMOOTHING_WINDOW = 5


def check_files():
    missing = []
    for path, name in [(LABEL_MAP_PATH, 'label_map.json'),
                       (MODEL_PATH, 'best_model.pth'),
                       (SCALER_PATH, 'scaler.npz')]:
        if not os.path.exists(path):
            missing.append(name)
    if missing:
        print(f"ERROR: Missing required file(s): {', '.join(missing)}")
        print("Run train.py first to generate these files.")
        sys.exit(1)


def extract_keypoints(results):
    pose = np.zeros(75)
    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark[:25]]).flatten()

    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()

    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, left_hand, right_hand])


def main():
    check_files()

    with open(LABEL_MAP_PATH, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(label_map)
    model = VSLModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Model loaded. {num_classes} classes.")

    scaler_data = np.load(SCALER_PATH)
    scaler_mean = torch.from_numpy(scaler_data['mean']).float().to(device)
    scaler_std = torch.from_numpy(scaler_data['std']).float().to(device)

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    sequence = deque(maxlen=SEQ_LENGTH)
    pred_history = deque(maxlen=SMOOTHING_WINDOW)
    word_buffer = WordBuffer(threshold=CONFIDENCE_THRESHOLD)
    composer = SentenceComposer()
    print("Press 's' = ghép câu · 'c' = xóa · 'u' = xóa từ cuối · 'q' = thoát.")
    if composer.ready:
        print(f"LLM: {composer.model} @ {composer.base_url}")
    else:
        print("LLM: chưa kích hoạt (thêm VSL_LLM_API_KEY vào .env) -> ghép cơ bản.")
    last_printed = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        prediction_text = "Waiting..."
        color = (0, 255, 0)
        top3_text = []

        if len(sequence) == SEQ_LENGTH:
            input_tensor = torch.from_numpy(
                np.array(sequence, dtype=np.float32)
            ).unsqueeze(0).to(device)
            input_tensor = (input_tensor - scaler_mean) / scaler_std

            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)
                top_probs, top_indices = torch.topk(probs, k=3, dim=1)

            top3_probs = top_probs[0].cpu().numpy()
            top3_idxs = top_indices[0].cpu().numpy()
            top3_labels = [idx_to_label.get(i, "?") for i in top3_idxs]

            pred_history.append(top3_idxs[0])
            smoothed_pred = max(set(pred_history), key=list(pred_history).count)

            committed = word_buffer.feed(
                idx_to_label.get(smoothed_pred) if has_hand else None,
                float(top3_probs[0]),
            )
            if committed and len(word_buffer.words) >= AUTO_SENTENCE_WORDS:
                composer.compose(list(word_buffer.words))

            top3_text = []
            for i in range(3):
                confidence = top3_probs[i]
                label = top3_labels[i]
                prefix = ">>" if i == 0 else "  "
                top3_text.append(f"{prefix} {label[:35]:35s} {confidence:.1%}")

            if top3_probs[0] >= CONFIDENCE_THRESHOLD:
                prediction_text = top3_labels[0]
                color = (0, 255, 0)
            else:
                prediction_text = "Unknown"
                color = (0, 0, 255)

        cv2.putText(frame, prediction_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        for i, text in enumerate(top3_text):
            y_offset = 110 + i * 35
            cv2.putText(frame, text, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        h, w = frame.shape[:2]
        if word_buffer.words:
            cv2.putText(frame, "words: " + " ".join(word_buffer.words),
                        (30, h - 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        status_prefix = {"composing": "...dang ghep... ", "api": "[LLM] ",
                         "fallback": "[co ban] "}.get(composer.status, "")
        if composer.sentence:
            cv2.putText(frame, status_prefix + composer.sentence,
                        (30, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255) if composer.status == "api" else (180, 180, 255), 2)
            if composer.sentence != last_printed:
                last_printed = composer.sentence
                print(f"CÂU ({composer.status}): {composer.sentence}")
        cv2.putText(frame, "'s' cau · 'c' xoa · 'u' xoa tu cuoi · 'q' thoat",
                    (30, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow('VSL Real-Time Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            composer.compose(list(word_buffer.words), force=True)
        elif key == ord('c'):
            word_buffer.clear()
            composer.reset()
            last_printed = ""
            print("Đã xóa buffer.")
        elif key == ord('u'):
            word_buffer.undo()
            print(f"Buffer: {' '.join(word_buffer.words) or '(trống)'}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
