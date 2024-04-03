import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)

finger_tips =[8, 12, 16, 20]
tipIds = [4, 8, 12, 16, 20]

finger_fold_status = []

def countFingers(image, hand_landmarks, handNo=0):
    fingers = []

    if hand_landmarks:
        landmarks = hand_landmarks.landmark

        for lm_index in tipIds:
            finger_tip_y = landmarks[lm_index].y
            finger_bottom_y = landmarks[lm_index - 2].y

            if lm_index != 4:
                if finger_tip_y < finger_bottom_y:
                    fingers.append(1)
                    print("FINGER with id ", lm_index, "is Closed")
                else:
                    fingers.append(0)

    totalFingers = fingers.count(1)

    text = f'Fingers: {totalFingers}'
    cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return fingers

def drawHandLandmarks(image, hand_landmarks):
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

while True:
    success, image = cap.read()

    image = cv2.flip(image, 1)

    # Detecta los puntos de referencia de las manos.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Obtén la posición de los puntos de referencia del resultado procesado.
    hand_landmarks = results.multi_hand_landmarks

    # Dibuja los puntos de referencia.
    drawHandLandmarks(image, hand_landmarks)

    # Obtén las posiciones de los dedos de la mano.
    fingers = countFingers(image, hand_landmarks)
    print(f"Fingers status: {fingers}")

    for idx, tip_id in enumerate(finger_tips):
        lm = hand_landmarks.landmark[tip_id]
        height, width, _ = image.shape
        center_x, center_y = int(lm.x * width), int(lm.y * height)
        
        # Dibuja círculos alrededor de las puntas de los dedos
        cv2.circle(image, (center_x, center_y), 10, (255, 0, 0), cv2.FILLED)

        # Comprueba si el dedo está doblado
        if fingers[idx] == 1:
            cv2.circle(image, (center_x, center_y), 20, (0, 255, 0), cv2.FILLED)
            finger_fold_status.append(True)
        else:
            cv2.circle(image, (center_x, center_y), 20, (0, 0, 255), cv2.FILLED)
            finger_fold_status.append(False)

    # Comprueba si todos los dedos están doblados
    if all(finger_fold_status):
        thumb_tip = hand_landmarks.landmark[4]
        pinky_tip = hand_landmarks.landmark[20]
        if thumb_tip.y < pinky_tip.y:
            cv2.putText(image, "Me gusta", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Disgusto", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Controlador de medios", image)

    # Cierra la ventana al presionar la barra espaciadora.
    key = cv2.waitKey(1)
    if key == 32:
        break

cap.release()
cv2.destroyAllWindows()

