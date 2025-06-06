import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp
import pygame
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
import screen_brightness_control as sbc




# Model ve etiketler
try:
    model = keras.models.load_model("C:\\Users\\asus\\Downloads\\hand_model5.keras")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    exit()

gesture_labels = ['A', 'F', 'L', 'Y']

# MediaPipe el tespiti
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

# Webcam başlat
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera açılamadı")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı")
        break

    # MediaPipe için RGB formatına çevir
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # El sınırlarını çiz (opsiyonel)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # El bölgesini kırp (bounding box hesapla)
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Kenar boşluğu ekle
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)

            # El bölgesini kırp
            hand_color = frame[y_min - 100:y_max + 100, x_min- 100:x_max + 100]
            if hand_color.size == 0:
                continue

            # Model için preprocessing
            hand_rgb = cv2.cvtColor(hand_color, cv2.COLOR_BGR2RGB)
            hand_resized = cv2.resize(hand_rgb, (224, 224))
            hand_input = np.expand_dims(hand_resized, axis=0)  # Normalizasyon

            # Tahmin
            predictions = model.predict(hand_input)
            print(predictions.round(2))
            gesture_index = np.argmax(predictions)
            gesture = gesture_labels[gesture_index]

            current_volume = volume.GetMasterVolumeLevelScalar()
            # current_brightness = sbc.get_brightness()

            if gesture == 'A':
                volume.SetMasterVolumeLevelScalar(min(current_volume + 0.05, 1.0), None)
                print("Səs artdı")

            elif gesture == 'F':
                volume.SetMasterVolumeLevelScalar(max(current_volume - 0.05, 0.0), None)
                print("Səs azaldı")


                #isiq
            # if gesture == 'Y':
            #    sbc.set_brightness(min(current_brightness + 5, 100))
            #    print("İşıq artdı")

            # elif gesture == 'L':
            #     sbc.set_brightness(max(current_brightness - 5, 0))
            #     print("İşıq azaldı")

            

            # Sadə səs göstəricisi
            vol = int(current_volume * 100)
            cv2.rectangle(frame, (50, 400), (50 + vol * 3, 430), (0, 255, 0), -1)
            cv2.putText(frame, f"{vol}%", (50, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

           
            # cv2.rectangle(frame, (x_min-50, y_min+50), (x_max-50, y_max+50), (255, 0, 0), 2)
            cv2.putText(frame, gesture, (x_min, y_min - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        


    # Görüntüyü göster
    cv2.imshow('Webcam Feed', frame)

    # 'q' ile çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizle
cap.release()
cv2.destroyAllWindows()
hands.close()


