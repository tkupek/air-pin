import cv2
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model


# Frame Rate to predict the frame, 50 is a good value
CLASS_FRAMES = 20
WINDOW_NAME = 'AirPIN - Live Demo'
STREAM_URL = 'rtmp://192.168.178.32/live'
label_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'null']
resize = (int(1280 / 4), int(720 / 4))

rec_level = 4


def predict(model, img):
    try:
        img = np.array(img)/255.0
        img = np.float32(np.expand_dims(img, 0))
        preds = model.predict(img)
        return tf.argmax(preds, 1).numpy()[0]
    except Exception:
        return -1


def add_text(img, pred_text):
    img = np.float32(img / 255.0)
    rectangle_bgr = (255, 255, 255)
    (text_width, text_height) = cv2.getTextSize(pred_text, 5, fontScale=4, thickness=2)[0]
    text_offset_x = int(img.shape[1] / 2) - int(text_width / 2)
    text_offset_y = int(img.shape[0] / 2)
    box_coords = ((text_offset_x - 10, text_offset_y + 10), (text_offset_x + text_width + 10, text_offset_y - text_height - 10))
    cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(img=img, text=pred_text, org=(text_offset_x, text_offset_y), fontFace=5, fontScale=4, color=(0, 0, 0), thickness=2)
    return img


if __name__ == '__main__':
    model = None

    pin = []
    digits = []
    skip_frames = 0

    if CLASS_FRAMES is not 0:
        print('load model')
        model = load_model(os.path.join('cnn', 'model', 'airpin-model_v1.h5'))

    print('Showing camera feed. Click window or press any key to stop.')
    cameraCapture = cv2.VideoCapture(STREAM_URL)
    cv2.namedWindow(WINDOW_NAME)

    count_frames = 0
    while True:
        success, frame = cameraCapture.read()

        if frame is None:
            print('Got empty frame. Will die now.')
            continue

        if skip_frames > 0:
            frame = np.ones(frame.shape) * 255
            skip_frames -= 1

        elif CLASS_FRAMES is not 0:
            if count_frames % CLASS_FRAMES == 0 and len(pin) != 4:
                class_frame = cv2.resize(frame, resize)
                pred = predict(model, class_frame)
                digits.append(pred)
                print(pred)

                if len(digits) >= rec_level and len(set(digits[-rec_level:])) is 1 and digits[-1] != 10:
                    # digit recognized
                    pin.append(digits[-1])
                    print('recognized ' + label_names[digits[-1]])
                    skip_frames = 5
                    digits = []

        if len(pin) == 4:
            skip_frames = 0
            pin_numbers = np.array([label_names[x] for x in pin])
            pred_text = ' - '.join(pin_numbers)
            frame = add_text(frame, pred_text)

        cv2.imshow(WINDOW_NAME, frame)
        count_frames += 1

        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    cv2.destroyWindow(WINDOW_NAME)
    cameraCapture.release()
