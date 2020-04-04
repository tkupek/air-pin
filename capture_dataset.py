import os
import cv2

# Frame Rate to capture the dataset, 20 is a good value
CAPTURE_FRAMES = 20
WINDOW_NAME = 'AirPIN - Capture Mode'
STREAM_URL = 'rtmp://192.168.178.32/live'


if __name__ == '__main__':
    print('Showing camera feed. Click window or press any key to stop.')
    cameraCapture = cv2.VideoCapture(STREAM_URL)
    cv2.namedWindow(WINDOW_NAME)

    count_frames = 0
    while True:
        _, frame = cameraCapture.read()

        if frame is None:
            print('Got empty frame. Will die now...')
            break

        if CAPTURE_FRAMES is not 0 and count_frames % CAPTURE_FRAMES == 0:
            # Capture mode
            filename = os.path.join('captures', 'capture_' + str(count_frames) + '.jpg')
            cv2.imwrite(filename, frame)

        cv2.imshow(WINDOW_NAME, frame)
        count_frames += 1

        if cv2.waitKey(1) & 0xff == ord("q"):
            break

    cv2.destroyWindow(WINDOW_NAME)
    cameraCapture.release()
