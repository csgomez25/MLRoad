import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    exit("Error: Webcam not accessible.")

# Set window size
window_width, window_height = 640, 480
cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Webcam', window_width, window_height)

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
