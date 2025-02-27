import cv2
import mediapipe as mp
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not access webcam.")
    exit()
mp_face_detection = mp.solutions.face_detection
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, img = webcam.read()
        if not ret:
            print("Failed to grab frame.")
            break
        H, W, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = face_detection.process(img_rgb)
        if out.detections:
            for detection in out.detections:
                location_data = detection.location_data
                bbox = location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * W))
                y1 = max(0, int(bbox.ymin * H))
                w = int(bbox.width * W)
                h = int(bbox.height * H)
                x2 = min(W, x1 + w)
                y2 = min(H, y1 + h)
                img_box = img.copy()
                cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('Step 1: Face Detection', img_box)
                cv2.waitKey(500)
                img_blur = img.copy()
                img_blur[y1:y2, x1:x2] = cv2.blur(img_blur[y1:y2, x1:x2], (40, 40))
                cv2.imshow('Step 2: Face Blurred', img_blur)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
webcam.release()
cv2.destroyAllWindows()
