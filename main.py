import cv2
import numpy as np
from lib_detection import load_model, detect_lp, im2single


# Arrange the contour from left to right
def sort_contours(contours):
    reverse = False
    i = 0
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, bounding_boxes),
                                            key=lambda b: b[1][i], reverse=reverse))
    return contours


# Character definition
char_list = '0123456789ABCDEFGHKLMNPRSTUVXYZ'


# Fine tune license plate, remove irrational characters
def fine_tune(lp):
    new_string = ""
    for i in range(len(lp)):
        if lp[i] in char_list:
            new_string += lp[i]
    return new_string


# Load model LP detection
wpod_net_path = "wpod-net_update1.json"
wpod_net = load_model(wpod_net_path)

cap = cv2.VideoCapture('test/6344170718708548413.mp4')
# The largest and smallest sizes of an image size
d_max = 608
d_min = 288

# Configure model parameters for SVM
digit_w = 30  # Character width
digit_h = 60  # Character height

model_svm = cv2.ml.SVM_load('svm.xml')

while cap.isOpened():
    ret, frame = cap.read()

    # Read img
    image_vehicle = frame

    # Get the ratio between W and H of the figure and find the smallest dimensions

    ratio = float(max(image_vehicle.shape[:2])) / min(image_vehicle.shape[:2])
    side = int(ratio * d_min)
    bound_dim = min(side, d_max)

    _, license_plate_images = detect_lp(wpod_net, im2single(image_vehicle), bound_dim, lp_threshold=0.5)

    if len(license_plate_images):

        # Convert license plate image
        license_plate_images[0] = cv2.convertScaleAbs(license_plate_images[0], alpha=255.0)

        roi = license_plate_images[0]

        # Convert image to gray
        gray = cv2.cvtColor(license_plate_images[0], cv2.COLOR_BGR2GRAY)

        # Apply a threshold to separate the numbers from the background
        binary = cv2.threshold(gray, 127, 255,
                               cv2.THRESH_BINARY_INV)[1]

        # Morphology
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w

            # Choose a contour
            if 1.5 <= ratio <= 3.5:
                if h / roi.shape[0] >= 0.6:

                    # Draw a border around the character
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Character separation and prediction
                    curr_num = thre_mor[y:y + h, x:x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)

                    cv2.imshow("xxx", curr_num)

                    curr_num = np.array(curr_num, dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    # Predicted by model SVM
                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    # ASCII code
                    if result <= 9:
                        result = str(result)
                    else:
                        result = chr(result)

                    plate_info += result

        # Image number the given
        cv2.putText(image_vehicle, fine_tune(plate_info), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255),
                    lineType=cv2.LINE_AA)

        # Show image result
        print("License plate", plate_info)
        cv2.imshow("Output", image_vehicle)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
