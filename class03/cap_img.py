import cv2 as cv
import os

img_path = './images/'

# Create directory if it does not exist
if not os.path.exists(img_path):
    os.makedirs(img_path)

# Find the next available index to avoid overwriting
existing_files = [f for f in os.listdir(img_path) if f.startswith("data_") and f.endswith(".jpg")]
indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
counter = max(indices) + 1 if indices else 1

cap = cv.VideoCapture(2)

while True:
    _, img = cap.read()

    cv.imshow('original', img)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'):
        filename = os.path.join(img_path, f"data_{counter}.jpg")
        cv.imwrite(filename, img)
        print(f"Saving {filename}")
        counter += 1
        continue

cap.release()
cv.destroyAllWindows()
