import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import time
import copy

LANDMARK_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 255)


def generate_pascal_colormap(size=256):
    # BGR colormap
    colormap = np.zeros((size, 3), dtype=int)
    ind = np.arange(size, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap[1:].tolist()

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def draw_on_img(img, faces, colormap, extra_text):
    for idx, face in enumerate(faces):

        # Draw extra_text on the left_top
        cv2.putText(img, extra_text, (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=TEXT_COLOR, thickness=2)

        # Draw bbox
        bbox = face.bbox.astype(np.int).flatten()
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        color = colormap[idx]
        cv2.rectangle(img, pt1=start_point, pt2=end_point,  color=(128, 128, 128), thickness=2)

        # Draw landmark
        landmarks = face.landmark.astype(np.int).flatten()
        Xs = [landmarks[i] for i in range(0, len(landmarks), 2)]
        Ys = [landmarks[i] for i in range(1, len(landmarks), 2)]
        for X, Y in zip(Xs, Ys):
            cv2.circle(img, (X, Y), radius=1, color=LANDMARK_COLOR, thickness=-1)

        # Draw text
        text_cord = (bbox[0], bbox[1] - 8)
        gender = 'Male'
        if face.gender == 0:
            gender = 'Female'
        text = "{}:{}:{}".format(idx, gender, face.age)
        cv2.putText(img, text, text_cord, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=TEXT_COLOR)
