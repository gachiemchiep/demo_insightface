"""1. Getting Started with Pre-trained Model from InsightFace
=======================================================


In this tutorial, we will demonstrate how to load a pre-trained model from :ref:`insightface-model-zoo`
and analyze faces from images.

Step by Step
------------------

Let's first try out a pre-trained insightface model with a few lines of python code.

First, please follow the `installation guide <../../index.html#installation>`__
to install ``MXNet`` and ``insightface`` if you haven't done so yet.
"""

import insightface
import urllib
import urllib.request
import cv2
import numpy as np
import time
import copy

LANDMARK_COLOR = (0, 255, 0)
TEXT_COLOR = (0, 0, 139)

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


def draw_on_img(img, faces, colormap):
    for idx, face in enumerate(faces):
        # Draw bbox
        bbox = face.bbox.astype(np.int).flatten()
        start_point = (int(bbox[0]), int(bbox[1]))
        end_point = (int(bbox[2]), int(bbox[3]))
        color = colormap[idx]
        cv2.rectangle(img, pt1=start_point, pt2=end_point,  color=color, thickness=2)

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


################################################################
#
# Then, we download and show the example image:

url = 'https://github.com/deepinsight/insightface/blob/master/sample-images/t1.jpg?raw=true'
img = url_to_image(url)

################################################################
# Init FaceAnalysis module by its default models
#     'arcface_r100_v1': arcface_r100_v1,
#     #'arcface_mfn_v1': arcface_mfn_v1,
#     #'arcface_outofreach_v1': arcface_outofreach_v1,
#     'retinaface_r50_v1': retinaface_r50_v1,
#     'retinaface_mnet025_v1': retinaface_mnet025_v1,
#     'retinaface_mnet025_v2': retinaface_mnet025_v2,
#     'genderage_v1': genderage_v1,

model = insightface.app.FaceAnalysis()

################################################################
# Use CPU to do all the job. Please change ctx-id to a positive number if you have GPUs
#

ctx_id = 0

################################################################
# Prepare the enviorment
# The nms threshold is set to 0.4 in this example.
#

model.prepare(ctx_id=ctx_id, nms=0.4)

################################################################
# Analysis faces in this image
# inference time = 0.2 second
# landmark = interest point on the face such as eye, nose, etc
# for i in range(100):
#     t0 = time.time()
#     faces = model.get(img)
#     print(time.time() - t0)

img_draw = img.copy()

faces = model.get(img)

colormap = generate_pascal_colormap(256)
draw_on_img(img_draw, faces, colormap)

cv2.imshow("insightface demo", img_draw)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()

# for idx, face in enumerate(faces):
#   print("Face [%d]:"%idx)
#   print("\tage:%d"%(face.age))
#   gender = 'Male'
#   if face.gender==0:
#     gender = 'Female'
#   print("\tgender:%s"%(gender))
#   print("\tembedding shape:%s"%face.embedding.shape)
#   print("\tbbox:%s"%(face.bbox.astype(np.int).flatten()))
#   print("\tlandmark:%s"%(face.landmark.astype(np.int).flatten()))
#   print("")



