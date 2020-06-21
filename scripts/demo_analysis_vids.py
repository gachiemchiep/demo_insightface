import insightface
import cv2
import argparse
import numpy as np
from util import generate_pascal_colormap, draw_on_img
import time
import os

def get_args():
    parser = argparse.ArgumentParser(description='Demo insightface on Video/Webcam.')
    parser.add_argument('--video', help='Location of video file.', type=str)
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    # capture from webcam
    if not args.video:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.video)

    # see https://docs.opencv.org/3.4/dd/d43/tutorial_py_video_display.html
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("out.avi", fourcc, cap_fps, (cap_w, cap_h))

    # Init insightface
    model = insightface.app.FaceAnalysis()

    # Use CPU
    ctx_id = -1
    # Use GPU
    # ctx_id = 0

    # config the model
    model.prepare(ctx_id=ctx_id, nms=0.4)

    # warm up network
    print("Warming up")
    input_data = np.random.uniform(size=(224, 224, 3)).astype("float32")
    model.get(input_data)

    # colormap for drawing
    colormap = generate_pascal_colormap()

    print("Start processing")
    while True:
        t0 = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_debug = frame.copy()

        faces = model.get(frame)
        t1 = time.time()
        fps = "{:03.1f}".format( 1 / (t1 - t0) )

        draw_on_img(frame_debug, faces, colormap, extra_text=fps)

        # Display the resulting frame
        cv2.imshow('frame', frame_debug)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out.write(frame_debug)

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
