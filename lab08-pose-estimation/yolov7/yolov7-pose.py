import time
import torch
import cv2
import numpy as np
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts


def pose_video(frame):
    mapped_img = frame.copy()

    img = letterbox(frame, input_size, stride=64, auto=True)[0]

    img = transforms.ToTensor()(img)

    img = torch.tensor(np.array([img.numpy()]))

    img = img.to(device)

    with torch.no_grad():
        t1 = time.time()

        output, _ = model(img)

        t2 = time.time()

        fps = 1 / (t2 - t1)

        output = non_max_suppression_kpt(
            output,
            0.25,
            0.65,
            nc=1,
            nkpt=17,
            kpt_label=True
        )

        output = output_to_keypoint(output)

    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

    return nimg, fps


# ---------------------------------------------------------------------------- #
# Input size
input_size = 256

# ---------------------------------------------------------------------------- #
# Device selection

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print("Selected Device :", device)

# ---------------------------------------------------------------------------- #
# Load model

weights = torch.load(
    "yolov7-w6-pose.pt",
    map_location=torch.device("cpu"),
    weights_only=False
)

model = weights["model"]

_ = model.float().eval()

model.to(device)

# ---------------------------------------------------------------------------- #
# Video selection

videos = [
    "skydiving",
    "far-away"
]

file_name = videos[0] + ".mp4"

vid_path = "../media/" + file_name

# ---------------------------------------------------------------------------- #
# Video capture

cap = cv2.VideoCapture(vid_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))

ret, frame = cap.read()

h, w, _ = frame.shape

save_name = videos[0]

# ---------------------------------------------------------------------------- #
# Video writer

out = cv2.VideoWriter(
    f"{save_name}_yolo7.avi",
    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
    10,
    (w, h)
)

# ---------------------------------------------------------------------------- #

if __name__ == "__main__":

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to read frame. Exiting ..")
            break

        img, fps_ = pose_video(frame)

        cv2.putText(
            img,
            "FPS : {:.2f}".format(fps_),
            (200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            "YOLOv7",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Output", img[..., ::-1])

        out.write(img[..., ::-1])

        key = cv2.waitKey(1)

        if key == ord("q"):
            break

    cap.release()

    out.release()

    cv2.destroyAllWindows()