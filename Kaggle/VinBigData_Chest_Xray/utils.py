import PIL
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches

from params import CLASS_IDNAME


CLRS = mpl.colormaps.get("jet")(np.linspace(0, 1, len(CLASS_IDNAME)))[:, :-1]


def show_image(
    image: PIL.Image,
    bboxes: np.array,
    labels: np.array,
    image_name: str,
    size: np.array,
    confidences: np.array = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    image = image.resize((size[1], size[0]))  # width, height
    ax.imshow(image, cmap="gray")
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        if label == 14:
            break
        rect = patches.Rectangle(
            xy=(bbox[0], bbox[1]),
            width=bbox[2] - bbox[0],
            height=bbox[3] - bbox[1],
            facecolor="none",
            edgecolor=CLRS[label],
            linewidth=1.5,
        )
        ax.add_patch(rect)
        txt = f"{CLASS_IDNAME[label]}"
        if confidences is not None:
            txt += f": {confidences[i]:.2f}"
        ax.text(x=bbox[0], y=bbox[1] - 10, s=txt, c=CLRS[label])
    plt.title(image_name)
    plt.axis("off")
    plt.show()