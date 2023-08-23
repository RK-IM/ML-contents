import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches


def show_image(image, bboxes):
    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox in bboxes:
        rect = patches.Rectangle(xy=(bbox[0], bbox[1]),
                                width=bbox[2]-bbox[0],
                                height=bbox[3]-bbox[1],
                                facecolor="none",
                                edgecolor="w",
                                linewidth=1)
        ax.add_patch(rect)
    plt.axis("off")
    plt.show()


def show_image_with_predict(model, image, confidence_score=0.25):
    model.eval()
    bboxes, confidences, _ = model.predict_sample(image)
    if len(confidences) == 1 and confidences[0].sum() == 0:
        plt.imshow(image)
        return
    
    idx = np.where(confidences[0] > confidence_score)
    bboxes = np.array(bboxes[0])[idx]
    confidences = np.array(confidences[0])[idx]

    fig, ax = plt.subplots()
    ax.imshow(image)
    for bbox, confidence in zip(bboxes, confidences):
        rect = patches.Rectangle(xy=(bbox[0], bbox[1]),
                                width=bbox[2]-bbox[0],
                                height=bbox[3]-bbox[1],
                                facecolor="none",
                                edgecolor="w",
                                linewidth=1)
        ax.add_patch(rect)
        ax.text(x=bbox[0], y=bbox[1]-5, s=f"{confidence:.2f}", c="w")

    plt.axis("off")
    plt.show()