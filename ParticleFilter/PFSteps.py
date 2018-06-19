import cv2
import numpy as np


def preprocess(img):
    blur = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow('Binary', binary)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()
    return binary

def initialization(img, N, box_size):
    h, w = img.shape[:2]
    test = np.zeros_like(img)
    centers = []
    particles = []
    color = (0, 0, 255)

    for i in range(N):
        # centers[i] = np.array((np.random.uniform(box_size[0]//2, w - box_size[0]//2), np.random.uniform(box_size[1]//2, h - box_size[0]//2)))
        x = int(np.random.uniform(box_size[0]//2, w - box_size[0]//2))
        y = int(np.random.uniform(box_size[1]//2, h - box_size[0]//2))
        centers.append((x, y))
        # print(x, y)
    for x, y in centers:
        cv2.rectangle(img, (x - box_size[0]//2, y - box_size[1]//2), (x + box_size[0]//2, y + box_size[1]//2), color, 1)
        particles.append((x, y, box_size[0], box_size[1]))
    cv2.imshow('test', img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    print(particles)