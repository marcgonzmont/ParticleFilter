import cv2
import numpy as np
from numba import jit
from math import fabs

delay = 20

def cleanImage(bin_img):
    '''
    Clean binary image
    :param bin_img: binary image from background subtraction
    :return: binary image cleaned
    '''
    n = 3
    iter = 4 #3
    iter_2 = 4
    kernel = np.ones((n, n), np.uint8)
    cleaned = bin_img.copy()

    # cleaned = cv2.dilate(cleaned, kernel= kernel2, iterations= iter_2)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel= kernel, iterations= iter)
    # cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel= kernel, iterations= iter)
    cleaned = cv2.dilate(cleaned, kernel= kernel, iterations= iter_2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel=kernel, iterations=iter)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel=kernel, iterations=iter)

    return cleaned

@jit
def backSubtrDiff(frame1, frame2, sensitivity, show= False):
    '''
    Background subtraction from image difference
    :param frame1:
    :param frame2:
    :param sensitivity: lower level for threshold
    :param show: bool to show the result
    :return: binary mask
    '''
    frame1_gr = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2_gr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame1_gr = cv2.GaussianBlur(frame1_gr, (5, 5), 0)
    frame2_gr = cv2.GaussianBlur(frame2_gr, (5, 5), 0)
    # diff = cv2.absdiff(frame1_gr, frame2_gr)
    diff = cv2.subtract(frame1_gr, frame2_gr)
    # thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY )[1]
    thresh = cleanImage(thresh)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[1]
    w = h = 0
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)

        if show:
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imshow('Thresh', thresh)
            cv2.imshow('Frame', frame1)
            cv2.waitKey(delay)
            # cv2.destroyAllWindows()
    box_size = (h, w)
    # print(box_size)
    return thresh, box_size

@jit
def backSubtrMOG(frame, bgSubtractor, history, show= False):
    '''
    Backgrownd subtractor
    :param frame: frame where apply the subtraction
    :param bgSubtractor: createBackgroundSubtractorMOG object
    :param history: length of the history
    :param show: bool to show the result
    :return:
    '''
    frame1_gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame1_gr = cv2.GaussianBlur(frame1_gr, (5, 5), 0)
    mask = bgSubtractor.apply(frame1_gr, learningRate= 1.0/history)
    mask = cleanImage(mask)
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[1]
    w = h = 0
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        (x, y, w, h) = cv2.boundingRect(contour)

        if show:
            copy = frame.copy()
            cv2.rectangle(copy, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imshow('Thresh', mask)
            cv2.imshow('Object detected', copy)
            cv2.waitKey(delay)
    else:
        if show:
            cv2.imshow('Thresh', mask)
            cv2.waitKey(delay)
    box_size = (h, w)
    # print(box_size)
    return mask, box_size

@jit
def initialization(img, N, box_size, show= False):
    '''
    Initialize particles and weights
    :param img: image to get size
    :param N: number of particles
    :param box_size: bbox size
    :param show: bool to show the result
    :return: particles and weights
    '''
    h, w = img.shape[:2]
    b_h, b_w = box_size
    particles = []
    weights = np.zeros(N, dtype= np.int16)

    for i in range(N):
        x = int(np.random.uniform(b_w//2, w - b_w//2))
        y = int(np.random.uniform(b_h//2, h - b_h//2))
        particles.append([x, y, b_h, b_w])

    if show:
        color = (0, 0, 255)
        for x, y, w, h in particles:
            cv2.rectangle(img, (x - b_w//2, y - b_h//2), (x + b_w//2, y + b_h//2), color, 1)
            # cv2.circle(img, (x, y), 1, (255, 0, 0), 2)
        cv2.imshow('Initialization', img)
        cv2.waitKey(delay)

    return particles, weights

@jit
def evaluation(mask, particles, weights, show= False):
    '''
    Evaluation of particles based on number of white pixels in each one
    :param mask: binary image
    :param particles: array os particles
    :param weights: array of weights
    :param show: bool to show the result
    :return: cumulative sum and weights updated
    '''
    if show:
        copy = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for i, (particle, weight) in enumerate(zip(particles, weights)):
            rect = mask[int(particle[1] - particle[2] // 2): int(particle[1] + particle[2]),
                   int(particle[0] - particle[3] // 2): int(particle[0] + particle[3])]
            weights[i] = np.count_nonzero(rect)
            cv2.rectangle(copy, (particle[0] - particle[3] // 2, particle[1] - particle[2] // 2),
                          (particle[0] + particle[3] // 2, particle[1] + particle[2] // 2), (0, 255, 0), 1)
        cv2.imshow('Evaluation', copy)
        cv2.waitKey(delay)
    else:
        for i, (particle, weight) in enumerate(zip(particles, weights)):
            rect = mask[int(particle[1] - particle[2]//2) : int(particle[1] + particle[2]) ,
                   int(particle[0] - particle[3]//2) : int(particle[0] + particle[3])]
            weights[i] = np.count_nonzero(rect)
    summ = weights.sum()

    return summ, weights

@jit
def estimation(frame, particles, weights):
    '''
    Estimate object position
    :param frame: current frame
    :param particles: array of particles
    :param weights: arrat of weights
    :return:
    '''
    draw_frame = frame.copy()
    particle_idx = np.argmax(weights)
    particle = particles[particle_idx]
    color = (0, 255, 0)
    cv2.rectangle(draw_frame, (int(particle[0] - particle[3]//2), int(particle[1] - particle[2]//2)),
                  (int(particle[0] + particle[3]), int(particle[1] + particle[2])), color, 2)

    return draw_frame

@jit
def selection(weights):
    '''
    Select particles for the next generation using the roulette method
    :param weights: array of weights
    :return: array of particles index
    '''
    # Normalization
    weights = weights / weights.sum()
    weights_cum = np.cumsum(weights)
    randoom_values = np.zeros(len(weights_cum))
    # Roulette method
    for state in range(len(weights_cum)):
        prob = np.random.rand()
        for acc_idx in range(len(weights_cum)):
            if weights_cum[acc_idx] >= prob:
                randoom_values[state] = acc_idx
                break

    return randoom_values.astype(int)


@jit
def diffusion(particles, random_values, box_size):
    '''
    Add gaussian noise to new particles generation and avoid degeneration
    :param particles: array of particles
    :param random_values: array of particles index
    :param box_size: size of bbox
    :return: new particles generation
    '''
    for i, particle in enumerate(particles):
        particle[0] = int(fabs(np.random.normal(0, 15) + particles[random_values[i]][0] ))  # x
        particle[1] = int(fabs(np.random.normal(0, 15) + particles[random_values[i]][1] ))  # y
        particle[2] = box_size[0]   # bbox_h
        particle[3] = box_size[1]   # bbox_h

    return particles