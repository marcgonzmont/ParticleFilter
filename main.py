import argparse
from myPackage import tools as tl
from ParticleFilter import PFSteps as PFS
from os.path import isdir
import cv2

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required= True,
                    help="-p Sequences path")
    args = vars(ap.parse_args())

    # Configuration
    seq_idx = 0                     # sequence: 0; video: [0,2]
    video = False                    # if video file will be used, else sequences
    box_size_aux = (15, 15)         # initialize bbox size
    N = 100                         # number of particles
    show = False                    # to show result of determinate functions
    debug = False                   # to show all particles in each iteration
    # sensitivity = 20              # lower level for threshold
    history = 10                    # history for background subtractor
    count_down = iter = 2           # number of frames without object detection to reinitialize
    delay = PFS.delay                      # fot cv2.waitKey

    # Background subtractor model
    # bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()

    # Get folders with the sequences or videos
    path = args["path"]

    if video:
        sequences = tl.natSort(tl.getSamples(path))
    else:
        sequences = tl.natSort(tl.getSequences(path))
    print("\nSequences available:\n{}\n\n".format(sequences))
    # Get the frames of the sequence and make results folder
    seq_selected = sequences[seq_idx]
    if isdir(seq_selected):
        # Get all frames
        frames = tl.natSort(tl.getSamples(seq_selected))
        frame = cv2.imread(frames[0])
        # Initialization
        particles, weights = PFS.initialization(frame, N, box_size_aux)
        for fr_idx in range(len(frames)):
            frame1 = cv2.imread(frames[fr_idx])
            # diff_binary, box_size = PFS.backSubtrDiff(frame1, frame2, sensitivity)
            mask, box_size = PFS.backSubtrMOG(frame1, bgSubtractor, history)
            # Mask empty and no object detected
            if box_size == (0,0):
                box_size = box_size_aux
                count_down -= 1
                if count_down == 0:
                    # Reinitialization
                    particles, weights = PFS.initialization(frame1, N, box_size)
                    count_down = iter
            # Evaluation
            summ, weights = PFS.evaluation(mask, particles, weights, show)
            # break
            # print(summ)
            draw_frame = frame1.copy()
            # Object detected
            if summ != 0:
                # Estimation
                draw_frame = PFS.estimation(draw_frame, particles, weights)
                # Selection
                selected_particles = PFS.selection(weights)
                # Diffusion
                particles = PFS.diffusion(particles, selected_particles, box_size)
                if debug:
                    color = (0, 0, 255)
                    copy = draw_frame.copy()
                    for x, y, w, h in particles:
                        cv2.rectangle(copy, (x - box_size[1] // 2, y - box_size[0] // 2), (x + box_size[1] // 2, y + box_size[0] // 2), color, 1)
                        # cv2.circle(img, (x, y), 1, (255, 0, 0), 2)
                    cv2.imshow('Debug', copy)
                    cv2.waitKey(delay)
            else:
                # Reinitialization
                particles, weights = PFS.initialization(frame1, N, box_size)
            cv2.imshow('Particle Filter', draw_frame)
            cv2.waitKey(delay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    else:
        # Read first frame
        video = cv2.VideoCapture(seq_selected)
        ret, frame = video.read()
        particles, weights = PFS.initialization(frame, N, box_size_aux)
        # Release video
        video.release()
        # Reopen the video
        video = cv2.VideoCapture(seq_selected)
        while video.isOpened():
            ret, frame1 = video.read()
            if ret:
                mask, box_size = PFS.backSubtrMOG(frame1, bgSubtractor, history)
                # Mask empty and no object detected
                if box_size == (0, 0):
                    box_size = box_size_aux
                    count_down -= 1
                    if count_down == 0:
                        # Reinitialization
                        particles, weights = PFS.initialization(frame1, N, box_size)
                        count_down = iter
                    draw_frame = frame1.copy()
                else:
                    # Evaluation
                    summ, weights = PFS.evaluation(mask, particles, weights, show)

                    draw_frame = frame1.copy()
                    if summ != 0:
                        # Estimation
                        draw_frame = PFS.estimation(draw_frame, particles, weights)
                        # Selection
                        selected_particles = PFS.selection(weights)
                        # Diffusion
                        particles = PFS.diffusion(particles, selected_particles, box_size)
                        # break
                        if debug:
                            color = (0, 0, 255)
                            copy = draw_frame.copy()
                            for x, y, w, h in particles:
                                cv2.rectangle(copy, (x - box_size[1] // 2, y - box_size[0] // 2),
                                              (x + box_size[1] // 2, y + box_size[0] // 2), color, 1)
                                # cv2.circle(img, (x, y), 1, (255, 0, 0), 2)
                            cv2.imshow('Debug', copy)
                            cv2.waitKey(delay)
                    else:
                        # Reinitialization
                        particles, weights = PFS.initialization(frame1, N, box_size)
                cv2.imshow('Particle Filter', draw_frame)
                cv2.waitKey(delay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    video.release()
                    break
            else:
                break
        video.release()
    cv2.destroyAllWindows()