import argparse
from myPackage import tools as tl
from ParticleFilter import PFSteps as PFS
from os.path import basename, altsep, exists
import cv2
from PIL import Image
import numpy as np
import time
import sys

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    # group = ap.add_mutually_exclusive_group(required= True)
    # group.add_argument("-p", "--path",
    #                 help="-p Sequences path")
    # group.add_argument("-v", "--video",
    #                    help="-v Video file")
    ap.add_argument("-p", "--path", required= True,
                    help="-p Sequences path")
    ap.add_argument("-r", "--results", required= False,
                    help="-r Results path")
    args = vars(ap.parse_args())

    # Configuration
    seq_idx = 0
    conf_seq = {0: '.jpg', 1: '.gif'}
    save = False
    box_size = (15, 15)
    N = 500

    # Get folders with the sequences
    sequences = tl.natSort(tl.getSequences(args["path"]))

    # Get the frames of the sequence and make results folder
    seq_selected = sequences[seq_idx]
    ext = conf_seq[seq_idx]
    name_sequence = basename(seq_selected)
    if args.get("results"):
        results_path = altsep.join((args["results"], name_sequence))
        if not exists(results_path) and save:
            tl.makeDir(results_path)

    frames = tl.natSort(tl.getSamples(seq_selected, ext))
    # print(frames)
    if ext == '.gif':
        img = Image.open(frames[0])
        img_rgb = img.convert('RGB')
        img_rgb = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
        binary = PFS.preprocess(img)

    elif ext == '.jpg':
        img = cv2.imread(frames[0])
        binary = PFS.preprocess(img)
        PFS.initialization(img, N, box_size)


    # for i, frame in enumerate(frames):
    #     if ext == '.gif':
    #         img = Image.open(frame)
    #         img_rgb = img.convert('RGB')
    #         img_rgb = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    #
    #     elif ext == '.jpg':
    #         img = cv2.imread(frame)
    #         binary = PFS.preprocess(img)
    #
    #     if save:
    #         result_name = altsep.join((results_path, ''.join((name_sequence, '-', i, '.png'))))

            # cv2.imwrite(result_name, result)