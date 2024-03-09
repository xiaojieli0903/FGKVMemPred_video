# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import av
import cv2

import os
import subprocess
import sys
from joblib import Parallel, delayed

prefix = sys.argv[1]
mode = sys.argv[2]
file_src = sys.argv[3]
older_path = f'{prefix}/{mode}/'
output_path = f'{prefix}/{mode}_256/'

file_list = []

f = open(file_src, 'r')

for line in f:
    rows = line.split(' ')
    fname = rows[0]
    file_list.append(fname)

f.close()


def downscale_clip(inname, outname):
    status = False
    inname = '"%s"' % inname
    outname = '"%s"' % outname
    command = "ffmpeg  -loglevel panic -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {}".format(
        inname, outname)
    try:
        output = subprocess.check_output(command,
                                         shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        print('resize error: ', err.output)
        return status, err.output

    status = os.path.exists(outname)
    return status, 'Downscaled'


def downscale_clip_wrapper(row):

    nameset = row.split('/')
    videoname = nameset[-1].split(' ')[0]

    output_folder = output_path
    if os.path.isdir(output_folder) is False:
        try:
            os.mkdir(output_folder)
        except:
            print(f'{output_folder} exists')

    inname = folder_path + '/' + videoname
    outname = output_path + '/' + videoname
    if os.path.exists(outname):
        try:
            av.open(outname)
            cap = cv2.VideoCapture(outname)
            flag, frame = cap.read()
            assert flag
            return True
        except:
            os.unlink(outname)
            print(f'resizing {inname} again')
    downscaled, log = downscale_clip(inname, outname)
    print(outname, downscaled)
    downscaled = True
    return downscaled


status_lst = Parallel(n_jobs=16)(delayed(downscale_clip_wrapper)(row)
                                 for row in file_list)
