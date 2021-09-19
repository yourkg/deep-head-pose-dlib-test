import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
from six.moves import xrange
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils

from skimage import io
import dlib

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='hopenet_robust_alpha1.pkl', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='mmod_human_face_detector.dat', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video',default='test.mp4')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file',default="outvidieo")
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames',default=2600, type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([transforms.Resize(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(video_path)

    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, args.fps, (width, height))

    # # Old cv2
    # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

    txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

    frame_num = 100

    while frame_num <= args.n_frames:
        print(frame_num)

        ret,frame = video.read()
        print("2")
        if ret == False:
            break
        #cv2.imshow('testim',frame)
        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        print(3)
        # Dlib detect
        dets =cnn_face_detector(cv2_frame, 1)
        #print(dets)
        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            print("->1")
            face = det.rect
            cv2.rectangle(frame, (face.left(),face.top()), (face.right(),face.bottom()), (0,0,255),2)
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence
            print("->2")
            if conf > 0.5:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                print("->3",y_min,y_max,x_min,x_max)
                # Crop image
                img = cv2_frame[int(y_min):int(y_max),int(x_min):int(x_max)]
                img = Image.fromarray(img)
                print("->4")
                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)
                print("->5")
                yaw, pitch, roll = model(img)
                print("->6")
                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                print("->7")
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                print("->8")
                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                print("->9")
                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                print("->10")
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                print("->11")
                # Plot expanded bounding box
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

        print("->12")
        out.write(frame)
        print("->13")
        frame_num += 1

    out.release()
    video.release()
