import numpy as np
import timeit
import shutil
from PIL import Image

from os.path import expanduser
import glob
import cv2

from CarTracker import *
###################
tracking_enabled = True
laneCut_enabled = True
###################

home_dir = expanduser("~")

# Make sure that the work directory is caffe_root
caffe_root = '{}/rrc_detection_ResNet/'.format(home_dir)

# modify the name of the dataset to be used and its input size as needed.
# some input sizes may not work. currently, the following sizes have been tested:
# 768x768, 1024x500, 1280x768, 2560x768
training_dataset_name = "KITTI"
image_width = 2560
image_height = 768
image_format = "png"

# validation mode. either perform validation on training or testing set.
#validation_mode = 'train'
validation_mode = 'test'

testing_dataset_name = "KITTI_Tracking"

import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
from google.protobuf import text_format
from caffe.proto import caffe_pb2

import caffe
from _ensemble import *
caffe.set_device(0)
caffe.set_mode_gpu()

model_dir = 'models/ResNet/{}/RRC_{}x{}/'.format(training_dataset_name,image_width,image_height)
model_def = '{}/test.prototxt'.format(model_dir)
model_weights_basename = "Res_{}_RRC_{}x{}_iter".format(training_dataset_name,image_width,image_height)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(model_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_".format(model_weights_basename))[1])
    if iter > max_iter:
      max_iter = iter
model_weights = '{}/{}_{}.caffemodel'.format(model_dir, model_weights_basename, max_iter)

voc_labelmap_file = caffe_root+'data/'+training_dataset_name+'/labelmap_voc.prototxt'
input_dir = '{}/tl_temp/lane/'.format(home_dir)
result_img_dir = '{}/tl_temp/car/'.format(home_dir) # previously named as 'save_dir'
result_txt_dir = '{}/tl_temp/car/'.format(home_dir)

detection_out_num = 3
if not(os.path.exists(input_dir)):
    os.makedirs(input_dir)
if not(os.path.exists(result_txt_dir)):
    os.makedirs(result_txt_dir)
if not(os.path.exists(result_img_dir)):
    os.makedirs(result_img_dir)
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap) 
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
net.blobs['data'].reshape(1,3,image_height,image_width)
    
def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if (type(labels) is not np.ndarray) and (type(labels) is not list):
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

total_time = 0

if tracking_enabled:
    trackers = []

# Because we are processing in real-time within Detection Pipeline,
# input raw images are to be received with timestamp string as its filename.
fileQueue = []
os.system('clear')
while 1:
    #print 'waiting for camera input image...'

    while len(fileQueue) == 0:
        fileQueue = glob.glob(input_dir+"/*.{}".format(image_format))
        fileQueue.sort()
    img_file = fileQueue[0] # the propagated image file
    coord_file = fileQueue[0].replace(image_format, 'bmp') # we assume lane regions are output as bitmap

    print ("received input {}".format(img_file))
    det_total = np.zeros([0,6],float)
    ensemble_num = 0
    img_idx = img_file.split('/')[-1].split('.')[0]
    #print 'processing image #{}...'.format(img_idx)

    result_file = open(result_txt_dir+"{}.txt".format(img_idx),'w')

    if tracking_enabled:
        # Car detection from tracking is set to have higher priority than detection from net forwarding.
        # For each image, if a car is detected from tracking, net forwarding step is skipped.
        # This greatly reduces computation time, and can compensate cases
        # when the bounding box of a car fails to be detected from net forwarding
        # (including cases when confidence score of the box is too low)
        cv_image_input = cv2.imread(img_file)
        hasValidTrackingRegion = False
        for tracker in trackers:
            try:
                ok, bbox = tracker.updateTracker(cv_image_input)
            except:
                ok = False
                bbox = None
            if ok:
                xmin = np.max([bbox[0], 0])
                ymin = bbox[1]
                xmax = bbox[0]+bbox[2]
                ymax = bbox[1]+bbox[3]
                score = 2.0 # A non-standard value for confidence score. But this is used for indicating that given result is from tracking.
                label = 'Car'
                result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
            else:
                del tracker
            hasValidTrackingRegion = hasValidTrackingRegion or ok
        if hasValidTrackingRegion:
            shutil.move(img_file, img_file.replace('lane','car'))
            #if os.path.exists(coord_file): os.remove(coord_file)
            fileQueue.pop(0)
            continue
    # we pass this point when tracking fails to keep track of the ego-lane car.

    image = caffe.io.load_image(img_file)

    # lane region file may not exist at this point, but this is normal operation
    # since lane detection module can fail to detect ego-lane region.
    if laneCut_enabled and os.path.exists(coord_file):
        lane_region = cv2.imread(coord_file)
        image = cv2.bitwise_and(image, image, mask=lane_region[:,:,0])

    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image

    t1 = timeit.Timer("net.forward()","from __main__ import net")
    current_time = t1.timeit(2)
    #print current_time
    total_time += current_time

    # Forward pass.
    net_out = net.forward()
    for out_i in range(2,detection_out_num + 1):
        detections = net_out['detection_out%d'%(out_i)].copy()

        if detections.shape[2] == 0:
            ensemble_num = ensemble_num + 1
            continue

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.001
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.001]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(voc_labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]* image.shape[1]
        top_ymin = det_ymin[top_indices]* image.shape[0]
        top_xmax = det_xmax[top_indices]* image.shape[1]
        top_ymax = det_ymax[top_indices]* image.shape[0]

        # Sometimes the Resnet-RRC may go 'nuts' and claim that ego-lane cars exist outside ego-lane region.
        # We prevent this here, by computing ratio of non-zero pixels to total number of pixels.
        if laneCut_enabled:
            validCount = len(top_indices)
            for i in range(len(top_indices)):
                if np.count_nonzero(image[int(top_ymin[i]):int(top_ymax[i]),int(top_xmin[i]):int(top_xmax[i]),:]) / float(top_ymax[i]-top_ymin[i]) / float(top_xmax[i]-top_xmin[i]) <= 0.001:
                    print("This detection at {} {} {} {} seems to reside outside ego-lane region. skip.".format(top_xmin[i], top_ymin[i], top_xmax[i], top_ymax[i]))
                    validCount -= 1
        
        if validCount == 0:
            print("No cars seem to be in ego-lane.")
            ensemble_num = ensemble_num + 1
            continue
        det_this = np.concatenate((top_xmin.reshape(-1,1),top_ymin.reshape(-1,1),
                                   top_xmax.reshape(-1,1),top_ymax.reshape(-1,1),
                                   top_conf.reshape(-1,1),det_label[top_indices].reshape(-1,1)),1)

        ensemble_num = ensemble_num + 1
        det_total = np.concatenate((det_total,det_this),0)
    # evaluate the flipped image
    image_flip = image[:,::-1,:]
    transformed_image = transformer.preprocess('data', image_flip)
    net.blobs['data'].data[...] = transformed_image
    net_out = net.forward()
    for out_i in range(2,detection_out_num + 1):
        detections = net_out['detection_out%d'%(out_i)].copy()

        if detections.shape[2] == 0:
            ensemble_num = ensemble_num + 1
            continue

        temp = detections[0,0,:,3].copy()
        detections[0,0,:,3] = 1-detections[0,0,:,5]
        detections[0,0,:,5] = 1-temp

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.1.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(voc_labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]* image.shape[1]
        top_ymin = det_ymin[top_indices]* image.shape[0]
        top_xmax = det_xmax[top_indices]* image.shape[1]
        top_ymax = det_ymax[top_indices]* image.shape[0]

        # Sometimes the Resnet-RRC may go 'nuts' and claim that ego-lane cars exist outside ego-lane region.
        # We prevent this here, by computing ratio of non-zero pixels to total number of pixels.
        if laneCut_enabled:
            validCount = len(top_indices)
            for i in range(len(top_indices)):
                if np.count_nonzero(image[int(top_ymin[i]):int(top_ymax[i]),int(top_xmin[i]):int(top_xmax[i]),:]) / float(top_ymax[i]-top_ymin[i]) / float(top_xmax[i]-top_xmin[i]) <= 0.001:
                    print("This detection at {} {} {} {} seems to reside outside ego-lane region. skip.".format(top_xmin[i], top_ymin[i], top_xmax[i], top_ymax[i]))
                    validCount -= 1
        
        if validCount == 0:
            print("No cars seem to be in ego-lane.")
            ensemble_num = ensemble_num + 1
            continue

        det_this = np.concatenate((top_xmin.reshape(-1,1),top_ymin.reshape(-1,1),
                                   top_xmax.reshape(-1,1),top_ymax.reshape(-1,1),
                                   top_conf.reshape(-1,1),det_label[top_indices].reshape(-1,1)),1)
        ensemble_num = ensemble_num + 1
        det_total = np.concatenate((det_total,det_this),0)

    if det_total.shape[0] == 0:
        # Cars NOT detected! we roll back the lane cut and try again.
        laneCut_enabled = False
        continue
        
    #ensemble different outputs
    det_results = det_ensemble(det_total,ensemble_num)
    idxs = np.where(det_results[:,4] > 0.0001)[0]
    top_xmin = det_results[idxs,0]
    top_ymin = det_results[idxs,1]
    top_xmax = det_results[idxs,2]
    top_ymax = det_results[idxs,3]
    top_conf = det_results[idxs,4]
    top_label = get_labelname(voc_labelmap,det_results[idxs,5])

    if tracking_enabled:
        trackers = []

    for i in xrange(top_conf.shape[0]):
        xmin = top_xmin[i]
        ymin = top_ymin[i]
        xmax = top_xmax[i]
        ymax = top_ymax[i]
        h = float(ymax - ymin)
        w = float(xmax - xmin)
        if (w==0) or (h==0):
            continue

        # If you want, you may add some criteria necessary to eliminate false positives below.
        # Un-comment the conditionals and/or adjust parameters, and add new ones if you like.

        # 1. cars that are mostly outside of view are discarded.
        if (xmin<image.shape[1]*0.05)or(xmax > image.shape[1]*0.95):
            continue

        # 2. cars that are too big or are too close to the driver are discarded.
        #if (w >= 0.6*image_width) or (h >= 0.6*image_height):
        #    continue

        score = top_conf[i]
        label = top_label[i]

        if tracking_enabled:
            cv_image_input = cv2.imread(img_file)
            tracker = CarTracker()

            tracker.initializeTracker(cv_image_input, (xmin, ymin, xmax-xmin, ymax-ymin), "MEDIANFLOW", score)
            trackers.append(tracker)

        if score > 0.02:
            result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
        else:
            result_file.write('')
    result_file.close()
    shutil.move(img_file, img_file.replace('lane','car'))
    #if os.path.exists(coord_file): os.remove(coord_file)
    laneCut_enabled = True
    fileQueue.pop(0)

