import numpy as np
import timeit
import time
from PIL import Image
from PIL import ImageDraw

from os.path import expanduser
import glob
import cv2

from CarTracker import *
###################
tracking_enabled = True
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

# modify img_dir to your path of testing images of cars
img_dir = '{}/data/{}/{}ing/image_2/'.format(home_dir,testing_dataset_name,validation_mode)


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
result_img_dir = model_dir+'result-{}_{}_images/'.format(validation_mode,testing_dataset_name) # previously named as 'save_dir'
result_txt_dir = model_dir+'result-{}_{}/'.format(validation_mode,testing_dataset_name) # previously named as 'txt_dir'

detection_out_num = 3
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

img_file_list = glob.glob("{}/*.{}".format(img_dir,image_format))
img_file_list.sort()
num_img = len(img_file_list)

if tracking_enabled:
    trackers = []

for img_file in img_file_list:
    det_total = np.zeros([0,6],float)
    ensemble_num = 0
    img_idx = img_file.split('/')[-1].split('.')[0]
    #print 'processing image #{}...'.format(img_idx)

    result_file = open(result_txt_dir+"{}.txt".format(img_idx),'w')
    img = Image.open(img_dir + "{}.{}".format(img_idx,image_format))
    draw = ImageDraw.Draw(img)
    bbox_color_array = [(255,255,0),(0,255,0),(0,255,255)]

    if tracking_enabled:
        # Car detection from tracking is set to have higher priority than detection from net forwarding.
        # For each image, if a car is detected from tracking, net forwarding step is skipped.
        # This greatly reduces computation time, and can compensate cases
        # when the bounding box of a car fails to be detected from net forwarding
        # (including cases when confidence score of the box is too low)
        cv_image_input = cv2.imread(img_file)
        cv_image_input_height = 0
        cv_image_input_width = 0

        # For a tracker to work properly, a series of input images must have same size.
        # When the input images are ego-lane images, we re-pad them to match with full-scene frame.
        if training_dataset_name == "SKKU_laneCar":
            cv_image_input_height, cv_image_input_width = cv_image_input.shape[:2]
            cv_image_input = cv2.copyMakeBorder( cv_image_input, 672 - cv_image_input_height, 0, 0, 1280 - cv_image_input_width, cv2.BORDER_CONSTANT)

        hasValidTrackingRegion = False
        for tracker in trackers:
            startTime = time.time()
            ok, bbox = tracker.updateTracker(cv_image_input)
            current_time = time.time() - startTime
            #print current_time
            total_time += current_time
            if ok:
                print(bbox)
                # the bbox coords are in tracker's point of view (i.e. full-scene frame).
                # We adjust them to the ego-lane frame.
                if training_dataset_name == "SKKU_laneCar":
                    xmin_tracker = np.max([bbox[0], 0])
                    ymin_tracker = bbox[1]
                    xmax_tracker = bbox[0]+bbox[2]
                    ymax_tracker = bbox[1]+bbox[3]

                    xmin = xmin_tracker
                    ymin = ymin_tracker - (672 - cv_image_input_height)
                    xmax = xmax_tracker
                    ymax = ymax_tracker - (672 - cv_image_input_height)
                else:
                    xmin = np.max([bbox[0], 0])
                    ymin = bbox[1]
                    xmax = bbox[0]+bbox[2]
                    ymax = bbox[1]+bbox[3]

                score = 2.0 # A non-standard value for confidence score. But this is used for indicating that given result is from tracking.
                label = 'Car'
                result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))

                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(255,0,0),width=5)
            else:
                del tracker
            hasValidTrackingRegion = hasValidTrackingRegion or ok
        if hasValidTrackingRegion:
            img.save(result_img_dir+"{}.png".format(img_idx))
            continue

    image = caffe.io.load_image(img_file)
    
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

        det_this = np.concatenate((top_xmin.reshape(-1,1),top_ymin.reshape(-1,1),
                                   top_xmax.reshape(-1,1),top_ymax.reshape(-1,1),
                                   top_conf.reshape(-1,1),det_label[top_indices].reshape(-1,1)),1)
        ensemble_num = ensemble_num + 1
        det_total = np.concatenate((det_total,det_this),0)

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
        #if (h/w >=2)and((xmin<50)or(xmax > 1230)):
        #    continue

        # 2. cars that are too big or are too close to the driver are discarded.
        #if (w >= 0.6*image_width) or (h >= 0.6*image_height):
        #    continue

        # 2-1. cars that are too far away are discarded.
        #if (w < 0.06*image_width) and (h < 0.06*image_height):
        #    continue

        # 3. cars with low confidence scores are discarded.
        #if top_conf[i] < 0.1:
        #    continue

        score = top_conf[i]
        label = top_label[i]

        if tracking_enabled:
            cv_image_input = cv2.imread(img_file)
            tracker = CarTracker()

            # For a tracker to work properly, a series of input images must have same size.
            # When the input images are ego-lane images, we re-pad them to match with the original size of all-lane scene image.
            # The bbox coords in the tracker's point of view is calibrated accordingly.
            if training_dataset_name == "SKKU_laneCar":
                cv_image_input_height, cv_image_input_width = cv_image_input.shape[:2]
                cv_image_input = cv2.copyMakeBorder( cv_image_input, 672 - cv_image_input_height, 0, 0, 1280 - cv_image_input_width, cv2.BORDER_CONSTANT)
                xmin_tracker = xmin
                ymin_tracker = ymin + (672 - cv_image_input_height)
                xmax_tracker = xmax
                ymax_tracker = ymax + (672 - cv_image_input_height)

                tracker.initializeTracker(cv_image_input, (xmin_tracker, ymin_tracker, xmax_tracker-xmin_tracker, ymax_tracker-ymin_tracker), "TLD", score)
            else:
                tracker.initializeTracker(cv_image_input, (xmin, ymin, xmax-xmin, ymax-ymin), "MEDIANFLOW", score)
            trackers.append(tracker)

        if score > 0.1: # sufficiently confident detection
            draw.line(((xmin+1,ymin+1),(xmin+1,ymax+1),(xmax+1,ymax+1),(xmax+1,ymin+1),(xmin+1,ymin+1)),fill=(128,128,128),width=3)
            draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=bbox_color_array[1%3],width=3)
            draw.text((xmin+1,ymax+1),'%.2f'%(score), fill=(64,64,64))
            draw.text((xmin,ymax),'%.2f'%(score), fill=bbox_color_array[1%3])
            result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
        elif score > 0.02: # not so confident, but somehow a valid detection
            draw.line(((xmin+1,ymin+1),(xmin+1,ymax+1),(xmax+1,ymax+1),(xmax+1,ymin+1),(xmin+1,ymin+1)),fill=(128,128,128),width=3)
            draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=bbox_color_array[1%3],width=3)
            draw.text((xmin+1,ymax+1),'%.2f'%(score), fill=(64,64,64))
            draw.text((xmin,ymax),'%.2f'%(score), fill=bbox_color_array[2%3])
            result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
    img.save(result_img_dir+"{}.png".format(img_idx))

print("Total time = {}".format(total_time))
print("Average FPS = {}".format(num_img/total_time))
