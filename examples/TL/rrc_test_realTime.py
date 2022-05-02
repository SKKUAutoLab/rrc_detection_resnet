import numpy as np
import timeit
from PIL import Image
from PIL import ImageDraw 

from os.path import expanduser
import glob

# Opposite-side TL detection -- enable this only if cars are 'perfectly' facing rear side.
opposite_enabled = False

home_dir = expanduser("~")

# Make sure that the work directory is caffe_root
caffe_root = '{}/rrc_detection_ResNet/'.format(home_dir)

# modify the name of the dataset to be used and its input size as needed.
# some input sizes may not work. currently, the following sizes have been tested:
# 768x768, 1024x500, 1280x768, 2560x768
training_dataset_name = "KITTI_carTL"
image_width = 768
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
caffe.set_device(1) # set this device number to zero if you only have 1 GPU
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
input_dir = '{}/tl_temp/car/'.format(home_dir)
result_img_dir = '{}/tl_temp/TL/'.format(home_dir)
result_txt_dir = '{}/tl_temp/TL_txt/'.format(home_dir)

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
num_img = 0
# Because we are processing in real-time within Detection Pipeline,
# input raw images are to be received with timestamp string as its filename.
fileQueue = []
os.system('clear')
while 1:
    #print 'waiting for car input image...'

    while len(fileQueue) == 0:
        fileQueue = glob.glob(input_dir+"/*.{}".format(image_format))
        fileQueue.sort()
    img_file = fileQueue[0] # the propagated image file
    coord_file = fileQueue[0].replace(image_format, 'txt') # txt file with detected car coords

    print ("received input {}".format(img_file))
    car_xmin, car_ymin, car_xmax, car_ymax = 0, 0, 0, 0
    lines = []
    with open(coord_file) as f:
        lines = f.readlines()
        if len(lines) == 0:
            print 'cars not detected. skip this frame.'
            if os.path.exists(img_file): os.remove(img_file)
            fileQueue.pop(0)
            continue
    img_idx = img_file.split('/')[-1].split('.')[0]
    result_file = open(result_txt_dir+"{}.txt".format(img_idx),'w')
    image = caffe.io.load_image(img_file)
    img = Image.open(img_file)
    draw = ImageDraw.Draw(img)
    for line in lines:
        tokens = line.strip().split(' ')
        car_xmin = max(float(tokens[4]), 0)
        car_ymin = max(float(tokens[5]), 0)
        car_xmax = float(tokens[6])
        car_ymax = float(tokens[7])

        det_total = np.zeros([0,6],float)
        num_img += 1
        ensemble_num = 0
        image_car = image[int(car_ymin):int(car_ymax), int(car_xmin):int(car_xmax), :] # crop propagated img according to coords
        transformed_image = transformer.preprocess('data', image_car)
        net.blobs['data'].data[...] = transformed_image

        t1 = timeit.Timer("net.forward()","from __main__ import net")
        current_time = t1.timeit(2)
        print current_time
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
            top_xmin = det_xmin[top_indices]* image_car.shape[1]
            top_ymin = det_ymin[top_indices]* image_car.shape[0]
            top_xmax = det_xmax[top_indices]* image_car.shape[1]
            top_ymax = det_ymax[top_indices]* image_car.shape[0]

            det_this = np.concatenate((top_xmin.reshape(-1,1),top_ymin.reshape(-1,1),
                                    top_xmax.reshape(-1,1),top_ymax.reshape(-1,1),
                                    top_conf.reshape(-1,1),det_label[top_indices].reshape(-1,1)),1)

            ensemble_num = ensemble_num + 1
            det_total = np.concatenate((det_total,det_this),0)
    #   evaluate the flipped image
        image_car_flip = image_car[:,::-1,:]
        transformed_image = transformer.preprocess('data', image_car_flip)
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
            top_xmin = det_xmin[top_indices]* image_car.shape[1]
            top_ymin = det_ymin[top_indices]* image_car.shape[0]
            top_xmax = det_xmax[top_indices]* image_car.shape[1]
            top_ymax = det_ymax[top_indices]* image_car.shape[0]

            det_this = np.concatenate((top_xmin.reshape(-1,1),top_ymin.reshape(-1,1),
                                    top_xmax.reshape(-1,1),top_ymax.reshape(-1,1),
                                    top_conf.reshape(-1,1),det_label[top_indices].reshape(-1,1)),1)
            ensemble_num = ensemble_num + 1
            det_total = np.concatenate((det_total,det_this),0)

        if opposite_enabled:
            det_opposite = det_total.copy()
            det_opposite[:,0] = (car_xmax-car_xmin) - det_total[:,2]
            det_opposite[:,2] = (car_xmax-car_xmin) - det_total[:,0]
            det_total = np.concatenate((det_total,det_opposite),0)
            ensemble_num = ensemble_num * 2

        #ensemble different outputs
        det_results = det_ensemble(det_total,ensemble_num)

        # this sorting is mandatory if we are to deal with low-score TL blobs properly in the code afterwards.
        #det_results = det_results[np.flip(det_results[:,4].argsort())]

        idxs = np.where(det_results[:,4] > 0.0001)[0]
        top_xmin = det_results[idxs,0]
        top_ymin = det_results[idxs,1]
        top_xmax = det_results[idxs,2]
        top_ymax = det_results[idxs,3]
        top_conf = det_results[idxs,4]
        top_label = get_labelname(voc_labelmap,det_results[idxs,5])

        for i in xrange(top_conf.shape[0]):
            xmin = top_xmin[i] + car_xmin
            ymin = top_ymin[i] + car_ymin
            xmax = top_xmax[i] + car_xmin
            ymax = top_ymax[i] + car_ymin
            h = float(ymax - ymin)
            w = float(xmax - xmin)
            if (w==0) or (h==0):
                continue

            # Depending on nation-wise legislations, there exists minimum rear-face ground clearance
            # which is measured as 0.08~0.14 meters. Here, overall height of passenger cars are about 1.5 meters.
            # Based on this, you may choose to apply the following condition for removing false positives if you want.
            if (top_ymin[i] >= (car_ymax-car_ymin) * 0.9):
                continue

            # Some cars have multiple pairs of taillight blobs (main and sub blobs), but it is best to detect main blobs only.
            # Conditional statement for skipping sub blobs based on vertical position goes here, if ever necessary.
            if (top_ymin[i] >= (car_ymax-car_ymin) * 0.4):
                continue

            score = top_conf[i]
            label = top_label[i]

            if score >= 0.1: # sufficiently confident detection
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(0,255,0))
                draw.text((xmin,ymin),'%.2f'%(score),fill=(255,255,255))
            elif score >= 0.005: # not so confident, but somehow a valid detection
                draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(255,0,255))
                draw.text((xmin,ymin),'%.2f'%(score),fill=(255,255,255))
            result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
    result_file.close()
    img.save(result_img_dir+"{}.png".format(img_idx))
    if os.path.exists(img_file): os.remove(img_file)
    fileQueue.pop(0)
    if total_time > 0:
        print("Average FPS = {}\r".format(num_img/total_time))
