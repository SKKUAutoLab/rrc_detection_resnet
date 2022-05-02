import numpy as np
import timeit
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from os.path import expanduser
import glob

home_dir = expanduser("~")

# Make sure that the work directory is caffe_root
caffe_root = '{}/rrc_detection_ResNet/'.format(home_dir)

# modify the name of the dataset to be used and its input size as needed.
# some input sizes may not work. currently, the following sizes have been tested:
# 768x768, 1024x500, 1280x768, 2560x768
dataset_name = "KITTI"
image_width = 2560
image_height = 768

# validation mode. either perform validation on training or testing set.
#validation_mode = 'train'
validation_mode = 'test'

# modify img_dir to your path of testing images of objects in traffic
img_dir = '{}/data/{}/{}ing/image_2/'.format(home_dir,dataset_name,validation_mode)

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

model_dir = 'models/ResNet/{}/RRC_{}x{}/'.format(dataset_name,image_width,image_height)
model_def = '{}/test.prototxt'.format(model_dir)
model_weights_basename = "Res_{}_RRC_{}x{}_iter".format(dataset_name,image_width,image_height)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(model_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_".format(model_weights_basename))[1])
    if iter > max_iter:
      max_iter = iter
model_weights = '{}/{}_{}.caffemodel'.format(model_dir, model_weights_basename, max_iter)

voc_labelmap_file = caffe_root+'data/'+dataset_name+'/labelmap_voc.prototxt'
result_img_dir = model_dir+'result-{}_images/'.format(validation_mode) # previously named as 'save_dir'
result_txt_dir = model_dir+'result-{}/'.format(validation_mode) # previously named as 'txt_dir'

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

img_file_list = glob.glob("{}/*.png".format(img_dir))
img_file_list.sort()
num_img = len(img_file_list)

for img_file in img_file_list:
    det_total = np.zeros([0,6],float)
    ensemble_num = 0
    img_idx = img_file.split('/')[-1].split('.')[0]
    print 'processing image #{}...'.format(img_idx)
    image = caffe.io.load_image(img_file)
    
    transformed_image = transformer.preprocess('data', image)
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
    idxs = np.where(det_results[:,4] >= 0.01)[0]
    top_xmin = det_results[idxs,0]
    top_ymin = det_results[idxs,1]
    top_xmax = det_results[idxs,2]
    top_ymax = det_results[idxs,3]
    top_conf = det_results[idxs,4]
    top_label = get_labelname(voc_labelmap,det_results[idxs,5])
    result_file = open(result_txt_dir+"{}.txt".format(img_idx),'w')
    img = Image.open(img_dir + "{}.png".format(img_idx))
    draw = ImageDraw.Draw(img)
    bbox_color_array = [(255,255,0),(0,255,0),(0,255,255)]
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed-Bold.ttf', 16)
    for i in xrange(top_conf.shape[0]):
        xmin = top_xmin[i]
        ymin = top_ymin[i]
        xmax = top_xmax[i]
        ymax = top_ymax[i]
        h = float(ymax - ymin)
        w = float(xmax - xmin)
        if (w==0) or (h==0):
           continue
        if (h/w >=2)and((xmin<10)or(xmax > 1230)):
           continue
        score = top_conf[i]
        label = top_label[i]
        if score > 0.1:
            draw.line(((xmin+1,ymin+1),(xmin+1,ymax+1),(xmax+1,ymax+1),(xmax+1,ymin+1),(xmin+1,ymin+1)),fill=(128,128,128),width=3)
            draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=bbox_color_array[i%3],width=3)
            draw.text((xmin,ymin),'%.2f'%(score),font=fnt,fill=bbox_color_array[i%3])
        elif score > 0.02:
            draw.line(((xmin+1,ymin+1),(xmin+1,ymax+1),(xmax+1,ymax+1),(xmax+1,ymin+1),(xmin+1,ymin+1)),fill=(128,128,128),width=3)
            draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=bbox_color_array[i%3],width=3)
            draw.text((xmin,ymin),'%.2f'%(score),font=fnt,fill=bbox_color_array[i%3])
        result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
    img.save(result_img_dir+"{}.png".format(img_idx))

print("Total time = {}".format(total_time))
print("Average FPS = {}".format(num_img/total_time))
