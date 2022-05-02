from __future__ import print_function

import math
import os
import stat
import subprocess
import sys

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.

# Select only one of the following dataset options
#dataset_name = "SKKU_TL" # SKKU dataset with scene frame coords.
#dataset_name = "KITTI_TL" # KITTI dataset with scene frame coords.
#dataset_name = "TL_Combined" # combined dataset with scene frame coords.
dataset_name = "SKKU_laneTL" # SKKU dataset with lane frame coords instead of scene frame coords.
#dataset_name = "SKKU_carTL" # SKKU dataset with car frame coords instead of scene frame coords.
#dataset_name = "KITTI_carTL" # KITTI dataset with car frame coords instead of scene frame coords.

caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, 'python')

import caffe
from caffe.model_libs import *
from caffe.proto import caffe_pb2

# Add extra layers on top of a "base" network
def AddExtraLayers(net):
    ##  For ResNet-18
    ConvBNLayer(net, "res4b", "conv4_3r", False, True, 256, 3, 1, 1)
    ConvBNLayer(net, "res5b", "fc7r", False, True, 256, 3, 1, 1)
    ConvBNLayer(net, "res5b_relu", "conv6_1", False, True, 256, 1, 0, 1)

    from_layer = "conv6_1"
    out_layer = "conv6_2"
    ConvBNLayer(net, "conv6_1", "conv6_2", False, True, 256, 3, 1, 2)

    for i in xrange(7, 9):
      from_layer = out_layer
      out_layer = "conv{}_1".format(i)
      ConvBNLayer(net, from_layer, out_layer, False, True, 128, 1, 0, 1)

      from_layer = out_layer
      out_layer = "conv{}_2".format(i)
      ConvBNLayer(net, from_layer, out_layer, False, True, 256, 3, 1, 2)

    return net

rolling_time = 4 # number of rolling operations to perform per iteration
branch_num = 4
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

# The database file for training data. Created by data/{dataset_name}/create_data.sh
train_data = "data/{}/lmdb/{}_training_lmdb".format(dataset_name,dataset_name)
# The database file for testing data. Created by data/{dataset_name}/create_data.sh
test_data = "data/{}/lmdb/{}_testing_lmdb".format(dataset_name,dataset_name)

# If true, do batch sampling on images.
sampling_enabled = False

# Specify the batch sampler.

# working values for resize_width: 246, 1024, 1272, 1280, 2560.
# working values for resize_height: 246, 500, 504, 756, 768, 1280.
if dataset_name == "KITTI_TL":
    resize_width = 2560
    resize_height = 768
elif dataset_name == "TL_Combined":
    resize_width = 2560
    resize_height = 1280
elif dataset_name == "KITTI_carTL" or dataset_name == "SKKU_carTL":
    resize_width = 768
    resize_height = 768
elif dataset_name == "SKKU_laneTL":
    resize_width = 1024
    resize_height = 500
else: # SKKU_TL or other dataset
    resize_width = 1280
    resize_height = 768

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
learning_rate = 0.0005

model_name = "Res_{}_RRC_{}x{}".format(dataset_name,resize_width, resize_height) # The name of the model. Modify it if you want.
save_dir = "models/ResNet/{}/RRC_{}x{}".format(dataset_name,resize_width, resize_height) # Directory which stores the model .prototxt and snapshot of models.
job_dir = "jobs/ResNet/{}/RRC_{}x{}".format(dataset_name,resize_width, resize_height) # Directory which stores the job script and log file.

# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)

snapshot_prefix = "{}/{}".format(save_dir, model_name) # snapshot prefix.
job_file = "{}/{}.sh".format(job_dir, model_name) # job script path.

# Stores the test image names and sizes. Created by data/{dataset_name}/create_list.sh
name_size_file = "data/{}/testing_name_size.txt".format(dataset_name)
pretrain_model = "models/ResNet/ResNet-18.caffemodel" # The pretrained model.

label_map_file = "data/{}/labelmap_voc.prototxt".format(dataset_name) # Stores LabelMapItem.

num_outputs=[256,256,256,256,256]
rolling_rate = 0.075

# Solver parameters.
# Defining which GPUs to use.
gpus = "0,1,2,3"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 1
accum_batch_size = 8
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

# MultiBoxLoss parameters.
num_classes = 8
background_label_id=0
code_type = P.PriorBox.CENTER_SIZE
if sampling_enabled:
    batch_sampler = [
        {
                'sampler': {},
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
                'sample_constraint': {'min_jaccard_overlap': 0.1, },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': { 'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
                'sample_constraint': { 'min_jaccard_overlap': 0.3, },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': { 'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
                'sample_constraint': { 'min_jaccard_overlap': 0.5, },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': { 'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
                'sample_constraint': { 'min_jaccard_overlap': 0.7, },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': { 'min_scale': 0.3, 'max_scale': 1.0,  'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
                'sample_constraint': { 'min_jaccard_overlap': 0.9, },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': { 'min_scale': 0.3, 'max_scale': 1.0,  'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
                'sample_constraint': { 'max_jaccard_overlap': 1.0, },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]
else:
    batch_sampler = [
        {
                'sampler': {},
                'max_trials': 1,
                'max_sample': 1,
        },
        ]
train_transform_param = {
        'mirror': True,
        'hsv': True,
        'gaussianblur': True,
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }

multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': 2.,
    'num_classes': num_classes,
    'share_location': True,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.7,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': True,
    'do_neg_mining': True,
    'neg_pos_ratio': 3.,
    'neg_overlap': 0.5,
    'code_type': code_type,
    }
loss_param = {
    'normalization': P.Loss.VALID,
    }

# parameters for generating priors.
# minimum dimension of input image
min_dim = min(resize_width,resize_height) 

mbox_source_layers = ['conv4_3r', 'fc7r', 'conv6_2', 'conv7_2', 'conv8_2']
# in percent %
min_ratio = 15
max_ratio = 85
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 6.7 / 100.] + min_sizes
max_sizes = [[]] + max_sizes
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3]]

# variance used to encode/decode prior bboxes.
# must be set to an array in accordance to P.PriorBox.CENTER_SIZE.
prior_variance = [0.1, 0.1, 0.2, 0.2]

flip = True
clip = True

# Evaluate on whole test set.
num_test_image = 3575
test_batch_size = 1
test_iter = num_test_image / test_batch_size

solver_param = {
    # Train parameters
    'base_lr': learning_rate,
    'weight_decay': 0.0005,
    'lr_policy': "step",
    'stepsize': 25000,
    'gamma': 0.5,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 150000,
    'snapshot': 1000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 1200000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }

# parameters for generating detection output.
det_out_param = {
    'num_classes': num_classes,
    'share_location': True,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    'keep_top_k': 200,
    'confidence_threshold': 0.001,
    'code_type': code_type,
    }

# Check files.
os.path.exists(train_data)
os.path.exists(test_data)
os.path.exists(label_map_file)
os.path.exists(pretrain_model)

# Make directories if they do not exist.
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(job_dir):
    os.makedirs(job_dir)

##########################Create train net.########################################
net = caffe.NetSpec()
net.data, net.label = L.AnnotatedData(name="data",
    annotated_data_param=dict(label_map_file=label_map_file,batch_sampler=batch_sampler),
    data_param=dict(batch_size=batch_size_per_device, backend=P.Data.LMDB, source=train_data),
    ntop=2, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), transform_param=train_transform_param)

ResNet18Body(net, from_layer='data', use_pool5=False)
AddExtraLayers(net)

mbox_layers = CreateMultiBoxHead_share_2x(net, data_layer='data', from_layers=mbox_source_layers,
        min_sizes=min_sizes, max_sizes=max_sizes, aspect_ratios=aspect_ratios, num_classes=num_classes, flip=flip, clip=clip,
        prior_variance=prior_variance, branch_num=branch_num)

mbox_layers.append(net.label)
net["mbox_loss"] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        loss_param=loss_param, propagate_down=[True, True, False, False])

for roll_idx in range(1,rolling_time+1):
    roll_layers = CreateRollingStruct(net,from_layersbasename=mbox_source_layers,num_outputs=num_outputs,rolling_rate=rolling_rate,roll_idx=roll_idx)
    mbox_layers = CreateMultiBoxHead_share_2x_without_prior(net, from_layers=roll_layers, max_sizes=max_sizes, aspect_ratios=aspect_ratios, num_classes=num_classes, flip=flip,
            layers_names=mbox_source_layers, roll_idx=roll_idx,branch_num=branch_num)
    mbox_layers.append(net["mbox_priorbox"])

    mbox_layers.append(net.label)
    net["mbox_loss%d"%(roll_idx+1)] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
            loss_param=loss_param, propagate_down=[True, True, False, False])

with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

#############################Create test net.################################
net = caffe.NetSpec()

net.data, net.label = L.AnnotatedData(name="data",
    annotated_data_param=dict(label_map_file=label_map_file,batch_sampler=[{}]),
    data_param=dict(batch_size=test_batch_size, backend=P.Data.LMDB, source=test_data),
    ntop=2, include=dict(phase=caffe_pb2.Phase.Value('TEST')), transform_param=test_transform_param)

ResNet18Body(net, from_layer='data', use_pool5=False)
AddExtraLayers(net)

data_layer = 'data'
from_layers=mbox_source_layers
Num = len(from_layers)
priorbox_layers = []
min_sizes_2x = []
max_sizes_2x = []

#  calculate the priorbox scales
for i in range(0,len(min_sizes)-1):
    for j in range(0,branch_num):
        min_sizes_2x.append(min_sizes[i] + j * (min_sizes[i+1] - min_sizes[i])/branch_num)
min_sizes_2x.append(min_sizes[-1])
for i in range(0,len(max_sizes)-1):
    if not(max_sizes[i]):
        for j in range(0,branch_num):
            max_sizes_2x.append([])
    else:
        for j in range(1,branch_num+1):
            max_sizes_2x.append(min_sizes_2x[branch_num*i + j])
max_sizes_2x.append(max_sizes[-1])

# Add L2 Normalization layer on conv4_3.
from_layer = mbox_source_layers[0]
norm_name = "{}_norm".format(from_layer)
net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=20), across_spatial=False, channel_shared=False)

#  create multi priorbox layer
for i in range(0,Num):
    from_layer = from_layers[i]

    aspect_ratio = []
    if len(aspect_ratios) > i:
        aspect_ratio = aspect_ratios[i]

    if (i == Num - 1):
        repeat_times = 1
    else:
        repeat_times = branch_num

    for mbox_idx in range(0,repeat_times):
          name = "{}_mbox_priorbox{}".format(from_layer,mbox_idx)
          if max_sizes and max_sizes[i]:
              if aspect_ratio:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], 
                                        max_size=max_sizes_2x[branch_num*i + mbox_idx],aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
              else:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], 
                                        max_size=max_sizes_2x[branch_num*i + mbox_idx],clip=clip, variance=prior_variance)
          else:
              if aspect_ratio:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                         aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
              else:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                         clip=clip, variance=prior_variance)
          priorbox_layers.append(net[name])

net["mbox_priorbox"] = L.Concat(*priorbox_layers, axis=2)
rolling_time = 2
#==============================================================================    
for roll_idx in range(1,rolling_time+1):
    roll_layers = CreateRollingStruct(net,from_layersbasename=mbox_source_layers,num_outputs=num_outputs,rolling_rate=rolling_rate,roll_idx=roll_idx)       
    mbox_layers = CreateMultiBoxHead_share_2x_without_prior(net, from_layers=roll_layers,
            max_sizes=max_sizes, aspect_ratios=aspect_ratios, num_classes=num_classes, flip=flip,
            layers_names=mbox_source_layers,roll_idx=roll_idx,branch_num=branch_num)
    mbox_layers.append(net["mbox_priorbox"])

    conf_name = "mbox_conf%d"%(roll_idx+1)

    reshape_name = "{}_reshape".format(conf_name)
    net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
    softmax_name = "{}_softmax".format(conf_name)
    net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
    flatten_name = "{}_flatten".format(conf_name)
    net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
    mbox_layers[1] = net[flatten_name]
    
    net['detection_out%d'%(roll_idx+1)] = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param, include=dict(phase=caffe_pb2.Phase.Value('TEST')))
#==============================================================================

with open(test_net_file, 'w') as f:
    net_param = net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]    
    net_param.name = '{}_test'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(save_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(save_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(save_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(save_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)

