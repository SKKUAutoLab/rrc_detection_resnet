import numpy as np
from caffe import layers as L
from caffe import params as P
import copy

def ConvBNLayer(net, from_layer, out_layer, use_bn, use_relu, num_output,
    kernel_size, pad, stride, share_weight = ''):
  if use_bn:
    # parameters for convolution layer with batchnorm.
    kwargs = {
        'param': [dict(lr_mult=1, decay_mult=1)],
        'weight_filler': dict(type='gaussian', std=0.01),
        'bias_term': False,
        }
    # parameters for scale bias layer after batchnorm.
    sb_kwargs = {
        'bias_term': True,
        'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=1, decay_mult=0)],
        'filler': dict(type='constant', value=1.0),
        'bias_filler': dict(type='constant', value=0.0),
        }
  else:
    if share_weight == '':
      kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }
    else:   
      kwargs = {
            'param': [dict(name='%s_w'%(share_weight),lr_mult=1, decay_mult=1), dict(name='%s_b'%(share_weight),lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)
            }
  conv_name = out_layer
  net[conv_name] = L.Convolution(net[from_layer], num_output=num_output,kernel_size=kernel_size, pad=pad, stride=stride, **kwargs)
  if use_bn:
    bn_name = '{}_bn'.format(out_layer)
    net[bn_name] = L.BatchNorm(net[conv_name], in_place=True, eps=0.001,
        param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)])
    sb_name = '{}_scale'.format(out_layer)
    net[sb_name] = L.Scale(net[bn_name], in_place=True, **sb_kwargs)
  if use_relu:
    relu_name = '{}_relu'.format(conv_name)
    net[relu_name] = L.ReLU(net[conv_name], in_place=True)

def ResBody(net, from_layer, block_name, out2a, out2b, out2c, stride, use_branch1):

  if use_branch1:
    branch1 = 'res{}_{}'.format(block_name,'branch1')
    ConvBNLayer(net, from_layer, branch1, use_bn=True, use_relu=False,num_output=out2c, kernel_size=1, pad=0, stride=stride)
  else:
    branch1 = from_layer

  out_name = 'res{}_{}'.format(block_name,'branch2a')
  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,num_output=out2a, kernel_size=1, pad=0, stride=stride)
  from_layer = out_name

  out_name = 'res{}_{}'.format(block_name,'branch2b')
  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True,num_output=out2b, kernel_size=3, pad=1, stride=1)
  from_layer = out_name

  branch2 = 'res{}_{}'.format(block_name,'branch2c')
  ConvBNLayer(net, from_layer, branch2, use_bn=True, use_relu=False,num_output=out2c, kernel_size=1, pad=0, stride=1)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)

def ResBodyReduced(net, from_layer, block_name, out2a, out2b, stride, use_branch1):

  if use_branch1:
    branch1 = 'res{}_{}'.format(block_name,'branch1')
    ConvBNLayer(net, from_layer, branch1, use_bn=True, use_relu=False, num_output=out2b, kernel_size=1, pad=0, stride=stride)
  else:
    branch1 = from_layer

  out_name = 'res{}_{}'.format(block_name,'branch2a')
  ConvBNLayer(net, from_layer, out_name, use_bn=True, use_relu=True, num_output=out2a, kernel_size=3, pad=1, stride=stride)

  branch2 = 'res{}_{}'.format(block_name,'branch2b')
  ConvBNLayer(net, out_name, branch2, use_bn=True, use_relu=True, num_output=out2b, kernel_size=3, pad=1, stride=1)

  res_name = 'res{}'.format(block_name)
  net[res_name] = L.Eltwise(net[branch1], net[branch2])
  relu_name = '{}_relu'.format(res_name)
  net[relu_name] = L.ReLU(net[res_name], in_place=True)

def VGGNetBody(net, from_layer, need_fc=True, fully_conv=False, reduced=False,
        dilated=False, nopool=False, dropout=True, freeze_layers=[]):
    kwargs = {
            'param': [dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}

    assert from_layer in net.keys()
    net.conv1_1 = L.Convolution(net[from_layer], num_output=64, pad=1, kernel_size=3, **kwargs)

    net.relu1_1 = L.ReLU(net.conv1_1, in_place=True)
    net.conv1_2 = L.Convolution(net.relu1_1, num_output=64, pad=1, kernel_size=3, **kwargs)
    net.relu1_2 = L.ReLU(net.conv1_2, in_place=True)

    if nopool:
        name = 'conv1_3'
        net[name] = L.Convolution(net.relu1_2, num_output=64, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool1'
        net.pool1 = L.Pooling(net.relu1_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv2_1 = L.Convolution(net[name], num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_1 = L.ReLU(net.conv2_1, in_place=True)
    net.conv2_2 = L.Convolution(net.relu2_1, num_output=128, pad=1, kernel_size=3, **kwargs)
    net.relu2_2 = L.ReLU(net.conv2_2, in_place=True)

    if nopool:
        name = 'conv2_3'
        net[name] = L.Convolution(net.relu2_2, num_output=128, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool2'
        net[name] = L.Pooling(net.relu2_2, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv3_1 = L.Convolution(net[name], num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_1 = L.ReLU(net.conv3_1, in_place=True)
    net.conv3_2 = L.Convolution(net.relu3_1, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_2 = L.ReLU(net.conv3_2, in_place=True)
    net.conv3_3 = L.Convolution(net.relu3_2, num_output=256, pad=1, kernel_size=3, **kwargs)
    net.relu3_3 = L.ReLU(net.conv3_3, in_place=True)

    if nopool:
        name = 'conv3_4'
        net[name] = L.Convolution(net.relu3_3, num_output=256, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool3'
        net[name] = L.Pooling(net.relu3_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv4_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_1 = L.ReLU(net.conv4_1, in_place=True)
    net.conv4_2 = L.Convolution(net.relu4_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_2 = L.ReLU(net.conv4_2, in_place=True)
    net.conv4_3 = L.Convolution(net.relu4_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu4_3 = L.ReLU(net.conv4_3, in_place=True)

    if nopool:
        name = 'conv4_4'
        net[name] = L.Convolution(net.relu4_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
    else:
        name = 'pool4'
        net[name] = L.Pooling(net.relu4_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

    net.conv5_1 = L.Convolution(net[name], num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_1 = L.ReLU(net.conv5_1, in_place=True)
    net.conv5_2 = L.Convolution(net.relu5_1, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_2 = L.ReLU(net.conv5_2, in_place=True)
    net.conv5_3 = L.Convolution(net.relu5_2, num_output=512, pad=1, kernel_size=3, **kwargs)
    net.relu5_3 = L.ReLU(net.conv5_3, in_place=True)

    if need_fc:
        if dilated:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=1, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, pad=1, kernel_size=3, stride=1)
        else:
            if nopool:
                name = 'conv5_4'
                net[name] = L.Convolution(net.relu5_3, num_output=512, pad=1, kernel_size=3, stride=2, **kwargs)
            else:
                name = 'pool5'
                net[name] = L.Pooling(net.relu5_3, pool=P.Pooling.MAX, kernel_size=2, stride=2)

        if fully_conv:
            if dilated:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=6, kernel_size=3, dilation=6, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=6, kernel_size=7, dilation=2, **kwargs)
            else:
                if reduced:
                    net.fc6 = L.Convolution(net[name], num_output=1024, pad=3, kernel_size=3, dilation=3, **kwargs)
                else:
                    net.fc6 = L.Convolution(net[name], num_output=4096, pad=3, kernel_size=7, **kwargs)

            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)

            if reduced:
                net.fc7 = L.Convolution(net.relu6, num_output=1024, kernel_size=1, **kwargs)
            else:
                net.fc7 = L.Convolution(net.relu6, num_output=4096, kernel_size=1, **kwargs)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)
        else:
            net.fc6 = L.InnerProduct(net.pool5, num_output=4096)
            net.relu6 = L.ReLU(net.fc6, in_place=True)
            if dropout:
                net.drop6 = L.Dropout(net.relu6, dropout_ratio=0.5, in_place=True)
            net.fc7 = L.InnerProduct(net.relu6, num_output=4096)
            net.relu7 = L.ReLU(net.fc7, in_place=True)
            if dropout:
                net.drop7 = L.Dropout(net.relu7, dropout_ratio=0.5, in_place=True)

    # Update freeze layers.
    kwargs['param'] = [dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0)]
    layers = net.keys()
    for freeze_layer in freeze_layers:
        if freeze_layer in layers:
            net.update(freeze_layer, kwargs)

    return net


def ResNet18Body(net, from_layer, use_pool5=True):
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True, num_output=64, kernel_size=7, pad=3, stride=2)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBodyReduced(net, 'pool1', '2a', out2a=64, out2b=64, stride=1, use_branch1=True)
    ResBodyReduced(net, 'res2a', '2b', out2a=64, out2b=64, stride=1, use_branch1=False)

    ResBodyReduced(net, 'res2b', '3a', out2a=128, out2b=128, stride=2, use_branch1=True)
    ResBodyReduced(net, 'res3a', '3b', out2a=128, out2b=128, stride=1, use_branch1=False)

    ResBodyReduced(net, 'res3b', '4a', out2a=256, out2b=256, stride=2, use_branch1=True)
    ResBodyReduced(net, 'res4a', '4b', out2a=256, out2b=256, stride=1, use_branch1=False)

    ResBodyReduced(net, 'res4b', '5a', out2a=512, out2b=512, stride=2, use_branch1=True)
    ResBodyReduced(net, 'res5a', '5b', out2a=512, out2b=512, stride=1, use_branch1=False)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5b, pool=P.Pooling.AVE, global_pooling=True)

    return net

def ResNet50Body(net, from_layer, use_pool5=True):
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,num_output=64, kernel_size=7, pad=3, stride=2)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 4):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 6):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=2, use_branch1=True)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net

def ResNet101Body(net, from_layer, use_pool5=True):
    ConvBNLayer(net, from_layer, 'conv1', use_bn=True, use_relu=True,num_output=64, kernel_size=7, pad=3, stride=2)

    net.pool1 = L.Pooling(net.conv1, pool=P.Pooling.MAX, kernel_size=3, stride=2)

    ResBody(net, 'pool1', '2a', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=True)
    ResBody(net, 'res2a', '2b', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)
    ResBody(net, 'res2b', '2c', out2a=64, out2b=64, out2c=256, stride=1, use_branch1=False)

    ResBody(net, 'res2c', '3a', out2a=128, out2b=128, out2c=512, stride=2, use_branch1=True)

    from_layer = 'res3a'
    for i in xrange(1, 4):
      block_name = '3b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=128, out2b=128, out2c=512, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '4a', out2a=256, out2b=256, out2c=1024, stride=2, use_branch1=True)

    from_layer = 'res4a'
    for i in xrange(1, 23):
      block_name = '4b{}'.format(i)
      ResBody(net, from_layer, block_name, out2a=256, out2b=256, out2c=1024, stride=1, use_branch1=False)
      from_layer = 'res{}'.format(block_name)

    ResBody(net, from_layer, '5a', out2a=512, out2b=512, out2c=2048, stride=2, use_branch1=True)
    ResBody(net, 'res5a', '5b', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)
    ResBody(net, 'res5b', '5c', out2a=512, out2b=512, out2c=2048, stride=1, use_branch1=False)

    if use_pool5:
      net.pool5 = L.Pooling(net.res5c, pool=P.Pooling.AVE, global_pooling=True)

    return net

def CreateRollingStruct(net,from_layersbasename=[],num_outputs=[],rolling_rate=0.25,roll_idx=1):
    roll_layers=[]
    factor = 2
    from_layers = copy.copy(from_layersbasename)
    assert len(from_layers) == len(num_outputs)

    if roll_idx == 1: 
        from_layers[0] = '%s_norm'%(from_layersbasename[0]) # need normalization? so name the layer differently from other layers?
    else:
        for i in range(len(from_layersbasename)):
            from_layers[i] = '%s_%d'%(from_layersbasename[i],roll_idx)

    for i in range(len(from_layersbasename)): 
        f_layers = []
        num_out = int(num_outputs[i]*rolling_rate)

        if i > 0:
            o_layer='%s_r%d'%(from_layersbasename[i-1],roll_idx)
            kwargs = {
            'param': [dict(name='%s_r_w'%(from_layersbasename[i-1]),lr_mult=1, decay_mult=1), dict(name='%s_r_b'%(from_layersbasename[i-1]),lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}
            net[o_layer]=L.Convolution(net[from_layers[i-1]],num_output=num_out, kernel_size=1, stride=1, **kwargs)
            net['%s_relu'%(o_layer)]=L.ReLU(net[o_layer],in_place=True)
            net['%s_pool'%(o_layer)]=L.Pooling(net[o_layer],pool=P.Pooling.MAX, kernel_size=2, stride=2,in_place=True)
            f_layers.append(net[o_layer])
            
        f_layers.append(net[from_layers[i]])

        if i < len(from_layersbasename)-1:
            o_layer='%s_l%d'%(from_layersbasename[i+1],roll_idx)
            kwargs = {
            'param': [dict(name='%s_l_w'%(from_layersbasename[i+1]),lr_mult=1, decay_mult=1), dict(name='%s_l_b'%(from_layersbasename[i+1]),lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}
            net[o_layer]=L.Convolution(net[from_layers[i+1]],num_output=num_out, kernel_size=1, stride=1, **kwargs)
            net['%s_relu'%(o_layer)]=L.ReLU(net[o_layer],in_place=True)
            kwargs = {
            'param': [dict(lr_mult=0, decay_mult=0)],
            }
            convolution_param = {
                'pad': int(np.ceil((factor - 1) / 2.)),
                'kernel_size': int(2 * factor - factor % 2),
                'stride': int(factor),
                'weight_filler': dict(type='bilinear'),
                'bias_term': False,
                'num_output': num_out,
                'group': num_out
            }
            net['%s_deconv'%(o_layer)]=L.Deconvolution(net[o_layer],convolution_param=convolution_param, **kwargs)
            f_layers.append(net['%s_deconv'%(o_layer)])

        o_layer='%s_concat_%d'%(from_layersbasename[i],roll_idx)
        net[o_layer]=L.Concat(*f_layers,axis = 1)
        
        from_layer = o_layer
        o_layer = '%s_%d'%(from_layersbasename[i],roll_idx+1)
        kwargs = {
            'param': [dict(name='%s_cw'%(from_layersbasename[i]),lr_mult=1, decay_mult=1), dict(name='%s_cb'%(from_layersbasename[i]),lr_mult=2, decay_mult=0)],
            'weight_filler': dict(type='xavier'),
            'bias_filler': dict(type='constant', value=0)}
        net[o_layer]=L.Convolution(net[from_layer],num_output=num_outputs[i], kernel_size=1, stride=1, **kwargs)
        net['%s_relu'%(o_layer)]=L.ReLU(net[o_layer],in_place=True)
        
        roll_layers.append(o_layer)
        
    return roll_layers

# multi loc conv priorbox layers generate after a feature layer
def CreateMultiBoxHead_share_2x(net, data_layer="data", num_classes=[], from_layers=[],
                             min_sizes=[], max_sizes=[], prior_variance=[0.1],
                             aspect_ratios=[], flip=True, clip=True, branch_num = 2):
    assert num_classes > 0, "num_classes must be positive number"
    assert len(from_layers) == len(min_sizes), "from_layers and min_sizes should have same length"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"
    assert data_layer in net.keys(), "data_layer is not in net's layers"

    num = len(from_layers)
    priorbox_layers = []
    loc_layers = []
    conf_layers = []
    min_sizes_2x = []
    max_sizes_2x = []
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

    for i in range(0, num):
        from_layer = from_layers[i]

        # Add Normalization layer for the first base layer only (conv4_3r)
        if i == 0:
            norm_name = "{}_norm".format(from_layer)
            net[norm_name] = L.Normalize(net[from_layer],scale_filler=dict(type="constant", value=20),across_spatial=False, channel_shared=False)
            from_layer = norm_name

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]

        if max_sizes and max_sizes[i]:
            num_priors_per_location = 2 + len(aspect_ratio)
        else:
            num_priors_per_location = 1 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        if (i == num - 1):
            repeat_times = 1
        else:
            repeat_times = branch_num

        for mbox_idx in range(0,repeat_times):
            # Create location prediction layer.
            name = "{}_mbox_loc{}".format(from_layer,mbox_idx)
            num_loc_output = num_priors_per_location * 4
            share_weight='{}_loc{}'.format(from_layers[i],mbox_idx)
            kwargs = {
                'param': [dict(name='%s_w'%(share_weight),lr_mult=1, decay_mult=1), dict(name='%s_b'%(share_weight),lr_mult=2, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)
            }
            net[name] = L.Convolution(net[from_layer], num_output=num_loc_output, kernel_size=3, pad=1, stride=1, **kwargs)

            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            loc_layers.append(net[flatten_name])

            # Create confidence prediction layer.
            name = "{}_mbox_conf{}".format(from_layer,mbox_idx)
            num_conf_output = num_priors_per_location * num_classes
            share_weight='{}_conf{}'.format(from_layers[i],mbox_idx)
            kwargs = {
                'param': [dict(name='%s_w'%(share_weight),lr_mult=1, decay_mult=1), dict(name='%s_b'%(share_weight),lr_mult=2, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)
            }
            net[name] = L.Convolution(net[from_layer], num_output=num_conf_output, kernel_size=3, pad=1, stride=1, **kwargs)

            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            conf_layers.append(net[flatten_name])

            # Create prior generation layer.
            name = "{}_mbox_priorbox{}".format(from_layer,mbox_idx)
            if max_sizes and max_sizes[i]:
                if aspect_ratio:
                    net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], max_size=max_sizes_2x[branch_num*i + mbox_idx],
                                             aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
                else:
                    net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], max_size=max_sizes_2x[branch_num*i + mbox_idx],
                                             clip=clip, variance=prior_variance)
            else:
                if aspect_ratio:
                    net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                             aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
                else:
                    net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                             clip=clip, variance=prior_variance)
            priorbox_layers.append(net[name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    net["mbox_loc"] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net["mbox_loc"])
    net["mbox_conf"] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net["mbox_conf"])
    net["mbox_priorbox"] = L.Concat(*priorbox_layers, axis=2)
    mbox_layers.append(net["mbox_priorbox"])

    return mbox_layers

# multi loc conv priorbox layers generate after a feature layer
def CreateMultiBoxHead_share_2x_without_prior(net, num_classes=[], from_layers=[],
                             max_sizes=[], aspect_ratios=[], flip=True,
                             layers_names=[],roll_idx=1,branch_num = 2):
    assert len(from_layers) == len(layers_names)
    assert num_classes > 0, "num_classes must be positive number"
    if max_sizes:
        assert len(from_layers) == len(max_sizes), "from_layers and max_sizes should have same length"

    num = len(from_layers)
    loc_layers = []
    conf_layers = []


    for i in range(0, num):
        from_layer = from_layers[i]

        # Estimate number of priors per location given provided parameters.
        aspect_ratio = []
        if len(aspect_ratios) > i:
            aspect_ratio = aspect_ratios[i]
            if type(aspect_ratio) is not list:
                aspect_ratio = [aspect_ratio]

        if max_sizes and max_sizes[i]:
            num_priors_per_location = 2 + len(aspect_ratio)
        else:
            num_priors_per_location = 1 + len(aspect_ratio)
        if flip:
            num_priors_per_location += len(aspect_ratio)

        if (i == num - 1):
            repeat_times = 1
        else:
            repeat_times = branch_num

        for mbox_idx in range(0,repeat_times):
            # Create location prediction layer.
            name = "{}_mbox_loc{}{}".format(from_layer, roll_idx+1,mbox_idx)
            num_loc_output = num_priors_per_location * 4
            share_weight='{}_loc{}'.format(layers_names[i],mbox_idx)
            kwargs = {
                'param': [dict(name='%s_w'%(share_weight),lr_mult=1, decay_mult=1), dict(name='%s_b'%(share_weight),lr_mult=2, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)
            }
            net[name] = L.Convolution(net[from_layer], num_output=num_loc_output, kernel_size=3, pad=1, stride=1, **kwargs)


            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            loc_layers.append(net[flatten_name])

            # Create confidence prediction layer.
            name = "{}_mbox_conf{}{}".format(from_layer, roll_idx+1,mbox_idx)
            num_conf_output = num_priors_per_location * num_classes;
            share_weight='{}_conf{}'.format(layers_names[i],mbox_idx)
            kwargs = {
                'param': [dict(name='%s_w'%(share_weight),lr_mult=1, decay_mult=1), dict(name='%s_b'%(share_weight),lr_mult=2, decay_mult=0)],
                'weight_filler': dict(type='xavier'),
                'bias_filler': dict(type='constant', value=0)
            }
            net[name] = L.Convolution(net[from_layer], num_output=num_conf_output, kernel_size=3, pad=1, stride=1, **kwargs)

            permute_name = "{}_perm".format(name)
            net[permute_name] = L.Permute(net[name], order=[0, 2, 3, 1])
            flatten_name = "{}_flat".format(name)
            net[flatten_name] = L.Flatten(net[permute_name], axis=1)
            conf_layers.append(net[flatten_name])

    # Concatenate priorbox, loc, and conf layers.
    mbox_layers = []
    name = "mbox_loc{}".format(roll_idx+1)
    net[name] = L.Concat(*loc_layers, axis=1)
    mbox_layers.append(net[name])
    name = "mbox_conf{}".format(roll_idx+1)
    net[name] = L.Concat(*conf_layers, axis=1)
    mbox_layers.append(net[name])

    return mbox_layers

