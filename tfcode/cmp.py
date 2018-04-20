# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Code for setting up the network for CMP.

Sets up the mapper and the planner.
"""

import sys, os, numpy as np
import matplotlib.pyplot as plt
import copy
import argparse, pprint
import time


import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim import arg_scope

import logging
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from src import utils 
import src.file_utils as fu
import tfcode.nav_utils as nu 
import tfcode.cmp_utils as cu 
import tfcode.cmp_summary as cmp_s
from tfcode import tf_utils
from tensorflow.python.framework import ops
from copy import deepcopy
import pdb

value_iteration_network = cu.value_iteration_network
rotate_preds            = cu.rotate_preds
deconv                  = cu.deconv
get_visual_frustum      = cu.get_visual_frustum
fr_v2                   = cu.fr_v2

setup_train_step_kwargs = nu.default_train_step_kwargs
compute_losses_multi_or = nu.compute_losses_multi_or
#Tri
rl_compute_losses_multi_or = nu.rl_compute_losses_multi_or

get_repr_from_image     = nu.get_repr_from_image

_save_d_at_t            = nu.save_d_at_t
_save_all               = nu.save_all
_eval_ap                = nu.eval_ap
_eval_dist              = nu.eval_dist
_plot_trajectories      = nu.plot_trajectories

_vis_readout_maps       = cmp_s._vis_readout_maps
_vis                    = cmp_s._vis
_summary_vis            = cmp_s._summary_vis
_summary_readout_maps   = cmp_s._summary_readout_maps
_add_summaries          = cmp_s._add_summaries

#Tri
def copy_variable_to_graph(org_instance, to_graph, namespace,
                           copied_variables={},fix_shape=True):
    """
    Copies the Variable instance 'org_instance' into the graph
    'to_graph', under the given namespace.
    The dict 'copied_variables', if provided, will be updated with
    mapping the new variable's name to the instance.
    """
 
    if not isinstance(org_instance, tf.Variable):
        raise TypeError(str(org_instance) + " is not a Variable")
 
    #The name of the new variable
    if namespace != '':
        new_name = (namespace + '/' +
                    org_instance.name[:org_instance.name.index(':')])
        new_name_full = namespace + '/' + org_instance.name
    else:
        new_name = org_instance.name[:org_instance.name.index(':')]
        new_name_full = org_instance.name

    if new_name_full in copied_variables:
        return to_graph.get_tensor_by_name(copied_variables[new_name_full].name)
 
    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given namespace,
    #except the special ones required for variable initialization and
    #training.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
           # if (name == ops.GraphKeys.VARIABLES or
            if (name == ops.GraphKeys.GLOBAL_VARIABLES or 
                name == ops.GraphKeys.TRAINABLE_VARIABLES or
                namespace == ''):
                collections.append(name)
            else:
                collections.append(namespace + '/' + name)
 
    #See if its trainable.
    trainable = (org_instance in org_instance.graph.get_collection(
        ops.GraphKeys.TRAINABLE_VARIABLES))
    #Get the initial value
    #pdb.set_trace()
    with org_instance.graph.as_default():
        temp_session = tf.Session()
        init_value = temp_session.run(org_instance.initialized_value())
 
    #Initialize the new variable
    with to_graph.as_default():
        new_var = tf.Variable(init_value,
                              trainable,
                              name=new_name,
                              collections=collections,
                              validate_shape=fix_shape)
 
    #Add to the copied_variables dict
    copied_variables[new_var.name] = new_var
 
    return new_var
 
 
def copy_to_graph(org_instance, to_graph, copied_variables={}, namespace=""):
    """
    Makes a copy of the Operation/Tensor instance 'org_instance'
    for the graph 'to_graph', recursively. Therefore, all required
    structures linked to org_instance will be automatically copied.
    'copied_variables' should be a dict mapping pertinent copied variable
    names to the copied instances.
     
    The new instances are automatically inserted into the given 'namespace'.
    If namespace='', it is inserted into the graph's global namespace.
    However, to avoid naming conflicts, its better to provide a namespace.
    If the instance(s) happens to be a part of collection(s), they are
    are added to the appropriate collections in to_graph as well.
    For example, for collection 'C' which the instance happens to be a
    part of, given a namespace 'N', the new instance will be a part of
    'N/C' in to_graph.
 
    Returns the corresponding instance with respect to to_graph.
 
    TODO: Order of insertion into collections is not preserved
    """
 
    #The name of the new instance
    if namespace != '':
        new_name = namespace + '/' + org_instance.name
    else:
        new_name = org_instance.name
 
    #If a variable by the new name already exists, return the
    #correspondng tensor that will act as an input
    if new_name in copied_variables:
        return to_graph.get_tensor_by_name(
            copied_variables[new_name].name)
 
    #If an instance of the same name exists, return appropriately
    try:
        already_present = to_graph.as_graph_element(new_name,
                                                    allow_tensor=True,
                                                    allow_operation=True)
        return already_present
    except:
        pass
 
    #Get the collections that the new instance needs to be added to.
    #The new collections will also be a part of the given namespace.
    collections = []
    for name, collection in org_instance.graph._collections.items():
        if org_instance in collection:
            if namespace == '':
                collections.append(name)
            else:
                collections.append(namespace + '/' + name)
     
    #Take action based on the class of the instance
 
    #if isinstance(org_instance, tf.python.framework.ops.Tensor):
    if isinstance(org_instance, ops.Tensor):
 
        #If its a Tensor, it is one of the outputs of the underlying
        #op. Therefore, copy the op itself and return the appropriate
        #output.
        op = org_instance.op
        new_op = copy_to_graph(op, to_graph, copied_variables, namespace)
        output_index = op.outputs.index(org_instance)
        new_tensor = new_op.outputs[output_index]
        #Add to collections if any
        for collection in collections:
            to_graph.add_to_collection(collection, new_tensor)
 
        return new_tensor
 
    #elif isinstance(org_instance, tf.python.framework.ops.Operation):
    elif isinstance(org_instance, ops.Operation):
 
        op = org_instance
 
        #If it has an original_op parameter, copy it
        if op._original_op is not None:
            new_original_op = copy_to_graph(op._original_op, to_graph,
                                            copied_variables, namespace)
        else:
            new_original_op = None
 
        #If it has control inputs, call this function recursively on each.
        new_control_inputs = [copy_to_graph(x, to_graph, copied_variables,
                                            namespace)
                              for x in op.control_inputs]
 
        #If it has inputs, call this function recursively on each.
        new_inputs = [copy_to_graph(x, to_graph, copied_variables,
                                    namespace)
                      for x in op.inputs]
 
        #Make a new node_def based on that of the original.
        #An instance of tensorflow.core.framework.graph_pb2.NodeDef, it
        #stores String-based info such as name, device and type of the op.
        #Unique to every Operation instance.
        new_node_def = deepcopy(op._node_def)
        #Change the name
        new_node_def.name = new_name
 
        #Copy the other inputs needed for initialization
        output_types = op._output_types[:]
        input_types = op._input_types[:]
 
        #Make a copy of the op_def too.
        #Its unique to every _type_ of Operation.
        op_def = deepcopy(op._op_def)
 
        #Initialize a new Operation instance
        #new_op = tf.python.framework.ops.Operation(new_node_def,
        new_op = ops.Operation(new_node_def,
                                                   to_graph,
                                                   new_inputs,
                                                   output_types,
                                                   new_control_inputs,
                                                   input_types,
                                                   new_original_op,
                                                   op_def)
        #Use Graph's hidden methods to add the op
        to_graph._add_op(new_op)
        to_graph._record_op_seen_by_control_dependencies(new_op)
        #pdb.set_trace()
        for device_function in reversed(to_graph._device_function_stack):
            if device_function is not None:
               new_op._set_device(device_function(new_op))
 
        return new_op
 
    else:
        raise TypeError("Could not copy instance: " + str(org_instance))
 
 
def get_copied(original, graph, copied_variables={}, namespace=""):
    """
    Get a copy of the instance 'original', present in 'graph', under
    the given 'namespace'.
    'copied_variables' is a dict mapping pertinent variable names to the
    copy instances.
    """
 
    #The name of the copied instance
    if namespace != '':
        new_name = namespace + '/' + original.name
    else:
        new_name = original.name
 
    #If a variable by the name already exists, return it
    if new_name in copied_variables:
        return copied_variables[new_name]
 
    return graph.as_graph_element(new_name, allow_tensor=True,
                                  allow_operation=True)

def clone_vars(m):
  """cloning all the variables & operations to a new graph."""
  #pdb.set_trace()
  namespace = "cloned"
  #cloned_vars = []
  #if m.cloned_vars is not None:
  if hasattr(m, 'cloned_vars'):
    cloned_vars = m.cloned_vars
  else:
    cloned_vars = {}
  #cloned_graph = tf.Graph()_
  #cloned_graph = tf.get_default_graph()
  graph = tf.get_default_graph()
  #graph = tf.Graph()
  #vars_to_restore = slim.get_variables_to_restore()
  vars_to_restore = slim.get_model_variables()
  #pdb.set_trace()
  for var in vars_to_restore[-3:]:
    #new_var = tf.contrib.copy_graph.copy_variable_to_graph(var,cloned_graph,namespace)
    #cloned_vars.append(new_var)
    ###new_var = copy_variable_to_graph(var,cloned_graph,namespace,cloned_vars,fix_shape=True)
    new_var = copy_variable_to_graph(var,graph,namespace,cloned_vars,fix_shape=True)

  #pdb.set_trace()
  #print "finished cloning variables"
  #cloned_action_logits_op = tf.contrib.copy_graph.copy_op_to_graph(m.action_logits_op,cloned_graph,cloned_vars,namespace) 
  #cloned_action_logits_op = copy_to_graph(m.action_logits_op,cloned_graph,cloned_vars,namespace)
  #cloned_action_logits_op = copy_to_graph(m.action_logits_op,graph,cloned_vars,namespace)
  #print "finished cloning op"
  #m.cloned_graph = cloned_graph
  m.cloned_vars = cloned_vars
  #m.cloned_action_logits_op = cloned_action_logits_op
  m.cloned_namespace = namespace
  print "finished cloning model ----------------------"
  #pdb.set_trace()


def clonemodel(m):
  """cloning all the variables & operations to a new graph."""
  #pdb.set_trace()
  namespace = "cloned"
  #cloned_vars = []
  #cloned_vars = {}
  #if m.cloned_vars is not None:
  if hasattr(m, 'cloned_vars'):
    cloned_vars = m.cloned_vars
  else:
    cloned_vars = {}
  #cloned_graph = tf.Graph()_
  #cloned_graph = tf.get_default_graph()
  graph = tf.get_default_graph()
  #graph = tf.Graph()
  #vars_to_restore = slim.get_variables_to_restore()
  vars_to_restore = slim.get_model_variables()
  #pdb.set_trace()
  #for var in vars_to_restore:
  for var in vars_to_restore[-1:]:
    #new_var = tf.contrib.copy_graph.copy_variable_to_graph(var,cloned_graph,namespace)
    #cloned_vars.append(new_var)
    ###new_var = copy_variable_to_graph(var,cloned_graph,namespace,cloned_vars,fix_shape=True)
    new_var = copy_variable_to_graph(var,graph,namespace,cloned_vars,fix_shape=True)

  #pdb.set_trace()
  print "finished cloning variables"
  #cloned_action_logits_op = tf.contrib.copy_graph.copy_op_to_graph(m.action_logits_op,cloned_graph,cloned_vars,namespace) 
  #cloned_action_logits_op = copy_to_graph(m.action_logits_op,cloned_graph,cloned_vars,namespace)
  cloned_action_logits_op = copy_to_graph(m.action_logits_op,graph,cloned_vars,namespace)
  print "finished cloning op"
  #m.cloned_graph = cloned_graph
  m.cloned_vars = cloned_vars
  m.cloned_action_logits_op = cloned_action_logits_op
  m.cloned_namespace = namespace
  print "finished cloning model ----------------------"
  #pdb.set_trace()

def set_copying_ops(m):
  copying_ops = []
  model_vars = slim.get_model_variables()
  for var in model_vars:
    new_name = (m.cloned_namespace + '/' + var.name)
    if new_name in m.cloned_vars.keys():
      #pdb.set_trace()
      new_var = m.cloned_vars[new_name]
      copy_op = new_var.assign(var)
      copying_ops.append(copy_op)

  m.copying_ops = copying_ops

def set_tmp_params(m):
  m.rl_num_explore_steps = 100
  m.rl_datapool_size = 1000
  m.rl_datapool = []
  m.rl_discount_factor = 0.9
  m.rl_rand_act_prob_start = 1.0
  m.rl_rand_act_prob_end = 0.1
  m.rl_rand_act_anneal_time = 5000
  m.rl_target_net_update_freq = 1000

def _inputs(problem):
  # Set up inputs.
  with tf.name_scope('inputs'):
    inputs = []
    inputs.append(('orig_maps', tf.float32, 
                   (problem.batch_size, 1, None, None, 1)))
    inputs.append(('goal_loc', tf.float32, 
                   (problem.batch_size, problem.num_goals, 2)))
    common_input_data, _ = tf_utils.setup_inputs(inputs)

    inputs = []
    if problem.input_type == 'vision':
      # Multiple images from an array of cameras.
      inputs.append(('imgs', tf.float32, 
                     (problem.batch_size, None, len(problem.aux_delta_thetas)+1,
                      problem.img_height, problem.img_width,
                      problem.img_channels)))
    elif problem.input_type == 'analytical_counts':
      for i in range(len(problem.map_crop_sizes)):
        inputs.append(('analytical_counts_{:d}'.format(i), tf.float32, 
                      (problem.batch_size, None, problem.map_crop_sizes[i],
                       problem.map_crop_sizes[i], problem.map_channels)))

    if problem.outputs.readout_maps: 
      for i in range(len(problem.readout_maps_crop_sizes)):
        inputs.append(('readout_maps_{:d}'.format(i), tf.float32, 
                      (problem.batch_size, None,
                       problem.readout_maps_crop_sizes[i],
                       problem.readout_maps_crop_sizes[i],
                       problem.readout_maps_channels)))

    for i in range(len(problem.map_crop_sizes)):
      inputs.append(('ego_goal_imgs_{:d}'.format(i), tf.float32, 
                    (problem.batch_size, None, problem.map_crop_sizes[i],
                     problem.map_crop_sizes[i], problem.goal_channels)))
      for s in ['sum_num', 'sum_denom', 'max_denom']:
        inputs.append(('running_'+s+'_{:d}'.format(i), tf.float32,
                       (problem.batch_size, 1, problem.map_crop_sizes[i],
                        problem.map_crop_sizes[i], problem.map_channels)))

    inputs.append(('incremental_locs', tf.float32, 
                   (problem.batch_size, None, 2)))
    inputs.append(('incremental_thetas', tf.float32, 
                   (problem.batch_size, None, 1)))
    inputs.append(('step_number', tf.int32, (1, None, 1)))
    inputs.append(('node_ids', tf.int32, (problem.batch_size, None,
                                          problem.node_ids_dim)))
    inputs.append(('perturbs', tf.float32, (problem.batch_size, None,
                                            problem.perturbs_dim)))
    
    # For plotting result plots
    inputs.append(('loc_on_map', tf.float32, (problem.batch_size, None, 2)))
    inputs.append(('gt_dist_to_goal', tf.float32, (problem.batch_size, None, 1)))

    step_input_data, _ = tf_utils.setup_inputs(inputs)

    inputs = []
    inputs.append(('action', tf.int32, (problem.batch_size, None, problem.num_actions)))
    #Tri
    inputs.append(('action_one', tf.int32, (None, 1)))
    inputs.append(('target',tf.float32, (None, 1)))

    train_data, _ = tf_utils.setup_inputs(inputs)
    train_data.update(step_input_data)
    train_data.update(common_input_data)
  return common_input_data, step_input_data, train_data 

def readout_general(multi_scale_belief, num_neurons, strides, layers_per_block,
                    kernel_size, batch_norm_is_training_op, wt_decay):
  multi_scale_belief = tf.stop_gradient(multi_scale_belief)
  with tf.variable_scope('readout_maps_deconv'):
    x, outs = deconv(multi_scale_belief, batch_norm_is_training_op,
                     wt_decay=wt_decay, neurons=num_neurons, strides=strides,
                     layers_per_block=layers_per_block, kernel_size=kernel_size,
                     conv_fn=slim.conv2d_transpose, offset=0,
                     name='readout_maps_deconv')
    probs = tf.sigmoid(x)
  return x, probs


def running_combine(fss_logits, confs_probs, incremental_locs,
                    incremental_thetas, previous_sum_num, previous_sum_denom,
                    previous_max_denom, map_size, num_steps):
  # fss_logits is B x N x H x W x C
  # confs_logits is B x N x H x W x C
  # incremental_locs is B x N x 2
  # incremental_thetas is B x N x 1
  # previous_sum_num etc is B x 1 x H x W x C

  with tf.name_scope('combine_{:d}'.format(num_steps)):
    running_sum_nums_ = []; running_sum_denoms_ = [];
    running_max_denoms_ = [];

    fss_logits_ = tf.unstack(fss_logits, axis=1, num=num_steps)
    confs_probs_ = tf.unstack(confs_probs, axis=1, num=num_steps)
    incremental_locs_ = tf.unstack(incremental_locs, axis=1, num=num_steps)
    incremental_thetas_ = tf.unstack(incremental_thetas, axis=1, num=num_steps)
    running_sum_num = tf.unstack(previous_sum_num, axis=1, num=1)[0]
    running_sum_denom = tf.unstack(previous_sum_denom, axis=1, num=1)[0]
    running_max_denom = tf.unstack(previous_max_denom, axis=1, num=1)[0]

    for i in range(num_steps):
      # Rotate the previous running_num and running_denom
      running_sum_num, running_sum_denom, running_max_denom = rotate_preds(
          incremental_locs_[i], incremental_thetas_[i], map_size,
          [running_sum_num, running_sum_denom, running_max_denom],
          output_valid_mask=False)[0]
      # print i, num_steps, running_sum_num.get_shape().as_list()
      running_sum_num = running_sum_num + fss_logits_[i] * confs_probs_[i]
      running_sum_denom = running_sum_denom + confs_probs_[i]
      running_max_denom = tf.maximum(running_max_denom, confs_probs_[i])
      running_sum_nums_.append(running_sum_num)
      running_sum_denoms_.append(running_sum_denom)
      running_max_denoms_.append(running_max_denom)

    running_sum_nums = tf.stack(running_sum_nums_, axis=1)
    running_sum_denoms = tf.stack(running_sum_denoms_, axis=1)
    running_max_denoms = tf.stack(running_max_denoms_, axis=1)
    return running_sum_nums, running_sum_denoms, running_max_denoms

def get_map_from_images(imgs, mapper_arch, task_params, freeze_conv, wt_decay,
                        is_training, batch_norm_is_training_op, num_maps,
                        split_maps=True):
  # Hit image with a resnet.
  n_views = len(task_params.aux_delta_thetas) + 1
  out = utils.Foo()

  images_reshaped = tf.reshape(imgs, 
      shape=[-1, task_params.img_height,
             task_params.img_width,
             task_params.img_channels], name='re_image')

  x, out.vars_to_restore = get_repr_from_image(
      images_reshaped, task_params.modalities, task_params.data_augment,
      mapper_arch.encoder, freeze_conv, wt_decay, is_training)

  # Reshape into nice things so that these can be accumulated over time steps
  # for faster backprop.
  sh_before = x.get_shape().as_list()
  out.encoder_output = tf.reshape(x, shape=[task_params.batch_size, -1, n_views] + sh_before[1:])
  x = tf.reshape(out.encoder_output, shape=[-1] + sh_before[1:])

  # Add a layer to reduce dimensions for a fc layer.
  if mapper_arch.dim_reduce_neurons > 0:
    ks = 1; neurons = mapper_arch.dim_reduce_neurons;
    init_var = np.sqrt(2.0/(ks**2)/neurons)
    batch_norm_param = mapper_arch.batch_norm_param
    batch_norm_param['is_training'] = batch_norm_is_training_op
    out.conv_feat = slim.conv2d(x, neurons, kernel_size=ks, stride=1,
                    normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_param,
                    padding='SAME', scope='dim_reduce',
                    weights_regularizer=slim.l2_regularizer(wt_decay),
                    weights_initializer=tf.random_normal_initializer(stddev=init_var))
    reshape_conv_feat = slim.flatten(out.conv_feat)
    sh = reshape_conv_feat.get_shape().as_list()
    out.reshape_conv_feat = tf.reshape(reshape_conv_feat, shape=[-1, sh[1]*n_views])

  with tf.variable_scope('fc'):
    # Fully connected layers to compute the representation in top-view space.
    fc_batch_norm_param = {'center': True, 'scale': True, 
                           'activation_fn':tf.nn.relu,
                           'is_training': batch_norm_is_training_op}
    f = out.reshape_conv_feat
    out_neurons = (mapper_arch.fc_out_size**2)*mapper_arch.fc_out_neurons
    neurons = mapper_arch.fc_neurons + [out_neurons]
    f, _ = tf_utils.fc_network(f, neurons=neurons, wt_decay=wt_decay,
                               name='fc', offset=0,
                               batch_norm_param=fc_batch_norm_param,
                               is_training=is_training,
                               dropout_ratio=mapper_arch.fc_dropout)
    f = tf.reshape(f, shape=[-1, mapper_arch.fc_out_size,
                             mapper_arch.fc_out_size,
                             mapper_arch.fc_out_neurons], name='re_fc')

  # Use pool5 to predict the free space map via deconv layers.
  with tf.variable_scope('deconv'):
    x, outs = deconv(f, batch_norm_is_training_op, wt_decay=wt_decay,
                     neurons=mapper_arch.deconv_neurons,
                     strides=mapper_arch.deconv_strides,
                     layers_per_block=mapper_arch.deconv_layers_per_block,
                     kernel_size=mapper_arch.deconv_kernel_size,
                     conv_fn=slim.conv2d_transpose, offset=0, name='deconv')

  # Reshape x the right way.
  sh = x.get_shape().as_list()
  x = tf.reshape(x, shape=[task_params.batch_size, -1] + sh[1:])
  out.deconv_output = x

  # Separate out the map and the confidence predictions, pass the confidence
  # through a sigmoid.
  if split_maps:
    with tf.name_scope('split'):
      out_all = tf.split(value=x, axis=4, num_or_size_splits=2*num_maps)
      out.fss_logits = out_all[:num_maps]
      out.confs_logits = out_all[num_maps:]
    with tf.name_scope('sigmoid'):
      out.confs_probs = [tf.nn.sigmoid(x) for x in out.confs_logits]
  return out

def init_fn_tri(sess,m):
  pdb.set_trace()
  print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
  assign_fn = slim.assign_from_checkpoint_fn(args.solver.pretrained_path,
                                                 m.vision_ops.vars_to_restore)
  assign_fn(sess)
  for op in m.copying_ops:
    sess.run(op) 

def setup_to_run(m, args, is_training, batch_norm_is_training, summary_mode):
  assert(args.arch.multi_scale), 'removed support for old single scale code.'
  # Set up the model.
  tf.set_random_seed(args.solver.seed)
  task_params = args.navtask.task_params

  batch_norm_is_training_op = \
      tf.placeholder_with_default(batch_norm_is_training, shape=[],
                                  name='batch_norm_is_training_op') 

  # Setup the inputs
  m.input_tensors = {}
  m.train_ops = {}
  m.input_tensors['common'], m.input_tensors['step'], m.input_tensors['train'] = \
      _inputs(task_params)

  m.init_fn = None

  if task_params.input_type == 'vision':
    m.vision_ops = get_map_from_images(
        m.input_tensors['step']['imgs'], args.mapper_arch,
        task_params, args.solver.freeze_conv,
        args.solver.wt_decay, is_training, batch_norm_is_training_op,
        num_maps=len(task_params.map_crop_sizes))

    # Load variables from snapshot if needed.
    if args.solver.pretrained_path is not None:
      #pdb.set_trace()
      #clone_vars(m)
      #set_copying_ops(m)

      def init_fn_tri2(s):
        pdb.set_trace()
        print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
        assign_fn = slim.assign_from_checkpoint_fn(args.solver.pretrained_path,
                                                 m.vision_ops.vars_to_restore)
        assign_fn(s)
        for op in m.copying_ops:
          s.run(op)

      #m.init_fn = lambda s: init_fn_tri(s,m)
      #m.init_fn = init_fn_tri2
      m.init_fn = slim.assign_from_checkpoint_fn(args.solver.pretrained_path,
                                                 m.vision_ops.vars_to_restore)

    # Set up caching of vision features if needed.
    if args.solver.freeze_conv:
      m.train_ops['step_data_cache'] = [m.vision_ops.encoder_output]
    else:
      m.train_ops['step_data_cache'] = []

    # Set up blobs that are needed for the computation in rest of the graph.
    m.ego_map_ops = m.vision_ops.fss_logits
    m.coverage_ops = m.vision_ops.confs_probs
    
    # Zero pad these to make them same size as what the planner expects.
    for i in range(len(m.ego_map_ops)):
      if args.mapper_arch.pad_map_with_zeros_each[i] > 0:
        paddings = np.zeros((5,2), dtype=np.int32)
        paddings[2:4,:] = args.mapper_arch.pad_map_with_zeros_each[i]
        paddings_op = tf.constant(paddings, dtype=tf.int32)
        m.ego_map_ops[i] = tf.pad(m.ego_map_ops[i], paddings=paddings_op)
        m.coverage_ops[i] = tf.pad(m.coverage_ops[i], paddings=paddings_op)
  
  elif task_params.input_type == 'analytical_counts':
    m.ego_map_ops = []; m.coverage_ops = []
    for i in range(len(task_params.map_crop_sizes)):
      ego_map_op = m.input_tensors['step']['analytical_counts_{:d}'.format(i)]
      coverage_op = tf.cast(tf.greater_equal(
          tf.reduce_max(ego_map_op, reduction_indices=[4],
                        keep_dims=True), 1), tf.float32)
      coverage_op = tf.ones_like(ego_map_op) * coverage_op
      m.ego_map_ops.append(ego_map_op)
      m.coverage_ops.append(coverage_op)
      m.train_ops['step_data_cache'] = []
  
  num_steps = task_params.num_steps
  num_goals = task_params.num_goals

  map_crop_size_ops = []
  for map_crop_size in task_params.map_crop_sizes:
    map_crop_size_ops.append(tf.constant(map_crop_size, dtype=tf.int32, shape=(2,)))

  with tf.name_scope('check_size'):
    is_single_step = tf.equal(tf.unstack(tf.shape(m.ego_map_ops[0]), num=5)[1], 1)

  fr_ops = []; value_ops = [];
  fr_intermediate_ops = []; value_intermediate_ops = [];
  crop_value_ops = [];
  resize_crop_value_ops = [];
  confs = []; occupancys = [];

  previous_value_op = None
  updated_state = []; state_names = [];

  for i in range(len(task_params.map_crop_sizes)):
    map_crop_size = task_params.map_crop_sizes[i]
    with tf.variable_scope('scale_{:d}'.format(i)): 
      # Accumulate the map.
      fn = lambda ns: running_combine(
             m.ego_map_ops[i],
             m.coverage_ops[i],
             m.input_tensors['step']['incremental_locs'] * task_params.map_scales[i],
             m.input_tensors['step']['incremental_thetas'],
             m.input_tensors['step']['running_sum_num_{:d}'.format(i)],
             m.input_tensors['step']['running_sum_denom_{:d}'.format(i)],
             m.input_tensors['step']['running_max_denom_{:d}'.format(i)],
             map_crop_size, ns)

      running_sum_num, running_sum_denom, running_max_denom = \
          tf.cond(is_single_step, lambda: fn(1), lambda: fn(num_steps*num_goals))
      updated_state += [running_sum_num, running_sum_denom, running_max_denom]
      state_names += ['running_sum_num_{:d}'.format(i),
                      'running_sum_denom_{:d}'.format(i),
                      'running_max_denom_{:d}'.format(i)]

      # Concat the accumulated map and goal
      occupancy = running_sum_num / tf.maximum(running_sum_denom, 0.001)
      conf = running_max_denom
      # print occupancy.get_shape().as_list()

      # Concat occupancy, how much occupied and goal.
      with tf.name_scope('concat'):
        sh = [-1, map_crop_size, map_crop_size, task_params.map_channels]
        occupancy = tf.reshape(occupancy, shape=sh)
        conf = tf.reshape(conf, shape=sh)

        sh = [-1, map_crop_size, map_crop_size, task_params.goal_channels]
        #Tri: remove goal inputs when perform exploration task
        #goal = tf.reshape(m.input_tensors['step']['ego_goal_imgs_{:d}'.format(i)], shape=sh)
        #to_concat = [occupancy, conf, goal]
        #pdb.set_trace()
        to_concat = [occupancy, conf]

        if previous_value_op is not None:
          to_concat.append(previous_value_op)

        x = tf.concat(to_concat, 3)

      # Pass the map, previous rewards and the goal through a few convolutional
      # layers to get fR.
      fr_op, fr_intermediate_op = fr_v2(
         x, output_neurons=args.arch.fr_neurons,
         inside_neurons=args.arch.fr_inside_neurons,
         is_training=batch_norm_is_training_op, name='fr',
         wt_decay=args.solver.wt_decay, stride=args.arch.fr_stride)

      # Do Value Iteration on the fR
      if args.arch.vin_num_iters > 0:
        value_op, value_intermediate_op = value_iteration_network(
            fr_op, num_iters=args.arch.vin_num_iters,
            val_neurons=args.arch.vin_val_neurons,
            action_neurons=args.arch.vin_action_neurons,
            kernel_size=args.arch.vin_ks, share_wts=args.arch.vin_share_wts,
            name='vin', wt_decay=args.solver.wt_decay)
      else:
        value_op = fr_op
        value_intermediate_op = []

      # Crop out and upsample the previous value map.
      remove = args.arch.crop_remove_each
      if remove > 0:
        crop_value_op = value_op[:, remove:-remove, remove:-remove,:]
      else:
        crop_value_op = value_op
      crop_value_op = tf.reshape(crop_value_op, shape=[-1, args.arch.value_crop_size,
                                                       args.arch.value_crop_size,
                                                       args.arch.vin_val_neurons])
      if i < len(task_params.map_crop_sizes)-1:
        # Reshape it to shape of the next scale.
        previous_value_op = tf.image.resize_bilinear(crop_value_op,
                                                     map_crop_size_ops[i+1],
                                                     align_corners=True)
        resize_crop_value_ops.append(previous_value_op)
      
      occupancys.append(occupancy)
      confs.append(conf)
      value_ops.append(value_op)
      crop_value_ops.append(crop_value_op)
      fr_ops.append(fr_op)
      fr_intermediate_ops.append(fr_intermediate_op)
  
  m.value_ops = value_ops
  m.value_intermediate_ops = value_intermediate_ops
  m.fr_ops = fr_ops
  m.fr_intermediate_ops = fr_intermediate_ops
  m.final_value_op = crop_value_op
  m.crop_value_ops = crop_value_ops
  m.resize_crop_value_ops = resize_crop_value_ops
  m.confs = confs
  m.occupancys = occupancys

  sh = [-1, args.arch.vin_val_neurons*((args.arch.value_crop_size)**2)]
  m.value_features_op = tf.reshape(m.final_value_op, sh, name='reshape_value_op')
  
  # Determine what action to take.
  with tf.variable_scope('action_pred'):
    batch_norm_param = args.arch.pred_batch_norm_param
    if batch_norm_param is not None:
      batch_norm_param['is_training'] = batch_norm_is_training_op
    m.action_logits_op, _ = tf_utils.fc_network(
        m.value_features_op, neurons=args.arch.pred_neurons,
        wt_decay=args.solver.wt_decay, name='pred', offset=0,
        num_pred=task_params.num_actions,
        batch_norm_param=batch_norm_param) 
    m.action_prob_op = tf.nn.softmax(m.action_logits_op)

  init_state = tf.constant(0., dtype=tf.float32, shape=[
      task_params.batch_size, 1, map_crop_size, map_crop_size,
      task_params.map_channels])

  m.train_ops['state_names'] = state_names
  m.train_ops['updated_state'] = updated_state
  m.train_ops['init_state'] = [init_state for _ in updated_state]

  m.train_ops['step'] = m.action_prob_op
  m.train_ops['common'] = [m.input_tensors['common']['orig_maps'],
                           m.input_tensors['common']['goal_loc']]
  m.train_ops['batch_norm_is_training_op'] = batch_norm_is_training_op
  m.loss_ops = []; m.loss_ops_names = [];

  if args.arch.readout_maps:
    with tf.name_scope('readout_maps'):
      all_occupancys = tf.concat(m.occupancys + m.confs, 3)
      readout_maps, probs = readout_general(
          all_occupancys, num_neurons=args.arch.rom_arch.num_neurons,
          strides=args.arch.rom_arch.strides, 
          layers_per_block=args.arch.rom_arch.layers_per_block, 
          kernel_size=args.arch.rom_arch.kernel_size,
          batch_norm_is_training_op=batch_norm_is_training_op,
          wt_decay=args.solver.wt_decay)

      gt_ego_maps = [m.input_tensors['step']['readout_maps_{:d}'.format(i)]
                     for i in range(len(task_params.readout_maps_crop_sizes))]
      m.readout_maps_gt = tf.concat(gt_ego_maps, 4)
      gt_shape = tf.shape(m.readout_maps_gt)
      m.readout_maps_logits = tf.reshape(readout_maps, gt_shape)
      m.readout_maps_probs = tf.reshape(probs, gt_shape)

      # Add a loss op
      m.readout_maps_loss_op = tf.losses.sigmoid_cross_entropy(
          tf.reshape(m.readout_maps_gt, [-1, len(task_params.readout_maps_crop_sizes)]), 
          tf.reshape(readout_maps, [-1, len(task_params.readout_maps_crop_sizes)]),
          scope='loss')
      m.readout_maps_loss_op = 10.*m.readout_maps_loss_op

  ewma_decay = 0.99 if is_training else 0.0
  weight = tf.ones_like(m.input_tensors['train']['action'], dtype=tf.float32,
                        name='weight')
  #m.reg_loss_op, m.data_loss_op, m.total_loss_op, m.acc_ops = \
  #  compute_losses_multi_or(m.action_logits_op,
  #                          m.input_tensors['train']['action'], weights=weight,
  #                          num_actions=task_params.num_actions,
  #                          data_loss_wt=args.solver.data_loss_wt,
  #                          reg_loss_wt=args.solver.reg_loss_wt,
  #                          ewma_decay=ewma_decay)

  #Tri
  m.reg_loss_op, m.data_loss_op, m.total_loss_op, m.acc_ops = \
    rl_compute_losses_multi_or(m.action_logits_op,m.input_tensors['train']['action'],
                            m.input_tensors['train']['action_one'],m.input_tensors['train']['target'], weights=weight,
                            num_actions=task_params.num_actions,
                            data_loss_wt=args.solver.data_loss_wt,
                            reg_loss_wt=args.solver.reg_loss_wt,
                            ewma_decay=ewma_decay)
  
  if args.arch.readout_maps:
    m.total_loss_op = m.total_loss_op + m.readout_maps_loss_op
    m.loss_ops += [m.readout_maps_loss_op]
    m.loss_ops_names += ['readout_maps_loss']

  m.loss_ops += [m.reg_loss_op, m.data_loss_op, m.total_loss_op]
  m.loss_ops_names += ['reg_loss', 'data_loss', 'total_loss']

  if args.solver.freeze_conv:
    vars_to_optimize = list(set(tf.trainable_variables()) -
                            set(m.vision_ops.vars_to_restore))
  else:
    vars_to_optimize = None

  m.lr_op, m.global_step_op, m.train_op, m.should_stop_op, m.optimizer, \
  m.sync_optimizer = tf_utils.setup_training(
      m.total_loss_op, 
      args.solver.initial_learning_rate, 
      args.solver.steps_per_decay,
      args.solver.learning_rate_decay, 
      args.solver.momentum,
      args.solver.max_steps, 
      args.solver.sync, 
      args.solver.adjust_lr_sync,
      args.solver.num_workers, 
      args.solver.task,
      vars_to_optimize=vars_to_optimize,
      clip_gradient_norm=args.solver.clip_gradient_norm,
      typ=args.solver.typ, momentum2=args.solver.momentum2,
      adam_eps=args.solver.adam_eps)

  if args.arch.sample_gt_prob_type == 'inverse_sigmoid_decay':
    m.sample_gt_prob_op = tf_utils.inverse_sigmoid_decay(args.arch.isd_k,
                                                         m.global_step_op)
  elif args.arch.sample_gt_prob_type == 'zero':
    m.sample_gt_prob_op = tf.constant(-1.0, dtype=tf.float32)

  elif args.arch.sample_gt_prob_type.split('_')[0] == 'step':
    step = int(args.arch.sample_gt_prob_type.split('_')[1])
    m.sample_gt_prob_op = tf_utils.step_gt_prob(
        step, m.input_tensors['step']['step_number'][0,0,0])

  m.sample_action_type = args.arch.action_sample_type
  m.sample_action_combine_type = args.arch.action_sample_combine_type

  m.summary_ops = {
      summary_mode: _add_summaries(m, args, summary_mode,
                                   args.summary.arop_full_summary_iters)}
  #Tri
  clonemodel(m)
  set_copying_ops(m)
  set_tmp_params(m)  
  
  #pdb.set_trace()
  m.init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

  #pdb.set_trace()
  m.saver_op = tf.train.Saver(keep_checkpoint_every_n_hours=4,
                              write_version=tf.train.SaverDef.V2)#,var_list=slim.get_variables_to_restore(exclude=['cloned']))


  return m
