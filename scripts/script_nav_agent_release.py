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

r""" Script to train and test the grid navigation agent.
Usage:
  1. Testing a model.
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+bench_test \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r

  2. Training a model (locally).
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+train_train \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r_

  3. Training a model (distributed).
  # See https://www.tensorflow.org/deploy/distributed on how to setup distributed
  # training.
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:/opt/cudnnv51/lib64 \
    PYTHONPATH='.' PYOPENGL_PLATFORM=egl python scripts/script_nav_agent_release.py \
    --config_name cmp.lmap_Msc.clip5.sbpd_d_r2r+train_train \
    --logdir output/cmp.lmap_Msc.clip5.sbpd_d_r2r_ \
    --ps_tasks $num_ps --master $master_name --task $worker_id
"""

import pdb
import sys, os, numpy as np
import copy
import argparse, pprint
import time
import cProfile
import platform


import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables

import logging
from tensorflow.python.platform import gfile
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from cfgs import config_cmp
from cfgs import config_vision_baseline
import datasets.nav_env as nav_env
import src.file_utils as fu 
import src.utils as utils
import tfcode.cmp as cmp 
from tfcode import tf_utils
from tfcode import vision_baseline_lstm
#Tri
from copy import deepcopy

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '',
                    'The address of the tensorflow master')
flags.DEFINE_integer('ps_tasks', 0, 'The number of parameter servers. If the '
                     'value is 0, then the parameters are handled locally by '
                     'the worker.')
flags.DEFINE_integer('task', 0, 'The Task ID. This value is used when training '
                     'with multiple workers to identify each worker.')

flags.DEFINE_integer('num_workers', 1, '')

flags.DEFINE_string('config_name', '', '')

flags.DEFINE_string('logdir', '', '')

flags.DEFINE_integer('solver_seed', 0, '')

flags.DEFINE_integer('delay_start_iters', 20, '')

logging.basicConfig(level=logging.INFO)

def main(_):
  _launcher(FLAGS.config_name, FLAGS.logdir)

def _launcher(config_name, logdir):
  args = _setup_args(config_name, logdir)

  fu.makedirs(args.logdir)

  if args.control.train:
    _train(args)

  if args.control.test:
    _test(args)

def get_args_for_config(config_name):
  configs = config_name.split('.')
  type = configs[0]
  config_name = '.'.join(configs[1:])
  if type == 'cmp':
    args = config_cmp.get_args_for_config(config_name)
    args.setup_to_run = cmp.setup_to_run
    args.setup_train_step_kwargs = cmp.setup_train_step_kwargs

  elif type == 'bl':
    args = config_vision_baseline.get_args_for_config(config_name)
    args.setup_to_run = vision_baseline_lstm.setup_to_run
    args.setup_train_step_kwargs = vision_baseline_lstm.setup_train_step_kwargs

  else:
    logging.fatal('Unknown type: {:s}'.format(type))
  return args

def _setup_args(config_name, logdir):
  args = get_args_for_config(config_name)
  args.solver.num_workers = FLAGS.num_workers
  args.solver.task = FLAGS.task
  args.solver.ps_tasks = FLAGS.ps_tasks
  args.solver.master = FLAGS.master
  args.solver.seed = FLAGS.solver_seed
  args.logdir = logdir
  args.navtask.logdir = None
  return args

##Tri
#def copy_variable_to_graph(org_instance, to_graph, namespace,
#                           copied_variables={},fix_shape=True):
#    """
#    Copies the Variable instance 'org_instance' into the graph
#    'to_graph', under the given namespace.
#    The dict 'copied_variables', if provided, will be updated with
#    mapping the new variable's name to the instance.
#    """
# 
#    if not isinstance(org_instance, tf.Variable):
#        raise TypeError(str(org_instance) + " is not a Variable")
# 
#    #The name of the new variable
#    if namespace != '':
#        new_name = (namespace + '/' +
#                    org_instance.name[:org_instance.name.index(':')])
#    else:
#        new_name = org_instance.name[:org_instance.name.index(':')]
# 
#    #Get the collections that the new instance needs to be added to.
#    #The new collections will also be a part of the given namespace,
#    #except the special ones required for variable initialization and
#    #training.
#    collections = []
#    for name, collection in org_instance.graph._collections.items():
#        if org_instance in collection:
#           # if (name == ops.GraphKeys.VARIABLES or
#            if (name == ops.GraphKeys.GLOBAL_VARIABLES or 
#                name == ops.GraphKeys.TRAINABLE_VARIABLES or
#                namespace == ''):
#                collections.append(name)
#            else:
#                collections.append(namespace + '/' + name)
# 
#    #See if its trainable.
#    trainable = (org_instance in org_instance.graph.get_collection(
#        ops.GraphKeys.TRAINABLE_VARIABLES))
#    #Get the initial value
#    #pdb.set_trace()
#    with org_instance.graph.as_default():
#        temp_session = tf.Session()
#        init_value = temp_session.run(org_instance.initialized_value())
# 
#    #Initialize the new variable
#    with to_graph.as_default():
#        new_var = tf.Variable(init_value,
#                              trainable,
#                              name=new_name,
#                              collections=collections,
#                              validate_shape=fix_shape)
# 
#    #Add to the copied_variables dict
#    copied_variables[new_var.name] = new_var
# 
#    return new_var
# 
# 
#def copy_to_graph(org_instance, to_graph, copied_variables={}, namespace=""):
#    """
#    Makes a copy of the Operation/Tensor instance 'org_instance'
#    for the graph 'to_graph', recursively. Therefore, all required
#    structures linked to org_instance will be automatically copied.
#    'copied_variables' should be a dict mapping pertinent copied variable
#    names to the copied instances.
#     
#    The new instances are automatically inserted into the given 'namespace'.
#    If namespace='', it is inserted into the graph's global namespace.
#    However, to avoid naming conflicts, its better to provide a namespace.
#    If the instance(s) happens to be a part of collection(s), they are
#    are added to the appropriate collections in to_graph as well.
#    For example, for collection 'C' which the instance happens to be a
#    part of, given a namespace 'N', the new instance will be a part of
#    'N/C' in to_graph.
# 
#    Returns the corresponding instance with respect to to_graph.
# 
#    TODO: Order of insertion into collections is not preserved
#    """
# 
#    #The name of the new instance
#    if namespace != '':
#        new_name = namespace + '/' + org_instance.name
#    else:
#        new_name = org_instance.name
# 
#    #If a variable by the new name already exists, return the
#    #correspondng tensor that will act as an input
#    if new_name in copied_variables:
#        return to_graph.get_tensor_by_name(
#            copied_variables[new_name].name)
# 
#    #If an instance of the same name exists, return appropriately
#    try:
#        already_present = to_graph.as_graph_element(new_name,
#                                                    allow_tensor=True,
#                                                    allow_operation=True)
#        return already_present
#    except:
#        pass
# 
#    #Get the collections that the new instance needs to be added to.
#    #The new collections will also be a part of the given namespace.
#    collections = []
#    for name, collection in org_instance.graph._collections.items():
#        if org_instance in collection:
#            if namespace == '':
#                collections.append(name)
#            else:
#                collections.append(namespace + '/' + name)
#     
#    #Take action based on the class of the instance
# 
#    #if isinstance(org_instance, tf.python.framework.ops.Tensor):
#    if isinstance(org_instance, ops.Tensor):
# 
#        #If its a Tensor, it is one of the outputs of the underlying
#        #op. Therefore, copy the op itself and return the appropriate
#        #output.
#        op = org_instance.op
#        new_op = copy_to_graph(op, to_graph, copied_variables, namespace)
#        output_index = op.outputs.index(org_instance)
#        new_tensor = new_op.outputs[output_index]
#        #Add to collections if any
#        for collection in collections:
#            to_graph.add_to_collection(collection, new_tensor)
# 
#        return new_tensor
# 
#    #elif isinstance(org_instance, tf.python.framework.ops.Operation):
#    elif isinstance(org_instance, ops.Operation):
# 
#        op = org_instance
# 
#        #If it has an original_op parameter, copy it
#        if op._original_op is not None:
#            new_original_op = copy_to_graph(op._original_op, to_graph,
#                                            copied_variables, namespace)
#        else:
#            new_original_op = None
# 
#        #If it has control inputs, call this function recursively on each.
#        new_control_inputs = [copy_to_graph(x, to_graph, copied_variables,
#                                            namespace)
#                              for x in op.control_inputs]
# 
#        #If it has inputs, call this function recursively on each.
#        new_inputs = [copy_to_graph(x, to_graph, copied_variables,
#                                    namespace)
#                      for x in op.inputs]
# 
#        #Make a new node_def based on that of the original.
#        #An instance of tensorflow.core.framework.graph_pb2.NodeDef, it
#        #stores String-based info such as name, device and type of the op.
#        #Unique to every Operation instance.
#        new_node_def = deepcopy(op._node_def)
#        #Change the name
#        new_node_def.name = new_name
# 
#        #Copy the other inputs needed for initialization
#        output_types = op._output_types[:]
#        input_types = op._input_types[:]
# 
#        #Make a copy of the op_def too.
#        #Its unique to every _type_ of Operation.
#        op_def = deepcopy(op._op_def)
# 
#        #Initialize a new Operation instance
#        #new_op = tf.python.framework.ops.Operation(new_node_def,
#        new_op = ops.Operation(new_node_def,
#                                                   to_graph,
#                                                   new_inputs,
#                                                   output_types,
#                                                   new_control_inputs,
#                                                   input_types,
#                                                   new_original_op,
#                                                   op_def)
#        #Use Graph's hidden methods to add the op
#        to_graph._add_op(new_op)
#        to_graph._record_op_seen_by_control_dependencies(new_op)
#        #pdb.set_trace()
#        for device_function in reversed(to_graph._device_function_stack):
#            if device_function is not None:
#               new_op._set_device(device_function(new_op))
# 
#        return new_op
# 
#    else:
#        raise TypeError("Could not copy instance: " + str(org_instance))
# 
# 
#def get_copied(original, graph, copied_variables={}, namespace=""):
#    """
#    Get a copy of the instance 'original', present in 'graph', under
#    the given 'namespace'.
#    'copied_variables' is a dict mapping pertinent variable names to the
#    copy instances.
#    """
# 
#    #The name of the copied instance
#    if namespace != '':
#        new_name = namespace + '/' + original.name
#    else:
#        new_name = original.name
# 
#    #If a variable by the name already exists, return it
#    if new_name in copied_variables:
#        return copied_variables[new_name]
# 
#    return graph.as_graph_element(new_name, allow_tensor=True,
#                                  allow_operation=True)
#
#def clonemodel(m):
#  """cloning all the variables & operations to a new graph."""
#  #pdb.set_trace()
#  namespace = "cloned"
#  #cloned_vars = []
#  cloned_vars = {}
#  #cloned_graph = tf.Graph()_
#  #cloned_graph = tf.get_default_graph()
#  graph = tf.get_default_graph()
#  #graph = tf.Graph()
#  #vars_to_restore = slim.get_variables_to_restore()
#  vars_to_restore = slim.get_model_variables()
#  #pdb.set_trace()
#  for var in vars_to_restore[:3]:
#    #new_var = tf.contrib.copy_graph.copy_variable_to_graph(var,cloned_graph,namespace)
#    #cloned_vars.append(new_var)
#    ###new_var = copy_variable_to_graph(var,cloned_graph,namespace,cloned_vars,fix_shape=True)
#    new_var = copy_variable_to_graph(var,graph,namespace,cloned_vars,fix_shape=True)
#
#  #pdb.set_trace()
#  print "finished cloning variables"
#  #cloned_action_logits_op = tf.contrib.copy_graph.copy_op_to_graph(m.action_logits_op,cloned_graph,cloned_vars,namespace) 
#  #cloned_action_logits_op = copy_to_graph(m.action_logits_op,cloned_graph,cloned_vars,namespace)
#  cloned_action_logits_op = copy_to_graph(m.action_logits_op,graph,cloned_vars,namespace)
#  print "finished cloning op"
#  #m.cloned_graph = cloned_graph
#  m.cloned_vars = cloned_vars
#  m.cloned_action_logits_op = cloned_action_logits_op
#  m.cloned_namespace = namespace
#  print "finished cloning model ----------------------"
#  #pdb.set_trace()
#
#def set_copying_ops(m):
#  copying_ops = []
#  model_vars = slim.get_model_variables()
#  for var in model_vars:
#    new_name = (m.cloned_namespace + '/' + var.name)
#    if new_name in m.cloned_vars.keys():
#      new_var = m.cloned_vars[new_name]
#      copy_op = new_var.assign(var)
#      copying_ops.append(copy_op)
#
#  m.copying_ops = copying_ops
#
#def set_tmp_params(m):
#  m.rl_num_explore_steps = 100
#  m.rl_datapool_size = 100000
#  m.rl_datapool = []
#  m.rl_discount_factor = 0.9
#  m.rl_rand_act_prob_start = 1.0
#  m.rl_rand_act_prob_end = 0.1
#  m.rl_rand_act_anneal_time = 5000
#  m.rl_target_net_update_freq = 1000

def _train(args):
  #pdb.set_trace()
  container_name = ""

  R = lambda: nav_env.get_multiplexer_class(args.navtask, args.solver.task)
  m = utils.Foo()
  m.tf_graph = tf.Graph()

  config = tf.ConfigProto()
  config.device_count['GPU'] = 1

  with m.tf_graph.as_default():
    with tf.device(tf.train.replica_device_setter(args.solver.ps_tasks,
                                          merge_devices=True)):
      with tf.container(container_name):
        m = args.setup_to_run(m, args, is_training=True,
                             batch_norm_is_training=True, summary_mode='train')

        train_step_kwargs = args.setup_train_step_kwargs(
            m, R(), os.path.join(args.logdir, 'train'), rng_seed=args.solver.task,
            is_chief=args.solver.task==0,
            num_steps=args.navtask.task_params.num_steps*args.navtask.task_params.num_goals, iters=1,
            train_display_interval=args.summary.display_interval,
            dagger_sample_bn_false=args.arch.dagger_sample_bn_false)

        delay_start = (args.solver.task*(args.solver.task+1))/2 * FLAGS.delay_start_iters
        logging.error('delaying start for task %d by %d steps.',
                      args.solver.task, delay_start)

        #Tri
        #clonemodel(m)
        #set_copying_ops(m)
        #set_tmp_params(m)
        #generating data for testing the learning process during training
        #obj = train_step_kwargs['obj']
        rng_data = train_step_kwargs['rng_data']
        #m.e1 = obj.sample_env(rng_data)
        #m.init_env_state1 = m.e1.reset(rng_data)
        #m.e2 = obj.sample_env(rng_data)
        #m.init_env_state2 = m.e2.reset(rng_data)
        m.rng_data = deepcopy(rng_data)
        

        additional_args = {}
        final_loss = slim.learning.train(
            train_op=m.train_op,
            logdir=args.logdir,
            master=args.solver.master,
            is_chief=args.solver.task == 0,
            number_of_steps=args.solver.max_steps,
            train_step_fn=tf_utils.train_step_custom_online_sampling,
            train_step_kwargs=train_step_kwargs,
            global_step=m.global_step_op,
            init_op=m.init_op,
            init_fn=m.init_fn,
            sync_optimizer=m.sync_optimizer,
            saver=m.saver_op,
            startup_delay_steps=delay_start,
            summary_op=None, session_config=config, **additional_args)

def _test(args):
  #pdb.set_trace();
  args.solver.master = ''
  container_name = ""
  checkpoint_dir = os.path.join(format(args.logdir))
  logging.error('Checkpoint_dir: %s', args.logdir)

  config = tf.ConfigProto();
  config.device_count['GPU'] = 1;

  m = utils.Foo()
  m.tf_graph = tf.Graph()

  rng_data_seed = 0; rng_action_seed = 0;
  R = lambda: nav_env.get_multiplexer_class(args.navtask, rng_data_seed)
  with m.tf_graph.as_default():
    with tf.container(container_name):
      m = args.setup_to_run(
        m, args, is_training=False,
        batch_norm_is_training=args.control.force_batchnorm_is_training_at_test,
        summary_mode=args.control.test_mode)
      train_step_kwargs = args.setup_train_step_kwargs(
        m, R(), os.path.join(args.logdir, args.control.test_name),
        rng_seed=rng_data_seed, is_chief=True,
        num_steps=args.navtask.task_params.num_steps*args.navtask.task_params.num_goals,
        iters=args.summary.test_iters, train_display_interval=None,
        dagger_sample_bn_false=args.arch.dagger_sample_bn_false)

      saver = slim.learning.tf_saver.Saver(variables.get_variables_to_restore())

      sv = slim.learning.supervisor.Supervisor(
          graph=ops.get_default_graph(), logdir=None, init_op=m.init_op,
          summary_op=None, summary_writer=None, global_step=None, saver=m.saver_op)

      last_checkpoint = None
      reported = False
      while True:
        last_checkpoint_ = None
        while last_checkpoint_ is None:
          last_checkpoint_ = slim.evaluation.wait_for_new_checkpoint(
            checkpoint_dir, last_checkpoint, seconds_to_sleep=10, timeout=60)
        if last_checkpoint_ is None: break

        last_checkpoint = last_checkpoint_
        checkpoint_iter = int(os.path.basename(last_checkpoint).split('-')[1])

        logging.info('Starting evaluation at %s using checkpoint %s.',
                     time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()),
                     last_checkpoint)

        if (args.control.only_eval_when_done == False or 
            checkpoint_iter >= args.solver.max_steps):
          start = time.time()
          logging.info('Starting evaluation at %s using checkpoint %s.', 
                       time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()),
                       last_checkpoint)

          with sv.managed_session(args.solver.master, config=config,
                                  start_standard_services=False) as sess:
            sess.run(m.init_op)
            sv.saver.restore(sess, last_checkpoint)
            sv.start_queue_runners(sess)
            if args.control.reset_rng_seed:
              train_step_kwargs['rng_data'] = [np.random.RandomState(rng_data_seed),
                                               np.random.RandomState(rng_data_seed)]
              train_step_kwargs['rng_action'] = np.random.RandomState(rng_action_seed)
            vals, _ = tf_utils.train_step_custom_online_sampling(
                sess, None, m.global_step_op, train_step_kwargs,
                mode=args.control.test_mode)
            should_stop = False

            if checkpoint_iter >= args.solver.max_steps: 
              should_stop = True

            if should_stop:
              break

if __name__ == '__main__':
  app.run()
