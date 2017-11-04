import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
from utils import *



class DeepIRLP:


  def __init__(self, n_input, lr, n_h1=400, n_h2=300, l1=10, l2=10, name='deep_siamese_irl_policy', log_dir='/tmp/tensorflow/siamese'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    #self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    self.sess = tf.InteractiveSession()
    # self.input_s, self.input_inv, self.reward, self.theta, self.keep_prob = self._build_network(self.name)

    self.input_s, self.input_inv, self.reward, self.theta = self._build_network(self.name)
    self.optimizer = tf.train.GradientDescentOptimizer(lr)
    
    with tf.name_scope('reward_grad'):
      self.grad_r = tf.placeholder(tf.float32, [None, 1])

    with tf.name_scope("add_l1"):
      l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=0.005, scope=None
      )
      self.l1_loss = tf.contrib.layers.apply_regularization(l1_regularizer, self.theta)

    with tf.name_scope('add_l2'):
      self.l2_loss = tf.add_n([tf.nn.l2_loss(v, name) for v in self.theta])

    with tf.name_scope('weigths_grad_wrt_l1'):
      self.grad_l1 = tf.gradients(self.l1_loss, self.theta)

    with tf.name_scope('weigths_grad_wrt_l2'):
      self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

    with tf.name_scope('weigths_grads'):
      self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
      # apply l2 loss gradients
      self.grad_theta = [tf.add(self.grad_l1[i], self.grad_theta[i]) for i in range(len(self.grad_l1))]
      self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
      self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

    with tf.name_scope('grad_norms'):
      self.grad_norms = tf.global_norm(self.grad_theta)

    with tf.name_scope('train'):
      self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))

    self.merged = tf.summary.merge_all()
    self.sess.run(tf.global_variables_initializer())

  def reset_graph(self):
      tf.reset_default_graph()

  def variable_summaries(self, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
      tf.summary.scalar('stddev', stddev)
      tf.summary.scalar('max', tf.reduce_max(var))
      tf.summary.scalar('min', tf.reduce_min(var))
      tf.summary.histogram('histogram', var)

  def conv2d(self, x, n_kernel, k_sz, stride=1, name='normal'):
    """convolutional layer with relu activation wrapper
    Args:
      x:          4d tensor [batch, height, width, channels]
      n_kernel:   number of kernels (output size)
      k_sz:       2d array, kernel size. e.g. [8,8]
      stride:     stride
    Returns
      a conv2d layer
    """
    with tf.name_scope(name):
      with tf.name_scope('weights'):
        W = tf.Variable(tf.random_normal([k_sz[0], k_sz[1], int(x.get_shape()[3]), n_kernel]))
        self.variable_summaries(W)
      with tf.name_scope('biases'):
        b = tf.Variable(tf.random_normal([n_kernel]))
        self.variable_summaries(b)
      # - strides[0] and strides[1] must be 1
      # - padding can be 'VALID'(without padding) or 'SAME'(zero padding)
      #     - http://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
      with tf.name_scope('conv'):
        conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, b)  # add bias term
        tf.summary.histogram('pre_activations', conv)
      
      activations = tf.nn.relu(conv, name='activation')
      tf.summary.histogram('activation', activations)
      # rectified linear unit: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
      return activations


  def fc(self, x, n_output, scope="fc", activation_fn=None, initializer=None):
    """fully connected layer with relu activation wrapper
    Args
      x:          2d tensor [batch, n_input]
      n_output    output size
    """
    with tf.name_scope(scope):
      if initializer is None:
        # default initialization
        with tf.name_scope('weights'):
          W = tf.Variable(tf.random_normal([int(x.get_shape()[1]), n_output]))
          self.variable_summaries(W)
        with tf.name_scope('biases'):
          b = tf.Variable(tf.random_normal([n_output]))
          self.variable_summaries(b)
      else:
        with tf.name_scope('weights'):
          W = tf.get_variable("W", shape=[int(x.get_shape()[1]), n_output], initializer=initializer)
          self.variable_summaries(W)
        with tf.name_scope('biases'):
          b = tf.get_variable("b", shape=[n_output],
                            initializer=tf.constant_initializer(.0, dtype=tf.float32))
          self.variable_summaries(b)
      with tf.name_scope('Wx_plus_b'):
        fc1 = tf.add(tf.matmul(x, W), b)
        tf.summary.histogram('pre_activations', fc1)

      if not activation_fn is None:
        activations = activation_fn(fc1)
        tf.summary.histogram('activations', activations)
      
      return fc1


  def flatten(self, x, name='normal'):
    """flatten a 4d tensor into 2d
    Args
      x:          4d tensor [batch, height, width, channels]
    Returns a flattened 2d tensor
    """
    with tf.name_scope(name+'flatten_reshape'):
      x_reshaped = tf.reshape(x, [-1, int(x.get_shape()[1] * x.get_shape()[2] * x.get_shape()[3])])
      # tf.summary.image(name+'_flatten', x_reshaped, 100)
    
    return x_reshaped

  def max_pool(self, x, k_sz=[2, 2]):
    """max pooling layer wrapper
    Args
      x:      4d tensor [batch, height, width, channels]
      k_sz:   The size of the window for each dimension of the input tensor
    Returns
      a max pooling layer
    """
    return tf.nn.max_pool(
        x, ksize=[
            1, k_sz[0], k_sz[1], 1], strides=[
            1, k_sz[0], k_sz[1], 1], padding='SAME')

  def _build_network(self, name):
    with tf.name_scope('input'):
      input_s = tf.placeholder(tf.float32, [None, self.n_input], name='input-s')
      input_inv = tf.placeholder(tf.float32, [None, self.n_input], name='inverted_input-s')
    with tf.name_scope('input_reshape'):
      img_in = tf.reshape(input_s, shape=[-1, 1, self.n_input, 1])
      # tf.summary.image('input', img_in, 100)
      img_inv = tf.reshape(input_inv, shape=[-1, 1, self.n_input, 1])
      # tf.summary.image('inverted-input', img_inv, 100)
    with tf.variable_scope(name):
      cnv1 = self.conv2d(img_in, 2, (2,2), name='siamese_normal')
      # avg_conv_1 = tf.nn.pool(cnv1, [2,2], pooling_type='AVG', padding='SAME', strides=[2,2])
      # avg_conv_1 = tf.nn.pool(cnv1, [2,2], pooling_type='AVG', padding='SAME')
      fltn_conv = self.flatten(cnv1, name='siamese_normal')
      # fc1 = tf_utils.fc(fltn_conv, self.n_h2, scope="fc1", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))

      cnv1_inv = self.conv2d(img_inv, 2, (2, 2), name='siamese_inverted')
      # max_conv_2 = tf.nn.pool(tf.negative(cnv1_inv), [2,2], pooling_type='MAX', padding='SAME')
      # max_conv_2 = tf.nn.pool(cnv1_inv, [2,2], pooling_type='MAX', padding='SAME')
      fltn_conv_inv = self.flatten(cnv1_inv, name='siamese_inverted')
      # fc1_inv = tf_utils.fc(fltn_conv_inv, self.n_h2, scope="fc1_inv", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))

      subt = tf.subtract(fltn_conv_inv, fltn_conv)
      subt_in = tf.reshape(subt, [-1, 1, subt.shape.as_list()[1], 1])
      cnv3 = self.conv2d(subt_in, 2, (2, 2), name='third_cnn')
      fltn_3 = self.flatten(cnv3, name='third_normal')
    #   with tf.name_scope('dropout'):
    #     keep_prob = tf.placeholder(tf.float32)
    #     tf.summary.scalar('dropout_keep_probability', keep_prob)
    #     dropped = tf.nn.dropout(fltn_3, keep_prob)
      # comb = tf.concat([fltn_conv, fltn_conv_inv], 1)
      # fc_p1 = tf_utils.fc(comb, fltn_conv.shape[1], scope="fc_p1", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      fc_p1 = self.fc(fltn_3, self.n_h1, scope="fc_p1", activation_fn=tf.nn.elu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      fc_p2 = self.fc(fc_p1, self.n_h2, scope="fc_p2", activation_fn=tf.nn.elu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      # with tf.name_scope('dropout'):
      #   keep_prob = tf.placeholder(tf.float32)
      #   tf.summary.scalar('dropout_keep_probability', keep_prob)
      #   dropped = tf.nn.dropout(fc_p2, keep_prob)
      # subt = tf.subtract(fc1, fc1_inv)
      # comb = tf.concat([fc1, fc1_inv], 1)
      # fc_p = tf_utils.fc(comb, fc1.shape, scope="fc_p", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      # fc_p1 = tf_utils.fc(fc_p, 2*self.n_h1, scope="fc_p1", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      # blah = tf.multiply(tf.divide(fc2, fc1_p), 0.35)
      # comb = tf.concat([fc1, fc1_inv], 1)
      # fc_p1 = tf_utils.fc(subt, self.n_h2, scope="fc_p1", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN2"))
      # fc_p2 = tf_utils.fc(fc_p1, self.n_h2, scope="fc_p2", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      reward = self.fc(fc_p2, 1, scope="reward")
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    # return input_s, input_inv, reward, theta, keep_prob
    return input_s, input_inv, reward, theta

  def get_theta(self):
    return self.sess.run(self.theta)


  def get_rewards(self, states, states_inv, keep_prob):
    # rewards, mergede = self.sess.run([self.reward, self.merged], feed_dict={self.input_s: states, self.input_inv: states_inv, self.keep_prob:keep_prob})
    rewards, mergede = self.sess.run([self.reward, self.merged], feed_dict={self.input_s: states, self.input_inv: states_inv})
    return rewards, mergede


  def apply_grads(self, feat_map, feat_map_inv, grad_r, keep_prob):
    grad_r = np.reshape(grad_r, [-1, 1])
    feat_map = np.reshape(feat_map, [-1, self.n_input])
    feat_map_inv = np.reshape(feat_map_inv, [-1, self.n_input])
    # _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
    #   feed_dict={self.grad_r: grad_r, self.input_s: feat_map, self.input_inv: feat_map_inv, self.keep_prob:keep_prob})
    # _, grad_theta, l1_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l1_loss, self.grad_norms], 
    #   feed_dict={self.grad_r: grad_r, self.input_s: feat_map, self.input_inv: feat_map_inv, self.keep_prob:keep_prob})
    # _, grad_theta, l1_loss, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l1_loss, self.l2_loss, self.grad_norms], 
    #   feed_dict={self.grad_r: grad_r, self.input_s: feat_map, self.input_inv: feat_map_inv, self.keep_prob:keep_prob})
    _, grad_theta, l1_loss, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l1_loss, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map, self.input_inv: feat_map_inv})
    return grad_theta, l1_loss, l2_loss, grad_norms



def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
  """compute the expected states visition frequency p(s| theta, T) 
  using dynamic programming

  inputs:
    P_a     NxNxN_ACTIONS matrix - transition dynamics
    gamma   float - discount factor
    trajs   list of list of Steps - collected from expert
    policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

  
  returns:
    p       Nx1 vector - state visitation frequencies
  """
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  T = len(trajs[0])
  # mu[s, t] is the prob of visiting state s at time t
  mu = np.zeros([N_STATES, T]) 

  for traj in trajs:
    mu[traj[0], 0] += 1
  mu[:,0] = mu[:,0]/len(trajs)

  for s in range(N_STATES):
    for t in range(T-1):
      if deterministic:
        mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
      else:
        mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
  p = np.sum(mu, 1)
  return p


def demo_svf(trajs, n_states):
  """
  compute state visitation frequences from demonstrations
  
  input:
    trajs   list of list of Steps - collected from expert
  returns:
    p       Nx1 vector - state visitation frequences   
  """

  p = np.zeros(n_states)
  for traj in trajs:
    for step in traj:
      p[step.cur_state] += 1
  p = p/len(trajs)
  return p

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions
        are ints. NumPy array with shape (T, L, 2) where T is the number of
        trajectories and L is the trajectory length.
    -> State visitation frequencies vector with shape (N,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def deep_siamese_maxent_irl(feat_map, feat_map_inv, P_a, gamma, trajs, trajs_inv, lr, n_iters):
  """
  Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

  inputs:
    feat_map    NxD matrix - the features for each state
    P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
                                       landing at state s1 when taking action 
                                       a at state s0
    gamma       float - RL discount factor
    trajs       a list of demonstrations
    lr          float - learning rate
    n_iters     int - number of optimization steps

  returns
    rewards     Nx1 vector - recoverred state rewards
  """

  # tf.set_random_seed(1)
  
  N_STATES, _, N_ACTIONS = np.shape(P_a)

  # init nn model
  nn_r = DeepIRLP(feat_map.shape[1], lr, 3, 3)

  # find state visitation frequencies using demonstrations
  mu_D = find_svf(N_STATES, trajs)
  values = np.zeros(feat_map.shape[0])

  log_dir='/tmp/tensorflow/siamese'

  train_writer = tf.summary.FileWriter(log_dir + '/train', nn_r.sess.graph)
  test_writer = tf.summary.FileWriter(log_dir + '/test')
  # training 
  for iteration in range(n_iters):
    if iteration % (n_iters/10) == 0:
      print('iteration: {}'.format(iteration))
      
    
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    # compute the reward matrix
    rewards, summary = nn_r.get_rewards(feat_map, feat_map_inv, 0.75)
  
    train_writer.add_run_metadata(run_metadata, 'step%03d' % iteration)
    train_writer.add_summary(summary, iteration)
    # compute policy
    values, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)

    
    # compute expected svf
    mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
    
    # compute gradients on rewards:
    with tf.name_scope('error'):
      grad_r = mu_D - mu_exp
      tf.summary.scalar('grad_r', grad_r)

    # apply gradients to the neural network
    # grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, feat_map_inv, grad_r, 0.5)
    # grad_theta, l1_loss, grad_norm = nn_r.apply_grads(feat_map, feat_map_inv, grad_r, 0.5)
    grad_theta, l1_loss, l2_loss, grad_norm = nn_r.apply_grads(feat_map, feat_map_inv, grad_r, 0.75)

  rewards, mergede = nn_r.get_rewards(feat_map, feat_map_inv, 1.0)
  test_writer.add_summary(mergede, 0)

  train_writer.close()
  test_writer.close()
  nn_r.reset_graph()
  # print(rewards)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)





