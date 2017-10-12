import numpy as np
import tensorflow as tf
import mdp.gridworld as gridworld
import mdp.value_iteration as value_iteration
import img_utils
import tf_utils
from utils import *



class DeepIRLP:


  def __init__(self, n_input, lr, n_h1=400, n_h2=300, l2=10, name='deep_irl_policy'):
    self.n_input = n_input
    self.lr = lr
    self.n_h1 = n_h1
    self.n_h2 = n_h2
    self.name = name
    with tf.device('/gpu:1'):
      self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
      self.input_s, self.input_inv, self.reward, self.theta = self._build_network(self.name)
      self.optimizer = tf.train.GradientDescentOptimizer(lr)
      
      self.grad_r = tf.placeholder(tf.float32, [None, 1])
      self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta])
      self.grad_l2 = tf.gradients(self.l2_loss, self.theta)

      self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r)
      # apply l2 loss gradients
      self.grad_theta = [tf.add(l2*self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))]
      self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0)

      self.grad_norms = tf.global_norm(self.grad_theta)
      self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta))
    self.sess.run(tf.global_variables_initializer())


  def _build_network(self, name):
    input_s = tf.placeholder(tf.float32, [None, self.n_input])
    input_inv = tf.placeholder(tf.float32, [None, self.n_input])
    img_in = tf.reshape(input_s, shape=[-1, 1, 4, 1])
    img_inv = tf.reshape(input_inv, shape=[-1, 1, 4, 1])
    with tf.variable_scope(name):
      cnv1 = tf_utils.conv2d(img_in, 2, (2,2))
      # max_conv_p = tf_utils.max_pool(cnv1_p)
      fltn_conv = tf_utils.flatten(cnv1)
      fc1 = tf_utils.fc(fltn_conv, self.n_h2, scope="fc1", activation_fn=tf.nn.elu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))

      cnv1_inv = tf_utils.conv2d(img_inv, 2, (2, 2))
      # max_conv_p = tf_utils.max_pool(cnv1_p)
      fltn_conv_inv = tf_utils.flatten(cnv1_inv)
      fc1_inv = tf_utils.fc(fltn_conv_inv, self.n_h2, scope="fc1_inv", activation_fn=tf.nn.elu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))

      subt = tf.subtract(fc1, fc1_inv)
      # blah = tf.multiply(tf.divide(fc2, fc1_p), 0.35)
      # comb = tf.concat([fc1, fc1_inv], 1)
      fc_p1 = tf_utils.fc(subt, 2*self.n_h1, scope="fc_p1", activation_fn=tf.nn.elu,
        initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      # fc_p2 = tf_utils.fc(fc_p1, self.n_h2, scope="fc_p2", activation_fn=tf.nn.elu,
      #   initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_IN"))
      reward = tf_utils.fc(fc_p1, 1, scope="reward")
    theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    return input_s, input_inv, reward, theta


  def get_theta(self):
    return self.sess.run(self.theta)


  def get_rewards(self, states, states_inv):
    rewards = self.sess.run(self.reward, feed_dict={self.input_s: states, self.input_inv: states_inv})
    return rewards


  def apply_grads(self, feat_map, feat_map_inv, grad_r):
    grad_r = np.reshape(grad_r, [-1, 1])
    feat_map = np.reshape(feat_map, [-1, self.n_input])
    feat_map_inv = np.reshape(feat_map_inv, [-1, self.n_input])
    _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms], 
      feed_dict={self.grad_r: grad_r, self.input_s: feat_map, self.input_inv: feat_map_inv})
    return grad_theta, l2_loss, grad_norms



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
  # training 
  for iteration in range(n_iters):
    if iteration % (n_iters/10) == 0:
      print('iteration: {}'.format(iteration))
    
    # compute the reward matrix
    rewards = nn_r.get_rewards(feat_map, feat_map_inv)
    
    # compute policy
    values, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=True)

    
    # compute expected svf
    mu_exp = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True)
    
    # compute gradients on rewards:
    grad_r = mu_D - mu_exp

    # apply gradients to the neural network
    grad_theta, l2_loss, grad_norm = nn_r.apply_grads(feat_map, feat_map_inv, grad_r)
    

  rewards = nn_r.get_rewards(feat_map, feat_map_inv)
  # print(rewards)
  # return sigmoid(normalize(rewards))
  return normalize(rewards)





