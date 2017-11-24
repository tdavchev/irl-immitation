import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple


import img_utils
from mdp import objectworld
# from mdp import gridworld
# from mdp import value_iteration
import google_drive_upload
from value_iterationn import find_policy
from value_iterationn import find_inverted_policy
from deep_siamese_maxent_irl import *
from deep_maxent_irl import *
from maxent_irl import *
from utils import *
from lp_irl import *
Step = namedtuple('Step','cur_state action next_state reward done')


PARSER = argparse.ArgumentParser(description=None)
PARSER.add_argument('-hei', '--height', default=12, type=int, help='height of the gridworld')
PARSER.add_argument('-wid', '--width', default=12, type=int, help='width of the gridworld')
PARSER.add_argument('-exp_no', '--exp_no', default=1, type=int, help='experiment number')
PARSER.add_argument('-g', '--gamma', default=0.9, type=float, help='discount factor')
PARSER.add_argument('-a', '--act_random', default=0.3, type=float, help='probability of acting randomly')
PARSER.add_argument('-t', '--n_trajs', default=200, type=int, help='number of expert trajectories')
PARSER.add_argument('-l', '--l_traj', default=20, type=int, help='length of expert trajectory')
PARSER.add_argument('--rand_start', dest='rand_start', action='store_true', help='when sampling trajectories, randomly pick start positions')
PARSER.add_argument('--no-rand_start', dest='rand_start',action='store_false', help='when sampling trajectories, fix start positions')
PARSER.set_defaults(rand_start=True)
PARSER.add_argument('-lr', '--learning_rate', default=0.02, type=float, help='learning rate')
PARSER.add_argument('-ni', '--n_iters', default=20, type=int, help='number of iterations')
ARGS = PARSER.parse_args()
print(ARGS)


GAMMA = ARGS.gamma
ACT_RAND = ARGS.act_random
R_MAX = 1 # the constant r_max does not affect much the recoverred reward distribution
H = ARGS.height
W = ARGS.width
N_TRAJS = ARGS.n_trajs
L_TRAJ = ARGS.l_traj
RAND_START = ARGS.rand_start
LEARNING_RATE = ARGS.learning_rate
N_ITERS = ARGS.n_iters
EXP_NO = ARGS.exp_no


def main(grid_size, discount, n_objects, n_colours, n_trajectories, epochs,
         learning_rate, structure, exp_no):

    wind = 0.3
    trajectory_length = 8
    l1 = l2 = 0

    ow = objectworld.Objectworld(grid_size, n_objects, n_colours, wind,
                                    discount)
    for key in ow.objects.keys():
        print(ow.objects[key])
        print(key)
        print("...")
    print("The objects ^^^")
    rewards_gt = np.array([ow.reward(s) for s in range(ow.n_states)])
    policy_gt = find_policy(ow.n_states, ow.n_actions, ow.transition_probability,
                            rewards_gt, ow.discount, stochastic=False)
    trajs = ow.generate_trajectories(N_TRAJS, L_TRAJ, lambda s: policy_gt[s])
    feat_map = ow.feature_matrix(ow.objects, discrete=False)

    # rewards_inv = np.array([ow.inverse_reward(s_inv) for s_inv in range(ow.n_states)])
    # policy_inv = find_inverted_policy(ow.n_states, ow.n_actions, ow.transition_probability,
    #                      rewards_inv, ow.discount, stochastic=False)
    # trajs_inv = ow.generate_inverse_trajectories(N_TRAJS, L_TRAJ, lambda s_inv: policy_inv[s_inv])
    # feat_map_inv = ow.inv_feature_matrix(ow.inverted_objects, discrete=False)

    # feat_map_inv = ow.feature_matrix(ow.objects, discrete=False)
    print('LP IRL training ..')
    rewards_lpirl = lp_irl(ow.transition_probability, policy_gt, gamma=0.3, l1=10, R_max=R_MAX)
#    print('Max Ent IRL training ..')
#    rewards_maxent = maxent_irl(feat_map, ow.transition_probability, GAMMA, trajs, LEARNING_RATE*2, N_ITERS*2)
#    print('Deep Max Ent IRL training ..')
#    rewards_deep = deep_maxent_irl(feat_map, ow.transition_probability, GAMMA, trajs, LEARNING_RATE, N_ITERS)
    #print('Deep Siamese Max Ent IRL training ..')
    #rewards = deep_siamese_maxent_irl(feat_map, feat_map_inv, ow.transition_probability, GAMMA, trajs, trajs_inv, LEARNING_RATE, N_ITERS)

    # plots
    #print("na maika ti v putkata?")
    plt.figure(figsize=(20,5))
    plt.subplot(1, 5, 1)
    img_utils.heatmap2d(np.reshape(rewards_gt, (H,W), order='F'), 'Ground Truth', block=False, text=False)
#    plt.subplot(1, 5, 2)
#    img_utils.heatmap2d(np.reshape(rewards_lpirl, (H,W), order='F'), 'LP', block=False, text=False)
#    plt.subplot(1, 5, 3)
#    img_utils.heatmap2d(np.reshape(rewards_maxent, (H,W), order='F'), 'Maxent', block=False, text=False)
#    plt.subplot(1, 5, 4)
#    img_utils.heatmap2d(np.reshape(rewards_deep, (H,W), order='F'), 'Deep Maxent', block=False, text=False)
    #plt.subplot(1, 5, 5)
    #img_utils.heatmap2d(np.reshape(rewards, (H,W), order='F'), 'Deep Siamese Maxent', block=False, text=False)
    plt.savefig('irl-immitation_' + str(exp_no) + '.png')
    #google_drive_upload.upload()




if __name__ == "__main__":
    if tf.gfile.Exists('/tmp/tensorflow/siamese'):
        tf.gfile.DeleteRecursively('/tmp/tensorflow/siamese')
    tf.gfile.MakeDirs('/tmp/tensorflow/siamese')
    main(12, 0.9, 15, 2, 20, 50, 0.01, (3, 3), EXP_NO)
