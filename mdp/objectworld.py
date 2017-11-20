"""
Implements the objectworld MDP described in Levine et al. 2011.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import math
from itertools import product

import numpy as np
import numpy.random as rn

from mdp.gridworldd import Gridworld

class OWObject(object):
    """
    Object in objectworld.
    """

    def __init__(self, inner_colour, outer_colour):
        """
        inner_colour: Inner colour of object. int.
        outer_colour: Outer colour of object. int.
        -> OWObject
        """

        self.inner_colour = inner_colour
        self.outer_colour = outer_colour

    def __str__(self):
        """
        A string representation of this object.

        -> __str__
        """

        return "<OWObject (In: {}) (Out: {})>".format(self.inner_colour,
                                                      self.outer_colour)

    def get_inner_colour(self):
        return self.inner_colour

    def get_outer_colour(self):
        return self.outer_colour

class Objectworld(Gridworld):
    """
    Objectworld MDP.
    """

    def __init__(self, grid_size, n_objects, n_colours, wind, discount):
        """
        grid_size: Grid size. int.
        n_objects: Number of objects in the world. int.
        n_colours: Number of colours to colour objects with. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Objectworld
        """

        super(Objectworld, self).__init__(grid_size, wind, discount)

        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
        self.n_actions = len(self.actions)
        self.n_objects = n_objects
        self.n_colours = n_colours

        # Generate objects.
        self.objects = {}
        self.inverted_objects = {}
        # # self.wrong_objects = {}
        # listo = [(6, 9), (5, 3), (2, 1), (5, 4), (7, 9), (9, 5), (2, 8), (2, 3), (4, 3), (7, 7)]
        # objo = [(1, 1), (1, 1), (1, 0), (0, 1), (0, 1), (0, 1), (0, 1), (0, 0), (0, 1), (0, 1)]
        # example 3
        # listo = [(6, 9), (2, 3), (4, 1), (2, 8), (5, 9), (5, 4), (7, 4), (1, 3), (7, 7), (0, 5)]
        # objo = [(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (0, 1), (1, 1), (0, 1), (0, 1)]
        # example 4
        # listo = [(4, 5), (7, 3), (3, 3), (1, 2), (2, 8), (8, 8), (9, 7), (3, 4), (3, 8), (5, 6), (6, 8), (6, 0), (6, 7), (0, 1), (1, 6)]
        # objo = [(2, 1), (2, 0), (1, 1), (1, 0), (0, 0), (0, 0), (1, 0), (2, 0), (2, 0), (0, 2), (0, 2), (0, 1), (1, 2), (2, 0), (0, 2)]
        # listo = [(0, 0), (1, 0), (3, 0), (4, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0),
        #          (0, 1), (1,1), (3,1), (4, 1), (5, 1), (1, 2), (3, 2), (4, 2), (4, 3), (9, 4), (8, 5), (9, 5),
        # #          (5, 6), (6, 6), (7, 6), (8, 6), (9, 6), (6, 5)]
        # objo = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),
        #         (1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0),(1, 0), (0, 1)]
        # listo = [(6, 5)]
        # listo = [(3, 3), (9, 10)] # so x goes down the y axis and y goes towards the x axis ...
        # 13x13
        # listo = [(0, 0), (0, 1), (0, 3), (0, 4), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13),
        #          (1, 0), (1,1), (1,3), (1, 4), (1, 5), (2, 1), (2, 3), (2, 4), (3, 3), (4, 9), (5, 8), (5, 9),
        #          (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (4, 6)]

        # objo = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
        #         (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
        #         (1, 0), (1, 0), (1, 0), (0, 1)]

        # 12x12

        listo = [(0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
            (1, 0), (1, 1), (1, 2), (2,0), (2,1), (2, 2), (3, 0), (3, 1), (3, 2),
            (4, 10), (4, 11), (5, 9), (5, 10), (5, 11), (6, 9), (6, 10), (6, 11),
            (7, 9), (7, 10), (7, 11),
            (8, 5), (8, 6), (8, 7), (8, 9), (8, 10), (8, 11), (1, 6)]
    
        objo = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
        (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
                (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 1)]

        # listo = [(0, 0), (0, 1), (0, 2), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        #         (1, 0), (1, 1), (1, 2), (1, 11),
        #         (2, 9), (2, 10), (2, 11),
        #         (3, 6), (3, 7), (3, 9), (3, 10), (3, 11), (1, 6)]
        
        # objo = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
        #         (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 1)]

        # 7x7
        # listo = [(0, 0), (0, 1), (0, 4), (0, 5), (0, 6),
        #          (1, 0), (1,1), (1,2), (2, 0), (2, 1), (3, 0), (4, 5), (5, 4), (5, 5),
        #          (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (5, 3)]

        
        # objo = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0),
        #         (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 1)]
        for i in range(len(listo)):
            obj = OWObject(objo[i][0], objo[i][1])
            self.objects[listo[i][0], listo[i][1]] = obj
        # for _ in range(self.n_objects):
        #     obj = OWObject(rn.randint(self.n_colours),
        #                    rn.randint(self.n_colours))

        #     while True:
        #         x = rn.randint(self.grid_size)
        #         y = rn.randint(self.grid_size)

        #         if (x, y) not in self.objects:
        #             break

        #     self.objects[x, y] = obj

        # for _ in range(self.n_objects):
        #     obj = OWObject(rn.randint(self.n_colours),
        #                    rn.randint(self.n_colours))

        #     while True:
        #         x = rn.randint(self.grid_size)
        #         y = rn.randint(self.grid_size)

        #         if (x, y) not in self.inverted_objects:
        #             break

        #     self.inverted_objects[x, y] = obj

        self.invert_world()
        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for j in range(self.n_actions)]
              for k in range(self.n_states)]
             for i in range(self.n_states)])

    def invert_world(self):
        # tam kade e s obekt e bez i obratno
        for state in range(self.grid_size**2):
            if self.int_to_point(state) not in self.objects.keys():
                inv_pt = self.int_to_point(state)
                self.inverted_objects[inv_pt[0], inv_pt[1]] = OWObject(rn.randint(self.n_colours), rn.randint(self.n_colours))

        # bukvalno obrushtash sveta nadolo s glavata
        # for key in self.objects.keys():
        #     obj = self.point_to_int(key)
        #     inv_obj = (self.grid_size**2-1) - obj
        #     inv_pt = self.int_to_point(inv_obj)
        #     self.inverted_objects[inv_pt[0], inv_pt[1]] = OWObject(self.objects[key].get_inner_colour(), self.objects[key].get_outer_colour())

    def feature_vector(self, i, objects, discrete=True):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """

        sx, sy = self.int_to_point(i)

        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) in objects:
                    dist = math.hypot((x - sx), (y - sy))
                    obj = objects[x, y]
                    if obj.inner_colour in nearest_inner:
                        if dist < nearest_inner[obj.inner_colour]:
                            nearest_inner[obj.inner_colour] = dist
                    else:
                        nearest_inner[obj.inner_colour] = dist
                    if obj.outer_colour in nearest_outer:
                        if dist < nearest_outer[obj.outer_colour]:
                            nearest_outer[obj.outer_colour] = dist
                    else:
                        nearest_outer[obj.outer_colour] = dist

        # Need to ensure that all colours are represented.
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colours*self.grid_size,))
            i = 0
            for c in range(self.n_colours):
                for d in range(1, self.grid_size+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
            assert i == 2*self.n_colours*self.grid_size
            assert (state >= 0).all()
        else:
            # Continuous features.
            state = np.zeros((2*self.n_colours))
            i = 0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state

    def inv_feature_vector(self, i, objects, discrete=True):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """

        sx, sy = self.int_to_point(i)

        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if (x, y) in objects:
                    dist = math.hypot((x - sx), (y - sy))
                    obj = objects[x, y]
                    if obj.inner_colour in nearest_inner:
                        if dist < nearest_inner[obj.inner_colour]:
                            nearest_inner[obj.inner_colour] = dist
                    else:
                        nearest_inner[obj.inner_colour] = dist
                    if obj.outer_colour in nearest_outer:
                        if dist < nearest_outer[obj.outer_colour]:
                            nearest_outer[obj.outer_colour] = dist
                    else:
                        nearest_outer[obj.outer_colour] = dist

        # Need to ensure that all colours are represented.
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:
            state = np.zeros((2*self.n_colours*self.grid_size,))
            i = 0
            for c in range(self.n_colours):
                for d in range(1, self.grid_size+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
            assert i == 2*self.n_colours*self.grid_size
            assert (state >= 0).all()
        else:
            # Continuous features.
            state = np.zeros((2*self.n_colours))
            i = 0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1

        return state

    def inv_feature_matrix(self, objects, discrete=True):
        """
        Get the feature matrix for this objectworld.

        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> NumPy array with shape (n_states, n_states).
        """

        return np.array([self.inv_feature_vector(i, objects, discrete)
                         for i in range(self.n_states)])

    def feature_matrix(self, objects, discrete=True):
        """
        Get the feature matrix for this objectworld.

        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> NumPy array with shape (n_states, n_states).
        """

        return np.array([self.feature_vector(i, objects, discrete)
                         for i in range(self.n_states)])

    def reward(self, state_int):
        """
        Get the reward for a state int.

        state_int: State int.
        -> reward float
        """

        x, y = self.int_to_point(state_int)
        near_c0 = False
        near_c1 = False
        near_c12 = False
        near_c22 = False
        near_c23 = False
        # for (dx, dy) in product(range(-3, 4), range(-3, 4)):
        #     if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
        #         if (abs(dx) + abs(dy) <= 3 and
        #                 (x+dx, y+dy) in self.objects and
        #                 self.objects[x+dx, y+dy].outer_colour == 0):
        #             near_c0 = True
        #         if (abs(dx) + abs(dy) <= 2 and
        #                 (x+dx, y+dy) in self.objects and
        #                 self.objects[x+dx, y+dy].outer_colour == 1):
        #             near_c1 = True

        # if (x, y) in self.objects:
        #     return -1

        for (dx, dy) in product(range(-3, 4), range(-3, 4)):
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                if (abs(dx) + abs(dy) <= 0 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                if (abs(dx) + abs(dy) <= 1 and
                        (dx + x, dy + y) in self.objects and
                        self.objects[dx + x, dy + y].outer_colour == 1):
                    near_c12 = True

        if (x, y) in self.objects and self.objects[x, y].outer_colour == 1:
            near_c1 = True

        if (x, y) in [(3, 5), (3, 6), (3, 7), (4, 5), (4, 6), (4, 7), (5, 5), (5, 6), (5, 7)]:
            near_c22 = True

        if (x, y) in [(6, 4), (6, 5), (6, 6), (6, 7), (6, 8),
        (7, 4), (7, 5), (7, 6), (7, 7), (7, 8)]:
            near_c23 = True

        # if near_c0 and near_c22:
        #     return -0.1
        if near_c23:
            return 0

        if near_c22:
            return 0.5
        
        if near_c1:
            return -1

        if near_c12:
            return -0.3

        if near_c0:
            return -0.7
        
        return 1#0

    def inverse_reward(self, state_int):
        """
        Get the reward for a state int.

        state_int: State int.
        -> reward float
        """

        x, y = self.int_to_point(state_int)

        near_c0 = False
        near_c1 = False
        for (dx, dy) in product(range(-3, 4), range(-3, 4)):
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                if (abs(dx) + abs(dy) <= 3 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
                if (abs(dx) + abs(dy) <= 2 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 1):
                    near_c1 = True

        if near_c0 and near_c1:
            return 1
        if near_c0:
            return 0#-1
        
        return -1#0

    # def inverse_reward(self, state_int):
    #     """
    #     Get the reward for a state int.

    #     state_int: State int.
    #     -> reward float
    #     """

    #     x, y = self.int_to_point(state_int)

    #     near_c0 = False
    #     near_c1 = False
    #     for (dx, dy) in product(range(-3, 4), range(-3, 4)):
    #         if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
    #             if (abs(dx) + abs(dy) <= 3 and
    #                     (x+dx, y+dy) in self.inverted_objects and
    #                     self.inverted_objects[x+dx, y+dy].outer_colour == 0):
    #                 near_c0 = True
    #             if (abs(dx) + abs(dy) <= 2 and
    #                     (x+dx, y+dy) in self.inverted_objects and
    #                     self.inverted_objects[x+dx, y+dy].outer_colour == 1):
    #                 near_c1 = True

    #     if near_c0 and near_c1:
    #         return -1#1
    #     if near_c0:
    #         return 0#-1
    #     return 1

    def generate_trajectories(self, n_trajectories, trajectory_length, policy):
        """
        Generate n_trajectories trajectories with length trajectory_length.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        -> [[(state int, action int, reward float)]]
        """

        return super(Objectworld, self).generate_trajectories(n_trajectories, trajectory_length,
                                             policy,
                                             True)

    def generate_inverse_trajectories(self, n_trajectories, trajectory_length, policy):
        """
        Generate n_trajectories trajectories with length trajectory_length.

        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        -> [[(state int, action int, reward float)]]
        """

        return super(Objectworld, self).generate_inverse_trajectories(n_trajectories, trajectory_length,
                                             policy,
                                             True)

    # def generate_trajectories(self, n_trajectories, trajectory_length, obj, policy,
    #                                 random_start=False):
    #     """
    #     Generate n_trajectories trajectories with length trajectory_length,
    #     following the given policy.

    #     n_trajectories: Number of trajectories. int.
    #     trajectory_length: Length of an episode. int.
    #     policy: Map from state integers to action integers.
    #     random_start: Whether to start randomly (default False). bool.
    #     -> [[(state int, action int, reward float)]]
    #     """
    #     keep = []
    #     trajectories = []
    #     for _ in range(n_trajectories):
    #         if random_start:
    #             sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
    #         else:
    #             sx, sy = 0, 0

    #         trajectory = []
    #         for _ in range(trajectory_length):
    #             if rn.random() < self.wind:
    #                 action = self.actions[rn.randint(0, 4)]
    #             else:
    #                 # Follow the given policy.
    #                 keep.append(policy(self.point_to_int((sx, sy))))
    #                 action = self.actions[policy(self.point_to_int((sx, sy)))]

    #             if (0 <= sx + action[0] < self.grid_size and
    #                     0 <= sy + action[1] < self.grid_size):
    #                 next_sx = sx + action[0]
    #                 next_sy = sy + action[1]
    #             else:
    #                 next_sx = sx
    #                 next_sy = sy

    #             state_int = self.point_to_int((sx, sy))
    #             action_int = self.actions.index(action)
    #             next_state_int = self.point_to_int((next_sx, next_sy))
    #             reward = self.reward(next_state_int, obj)
    #             trajectory.append((state_int, action_int, reward))

    #             sx = next_sx
    #             sy = next_sy

    #         trajectories.append(trajectory)

    #     return np.array(trajectories)

    def optimal_policy(self, state_int):
        raise NotImplementedError(
            "Optimal policy is not implemented for Objectworld.")
    def optimal_policy_deterministic(self, state_int):
        raise NotImplementedError(
            "Optimal policy is not implemented for Objectworld.")
