import numpy as np
import time
import random
import math

np.random.seed(1376)


class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0
        self.h_bs = 1.5
        self.h_ms = 1.5
        self.fc = 2 #GHz
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db


class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25
        self.h_ms = 1.5
        self.Decorrelation_distance = 50
        self.BS_position = [750 / 2, 1299 / 2]  # center of the grids
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - self.BS_position[0])
        d2 = abs(position_A[1] - self.BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(
            math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)


class Vehicle:

    # Vehicle simulator: include all the information for a vehicle
    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []


class Environ:

    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, size_platoon, n_RB,
                 V2I_min, BW, V2V_SIZE, Gap):
        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.vehicles = []

        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.delta_distance = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []
        self.V2V_pathloss = []
        self.V2V_channels_abs = []

        self.V2I_min = V2I_min
        self.sig2_dB = -114
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.gap = Gap
        self.v_length = 0

        self.change_direction_prob = 0.4
        self.n_RB = n_RB
        self.n_Veh = n_veh
        self.size_platoon = size_platoon
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms
        self.bandwidth = BW  # bandwidth per RB, 180,000 MHz
        self.V2V_demand_size = V2V_SIZE  # V2V payload: 4000 Bytes every 100 ms

        self.Interference_all = np.zeros(int(self.n_Veh / self.size_platoon)) + self.sig2

    def add_new_platoon(self, start_position, start_direction, start_velocity, size_platoon):
        for i in range(size_platoon):
            if start_direction == 'u':
                self.vehicles.append(Vehicle([start_position[0], start_position[1] - i * (self.gap + self.v_length)],
                                             start_direction, start_velocity))
            if start_direction == 'd':
                self.vehicles.append(Vehicle([start_position[0], start_position[1] + i * (self.gap + self.v_length)],
                                             start_direction, start_velocity))
            if start_direction == 'l':
                self.vehicles.append(Vehicle([start_position[0] + i*(self.gap + self.v_length), start_position[1]],
                                             start_direction, start_velocity))
            if start_direction == 'r':
                self.vehicles.append(Vehicle([start_position[0] - i*(self.gap + self.v_length), start_position[1]],
                                             start_direction, start_velocity))

    def add_new_platoon_by_number(self, number_vehicle, size_platoon):
        # due to the importance of initial positioning of platoons for RL, we have allocated their positions as follows:
        for i in range(int(number_vehicle / size_platoon)):

            if i == 0:
                ind = 2
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)] # position of platoon leader
                self.add_new_platoon(start_position, 'd', np.random.randint(10, 15), size_platoon)
            elif i == 1:
                ind = 2
                AoI = 100
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]  # position of platoon leader
                self.add_new_platoon(start_position, 'u', np.random.randint(10, 15), size_platoon)
            elif i == 2:
                ind = 2
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'l', np.random.randint(10, 15), size_platoon)
            elif i == 3:
                ind = 2
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'r', np.random.randint(10, 15), size_platoon)
            elif i == 4:
                ind = 4
                start_position = [self.down_lanes[ind], np.random.randint(0, self.height)] # position of platoon leader
                self.add_new_platoon(start_position, 'd', np.random.randint(10, 15), size_platoon)
            elif i == 5:
                ind = 4
                start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]  # position of platoon leader
                self.add_new_platoon(start_position, 'u', np.random.randint(10, 15), size_platoon)
            elif i == 6:
                ind = 4
                start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'l', np.random.randint(10, 15), size_platoon)
            elif i == 7:
                ind = 4
                start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]  # position of platoon leader
                self.add_new_platoon(start_position, 'r', np.random.randint(10, 15), size_platoon)

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.delta_distance = np.asarray([c.velocity * self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each platoon
        # ===============
        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            # ================================================================================================
            if self.vehicles[i].direction == 'u':
                if i % self.size_platoon == 0:
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] <= self.left_lanes[j]) and \
                                ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.vehicles[i].position[0] -
                                                             (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])),
                                                             self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] <= self.right_lanes[j]) and \
                                    ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] +
                                                                 (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])),
                                                                 self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    if change_direction == False:
                        self.vehicles[i].position[1] += delta_distance
                else:
                    follow_index = int(np.floor(i / self.size_platoon))  # vehicle i belongs to which platoon?
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[1] += delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'r':
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
                        else:
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] + \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
            # ================================================================================================
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                if i % self.size_platoon == 0:
                    for j in range(len(self.left_lanes)):
                        if (self.vehicles[i].position[1] >= self.left_lanes[j]) and \
                                ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.vehicles[i].position[0] -
                                                             (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])),
                                                             self.left_lanes[j]]
                                self.vehicles[i].direction = 'l'
                                change_direction = True
                                break
                    if change_direction == False:
                        for j in range(len(self.right_lanes)):
                            if (self.vehicles[i].position[1] >= self.right_lanes[j]) and \
                                    (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.vehicles[i].position[0] +
                                                                 (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])),
                                                                 self.right_lanes[j]]
                                    self.vehicles[i].direction = 'r'
                                    change_direction = True
                                    break
                    if change_direction == False:
                        self.vehicles[i].position[1] -= delta_distance
                else:
                    follow_index = int(np.floor(i / self.size_platoon))  # vehicle i belongs to which platoon?
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[1] -= delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'r':
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
                        else:
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0] + \
                                                           int(i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1]
            # ================================================================================================
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                if i % self.size_platoon == 0:
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] <= self.up_lanes[j]) and \
                                ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] +
                                                             (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] <= self.down_lanes[j]) and \
                                    ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] -
                                                                 (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                    if change_direction == False:
                        self.vehicles[i].position[0] += delta_distance
                else:
                    follow_index = int(np.floor(i / self.size_platoon))
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[0] += delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'u':
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
                        else:
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] + \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
            # ================================================================================================
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                if i % self.size_platoon == 0:
                    for j in range(len(self.up_lanes)):
                        if (self.vehicles[i].position[0] >= self.up_lanes[j]) and \
                                ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                            if (np.random.uniform(0, 1) < self.change_direction_prob):
                                self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] +
                                                             (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'u'
                                break
                    if change_direction == False:
                        for j in range(len(self.down_lanes)):
                            if (self.vehicles[i].position[0] >= self.down_lanes[j]) and \
                                    ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                                if (np.random.uniform(0, 1) < self.change_direction_prob):
                                    self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] -
                                                                 (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                    change_direction = True
                                    self.vehicles[i].direction = 'd'
                                    break
                        if change_direction == False:
                            self.vehicles[i].position[0] -= delta_distance

                else:
                    follow_index = int(np.floor(i / self.size_platoon))
                    if self.vehicles[i].direction == self.vehicles[follow_index * self.size_platoon].direction:
                        self.vehicles[i].position[0] -= delta_distance
                    else:
                        change_direction = True
                        self.vehicles[i].direction = self.vehicles[follow_index * self.size_platoon].direction
                        if self.vehicles[i].direction == 'u':
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] - \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
                        else:
                            self.vehicles[i].position[1] = self.vehicles[follow_index * self.size_platoon].position[1] + \
                                                           (i % self.size_platoon) * (self.gap + self.v_length)
                            self.vehicles[i].position[0] = self.vehicles[follow_index * self.size_platoon].position[0]
            # ================================================================================================
            # if it comes to an exit
            if i % self.size_platoon == 0:
                if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or \
                        (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                    if (self.vehicles[i].direction == 'u'):
                        self.vehicles[i].direction = 'r'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                    else:
                        if (self.vehicles[i].direction == 'd'):
                            self.vehicles[i].direction = 'l'
                            self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                        else:
                            if (self.vehicles[i].direction == 'l'):
                                self.vehicles[i].direction = 'u'
                                self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                            else:
                                if (self.vehicles[i].direction == 'r'):
                                    self.vehicles[i].direction = 'd'
                                    self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1

    def renew_channel(self, number_vehicle, size_platoon):
        """ Renew slow fading channel """

        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))
        self.V2I_pathloss = np.zeros((len(self.vehicles)))

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))
        for i in range(len(self.vehicles)):
            for j in range(i + 1, len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = \
                    self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j],
                                                   self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j, i] = self.V2V_pathloss[i][j] = \
                    self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position)

        self.V2V_channels_abs = self.V2V_pathloss + self.V2V_Shadowing

        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)

        self.V2I_channels_abs = self.V2I_pathloss + self.V2I_Shadowing

    def renew_channels_fastfading(self):

        """ Renew fast fading channel """
        V2V_channels_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_channels_with_fastfading = V2V_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2V_channels_with_fastfading.shape)) / math.sqrt(2))

        V2I_channels_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_channels_with_fastfading = V2I_channels_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_channels_with_fastfading.shape) +
                   1j * np.random.normal(0, 1, V2I_channels_with_fastfading.shape)) / math.sqrt(2))

    def Revenue_function(self, quantity, threshold):
        # G function definition in the paper
        revenue = 0
        if quantity >= threshold:
            revenue = 1
        else:
            revenue = 0
        return revenue

    def Compute_Performance_Reward_Train(self, platoons_actions):

        sub_selection = platoons_actions[:, 0].astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)            # channel_selection_part
        platoon_decision = platoons_actions[:, 1].astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)         # platoon selection Intra/Inter platoon communication
        power_selection = platoons_actions[:, 2].reshape(int(self.n_Veh / self.size_platoon), 1)                        # platoon selection Intra/Inter platoon communication
        # ------------ Compute Interference --------------------
        self.platoon_V2I_Interference = np.zeros(int(self.n_Veh / self.size_platoon))  # V2I interferences
        self.platoon_V2I_Signal = np.zeros(int(self.n_Veh / self.size_platoon))  # V2I signals
        self.platoon_V2V_Interference = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon-1])  # V2V interferences
        self.platoon_V2V_Signal = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon-1])  # V2V signals

        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                for k in range(len(indexes)):
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 0: # platoon j has selected the inter-platoon communication
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #     continue
                        self.platoon_V2I_Interference[indexes[j, 0]] += \
                            10 ** ((power_selection[indexes[k, 0], 0] - self.V2I_channels_with_fastfading[indexes[k, 0]*self.size_platoon, i] +
                                                       self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 1: # platoon j has selected the intra-platoon communication
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #     continue
                        for l in range(self.size_platoon-1):
                            self.platoon_V2V_Interference[indexes[j, 0], l] += \
                                10 ** ((power_selection[indexes[k, 0], 0] - self.V2V_channels_with_fastfading[indexes[k, 0]*self.size_platoon, indexes[j, 0]*self.size_platoon + (l + 1), i] +
                                        2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # computing the platoons inter/intra-platoon signals
        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                if platoon_decision[indexes[j, 0], 0] == 0:
                    self.platoon_V2I_Signal[indexes[j, 0]] = 10 ** ((power_selection[indexes[j, 0], 0] - self.V2I_channels_with_fastfading[indexes[j, 0]*self.size_platoon, i] +
                                                       self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                elif platoon_decision[indexes[j, 0], 0] == 1:
                    for l in range(self.size_platoon - 1):
                        self.platoon_V2V_Signal[indexes[j, 0], l] += 10 ** ((power_selection[indexes[j, 0], 0] - self.V2V_channels_with_fastfading[indexes[j, 0] * self.size_platoon, indexes[j, 0] * self.size_platoon + (l + 1), i] +
                                    2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        V2I_Rate = np.log2(1 + np.divide(self.platoon_V2I_Signal, (self.platoon_V2I_Interference + self.sig2)))
        V2V_Rate = np.log2(1 + np.divide(self.platoon_V2V_Signal, (self.platoon_V2V_Interference + self.sig2)))

        self.interplatoon_rate = V2I_Rate * self.time_fast * self.bandwidth
        self.intraplatoon_rate = (V2V_Rate * self.time_fast * self.bandwidth).min(axis=1)

        platoons_AoI = self.Age_of_Information(self.interplatoon_rate)
        # self.V2I_demand -= self.interplatoon_rate
        self.V2V_demand -= self.intraplatoon_rate
        # self.V2I_demand[self.V2I_demand < 0] = 0
        self.V2V_demand[self.V2V_demand <= 0] = 0

        self.individual_time_limit -= self.time_fast
        self.active_links[
        np.multiply(self.active_links, self.V2V_demand <= 0)] = 0  # transmission finished, turned to "inactive"
        reward_elements = self.intraplatoon_rate / 10000
        # reward_elements = np.zeros(int(self.n_Veh / self.size_platoon))
        reward_elements[self.V2V_demand <= 0] = 1

        return platoons_AoI, self.interplatoon_rate, self.intraplatoon_rate, self.V2V_demand, reward_elements

    def Age_of_Information(self, V2I_rate):
        # computing the platoons age of information

        for i in range(int(self.n_Veh / self.size_platoon)):
            if V2I_rate[i] >= self.V2I_min:
                self.AoI[i] = 1
            else:
                self.AoI[i] += 1
                if self.AoI[i] >= (self.time_slow / self.time_fast):
                    self.AoI[i] = (self.time_slow / self.time_fast)
        return self.AoI

    def act_for_training(self, actions):

        per_user_reward = np.zeros(int(self.n_Veh / self.size_platoon))
        action_temp = actions.copy()
        platoon_AoI, C_rate, V_rate, Demand, elements = self.Compute_Performance_Reward_Train(action_temp)
        V2V_success = 1 - np.sum(self.active_links) / (int(self.n_Veh / self.size_platoon))  # V2V success rates

        for i in range(int(self.n_Veh / self.size_platoon)):
            
            per_user_reward[i] = (-4.95)*(Demand[i]/self.V2V_demand_size) - \
                                 platoon_AoI[i]/20 + (0.05)*self.Revenue_function(C_rate[i], self.V2I_min) - \
                                 0.5 * math.log(action_temp[i, 2], 5)
        global_reward = np.mean(per_user_reward)
        return per_user_reward, global_reward, platoon_AoI, C_rate, V_rate, Demand, V2V_success

    def act_for_testing(self, actions):

        action_temp = actions.copy()
        platoon_AoI, C_rate, V_rate, Demand, elements = self.Compute_Performance_Reward_Train(action_temp)
        V2V_success = 1 - np.sum(self.active_links) / (int(self.n_Veh / self.size_platoon))  # V2V success rates

        return platoon_AoI, C_rate, V_rate, Demand, elements, V2V_success

    def Compute_Interference(self, platoons_actions):

        sub_selection = platoons_actions[:, 0].copy().astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)
        platoon_decision = platoons_actions[:, 1].copy().astype('int').reshape(int(self.n_Veh / self.size_platoon), 1)
        power_selection = platoons_actions[:, 2].copy().reshape(int(self.n_Veh / self.size_platoon), 1)
        # ------------ Compute Interference --------------------
        V2I_Interference_state = np.zeros(int(self.n_Veh / self.size_platoon)) + self.sig2
        V2V_Interference_state = np.zeros([int(self.n_Veh / self.size_platoon), self.size_platoon - 1]) + self.sig2

        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                for k in range(len(indexes)):
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 0:
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #     continue
                        V2I_Interference_state[indexes[j, 0]] += \
                            10 ** ((power_selection[indexes[k, 0], 0] - self.V2I_channels_with_fastfading[
                                indexes[k, 0] * self.size_platoon, i] +
                                    self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
                    if indexes[j, 0] != indexes[k, 0] and platoon_decision[indexes[j, 0], 0] == 1:
                        # if not self.active_links[indexes[k, 0]] and platoon_decision[indexes[k, 0], 0] == 1:
                        #     continue
                        for l in range(self.size_platoon - 1):
                            V2V_Interference_state[indexes[j, 0], l] += \
                                10 ** ((power_selection[indexes[k, 0], 0] - self.V2V_channels_with_fastfading[
                                    indexes[k, 0] * self.size_platoon, indexes[j, 0] * self.size_platoon + (l + 1), i] +
                                        2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        self.V2I_Interference_all = 10 * np.log10(V2I_Interference_state)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference_state)
        for i in range(int(self.n_Veh / self.size_platoon)):
            if platoon_decision[i, 0] == 0:
                self.Interference_all[i] = self.V2I_Interference_all[i]
            else:
                self.Interference_all[i] = np.max(self.V2V_Interference_all[i, :])
    def new_random_game(self, n_Veh=0):

        # make a new game
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_platoon_by_number(int(self.n_Veh), self.size_platoon)
        self.renew_channel(int(self.n_Veh), self.size_platoon)
        self.renew_channels_fastfading()

        self.V2V_demand = self.V2V_demand_size * np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16)
        self.individual_time_limit = self.time_slow * np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16)
        self.active_links = np.ones((int(self.n_Veh / self.size_platoon)), dtype='bool')
        self.AoI = np.ones(int(self.n_Veh / self.size_platoon), dtype=np.float16)*100
