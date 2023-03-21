import gym
from gym import spaces
from gym.spaces import Dict, Tuple, Discrete, Box
import numpy as np
import random
import threading as th
import time
from enemy_detection_client import SCREEN_WIDTH, SCREEN_HEIGHT
from enemy_detection_client import EnemyDetectorClient
from GameInterface.game_interface_client import GameClient

from gsi_client import client
from awpy.data import NAV_CSV
# importing input library
from pynput import mouse, keyboard
from pynput.mouse import Button
from pynput.keyboard import Key

TRAINING = True

TIME_STEPS = 400


class CSGO_Env_Utils:

    def location_domain(max_x, min_x, max_y, min_y, max_z, min_z):
        return Box(
            low=np.array([min_x, min_y, min_z]),
            high=np.array([max_x, max_y, max_z]),
            dtype=np.int32)
        # int to round off
        # return Tuple((
        #     spaces.Discrete(max_x - min_x + 1, start =min_x), #x
        #     spaces.Discrete(max_y - min_y + 1, start =min_y), #y
        #     spaces.Discrete(max_z - min_z + 1, start =min_z), #z
        # ))

    def forward():
        # each forward has domain [-1:1]
        return Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

    def action_space_domain():
        return Tuple([
            Discrete(2),  # 0 for ignore, 1 for no movement key
            # shift pressed? #shift pressed == walking, else running
            Discrete(2),
            # ctrl pressed? #crouching basically, ctrl pressed == crouching, else standing
            Discrete(2),
            # space pressed? #jumping basically, space pressed == jumping, else standing
            Discrete(2),
            Discrete(2),  # fire? #fire == 1, else 0 #left mouse click
            # # 0 for no button pressed, 1 for 'w', 2 for 'a', 3 for 's', 4 for 'd',
            # Discrete(5),
            #  0 for 'w', 1 for 'a', 2 for 's', 3 for 'd', <Binary Numbers>
            Discrete(2), 
            Discrete(2), 
            Discrete(2), #1 means move mouse cursor to right by 5, 0 means no movement
            Discrete(2), #1 means move mouse cursor to left by 5 0 means no movement
        ])

    def observation_space_domain(max_x, min_x, max_y, min_y, max_z, min_z, SCREEN_HEIGHT, SCREEN_WIDTH):
        return Dict({
            'obs_type': spaces.Discrete(2),  # 0 for partial, 1 for  complete
            'enemy': Dict({
                'position': Dict({
                    # data['areaId'] map id, 10 char for buffer
                    'areaId': spaces.Text(10),
                    'location': CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
                    'forward': CSGO_Env_Utils.forward(),
                    'time_seen': spaces.Discrete(TIME_STEPS),
                }),
                'enemy_screen_location':  Box(low=np.array([0, 0]), high=np.array([SCREEN_HEIGHT, SCREEN_WIDTH]), dtype=np.int32),
                'health': spaces.Discrete(100),
            }),
            'agent': Dict({
                'position': Dict({
                    # data['areaId'] map id, 10 char for buffer
                    'areaId': spaces.Text(10),
                    'location': CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
                    'forward': CSGO_Env_Utils.forward(),
                }),
                'agent_gun': spaces.Discrete(1),  # fixed
                'agent_bullets': spaces.Discrete(30),  # fixed
                'health':  spaces.Discrete(100),
            }),
            'bomb_location': Dict({
                # data['areaId'] map id, 10 char for buffer
                'areaId': spaces.Text(10),
                'location': CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
            }),
            # fixed, 0 for not defusing, 1 for defusing
            'bomb_defusing': Tuple([spaces.Discrete(2), spaces.Discrete(TIME_STEPS)]),
            'current_time': spaces.Discrete(TIME_STEPS),
            # 0 for ongoing, 1 for agent win, 2 for enemy win
            'winner': spaces.Discrete(3),
        })

    # function from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    def generate_set_of_goals(site):
        #hand picked goals sadly
        B = [
            # Tuple([-2157.99, 1814.03, 68.03]),
            # Tuple([-1639.76, 1620.03, 66.41]),
            # Tuple([-1394.03, 1978.68, 70.08]),
            # Tuple([-1819.93, 2477.03, 94.81]),
            # Tuple([-2084.75, 3117.96, 99.53]),
            # Tuple([-1362.03, 2755.43, 82.11]),
            # Tuple([-1271,41, 2481.42, 108.06]), 
            (-2157.99, 1814.03, 68.03),
            (-1639.76, 1620.03, 66.41),
            (-1394.03, 1978.68, 70.08),
            (-1819.93, 2477.03, 94.81),
            (-2084.75, 3117.96, 99.53),
            (-1362.03, 2755.43, 82.11),
            (-1271,41, 2481.42, 108.06), 
        ]
        A = [
            (384.29,1935.1,160.07),
            (543.96, 2763.54, 161.43),
            (1235.10,2460.96,161.89),
            (1051.03, 3059.96, 195.22),
            (1329.03, 2407.05, 102.65),
            (1763.37, 1999.96, 65.14),#a bombsite
        ]
        if site == 'BombsiteA':
            return A
        return B
        
    def get_enemy_spawn_points(self, bombsite_choice):
        pass
        
    # def _generate_complete_goal_state(self, goal, obs):
        # 2 kind of goal nature: Aggressive or Passive

        # Aggressive Goal States are one where puts emphasis in closing the distance
        # between player and agent
        # if goal == 0:  # 0 for aggressive, None for Passive
        #     # we form a circle around the enemy, (need not check for enemy location bc complete)
        #     enemy_loc = obs['enemy']['position']['location']
        #     agent_loc = obs['agent']['position']['location']

        # else:  # Passive

            # Passive Goal States are one where puts emphasis in keeping the ratio distances
            # of d(player, bomb) and d(agent,enemy) at 1
            # if enemy location not known, then we want d(player,bomb) < 500

            # generate all possible goal states
            # goal state is a tuple of (x, y, z, enemy_x, enemy_y, enemy_z)
            # goal state is the state that the a


class CSGO_Env(gym.Env):
    MAP_NAME = 'de_dust2'
    MAP_DATA = NAV_CSV[NAV_CSV["mapName"] == MAP_NAME]
    # SCREEN_HEIGHT = ENEMY_SCREEN_DETECTOR.re_x
    # SCREEN_WIDTH = ENEMY_SCREEN_DETECTOR.re_y
    OBSERVING_TIME = 0.1
    ACTION_TIME = 0.1
    # Env is made up of segmented areas of map. Each area is represented by a super node
    # do we include screen location of enemy in observation space?

    def __init__(self):
        self._init_para()
        # observation space can be abstractly thought of as a set of nodes
        # each node encapsulates the following
        self.lock = th.RLock()
        self._obs = None
        self._part_obs = None
        self._reward = 0
        self._set_of_goals = None
        self._goal_state = None
        self._partial_goal_state = None
        self._time_of_goal_state = None
        # self._prev_action = None
        self.observation_space = CSGO_Env_Utils.observation_space_domain(
            self.max_x, self.min_x, self.max_y, self.min_y, self.max_z, self.min_z, SCREEN_HEIGHT, SCREEN_WIDTH)
        self.action_space = CSGO_Env_Utils.action_space_domain()
        self.goal_space = CSGO_Env_Utils.location_domain(self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z)


        # CSGO_Env_Utils.start_game(
        #     self.MAP_NAME, self.MAP_DATA, self.keyboard_controller, self.mouse_controller)

    def _init_para(self):
        bombsite_choice = random.choice(['BombsiteA', 'BombsiteB'])
        self.min_x, self.max_x = None, None
        self.min_y, self.max_y = None, None
        self.min_z, self.max_z = None, None
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        self._set_of_goals = CSGO_Env_Utils.generate_set_of_goals(bombsite_choice)
        self._goal_state = random.choice(self._set_of_goals)
        self._partial_goal_state = self._make_partial_goal(self._goal_state)

        for i in range(len(CSGO_Env.MAP_DATA)):
            data = CSGO_Env.MAP_DATA.iloc[i]
            if self.min_x is None or self.min_x > float(data["northWestX"]):
                self.min_x = float(data["northWestX"])

            if self.max_x is None or self.max_x < float(data["southEastX"]):
                self.max_x = float(data["southEastX"])

            if self.min_y is None or self.min_y > float(data["northWestY"]):
                self.min_y = float(data["northWestY"])

            if self.max_y is None or self.max_y < float(data["southEastY"]):
                self.max_y = float(data["southEastY"])

            if self.min_z is None or self.min_z > float(data["northWestZ"]):
                self.min_z = float(data["northWestZ"])

            if self.max_z is None or self.max_z < float(data["southEastZ"]):
                self.max_z = float(data["southEastZ"])
        # then init game
        GameClient.send_action("configure")
        GameClient.send_action(f"start {bombsite_choice}")
       

    #(observation, reward, done, info)
    # each step corresponds to 0.1 seconds (OBSERVING_TIME or ACTION_TIME)
    def step(self, action):
        

        # get information and img
        information = {}
        information['player'] = client.get_info("player")
        information['phase_countdowns'] = client.get_info("phase_countdowns")
        information['round'] = client.get_info("round")
        information['allplayers'] = client.get_info("allplayers")
        information['bomb'] = client.get_info("bomb")
        
        round_info = information["round"]
        match_result = round_info['win_team']
        if match_result == 'T':
            match_result = 1
        elif match_result == 'CT':
            match_result = 2
        else:
            match_result = 0
            
        # create threads
        observing_thread = th.Timer(
            CSGO_Env.OBSERVING_TIME, self._get_state, args=(information))
        # observing_thread_2 = th.Timer(CSGO_Env.OBSERVING_TIME, self._get_full_state, args = (lock,))
        action_thread = th.Timer(CSGO_Env.ACTION_TIME,
                                 self._apply_action, args=(action, (match_result>0)))

        # get current_state
        prev_observation = self._obs
        prev_part_observation = self._part_obs

        # start observing
        observing_thread.start()

        # apply action at the same time
        action_thread.start()

        # create reward thread
        # this is so to ensure that we calculate reward only after we have the next observation
        reward_thread = th.Thread(target=self._get_reward, args=(
            prev_observation, prev_part_observation, action))

        # #get next observation
        # observing_thread_2.start()

        # reward thread has to wait for observing thread, so we ensure all threads are properly ran through in the function
        # self._obs records the latest _obs
        reward_thread.start()
        reward_thread.join()

        return self._obs, self._part_obs, self._reward, self._is_done(), self._goal_state, self._partial_goal_state

    def get_current_observation(self):
        return self._obs
    
    def get_current_partial_observation(self):
        return self._part_obs

    # TODO: Fill in the blank <models> after finishing implementing them
    def _get_state(self, information):
        with self.lock:

            # process img
            # img = information['img']
            enemy_information = EnemyDetectorClient.get_enemy_info()

            # enemy_on_radar = ENEMY_RADAR_DETECTOR.scan_for_enemy(img)
            enemy_on_radar = enemy_information['enemy_on_screen'] 
            enemy_screen_coords = enemy_information['enemy_screen_coords']
            # #now check if see enemy on screen
            # if enemy_on_radar:
                #now check if see enemy on screen
                
                #now we tackle this
                # enemy_screen_coords = ENEMY_SCREEN_DETECTOR.scan_for_enemy(img)
            # enemy_screen_coords = ENEMY_SCREEN_DETECTOR.scan_for_enemy(img)

            # information['enemy_coords_on_screen'] = enemy_screen_coords
            partial_information = information.copy()

            # img no longer needed
            # information.pop('img')
            self._obs = self._get_full_state(information)

            # Making partial state
            # blanks for partial state as they have no access
            # unless certain conditions are met
            # this is to make the agent's detection system
            # more sophisticated, eg being able to identify
            # location of an object (in exact coordinates),
            # by 'seeing' it on the screen <Computer Vision>

            # direct access to player information and time only
            # if see enemy on radar, agent will know enemy location
            # if see enemy on radar, we start to see if we can see enemy on screen
            # bomb location is always known by agent
            # if enemy tries to defuse bomb, agent will know
            # if not enemy_on_radar or not (information['bomb']['state'] == 'defusing' and self._obs['bomb defusing'][0] == 0):
            if enemy_screen_coords is not None or not (information['bomb']['state'] == 'defusing' and self._obs['bomb_defusing'][0] == 0):
                self._obs['enemy_coords_on_screen'] = enemy_screen_coords
                partial_information['allplayers'] = None

            self._part_obs = self._get_partial_state( partial_information)
            self._generate_goal()

    def _get_full_state(self, information):
        # here we assume TRAINING is true
        with self.lock:
            self._obs = self._make_complete_observation(information)

    def _get_partial_state(self, information):
        # here we assume TRAINING is true
        with self.lock:
            self._part_obs = self._make_partial_observation(information)

# Reward function
    # TODO: adapt reward for partial state
    def _get_reward(self, prev_obs, prev_part_obs, action):
        with self.lock:
            if prev_obs is None:
                self._reward = 0
            else:
                self._reward = self._calculate_reward(
                    prev_obs, prev_part_obs, action)

    # Design this please brother you my god fr
    def _calculate_reward(self, prev_obs, prev_part_obs, action):
        # divide all by 10 since now step is in 0.1 seconds <if reward is time based>
        # +0.01 for every survived second
        # -0.25 if bomb is
        # +0.01 for every survived iteration
        # -0.25 if bomb is heard/seen to be defused, reset to +0.01 by seeing (with 90% certainty) bomb is not defused.
        # +1 for win. -1 for lose
        # -0.001 for making noise<Running, Firing gun, Reloading> if opponent not seen since 5 sec
        # +0.1 for spotting enemy
        # -0.2 for getting hit by enemy
        # +0.2 for hitting enemy

        # first check if the game is lost or won
        if self._obs['winner'] != 0:
            return 1 if self._obs['winner'] == 1 else -1

        # if game ongoing
        else:
            cost = 0
            reward = 0
            # check if bomb is being defused
            prev_bomb_defusing = prev_obs['bomb_defusing'][0]
            prev_info_timestamp = prev_obs['bomb_defusing'][1]
            bomb_defusing = self._obs['bomb_defusing'][0]
            info_timestamp = self._obs['bomb_defusing'][1]
            prev_enemy_health = prev_obs['enemy']['health']
            cur_enemy_health = self._obs['enemy']['health']

            # reward +0.5 if bomb is prevented from defusing
            if prev_bomb_defusing == 1 and bomb_defusing == 0 and info_timestamp > prev_info_timestamp:
                reward += 0.25
            else:
                pass

            # if bomb defusing, we penalize per timestep, unless enemy is being hit
            if bomb_defusing == 1:
                if prev_enemy_health > cur_enemy_health:
                    reward += 0.1725
                else:
                    reward -= 0.0125

            # if bomb not defusing we take note focus on finding the enemy and hiding information
            else:
                if prev_enemy_health > cur_enemy_health:
                    reward += 0.125
                else:
                    if action[3] == 6:
                        cost += 0.0025  # cost for making noise and wasting gun ammo

                # now account for reward from partial state, specifically
                # gaining new information about enemy location
                # with regards to information, we specifically look at the enemy location
                prev_enemy_position = prev_part_obs['enemy']['position']
                prev_enemy_location = prev_enemy_position['location']
                prev_enemy_timestamp = prev_enemy_position['time_seen']

                curr_enemy_position = self._part_obs['enemy']['position']
                curr_enemy_location = curr_enemy_position['location']
                curr_enemy_timestamp = curr_enemy_position['time_seen']

                # enemy recently seen, reward += 0.1
                if curr_enemy_location is not None and \
                        prev_enemy_location is None and \
                        curr_enemy_timestamp > prev_enemy_timestamp:  # need to replace here to accommodate for the fact that full state should not receive this reward
                    reward += 0.05
                # enemy kept track of, reward += 0.001
                elif curr_enemy_location is None and \
                        prev_enemy_location is not None:
                    reward += 0.001

                # if shift/ctrl not press when moving , penalize for making sound
                # action[2] corr to ctrl, action[1] corr to shift
                # action[0] corr to movement key, action[2] corr to jump key
                if (action[2] == 0 or action[1] == 0) and \
                        (action[0] == 0 or action[3] != 0):
                    cost += 0.0005

                # +0.02 if near goal state and not defusing bomb
                if self._near_goal_state() and bomb_defusing == 0:
                    reward += 0.001

            agent_health = self._obs['agent']['health']
            prev_agent_health = prev_obs['agent']['health']
            if prev_agent_health > agent_health:
                if bomb_defusing == 1:
                    pass
                else:
                    if agent_health <= 50:
                        reward -= 0.2
                    else:
                        reward -= 0.1
        return reward - cost

# Action
    # way we apply action might result very straight forward
    # if action dont explicitly state to press a key, we release it
    def _apply_action(self, action, done):
        # sleep to run through the timed thread
        GameClient.send_action(action, done)
        time.sleep(self.ACTION_TIME)
    # TODO: Change datatype
    # DONE, TODO: Check

# State implementation
    def _make_complete_observation(self, information):
        agent = information["player"]
        phase_cd = information["phase_countdowns"]
        round_info = information["round"]
        match_result = round_info['win_team']
        if match_result == 'T':
            match_result = 1
        elif match_result == 'CT':
            match_result = 2
        else:
            match_result = 0
        enemy = {}
        # players = GSI_SERVER_TRAINING.get_info("allplayers")#returns dictionary of bombstate
        players = information["allplayers"]  # returns dictionary of bombstate
        for player in players:
            if player['name'] != agent['name']:
                enemy = player
                break
        # bomb = GSI_SERVER_TRAINING.get_info("bomb")
        bomb = information["bomb"]
        agent_weapon = [weapon for weapon in agent['weapons']
                        if weapon['state'] == 'active'][0]
        enemy_screen_coord = information['enemy_screen_coords'].get(
            'head', None)
        if enemy_screen_coord is None:
            enemy_screen_coord = information['enemy_screen_coords'].get(
                'body', None)
        return{
            'obs_type': 1,
            'enemy': {
                'position': {
                    'areaId': None,
                    # 'location' : np.fromstring(enemy['position']),
                    'location': np.array(enemy['position'].split(','), dtype=np.float32),
                    'forward': np.array(enemy['forward'].split(','), dtype=np.float32),
                    'time_seen': float(phase_cd['phase_ends_in']),
                },
                'enemy_coord_on_screen': enemy_screen_coord if enemy_screen_coord != (None, None) else (None, None),
                'health': int(enemy['state']['health']),
            },
            'agent': {
                'position': {
                    'areaId': None,
                    'location': np.array(agent['position'].split(','), dtype=np.float32),
                    'forward': np.array(agent['forward'].split(','), dtype=np.float32),
                },
                'agent_gun': agent_weapon['name'],
                'agent_bullets': int(agent_weapon['ammo_clip']),
                'health': int(agent['state']['health']),
            },
            # 'bomb location' : np.fromstring(bomb['position']),
            'bomb_location': {
                'areaId': None,
                'location': np.array(bomb['position'].split(','), dtype=np.float32),
            },
            'bomb_defusing': (1 if bomb['state'] == 'defusing' else 0,  int(phase_cd['phase_ends_in'])),
            'current_time': int(phase_cd['phase_ends_in']),
            'winner': match_result

        }

    # TODO: after finishing partial state, sync the making of complete and partial so that they receive the same information [DONE]
    # TODO: Fill in the blank <models> after finishing implementing them
    def _make_partial_state(self, information):
        # if enemy not on bomb, we know bomb not defusing
        # if see bomb on screen, agent will know whether bomb is defused

        # Get player information first
        agent = information["player"]
        agent_weapon = [weapon for weapon in agent['weapons']
                        if weapon['state'] == 'active'][0]

        # get time
        phase_cd = information["phase_countdowns"]

        # get round information
        round_info = information["round"]
        match_result = round_info['win_team']
        if match_result == 'T':
            match_result = 1
        elif match_result == 'CT':
            match_result = 2
        else:
            match_result = 0

        # get prev obv of enemy
        enemy = self._part_obs['enemy']
        enemy_pos = enemy['position'] if enemy is not None else None
        enemy_loc = enemy_pos['location'] if enemy_position is not None else None
        enemy_forw = enemy_pos['forward'] if enemy_position is not None else None
        time_of_info_enemy = enemy_pos['time_seen'] if enemy is not None else None

        # get prev obv of bomb
        bomb_state, time_of_info_bomb = self._part_obs['bomb_defusing']

        # check if we have seen the bomb
        # here we area also 'simulating' the ability of the agent to be able to
        # distinguish whether the bomb is defusing or not with just the image of the bomb
        # This is not an easy task since the vision cues for defusing is not very clear
        # To simulate this ability, once we have seen the bomb on the screen, we will
        # stream the state of the bomb from the game server
        # img = information['img']
        # bomb_seen_on_screen = BOMB_SCREEN_DETECTOR.scan_for_bomb(img)
        curr_bomb_state = information['bomb']['state']

        # now if our time_of_info_bomb is way too old <5s, we delete the information as irrelevant
        if time_of_info_bomb is None or int(phase_cd['phase_ends_in']) - time_of_info_bomb > 5:
            bomb_state = None
            time_of_info_bomb = int(phase_cd['phase_ends_in'])

        # same goes for time_of_info_enemy
        if time_of_info_enemy is None or int(phase_cd['phase_ends_in']) - time_of_info_enemy > 5:
            enemy = None
            enemy_pos = None
            enemy_loc = None
            enemy_forw = None
            time_of_info_enemy = int(phase_cd['phase_ends_in'])

        # now we see if we have 'seen' enemy
        if information['allplayers'] is not None:
            # returns dictionary of bombstate
            players = information["allplayers"]

            for player in players:
                if player['name'] != agent['name']:
                    enemy = player
                    break

            enemy_loc = np.array(
                enemy['position'].split(','), dtype=np.float32)
            enemy_forw = np.array(
                enemy['forward'].split(','), dtype=np.float32)
            time_of_info_enemy = int(phase_cd['phase_ends_in'])
            bomb_loc = np.array(
                information["bomb"]['position'].split(','), dtype=np.float32)

           # check if enemy location is close to bomb location
            if(sum((enemy_loc - bomb_loc)**2) < 30):
                # so if enemy is close to the bomb, there is a chance that the enemy
                # is defusing, so we have to see if we have seen the bomb on the screen
                # to confirm its status
                # if yes, we know bomb true state
                if bomb_seen_on_screen:
                    bomb_state_info = information['bomb']['state']
                    time_of_info_bomb = int(phase_cd['phase_ends_in'])
                    if bomb_state_info == 'defusing':
                        bomb_state = 1
                    else:
                        bomb_state = 0
                # if no, we assume the bomb to be defusing
                else:
                    time_of_info_bomb = int(phase_cd['phase_ends_in'])
                    bomb_state = 1
            # if enemy is far from bomb, we know bomb is not defusing
            else:
                bomb_state = 0
                time_of_info_bomb = int(phase_cd['phase_ends_in'])
                # bomb_state_info = information['bomb']['state']
                # time_of_info_bomb = int(phase_cd['phase_ends_in'])
                # if bomb_state_info == 'defusing':
                #     bomb_state = 1
                # else:
                #     bomb_state = 0

        # if we have not seen player, but have seen bomb, we know its state
        if bomb_seen_on_screen:
            bomb_state = information['bomb']['state']
            time_of_info_bomb = int(phase_cd['phase_ends_in'])
            if bomb_state == 'defusing':
                bomb_state = 1
            else:
                bomb_state = 0

        # now check if enemy tried to defuse the bomb
        # done by comparing prev state of bomb and current state of bomb
        if bomb_state != 'defusing' and curr_bomb_state == 'defusing':
            bomb_state = 1
            time_of_info_bomb = int(phase_cd['phase_ends_in'])

        # update time of no information about bomb and enemy, if
        # there is no information about bomb and enemy
        if bomb_state is None:
            time_of_info_bomb = int(phase_cd['phase_ends_in'])

        if enemy is None:
            time_of_info_enemy = int(phase_cd['phase_ends_in'])

        bomb = information["bomb"]
        enemy_screen_coord = information['enemy_screen_coords'].get(
            'head', None)
        if enemy_screen_coord is None:
            enemy_screen_coord = information['enemy_screen_coords'].get(
                'body', None)
        return{
            'obs_type': 0,
            'enemy': {
                'position': {
                    'areaId': None,
                    # 'location' : np.fromstring(enemy['position']),
                    'location': enemy_loc,
                    'forward': enemy_pos,
                    'time_seen': time_of_info_enemy,
                },
                'enemy_coord_on_screen': enemy_screen_coord if enemy_screen_coord is not None else None,
                'health': int(enemy['state']['health']),
            },
            'agent': {
                'position': {
                    'areaId': None,
                    'location': np.array(agent['position'].split(','), dtype=np.float32),
                    'forward': np.array(agent['forward'].split(','), dtype=np.float32),
                },
                'agent_gun': agent_weapon['name'],
                'agent_bullets': int(agent_weapon['ammo_clip']),
                'health': int(agent['state']['health']),
            },
            # 'bomb location' : np.fromstring(bomb['position']),
            'bomb_location': {
                'areaId': None,
                'location': np.array(bomb['position'].split(','), dtype=np.float32),
            },
            'bomb_defusing': (bomb_state,  time_of_info_bomb),
            'current_time': int(phase_cd['phase_ends_in']),
            'winner': match_result

        }


    def reset(self):
        bombsite_choice = random.choice(['BombsiteA', 'BombsiteB'])
        GameClient.send_action(f"restart {bombsite_choice}")
        # get information and img
        information = {}
        information['player'] = client.get_info("player")
        information['phase_countdowns'] = client.get_info("phase_countdowns")
        information['round'] = client.get_info("round")
        information['allplayers'] = client.get_info("allplayers")
        information['bomb'] = client.get_info("bomb")
        self._get_state(information)
        return self._obs, self._part_obs, self._goal_state, self._partial_goal_state

# Goals implementation
    # Goals are basically a strategic location in the map
    # Extending on the idea of a universal policy
    # want to see if universal dynamic policy can be achieved with dynamic goals
    # so goal are safe spot in the map,
    # We give the agent a certain time to reach the goal state
    # if it does not reach the goal state, we dont really care, but if it does, we give it a reward
    # at goal state, we start rolling for another goal state, and the probability is varied wrt to time.
    # we let the environment take care of this, meaning goal generation will be running off sync with the environment steps,
    # tho it is still very much dependent on the environment observation
    # rolling probability can be done with k++ mean clustering
    # we care more about positioning and bomb state.
    # since bomb state is a binary, we use it to add bonus when its not defusing, and deduct bonus when its defusing and
    # agent is chasing sub goal
    # other than that, goal inherits other attributes of the environment
    # we can even try to reduce the dimension of goals!
    # generate goal state wrt to a probability distribution that is dependent of
    # time_of_goal_state and distance_to_goal_state
    # we prefer 'closer' goal states, and we want probability of
    # choosing current goal state to degrade over time
    # so we use time as a exponential decay function to the probability of choosing the current goal state
    # then we generate a different prob distribution for the rest of the goal states
    # by weighting the probability of choosing a given goal state by the distance to the goal state from the player
    # we also want to explore the map, so there is a 0.2 chance that we will generate a random goal state
    def _generate_goal(self):
        # roll to see if we stay
        probability_of_staying = np.exp(-self._time_of_goal_state**2/10000)
        roll = np.random.rand()
        if roll > probability_of_staying:
            roll = np.random.rand()
            # no stay
            # generate other goals
            other_goals = self._set_of_goals.copy()
            other_goals.remove(self._curr_goal_state)
            other_goals = np.array(list(other_goals))
            self._time_of_goal_state = 0
            # random goal or no
            if roll > 0.8:
                # heuristic goal

                # generate complete goal_states, and with that we generate partial goal state
                curr_loc = self._obs['agent']['position']['location']
                enemy_loc = self._obs['enemy']['position']['location']
                # want close to agent
                w1 = [np.abs(np.linalg.norm(curr_loc - goal))
                      for goal in other_goals]

                # want far from enemy
                w2 = [np.abs(np.linalg.norm(enemy_loc - goal))
                      for goal in other_goals]

                np.linalg.norm(arr)
                w = w1/w2
                w = w / np.linalg.norm(w)

                # now we roll again
                index = np.random.choice(len(other_goals), p=w)
                self._goal_state = other_goals[index]
                self._partial_goal_state = self._make_partial_goal(
                    self._goal_state)
            else:
                # random goal
                self._goal_state = other_goals[np.random.randint(
                    len(other_goals))]
                self._partial_goal_state = self._make_partial_goal(
                    self._goal_state)
        else:
            self._time_of_goal_state += 1

    def _near_goal_state(self):
        goal_state_as_arr = np.array(list(self._goal_state))
        curr_loc = np.array(self._obs['agent']['position']['location'])
        return np.linalg.norm(goal_state_as_arr - curr_loc) <= 50

    def _make_partial_goal(self, goal_state):
        # make partial goal state as of now we just copy the goal state
        return goal_state

