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
import math
floor = math.floor

from gsi_client import client
from awpy.data import NAV_CSV

# importing input library
from pynput import mouse, keyboard
from pynput.mouse import Button
from pynput.keyboard import Key
import time
from gym.spaces.utils import flatdim

TRAINING = True
TIME_STEPS = 400



class Observation:
    map_domain = None
    def set_map_domain(map_domain):
        Observation.map_domain = map_domain


class CompleteObservation(Observation):
    
    def __init__(self):
        self.agent_location = None #location of agent
        self.agent_forward = None #forward of agent
        self.agent_health = None #health of agent
        self.time = None #time of the game
        self.agent_bullet = None #number of bullets in the agent's magazine

        self.enemy_location = None #location of enemy
        self.enemy_forward = None #forward of enemy
        self.enemy_health = None #health of enemy
        
        self.enemy_last_seen = None #last seen location of enemy
        self.enemy_on_screen_now = None #is enemy on screen now
        self.enemy_on_radar_now = None #is enemy on radar now
        
        
        self.match_result = None
        self.bomb_location = None
        self.bomb_defusing = None
        self.bomb_info_time = None 

    def update_observation(self, information):
        agent = information["player"]
        phase_cd = information["phase_countdowns"]
        round_info = information["round"]
        match_result = 0
        agent_health = agent['state']['health']
        enemy = {}
        players = information["allplayers"]  # returns dictionary of bombstate
        
        for player in players:
            if players[player]['name'] != agent['name']:
                enemy = players[player]
                break
        enemy_health = enemy['state']['health']
        if "bomb" in round_info.keys():
            bomb_state = round_info['bomb']
            if bomb_state == 'exploded' or enemy_health <= 0:
                match_result = 1 # bomb exploded or enemy killed, Terrorist win
            elif bomb_state == 'defused':
                #we do not care if agent dies as bomb can still explode after agent death
                match_result = 2 # bomb defused, Counter Terrorist win

        # bomb = GSI_SERVER_TRAINING.get_info("bomb")
        bomb = information["bomb"]
        agent_weapon = [agent['weapons'][weapon] for weapon in agent['weapons']
                        if agent['weapons'][weapon]['state'] == 'active'][0]
        enemy_on_radar = information['enemy_on_radar']
        enemy_screen_coord = information['enemy_screen_coords'].get(
            'head', None)
        if enemy_screen_coord is None:
            enemy_screen_coord = information['enemy_screen_coords'].get(
                'body', None)

        self.enemy_location = np.array(enemy['position']['location'].split(','), dtype=np.float32)
        self.enemy_forward = np.array(enemy['forward'].split(','), dtype=np.float32)
        self.enemy_last_seen = int(float(phase_cd['phase_ends_in']))
        self.enemy_on_screen_now = 1 if enemy_screen_coord is not None else 0
        self.enemy_on_radar_now = 1 if enemy_on_radar == 'visible' else 0
        self.enemy_health = int(enemy['state']['health'])
        self.agent_location = np.array(agent['position'].split(','), dtype=np.float32)
        self.agent_forward = np.array(agent['forward'].split(','), dtype=np.float32)
        self.agent_health = int(agent['state']['health'])
        self.agent_bullet = int(agent_weapon['ammo_clip']) if "ammo_clip" in agent_weapon.keys() else 0
        self.bomb_location = np.array(bomb['position'].split(','), dtype=np.float32)
        self.bomb_defusing = 1 if bomb['state'] == 'defusing' else 0
        self.bomb_info_time =  int(float(phase_cd['phase_ends_in']))
        self.time = int(float(phase_cd['phase_ends_in']))
        self.match_result = match_result
    #converts the observation to an array
    def convert_to_array(self):
        
        #agent info first
        agent_location = CSGO_Env_Utils.location_to_array(self.agent_location, Observation.map_domain)
        agent_forward = CSGO_Env_Utils.forward_to_array(self.agent_forward)
        agent_health = CSGO_Env_Utils.health_to_array(self.agent_health)
        time = CSGO_Env_Utils.time_to_array(self.time)
        
        #enemy info
        enemy_location = CSGO_Env_Utils.location_to_array(self.enemy_location, Observation.map_domain)
        enemy_forward = CSGO_Env_Utils.forward_to_array(self.enemy_forward)
        enemy_health = CSGO_Env_Utils.health_to_array(self.enemy_health)
        enemy_last_seen = CSGO_Env_Utils.time_to_array(self.enemy_last_seen)
        enemy_on_screen_now = CSGO_Env_Utils.bool_to_array(self.enemy_on_screen_now)
        enemy_on_radarr_now = CSGO_Env_Utils.bool_to_array(self.enemy_on_radar_now)
        
        
        #bomb info
        bomb_location = CSGO_Env_Utils.location_to_array(self.bomb_location, Observation.map_domain)
        bomb_defusing = CSGO_Env_Utils.bool_to_array(self.bomb_defusing)
        bomb_info_time = CSGO_Env_Utils.time_to_array(self.bomb_info_time)
        
        
        return np.concatenate((agent_location, agent_forward, agent_health, time, enemy_location, enemy_forward, enemy_health, enemy_last_seen, enemy_on_screen_now, enemy_on_radarr_now, bomb_location, bomb_defusing, bomb_info_time))
    
class PartialObservation(Observation):
    
    def __init__(self):
        self.agent_location = None #location of agent
        self.agent_forward = None #forward of agent
        self.agent_health = None #health of agent
        self.time = None #time of the game
        self.agent_bullet = None #number of bullets in the agent's magazine

        self.enemy_location = None #location of enemy
        self.enemy_forward = None #forward of enemy
        self.enemy_health = None #health of enemy
        
        self.enemy_last_seen = None #last seen location of enemy
        self.enemy_on_screen_now = None #is enemy on screen now
        self.enemy_on_radar_now = None #is enemy on radar now
        
        
        self.match_result = None
        self.bomb_location = None
        self.bomb_defusing = None
        self.bomb_info_time = None
        

    def update_observation(self, information):
        # if enemy not on bomb, we know bomb not defusing
        # if see bomb on screen, agent will know whether bomb is defused

        # Get player information first
        agent = information["player"]
        agent_weapon = [agent['weapons'][weapon] for weapon in agent['weapons']
                        if agent['weapons'][weapon]['state'] == 'active'][0]
        # get time
        phase_cd = information["phase_countdowns"]

        # get round information
        round_info = information["round"]
        match_result = 0
        if "bomb" in round_info.keys():
            bomb_state = round_info['bomb']

            if bomb_state == 'exploded':
                match_result = 1 # bomb exploded, Terrorist win
                
            elif bomb_state == 'defused':
                match_result = 2 # bomb defused or agent killed, Counter Terrorist win
        
        # get prev obv of enemy
        enemy_loc = self.enemy_location if self.enemy_location is not None else None
        enemy_forw = self.enemy_forward if self.enemy_forward is not None else None
        enemy_health = self.enemy_health if self.enemy_health is not None else 100
        time_of_info_enemy = self.enemy_last_seen if self.enemy_last_seen is not None else None

        # get prev obv of bomb
        bomb_state = self.bomb_defusing if self.bomb_defusing is not None else 0
        time_of_info_bomb = self.bomb_info_time if self.bomb_info_time is not None else None

        # check if we have seen the bomb
        # here we area also 'simulating' the ability of the agent to be able to
        # distinguish whether the bomb is defusing or not with just the image of the bomb
        # This is not an easy task since the vision cues for defusing is not very clear
        # To simulate this ability, once we have seen the bomb on the screen, we will
        # stream the state of the bomb from the game server
        curr_bomb_state = information['bomb']['state']


        # Decaying information, we set a expiry time for the information we collected
        # now if our time_of_info_bomb is way too old <5s, we delete the information as irrelevant
        if time_of_info_bomb is None or int(float(phase_cd['phase_ends_in'])) - time_of_info_bomb > 5:
            bomb_state = None
            time_of_info_bomb = int(float(phase_cd['phase_ends_in']))

        # same goes for time_of_info_enemy
        if time_of_info_enemy is None or int(float(phase_cd['phase_ends_in'])) - time_of_info_enemy > 5:
            enemy = None
            enemy_pos = None
            enemy_loc = None
            enemy_forw = None
            time_of_info_enemy = int(float(phase_cd['phase_ends_in']))

        # now we see if we have 'seen' enemy
        if information['allplayers'] is not None:
            # returns dictionary of bombstate
            players = information["allplayers"]
            for player in players:
                if players[player]['name'] != agent['name']:
                    enemy = players[player]
                    break

            enemy_loc = np.array(
                enemy['position'].split(','), dtype=np.float32)
            enemy_forw = np.array(
                enemy['forward'].split(','), dtype=np.float32)
            enemy_health = enemy['state']['health']
            time_of_info_enemy = int(float(phase_cd['phase_ends_in']))
            bomb_loc = np.array(
                information["bomb"]['position'].split(','), dtype=np.float32)

           # check if enemy location is far away from bomb location
           # if yes then we know that enemy is not defusing the bomb
            if(sum((enemy_loc - bomb_loc)**2) > 30):
                bomb_state = 0
                time_of_info_bomb = int(float(phase_cd['phase_ends_in']))

        # now check if enemy tried to defuse the bomb
        # done by comparing prev state of bomb and current state of bomb
        if bomb_state != 1 and curr_bomb_state == 1:
            bomb_state = 1
            time_of_info_bomb = int(float(phase_cd['phase_ends_in']))

        # update time of no information about bomb and enemy, if
        # there is no information about bomb and enemy
        if bomb_state is None:
            time_of_info_bomb = int(float(phase_cd['phase_ends_in']))

        if enemy is None:
            time_of_info_enemy = int(float(phase_cd['phase_ends_in']))

        bomb = information["bomb"]
        enemy_on_radar = information['enemy_on_radar']
        enemy_screen_coord = information['enemy_screen_coords'].get(
            'head', None)
        if enemy_screen_coord is None:
            enemy_screen_coord = information['enemy_screen_coords'].get(
                'body', None)
        
        enemy_health = enemy
        self.enemy_location = enemy_loc
        self.enemy_forward = enemy_forw
        self.enemy_health = enemy_health
        self.enemy_last_seen = time_of_info_enemy
        self.enemy_on_screen_now = 1 if enemy_screen_coord is not None else 0
        self.enemy_on_radar_now = 1 if enemy_on_radar else 0
        self.agent_location = np.array(agent['position'].split(','), dtype=np.float32)
        self.agent_forward = np.array(agent['forward'].split(','), dtype=np.float32)
        self.agent_health = int(agent['state']['health'])
        self.agent_bullet = int(agent_weapon['ammo_clip']) if "ammo_clip" in agent_weapon.keys() else 0
        self.bomb_location = np.array(bomb['position'].split(','), dtype=np.float32)
        self.bomb_defusing = bomb_state
        self.bomb_info_time =  time_of_info_bomb
        self.time = int(float(phase_cd['phase_ends_in']))
        self.match_result = match_result
        
    #converts the observation to an array
    def convert_to_array(self):
        
        #agent info first
        agent_location = CSGO_Env_Utils.location_to_array(self.agent_location, Observation.map_domain)
        agent_forward = CSGO_Env_Utils.forward_to_array(self.agent_forward)
        agent_health = CSGO_Env_Utils.health_to_array(self.agent_health)
        time = CSGO_Env_Utils.time_to_array(self.time)
        
        #enemy info
        enemy_location = CSGO_Env_Utils.location_to_array(self.enemy_location, Observation.map_domain)
        enemy_forward = CSGO_Env_Utils.forward_to_array(self.enemy_forward)
        enemy_health = CSGO_Env_Utils.health_to_array(self.enemy_health)
        enemy_last_seen = CSGO_Env_Utils.time_to_array(self.enemy_last_seen)
        enemy_on_screen_now = CSGO_Env_Utils.bool_to_array(self.enemy_on_screen_now)
        enemy_on_radarr_now = CSGO_Env_Utils.bool_to_array(self.enemy_on_radar_now)
        
        
        #bomb info
        bomb_location = CSGO_Env_Utils.location_to_array(self.bomb_location, Observation.map_domain)
        bomb_defusing = CSGO_Env_Utils.bool_to_array(self.bomb_defusing)
        bomb_info_time = CSGO_Env_Utils.time_to_array(self.bomb_info_time)
        
        
        return np.concatenate((agent_location, agent_forward, agent_health, time, enemy_location, enemy_forward, enemy_health, enemy_last_seen, enemy_on_screen_now, enemy_on_radarr_now, bomb_location, bomb_defusing, bomb_info_time))

class Goal:
    
    def __init__(self):
        self.goal_index = 0
        self.goal = None
 
    def update_goal(self, goal_index, goal):
        self.goal_index = goal_index
        self.goal = goal
    
    def convert_to_array(self,goal_size):
        arr = np.zeros(goal_size)
        arr[self.goal_index] = 1
        return arr
    
         
        
class CSGO_Env_Utils:
    
    def match_result_to_array(match_result):
        # arr = np.zeros(3)
        # arr[match_result] = 1
        # return arr
        return np.array(match_result)
    
    def bool_to_array(bool):
        # arr = np.zeros(2)
        # if bool is None:
        #     return arr
        # arr[bool] = 1
        # return arr
        return np.array(bool)
    
    def bullet_to_array(bullet):
        # bullet = bullet.clip(0, 30)
        # arr = np.zeros(30)
        # arr[bullet] = 1
        # return arr
        return np.array(bullet)
    
    def forward_to_array(forward, STEP_SIZE = 0.25):
        # arr = np.zeros((int(2/STEP_SIZE), int(2/STEP_SIZE), int(2/STEP_SIZE)))
        # if forward is None:
        #     arr.flatten()
        #     return arr
        # forward = np.array(forward, dtype=np.float32)//STEP_SIZE + 1//STEP_SIZE
        # forward = np.array(forward, dtype=np.int32)
        # arr[forward[0], forward[1], forward[2]] = 1
        # arr.flatten()
        # return arr
        arr = np.array([np.nan, np.nan, np.nan])
        if forward is not None:
            arr = np.array(forward, dtype=np.float32)
        arr.flatten()
        return arr

    def health_to_array(health):
        # arr = np.zeros(100)
        # arr[health] = 1
        # return arr
        return np.array(health)
    
    def time_to_array(time):
        # arr = np.zeros(TIME_STEPS)
        # if time is not None:
            # arr[time] = 1
        # return arr
        return np.array(time)
    
    def location_to_array(location, domain, DISCRETE_STEP_SIZE = 50):
        # max_x = domain['max_x']
        # min_x = domain['min_x']
        # max_y = domain['max_y']
        # min_y = domain['min_y']
        # max_z = domain['max_z']
        # min_z = domain['min_z']
        # arr = np.zeros(floor((max_x - min_x) / DISCRETE_STEP_SIZE), floor((max_y - min_y) / DISCRETE_STEP_SIZE), floor((max_z - min_z) / DISCRETE_STEP_SIZE))
        # location = np.array(location, dtype=np.int32)//DISCRETE_STEP_SIZE
        arr = np.array([np.nan, np.nan, np.nan])
        if location is not None:
            arr = np.array([location[0], location[1], location[2]], dtype=np.int32) 
        arr.flatten()
        return arr
    
    def location_domain(max_x, min_x, max_y, min_y, max_z, min_z):
        return Box(
            low=np.array([min_x, min_y, min_z]),
            high=np.array([max_x, max_y, max_z]),
            dtype=np.int32,
            shape=(3,))
    
    def location_domain_size(max_x, min_x, max_y, min_y, max_z, min_z, DISCRETE_STEP_SIZE = 50):
        # return (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        return floor((max_x - min_x) / DISCRETE_STEP_SIZE) * floor((max_y - min_y) / DISCRETE_STEP_SIZE) * floor((max_z - min_z) / DISCRETE_STEP_SIZE)


    def forward():
        # each forward has domain [-1:1]
        return Box(
            low=np.array([-1, -1, -1]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

    def forward_domain_size(STEP_SIZE = 0.25):
        return floor(2/STEP_SIZE) * floor(2/STEP_SIZE) * floor(2/STEP_SIZE)
    
    
    def action_space_domain():
        return Tuple([
            Discrete(1),  # 0 for ignore, 1 for no movement key
            # shift pressed? #shift pressed == walking, else running
            Discrete(1),
            # ctrl pressed? #crouching basically, ctrl pressed == crouching, else standing
            Discrete(1),
            # space pressed? #jumping basically, space pressed == jumping, else standing
            Discrete(1),
            Discrete(1),  # fire? #fire == 1, else 0 #left mouse click
            # # 0 for no button pressed, 1 for 'w', 2 for 'a', 3 for 's', 4 for 'd',
            # Discrete(5),
            #  0 for 'w', 1 for 'a', 2 for 's', 3 for 'd', <Binary Numbers>
            Discrete(1), 
            Discrete(1), 
            Discrete(1), #1 means move mouse cursor to right by 5, 0 means no movement
            Discrete(1), #1 means move mouse cursor to left by 5 0 means no movement
        ])
    
    def action_space_domain_size():
        return 2**10 -1
    
    def observation_space_domain(max_x, min_x, max_y, min_y, max_z, min_z, SCREEN_HEIGHT, SCREEN_WIDTH):
        return Dict({
            # 'obs_type': spaces.Discrete(2),  # 0 for partial, 1 for  complete
            'enemy': Dict({
                'position': Dict({
                    # data['areaId'] map id, 10 char for buffer
                    # 'areaId': spaces.Text(10),
                    'location': CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
                    'forward': CSGO_Env_Utils.forward(),
                    # 'time_seen': spaces.Discrete(TIME_STEPS),
                    'time_seen': spaces.Box(low=0, high=TIME_STEPS, shape=(1,), dtype=np.int32)
                }),
                'enemy_screen_coords': Box(low=np.array([0, 0]), high=np.array([SCREEN_HEIGHT, SCREEN_WIDTH]), dtype=np.int32),
                'health': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                # 'health': spaces.Discrete(100),
            }),
            'agent': Dict({
                'position': Dict({
                    # data['areaId'] map id, 10 char for buffer
                    # 'areaId': spaces.Text(10),
                    'location': CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
                    'forward': CSGO_Env_Utils.forward(),
                }),
                'agent_gun': spaces.Discrete(1),  # fixed
                # 'agent_bullets': spaces.Discrete(30),  # fixed
                'agent_bullets': spaces.Box(low=0,high=30,dtype=np.int32,shape=(1,),),  # fixed
                'health': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                # 'health':  spaces.Discrete(100),
            }),
            'bomb_location': Dict({
                # data['areaId'] map id, 10 char for buffer
                # 'areaId': spaces.Text(10),
                'location': CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
            }),
            # fixed, 0 for not defusing, 1 for defusing
            # 'bomb_defusing': Tuple([spaces.Discrete(2), spaces.Discrete(TIME_STEPS)]),
            # 'bomb_defusing': Tuple([spaces.Discrete(2), spaces.Box(low=0, high=TIME_STEPS, shape=(1,), dtype=np.int32)]),
            'bomb_defusing': Tuple([spaces.Box(low=0, high=2, shape=(1,), dtype=np.int32), spaces.Box(low=0, high=TIME_STEPS, shape=(1,), dtype=np.int32)]),
            # 'current_time': spaces.Discrete(TIME_STEPS),
            'current_time': spaces.Box(low=0, high=TIME_STEPS, shape=(1,), dtype=np.int32),
            # 0 for ongoing, 1 for agent win, 2 for enemy win
            # 'winner': spaces.Discrete(3),
            'winner': spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),
            
        })

    def observation_space_domain_size(max_x, min_x, max_y, min_y, max_z, min_z, SCREEN_HEIGHT, SCREEN_WIDTH):
        return CSGO_Env_Utils.location_domain_size(max_x, min_x, max_y, min_y, max_z, min_z) * \
            CSGO_Env_Utils.forward_domain_size() * \
                CSGO_Env_Utils.location_domain_size(max_x, min_x, max_y, min_y, max_z, min_z) * \
                    CSGO_Env_Utils.forward_domain_size() * \
                        100**2 * \
                           TIME_STEPS**2 * \
                               2 * 2 * 3  
    
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
        
        
class CSGO_Env(gym.Env):
    MAP_NAME = 'de_dust2'
    MAP_DATA = NAV_CSV[NAV_CSV["mapName"] == MAP_NAME]
    OBSERVING_TIME = 0.1
    ACTION_TIME = 0.1
    # Env is made up of segmented areas of map. Each area is represented by a super node
    # do we include screen location of enemy in observation space?

    def __init__(self):
        self.bombsite_choice = None
        self._obs = CompleteObservation()
        self._part_obs = PartialObservation()
        self._reward = 0
        self._set_of_goals = None
        self._goal_state = Goal()
        self._partial_goal_state = Goal()
        self._time_of_goal_state = None
        self._init_para()
        # observation space can be abstractly thought of as a set of nodes
        # each node encapsulates the following
        self.lock = th.RLock()

        # self._prev_action = None
        self.observation_space = CSGO_Env_Utils.observation_space_domain(
            self.max_x, self.min_x, self.max_y, self.min_y, self.max_z, self.min_z, SCREEN_HEIGHT, SCREEN_WIDTH)
        self.action_space = CSGO_Env_Utils.action_space_domain()
        self.goal_space = CSGO_Env_Utils.location_domain(self.max_x, self.min_x, self.max_y, self.min_y, self.max_z, self.min_z)

        self.action_space_size = CSGO_Env_Utils.action_space_domain_size()
        self.goal_space_size = len(self._set_of_goals)
        self.observation_space_size = CSGO_Env_Utils.observation_space_domain_size(self.max_x, self.min_x, self.max_y,\
            self.min_y, self.max_z, self.min_z, SCREEN_HEIGHT, SCREEN_WIDTH) 
        
        self.map_domain = {
            'min_x': self.min_x,
            'max_x': self.max_x,
            'min_y': self.min_y,
            'max_y': self.max_y,
            'min_z': self.min_z,
            'max_z': self.max_z,
        }
        Observation.set_map_domain(self.map_domain)
    
    def _init_para(self):
        self.bombsite_choice = random.choice(['BombsiteA', 'BombsiteB']) if self.bombsite_choice is None else self.bombsite_choice
        self.min_x, self.max_x = None, None
        self.min_y, self.max_y = None, None
        self.min_z, self.max_z = None, None
        # self.mouse_controller = mouse.Controller()
        # self.keyboard_controller = keyboard.Controller()
        self._set_of_goals = CSGO_Env_Utils.generate_set_of_goals(self.bombsite_choice)

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
        # GameClient.send_action(f"start {bombsite_choice}")
  

    #(observation, reward, done, info)
    # each step corresponds to 0.1 seconds (OBSERVING_TIME or ACTION_TIME)
    def step(self, action):
        

        # get round information to if its the end of the round
        round_info = client.get_info("round")
        match_result = 0
        if "bomb" in round_info.keys():
            bomb_state = round_info['bomb']

            if bomb_state == 'exploded':
                match_result = 1 # bomb exploded, Terrorist win
                
            elif bomb_state == 'defused':
                match_result = 2 # bomb defused, Counter Terrorist win

        self._apply_action(action, (match_result>0))

    
        #collect information after action applied
        information = {}
        information['player'] = client.get_info("player")
        information['phase_countdowns'] = client.get_info("phase_countdowns")
        information['round'] = client.get_info("round")
        information['allplayers'] = client.get_info("allplayers")
        information['bomb'] = client.get_info("bomb")
        
        round_info = information["round"]
        match_result = 0
        if "bomb" in round_info.keys():
            bomb_state = round_info['bomb']

            if bomb_state == 'exploded':
                match_result = 1 # bomb exploded, Terrorist win
                
            elif bomb_state == 'defused':
                match_result = 2 # bomb defused, Counter Terrorist win
                
                   # get current_state
        prev_observation = self._obs
        prev_part_observation = self._part_obs
    
        #get next state after action applied
        self._get_state(information)

        #get reward
        self._get_reward(prev_observation, prev_part_observation, action)


        info = {'goal state' : self._goal_state,'partial goal state': self._partial_goal_state}
        print("step taken, variables are: ")
        print(f"observation: {self._obs}")
        print(f"partial observation: {self._part_obs}")
        print(f"reward: {self._reward}")
        print(f"done: {self._is_done()}")
        print(f"info: {info}")
        
        return self._obs, self._part_obs, self._reward, self._is_done(), {'goal state' : self._goal_state,'partial goal state': self._partial_goal_state}

    def _is_done(self):
        if self._obs.winner is not None and self._obs.winner != 0:
            return True
        return False

    def get_current_observation(self):
        return self._obs
    
    def get_current_partial_observation(self):
        return self._part_obs

    # TODO: Fill in the blank <models> after finishing implementing them
    def _get_state(self, information):
        with self.lock:

            #get screen information
            enemy_information = EnemyDetectorClient.get_enemy_info()

            enemy_on_radar = enemy_information['enemy_on_screen'] 
            enemy_screen_coords = enemy_information['enemy_screen_coords']
            information['enemy_on_radar'] = enemy_on_radar
            information['enemy_screen_coords'] = enemy_screen_coords
            partial_information = information.copy()

            self._get_full_state(information)

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
            if enemy_screen_coords is not None or not (information['bomb']['state'] == 'defusing' and self._obs.bomb_defusing == 0):
                # self._obs['enemy']['enemy_screen_coords'] = enemy_screen_coords
                partial_information['allplayers'] = None

            self._get_partial_state(partial_information)
            self._generate_goal()

    def _get_full_state(self, information):
        # here we assume TRAINING is true
        with self.lock:
            self._obs.update_observation(information)

    def _get_partial_state(self, information):
        # here we assume TRAINING is true
        with self.lock:
            self._part_obs.update_observation(information)

# Reward function
    # TODO: adapt reward for partial state
    def _get_reward(self, prev_obs, prev_part_obs, action):
        with self.lock:
            if prev_obs is None:
                self._reward = 0
            else:
                self._reward = self._calculate_reward(
                    prev_obs, prev_part_obs, action)

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
        if self._obs.winner != 0:
            return 1 if self._obs.winner == 1 else -1

        # if game ongoing
        else:
            cost = 0
            reward = 0
            # check if bomb is being defused
            prev_bomb_defusing = prev_obs.bomb_defusing
            prev_info_timestamp = prev_obs.bomb_info_time
            bomb_defusing = self._obs.bomb_defusing
            info_timestamp = self._obs.bomb_info_time
            prev_enemy_health = prev_obs.enemy_health
            cur_enemy_health = self._obs.enemy_health

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
                prev_enemy_position = prev_part_obs.enemy_position
                prev_enemy_location = prev_enemy_position.enemy_location
                prev_enemy_timestamp = prev_enemy_position.enemy_last_seen

                curr_enemy_position = self._part_obs.enemy_position
                curr_enemy_location = curr_enemy_position.enemy_location
                curr_enemy_timestamp = curr_enemy_position.enemy_last_seen

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

            agent_health = self._obs.agent_health
            prev_agent_health = prev_obs.agent_health
            if prev_agent_health > agent_health:
                if bomb_defusing == 1:
                    #we do not penalise if fight is taken when bomb is defusing
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
        # convert action to binary form
        if action is not None:
            binary_action = action.convert_to_binary_form()
            
            GameClient.send_action(binary_action, done)
            time.sleep(self.ACTION_TIME)
    # TODO: Change datatype
    # DONE, TODO: Check

  
    def reset(self):
        start_time = time.time()
        self.bombsite_choice = random.choice(['BombsiteA', 'BombsiteB'])
        GameClient.send_action(f"restart {self.bombsite_choice}")
        print('Restarting game...')
        # get information and img
        information = {}
        information['player'] = client.get_info("player")
        information['phase_countdowns'] = client.get_info("phase_countdowns")
        information['round'] = client.get_info("round")
        information['allplayers'] = client.get_info("allplayers")
        information['bomb'] = client.get_info("bomb")
        self._get_state(information)
        print(f"took {time.time() - start_time} to reset")
        return self._obs, self._part_obs, 0, False,self._goal_state, self._partial_goal_state

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
        if self._time_of_goal_state == None:
            self._time_of_goal_state = 0
            goal = random.choice(self._set_of_goals)
            goal_index = self._set_of_goals.index(goal)
            self._goal_state.update_goal(goal_index, goal)
            self._partial_goal_state.update_goal(goal_index, goal)
            
        probability_of_staying = np.exp(-self._time_of_goal_state**2/10000)
        roll = np.random.rand()
        if roll > probability_of_staying:
            roll = np.random.rand()
            # no stay
            # generate other goals
            other_goals = self._set_of_goals.copy()
            if self._goal_state.goal in other_goals:
                other_goals.remove(self._goal_state.goal)
            other_goals = np.array(list(other_goals))
            self._time_of_goal_state = 0
            # random goal or no
            if roll > 0.8:
                # heuristic goal

                # generate complete goal_states, and with that we generate partial goal state
                curr_loc = self._obs.agent_location
                enemy_loc = self._obs.enemy_location
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
                # self._goal_state = tuple(other_goals[index])
                self._goal_state.update_goal(index, other_goals[index])
                self._partial_goal_state.update_goal(index, other_goals[index])
            else:
                # random goal
                rand_index = np.random.randint(len(other_goals))
                self._goal_state.update_goal(rand_index, other_goals[rand_index])
                # self._goal_state = other_goals[np.random.randint(
                #     len(other_goals))]
                self._partial_goal_state.update_goal(rand_index, other_goals[rand_index])
        else:
            self._time_of_goal_state += 1
    
    def _near_goal_state(self):
        goal_state_as_arr = np.array(list(self._goal_state.goal))
        curr_loc = np.array(self._obs.agent_location)
        return np.linalg.norm(goal_state_as_arr - curr_loc) <= 50

        # make partial goal state as of now we just copy the goal state
        return goal_state

 