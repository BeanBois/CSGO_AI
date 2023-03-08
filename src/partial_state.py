import gym
from gym import spaces
from gym.spaces import Dict, Sequence, Tuple, Discrete, Box
from awpy.data import NAV_CSV
from numpy import np
import threading as th
#import servers, they have started already btw
from csgo_gsi_python import TRAINING
import time
if TRAINING:
    from csgo_gsi_python import GSI_SERVER_TRAINING

#importing input library
from pynput import mouse, keyboard
from pynput.mouse import Button
from pynput.keyboard import Key

TIME_STEPS = 40

class CSGO_Env_Utils:
    
    def location_domain(max_x, min_x, max_y, min_y, max_z, min_z):
        return Box(
            low = np.array([min_x, min_y, min_z]), 
            high = np.array([max_x, max_y, max_z]), 
            dtype = np.int32)
        #int to round off
        # return Tuple((
        #     spaces.Discrete(max_x - min_x + 1, start =min_x), #x
        #     spaces.Discrete(max_y - min_y + 1, start =min_y), #y
        #     spaces.Discrete(max_z - min_z + 1, start =min_z), #z
        # ))
    
    def forward():
        #each forward has domain [-1:1]
        return Box(
            low = np.array([-1, -1, -1]),
            high = np.array([1, 1, 1]),
            dtype = np.float32
        )
    
    def action_space_domain(SCREEN_HEIGHT,SCREEN_WIDTH):
        return Tuple(
            Discrete(2), #shift pressed? #shift pressed == walking, else running
            Discrete(2), #ctrl pressed? #crouching basically, ctrl pressed == crouching, else standing
            Discrete(2), #space pressed? #jumping basically, space pressed == jumping, else standing
            Discrete(2), #fire? #fire == 1, else 0 #left mouse click
            Discrete(5), #0 for no button pressed, 1 for 'w', 2 for 'a', 3 for 's', 4 for 'd', 
            Box(low = np.array[0,0], high = np.array[SCREEN_HEIGHT,SCREEN_WIDTH], dtype = np.float32) #location of cursor
        )
    
    def observation_space_domain(max_x, min_x, max_y, min_y, max_z, min_z, SCREEN_HEIGHT, SCREEN_WIDTH):
        return Dict({
            'obs_type' : spaces.Discrete(2), #0 for partial, 1 for  complete
            'enemy' : {
                'position' :{
                    'areaId' : spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                    'location' : CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),

                    'forward' : CSGO_Env_Utils.forward(),
                    'time_seen' : spaces.Discrete(TIME_STEPS),
                },
                'health' : spaces.Discrete(100),
            },
            'enemy_screen_location' :{
                'pixel_location' : Box(low = np.array[0,0], high = np.array[SCREEN_HEIGHT,SCREEN_WIDTH], dtype = np.float32),
            },
            'agent' : {
                'position' : {
                        'areaId' : spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                        'location' : CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
                        'forward' : CSGO_Env_Utils.forward(),
                    },
                'agent_gun' : spaces.Discrete(1), #fixed
                'agent_bullets' : spaces.Discrete(30), #fixed
                'health' :  spaces.Discrete(100),                
            },
            'bomb_location' : {
                'areaId' : spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                'location' : CSGO_Env_Utils.location_domain(max_x, min_x, max_y, min_y, max_z, min_z),
                'forward' : CSGO_Env_Utils.forward(),
            },
            'bomb_defusing' : Tuple(spaces.Discrete(2), spaces.Discrete(TIME_STEPS))#fixed, 0 for not defusing, 1 for defusing
            'current_time' : spaces.Discrete(TIME_STEPS),
            'winner' : spaces.Discrete(3), # 0 for ongoing, 1 for agent win, 2 for enemy win
        })


    def initialise_game():
        pass
    
    def reset_game():
        pass
class CSGO_Env(gym.Env):
    MAP_NAME = 'de_dust2'
    MAP_DATA = NAV_CSV[NAV_CSV["mapName"] == MAP_NAME]
    SCREEN_HEIGHT = 768
    SCREEN_WIDTH = 1024
    OBSERVING_TIME = 0.1
    ACTION_TIME = 0.1
    #Env is made up of segmented areas of map. Each area is represented by a super node
    #do we include screen location of enemy in observation space?
    def __init__(self):
        self._init_para()
        #observation space can be abstractly thought of as a set of nodes
        #each node encapsulates the following
        self._obs = None
        self._reward = 0
        # self._prev_action = None
        self.observation_space = CSGO_Env_Utils.observation_space_domain(self.max_x, self.min_x, self.max_y, self.min_y, self.max_z, self.min_z, self.SCREEN_HEIGHT, self.SCREEN_WIDTH)
        self.action_space = CSGO_Env_Utils.action_space_domain(self.SCREEN_HEIGHT, self.SCREEN_WIDTH)
        
    def _init_para(self):
        self.min_x, self.max_x = None, None
        self.min_y, self.max_y = None, None
        self.minz, self.max_z = None, None
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        for data in CSGO_Env.MAP_DATA:
            if self.min_x is None or self.min_x > int(data["northWestX"]):
                self.min_x = int(data["northWestX"])
                
            if self.max_x is None or self.max_x < int(data["southEastX"]):
                self.max_x = int(data["southEastX"])
                
            if self.min_y is None or self.min_y > int(data["northWestY"]):
                self.min_y = int(data["northWestY"])
                
            if self.max_y is None or self.max_y < int(data["southEastY"]):
                self.max_y = int(data["southEastY"])
                
            if self.min_z is None or self.min_z > int(data["northWestZ"]):
                self.min_z = int(data["northWestZ"])
                
            if self.max_z is None or self.max_z < int(data["southEastZ"]):
                self.max_z = int(data["southEastZ"])

    #(observation, reward, done, info)
    #each step corresponds to 0.1 seconds (OBSERVING_TIME or ACTION_TIME)
    def step(self, action):
        if TRAINING:
            #create lock
            lock = th.RLock()            
            
            #create threads
            observing_thread= th.Timer(CSGO_Env.OBSERVING_TIME, self._get_full_state, args = (lock,))
            # observing_thread_2 = th.Timer(CSGO_Env.OBSERVING_TIME, self._get_full_state, args = (lock,))
            action_thread = th.Timer(CSGO_Env.ACTION_TIME, self._apply_action, args = (action,))
            

            
            #get current_state
            prev_observation = self._obs
            
            #start observing
            observing_thread.start()
            
            #apply action at the same time
            action_thread.start()
            
            #create reward thread
            #this is so to ensure that we calculate reward only after we have the next observation
            reward_thread = th.Thread(target = self._get_reward, args = (prev_observation, lock, action))
            
            # #get next observation
            # observing_thread_2.start()
            
            #reward thread has to wait for observing thread, so we ensure all threads are properly ran through in the function
            #self._obs records the latest _obs 
            reward_thread.start()
            reward_thread.join()
            
            return self._obs, self._reward, self._is_done(), {}
        else:
            pass
    
    def _get_full_state(self,lock):
        #here we assume TRAINING is true
        with lock:
            self._obs = self._make_observation_complete()
        
    def _get_reward(self, prev_obs ,lock, action):
        with lock:
            if prev_obs is None:
                self._reward = 0
            else:
                self._reward = self._calculate_reward(prev_obs,action)        
    #Design this please brother you my god fr 
    def _calculate_reward(self, prev_obs, action):
        #divide all by 10 since now step is in 0.1 seconds <if reward is time based>
        # +0.01 for every survived second
        # -0.25 if bomb is
        # +0.01 for every survived iteration 
        # -0.25 if bomb is heard/seen to be defused, reset to +0.01 by seeing (with 90% certainty) bomb is not defused. 
        # +1 for win. -1 for lose
        # -0.001 for making noise<Running, Firing gun, Reloading> if opponent not seen since 5 sec
        # +0.1 for spotting enemy
        # -0.2 for getting hit by enemy
        # +0.2 for hitting enemy 
        
        #first check if the game is lost or won
        if self._obs['winner'] != 0:
            return 1 if self._obs['winner'] == 1 else -1
        
        #if game ongoing
        else:
            cost = 0
            reward = 0
            #check if bomb is being defused
            prev_bomb_defusing = prev_obs['bomb_defusing'][0]
            prev_info_timestamp = prev_obs['bomb_defusing'][1]
            bomb_defusing = self._obs['bomb_defusing'][0]
            info_timestamp = self._obs['bomb_defusing'][1]
            prev_enemy_health = prev_obs['enemy']['health']
            cur_enemy_health = self._obs['enemy']['health']
            #reward +0.5 if bomb is prevented from defusing
            if prev_bomb_defusing == 1 and bomb_defusing == 0 and info_timestamp > prev_info_timestamp:
                reward += 0.5
            else:
                pass
            
            #if bomb defusing, we penalize per timestep, unless enemy is being hit
            if bomb_defusing == 1:
                if prev_enemy_health > cur_enemy_health:
                    reward += 0.35
                else:
                    reward -= 0.025
            
            #if bomb not defusing we take note focus on finding the enemy and hiding information
            else:
                if prev_enemy_health > cur_enemy_health:
                    reward += 0.25
                else:
                    if action[3] == 6:
                        cost +=  0.005 #cost for making noise and wasting gun ammo
                prev_enemy_location = prev_obs['enemy']['position']
                prev_enemy_timestamp = prev_enemy_location['time_seen']
                curr_enemy_location = self._obs['enemy']['position']
                curr_enemy_timestamp = curr_enemy_location['time_seen']
                obs_type = self._obs['obs_type']
                #enemy recently seen, reward += 0.1
                if curr_enemy_location is not None and \
                    curr_enemy_timestamp > prev_enemy_timestamp and\
                        obs_type == 0: #need to replace here to accommodate for the fact that full state should not receive this reward
                    reward += 0.1
                
                else:
                    reward += 0.001
                
                #if shift/ctrl not press when moving , penalize for making sound
                #action[1] corr to ctrl, action[0] corr to shift
                #action[4] corr to movement key, action[2] corr to jump key
                if (action[1] == 0 or action[0] == 0) and \
                    (action[4] != 0 or action[2] != 0):
                    cost += 0.001
            
            agent_health = self._obs['agent']['health']
            prev_agent_health = prev_obs['agent']['health']
            if prev_agent_health > agent_health:
                if bomb_defusing == 1:
                    pass
                else:
                    if agent_health <= 50:
                        reward -= 0.4
                    else:
                        reward -= 0.2
        return reward - cost

    #way we apply action might result very straight forward
    #if action dont explicitly state to press a key, we release it
    def _apply_action(self, action):
        shift_pressed = True if action[0] == 1 else False
        ctrl_pressed = True if action[1] == 1 else False
        spacebar_pressed = True if action[2] == 1 else False
        movement_button = None
        left_click = True if action[4] == 1 else False
        cursor_location = action[5]
        if action[3] == 1:
            movement_button = Key.w
        elif action[3] == 2:
            movement_button = Key.a
        elif action[3] == 3:
            movement_button = Key.s
        elif action[3] == 4:
            movement_button = Key.d
                
        if movement_button is not None:
            list_of_keys = [Key.w, Key.a, Key.s, Key.d] - [movement_button]
            self.keyboard_controller.release(*list_of_keys)
            if shift_pressed and ctrl_pressed and spacebar_pressed:
                with self.keyboard_controller.pressed(Key.shift, Key.ctrl, Key.space):
                    self.keyboard_controller.press(movement_button)
            elif shift_pressed and ctrl_pressed:
                self.keyboard_controller.release(Key.space)
                with self.keyboard_controller.pressed(Key.shift, Key.ctrl):
                    self.keyboard_controller.press(movement_button)
            elif shift_pressed and spacebar_pressed:
                self.keyboard_controller.release(Key.ctrl)
                with self.keyboard_controller.pressed(Key.shift, Key.space):
                    self.keyboard_controller.press(movement_button)
            elif ctrl_pressed and spacebar_pressed:
                self.keyboard_controller.release(Key.shift)
                with self.keyboard_controller.pressed(Key.ctrl, Key.space):
                    self.keyboard_controller.press(movement_button)
            elif shift_pressed:
                self.keyboard_controller.release(Key.ctrl, Key.space)
                with self.keyboard_controller.pressed(Key.shift):
                    self.keyboard_controller.press(movement_button)
            elif ctrl_pressed:
                self.keyboard_controller.release(Key.shift, Key.space)
                with self.keyboard_controller.pressed(Key.ctrl):
                    self.keyboard_controller.press(movement_button)
            elif spacebar_pressed:
                self.keyboard_controller.release(Key.shift, Key.ctrl)
                with self.keyboard_controller.pressed(Key.space):
                    self.keyboard_controller.press(movement_button)
            else:
                list_of_keys = [Key.w, Key.a, Key.s, Key.d] - [movement_button]
                self.keyboard_controller.release(Key.shift, Key.ctrl, Key.space, *list_of_keys)
                self.keyboard_controller.press(movement_button)
        
        else:
            list_of_keys = [Key.w, Key.a, Key.s, Key.d]
            self.keyboard_controller.release(*list_of_keys)
            if shift_pressed:
                self.keyboard_controller.press(Key.shift)
            else:
                self.keyboard_controller.release(Key.shift)
            if ctrl_pressed:
                self.keyboard_controller.press(Key.ctrl)
            else:
                self.keyboard_controller.release(Key.ctrl)
            if spacebar_pressed:
                self.keyboard_controller.press(Key.space)
            else:
                self.keyboard_controller.release(Key.space)
         
        if left_click:
            self.mouse_controller.position = cursor_location
            self.mouse_controller.click(Button.left)
        else:
            self.mouse_controller.position = cursor_location
        #sleep to run through the timed thread
        time.sleep(self.ACTION_TIME)   
        
    #TODO: Change datatype
    #DONE, TODO: Check
    def _make_observation_complete(self):
        agent = GSI_SERVER_TRAINING.get_info("player")
        phase_cd = GSI_SERVER_TRAINING.get_info("phase_countdowns")
        round_info = GSI_SERVER_TRAINING.get_info("round")
        match_result = round_info['win_team'] 
        if match_result == 'T':
            match_result = 1
        elif match_result == 'CT':
            match_result = 2
        else:
            match_result = 0
        enemy = {}
        players = GSI_SERVER_TRAINING.get_info("allplayers")#returns dictionary of bombstate
        for player in players:
            if player['name'] != agent['name']:
                enemy = player
                break
        bomb = GSI_SERVER_TRAINING.get_info("bomb")
        agent_weapon = [weapon for weapon in agent['weapons'] if weapon['state'] == 'active'][0]
        
        return{
            'obs_type' : 1,
            'enemy' : {
                'position' :{
                    'areaId' : None,
                    'location' : np.fromstring(enemy['position']),
                    'forward' : np.fromstring(enemy['forward']),
                    'time_seen' : float(phase_cd['phase_ends_in']),
                },
                'health' : int(enemy['state']['health']),
                },
            'agent' : {
                'position' : {
                    'areaId' : None,
                    'location' : np.fromstring(agent['position']),
                    'forward' : np.fromstring(agent['forward']),
                },
                'agent_gun' : agent_weapon['name'],
                'agent_bullets' : int(agent_weapon['ammo_clip']),
                'health' : int(agent['state']['health']),
                },
            'bomb location' : np.fromstring(bomb['position']),
            'bomb defusing' : 1 if bomb['state'] == 'defusing' else 0,
            'current_time' : int(phase_cd['phase_ends_in']),
            'winner' : match_result
            
        }
    
    def reset(self):
        

        
######BETA CODE########
"""
class Partial_State():
    
    def __init__(self, current_location, enemy_location = None, enemy_location_time = None, enemy_health = None,agent_gun = None, agent_bullets = None, 
                 agent_health = None, bomb_defusing = None, current_time = None):
        
        self.state = Dict({
            'enemy' : {
                'position' :{
                    'location ' : enemy_location.space,
                    'time_seen' : enemy_location_time, #spaces.Discrete(TIME_STEPS),
                },
                'health' : enemy_health, #spaces.Discrete(100)
            },
            'agent' :{
                'position' : current_location.space,
                'agent_gun' : agent_gun, # spaces.Discrete(1), #fixed
                'agent_bullets' : agent_bullets, #spaces.Discrete(30) #fixed
                'health' : agent_health,#spaces.Discrete(100),                
            },
            'bomb defusing' : bomb_defusing, # spaces.Discrete(2), #fixed, 0 for not defusing, 1 for defusing
            'current_time' : current_time, #spaces.Discrete(TIME_STEPS),
        })
            
class Partial_State_Space():
    
    def __init__(self, current_location, enemy_location = None, enemy_location_time = None, enemy_health = None,agent_gun = None, agent_bullets = None, 
                 agent_health = None, bomb_defusing = None, current_time = None):
        
        self.state = Dict({
            'enemy' : {
                'position' :{
                    'location ' : enemy_location.space, #SUPER NODE
                    'time_seen' : enemy_location_time, #spaces.Discrete(TIME_STEPS),
                },
                'health' : enemy_health, #spaces.Discrete(100)
            },
            'agent' :{
                'position' : current_location.space, #SUPER NODE
                'agent_gun' : agent_gun, # spaces.Discrete(1), #fixed
                'agent_bullets' : agent_bullets, #spaces.Discrete(30) #fixed
                'health' : agent_health,#spaces.Discrete(100),                
            },
            'bomb defusing' : bomb_defusing, # spaces.Discrete(2), #fixed, 0 for not defusing, 1 for defusing
            'current_time' : current_time, #spaces.Discrete(TIME_STEPS),
        })
            
class Action():
    
    def __init__(self, action_id, cursor_loc):
        self.action_id = action_id
        self.cursor_loc = cursor_loc


def __init__(self):
    self._init_para()
    #observation space can be abstractly thought of as a set of nodes
    #each node encapsulates the following
    self.observation_space = Dict({
        'enemy' : {
            'position' :{
                        'areaId' : spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                        'location' : CSGO_Env_Utils.location_domain(self.max_x, self.min_x, self.max_y, self.min_y, self.max_z, self.min_z),
                        #     Tuple((
                        #     spaces.Discrete(self.max_x - self.min_x + 1, start =self.min_x), #x
                        #     spaces.Discrete(self.max_y - self.min_y + 1, start =self.min_y), #y
                        #     spaces.Discrete(self.max_z - self.min_z + 1, start =self.min_z), #z
                        # )),
                        #each forward has domain [-1:1]
                        'forward' : CSGO_Env_Utils.forward(),
                        #     Tuple((
                        #     spaces.Discrete(),
                        #     spaces.Discrete(), 
                        #     spaecs.Discrete(),
                        # ))
                    # Sequence([
                        # Tuple((
                        #     spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                        #     spaces.Discrete(int(data["southEastX"]) - int(data["northWestX"]) + 1, start =int(data["northWestX"])), #x
                        #     spaces.Discrete(int(data["southEastY"]) - int(data["northWestY"]) + 1, start =int(data["northWestY"])), #y
                        #     spaces.Discrete(int(data["southEastZ"]) - int(data["northWestZ"]) + 1, start =int(data["northWestZ"])), #z
                            
                        #     # spaces.Discrete(360), #p
                        #     # spaces.Discrete(360), #y
                        #     # spaces.Discrete(1), #r
                        # )),
                        # for data in CSGO_Env.MAP_DATA
                    # ]),
                'time_seen' : spaces.Discrete(TIME_STEPS),
            },
            'health' : spaces.Discrete(100)
        },
        'agent' :
            {
            'position' : 
                # Sequence([
                    {
                        'areaId' : spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                        'location' : CSGO_Env_Utils.location_domain(self.max_x, self.min_x, self.max_y, self.min_y, self.max_z, self.min_z),
                        #     Tuple((
                        #     spaces.Discrete(),
                        #     spaces.Discrete(),
                        #     spaces.Discrete(),
                        # )),
                        'forward' : CSGO_Env_Utils.forward(),
                        #     Tuple((
                        #     spaces.Discrete(),
                        #     spaces.Discrete(),
                        #     spaecs.Discrete(),
                        # ))
                    },
                    # for data in CSGO_Env.MAP_DATA
                # ]),
            'agent_gun' : spaces.Discrete(1), #fixed
            'agent_bullets' : spaces.Discrete(30), #fixed
            'health' :  spaces.Discrete(100),                
        },
        'bomb location' : 
        # Sequence([
            {
                'areaId' : spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                'location' : Tuple((
                    spaces.Discrete(),
                    spaces.Discrete(),
                    spaces.Discrete(),
                )),
                'forward' : Tuple((
                    spaces.Discrete(),
                    spaces.Discrete(),
                    spaecs.Discrete(),
                ))
            },
        'bomb defusing' : spaces.Discrete(2), #fixed, 0 for not defusing, 1 for defusing
        'current_time' : spaces.Discrete(TIME_STEPS),
    })
    
    self.action_space = Tuple(
        Discrete(8), #0 for walk, 1 for run, 2 for jump, 3 for crouch, 4 for stay, 5 for prone, 6 for shoot, 7 for aim
        Tuple(Discrete(1920), Discrete(1080)), #Mouse movement, controlling mouse location on screen basically
    )
    
    #way we apply action might result in very choppy actions so lookout for that
    #Implemented for smoothness now!
    #Basically we compare the current action and the previous action
    #if both action share the same key, we hold the key
    #if current action has a key that is not in the previous action, we press the key
    #if current action does not have a key that is in the previous action, we release the key
    def _apply_action(self, action):
        if self._prev_action is not None:
            #get prev action and current action
            shift_pressed = True if action[0] == 1 else False
            prev_shift_pressed = True if self._prev_act[0] == 1 else False
            if 
            
            ctrl_pressed = True if action[1] == 1 else False
            prev_ctrl_pressed = True if action[1] == 1 else False
            
            spacebar_pressed = True if action[2] == 1 else False
            prev_spacebar_pressed = True if self._prev_act[2] == 1 else False
            
            movement_button = action[3]
            left_click = True if action[4] == 1 else False
            cursor_location = action[5]
            keys_to_release = []
            keys_to_hold = []
            mouse_control = []
        else:
        
            
            self._send_keyboard_command(keys_to_press)
            self._send_mouse_command(mouse_control)
    """