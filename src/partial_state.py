import gym
from gym import spaces
from gym.spaces import Dict, Sequence, Tuple, Discrete, Box
from awpy.data import NAV_CSV
import numpy as np
import random
import threading as th
#import servers, they have started already btw
from csgo_gsi_python import TRAINING
import time
# if TRAINING:
#     from csgo_gsi_python import GSI_SERVER_TRAINING
from gsi_server import get_info
#importing input library
# from pynput import mouse, keyboard
# from pynput.mouse import Button
# from pynput.keyboard import Key

#path finding algorithm
import pathfinder as pf


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
            'bomb_defusing' : Tuple(spaces.Discrete(2), spaces.Discrete(TIME_STEPS)), #fixed, 0 for not defusing, 1 for defusing
            'current_time' : spaces.Discrete(TIME_STEPS),
            'winner' : spaces.Discrete(3), # 0 for ongoing, 1 for agent win, 2 for enemy win
        })

    #Use the csgo command console to configure 
    def configure_game(map_name, map_data, nav_mesh, keyboard_controller, mouse_controller):
        
        #configure game settings
        CSGO_Env_Utils.csgo_command('sv_cheats', '1') #allow cheats
        CSGO_Env_Utils.csgo_command('mp_buy_allow_grenades', '0') #dont allow grenades or any utilities
        CSGO_Env_Utils.csgo_command('mp_c4timer', '40') #Set bomb explode timer 
        CSGO_Env_Utils.csgo_command('mp_ct_default_primary', 'weapon_ak47') # Set CT default primary weapon
        CSGO_Env_Utils.csgo_command('mp_ct_default_secondary', 'weapon_ak47') # Set CT default secondary weapon
        CSGO_Env_Utils.csgo_command('mp_t_default_primary', 'weapon_ak47') # Set T default primary weapon
        CSGO_Env_Utils.csgo_command('mp_t_default_secondary', 'weapon_ak47') # Set T default secondary weapon
        
        #setting what weapons to can be used, for which team
        CSGO_Env_Utils.csgo_command('mp_weapons_allow_heavy', '-1' ) #-1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        CSGO_Env_Utils.csgo_command('mp_weapons_allow_pistols', '-1') #-1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        CSGO_Env_Utils.csgo_command('mp_weapons_allow_rifles', '-1') #-1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        CSGO_Env_Utils.csgo_command('mp_weapons_allow_smgs', '-1') #-1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        #ff_damage_bullet_penetration 0/1 to allow bullet penetration
        
        #interesting command mp_weapon_self_inflict_amount [_], 0 for no self inflict damage, 1 for self inflict damage for mimssing shot
        #intersting command mp_plant_c4_anywhere to plant anywhere
        CSGO_Env_Utils.csgo_command('mp_give_player_c4', '1') # Give T bomb
        CSGO_Env_Utils.csgo_command('mp_halftime', '0') # dont switch team in half time
        
        #sound distance
        play_sound_distance = 1000
        # CSGO_Env_Utils.csgo_command('play_distance', f'{play_sound_distance}') #dont show sound
        #interesting command -- playgmaesound [Sound] can use to train sound model
        #sound_device_list to list all sound device
        
        #radarvisdistance [Distance
        
        
        #bots
        #CSGO_Env_Utils.csgo_command('bot_goto_selected', '0') #navigation
        #CSGO_Env_Utils.csgo_command('bot_goto_mark', '0') #navigation
        
        CSGO_Env_Utils.csgo_command('bot_allow_grenades', '0') #dont allow grenades or any utilities
        CSGO_Env_Utils.csgo_command('bot_allow_machine_guns', '0') #dont allow grenades or any utilities
        CSGO_Env_Utils.csgo_command('bot_allow_pistols', '0') #dont allow grenades or any utilities
        CSGO_Env_Utils.csgo_command('bot_allow_rifles', '0') #dont allow grenades or any utilities
        CSGO_Env_Utils.csgo_command('bot_allow_snipers', '0') #dont allow grenades or any utilities
        CSGO_Env_Utils.csgo_command('bot_allow_shotguns', '0') #dont allow grenades or any utilities
        CSGO_Env_Utils.csgo_command('bot_allow_sub_machine_guns', '0') #dont allow grenades or any utilities
        
        
        
        CSGO_Env_Utils.csgo_command('notarget') # bot ignores player
        CSGO_Env_Utils.csgo_command('mp_random_spawn', '3') # random spawn for enemy bot, not agent
        CSGO_Env_Utils.csgo_command('mp_random_spawn_los', '1') # random spawn for enemy bot to ensure that enemy bot not in sight of agent
        #CSGO_Env_Utils.csgo_command('bot_max_vision_distance_override', '1', '30') # random spawn for enemy bot, not agent
        CSGO_Env_Utils.csgo_command('custom_bot_difficulty', '2') # [0:4] 0 :EASIEST, 1: EASY, 2: NORMAL, 3: HARD, 4: HARDEST
        CSGO_Env_Utils.csgo_command('bot_freeze', '1') # 1 bot              
        CSGO_Env_Utils.csgo_command('bot_loadout', 'weapon') # sets the bot loadout                        
                   

    def start_game(map_name, map_data, nav_mesh, keyboard_controller, mouse_controller, ): #server):
        #choose bombsite
        bombsite_choice = random.choice(['BombsiteA', 'BombsiteB'])
        spawn = map_data[map_data['areaName'] == 'TSpawn'].sample()
        bombsite = map_data[map_data['areaName'] == bombsite_choice].sample()
        
        path = nav_mesh.search_path(start = tuple([float(spawn['northWestX']), float(spawn['northWestY']), float(spawn['northWestZ'])]),
                                   finish = tuple([float(bombsite['northWestX']), float(bombsite['northWestY']), float(bombsite['northWestZ'])]),
                                    )
        bomb_position = Box(low = np.array([bombsite['northWestX'], bombsite['northWestY']]), 
                            high = np.array([bombsite['southEastX'], bombsite['southEastY']]), 
                            dtype = np.int32).sample()
       
        path.append(np.asarray(bomb_position))
        
        #first check if the player has a bomb
        

        
        #navigating to bomb site
        # CSGO_Env_Utils.navigate(path, keyboard_controller, mouse_controller, server)
        CSGO_Env_Utils.navigate(path, keyboard_controller, mouse_controller)

        
        #ensure that the player is at the bomb site
        # player_info = server.get_info("player")
        player_info = get_info("player")

        print('player info : ', player_info)
        curr_loc = np.array(player_info['position'].split(','), dtype = np.float32)

        # curr_loc = np.fromstring(player_info['location'], dtype = np.int32, sep = ',')
        assert (
            curr_loc[0] >= bombsite['northWestX'] and curr_loc[0] <= bombsite['southEastX'] and\
            curr_loc[1] >= bombsite['northWestY'] and curr_loc[1] <= bombsite['southEastY'] and\
            curr_loc[2] >= bombsite['northWestZ'] and curr_loc[2] <= bombsite['southEastZ']
        )

        #plant bomb
        
        #player switch to bomb
        keyboard_controller.press('5')
        
        
        #initialise bomb plant
        with mouse_controller.pressed(Button.left):
            # round_status = server.get_info('round')
            round_status = get_info('round')
            while 'bomb' not in round_status.keys():
                # round_status = server.get_info('round')
                round_status = get_info('round')
                if 'bomb' in round_status.keys():
                    if round_status['bomb'] == 'planted':
                        break
        
        print('bomb planted')   
    
    def csgo_command(command, keyboard_controller ,*args):
        #open terminal
        keyboard_controller.press('~')
        keyboard_controller.release('~')
        for char in command:
            keyboard_controller.press(char)
            keyboard_controller.release(char)
        
        for i in range(len(args)):
            keyboard_controller.press(Key.space)
            keyboard_controller.release(Key.space)
            for char in args[i]:
                keyboard_controller.press(char)
                keyboard_controller.release(char)
                
        #send command
        keyboard_controller.press(Key.enter)
        keyboard_controller.release(Key.enter)

        #close terminal 
        keyboard_controller.press('~')
        keyboard_controller.release('~')
        print('command sent')
        
    def navigate(path, keyboard_controller, mouse_controller, player = "player"):
        not_reached = True
        path.append(None)
        next_node = path.pop(0)
        while(not_reached):
            if next_node is None: #reached destination
                not_reached = False
            
            #rmb to enclose in try block
            # player_info = server.get_info("player")
            player_info = get_info("player")
            print('player info : ', player_info)
            # curr_loc = np.fromstring(player_info['position'], dtype = np.int32, sep = ',')
            curr_loc = np.array(player_info['position'].split(','), dtype = np.float32)
            print('current location : ', curr_loc)
            # forward = np.fromstring(player_info['forward'], dtype = np.float32, sep = ',') 
            forward = np.array(player_info['forward'].split(','), dtype = np.float32)
            print('next node : ', next_node)
            #find difference between current location and next node
            diff = next_node.flatten() - curr_loc[:2].flatten()
            print('difference : ', diff)
            #normalize values 
            norm_diff = diff / np.linalg.norm(diff)
            print(norm_diff)
            
            #find orientation of next node 
            while(np.dot(forward[:2], norm_diff.flatten()) < 0.90):
                mouse_pos = mouse_controller.position
                
                # mouse_pos = (mouse_pos[0], mouse_pos[1] + 1)
                mouse_controller.move(1,0)
                # mouse.position = mouse_pos
                # player_info = server.get_info('player')
                player_info = get_info('player')
                # forward = np.fromstring(player_info['forward'], dtype = np.float32, sep = ',')
                forward = np.array(player_info['forward'].split(','), dtype = np.float32)

                
                
            #hard forward walk to next node
            keyboard_controller.press('w')
            while(np.dot(curr_loc[:2], next_node) < 0.90):
                # player_info = server.get_info('player')
                player_info = get_info('player')
                # curr_loc = np.fromstring(player_info['position'], dtype = np.int32, sep = ',')
                curr_loc = np.array(player_info['position'].split(','), dtype = np.float32)
            
            keyboard_controller.release('w')
            
            #by here we should have reached the node
            
            if np.all(curr_loc == next_node):
                next_node = path.pop(0)
    

    def reset_game():
        pass
    
    #function from https://stackoverflow.com/questions/11144513/cartesian-product-of-x-and-y-array-points-into-single-array-of-2d-points
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
    
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
        CSGO_Env_Utils.start_game(self.MAP_NAME, self.MAP_DATA, self.nav_mesh, self.keyboard_controller, self.mouse_controller)
        
    def _init_para(self):
        self.min_x, self.max_x = None, None
        self.min_y, self.max_y = None, None
        self.minz, self.max_z = None, None
        vertices = []
        polygons = []
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        index=0
        for data in CSGO_Env.MAP_DATA:
            x_range = np.arange(float(data["northWestX"]), float(data["southEastX"]), 1)
            y_range = np.arange(float(data["northWestY"]), float(data["southEastY"]), 1)
            z_range = np.arange(float(data["northWestZ"]), float(data["southEastZ"]), 1)
            v = CSGO_Env_Utils.cartesian_product(x_range, y_range, z_range)
            for point in v:
                vertices.append(point)
            polygons.append(tuple(range(index, index+len(vertices))))
            index += len(vertices)
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
        self.navmesh = pf.PathFinder(vertices, polygons)
        #then initialise the game
          
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
        # agent = GSI_SERVER_TRAINING.get_info("player")
        # phase_cd = GSI_SERVER_TRAINING.get_info("phase_countdowns")
        # round_info = GSI_SERVER_TRAINING.get_info("round")
        agent = get_info("player")
        phase_cd = get_info("phase_countdowns")
        round_info = get_info("round")
        match_result = round_info['win_team'] 
        if match_result == 'T':
            match_result = 1
        elif match_result == 'CT':
            match_result = 2
        else:
            match_result = 0
        enemy = {}
        # players = GSI_SERVER_TRAINING.get_info("allplayers")#returns dictionary of bombstate
        players = get_info("allplayers")#returns dictionary of bombstate

        for player in players:
            if player['name'] != agent['name']:
                enemy = player
                break
        # bomb = GSI_SERVER_TRAINING.get_info("bomb")
        bomb = get_info("bomb")
        agent_weapon = [weapon for weapon in agent['weapons'] if weapon['state'] == 'active'][0]
        
        return{
            'obs_type' : 1,
            'enemy' : {
                'position' :{
                    'areaId' : None,
                    # 'location' : np.fromstring(enemy['position']),
                    'location' : np.array(enemy['position'].split(','), dtype=np.float32),
                    'forward' : np.array(enemy['forward'].split(','), dtype=np.float32),
                    'time_seen' : float(phase_cd['phase_ends_in']),
                },
                'health' : int(enemy['state']['health']),
                },
            'agent' : {
                'position' : {
                    'areaId' : None,
                    'location' : np.array(agent['position'].split(','), dtype=np.float32),
                    'forward' : np.array(agent['forward'].split(','), dtype=np.float32),
                },
                'agent_gun' : agent_weapon['name'],
                'agent_bullets' : int(agent_weapon['ammo_clip']),
                'health' : int(agent['state']['health']),
                },
            # 'bomb location' : np.fromstring(bomb['position']),
            'bomb location' : np.array(bomb['position'].split(','), dtype=np.float32),
            'bomb defusing' : 1 if bomb['state'] == 'defusing' else 0,
            'current_time' : int(phase_cd['phase_ends_in']),
            'winner' : match_result
            
        }
    
    def reset(self):
        CSGO_Env_Utils.reset_game()
        CSGO_Env_Utils.start_game()     


if __name__ == '__main__':
    MAP_NAME = 'de_dust2'
    MAP_DATA = NAV_CSV[NAV_CSV["mapName"] == MAP_NAME]
    index = 0
    polygons = []
    vertices = []
    for i in MAP_DATA.index:
        data = MAP_DATA.loc[i]  
        x_range = np.array([float(data["northWestX"]), float(data["southEastX"])])
        y_range = np.array([float(data["northWestY"]), float(data["southEastY"])])
        z_range = np.array([float(data["northWestZ"]), float(data["southEastZ"])])
        v = CSGO_Env_Utils.cartesian_product(x_range, y_range, z_range)
        for point in v:
            point = tuple(point)
            vertices.append(point)
        polygons.append([i for i in range(index, index + 8 )])
        index += 8
    navmesh = pf.PathFinder(vertices, polygons)
    keyboard_controller = keyboard.Controller()
    mouse_controller = mouse.Controller()
    CSGO_Env_Utils.start_game( MAP_NAME, MAP_DATA, navmesh, keyboard_controller, mouse_controller)
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