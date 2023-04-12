from pynput import mouse, keyboard
import socket
import json
from pynput.keyboard import Key
from pynput.mouse import Button
import numpy as np
import time
from gym.spaces import  Box
from awpy.data import NAV_CSV
import re
import random
from enemy_detector_server import ENEMY_SCREEN_DETECTOR
import select
class GameServer:
    def __init__(self, action_time= 0.1):
        self.keyboard_controller = keyboard.Controller()
        self.mouse_controller = mouse.Controller()
        self.ACTION_TIME = action_time
        # self.host = '192.168.1.241'
        # self.port = 5000
        # self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # self.client =('192.168.1.109', 5005)
        MAP_NAME = 'de_dust2'
        self.map_data = NAV_CSV[NAV_CSV["mapName"] == MAP_NAME]
        # self.socket.bind((self.host, self.port))


    def start_server(self):
        while True:
            self.get_action()
    
    def get_action(self, s, client):
        ready = select.select([s], [], [], 0.5)
        # print(ready)
        if ready[0]:
            data, addr = s.recvfrom(1024)
            data = data.decode('utf-8')
            data = json.loads(data)
            done = bool(data['done'])
            action = data['action']
            if action == 'configure':
                self.configure_game()
            elif action.startswith("start"):
                words = action.split()
                print(words)
                self.start_game(words[1])
            elif action.startswith('pause'):
                self.pause_game(client, s)
            elif action.startswith("restart"):
                words = action.split()
                self.start_game(words[1])
            elif action == 'endround':
                self.endround()
            elif action is None:
                pass
            else:
                tmp = data['action'].split(',')
                action = tmp[:-2]
                coord =  tmp[-2:]
                action = [int(i) for i in action]

                if coord[0] != 'None':
                    action.append(int(float(coord[0])))
                    action.append(int(float(coord[1])))
                else:
                    action.append(None)
                    action.append(None)
                if not done:
                    self._apply_action(action)
                    
            print('action applied')
        
        response = "done"
        s.sendto(response.encode('utf-8'), client)

    def pause_game(self, client, s):
        self.reset_controllers()
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')
        self.csgo_type_command(self.keyboard_controller, 'endround')
        self.csgo_type_command(self.keyboard_controller, 'bot_stop', '1')
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')

    def endround(self):
        self.reset_controllers()
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')
        
        self.csgo_type_command(self.keyboard_controller ,'endround')
        
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')
    
    def _apply_action(self, action):
        shift_pressed = True if action[1] == 1 else False
        ctrl_pressed = True if action[2] == 1 else False
        spacebar_pressed = True if action[3] == 1 else False
        movement_button = None
        left_click = True if action[10] == 1 else False
        # enemy_screen_coords = self._obs['enemy_coords_on_screen']
        cursor_location = (action[11], action[12])
        print('cursor location', cursor_location)
        if left_click:
            #if there is a target to aim at
            if cursor_location[0] is not None and cursor_location[1] is not None:
                self.mouse_controller.position = cursor_location
                # curr_cursor_position = self.mouse_controller.position
                # self.mouse_controller.move(
                #     cursor_location[0] - curr_cursor_position[0], cursor_location[1] - curr_cursor_position[1])
            self.mouse_controller.click(Button.left,1)
            # self.mouse_controller.release(Button.left)

        #Action to control the mouse
        # if action[6] == 0:
        #     if action[7] == 0:
        #         self.mouse_controller.move(1, 0)
        #     elif action[7] == 1:
        #         self.mouse_controller.move(-1, 0)
        #     # elif action[7] == 0 and action[8] == 1:
        #     #     self.mouse_controller.move(1, 0)
        #     # elif action[7] == 1 and action[8] == 1:
        #     #     self.mouse_controller.move(-1, 0)
        
        if action[7] == 1 and action[6] == 0:
            self.mouse_controller.move(-100, 0)
        if action[6] == 1 and action[7] == 0:
            self.mouse_controller.move(100, 0)
        if action[9] == 1 and action[8] == 0:
            self.mouse_controller.move(0, 50)
        if action[8] == 1 and action[9] == 0:
            self.mouse_controller.move(0, -50)    

        # we only set movement action if action[0] == 0
        # this is so as we prevent any keyboard-related inputs when action[0] == 1
        if action[0] == 0:
            if action[4] == 0 and action[5] == 0:
                movement_button = 'w'
            elif action[4] == 1 and action[5] == 0:
                movement_button = 'a'
            elif action[4] == 0 and action[5] == 1:
                movement_button = 's'
            elif action[4] == 1 and action[5] == 1:
                movement_button = 'd'

        #keyboard action allowed is implied by movement_button not being None
        if movement_button is not None:
            # list_of_keys = ['w', 'a', 's', 'd'] - [movement_button]
            list_of_keys = ['w', 'a', 's', 'd']
            list_of_keys.remove(movement_button)
            if len(list_of_keys) > 0:
                for key in list_of_keys:
                    self.keyboard_controller.release(key)
            # self.keyboard_controller.release(*list_of_keys)
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
                self.keyboard_controller.release(Key.space)
                self.keyboard_controller.release(Key.ctrl)
                with self.keyboard_controller.pressed(Key.shift):
                    self.keyboard_controller.press(movement_button)
            elif ctrl_pressed:
                self.keyboard_controller.release(Key.space)
                self.keyboard_controller.release(Key.shift)
                with self.keyboard_controller.pressed(Key.ctrl):
                    self.keyboard_controller.press(movement_button)
            elif spacebar_pressed:
                self.keyboard_controller.release(Key.ctrl)
                self.keyboard_controller.release(Key.shift)
                with self.keyboard_controller.pressed(Key.space):
                    self.keyboard_controller.press(movement_button)
            else:
                list_of_keys = ['w', 'a', 's', 'd'].remove(movement_button)
                if list_of_keys is not None:
                    for key in list_of_keys:
                        self.keyboard_controller.release(key)
                self.keyboard_controller.release(Key.shift)
                self.keyboard_controller.release(Key.ctrl)
                self.keyboard_controller.release(Key.space)
                # list_of_keys = ['w', 'a', 's', 'd'] - [movement_button]
                # self.keyboard_controller.release(
                    # Key.shift, Key.ctrl, Key.space, *list_of_keys)
                self.keyboard_controller.press(movement_button)

        #keyboard action not allowed is implied by movement_button being None
        else:
            list_of_keys = ['w', 'a', 's', 'd']
            for key in list_of_keys:
                self.keyboard_controller.release(key)
            # self.keyboard_controller.release(*list_of_keys)
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



        # sleep to run through the timed thread
        # time.sleep(self.ACTION_TIME)
    # TODO: Change datatype
    # DONE, TODO: Check
    
    def configure_game(self):
        self.reset_controllers()
        # open terminal
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')
        # configure game settings
        self.csgo_type_command(
            self.keyboard_controller, 'sv_cheats', '1')  # allow cheats
        
        #ignore win condition so game round does not end, this is crucial so that we can keep training for more than 36 rounds
        #without this, the game will end after 36 rounds and we will have to restart the game, which is not optimal
        self.csgo_type_command(
            self.keyboard_controller, 'mp_ignore_round_win_conditions', '1'
        )
        
        # dont allow grenades or any utilities
        self.csgo_type_command(
            self.keyboard_controller, 'mp_buy_allow_grenades', '0')
        self.csgo_type_command(
            self.keyboard_controller, 'mp_c4timer', '40')  # Set bomb explode timer
        self.csgo_type_command(
            self.keyboard_controller, 'mp_autokick', '0')
        # Set CT default primary weapon
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_ct_default_primary', 'weapon_m4a1')
        # # Set CT default secondary weapon
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_ct_default_secondary', 'weapon_usp_silencer')
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_t_default_primary', 'weapon_ak47')  # Set T default primary weapon
        # # Set T default secondary weapon
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_t_default_secondary', 'weapon_glock')

        # # setting what weapons to can be used, for which team
        # # -1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_weapons_allow_heavy', '0')
        # # -1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_weapons_allow_pistols', '-1')
        # # -1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_weapons_allow_rifles', '-1')
        # # -1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_weapons_allow_smgs', '0')
        # # -1 : all allow, 0: none allow, 2: only T allow, 3: only CT allow
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_weapons_allow_snipers', '0')

        # ff_damage_bullet_penetration 0/1 to allow bullet penetration

        # interesting command mp_weapon_self_inflict_amount [_], 0 for no self inflict damage, 1 for self inflict damage for mimssing shot
        # intersting command mp_plant_c4_anywhere to plant anywhere
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_give_player_c4', '1')  # Give T bomb
        # # dont switch team in half time
        # self.csgo_type_command(
        #     self.keyboard_controller, 'mp_halftime', '0')

        # sound distance
        # play_sound_distance = 1000
        # self.csgo_type_command(self.keyboard_controller, 'play_distance', f'{play_sound_distance}') #dont show sound
        # interesting command -- playgmaesound [Sound] can use to train sound model
        # sound_device_list to list all sound device

        # radarvisdistance [Distance

        # bots
        # self.csgo_type_command(self.keyboard_controller, 'bot_goto_selected', '0') #navigation
        # self.csgo_type_command(self.keyboard_controller, 'bot_goto_mark', '0') #navigation

        # dont allow grenades or any utilities
        self.csgo_type_command(
            self.keyboard_controller, 'bot_allow_grenades', '0')
        # dont allow grenades or any utilities
        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_allow_machine_guns', '0')
        # # dont allow grenades or any utilities
        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_allow_pistols', '1')
        # # dont allow grenades or any utilities
        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_allow_rifles', '1')
        # # dont allow grenades or any utilities
        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_allow_snipers', '0')
        # # dont allow grenades or any utilities
        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_allow_shotguns', '0')
        # # dont allow grenades or any utilities
        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_allow_sub_machine_guns', '0')
        # dont allow grenades or any utilities
        # self.csgo_type_command(
            # self.keyboard_controller, 'bot_allow_rogues', '0')

        # self.csgo_type_command(self.keyboard_controller, 'notarget') # bot ignores player
        # self.csgo_type_command(self.keyboard_controller, 'mp_random_spawn', '3') # random spawn for enemy bot, not agent
        # self.csgo_type_command(self.keyboard_controller, 'mp_random_spawn_los', '1') # random spawn for enemy bot to ensure that enemy bot not in sight of agent
        # self.csgo_type_command(self.keyboard_controller, 'bot_max_vision_distance_override', '1', '30') # random spawn for enemy bot, not agent
        # [0:4] 0 :EASIEST, 1: EASY, 2: NORMAL, 3: HARD, 4: HARDEST
        # self.csgo_type_command(
        #     self.keyboard_controller, 'custom_bot_difficulty', '2')
        # # 1 [0:4] 0 :EASIEST, 1: EASY, 2: NORMAL, 3: HARD, 4: HARDEST
        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_difficulty', '2')

        # self.csgo_type_command(
        #     self.keyboard_controller, 'bot_add_ct', 'normal')  # number of bot

        # close terminal
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')
        # self.socket.send("done".encode('utf-8'), self.server)

    def _get_bombsites_points(self, bombsite_choice):
        if bombsite_choice == 'BombsiteA':
            return[
            [-1366.031250, 2565.968750, 68.707779],
            [-1470.027832, 2565.968750, 69.762558],
            [-1554.365112, 2620.665771, 68.789566],
            [-1643.440796, 2678.800293, 72.187271],
            [-1648.400513, 2810.234863, 81.354736],
            [-1480.651001, 2782.294434, 73.810066],
            [-1423.031250, 2743.738281, 81.363113],
            [-1574.565796, 2670.532715, 69.824219],
            ]
        else:
            return[
                [987.968750 , 2444.031250 , 160.964554,], 
                [987.968750 , 2545.016846 , 160.028427,],   
                [1150.484863,  2550.376953,  160.254578,], 
                [1091.842529,  2390.350098,  161.772141,], 
                [1235.968750,  2435.178223,  162.825745,], 
                [1232.514893,  2348.031250,  163.591339,],              
                [1235.995605,  2582.184326,  163.423294,], 
            ]
    # server):
    def start_game(self, bombsite_choice):
        self.reset_controllers()
        # choose bombsite
        bombsite = self.map_data[self.map_data['areaName'] == bombsite_choice].sample()
        # enemy_spawn = self.map_data[self.map_data['areaName']
        #                        != bombsite_choice].sample()

        # bomb_position = Box(low=np.array([bombsite['northWestX'], bombsite['northWestY'], bombsite['northWestZ']]),
        #                     high=np.array(
        #                         [bombsite['southEastX'], bombsite['southEastY'], bombsite['southEastZ']]),
        #                     dtype=np.float32).sample()
        bomb_position = random.choice(self._get_bombsites_points(bombsite_choice))
        # enemy_position = Box(low=np.array([enemy_spawn['northWestX'], enemy_spawn['northWestY'], enemy_spawn['northWestZ']]),
        #                      high=np.array(
        #                          [enemy_spawn['southEastX'], enemy_spawn['southEastY'], enemy_spawn['southEastZ']]),
        #                      dtype=np.float32).sample()

        # open terminal
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')

        # first give the player the bomb
        # self.csgo_type_command(
            # self.keyboard_controller, 'mp_give_player_c4', '1')  # Give T bomb

        # then we spawn the enemy, but first we freeze bot first
        self.csgo_type_command(
            self.keyboard_controller, 'endround')  # 1 bot
        self.csgo_type_command(
            self.keyboard_controller, 'bot_stop', '1')  # 1 bot
        # self.csgo_type_command(
        #     self.self.keyboard_controller, 'setpos', f'{enemy_position[0]}', f'{enemy_position[1]}', f'{enemy_position[2]}')  # 1 bot
        # # check if player stuck

        # self.csgo_type_command(self.keyboard_controller, 'bot_place')

        # then we go to the bombsite
        self.csgo_type_command(
            self.keyboard_controller, 'setpos', f'{bomb_position[0]}', f'{bomb_position[1]}', f'{bomb_position[2]}')  # 1 bot
        # check if player stuck if yes noclip to unstuck

        # ensure that the player is at the bomb site
        # player_info = server.get_info("player")
        # player_info = client.get_info("player")

        # print('player info : ', player_info)
        # curr_loc = np.array(
        #     player_info['position'].split(','), dtype=np.float32)

        # # curr_loc = np.fromstring(player_info['location'], dtype = np.int32, sep = ',')
        # assert (
        #     curr_loc[0] >= bombsite['northWestX'] and curr_loc[0] <= bombsite['southEastX'] and
        #     curr_loc[1] >= bombsite['northWestY'] and curr_loc[1] <= bombsite['southEastY'] and
        #     curr_loc[2] >= bombsite['northWestZ'] and curr_loc[2] <= bombsite['southEastZ']
        # )

        #close terminal
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')

        # plant bomb

        # player switch to bomb
        self.keyboard_controller.press('5')
        self.keyboard_controller.release('5')

        # initialise bomb plant
        self.mouse_controller.press(Button.left)
        time.sleep(5)
        self.mouse_controller.release(Button.left)


        #open terminal
        self.keyboard_controller.press('`')
        time.sleep(0.1)
        self.keyboard_controller.release('`')

        self.csgo_type_command(self.keyboard_controller, 'bot_stop', '0')
            # # round_status = server.get_info('round')
            # round_status = client.get_info('round')
            # while 'bomb' not in round_status.keys():
            #     # round_status = server.get_info('round')
            #     round_status = client.get_info('round')
            #     if 'bomb' in round_status.keys():
            #         if round_status['bomb'] == 'planted':
            #             # unfreeze bot and start the game
            #             self.csgo_type_command(
            #                 self.keyboard_controller, 'bot_stop', '0')  # 1 bot
                        # break

        print('bomb planted')
        # close terminal
        self.keyboard_controller.press('`')
        time.sleep(0.1) 
        self.keyboard_controller.release('`')
        # self.socket.send("done".encode('utf-8'), self.server)

    def csgo_type_command(self, _ ,command, *args):

        for char in command:
            self.keyboard_controller.press(char)
            self.keyboard_controller.release(char)

        for i in range(len(args)):
            self.keyboard_controller.press(Key.space)
            self.keyboard_controller.release(Key.space)
            for char in args[i]:
                self.keyboard_controller.press(char)
                self.keyboard_controller.release(char)

        # send command
        self.keyboard_controller.press(Key.enter)
        self.keyboard_controller.release(Key.enter)
        time.sleep(0.5)
        print('command sent')

    def reset_controllers(self):
        if self.keyboard_controller.shift_pressed:
            self.keyboard_controller.release(Key.shift)
        if self.keyboard_controller.ctrl_pressed:
            self.keyboard_controller.release(Key.ctrl)
        # if self.mouse_controller.left_pressed:
        #     self.mouse_controller.release(Button.left)
        # if self.mouse_controller.right_pressed:
        #     self.mouse_controller.release(Button.right)


if __name__ == '__main__':
    pass

