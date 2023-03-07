

from input_data_utils import CSGOImageProcessor
from global_map import SuperNode, NanoNode
from csgo_gsi_python import GSI_SERVER
import gym
from gym import spaces
from gym.spaces import Dict, Sequence, Tuple, Discrete, Box
from global_map import SuperNode,Map
from awpy.data import NAV_CSV

TIME_STEPS = 40
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

class CSGO_Env(gym.Env):
    MAP_NAME = 'de_dust2'
    MAP_DATA = NAV_CSV[NAV_CSV["mapName"] == MAP_NAME]
    SCREEN_HEIGHT = 1080
    SCREEN_WIDTH = 1920
            x_lower_bound = 
            x_upper_bound =
            y_lower_bound = int(data["northWestY"])
            y_upper_bound = int(data["southEastY"])
            z_lower_bound = int(data["northWestZ"])
            z_upper_bound = int(data["southEastZ"])
    #Env is made up of segmented areas of map. Each area is represented by a super node
    def __init__(self):
        
        #observation space can be abstractly thought of as a set of nodes
        #each node encapsulates the following
        self.observation_space = Dict({
            'enemy' : {
                'position' :{
                    'location ' :   #SUPER NODE, Sequence of Tuples of Discrete ranges of x,y,z,p,y,r
                        Sequence([
                            Tuple((
                                spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                                spaces.Discrete(int(data["southEastX"]) - int(data["northWestX"]) + 1, start =int(data["northWestX"])), #x
                                spaces.Discrete(int(data["southEastY"]) - int(data["northWestY"]) + 1, start =int(data["northWestY"])), #y
                                spaces.Discrete(int(data["southEastZ"]) - int(data["northWestZ"]) + 1, start =int(data["northWestZ"])), #z
                                spaces.Discrete(360), #p
                                spaces.Discrete(360), #y
                                spaces.Discrete(1), #r
                            ))
                            for data in CSGO_Env.MAP_DATA
                        ]),
                    'time_seen' : spaces.Discrete(TIME_STEPS),
                },
                'health' : spaces.Discrete(100)
            },
            'agent' :
                {
                'position' : 
                    Sequence([
                        Tuple((
                            spaces.Text(10),  #data['areaId'] map id, 10 char for buffer
                            spaces.Discrete(int(data["southEastX"]) - int(data["northWestX"]) + 1, start =int(data["northWestX"])), #x
                            spaces.Discrete(int(data["southEastY"]) - int(data["northWestY"]) + 1, start =int(data["northWestY"])), #y
                            spaces.Discrete(int(data["southEastZ"]) - int(data["northWestZ"]) + 1, start =int(data["northWestZ"])), #z
                            spaces.Discrete(360), #p
                            spaces.Discrete(360), #y
                            spaces.Discrete(1), #r
                        ))
                        for data in CSGO_Env.MAP_DATA
                    ]),
                'agent_gun' : spaces.Discrete(1), #fixed
                'agent_bullets' : spaces.Discrete(30), #fixed
                'health' :  spaces.Discrete(100),                
            },
            'bomb defusing' : spaces.Discrete(2), #fixed, 0 for not defusing, 1 for defusing
            'current_time' : spaces.Discrete(TIME_STEPS),
        })
        
        self.action_space = Tuple(
            Discrete(8), #0 for walk, 1 for run, 2 for jump, 3 for crouch, 4 for stay, 5 for prone, 6 for shoot, 7 for aim
            Tuple(Discrete(1920), Discrete(1080)), #Mouse movement, controlling mouse location on screen basically
        )
        
    def step(self, action):
        