import pandas as pd
from awpy.data import NAV_CSV 
from awpy.parser import DemoParser
import os
import numpy as np
from gym.spaces import Box, Discrete, Tuple
from math import ceil
# NAV_CSV[NAV_CSV["mapName"] == "de_dust2"].head()

# demo_parser = DemoParser(
#     demo_id = "demotest_1", 
#     demofile = "C:\\Users\\beepo\\Desktop\\CSGO_AI\\Models\\Navigation\\demofiles\\demo1.dem",
#     parse_rate=128, 
#     trade_time=5, 
#     buy_style="hltv"
# )

# df = pd.read_json('demotest_1.json',orient='index')
# print(df)
# print(df.shape)
# print(df.rows)
# print(bool(df['gameRounds']))

# class NanoNode():

#     def __init__(self, x, y, z, pitch, yaw, roll):

#         self.x = x

#         self.y = y

#         self.z = z
        
#         self.pitch = pitch

#         self.yaw = yaw

#         self.roll = roll

#     def set_pitch_yaw_roll(self, pitch, yaw, roll):
#         self.pitch = pitch
#         self.yaw = yaw
#         self.roll = roll
    

# class SuperNode:
#     def __init__(self, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, z_lower_bound, z_upper_bound):
#         self.x_lower_bound = x_lower_bound #first axis
#         self.x_upper_bound = x_upper_bound #first axis
#         self.y_lower_bound = y_lower_bound 
#         self.y_upper_bound = y_upper_bound
#         self.z_lower_bound = z_lower_bound
#         self.z_upper_bound = z_upper_bound
#         self.PITCH_BOUNDS = (-90,90)
#         self.YAW_BOUNDS = (-180,180)
#         # self.ROLL_BOUNDS = (-90,90) #sixth axis
#         self.ROLL_BOUNDS = (0,0) #sixth axis
#         self.nodes = None #will be a six axis array
#         self.neighbors = []
#         self._init_nanonodes()
    
#     #This function is used to initialize the nanonodes within the super node
#     def _init_nanonodes(self):
#         nodes = []
#         # 20 is the step size
#         #we add 20 to the upper bound so that we cover the entire map
#         #we will later round them up to the nearest 20 when we get actual data
#         POS_OFFSET = 20
#         ANGLE_OFFSET = 45
#         for x in range(self.x_lower_bound, self.x_upper_bound + POS_OFFSET,POS_OFFSET):
#             for y in range(self.y_lower_bound, self.y_upper_bound + POS_OFFSET,POS_OFFSET):
#                 for z in range(self.z_lower_bound, self.z_upper_bound + POS_OFFSET,POS_OFFSET):
#                     #add one here to make sure we cover the entire orientation space
#                     for pitch in range(self.PITCH_BOUNDS[0], self.PITCH_BOUNDS[1] + 1,ANGLE_OFFSET):
#                         for yaw in range(self.YAW_BOUNDS[0], self.YAW_BOUNDS[1] + 1,ANGLE_OFFSET):
#                             for roll in range(self.ROLL_BOUNDS[0], self.ROLL_BOUNDS[1] + 1,ANGLE_OFFSET):
#                                 nodes.append(NanoNode(x,y,z,pitch,yaw,roll))
#         self.nodes = np.asarray(nodes)
#         return True

#     def add_neighbor(self, neighbor):
#         self.neighbors.append(neighbor)

#represents a location of an area in the map
class SuperNode:
    
    POS_OFFSET = 20
    ANGLE_OFFSET = 45
    
    def __init__(self, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, z_lower_bound, z_upper_bound):
        self.x_lower_bound = x_lower_bound #first axis
        self.x_upper_bound = x_upper_bound #first axis
        self.y_lower_bound = y_lower_bound 
        self.y_upper_bound = y_upper_bound
        self.z_lower_bound = z_lower_bound
        self.z_upper_bound = z_upper_bound
        self.PITCH_BOUNDS = (-90,90)
        self.YAW_BOUNDS = (-180,180)
        # self.ROLL_BOUNDS = (-90,90) #sixth axis
        self.ROLL_BOUNDS = (0,0) #sixth axis
        self.neighbors = []
        self.space = None
        self.init_space()
    #This function is used to initialize the nanonodes within the super node
    def init_space(self):
        shape = (
            (self.x_upper_bound - self.x_lower_bound) // self.POS_OFFSET + 1,
            (self.y_upper_bound - self.y_lower_bound) // self.POS_OFFSET + 1,
            (self.z_upper_bound - self.z_lower_bound) // self.POS_OFFSET + 1,
            180 // self.ANGLE_OFFSET,
            360 // self.ANGLE_OFFSET,
            1,
        )
        self.space = Tuple(
            Discrete(shape[0]),Discrete(shape[1]),Discrete(shape[2]),Discrete(shape[3]),Discrete(shape[4]),Discrete(shape[5]),
        )
        
        return True
    def get_space(self):
        return self.space
    
    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        
class Map:

    def __init__(self, map_name):
        self.map_name = map_name
        self.super_nodes = None
        self.FILEPATH_TO_MAP = "C:\\Users\\beepo\\Desktop\\CSGO_AI\\Models\\Navigation\\maps_files\\"
        self.make_map(False)
    
    def make_map(self,save=True):
        #see if map exists
        if not os.path.exists(self.FILEPATH_TO_MAP + self.map_name + ".npy"):
            print("Map not found, making new map")
            self.super_nodes = self._make_super_nodes(save)
        else:
            print("Map found, loading map")
            self.super_nodes = np.load(self.FILEPATH_TO_MAP + self.map_name + ".npy", allow_pickle=True)
            return True

    def _make_super_nodes(self, SAVE_MAP=True):
        #get the map bounds
        map_bounds = NAV_CSV[NAV_CSV["mapName"] == self.map_name]
        number_of_data, number_of_features = map_bounds.shape
        super_nodes = []
        #EG format of data 
        # mapName        de_dust2
        # areaId             3765
        # areaName      BombsiteB
        # northWestX        -2050
        # northWestY         1950
        # northWestZ      1.63846
        # southEastX        -2025
        # southEastY         2025
        # southEastZ     0.163017
        for i in range(number_of_data):
            data = map_bounds.iloc[i]
            x_lower_bound = int(data["northWestX"])
            x_upper_bound = int(data["southEastX"])
            y_lower_bound = int(data["northWestY"])
            y_upper_bound = int(data["southEastY"])
            z_lower_bound = int(data["northWestZ"])
            z_upper_bound = int(data["southEastZ"])
            super_nodes.append(SuperNode(x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, z_lower_bound, z_upper_bound))
        print('done making super nodes')
        #make super nodes
        self.super_nodes = np.asarray(super_nodes)
        if SAVE_MAP:
            np.save(self.FILEPATH_TO_MAP + self.map_name + ".npy", self.super_nodes)
            pd.DataFrame(self.super_nodes).to_csv(self.map_name + ".csv")
        return True

if __name__ == "__main__":
    map = Map("de_dust2")
    print(map.super_nodes[0].nodes[0].x)