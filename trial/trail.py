

import pathfinder as pf
from pathfinder import navmesh_baker as nmb
import numpy as np
from awpy.data import NAV_CSV
import random
from gym.spaces import Box, Discrete
def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


MAP_NAME = 'de_dust2'
MAP_DATA = NAV_CSV[NAV_CSV["mapName"] == MAP_NAME]
vertices = []
polygons = []
index=0
for i in range(len(MAP_DATA)):
    data = MAP_DATA.iloc[i]
    x_range = np.array([float(data["northWestX"]), float(data["southEastX"])])
    y_range = np.array([float(data["northWestY"]), float(data["southEastY"])])
    z_range = None
    #account for similarity in z_range
    if abs(data["northWestZ"] - data["southEastZ"]) <= 0.00001:
        z_range = np.array([float(data["northWestZ"]), float(data["northWestZ"]+0.00001)])
    else:
        z_range = np.array([float(data["northWestZ"]), float(data["southEastZ"])])
    v = cartesian_product(x_range, y_range, z_range)
    num_points = 0
    points_added = 0
    for point in v:
        vertices.append(tuple(point))
        points_added+=1
        
    polygons.append([i for i in range(index,index+points_added)])
    index += points_added
# print(vertices)
# print(polygons)
# # create baker object
baker = nmb.NavmeshBaker()

# # add geometry, for example a simple plane
# # the first array contains vertex positions, the second array contains polygons of the geometry
baker.add_geometry(vertices, polygons)

# # bake navigation mesh
baker.bake()
baker.save_to_text("navmesh_dust2.txt")
baker.save_to_binary("navmesh_dust2")
# # obtain polygonal description of the mesh
vertices, polygons = baker.get_polygonization()
navmesh = pf.PathFinder(vertices, polygons)
vertices,polygons = pf.read_from_text("navmesh_dust2.txt")


bombsite_choice = random.choice(['BombsiteA', 'BombsiteB'])
spawn = MAP_DATA[MAP_DATA['areaName'] == 'TSpawn'].sample()
bombsite = MAP_DATA[MAP_DATA['areaName'] == bombsite_choice].sample()

path = navmesh.search_path(start = tuple([float(spawn['northWestX']), float(spawn['northWestY']), float(spawn['northWestZ'])]),
                            finish = tuple([float(bombsite['northWestX']), float(bombsite['northWestY']), float(bombsite['northWestZ'])]),
                            )
path_np = np.asarray(path)
np.save("path", path_np)
bomb_position = Box(low = np.array([bombsite['northWestX'], bombsite['northWestY']]), 
                    high = np.array([bombsite['southEastX'], bombsite['southEastY']]), 
                    dtype = np.int32).sample()

path.append(np.asarray(bomb_position))
print(path)