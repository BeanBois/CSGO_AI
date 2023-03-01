import pandas as pd

from awpy.parser import DemoParser

demo_parser = DemoParser(
    demofile = "C:\\Users\\beepo\\Desktop\\CSGO_AI\\Models\\Navigation\\demofiles\\demo1.dem",
    demo_id = "demotest_1", 
    parse_rate=128, 
    trade_time=5, 
    buy_style="hltv"
)
df = pd.read_json('demotest_1.json',orient='index')
print(df)
print(df.shape)
print(df.rows)
print(bool(df['gameRounds']))
class SuperNode():

    def __init__(self, x, y, z):

        self.x = x

        self.y = y

        self.z = z

        self.neighbors = []

        self.visited = False

        self.parent = None

        self.cost = 0

        self.heuristic = 0

        self.f = 0