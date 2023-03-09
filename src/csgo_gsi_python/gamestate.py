# import information
from .information import Player, Map, Provider, PhaseCountdowns, Bomb, Round
class GameState:
    def __init__(self):
        self.player = Player()
        self.map = Map()
        self.provider = Provider()
        self.phase_countdowns = PhaseCountdowns()
        self.bomb = Bomb()
        self.round = Round()


class CompleteGameState(GameState):
    
    def __init__(self):
        #provides {position <position, forward> , {health, gun, bullets}}
        #forward is the direction of movement of the current player
        super().__init__()
        self.allplayers = {}
        
        
        
