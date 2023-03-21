
import information
# from . import information

class GameState:
    def __init__(self):
        self.player = information.Player()
        self.map = information.Map()
        self.provider = information.Provider()
        self.phase_countdowns = information.PhaseCountdowns()
        self.bomb = information.Bomb()
        self.round = information.Round()


class CompleteGameState(GameState):
    
    def __init__(self):
        #provides {position <position, forward> , {health, gun, bullets}}
        #forward is the direction of movement of the current player
        super().__init__()
        self.allplayers = {}
        
        
        
