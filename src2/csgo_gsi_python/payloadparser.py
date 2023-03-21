# import gamestate
# import gamestate
import json
# import information
from . import gamestate, information
# import gamestate
# import information
class PayloadParser:
    def parse_payload(self, payload, gamestate):
        # print(payload)
        for item in payload:
            if item == 'allplayers':
                
                for i in payload[item]:
                    getattr(gamestate,item)[i] = information.Player()
                    for j in payload[item][i]:
                        try:
                            setattr(
                                getattr(gamestate, item)[i],
                                j,
                                payload[item][i][j]
                            )
                        except:
                            pass
            for i in payload[item]:
                try:
                    setattr(getattr(gamestate, item), i, payload[item][i])
                except:
                    pass


if __name__ == "__main__":
    f = open('data_spectate.json', 'r')
    parser = PayloadParser()
    payload = json.load(f)
    cgs = gamestate.CompleteGameState() 
    data = parser.parse_payload(payload, cgs)
    print(cgs.allplayers['2'].position)