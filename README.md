# CSGO_AI

# The project/agent is train using 3 computers to compensate for the lack of computational power in 1 computer. The 3 computers are:
#   'Player': the computer playing the game. This involves streaming processed image data to 'Agent' and receiving actions to play from 'Agent' (src1)
#   'Spectator': the computer spectating the game for information only available to the spectators. {EG enemy location} (src2)
#   'Agent': the computer that is used to train the model. (src)
