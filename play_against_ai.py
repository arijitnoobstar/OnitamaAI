from play_onitama import *
import sys

if len(sys.argv)< 2:
    depth = 5
else:
    depth = sys.argv[1]

game_mode = "minimax v p"
play_onitama(game_mode, verbose = 1, minimax_depth = depth)