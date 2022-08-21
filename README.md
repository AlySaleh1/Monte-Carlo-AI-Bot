# Monte Carlo AI Bot
A game playing agent that competes against an adversarial opponent in a game. The agent uses Monte Carlo Tree Search and Minimax to decide the next best step in the game.

## Game Rules:
This is a turn-based two-player board game. Each player takes a turn by moving a number of steps and placing a wall on the perimeter of a block. 
Restrictions:
- players can not move through walls.
- two players can not be at the same block at the same time.
- the board is 8x8
- the players can between 0 and 4 steps at each  round

#### Game goal:
Throughout the game, the players are trying to create two separate closed zones, with each player in each of those two zones. The player with the laregst number of blocks in their zone wins.

#### Exampels
