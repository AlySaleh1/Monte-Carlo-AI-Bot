# Monte Carlo AI Bot
A game playing agent that competes against an adversarial opponent in a game. The agent uses Monte Carlo Tree Search and Minimax to decide the next best step in the game.

## Game Rules:
This is a turn-based two-player board game. Each player takes a turn by moving a number of steps and placing a wall on the perimeter of a block. 
Restrictions:
- players can not move through walls.
- two players can not be at the same block at the same time.
- the board is 8x8.
- the players can between 0 and 4 steps at each  round.

#### Game goal:
Throughout the game, the players are trying to create two separate closed zones, with each player in each of those two zones. The player with the largest number of blocks in their zone wins.


#### Examples:
Here is a situation where player A wins the game. Note that the number of blocks in player A's zone (20 blocks) is greater than the number of blocks in player B's zone.
[image](https://user-images.githubusercontent.com/78103711/185810666-f98ad90a-f542-4266-b76a-639a6f577e94.png)

#### Contributors:
- I worked on implementing the Monte Carlo Search and the Minimax algorithm.
- (Zhihao Huang)[https://github.com/zhihao2828] worked on the heuristics of the game and the Minimax algorithm. 
- The orginal code for the game can be found [here](https://github.com/comp424mcgill/Project-COMP424-2022-Winter).
