# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import time
from copy import deepcopy
import random
import numpy as np


@register_agent("pure_MCS")
class pure_MCS(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(pure_MCS, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.step_counter = 0
        self.tree = None

    @staticmethod
    def generate_all_possible_moves(chess_board, my_pos, adv_pos, max_step):
        # returns a winning move if possible (in a list of one element)
        # returns a list of continuing moves when there is no winning move
        # returns a list of game-ending moves that leads to a draw
        # if losing no matter what, error, but losing anyways so whatever
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        list_of_moves = []
        start_pos = deepcopy(my_pos)

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        # is_reached = False
        # while state_queue and not is_reached:
        while state_queue:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step > max_step:
                break
            for dir, move in enumerate(moves):
                # if there is a wall
                if chess_board[r, c, dir]:
                    continue
                # if there is no wall, then we can add this move to the list
                list_of_moves.append((cur_pos, dir))
                next_pos = tuple(map(sum, zip(cur_pos, move)))
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))
        return list_of_moves

    @staticmethod
    def check_endgame(chess_board, my_pos, adv_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        board_size = len(chess_board[0])
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                        moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score
        return True, p0_score, p1_score

    @staticmethod
    def is_valid_step(start_pos, end_pos, adv_pos, barrier_dir, chess_board, max_step):
        """
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False

        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))

        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    @staticmethod
    def set_barrier(r, c, dir, chess_board):
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = True

    @staticmethod
    def get_good_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        3 -> winning move
        2 -> list of not game-ending moves
        1 -> list of draw moves
        0 -> a losing move to avoid error
        """
        list_of_moves = self.generate_all_possible_moves(chess_board, my_pos, adv_pos, max_step)
        list_to_return = []
        draw_list = []  # if there is (1)no winning move and (2)only losing or draw moves, return draw moves

        for move in list_of_moves:
            copy_chess_board = deepcopy(chess_board)
            self.set_barrier(move[0][0], move[0][1], move[1], copy_chess_board)
            end, score1, score2 = self.check_endgame(copy_chess_board, move[0], adv_pos)
            if end:
                if score1 > score2:
                    return ([move], 3)  # winning move, immediately return it
                elif score1 == score2:
                    draw_list.append(move)
            else:
                list_to_return.append(move)
        if not list_to_return:  # if there is no move winning or continuing, only losing move, draw the game
            if not draw_list:  # if there is nothing left except a losing move, return a losing move
                return (list_of_moves[0], 0)
            return (draw_list, 1)  # else, return a move that draws the game
        return (list_to_return, 2)  # False is for the game not ending

    @staticmethod
    def get_second_good_moves(self, good_moves, chess_board, my_pos, adv_pos, max_step):
        continuing_moves = []
        for move in good_moves:
            copy_chess_board = deepcopy(chess_board)
            self.set_barrier(move[0][0], move[0][1], move[1], copy_chess_board)
            my_pos = move[0]
            game_ending_b = self.get_good_moves(self, copy_chess_board, adv_pos, my_pos, max_step)[1]
            if game_ending_b == 3: # if player B can win after this move, remove this move
                continue
            elif game_ending_b == 0: # if player B will unavoidably lose after this move, return this move
                return [move]
            else:
                continuing_moves.append(move)
        if not continuing_moves:
            return good_moves
        else:
            return continuing_moves


    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """

        self.step_counter += 1
        start_time = time.time()

        good_moves, game_ending = self.get_good_moves(self, chess_board, my_pos, adv_pos, max_step)
        if game_ending in (0,1,3): # if game is inevitably ending, return it immediately: 0 is losing, 1 is draw, 3 is winning
            return random.choice(good_moves)
        elif max_step < 4: # if game is not ending
            good_moves = self.get_second_good_moves(self, good_moves, chess_board, my_pos, adv_pos, max_step)
        if self.step_counter == 1:
            return self.pure_MCS_move(good_moves, chess_board, adv_pos, start_time, max_step, 29.9)
        else:
            return self.pure_MCS_move(good_moves, chess_board, adv_pos, start_time, max_step,1.9)

    '''
    input:
        available_moves: list of good moves that will not trap you
        returns: the best move according to pure MCS
    '''

    def pure_MCS_move(self, available_moves, chessboard, adv_pos, start_time, max_step, max_time):
        possible_states = []

        for move in available_moves:
            new_chessboard = deepcopy(chessboard)
            self.apply_step(new_chessboard, move)
            possible_states.append(MCTSnode(new_chessboard, (move[0][0], move[0][1]), adv_pos, move_from_parent=move))
        i = 0

        while time.time() - start_time < max_time:
            if i == len(possible_states):
                i = 0
            self.random_simulations(possible_states[i], max_step, start_time) ##added start time here
            i += 1

        max_win_node = possible_states[0]
        max_win_rate = self.win_rate(possible_states[0],-1)
        for i in range(len(possible_states)):
            if self.win_rate(possible_states[i],i, len(possible_states)) > max_win_rate:
                max_win_node = possible_states[i]
                max_win_rate = self.win_rate(possible_states[i],-2)
        return max_win_node.move_from_parent


    def win_rate(self, node,num, list_len=None):
        if node.games_played_amount == 0:
            return 0
        return (node.win_amount / node.games_played_amount)

    def apply_step(self, chessboard, move):
        self.set_barrier(move[0][0], move[0][1], move[1], chessboard)

    '''
    input:
        node to have the simulation run on
    post conditions:
        node's win rate is updated
    '''

    def random_simulations(self, node, max_step, start_time):
        win_number = 0
        game_number = 6

        for _ in range(game_number):  # 2 simulations
            my_turn = True
            my_new_pos = deepcopy(node.my_pos)
            adv_new_pos = deepcopy(node.adv_pos)
            chess_board_copy = deepcopy(node.chess_board_state)
            endgame = False, 0, 0  # endgame = (did the game end, my score, adv score)
            while not endgame[0]:
                if my_turn:
                    my_random_move = random_step(chess_board_copy, my_new_pos, adv_new_pos, max_step)
                    self.apply_step(chess_board_copy, my_random_move)
                    my_new_pos = my_random_move[0][0], my_random_move[0][1]
                    my_turn = False
                else:
                    adv_random_move = random_step(chess_board_copy, adv_new_pos, my_new_pos, max_step)
                    self.apply_step(chess_board_copy, adv_random_move)
                    adv_new_pos = adv_random_move[0][0], adv_random_move[0][1]
                    my_turn = True
                endgame = self.check_endgame(chess_board_copy, my_new_pos, adv_new_pos)
            if endgame[1] > endgame[2]:
                win_number += 1

        node.win_amount += win_number
        node.games_played_amount += game_number


def random_step(chess_board, my_pos, adv_pos, max_step):
    # Moves (Up, Right, Down, Left)
    ori_pos = deepcopy(my_pos)
    moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
    steps = np.random.randint(0, max_step + 1)

    # Random Walk
    for _ in range(steps):
        r, c = my_pos
        dir = np.random.randint(0, 4)
        m_r, m_c = moves[dir]
        my_pos = (r + m_r, c + m_c)

        # Special Case enclosed by Adversary
        k = 0
        while chess_board[r, c, dir] or my_pos == adv_pos:
            k += 1
            if k > 300:
                break
            dir = np.random.randint(0, 4)
            m_r, m_c = moves[dir]
            my_pos = (r + m_r, c + m_c)

        if k > 300:
            my_pos = ori_pos
            break

    # Put Barrier
    dir = np.random.randint(0, 4)
    r, c = my_pos
    while chess_board[r, c, dir]:
        dir = np.random.randint(0, 4)

    return my_pos, dir


class MCTSnode():
    def __init__(self, chess_board_state, my_pos, adv_pos, move_from_parent=None):
        self.chess_board_state = deepcopy(chess_board_state)
        self.move_from_parent = move_from_parent
        self.my_pos = my_pos
        self.adv_pos = adv_pos
        self.win_amount = 0
        self.games_played_amount = 0

    def won_MCTS_simulation(self):
        self.win_amount += 1
        self.games_played_amount += 1


    def lost_MCTS_simulation(self):
        self.games_played_amount += 1

    def add_child_node(self, child_node):
        self.child.append(child_node)
