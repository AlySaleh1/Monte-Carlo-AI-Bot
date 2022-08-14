# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import time
from copy import deepcopy
import logging
import random
import numpy as np


@register_agent("student_agent2")
class StudentAgent2(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    step_counter = 0
    def __init__(self):
        super(StudentAgent2, self).__init__()
        self.name = "student_agent2"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

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
                next_pos= tuple(map(sum, zip(cur_pos, move)))
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
        list_of_moves = self.generate_all_possible_moves(chess_board, my_pos, adv_pos, max_step)
        list_to_return = []
        draw_list = [] # if there is (1)no winning move and (2)only losing or draw moves, return draw moves

        for move in list_of_moves:
            copy_chess_board = deepcopy(chess_board)
            self.set_barrier(move[0][0], move[0][1], move[1], copy_chess_board)
            end, score1, score2 = self.check_endgame(copy_chess_board, move[0], adv_pos)
            if end:
                if score1 > score2:
                    return ([move], True)
                elif score1 == score2:
                    draw_list.append(move)
            else:
                list_to_return.append(move)
        if not list_to_return: # if there is no move winning or continuing, only losing move, draw the game
            if not draw_list: # if there is nothing left except a losing move, return a losing move
                return (list_of_moves[0], True)
            return (draw_list, True) # else, return a move that draws the game
        return (list_to_return, False)


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

        self.step_counter+=1
        start_time = time.time()



        good_moves, result_type = self.get_good_moves(self, chess_board, my_pos, adv_pos, max_step)
        return random.choice(good_moves)
