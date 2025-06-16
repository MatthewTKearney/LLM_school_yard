import itertools
import numpy as np
from game import GameBase, GameState, Game
from utils import to_tuple

class Sum15Base(GameBase):
    def __init__(self, offset=3, init_board=None):
        self.turn_idx = 0
        self.offset = offset
        self.goal_sum = 15 + 3*self.offset

        magic_square = (np.arange(1,10)+self.offset).tolist()
        
        if init_board is None:
            self.board = [magic_square, [], []] # remaining nums, player 1 nums, player -1 nums
        else:
            nums_chosen = init_board[0] + init_board[1]
            magic_square = [num for num in magic_square if not num in nums_chosen]
            self.board = [magic_square, sorted(init_board[0]), sorted(init_board[1])]

        # self.move_properties = self.get_move_properties()

    def similar_to(self, other):
        return False
    
    # def get_move_properties(self):
    #     center_values = list(range((self.dim-1)//2, self.dim//2+1))
    #     centers = [(x, y) for x in center_values for y in center_values]
    #     corners = [(x, y) for x in [0, self.dim-1] for y in [0, self.dim-1]]
    #     edges = [(x, y) for x in range(0, self.dim-1) for y in [0, self.dim-1]]
    #     edges += [(y, x) for x, y in edges]
    #     positions = set(centers + corners + edges)
    #     interiors = [(x, y) for x in range(self.dim) for y in range(self.dim) if not (x,y) in positions]
    
    #     return {
    #         "move_type": {
    #             **{move: "center" for move in centers},
    #             **{move: "edge" for move in edges},
    #             **{move: "corner" for move in corners},
    #             **{move: "interior" for move in interiors},
    #         }, 
    #     }

    def get_next_player(self):
        # 1 always starts
        return 1 if self.turn_idx % 2 == 0 else -1

    def get_legal_moves(self):
        if self.get_winner():
            return []
        return self.board[0]

    def get_winner(self):
        player1_nums = itertools.combinations(self.board[1], 3)
        for nums in player1_nums:
            if sum(nums) == self.goal_sum:
                return 1

        playerm1_nums = itertools.combinations(self.board[-1], 3)
        for nums in playerm1_nums:
            if sum(nums) == self.goal_sum:
                return -1

        if len(self.board[0]) > 0:
            return None
        return 0
    
    def get_game_properties(self):
        return {
            "goal_sum": self.goal_sum,
            "offset": self.offset,
        }

class Sum15State(GameState, Sum15Base):
    def __init__(self, board, turn_idx, offset, goal_sum):
        ## state
        self.board = board
        self.turn_idx = turn_idx
        self.offset = offset
        self.goal_sum = goal_sum
        
        ## properties
        self.init_properties()

    def __eq__(self, other):
        self_without_offset = [[num-self.offset for num in numset] for numset in self.board]
        other_without_offset = [[num-other.offset for num in numset] for numset in other.board]
        return self_without_offset == other_without_offset

    def __hash__(self):
        return hash(to_tuple(self.board))

    def __str__(self):
        s = f"""
        Numbers Left in Pot: {self.board[0]}
        Player A Numbers: {self.board[1]}
        Player B Numbers: {self.board[2]}
        """.strip()
        return s 

class Sum15(Game, Sum15Base):
    def move(self, num):
        if self.get_winner():
            return
        assert num in self.board[0]
        self.board[0].remove(num)
        player_idx = 1 if self.get_next_player() == 1 else 2
        self.board[player_idx] = sorted(self.board[player_idx] + [num])
        self.turn_idx += 1
    
    def get_state(self):
        return Sum15State(self.board, self.turn_idx, self.offset, self.goal_sum)