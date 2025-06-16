import numpy as np
from game import GameBase, GameState, Game
from utils import to_tuple

class TicTacToeBase(GameBase):
    def __init__(self, dim=3, init_board=None):
        self.turn_idx = 0
        self.dim = dim
    
        if init_board is None:
            self.board = np.array([[0]*self.dim]*self.dim)
        else:
            self.board = np.array(init_board)

    def get_next_player(self):
        # 1 always starts
        return 1 if self.turn_idx % 2 == 0 else -1

    def get_legal_moves(self):
        if self.get_winner():
            return []
        else:
            X, Y = np.mgrid[:self.dim, :self.dim]
            unplayed_squares = self.board == 0
            positions = np.array([X[unplayed_squares], Y[unplayed_squares]]).T
            return [tuple(pos) for pos in positions.tolist()]

    def get_winner(self):
        col_sums = np.sum(self.board, axis=0)
        row_sums = np.sum(self.board, axis=1)
        major_sum = np.sum(self.board[np.arange(self.dim), np.arange(self.dim)], keepdims=True)
        minor_sum = np.sum(self.board[np.arange(self.dim), np.arange(self.dim)[::-1]], keepdims=True)
        all_sums = np.concatenate([col_sums, row_sums, major_sum, minor_sum])
        
        if np.any(all_sums == self.dim):
            return 1
        elif np.any(all_sums == -self.dim):
            return -1
        elif np.any(self.board == 0):
            return None
        else:
            return 0

class TicTacToeState(GameState, TicTacToeBase):
    def __init__(self, board, turn_idx, dim):
        ## state
        self.board = board
        self.turn_idx = turn_idx
        self.dim = dim
        
        ## properties
        self.init_properties()

    def __eq__(self, other):
        return to_tuple(self.board.tolist()) == to_tuple(other.board.tolist())

    def __hash__(self):
        return hash(to_tuple(self.board.tolist()))

    def __str__(self):
        row_strs = []
        for row in self.board:
            row_str = "["
            for pos in row:
                pos_str = str(pos)
                if len(pos_str) == 1:
                    pos_str = " "+pos_str
                pos_str = "  " + pos_str
                row_str += pos_str
            row_str += "  ]"
            row_strs.append(row_str)
        return "\n".join(row_strs)

    def export_for_prompt(self):
        return self.board.tolist(), self.next_player

class TicTacToe(Game, TicTacToeBase):
    def move(self, pos):
        if self.get_winner():
            return
        assert self.board[*pos] == 0
        self.board[*pos] = self.get_next_player()
        self.turn_idx += 1
    
    def get_state(self):
        return TicTacToeState(self.board, self.turn_idx, self.dim)