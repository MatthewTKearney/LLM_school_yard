import numpy as np
from src.game import GameBase, GameState, Game
from src.utils import to_tuple

class ConnectFourBase(GameBase):
    def __init__(self, ncols=4, nrows=4, nwin=4, init_board=None):
        self.turn_idx = 0
        self.ncols = ncols
        self.nrows = nrows
        self.nwin = nwin
    
        if init_board is None:
            self.board = np.array([[0]*self.nrows]*self.ncols)
        else:
            self.board = np.array(init_board)

        # self.move_properties = self.get_move_properties()

    def similar_to(self, other):
        if not self.turn_idx == other.turn_idx:
            return False

        equivalent_board_states = {to_tuple(self.board.tolist()), to_tuple(self.board[:,::-1].tolist())}
        return to_tuple(other.board.tolist()) in equivalent_board_states
    
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
        else:
            return np.arange(len(self.board))[self.board[0] == 0].tolist()

    def get_winner(self):
        stacked_cols = np.array([self.board[:,idx:self.ncols-self.nwin+idx+1] for idx in range(self.nwin)])
        col_sums = np.sum(stacked_cols, axis=0).flatten()
        
        stacked_rows = np.array([self.board[idx:self.nrows-self.nwin+idx+1, :] for idx in range(self.nwin)])
        row_sums = np.sum(stacked_rows, axis=0).flatten()
        
        stacked_majors = np.array([self.board[idx:self.nrows-self.nwin+idx+1, idx:self.ncols-self.nwin+idx+1] for idx in range(self.nwin)])
        major_sums = np.sum(stacked_majors, axis=0).flatten()

        stacked_minors = np.array([self.board[self.nwin-idx-1:self.nrows-idx, idx:self.ncols-self.nwin+idx+1] for idx in range(self.nwin)])
        minor_sums = np.sum(stacked_minors, axis=0).flatten()

        all_sums = np.concatenate([col_sums, row_sums, major_sums, minor_sums])
        
        if np.any(all_sums == self.nwin):
            return 1
        elif np.any(all_sums == -self.nwin):
            return -1
        elif np.any(self.board == 0):
            return None
        else:
            return 0
        
    def get_game_properties(self):
        return {
            "nrows": self.nrows,
            "ncols": self.ncols,
            "nwin": self.nwin,
        }

class ConnectFourState(GameState, ConnectFourBase):
    def __init__(self, board, turn_idx, nrows, ncols, nwin):
        ## state
        self.board = board
        self.turn_idx = turn_idx
        self.nrows = nrows
        self.ncols = ncols
        self.nwin = nwin
        
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

class ConnectFour(Game, ConnectFourBase):
    def move(self, col):
        if self.get_winner():
            return
        assert self.board[0,col] == 0
        lowest_open_row = np.arange(self.board.shape[0])[self.board[:,col] == 0][-1]
        self.board[lowest_open_row, col] = self.get_next_player()
        self.turn_idx += 1
    
    def get_state(self):
        return ConnectFourState(self.board, self.turn_idx, self.nrows, self.ncols, self.nwin)