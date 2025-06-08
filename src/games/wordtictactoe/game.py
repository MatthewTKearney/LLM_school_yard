import itertools
import numpy as np
from src.game import GameBase, GameState, Game
from src.utils import to_tuple

import itertools
class WordTicTacToeBase(GameBase):
    def __init__(self, init_board=None):
        self.turn_idx = 0
        
        # words = {"eat", "bee", "less", "air", "bits", "lip", "soda", "book", "lot"}
        self.words = sorted(["pout", "toga", "oil", "sue", "apple", "dice", "null", "and", "pin"])
        self.winning_combinations = []
        for word1, word2, word3 in itertools.combinations(self.words, 3):
            overlapping_letters = set(list(word1)).intersection(set(list(word2))).intersection(set(list(word3)))
            if len(overlapping_letters)> 0:
                self.winning_combinations.append({word1, word2, word3})
        # print(self.winning_combinations)

        if init_board is None:
            self.board = [self.words, [], []] # remaining nums, player 1 nums, player -1 nums
        else:
            words_chosen = init_board[0] + init_board[1]
            remaining_words = [word for word in self.words if not word in words_chosen]
            self.board = [remaining_words, sorted(init_board[0]), sorted(init_board[1])]

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
        player1_words = set(self.board[1])
        # print(player1_words)
        for winning_combo in self.winning_combinations:
            if len(winning_combo.intersection(player1_words)) == 3:
                return 1

        playerm1_words = set(self.board[2])
        for winning_combo in self.winning_combinations:
            if len(winning_combo.intersection(playerm1_words)) == 3:
                return -1

        if len(self.board[0]) > 0:
            return None
        return 0

class WordTicTacToeState(GameState, WordTicTacToeBase):
    def __init__(self, board, turn_idx, words):
        super().__init__()
        ## state
        self.board = board
        self.turn_idx = turn_idx
        self.words = words
        
        ## properties
        self.init_properties()

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(to_tuple(self.board))

    def __str__(self):
        s = f"""
        Words Left in Pot: {self.board[0]}
        Player A Words: {self.board[1]}
        Player B Words: {self.board[2]}
        """.strip()
        return s 

    def export_for_prompt(self):
        return self.board, self.next_player

class WordTicTacToe(Game, WordTicTacToeBase):
    def move(self, word):
        if self.get_winner():
            return
        assert word in self.board[0]
        self.board[0].remove(word)
        player_idx = 1 if self.get_next_player() == 1 else 2
        self.board[player_idx] = sorted(self.board[player_idx] + [word])
        self.turn_idx += 1
    
    def get_state(self):
        return WordTicTacToeState(self.board, self.turn_idx, self.words)