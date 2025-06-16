import numpy as np
from game import GameBase, GameState, Game
from utils import to_tuple

class ChopsticksBase(GameBase):
    def __init__(self, max_finger_count=5, init_board=None):
        self.turn_idx = 0
        self.max_finger_count = max_finger_count
    
        if init_board is None:
            self.board = np.array([[1,1],[1,1]])
        else:
            self.board = np.array(init_board)

    def get_next_player(self):
        # 1 always starts
        return 1 if self.turn_idx % 2 == 0 else -1

    def get_legal_moves(self):
        # tap moves are of the format ("TAP", my_finger_idx, your_finger_idx)
        # split moves are of the format ("SPLIT", new_my_finger0, new_my_finger1)
        if self.get_winner():
            return []
        else:
            self_fingers = self.board[self.turn_idx%2]
            other_fingers = self.board[(self.turn_idx+1)%2]
            non_zero_self_fingers = np.arange(len(self_fingers))[self_fingers != 0].tolist()
            non_zero_other_fingers = np.arange(len(other_fingers))[other_fingers != 0].tolist()
            tap_moves = [("TAP", self_finger, other_finger) for self_finger in non_zero_self_fingers for other_finger in non_zero_other_fingers]

            finger_total = int(np.sum(self_fingers))
            finger_split_options = np.array([[x, finger_total-x] for x in np.arange(finger_total+1)])
            if len(finger_split_options) > 0:
                finger_split_options = finger_split_options[np.all(finger_split_options >= 0, axis=1) & np.all(finger_split_options < self.max_finger_count, axis=1) & np.all(finger_split_options != self_fingers[0], axis=1)]
            split_moves = [("SPLIT", *fingers) for fingers in finger_split_options.tolist()]
            
            return tap_moves + split_moves

    def get_winner(self):
        if np.all(self.board[1] == 0):
            return 1
        elif np.all(self.board[0] == 0):
            return -1
        else:
            return None
    
    def similar_to(self, other):
        return False
    
    def get_game_properties(self):
        return {
            "max_finger_count": self.max_finger_count,
        }

class ChopsticksState(GameState, ChopsticksBase):
    def __init__(self, board, turn_idx, max_finger_count):
        ## state
        self.board = board
        self.turn_idx = turn_idx
        self.max_finger_count = max_finger_count
        
        ## properties
        self.init_properties()

    def __eq__(self, other):
        return (to_tuple(self.board.tolist()) == to_tuple(other.board.tolist())) and (self.next_player == other.next_player)

    def __hash__(self):
        return hash((to_tuple(self.board.tolist()), self.next_player))

    def __str__(self):
        return f"Next Player: {self.next_player}\n" + "\n".join([str(row) for row in self.board.tolist()])

    def export_for_prompt(self):
        return self.board.tolist(), self.next_player

class Chopsticks(Game, ChopsticksBase):
    def move(self, move):
        if self.get_winner():
            return
            
        move_type, move = move[0], move[1:]
        assert len(move) == 2
        current_player_idx = self.turn_idx%2
        if move_type == "TAP":
            assert self.board[current_player_idx][move[0]] != 0
            assert self.board[1-current_player_idx][move[1]] != 0
            self.board[1-current_player_idx][move[1]] += self.board[current_player_idx][move[0]]
            self.board[1-current_player_idx][move[1]] %= self.max_finger_count
        elif move_type == "SPLIT":
            assert np.sum(self.board[current_player_idx]) == np.sum(move)
            assert not np.min(self.board[current_player_idx]) == np.min(move)
            assert not np.min(move) < 0
            assert not np.max(move) >= self.max_finger_count
            self.board[current_player_idx] = move
        else:
            raise ValueError(f"Move type {move_type} is not valid. Must be one of ('TAP', 'SPLIT')")
        self.turn_idx += 1
    
    def get_state(self):
        return ChopsticksState(self.board, self.turn_idx, self.max_finger_count)