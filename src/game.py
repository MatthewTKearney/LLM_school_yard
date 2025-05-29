from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

class GameBase(ABC):
    @abstractmethod
    def get_next_player(self):
        pass

    @abstractmethod
    def get_legal_moves(self):
        pass

    @abstractmethod
    def get_winner(self):
        """
        returns None if still in play, 0 if draw, 1 if player 1 wins, -1 if player 1 loses
        """
        pass

class GameState(GameBase, ABC):
    def init_properties(self):
        self.next_player = self.get_next_player()
        self.legal_moves = self.get_legal_moves()
        self.winner = self.get_winner()

    @abstractmethod
    def __eq__(self, other):
         pass

    @abstractmethod
    def __hash__(self):
         pass
    
    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def export_for_prompt(self):
        pass
         
class Game(GameBase, ABC):
    @abstractmethod
    def move(self, move):
        pass

    @abstractmethod
    def get_state(self):
        """
        returns hashable version of game state
        """
        pass

    def __eq__(self, other):
        return self.get_state() == other.get_state()

    def __str__(self):
        return self.get_state().__str__()