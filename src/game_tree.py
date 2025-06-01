import numpy as np
import os
import argparse
from copy import deepcopy

from src.games import GAME_PACKAGES
from src.utils import json_save, json_dumps
    
class GameStateTree():
    def __init__(self, game, node_lookup=None, root=False):
        self._game_state = game.get_state()
        self.init_children(game, node_lookup)
        self.size = self.calc_size()
        if root:
            print(self.size)
        self.optimal_outcome, self.move_to_outcome, self.optimal_num_turns = self.calc_optimality()
        self.win_critical, self.lose_critical = self.calc_critical_point_types()
        self.outcomes_to_num_moves = self.calc_num_moves_to_outcomes()
        self.win_difficulty, self.lose_difficulty = self.calc_critical_point_difficulty()

    def __getattr__(self, name):
        """GameStateTree inherits from self.game_state"""
        return getattr(self._game_state, name)

    def init_children(self, game, node_lookup=None):
        if not node_lookup:
            node_lookup = {}
        node_lookup[game.get_state()] = self
        
        self.children = []
        for move in self.legal_moves:
            game_copy = deepcopy(game)
            game_copy.move(move)
            if game_copy.get_state() in node_lookup:
                self.children.append(node_lookup[game_copy.get_state()])
            else:
                self.children.append(GameStateTree(game_copy, node_lookup=node_lookup))

    def calc_optimality(self):
        """Assuming both players play optimally, return the optimal outcome: 1 = current player wins, -1 = current player loses, 0 = draw"""
        if len(self.children) == 0:
            return self.winner*self.next_player, {}, 0
        else:
            child_outcomes = np.array([child.optimal_outcome if child.next_player==self.next_player else -1*child.optimal_outcome for child in self.children])
            move_to_outcome = dict([(move, outcome) for move, outcome in zip(self.legal_moves, child_outcomes)])
            optimal_outcome = np.max(child_outcomes)
            if optimal_outcome >= 0:
                optimal_num_turns = 1+np.min([child.optimal_num_turns for child, child_outcome in zip(self.children, child_outcomes) if child_outcome == optimal_outcome])
            else:
                optimal_num_turns = 1+np.max([child.optimal_num_turns for child, child_outcome in zip(self.children, child_outcomes)  if child_outcome == optimal_outcome])
            return optimal_outcome, move_to_outcome, optimal_num_turns

    def mapfilter_traverse(self, already_seen=None, filter_fxn=lambda x: True, map_fxn=lambda x: x):
        if not already_seen:
            already_seen = set()
        already_seen.add(id(self))
        filtered = []
        if filter_fxn(self):
            filtered.append(map_fxn(self))
        for child in self.children:
            if not id(child) in already_seen:
                 filtered += child.mapfilter_traverse(already_seen=already_seen, filter_fxn=filter_fxn, map_fxn=map_fxn)
        return filtered

    def calc_size(self):
        return len(self.mapfilter_traverse(filter_fxn=lambda x: True, map_fxn=lambda x: 1))

    def calc_critical_point_types(self):
        """
        Returns if this game state is a critical point for winning and if it is a critical point for losing
        A critical point for winning is a point at which there is at least one move that guarantees victory under optimal play and at least one that does not
        A critical point for winning is a point at which there is at least one move that guarantees loss under optimal play and at least one that does not
        """
        child_outcomes = np.array([child.optimal_outcome if child.next_player==self.next_player else -1*child.optimal_outcome for child in self.children])
        if len(np.unique(child_outcomes)) <= 1: # outcome is guaranteed
            return False, False 
        win_critical = np.any(child_outcomes == 1) # there is at least one path that guarantees victory (with optimal play) and one that does not
        lose_critical = np.any(child_outcomes == -1) # there is at least one path that guarantees loss (with optimal play) and one that does not
        return win_critical, lose_critical

    def calc_num_moves_to_outcomes(self):
        """
        Returns a dictionary that maps each possible outcome (1 - win, 0 - draw, -1 - loss) 
        to a list of the number of moves in each possible path to achieve that outcome
        """
        if len(self.children) == 0:
            return {self.winner*self.next_player: [0]}
        outcomes_to_num_moves = {}
        for child in self.children:
            child_outcomes_to_moves = child.outcomes_to_num_moves
            for child_outcome, child_moves in child_outcomes_to_moves.items():
                player_change = child.next_player*self.next_player
                outcomes_to_num_moves[child_outcome*player_change] = outcomes_to_num_moves.get(child_outcome*player_change, []) + [steps+1 for steps in child_moves]
        return outcomes_to_num_moves

    def calc_critical_point_difficulty(self):
        """
        If this game state is a critical point return the difficulty of winning (if win_critical) and losing (if lose_critical)
        Difficulty of winning is defined as the minimum number of turns needed to guarantee a win (under optimal play)
        Difficulty of avoiding the loss is defined as the maximum number of turns needed for the opponent to win
        among any of the moves that lead to a guaranteed loss (under optimal play)
        """
        win_difficulty, lose_difficulty = None, None
        if self.win_critical:
            win_difficulty = 1+np.min([child.optimal_num_turns for child in self.children if child.optimal_outcome*child.next_player==self.next_player])
        if self.lose_critical:
            lose_difficulty = 1+np.max([child.optimal_num_turns for child in self.children if -1*child.optimal_outcome*child.next_player==self.next_player])
        return win_difficulty, lose_difficulty
    
    def visualize(self, node_lookup=None, id_str="0", viz_fxn=lambda x: x._game_state.visualize()):
        if not node_lookup:
            node_lookup = {}
        print("NODE ID", id_str)
        viz_fxn(self)
        print()
        node_lookup[self] = (id_str, self)
        for child_idx, child in enumerate(self.children):
            print(f"CHILD OF {id_str}")
            if child in node_lookup:
                node_id, node = node_lookup[child]
                print("NODE ID", node_id)
                viz_fxn(node)
            else:
                child.visualize(node_lookup, f"{id_str}.{child_idx}", viz_fxn=viz_fxn)
            print()


def export_game_states(game_type, outdir=None, include_all_states=False):
    filter_fxn=lambda x: x.win_critical != x.lose_critical
    if include_all_states:
        filter_fxn=lambda x: True
    game = GAME_PACKAGES[game_type].GameClass()
    tree = GameStateTree(game)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{game_type}.json")
    
    def tree_to_prompt_state(tree):
        move_outcomes = np.array(list(tree.move_to_outcome.values()))
        optimal_outcome = np.max(move_outcomes)
        return {
            "board": tree.board.tolist(),
            "next_player": tree.next_player,
            "moves": [
                {
                    'move': move,
                    'outcome': tree.move_to_outcome[move],
                    'optimal_num_turns_after_move': child.optimal_num_turns,
                    'tree_size_after_move': child.size,
                }
                for move, child in zip(tree.legal_moves, tree.children)
            ],
            "optimal_moves": [move for move in tree.legal_moves if tree.move_to_outcome[move] == optimal_outcome],
            "optimal_outcome": optimal_outcome,
            "win_critical": tree.win_critical,
            "lose_critical": tree.lose_critical,
            "win_difficulty": tree.win_difficulty,
            "lose_difficulty": tree.lose_difficulty,
            "num_optimal_moves": np.sum(move_outcomes == optimal_outcome),
            "optimal_move_percent": np.mean(move_outcomes == optimal_outcome),
            "tree_size": tree.size
        }
    states = tree.mapfilter_traverse(filter_fxn=filter_fxn, map_fxn=tree_to_prompt_state)
    json_save(outpath, states)
    return states

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="Game to generate tree from", required=True, type=str)
    parser.add_argument("--data_root", help="Directory to save critical points in", default="./data/critical_points", type=str)
    parser.add_argument("--include_all_states", help="Whether to export only the critical points in the game tree or all states", action='store_true')
    args = parser.parse_args()

    export_game_states(args.game, args.data_root, args.include_all_states)

if __name__ == "__main__":
    main()