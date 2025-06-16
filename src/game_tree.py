import numpy as np
import os
import argparse
from copy import deepcopy

from games import GAME_PACKAGES
from utils import json_save, json_dumps
from functools import partial
import sys
    
from copy import copy


class GameStateTree():
    def __init__(self, game, node_lookup=None, root=False, idx=0):
        self._game_state = game.get_state()
        self.children = []
        self.parents = []

    def __getattr__(self, name):
        """GameStateTree inherits from self.game_state"""
        return getattr(self._game_state, name)

class GameTreeWrapper():
    def __init__(self, game):
        self.all_nodes, self.root, self.leaves = self.construct_tree(game)
        self.compute_subtree_sizes()
        # print(self.root.size)
        self.compute_optimal_outcome()
        self.compute_optimal_num_turns()
        self.compute_remaining_properties()

    def construct_tree(self, game):
        root = GameStateTree(game)
        visited = {root._game_state: root}
        nodes = [(root, game)] # if would rather BFS, then change to pqueue
        leaves = []
        all_nodes = []
        while len(nodes) > 0:
            node, game = nodes.pop()
            all_nodes.append(node)
            for move in node.legal_moves:
                game_copy = deepcopy(game)
                game_copy.move(move)
                game_copy_state = game_copy.get_state()
                if game_copy_state in visited:
                    visited[game_copy_state].parents.append(node)
                    node.children.append(visited[game_copy_state])
                else:
                    child_node = GameStateTree(game_copy)
                    child_node.parents.append(node)
                    node.children.append(child_node)
                    
                    visited[game_copy_state] = child_node
                    nodes.append((child_node, game_copy))
            if len(node.legal_moves) == 0:
                leaves.append(node)
        return set(all_nodes), root, leaves

    def get_root_size(self, root):
        def node_fxn(node, nodes_to_visit, visited, output):
            new_nodes = []
            output["count"] += 1
            for child in node.children:
                if not child in visited:
                    new_nodes.append(child)
                    visited.add(child)
            return new_nodes
        return self.tree_apply(node_fxn, root=root, output={"count": 0})["count"]

    def compute_subtree_sizes(self):
        for node in self.all_nodes:
            node.size = self.get_root_size(node)
            
    def compute_optimal_outcome(self):
        def node_fxn(node, nodes_to_visit, visited, output):
            new_nodes = []
            if len(node.children) == 0:
                node.optimal_outcome = node.winner*node.next_player
                new_nodes = [parent for parent in node.parents if not parent in nodes_to_visit and not parent in visited]
                visited.add(node)
            else:
                child_optimal_outcomes = [child.next_player*node.next_player*child.optimal_outcome for child in node.children if hasattr(child, 'optimal_outcome')]
                if max(child_optimal_outcomes) == 1 or len(child_optimal_outcomes) == len(node.children):
                    node.optimal_outcome = max(child_optimal_outcomes)
                    visited.add(node)
                    new_nodes = new_nodes = [parent for parent in node.parents if not parent in nodes_to_visit and not parent in visited]
                    
            return new_nodes

        def default_fxn(node):
            node.optimal_outcome = 0

        self.tree_apply(node_fxn, start_from_leaves=True, default_fxn=default_fxn)

    def compute_optimal_num_turns(self):
        def node_fxn(node, nodes_to_visit, visited, output):
            visited.add(node)
            new_nodes = []
            if len(node.children) == 0:
                node.optimal_num_turns = 0
                new_nodes = [parent for parent in node.parents if (not parent in nodes_to_visit) and (parent.optimal_outcome == node.optimal_outcome*node.next_player*parent.next_player)]
            else:
                child_optimal_outcomes = [child.next_player*node.next_player*child.optimal_outcome for child in node.children if hasattr(child, 'optimal_outcome') ]
                optimal_children = [child for child, outcome in zip(node.children, child_optimal_outcomes) if outcome == node.optimal_outcome]
                old_num_moves = node.optimal_num_turns if hasattr(node, "optimal_num_turns") else -1
                optimal_children_num_moves = [child.optimal_num_turns for child in optimal_children if hasattr(child, "optimal_num_turns")]
                if node.optimal_outcome >= 0:
                    node.optimal_num_turns = 1+np.min(optimal_children_num_moves)
                else:
                    node.optimal_num_turns = 1+np.max(optimal_children_num_moves)
                if not old_num_moves == node.optimal_num_turns:
                    for parent in node.parents:
                        if (not parent in nodes_to_visit) and (parent.optimal_outcome == node.optimal_outcome*node.next_player*parent.next_player):
                            new_nodes.append(parent)
            return new_nodes

        def default_fxn(node):
            node.optimal_num_turns = np.inf

        self.tree_apply(node_fxn, start_from_leaves=True, default_fxn=default_fxn)

    def compute_remaining_properties(self):
        def assign_props(node):
            child_optimal_outcomes = [child.next_player*node.next_player*child.optimal_outcome for child in node.children]
            unique_outcomes = set(child_optimal_outcomes)
            node.win_critical = node.optimal_outcome == 1 and len(unique_outcomes) > 1
            node.lose_critical = any([outcome == -1 for outcome in child_optimal_outcomes]) and len(unique_outcomes) > 1
            node.win_difficulty = node.optimal_num_turns if node.optimal_outcome == 1 else None
            node.lose_difficulty = 1+np.max([child.optimal_num_turns for child in node.children if child.optimal_outcome*node.next_player*child.next_player == -1]) if node.lose_critical else None
            node.move_to_outcome = {move: outcome for move, outcome in zip(node.legal_moves, child_optimal_outcomes)}
        for node in self.all_nodes:
            assign_props(node)
    
    
    def tree_apply(self, node_fxn, root=None, default_fxn=None, start_from_leaves=False, output=None):
        nodes = set([self.root if root is None else root])
        if start_from_leaves:
            nodes = set(self.leaves)
        visited = set(nodes)
        while len(nodes) > 0:
            node = nodes.pop()
            new_nodes = node_fxn(node, nodes, visited, output)
            nodes.update(set(new_nodes))
           
        if default_fxn:
            unvisited_nodes = self.all_nodes.difference(visited)
            for node in unvisited_nodes:
                default_fxn(node)
        return output
    
def export_game_states(game_type, outdir=None, include_all_states=False):
    filter_fxn=lambda x: x.win_critical != x.lose_critical
    if include_all_states:
        filter_fxn=lambda x: True
    game = GAME_PACKAGES[game_type].Game()
    tree = GameTreeWrapper(game)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{game_type}.json")
    
    def tree_to_prompt_state(tree, node_similarities={}):
        # similarity_idx = None
        # for node, idx in node_similarities.items():
        #     if node.similar_to(tree):
        #         similarity_idx = idx
        #         break
        # if not similarity_idx:
        #     similarity_idx = len(node_similarities)
        #     node_similarities[tree] = similarity_idx

        move_outcomes = np.array(list(tree.move_to_outcome.values()))
        optimal_outcome = np.max(move_outcomes)
        state_dict = {
            "board": tree.board,
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
            "tree_size": tree.size,
            # "similarity_idx": similarity_idx,
            **tree.get_game_properties(),
        }
        if hasattr(tree, "move_properties"):
            state_dict["move_properties"] = tree.move_properites
        return state_dict
    similarity_mapping = {}
    states = []
    for node in tree.all_nodes:
        if filter_fxn(node):
            states.append(tree_to_prompt_state(node, node_similarities=similarity_mapping))
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