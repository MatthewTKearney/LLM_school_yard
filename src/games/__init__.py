from src.games import tictactoe, connect_four

# game class located at package.game.Game
# prompt template located at package.prompt.create_prompt
# prompt parser located at package.prompt.response_to_move
# baseline strategies located at package.strategies.get_strategies

GAME_PACKAGES = {
    'tictactoe': tictactoe,
    "connect_four": connect_four
}