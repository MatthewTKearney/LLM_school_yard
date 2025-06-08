from src.games import tictactoe, connect_four, sum_15, wordtictactoe, chopsticks

# game class located at package.game.Game
# prompt template located at package.prompt.create_prompt
# prompt parser located at package.prompt.response_to_move
# baseline strategies located at package.strategies.get_strategies

GAME_PACKAGES = {
    'tictactoe': tictactoe,
    "connect_four": connect_four,
    "sum_15": sum_15,
    "word_tictactoe": wordtictactoe,
    "chopsticks": chopsticks,
}