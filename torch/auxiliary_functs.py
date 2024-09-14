import numpy as np
from chess import Board


def board_to_matrix(board: Board):
    """
    Converts a chess board to a matrix representation.

    You can also add a 14th dimension for even better results. The 13th channel shows where you can move pieces, but it doesn't show from which squares you can move them. I've already tried it; the results will be better, though blunders are inevitable, of course.

    Here is how you can modify the board_to_matrix function to include the 14th dimension (auxiliary_func.py, from line 22):

    for move in legal_moves:
        to_square = move.to_square
        from_square = move.from_square
        row_to, col_to = divmod(to_square, 8)
        row_from, col_from = divmod(from_square, 8)
        matrix[12, row_to, col_to] = 1
        matrix[13, row_from, col_from] = 1
    """
    # 8x8 board, 12 number of unique pieces
    # 13th board for legal moves (where we can move)

    matrix = np.zeros((13, 8, 8))
    piece_map = board.piece_map()

    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        piece_type = piece.piece_type - 1
        piece_color = 0 if piece.color else 6
        matrix[piece_type + piece_color, row, col] = 1

    legal_moves = board.legal_moves

    for move in legal_moves:
        to_square = move.to_square
        row_to, col_to = divmod(to_square, 8)
        matrix[12, row_to, col_to] = 1

    return matrix


def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: i for i, move in enumerate(set(moves))}
    return (
        np.array([move_to_int[move] for move in moves], dtype=np.float32),
        move_to_int,
    )
