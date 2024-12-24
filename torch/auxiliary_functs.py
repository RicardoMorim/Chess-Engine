import chess
import numpy as np
from chess import Board
import torch


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
	# 14th board for source squares (from where we can move)
	
	matrix = np.zeros((14, 8, 8))
	piece_map = board.piece_map()

	for square, piece in piece_map.items():
		row, col = divmod(square, 8)
		piece_type = piece.piece_type - 1
		piece_color = 0 if piece.color else 6
		matrix[piece_type + piece_color, row, col] = 1

	legal_moves = board.legal_moves
	
	for move in legal_moves:
		to_square = move.to_square
		from_square = move.from_square
		row_to, col_to = divmod(to_square, 8)
		row_from, col_from = divmod(from_square, 8)
		matrix[12, row_to, col_to] = 1    # Destination squares
		matrix[13, row_from, col_from] = 1 # Source squares
		
	return matrix


def create_input_for_nn(filtered_positions):
    """Process filtered positions into training data."""
    if not filtered_positions:
        raise ValueError("No positions provided")
        
    X = []
    y = []
    
    for board, move in filtered_positions:
        try:
            X.append(board_to_matrix(board))
            y.append(move.uci())
        except Exception as e:
            print(f"Error processing position: {str(e)}")
            continue
    
    if not X:
        raise ValueError("No valid positions were processed")
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"Processed {len(X)} positions into training data")
    return X, y

def encode_moves(moves):
	move_to_int = {move: i for i, move in enumerate(set(moves))}
	return (
		np.array([move_to_int[move] for move in moves], dtype=np.float32),
		move_to_int,
	)


def get_king_position(board):
	"""Get king positions for both sides"""
	white_king = board.king(chess.WHITE)
	black_king = board.king(chess.BLACK)
	return white_king, black_king

def calculate_king_safety(board, king_positions):
	"""Calculate king safety features"""
	white_king, black_king = king_positions
	safety_features = np.zeros((2, 8, 8))
	
	# Count attacking pieces around kings
	for color, king_pos in enumerate([white_king, black_king]):
		if king_pos is not None:
			row, col = divmod(king_pos, 8)
			for i in range(max(0, row-1), min(8, row+2)):
				for j in range(max(0, col-1), min(8, col+2)):
					if board.is_attacked_by(not bool(color), i*8 + j):
						safety_features[color, i, j] = 1
	
	return torch.tensor(safety_features, dtype=torch.float32)

def calculate_material_balance(board):
	"""Calculate material balance feature"""
	piece_values = {
		chess.PAWN: 1,
		chess.KNIGHT: 3,
		chess.BISHOP: 3,
		chess.ROOK: 5,
		chess.QUEEN: 9
	}
	
	material = np.zeros((1, 8, 8))
	for square, piece in board.piece_map().items():
		row, col = divmod(square, 8)
		value = piece_values.get(piece.piece_type, 0)
		material[0, row, col] = value if piece.color else -value
	
	return torch.tensor(material, dtype=torch.float32)