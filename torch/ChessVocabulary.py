import pickle
import os
import torch


class ChessVocabulary:
    def __init__(self, path="../model/chess_vocab.pkl"):
        self.path = path
        self.move_to_int = {}
        self.int_to_move = {}
        
    def load_or_create(self, moves=None):
        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                self.move_to_int = pickle.load(f)
                self.int_to_move = {v: k for k, v in self.move_to_int.items()}
        elif moves is not None:
            # Create standard chess move vocabulary
            all_possible_moves = self._generate_all_possible_moves()
            # Add moves from dataset
            all_moves = set(all_possible_moves) | set(moves)
            self.move_to_int = {move: i for i, move in enumerate(sorted(all_moves))}
            self.int_to_move = {v: k for k, v in self.move_to_int.items()}
            self.save()
            
    def _generate_all_possible_moves(self):
        """Generate all possible chess moves in UCI format"""
        files = 'abcdefgh'
        ranks = '12345678'
        moves = []
        
        # Generate regular moves
        for f1 in files:
            for r1 in ranks:
                for f2 in files:
                    for r2 in ranks:
                        moves.append(f"{f1}{r1}{f2}{r2}")
                        # Add promotion moves
                        if r2 in '18':
                            for piece in 'qrbn':
                                moves.append(f"{f1}{r1}{f2}{r2}{piece}")
        return moves
        
    def save(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.move_to_int, f)
            
    def encode_moves(self, moves):
        return torch.tensor([self.move_to_int.get(move, 0) for move in moves])
        
    def decode_move(self, index):
        return self.int_to_move.get(index)
        
    @property
    def vocab_size(self):
        return len(self.move_to_int)