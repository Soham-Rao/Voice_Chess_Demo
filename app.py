from __future__ import annotations

import random
import re
import os
import sys
import threading
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from difflib import get_close_matches

import chess
from PIL import Image, ImageTk


def ensure_tcl_tk_paths() -> bool:
    """Conda on Windows sometimes needs explicit Tcl/Tk library paths."""
    changed = False
    prefix = Path(sys.prefix)
    tcl_dir = prefix / "Library" / "lib" / "tcl8.6"
    tk_dir = prefix / "Library" / "lib" / "tk8.6"
    if tcl_dir.exists() and "TCL_LIBRARY" not in os.environ:
        os.environ["TCL_LIBRARY"] = str(tcl_dir)
        changed = True
    if tk_dir.exists() and "TK_LIBRARY" not in os.environ:
        os.environ["TK_LIBRARY"] = str(tk_dir)
        changed = True
    return changed


if ensure_tcl_tk_paths() and __name__ == "__main__" and os.environ.get("WIZARD_CHESS_TCL_READY") != "1":
    env = os.environ.copy()
    env["WIZARD_CHESS_TCL_READY"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import customtkinter as ctk

try:
    import speech_recognition as sr
except ImportError:  # The app still works with typed and clicked moves.
    sr = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    import torch
except ImportError:
    torch = None


BASE_DIR = Path(__file__).resolve().parent
PIECE_DIR = BASE_DIR / "pieces-basic-png"
WAND_FILE = BASE_DIR / "wand.png"
TRANSCRIBE_MODEL = os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
LOCAL_WHISPER_MODEL = os.environ.get("LOCAL_WHISPER_MODEL", "tiny.en")
LOCAL_WHISPER_DEVICE = os.environ.get("LOCAL_WHISPER_DEVICE", "auto")
USE_OPENAI_TRANSCRIBE = os.environ.get("USE_OPENAI_TRANSCRIBE") == "1"
CHESS_TRANSCRIBE_PROMPT = (
    "The speaker is controlling a chess game. Transcribe chess commands literally. "
    "Likely words include pawn, knight, bishop, rook, queen, king, from, to, "
    "and board squares a1 through h8. Examples: knight e3 to d5, pawn e4, "
    "rook b7, bishop c7, queen c4, f1 to e3. Do not turn square commands into long numbers."
)
LOCAL_WHISPER_CACHE = None
LOCAL_WHISPER_CACHE_DEVICE = None

BOARD_SIZE = 560
SQUARE_SIZE = BOARD_SIZE // 8

LIGHT_SQUARE = "#d8b66c"
DARK_SQUARE = "#5b3326"
GOLD = "#d7af42"
INK = "#1d130d"
PARCHMENT = "#f0d79a"
PANEL = "#211510"
PANEL_2 = "#321f18"
GLOW = "#7fb7ff"
BAD = "#b84a35"
GOOD = "#75b86b"

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

NAME_TO_PIECE = {name: piece for piece, name in PIECE_NAMES.items()}
NAME_TO_PIECE.update(
    {
        "horse": chess.KNIGHT,
        "castle": chess.ROOK,
        "night": chess.KNIGHT,
        "nite": chess.KNIGHT,
        "rishab": chess.BISHOP,
        "rishabh": chess.BISHOP,
        "bishup": chess.BISHOP,
        "rocket": chess.ROOK,
        "brook": chess.ROOK,
        "on": chess.PAWN,
        "one": chess.PAWN,
        "want": chess.PAWN,
        "won": chess.PAWN,
        "pond": chess.PAWN,
        "porn": chess.PAWN,
        "born": chess.PAWN,
        "pan": chess.PAWN,
        "queen": chess.QUEEN,
        "king": chess.KING,
    }
)

FILE_WORDS = {
    "a": "a",
    "ay": "a",
    "hey": "a",
    "b": "b",
    "be": "b",
    "bee": "b",
    "c": "c",
    "see": "c",
    "sea": "c",
    "d": "d",
    "dee": "d",
    "the": "d",
    "nine": "d",
    "9": "d",
    "e": "e",
    "ee": "e",
    "eat": "e",
    "f": "f",
    "eff": "f",
    "g": "g",
    "gee": "g",
    "h": "h",
    "aitch": "h",
    "age": "h",
}

SPEECH_DIGIT_FILES = {
    "1": "a",
    "2": "d",
    "3": "c",
    "4": "f",
    "5": "b",
    "6": "g",
    "7": "h",
    "8": "e",
    "9": "e",
}

RANK_WORDS = {
    "1": "1",
    "one": "1",
    "won": "1",
    "2": "2",
    "two": "2",
    "to": "2",
    "too": "2",
    "3": "3",
    "three": "3",
    "tree": "3",
    "4": "4",
    "four": "4",
    "for": "4",
    "food": "4",
    "5": "5",
    "five": "5",
    "6": "6",
    "six": "6",
    "sicks": "6",
    "7": "7",
    "seven": "7",
    "8": "8",
    "eight": "8",
    "ate": "8",
}

VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}


@dataclass
class ParsedCommand:
    piece_type: Optional[int]
    destination: Optional[chess.Square]
    origin: Optional[chess.Square] = None


@dataclass
class SpecialRule:
    kind: str = "capture"
    attacker: int = chess.KNIGHT
    target: int = chess.ROOK
    target_color: Optional[bool] = chess.BLACK
    message: str = "Mischief managed. The challenge is yours."


@dataclass
class MoveResult:
    ok: bool
    message: str
    move: Optional[chess.Move] = None
    game_over: bool = False
    victory: bool = False
    ambiguous_from: tuple[chess.Square, ...] = ()


def square_name(square: chess.Square) -> str:
    return chess.square_name(square)


def parse_square(text: str) -> Optional[chess.Square]:
    match = re.search(r"\b([a-h])\s*([1-8])\b", text.lower())
    if not match:
        return None
    return chess.parse_square(match.group(1) + match.group(2))


def spoken_square(text: str) -> Optional[chess.Square]:
    text = text.lower().strip()
    compact = re.sub(r"[^a-z0-9]", "", text)
    compact_aliases = {
        "seafood": "c4",
        "seafour": "c4",
        "seafor": "c4",
        "seafive": "c5",
        "seasix": "c6",
        "seaseven": "c7",
        "seathree": "c3",
        "eattree": "e3",
        "eatthree": "e3",
        "eate": "e8",
    }
    if compact in compact_aliases:
        return chess.parse_square(compact_aliases[compact])
    if re.fullmatch(r"[a-h][1-8]", compact):
        return chess.parse_square(compact)
    if re.fullmatch(r"9[1-8]", compact):
        return chess.parse_square("d" + compact[1])

    words = re.findall(r"[a-z0-9]+", text)
    for index, word in enumerate(words):
        if re.fullmatch(r"[a-h][1-8]", word):
            return chess.parse_square(word)
        if re.fullmatch(r"9[1-8]", word):
            return chess.parse_square("d" + word[1])
        file_name = FILE_WORDS.get(word)
        if file_name and index + 1 < len(words):
            rank = RANK_WORDS.get(words[index + 1])
            if rank:
                return chess.parse_square(file_name + rank)
    return None


def likely_square_from_tail(text: str) -> Optional[chess.Square]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    for size in (2, 1):
        if len(words) >= size:
            square = spoken_square(" ".join(words[-size:]))
            if square is not None:
                return square
    return spoken_square(text)


def decode_numeric_square_pair(text: str) -> tuple[Optional[chess.Square], Optional[chess.Square]]:
    digits = re.sub(r"\D", "", text)
    if len(digits) < 4:
        return None, None
    digits = digits[-4:]
    origin_file = SPEECH_DIGIT_FILES.get(digits[0])
    destination_file = SPEECH_DIGIT_FILES.get(digits[2])
    if not origin_file or not destination_file or digits[1] not in "12345678" or digits[3] not in "12345678":
        return None, None
    return chess.parse_square(origin_file + digits[1]), chess.parse_square(destination_file + digits[3])


def spoken_piece(text: str) -> Optional[int]:
    words = re.findall(r"[a-z]+", text.lower())
    if not words:
        return None
    for word in words[:3]:
        if word in NAME_TO_PIECE:
            return NAME_TO_PIECE[word]
    close = get_close_matches(words[0], list(NAME_TO_PIECE), n=1, cutoff=0.68)
    if close:
        return NAME_TO_PIECE[close[0]]
    return None


def normalize_command(raw: str) -> str:
    text = raw.lower().strip()
    text = text.replace("-", " ")
    text = re.sub(r"\bmove\b|\bcast\b|\bspell\b|\bplease\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_command(raw: str) -> ParsedCommand:
    text = normalize_command(raw)
    origin = None
    destination = None
    separator_match = re.search(r"\b(to|two|too|2)\b", text)
    numeric_origin, numeric_destination = decode_numeric_square_pair(text)
    if numeric_origin is not None and numeric_destination is not None:
        origin = numeric_origin
        destination = numeric_destination

    squares = re.findall(r"\b[a-h]\s*[1-8]\b", text)
    if squares:
        destination = chess.parse_square(squares[-1].replace(" ", ""))
        if len(squares) >= 2:
            origin = chess.parse_square(squares[-2].replace(" ", ""))

    from_to_match = re.search(r"\bfrom\s+(.+?)\s+\bto\b\s+(.+)$", text)
    if from_to_match:
        origin = spoken_square(from_to_match.group(1)) or origin
        destination = likely_square_from_tail(from_to_match.group(2)) or destination
    elif separator_match:
        before = text[: separator_match.start()]
        after = text[separator_match.end() :]
        origin = likely_square_from_tail(before) or origin
        destination = likely_square_from_tail(after) or destination

    if destination is None:
        destination = likely_square_from_tail(text)

    piece_type = spoken_piece(text)
    for name, piece in NAME_TO_PIECE.items():
        if re.search(rf"\b{name}s?\b", text):
            piece_type = piece
            break

    # Algebraic-ish fallback such as "ne5", "rb7", or "a4".
    compact = text.replace(" ", "")
    if piece_type is None and compact:
        first = compact[0]
        piece_type = {
            "n": chess.KNIGHT,
            "b": chess.BISHOP,
            "r": chess.ROOK,
            "q": chess.QUEEN,
            "k": chess.KING,
        }.get(first)
    if piece_type is None and destination is not None and origin is None:
        piece_type = chess.PAWN

    return ParsedCommand(piece_type=piece_type, destination=destination, origin=origin)


def piece_file(piece: chess.Piece) -> Path:
    color = "white" if piece.color == chess.WHITE else "black"
    return PIECE_DIR / f"{color}-{PIECE_NAMES[piece.piece_type]}.png"


def command_label(parsed: ParsedCommand) -> str:
    piece = PIECE_NAMES.get(parsed.piece_type, "piece")
    destination = square_name(parsed.destination) if parsed.destination is not None else "unknown square"
    if parsed.origin is not None:
        return f"{piece} from {square_name(parsed.origin)} to {destination}"
    return f"{piece} {destination}"


def explain_failed_move(board: chess.Board, parsed: ParsedCommand, color: bool, pseudo: bool = False) -> str:
    label = command_label(parsed)
    if parsed.piece_type is None and parsed.destination is None:
        return "I could not hear a full move. Try saying it as origin to target, like 'c3 to d5'."
    if parsed.piece_type is None:
        return f"{label} is invalid because I could not identify the piece. If you know the origin, say it like 'c3 to d5'."
    if parsed.destination is None:
        return f"{label} is invalid because I could not identify the square."

    own_piece = board.piece_at(parsed.destination)
    if own_piece and own_piece.color == color:
        return f"{label} is invalid because one of your own pieces is already on {square_name(parsed.destination)}."

    candidates = [
        square
        for square, piece in board.piece_map().items()
        if piece.color == color and piece.piece_type == parsed.piece_type and (parsed.origin is None or square == parsed.origin)
    ]
    if not candidates:
        return f"{label} is invalid because you do not have that piece available to move."

    generator = board.generate_pseudo_legal_moves() if pseudo else board.generate_legal_moves()
    rough_matches = [
        move
        for move in generator
        if move.to_square == parsed.destination
        and move.from_square in candidates
        and board.piece_at(move.from_square)
        and board.piece_at(move.from_square).piece_type == parsed.piece_type
    ]
    if rough_matches:
        return f"{label} is invalid because the move is blocked by the current challenge rules."

    pseudo_matches = [
        move
        for move in board.generate_pseudo_legal_moves()
        if move.to_square == parsed.destination and move.from_square in candidates
    ]
    if pseudo_matches and not pseudo:
        if board.is_check():
            return f"{label} is invalid because your king is in check and that move does not save it."
        return f"{label} is invalid because it would leave your king in check."

    if board.is_check() and not pseudo:
        return f"{label} is invalid because your king is in check and that move does not answer the threat."

    origins = ", ".join(square_name(square) for square in candidates)
    return f"{label} is invalid because no {PIECE_NAMES[parsed.piece_type]} from {origins} can move to {square_name(parsed.destination)}."


def infer_piece_from_origin(board: chess.Board, parsed: ParsedCommand, color: bool) -> ParsedCommand:
    if parsed.origin is None:
        return parsed
    piece = board.piece_at(parsed.origin)
    if piece and piece.color == color:
        return ParsedCommand(piece.piece_type, parsed.destination, parsed.origin)
    return parsed


def recover_obvious_origin(board: chess.Board, parsed: ParsedCommand, color: bool, pseudo: bool = False) -> ParsedCommand:
    if parsed.origin is None or parsed.destination is None or parsed.piece_type is None:
        return parsed
    piece = board.piece_at(parsed.origin)
    if piece and piece.color == color and piece.piece_type == parsed.piece_type:
        return parsed

    moves = board.generate_pseudo_legal_moves() if pseudo else board.generate_legal_moves()
    matches = []
    for move in moves:
        moving_piece = board.piece_at(move.from_square)
        if not moving_piece or moving_piece.color != color or moving_piece.piece_type != parsed.piece_type:
            continue
        if move.to_square == parsed.destination:
            matches.append(move)
    if len(matches) == 1:
        return ParsedCommand(parsed.piece_type, parsed.destination, matches[0].from_square)
    return parsed


class PieceImages:
    def __init__(self, size: int) -> None:
        self.size = size
        self.cache: dict[str, ImageTk.PhotoImage] = {}

    def get(self, piece: chess.Piece) -> ImageTk.PhotoImage:
        key = piece.symbol()
        if key not in self.cache:
            image = Image.open(piece_file(piece)).convert("RGBA").resize((self.size, self.size), Image.Resampling.LANCZOS)
            self.cache[key] = ImageTk.PhotoImage(image)
        return self.cache[key]


class CasualAI:
    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        legal = list(board.legal_moves)
        if not legal:
            return None

        scored = []
        for move in legal:
            score = random.randint(-8, 8)
            moving_piece = board.piece_at(move.from_square)
            captured = board.piece_at(move.to_square)
            if captured:
                score += VALUES[captured.piece_type] - (VALUES[moving_piece.piece_type] // 14 if moving_piece else 0)
            if move.promotion:
                score += VALUES.get(move.promotion, 0)

            board.push(move)
            if board.is_checkmate():
                score += 100000
            elif board.is_check():
                score += 85

            attackers = board.attackers(board.turn, move.to_square)
            if moving_piece and attackers:
                score -= VALUES[moving_piece.piece_type] // 5
            board.pop()
            scored.append((score, move))

        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: min(5, len(scored))]
        return random.choice(top)[1]


class StandardGame:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.ai = CasualAI()

    def reset(self) -> None:
        self.board.reset()

    def command_move(self, raw: str, color: bool = chess.WHITE) -> MoveResult:
        parsed = parse_command(raw)
        parsed = infer_piece_from_origin(self.board, parsed, color)
        parsed = recover_obvious_origin(self.board, parsed, color)
        return self.parsed_move(parsed, color)

    def parsed_move(self, parsed: ParsedCommand, color: bool = chess.WHITE) -> MoveResult:
        if self.board.turn != color:
            return MoveResult(False, "It is not your turn.")
        if parsed.piece_type is None or parsed.destination is None:
            return MoveResult(False, explain_failed_move(self.board, parsed, color))

        matches = []
        for move in self.board.legal_moves:
            piece = self.board.piece_at(move.from_square)
            if not piece or piece.color != color:
                continue
            if piece.piece_type != parsed.piece_type or move.to_square != parsed.destination:
                continue
            if parsed.origin is not None and move.from_square != parsed.origin:
                continue
            matches.append(move)

        if not matches:
            return MoveResult(False, explain_failed_move(self.board, parsed, color))
        if len(matches) > 1:
            origins = tuple(move.from_square for move in matches)
            origin_names = ", ".join(square_name(sq) for sq in origins)
            return MoveResult(False, f"Ambiguous spell. Say the origin: {origin_names}.", ambiguous_from=origins)

        move = matches[0]
        san = self.board.san(move)
        self.board.push(move)
        if self.board.is_checkmate():
            return MoveResult(True, f"Checkmate! {san} sealed the board. Gryffindor-level brilliance.", move=move, game_over=True, victory=True)
        if self.board.is_stalemate():
            return MoveResult(True, f"{san} ends in stalemate. The magic fizzles into a draw.", move=move, game_over=True)
        return MoveResult(True, f"You cast {san}.", move=move, game_over=self.board.is_game_over())

    def click_move(self, origin: chess.Square, destination: chess.Square) -> MoveResult:
        move = chess.Move(origin, destination)
        piece = self.board.piece_at(origin)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(destination) in (0, 7):
            move = chess.Move(origin, destination, promotion=chess.QUEEN)
        if move not in self.board.legal_moves:
            parsed = ParsedCommand(piece.piece_type if piece else None, destination, origin)
            return MoveResult(False, explain_failed_move(self.board, parsed, self.board.turn))
        san = self.board.san(move)
        self.board.push(move)
        if self.board.is_checkmate():
            return MoveResult(True, f"Checkmate! {san} sealed the board. Gryffindor-level brilliance.", move=move, game_over=True, victory=True)
        if self.board.is_stalemate():
            return MoveResult(True, f"{san} ends in stalemate. The magic fizzles into a draw.", move=move, game_over=True)
        return MoveResult(True, f"You cast {san}.", move=move, game_over=self.board.is_game_over())

    def ai_move(self) -> MoveResult:
        if self.board.is_game_over():
            return MoveResult(False, "The duel has already ended.", game_over=True)
        move = self.ai.choose_move(self.board)
        if move is None:
            return MoveResult(False, "The opponent has no legal reply.", game_over=True)
        san = self.board.san(move)
        self.board.push(move)
        if self.board.is_checkmate():
            return MoveResult(True, f"Checkmate. The rival's {san} pins your king. The castle falls silent.", move=move, game_over=True, victory=False)
        if self.board.is_stalemate():
            return MoveResult(True, f"Rival casts {san}. Stalemate: the spellwork dissolves into a draw.", move=move, game_over=True)
        return MoveResult(True, f"Rival casts {san}.", move=move, game_over=self.board.is_game_over())


class SpecialGame:
    def __init__(self) -> None:
        self.board = chess.Board(None)
        self.board.turn = chess.WHITE
        self.start_fen = self.board.board_fen()
        self.start_turn = chess.WHITE
        self.rule = SpecialRule()
        self.ai = CasualAI()
        self.game_over = False

    def set_from_editor(self, board: chess.Board, turn: bool, rule: SpecialRule) -> None:
        self.board = board.copy(stack=False)
        self.board.turn = turn
        self.start_fen = self.board.board_fen()
        self.start_turn = turn
        self.rule = rule
        self.game_over = False

    def reset(self) -> None:
        self.board = chess.Board(None)
        self.board.set_board_fen(self.start_fen)
        self.board.turn = self.start_turn
        self.game_over = False

    def pseudo_moves(self) -> list[chess.Move]:
        return list(self.board.generate_pseudo_legal_moves())

    def command_move(self, raw: str, color: bool = chess.WHITE) -> MoveResult:
        parsed = parse_command(raw)
        parsed = infer_piece_from_origin(self.board, parsed, color)
        parsed = recover_obvious_origin(self.board, parsed, color, pseudo=True)
        return self.parsed_move(parsed, color)

    def parsed_move(self, parsed: ParsedCommand, color: bool = chess.WHITE) -> MoveResult:
        if self.game_over:
            return MoveResult(False, "Restart the challenge to play again.", game_over=True)
        if self.board.turn != color:
            return MoveResult(False, "It is not your turn.")
        if parsed.piece_type is None or parsed.destination is None:
            return MoveResult(False, explain_failed_move(self.board, parsed, color, pseudo=True))

        matches = []
        for move in self.pseudo_moves():
            piece = self.board.piece_at(move.from_square)
            if not piece or piece.color != color:
                continue
            if piece.piece_type != parsed.piece_type or move.to_square != parsed.destination:
                continue
            if parsed.origin is not None and move.from_square != parsed.origin:
                continue
            matches.append(move)

        if not matches:
            print(
                "[challenge invalid]",
                {
                    "fen": self.board.board_fen(),
                    "turn": "white" if self.board.turn == chess.WHITE else "black",
                    "parsed": parsed,
                    "pieces": {square_name(sq): piece.symbol() for sq, piece in self.board.piece_map().items()},
                },
                flush=True,
            )
            return MoveResult(False, explain_failed_move(self.board, parsed, color, pseudo=True))
        if len(matches) > 1:
            origins = tuple(move.from_square for move in matches)
            origin_names = ", ".join(square_name(sq) for sq in origins)
            return MoveResult(False, f"Ambiguous spell. Say the origin: {origin_names}.", ambiguous_from=origins)

        return self.apply_move(matches[0], "You cast")

    def click_move(self, origin: chess.Square, destination: chess.Square) -> MoveResult:
        if self.game_over:
            return MoveResult(False, "Restart the challenge to play again.", game_over=True)
        piece = self.board.piece_at(origin)
        if not piece or piece.color != self.board.turn:
            return MoveResult(False, "Choose one of your active pieces.")
        move = chess.Move(origin, destination)
        if piece.piece_type == chess.PAWN and chess.square_rank(destination) in (0, 7):
            move = chess.Move(origin, destination, promotion=chess.QUEEN)
        if move not in self.pseudo_moves():
            parsed = ParsedCommand(piece.piece_type if piece else None, destination, origin)
            print(
                "[challenge invalid click]",
                {
                    "fen": self.board.board_fen(),
                    "turn": "white" if self.board.turn == chess.WHITE else "black",
                    "from": square_name(origin),
                    "to": square_name(destination),
                },
                flush=True,
            )
            return MoveResult(False, explain_failed_move(self.board, parsed, self.board.turn, pseudo=True))
        return self.apply_move(move, "You cast")

    def apply_move(self, move: chess.Move, prefix: str) -> MoveResult:
        moving_piece = self.board.piece_at(move.from_square)
        captured_piece = self.board.piece_at(move.to_square)
        move_text = f"{PIECE_NAMES[moving_piece.piece_type].title()} {square_name(move.to_square)}"
        self.board.push(move)

        if self.rule.kind == "capture" and moving_piece and captured_piece:
            if moving_piece.piece_type == self.rule.attacker and captured_piece.piece_type == self.rule.target:
                self.game_over = True
                return MoveResult(True, self.rule.message, move=move, game_over=True, victory=True)

        if self.rule.kind == "eliminate":
            targets_left = [
                piece
                for piece in self.board.piece_map().values()
                if piece.piece_type == self.rule.target
                and (self.rule.target_color is None or piece.color == self.rule.target_color)
            ]
            if not targets_left:
                self.game_over = True
                return MoveResult(True, self.rule.message, move=move, game_over=True, victory=True)

        if self.rule.kind == "checkmate" and self.has_both_kings() and self.board.is_checkmate():
            self.game_over = True
            return MoveResult(True, self.rule.message, move=move, game_over=True, victory=True)

        if not self.pseudo_moves():
            self.game_over = True
            return MoveResult(True, "No moves remain. The challenge is lost.", move=move, game_over=True, victory=False)

        return MoveResult(True, f"{prefix} {move_text}.", move=move)

    def ai_move(self) -> MoveResult:
        if self.game_over:
            return MoveResult(False, "The challenge has ended.", game_over=True)
        moves = self.pseudo_moves()
        if not moves:
            self.game_over = True
            return MoveResult(False, "No rival moves remain.", game_over=True)

        scored = []
        for move in moves:
            piece = self.board.piece_at(move.from_square)
            captured = self.board.piece_at(move.to_square)
            score = random.randint(-5, 5)
            if captured:
                score += VALUES[captured.piece_type]
            if piece and piece.color == chess.BLACK:
                scored.append((score, move))
        if not scored:
            self.game_over = True
            return MoveResult(False, "The rival has no pieces left.", game_over=True)

        scored.sort(key=lambda item: item[0], reverse=True)
        return self.apply_move(random.choice(scored[: min(4, len(scored))])[1], "Rival casts")

    def has_both_kings(self) -> bool:
        return self.board.king(chess.WHITE) is not None and self.board.king(chess.BLACK) is not None


class ChessBoardView(ctk.CTkFrame):
    def __init__(self, master, image_bank: PieceImages, on_square: Callable[[chess.Square], None]) -> None:
        super().__init__(master, fg_color="#140d0a", corner_radius=8)
        self.image_bank = image_bank
        self.on_square = on_square
        self.board: chess.Board = chess.Board()
        self.selected: Optional[chess.Square] = None
        self.legal_targets: set[chess.Square] = set()
        self.highlight_from: set[chess.Square] = set()
        self.last_move: Optional[chess.Move] = None
        self.flipped = False

        self.canvas = ctk.CTkCanvas(self, width=BOARD_SIZE, height=BOARD_SIZE, highlightthickness=0, bg="#140d0a")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self._click)

    def set_board(
        self,
        board: chess.Board,
        selected: Optional[chess.Square] = None,
        legal_targets: Optional[set[chess.Square]] = None,
        highlight_from: Optional[set[chess.Square]] = None,
        last_move: Optional[chess.Move] = None,
    ) -> None:
        self.board = board
        self.selected = selected
        self.legal_targets = legal_targets or set()
        self.highlight_from = highlight_from or set()
        self.last_move = last_move
        self.draw()

    def _coords_for_square(self, square: chess.Square) -> tuple[int, int, int, int]:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        if self.flipped:
            col = 7 - file_index
            row = rank_index
        else:
            col = file_index
            row = 7 - rank_index
        x1 = col * SQUARE_SIZE
        y1 = row * SQUARE_SIZE
        return x1, y1, x1 + SQUARE_SIZE, y1 + SQUARE_SIZE

    def _square_from_xy(self, x: int, y: int) -> Optional[chess.Square]:
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        if not (0 <= col <= 7 and 0 <= row <= 7):
            return None
        file_index = 7 - col if self.flipped else col
        rank_index = row if self.flipped else 7 - row
        return chess.square(file_index, rank_index)

    def _click(self, event) -> None:
        square = self._square_from_xy(event.x, event.y)
        if square is not None:
            self.on_square(square)

    def draw(self) -> None:
        self.canvas.delete("all")
        for square in chess.SQUARES:
            x1, y1, x2, y2 = self._coords_for_square(square)
            file_index = chess.square_file(square)
            rank_index = chess.square_rank(square)
            color = LIGHT_SQUARE if (file_index + rank_index) % 2 else DARK_SQUARE
            if self.last_move and square in (self.last_move.from_square, self.last_move.to_square):
                color = "#9b6f2b"
            if square in self.highlight_from:
                color = BAD
            if square == self.selected:
                color = GLOW
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#2a1a12", width=1)

            if square in self.legal_targets:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                self.canvas.create_oval(cx - 10, cy - 10, cx + 10, cy + 10, fill="#f6e6a8", outline="#6d5421")

            if chess.square_rank(square) == 0:
                self.canvas.create_text(x1 + 8, y2 - 10, text=chess.FILE_NAMES[file_index], fill="#fff4c8", anchor="w", font=("Georgia", 9, "bold"))
            if chess.square_file(square) == 0:
                self.canvas.create_text(x1 + 9, y1 + 10, text=str(rank_index + 1), fill="#fff4c8", anchor="w", font=("Georgia", 9, "bold"))

        for square, piece in self.board.piece_map().items():
            x1, y1, x2, y2 = self._coords_for_square(square)
            image = self.image_bank.get(piece)
            self.canvas.create_image((x1 + x2) // 2, (y1 + y2) // 2, image=image)


class WizardChessApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Wizard Chess")
        self.geometry("1180x760")
        self.minsize(1040, 700)

        self.images = PieceImages(58)
        self.normal = StandardGame()
        self.special = SpecialGame()
        self.editor_board = chess.Board(None)
        self.active_mode = "normal"
        self.selected_square: Optional[chess.Square] = None
        self.last_move: Optional[chess.Move] = None
        self.ambiguous_from: set[chess.Square] = set()
        self.listening = False
        self.fullscreen = False
        self.editor_piece = chess.Piece(chess.KNIGHT, chess.WHITE)
        self.undo_snapshot: Optional[dict] = None
        self.ai_after_id: Optional[str] = None
        self.listen_timer_after_id: Optional[str] = None
        self.listen_started_at = 0.0
        self.listen_duration_seconds = 9
        self.closing = False
        self.challenge_started = False
        self.wand_image = self.load_wand_image()

        self._build_ui()
        self.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.bind("<Escape>", lambda _event: self.set_fullscreen(False))
        self.protocol("WM_DELETE_WINDOW", self.close_app)
        self.refresh_all("Welcome to the enchanted board. Speak or type a move to begin.")

    def load_wand_image(self) -> Optional[ctk.CTkImage]:
        if not WAND_FILE.exists():
            return None
        image = Image.open(WAND_FILE).convert("RGBA")
        image.thumbnail((84, 84), Image.Resampling.LANCZOS)
        return ctk.CTkImage(light_image=image, dark_image=image, size=image.size)

    def _build_ui(self) -> None:
        self.configure(fg_color="#120b08")
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.title_bar = ctk.CTkFrame(self, fg_color="#160f0b", corner_radius=0)
        self.title_bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.title_bar.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self.title_bar, text="Wizard Chess", font=("Georgia", 32, "bold"), text_color=GOLD).grid(row=0, column=0, padx=22, pady=14, sticky="w")
        self.status_label = ctk.CTkLabel(self.title_bar, text="", font=("Segoe UI", 15), text_color="#f4dfaa")
        self.status_label.grid(row=0, column=1, padx=10, sticky="ew")
        ctk.CTkButton(self.title_bar, text="Fullscreen", command=self.toggle_fullscreen, width=126, fg_color="#6d4b1f", hover_color="#8f6329").grid(row=0, column=2, padx=8)
        ctk.CTkButton(self.title_bar, text="Restart", command=self.restart_current, width=104, fg_color="#6d2d22", hover_color="#8a3a2c").grid(row=0, column=3, padx=(0, 18))

        self.left_panel = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=8)
        self.left_panel.grid(row=1, column=0, padx=(18, 10), pady=16, sticky="ns")
        self.mode_label = ctk.CTkLabel(self.left_panel, text="Modes", font=("Georgia", 20, "bold"), text_color=GOLD)
        self.mode_label.pack(pady=(18, 8))
        self.mode_control = ctk.CTkSegmentedButton(self.left_panel, values=["Normal", "Challenge"], command=self.change_mode)
        self.mode_control.set("Normal")
        self.mode_control.pack(padx=16, pady=8, fill="x")

        self.spell_label = ctk.CTkLabel(self.left_panel, text="Spell Command", font=("Georgia", 18, "bold"), text_color="#f3d892")
        self.spell_label.pack(pady=(22, 8))
        self.command_entry = ctk.CTkEntry(self.left_panel, width=260, placeholder_text="knight e5")
        self.command_entry.pack(padx=16, pady=6)
        self.command_entry.bind("<Return>", lambda _event: self.submit_command())
        self.cast_button = ctk.CTkButton(self.left_panel, text="Cast Typed Spell", command=self.submit_command, fg_color="#5c3c23", hover_color="#7a5130")
        self.cast_button.pack(padx=16, pady=6, fill="x")
        self.undo_button = ctk.CTkButton(self.left_panel, text="Undo Last Spell", command=self.undo_last_spell, fg_color="#4d3023", hover_color="#6a4432", state="disabled")
        self.undo_button.pack(padx=16, pady=6, fill="x")
        self.wand_frame = ctk.CTkFrame(self.left_panel, fg_color=PANEL)
        self.wand_frame.pack(padx=16, pady=(10, 4), fill="x")
        self.listen_button = ctk.CTkButton(
            self.wand_frame,
            text="" if self.wand_image else "Sonorus",
            image=self.wand_image,
            command=self.listen_once,
            width=96,
            height=80,
            fg_color=PANEL,
            hover_color="#2d2119",
            border_width=0,
        )
        self.listen_button.pack()
        self.sonorus_label = ctk.CTkLabel(self.wand_frame, text="Sonorus", font=("Georgia", 16, "bold"), text_color="#d7af42")
        self.sonorus_label.pack(pady=(0, 2))
        self.voice_label = ctk.CTkLabel(self.left_panel, text="Voice idle", text_color="#b9cbe2", wraplength=240)
        self.voice_label.pack(padx=16, pady=(2, 16))
        self.listen_progress = ctk.CTkProgressBar(self.left_panel, width=240, height=10, progress_color=GOLD)
        self.listen_progress.set(0)
        self.listen_progress.pack(padx=16, pady=(0, 12))

        self.log_title = ctk.CTkLabel(self.left_panel, text="Game Log", font=("Georgia", 18, "bold"), text_color="#f3d892")
        self.log_title.pack(pady=(8, 8))
        self.log_box = ctk.CTkTextbox(self.left_panel, width=280, height=280, fg_color="#100a08", text_color="#f8e8bd", border_width=1, border_color="#5f4325", font=("Consolas", 13))
        self.log_box.pack(padx=16, pady=(0, 16), fill="both", expand=True)
        self.log_box.tag_config("ok", foreground="#baf2b2")
        self.log_box.tag_config("bad", foreground="#ffb8a8")
        self.log_box.tag_config("ai", foreground="#b9cbe2")
        self.log_box.tag_config("info", foreground="#f8e8bd")
        self.log_box.tag_config("rule", foreground="#d7af42")
        self.log_box.configure(state="disabled")

        self.center_panel = ctk.CTkFrame(self, fg_color="#0d0806", corner_radius=8)
        self.center_panel.grid(row=1, column=1, padx=8, pady=16)
        self.board_view = ChessBoardView(self.center_panel, self.images, self.handle_square_click)
        self.board_view.pack(padx=12, pady=12)

        self.right_panel = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=8, width=330)
        self.right_panel.grid(row=1, column=2, padx=(10, 18), pady=16, sticky="nsew")
        self.right_panel.grid_propagate(False)
        self._build_normal_panel()

    def _clear_right(self) -> None:
        for child in self.right_panel.winfo_children():
            child.destroy()

    def _build_normal_panel(self) -> None:
        self._clear_right()
        ctk.CTkLabel(self.right_panel, text="Normal Duel", font=("Georgia", 22, "bold"), text_color=GOLD).pack(pady=(18, 8))
        ctk.CTkLabel(
            self.right_panel,
            text="You command white. The rival replies with fast casual magic.",
            wraplength=285,
            text_color="#f1dcaa",
        ).pack(padx=18, pady=(0, 16))
        ctk.CTkButton(self.right_panel, text="New Normal Game", command=self.new_normal_game, fg_color="#5c3c23", hover_color="#7a5130").pack(padx=18, pady=8, fill="x")
        ctk.CTkButton(self.right_panel, text="Flip Board", command=self.flip_board, fg_color="#284a64", hover_color="#32607f").pack(padx=18, pady=8, fill="x")
        self.normal_info = ctk.CTkLabel(self.right_panel, text="", wraplength=285, text_color="#f8e8bd", justify="left")
        self.normal_info.pack(padx=18, pady=18, fill="x")

    def _build_challenge_panel(self) -> None:
        self.normal_info = None
        self._clear_right()
        ctk.CTkLabel(self.right_panel, text="Challenge Builder", font=("Georgia", 22, "bold"), text_color=GOLD).pack(pady=(14, 6))

        controls = ctk.CTkScrollableFrame(self.right_panel, fg_color=PANEL_2, corner_radius=8, width=300, height=610)
        controls.pack(padx=14, pady=10, fill="both", expand=True)

        ctk.CTkLabel(controls, text="Piece Palette", font=("Georgia", 16, "bold"), text_color="#f3d892").pack(pady=(10, 6))
        self.palette_color = ctk.CTkSegmentedButton(controls, values=["White", "Black"], command=lambda _v: self.update_editor_piece())
        self.palette_color.set("White")
        self.palette_color.pack(padx=8, pady=4, fill="x")
        self.palette_piece = ctk.CTkOptionMenu(controls, values=["knight", "rook", "bishop", "queen", "king", "pawn"], command=lambda _v: self.update_editor_piece())
        self.palette_piece.set("knight")
        self.palette_piece.pack(padx=8, pady=8, fill="x")
        ctk.CTkLabel(controls, text="Click the board to place. Right choice: Empty removes.", text_color="#d9c48d", wraplength=245).pack(padx=8, pady=(0, 8))
        ctk.CTkButton(controls, text="Use Empty Brush", command=self.use_empty_brush, fg_color="#4d3023", hover_color="#6a4432").pack(padx=8, pady=4, fill="x")
        ctk.CTkButton(controls, text="Clear Board", command=self.clear_editor, fg_color="#6d2d22", hover_color="#8a3a2c").pack(padx=8, pady=4, fill="x")
        ctk.CTkButton(controls, text="Sample Knights vs Rooks", command=self.sample_challenge, fg_color="#5c3c23", hover_color="#7a5130").pack(padx=8, pady=4, fill="x")

        ctk.CTkLabel(controls, text="Side To Move", font=("Georgia", 16, "bold"), text_color="#f3d892").pack(pady=(16, 6))
        self.turn_select = ctk.CTkSegmentedButton(controls, values=["White", "Black"])
        self.turn_select.set("White")
        self.turn_select.pack(padx=8, pady=4, fill="x")

        ctk.CTkLabel(controls, text="Win Condition", font=("Georgia", 16, "bold"), text_color="#f3d892").pack(pady=(16, 6))
        self.rule_kind = ctk.CTkOptionMenu(controls, values=["capture", "eliminate", "checkmate"])
        self.rule_kind.set("capture")
        self.rule_kind.pack(padx=8, pady=4, fill="x")
        self.attacker_select = ctk.CTkOptionMenu(controls, values=["knight", "rook", "bishop", "queen", "king", "pawn"])
        self.attacker_select.set("knight")
        self.attacker_select.pack(padx=8, pady=4, fill="x")
        self.target_select = ctk.CTkOptionMenu(controls, values=["rook", "knight", "bishop", "queen", "king", "pawn"])
        self.target_select.set("rook")
        self.target_select.pack(padx=8, pady=4, fill="x")
        self.target_color = ctk.CTkOptionMenu(controls, values=["Black", "White", "Either"])
        self.target_color.set("Black")
        self.target_color.pack(padx=8, pady=4, fill="x")

        ctk.CTkLabel(controls, text="Victory Message", font=("Georgia", 16, "bold"), text_color="#f3d892").pack(pady=(16, 6))
        self.victory_entry = ctk.CTkEntry(controls, placeholder_text="Mischief managed.")
        self.victory_entry.insert(0, "Mischief managed. You solved the charm.")
        self.victory_entry.pack(padx=8, pady=4, fill="x")
        ctk.CTkButton(controls, text="Start Challenge", command=self.start_challenge, fg_color="#2d5c39", hover_color="#3a7a4a").pack(padx=8, pady=(16, 6), fill="x")
        ctk.CTkButton(controls, text="Retry Challenge", command=self.retry_challenge, fg_color="#5c3c23", hover_color="#7a5130").pack(padx=8, pady=6, fill="x")

    def change_mode(self, value: str) -> None:
        self.active_mode = "normal" if value == "Normal" else "challenge"
        self.selected_square = None
        self.ambiguous_from = set()
        if self.active_mode == "normal":
            self._build_normal_panel()
        else:
            self._build_challenge_panel()
            self.refresh_editor_board("Design a challenge, then press Start Challenge.")
            return
        self.refresh_all("Mode changed.")

    def log(self, text: str, kind: str = "info") -> None:
        prefix = {
            "ok": "[YOU]",
            "ai": "[RIVAL]",
            "bad": "[INVALID]",
            "rule": "[EVENT]",
            "info": "[INFO]",
        }.get(kind, "[INFO]")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", "-" * 31 + "\n", "rule")
        self.log_box.insert("end", f"{prefix} ", kind)
        self.log_box.insert("end", text + "\n", kind)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, text: str, good: Optional[bool] = None) -> None:
        color = "#f4dfaa"
        if good is True:
            color = "#baf2b2"
        elif good is False:
            color = "#ffb8a8"
        self.status_label.configure(text=text, text_color=color)

    def current_board(self) -> chess.Board:
        if self.active_mode == "normal":
            return self.normal.board
        if self.special.start_fen != "8/8/8/8/8/8/8/8" or self.special.board.piece_map():
            return self.special.board
        return self.editor_board

    def refresh_all(self, status: str = "") -> None:
        board = self.normal.board if self.active_mode == "normal" else self.special.board
        self.board_view.set_board(
            board,
            selected=self.selected_square,
            legal_targets=self.legal_targets_for_selected(board),
            highlight_from=self.ambiguous_from,
            last_move=self.last_move,
        )
        if status:
            self.set_status(status)
        if self.active_mode == "normal":
            self.update_normal_info()

    def refresh_editor_board(self, status: str = "") -> None:
        self.board_view.set_board(self.editor_board, selected=None, legal_targets=set(), highlight_from=set(), last_move=None)
        if status:
            self.set_status(status)

    def legal_targets_for_selected(self, board: chess.Board) -> set[chess.Square]:
        if self.selected_square is None:
            return set()
        if self.active_mode == "normal":
            return {move.to_square for move in board.legal_moves if move.from_square == self.selected_square}
        return {move.to_square for move in self.special.pseudo_moves() if move.from_square == self.selected_square}

    def update_normal_info(self) -> None:
        if not getattr(self, "normal_info", None) or not self.normal_info.winfo_exists():
            return
        board = self.normal.board
        text = [
            f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}",
            f"Move: {board.fullmove_number}",
        ]
        if board.is_check():
            text.append("Check is on the board.")
        if board.is_game_over():
            text.append(f"Result: {board.result()}")
            text.append(board.outcome().termination.name.replace("_", " ").title())
        self.normal_info.configure(text="\n".join(text))

    def handle_square_click(self, square: chess.Square) -> None:
        if self.active_mode == "challenge" and not self.challenge_started:
            self.editor_click(square)
            return

        board = self.normal.board if self.active_mode == "normal" else self.special.board
        piece = board.piece_at(square)
        if self.selected_square is None:
            if piece and piece.color == board.turn and piece.color == chess.WHITE:
                self.selected_square = square
                self.ambiguous_from = set()
                self.refresh_all(f"Selected {PIECE_NAMES[piece.piece_type]} on {square_name(square)}.")
            return

        origin = self.selected_square
        self.selected_square = None
        snapshot = self.capture_undo_snapshot()
        if self.active_mode == "normal":
            result = self.normal.click_move(origin, square)
        else:
            result = self.special.click_move(origin, square)
        self.after_player_move(result, snapshot)

    def editor_click(self, square: chess.Square) -> None:
        if self.editor_piece is None:
            self.editor_board.remove_piece_at(square)
        else:
            self.editor_board.set_piece_at(square, self.editor_piece)
        self.refresh_editor_board(f"Edited {square_name(square)}.")

    def update_editor_piece(self) -> None:
        color = chess.WHITE if self.palette_color.get() == "White" else chess.BLACK
        piece_type = NAME_TO_PIECE[self.palette_piece.get()]
        self.editor_piece = chess.Piece(piece_type, color)

    def use_empty_brush(self) -> None:
        self.editor_piece = None
        self.set_status("Empty brush selected. Click a square to remove a piece.")

    def clear_editor(self) -> None:
        self.editor_board.clear()
        self.special = SpecialGame()
        self.challenge_started = False
        self.refresh_editor_board("Challenge board cleared.")

    def sample_challenge(self) -> None:
        self.editor_board.clear()
        self.challenge_started = False
        for sq in ("b1", "d1", "f1"):
            self.editor_board.set_piece_at(chess.parse_square(sq), chess.Piece(chess.KNIGHT, chess.WHITE))
        for sq in ("b8", "d8", "f8"):
            self.editor_board.set_piece_at(chess.parse_square(sq), chess.Piece(chess.ROOK, chess.BLACK))
        self.refresh_editor_board("Sample challenge loaded.")

    def build_special_rule(self) -> SpecialRule:
        target_color_value = self.target_color.get()
        target_color = None
        if target_color_value == "White":
            target_color = chess.WHITE
        elif target_color_value == "Black":
            target_color = chess.BLACK
        message = self.victory_entry.get().strip() or "Mischief managed."
        return SpecialRule(
            kind=self.rule_kind.get(),
            attacker=NAME_TO_PIECE[self.attacker_select.get()],
            target=NAME_TO_PIECE[self.target_select.get()],
            target_color=target_color,
            message=message,
        )

    def start_challenge(self) -> None:
        if not self.editor_board.piece_map():
            self.sample_challenge()
        turn = chess.WHITE if self.turn_select.get() == "White" else chess.BLACK
        board = self.editor_board.copy(stack=False)
        board.turn = turn
        self.special.set_from_editor(board, turn, self.build_special_rule())
        self.selected_square = None
        self.last_move = None
        self.ambiguous_from = set()
        self.challenge_started = True
        self.undo_snapshot = None
        self.undo_button.configure(state="disabled")
        self.refresh_all("Challenge started. White is your side.")
        self.log("Challenge started.", "rule")
        if self.special.board.turn == chess.BLACK:
            self.ai_after_id = self.after(450, self.run_ai_turn)

    def retry_challenge(self) -> None:
        self.special.reset()
        self.selected_square = None
        self.last_move = None
        self.ambiguous_from = set()
        self.challenge_started = True
        self.undo_snapshot = None
        self.undo_button.configure(state="disabled")
        self.refresh_all("Challenge reset.")

    def capture_undo_snapshot(self) -> dict:
        return {
            "mode": self.active_mode,
            "normal_board": self.normal.board.copy(stack=False),
            "special_board": self.special.board.copy(stack=False),
            "special_game_over": self.special.game_over,
            "challenge_started": self.challenge_started,
            "last_move": self.last_move,
        }

    def undo_last_spell(self) -> None:
        if not self.undo_snapshot:
            self.set_status("Undo is available once after a spell.", False)
            return
        if self.ai_after_id:
            try:
                self.after_cancel(self.ai_after_id)
            except Exception:
                pass
            self.ai_after_id = None

        snapshot = self.undo_snapshot
        self.normal.board = snapshot["normal_board"].copy(stack=False)
        self.special.board = snapshot["special_board"].copy(stack=False)
        self.special.game_over = snapshot["special_game_over"]
        self.challenge_started = snapshot["challenge_started"]
        self.last_move = snapshot["last_move"]
        self.selected_square = None
        self.ambiguous_from = set()
        self.undo_snapshot = None
        self.undo_button.configure(state="disabled")
        self.refresh_all("Undone. One-use rewind spent.")
        self.log("Last spell rewound. Undo is now spent.", "rule")

    def submit_command(self) -> None:
        command = self.command_entry.get().strip()
        if not command:
            self.set_status("Type a spell first.", False)
            return
        self.command_entry.delete(0, "end")
        if self.active_mode == "challenge" and not self.challenge_started:
            message = f"{command} was not cast because the challenge has not started. Press Start Challenge first."
            self.set_status(message, False)
            self.log(message, "bad")
            self.refresh_editor_board()
            return
        snapshot = self.capture_undo_snapshot()
        if self.active_mode == "normal":
            result = self.normal.command_move(command)
        else:
            result = self.special.command_move(command)
        self.after_player_move(result, snapshot)

    def after_player_move(self, result: MoveResult, snapshot: Optional[dict] = None) -> None:
        self.ambiguous_from = set(result.ambiguous_from)
        if not result.ok:
            self.set_status(result.message, False)
            self.log(result.message, "bad")
            self.refresh_all()
            return
        if snapshot:
            self.undo_snapshot = snapshot
            self.undo_button.configure(state="normal")
        self.last_move = result.move
        self.set_status(result.message, result.victory if result.game_over else True)
        self.log(result.message, "ok")
        self.refresh_all()

        if result.game_over:
            self.show_end_modal(result.victory, result.message)
            return
        self.ai_after_id = self.after(550, self.run_ai_turn)

    def run_ai_turn(self) -> None:
        if self.active_mode == "normal":
            if self.normal.board.turn != chess.BLACK or self.normal.board.is_game_over():
                return
            self.set_status("The rival studies the board...")
            result = self.normal.ai_move()
        else:
            if self.special.board.turn != chess.BLACK or self.special.game_over:
                return
            self.set_status("The rival studies the challenge...")
            result = self.special.ai_move()

        if result.ok:
            self.last_move = result.move
            self.log(result.message, "ai")
            self.set_status(result.message, not result.game_over)
        else:
            self.log(result.message, "bad")
            self.set_status(result.message, False)
        self.refresh_all()

        if result.game_over:
            self.show_end_modal(result.victory, result.message)

    def listen_once(self) -> None:
        if self.listening:
            return
        if sr is None:
            self.voice_label.configure(text="SpeechRecognition is not installed. Type the spell instead.")
            return
        self.listening = True
        self.listen_started_at = time.monotonic()
        self.listen_progress.set(0)
        self.listen_button.configure(state="disabled", fg_color="#6d4b1f")
        self.sonorus_label.configure(text="Sonorus: listening")
        mode = "offline Whisper" if self.can_use_local_whisper() else ("OpenAI" if self.can_use_openai_transcribe() else "local")
        self.voice_label.configure(text=f"Raise the wand and speak. {self.listen_duration_seconds}s left, then {mode} transcribes.")
        self.update_listen_timer()
        thread = threading.Thread(target=self._listen_worker, daemon=True)
        thread.start()

    def safe_voice_status(self, text: str, progress: Optional[float] = None) -> None:
        if self.closing:
            return
        def apply_status() -> None:
            if self.closing:
                return
            self.voice_label.configure(text=text)
            if progress is not None:
                self.listen_progress.set(max(0, min(1, progress)))
        try:
            self.after(0, apply_status)
        except Exception:
            pass

    def update_listen_timer(self) -> None:
        if not self.listening or self.closing:
            return
        elapsed = time.monotonic() - self.listen_started_at
        progress = min(1, elapsed / self.listen_duration_seconds)
        remaining = max(0, self.listen_duration_seconds - int(elapsed))
        self.listen_progress.set(progress)
        self.voice_label.configure(text=f"Listening... {remaining}s left.")
        if progress < 1:
            self.listen_timer_after_id = self.after(250, self.update_listen_timer)

    def can_use_openai_transcribe(self) -> bool:
        return USE_OPENAI_TRANSCRIBE and OpenAI is not None and bool(os.environ.get("OPENAI_API_KEY"))

    def can_use_local_whisper(self) -> bool:
        return whisper is not None

    def local_whisper_device(self) -> str:
        if LOCAL_WHISPER_DEVICE != "auto":
            return LOCAL_WHISPER_DEVICE
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def transcribe_with_openai(self, audio) -> str:
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wav_path = temp_file.name
                temp_file.write(audio.get_wav_data())
            client = OpenAI()
            with open(wav_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model=TRANSCRIBE_MODEL,
                    file=audio_file,
                    prompt=CHESS_TRANSCRIBE_PROMPT,
                )
            return transcription.text.strip()
        finally:
            if wav_path:
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def transcribe_with_local_whisper(self, audio) -> str:
        global LOCAL_WHISPER_CACHE, LOCAL_WHISPER_CACHE_DEVICE
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wav_path = temp_file.name
                temp_file.write(audio.get_wav_data())
            device = self.local_whisper_device()
            if LOCAL_WHISPER_CACHE is None or LOCAL_WHISPER_CACHE_DEVICE != device:
                self.safe_voice_status(f"Loading Whisper {LOCAL_WHISPER_MODEL} on {device}. First run can take a bit...", 1)
                LOCAL_WHISPER_CACHE = whisper.load_model(LOCAL_WHISPER_MODEL, device=device)
                LOCAL_WHISPER_CACHE_DEVICE = device
            self.safe_voice_status(f"Transcribing with Whisper {LOCAL_WHISPER_MODEL} on {device}...", 1)
            result = LOCAL_WHISPER_CACHE.transcribe(
                wav_path,
                language="en",
                fp16=device == "cuda",
                initial_prompt=CHESS_TRANSCRIBE_PROMPT,
            )
            return result.get("text", "").strip()
        finally:
            if wav_path:
                try:
                    Path(wav_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _listen_worker(self) -> None:
        text = ""
        error = ""
        try:
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=self.listen_duration_seconds)
            self.safe_voice_status("Recording captured. Preparing transcription...", 1)
            if self.can_use_local_whisper():
                try:
                    text = self.transcribe_with_local_whisper(audio)
                except Exception as exc:
                    error = f"Offline Whisper failed; local fallback used. {exc}"
                    self.safe_voice_status("Offline Whisper failed. Trying local speech fallback...", 1)
                    text = recognizer.recognize_google(audio)
            elif self.can_use_openai_transcribe():
                try:
                    self.safe_voice_status("Sending audio to OpenAI transcription...", 1)
                    text = self.transcribe_with_openai(audio)
                except Exception as exc:
                    error = f"OpenAI transcription failed; local fallback used. {exc}"
                    self.safe_voice_status("OpenAI transcription failed. Trying local speech fallback...", 1)
                    text = recognizer.recognize_google(audio)
            else:
                self.safe_voice_status("Transcribing with local speech fallback...", 1)
                text = recognizer.recognize_google(audio)
        except Exception as exc:
            error = error or str(exc)
        if not self.closing:
            try:
                self.after(0, lambda: self._listen_finished(text, error))
            except Exception:
                pass

    def _listen_finished(self, text: str, error: str) -> None:
        if self.closing:
            return
        self.listening = False
        if self.listen_timer_after_id:
            try:
                self.after_cancel(self.listen_timer_after_id)
            except Exception:
                pass
            self.listen_timer_after_id = None
        self.listen_progress.set(0)
        self.listen_button.configure(state="normal", fg_color=PANEL)
        self.sonorus_label.configure(text="Sonorus")
        if text:
            self.voice_label.configure(text=f"Heard: {text}")
            self.command_entry.delete(0, "end")
            self.command_entry.insert(0, text)
            self.submit_command()
        else:
            self.voice_label.configure(text=f"Voice spell failed. Type it instead. {error[:70]}")

    def new_normal_game(self) -> None:
        self.normal.reset()
        self.selected_square = None
        self.last_move = None
        self.ambiguous_from = set()
        self.undo_snapshot = None
        self.undo_button.configure(state="disabled")
        self.refresh_all("New normal duel started.")
        self.log("New normal duel started.", "rule")

    def restart_current(self) -> None:
        if self.active_mode == "normal":
            self.new_normal_game()
        else:
            self.retry_challenge()

    def flip_board(self) -> None:
        self.board_view.flipped = not self.board_view.flipped
        self.refresh_all("Board flipped.")

    def toggle_fullscreen(self) -> None:
        self.set_fullscreen(not self.fullscreen)

    def set_fullscreen(self, value: bool) -> None:
        self.fullscreen = value
        self.attributes("-fullscreen", value)
        if value:
            self.title_bar.grid_remove()
            self.right_panel.grid_remove()
            for widget in (
                self.mode_label,
                self.mode_control,
            ):
                widget.pack_forget()
            self.log_title.pack_forget()
            self.log_box.pack_forget()
            self.spell_label.pack(pady=(18, 8))
            self.command_entry.pack(padx=16, pady=6)
            self.cast_button.pack(padx=16, pady=6, fill="x")
            self.undo_button.pack(padx=16, pady=6, fill="x")
            self.wand_frame.pack(padx=16, pady=(10, 4), fill="x")
            self.voice_label.pack(padx=16, pady=(2, 14))
            self.listen_progress.pack(padx=16, pady=(0, 12))
            self.log_title.pack(pady=(8, 8))
            self.log_box.pack(padx=16, pady=(0, 16), fill="both", expand=True)
            self.center_panel.grid_configure(row=1, column=0, columnspan=2, padx=(18, 8), pady=12)
            self.left_panel.grid_configure(row=1, column=2, padx=(8, 18), pady=12, sticky="nsew")
            self.log_box.configure(width=340, height=430)
        else:
            self.title_bar.grid(row=0, column=0, columnspan=3, sticky="ew")
            self.right_panel.grid(row=1, column=2, padx=(10, 18), pady=16, sticky="nsew")
            self.log_title.pack_forget()
            self.log_box.pack_forget()
            self.mode_label.pack(pady=(18, 8))
            self.mode_control.pack(padx=16, pady=8, fill="x")
            self.spell_label.pack(pady=(22, 8))
            self.command_entry.pack(padx=16, pady=6)
            self.cast_button.pack(padx=16, pady=6, fill="x")
            self.undo_button.pack(padx=16, pady=6, fill="x")
            self.wand_frame.pack(padx=16, pady=(10, 4), fill="x")
            self.voice_label.pack(padx=16, pady=(2, 16))
            self.listen_progress.pack(padx=16, pady=(0, 12))
            self.log_title.pack(pady=(8, 8))
            self.log_box.pack(padx=16, pady=(0, 16), fill="both", expand=True)
            self.left_panel.grid_configure(row=1, column=0, padx=(18, 10), pady=16, sticky="ns")
            self.center_panel.grid_configure(row=1, column=1, columnspan=1, padx=8, pady=16)
            self.log_box.configure(width=280, height=280)

    def close_app(self) -> None:
        self.closing = True
        if self.listen_timer_after_id:
            try:
                self.after_cancel(self.listen_timer_after_id)
            except Exception:
                pass
            self.listen_timer_after_id = None
        if self.ai_after_id:
            try:
                self.after_cancel(self.ai_after_id)
            except Exception:
                pass
            self.ai_after_id = None
        self.destroy()

    def show_end_modal(self, victory: bool, message: str) -> None:
        modal = ctk.CTkToplevel(self)
        modal.title("Duel Complete")
        modal.geometry("430x260")
        modal.transient(self)
        modal.grab_set()
        modal.configure(fg_color="#160f0b")
        title = "Mischief Managed" if victory else "The Duel Is Lost"
        body = message
        if victory and "checkmate" not in message.lower():
            body = f"{message}\n\nThe Great Hall erupts. Your spellwork wins the board."
        elif not victory and "checkmate" not in message.lower():
            body = f"{message}\n\nThe board resets, and the portraits are already whispering about a rematch."
        ctk.CTkLabel(modal, text=title, font=("Georgia", 28, "bold"), text_color=GOOD if victory else BAD).pack(pady=(24, 10))
        ctk.CTkLabel(modal, text=body, wraplength=360, text_color="#f7e7bd", font=("Segoe UI", 15)).pack(padx=26, pady=12)
        row = ctk.CTkFrame(modal, fg_color="transparent")
        row.pack(pady=18)
        ctk.CTkButton(row, text="Retry", command=lambda: (modal.destroy(), self.restart_current()), fg_color="#5c3c23", hover_color="#7a5130").grid(row=0, column=0, padx=8)
        ctk.CTkButton(row, text="Close", command=modal.destroy, fg_color="#284a64", hover_color="#32607f").grid(row=0, column=1, padx=8)


def main() -> None:
    app = WizardChessApp()
    if "--smoke-close" in sys.argv:
        app.after(150, app.destroy)
    app.mainloop()


if __name__ == "__main__":
    main()
