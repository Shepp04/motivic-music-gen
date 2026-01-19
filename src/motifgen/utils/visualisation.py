import music21 as m21

def show_score(score: m21.stream.Score, listenable: bool=False) -> None:
    """Display a music21 score, optionally as a MIDI playback."""
    score.show()
    if listenable:
        score.show('midi')