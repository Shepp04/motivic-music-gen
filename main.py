
import random
import music21 as m21
from src.motifgen import pcfg, ngram, realise

if __name__ == "__main__":
    # 0. Config
    seed = 42
    num_bars = 4

    # 1. Generate a phrase plan
    grammar = pcfg.make_minimal_mvp_grammar()
    cfg = pcfg.SamplerConfig(seed=seed)
    events = grammar.sample_plan(num_bars=num_bars, cfg=cfg)

    # 2. Train the harmony ngram and sample functional (T | D | PD) and roman numeral plans
    pieces = [{"id": "bach/bwv8.6", "key": "B major", "melody_tokens": ["BAR", "R:1.0", "N:0:0.5", "N:0:0.5", "N:5:1.0", "N:0:1.0", "BAR", "N:2:1.0", "N:0:1.0", "N:10:2.0", "N:0:0.5", "N:10:0.5", "BAR", "N:9:1.0", "N:7:1.0", "R:1.0", "N:0:0.5", "N:10:0.5", "BAR", "N:9:1.0", "N:2:0.5", "N:0:0.5", "N:11:0.5", "N:7:0.5", "N:0:1.0", "BAR", "N:0:1.0", "N:11:1.0", "N:0:2.0", "BAR", "N:0:1.0", "N:11:1.0", "N:0:2.0", "BAR", "R:1.0", "N:7:0.5", "N:9:0.5", "N:10:1.0", "N:9:1.0", "BAR", "N:2:2.0", "N:4:0.5", "N:1:2.0", "BAR", "R:1.0", "N:2:0.5", "N:0:0.5", "N:11:1.0", "N:0:1.0", "BAR", "N:0:1.0", "N:11:1.0", "N:0:1.0", "N:7:0.5", "N:7:0.5", "BAR", "N:0:2.0", "N:10:0.5", "N:9:1.0", "N:2:1.0", "BAR", "N:1:1.0", "N:2:2.0", "N:1:1.0", "BAR", "N:2:2.0", "R:1.0", "N:5:0.5", "N:0:0.5", "BAR", "N:2:1.0", "N:9:0.5", "N:10:0.5", "N:0:2.0", "N:10:0.5", "BAR", "N:9:1.0", "N:7:0.5", "N:5:0.5", "N:4:1.0", "N:5:1.0", "BAR", "N:5:1.0", "N:4:1.0", "N:5:2.0"], "harmony_tokens": ["iv", "iv", "i", "iv", "IV", "bVII", "IV", "v75b3", "I6b5", "IV", "I742", "iii", "iiiob5", "IV", "IV7", "iii7", "I", "i54", "V", "I", "I", "ii652", "V", "I", "I532", "iii", "vi6", "vb3", "vi43", "iiiø7b53", "iiio6", "VI", "VI6#42", "iv", "IV7", "V42", "I752", "i4", "V", "I", "v4", "I6", "Ib753", "vi43", "ii42", "VI", "#vø7", "ii54", "VI", "ii", "#i7b2", "ii", "IV6b5", "ii42", "bVII6", "IV6", "I6b5", "IV", "bvii", "I42", "viiob753", "iv4", "I", "IV"]},
                {"id": "bach/bwv67.7", "key": "A major", "melody_tokens": ["BAR", "N:4:1.0", "N:0:1.0", "N:2:1.0", "N:4:1.0", "BAR", "N:7:1.0", "N:5:1.0", "N:5:1.0", "N:4:1.0", "BAR", "N:7:1.0", "N:5:1.0", "N:4:1.0", "N:2:1.0", "BAR", "N:2:1.0", "N:4:2.0", "BAR", "N:2:1.0", "N:2:1.0", "N:2:1.0", "N:4:1.0", "BAR", "N:2:1.0", "N:0:1.0", "N:2:1.0", "N:11:1.0", "BAR", "N:11:1.0", "N:0:1.0", "N:2:1.0", "N:4:1.0", "BAR", "N:2:0.5", "N:4:0.5", "N:5:1.0", "N:4:1.0", "N:2:1.0", "BAR", "N:0:2.0", "R:1.0"], "harmony_tokens": ["I", "I", "IV42", "viio6", "I", "V6", "IV6", "V65", "I", "I6", "IV752", "I", "ii65", "V", "I", "I", "I", "V742", "V6", "V", "I", "V7", "vi", "viio6", "III", "III", "vi", "V", "I", "iii64", "IV6", "I64", "ii65", "V7", "I", "I"]}]
    cfg = ngram.HarmonyConfig(num_bars=num_bars, k_func=3, alpha_func=0.25, seed=seed)
    model = ngram.train_harmony_model(pieces, cfg)
    func_plan, rn_plan = ngram.sample_harmony_plan(model)

    # 3. Get the motif
    key = m21.key.Key('C')
    motif_stream = m21.stream.Stream()
    motif_stream.append(m21.note.Note('C4', quarterLength=0.5))
    motif_stream.append(m21.note.Note('E4', quarterLength=0.5))
    motif_stream.append(m21.note.Note('A4', quarterLength=0.5))
    motif_stream.append(m21.note.Note('G4', quarterLength=0.5))

    # 4. Realise into a piece
    score = realise.realise_score(
        key_obj=key,
        motif_stream=motif_stream,
        events=events,
        rn_plan=rn_plan,
        num_bars=4,
        units_per_beat=2,
        beats_per_bar=4,
    )

    # Display the realised piece
    score.show()