# TODO
* Smooth harmony voice leading - Tonnetz? X
* Disable inversions + extensions (on the tonic) from cadences X
* Add a mode setting - only use scale degrees from the mode + suitable chords X
* Melody infilling using a second ngram model X
* Fitting the melody to the harmony X
* Separating major and minor pieces X

ISSUES
* melody infilling after cadence X
* Harmony is weird e.g. flats or 7th chords X
* Durations of motif aren't represented (because 0.25 is too short?) X
* Need to establish major or minor (in the training data and when generating a piece) X
* This could be future work: non-diatonic harmony.

MVP FEATURES
* Render in musescore 4 X
* Set certain instrument combos [C]
* Vary the accompaniment with alberti bass, arpeggiation, block chords X
* Make cadences robust X

THINGS TO MENTION IN REPORT
* Issue with leading tones in minor key, have to raise the 6th and 7th in both motif and infilled melody
* Representation of motif
* Octave wrap-around and avoiding large leaps in melody
* T, PD, D function for realistic harmony
* Cadence handling

NOTES
"the grammar generates symbolic form + development operators; the scheduler enforces temporal coverage constraints to prevent degenerate long infill stretches."