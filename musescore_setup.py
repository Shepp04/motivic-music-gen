import music21 as m21

us = m21.environment.UserSettings()

# macOS (typical)
us["musicxmlPath"] = "/Applications/MuseScore 4.app"
us["musescoreDirectPNGPath"] = "/Applications/MuseScore 4.app"

print("musicxmlPath =", us["musicxmlPath"])
print("musescoreDirectPNGPath =", us["musescoreDirectPNGPath"])