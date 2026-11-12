"""Class order and canonical score range used across service boundaries."""

CHOICES = ("A", "B", "C", "D", "E", "F")
SCORE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
SCORE_MIN = SCORE_LEVELS[0]
SCORE_MAX = SCORE_LEVELS[-1]
OFF_TOPIC = len(SCORE_LEVELS)
