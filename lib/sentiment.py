"""Class order and canonical score range used across service boundaries."""

CHOICES = ("A", "B", "C", "D", "E", "F")
SCORE_LEVELS = (0.0, 1.25, 2.5, 3.75, 5.0)
SCORE_MIN = SCORE_LEVELS[0]
SCORE_MAX = SCORE_LEVELS[-1]
OFF_TOPIC = len(SCORE_LEVELS)
