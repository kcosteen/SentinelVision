"""Choosing thresholds from data instead of guessing them.

Several constants in this project were picked by eye -- EAR 0.20, phone
confidence 0.5, gaze 0.35/0.65. Each one is a decision boundary, and a boundary
you can't justify is one you can't defend. This package measures them against
public labelled data and reports the operating point, so the number in the code
comes with evidence attached.
"""
