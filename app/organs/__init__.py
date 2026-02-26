"""Organ implementations (outside the core equation).

Organs can:
- observe state
- perform IO (web, sensors)
- emit events back into the core (which then apply matrices to the state vector)
"""

from .feedback import interpret_feedback, OllamaConfig as FeedbackConfig
from . import topic
from . import selfreport

