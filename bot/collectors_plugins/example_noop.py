"""Example collector plugin.

Plugin contract:
- expose collect(bot) -> int
- bot is AutonomousLearningBot instance
- return number of newly collected samples
"""


def collect(bot) -> int:
    # Keep as no-op example. Implement custom collection logic here.
    return 0
