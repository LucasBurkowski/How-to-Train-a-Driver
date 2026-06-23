"""FRC Reinforcement Learning Driver package."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("frc-rl-driver")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"
