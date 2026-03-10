# |---------------------------------------------------------|
# |                                                         |
# |                 Give Feedback / Get Help                |
# | https://github.com/getbindu/Bindu/issues/new/choose    |
# |                                                         |
# |---------------------------------------------------------|
#
#  Thank you users! We ❤️ you! - 🌻

"""health-insights-agent - HIA Medical Analysis Agent using Bindu Framework."""

from health_insights_agent.__version__ import __version__
from health_insights_agent.main import (
    handler,
    initialize_agent,
    initialize_all,
    main,
)

__all__ = [
    "__version__",
    "handler",
    "initialize_agent",
    "initialize_all",
    "main",
]
