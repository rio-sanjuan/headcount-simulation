"""Shared pytest configuration."""

import os

from hypothesis import HealthCheck, settings

# `dev` is the default. CI selects `ci` through HYPOTHESIS_PROFILE: more examples, no deadline
# (shared runners are noisy), and a reproduction blob printed on failure.
settings.register_profile("dev", max_examples=50)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))
