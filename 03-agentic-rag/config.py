"""Settings shared by every step of this project.

Change the model in ONE place and every step picks it up.
"""

import os

# Run step0_list_models.py to see which model names exist. Careful: that list
# can include retired models. gemini-2.5-flash was listed but returned 404 "no
# longer available to new users", and the error named this replacement.
# Model names change over time, so do not trust an old tutorial's name.
# To override without editing this file:
#   $env:GEMINI_MODEL = "some-model-name"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
