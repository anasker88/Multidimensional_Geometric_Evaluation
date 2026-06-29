# common/

Shared utilities used by every experiment (evaluation, patching, legacy SAE).

- **`prompting.py`** — prompt construction (multiple-choice / numeric / numeric-MC),
  chat-template application, and the cyclic answer-rotation mechanism. Imported as
  `from common.prompting import …`.
