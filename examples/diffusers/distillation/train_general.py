#!/usr/bin/env python3
"""Thin entry point for ``accelerate launch``.

``src/run.py`` uses relative imports so it cannot be invoked as a plain
script.  This wrapper sits one level up and delegates via an absolute import:

    accelerate launch train_general.py --config configs/my_config.yaml
"""

from src.run import main

if __name__ == "__main__":
    main()
