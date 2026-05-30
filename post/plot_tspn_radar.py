#!/usr/bin/env python3
"""Compatibility entry for demo TSPN visual analysis.

The former radar script depended on historical SEU paths. The TSPN-only demo
uses the scripted analysis pipeline as the single post-training visual path.
"""

from demo_tspn_analysis import main


if __name__ == "__main__":
    main()
