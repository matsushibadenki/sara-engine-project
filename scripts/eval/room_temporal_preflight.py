#!/usr/bin/env python3
"""One-shot registration, execution and independent replay of PIR-only task."""
import argparse
from sara_engine.evaluation.room_temporal import main
if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
