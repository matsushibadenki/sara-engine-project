#!/usr/bin/env python3
"""Register, execute once, or independently replay the room-motion screen."""
import argparse
from sara_engine.evaluation.motion_temporal import main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('register','execute','replay'))
    main(parser.parse_args().mode)
