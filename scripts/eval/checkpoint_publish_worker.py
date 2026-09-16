#!/usr/bin/env python3
"""Attempt one generation-checked event-stream checkpoint publication."""
from __future__ import annotations
import argparse,json
from sara_engine.learning.event_stream_engine import BoundedEventStreamEngine
def main():
    parser=argparse.ArgumentParser();parser.add_argument("--filename",required=True);parser.add_argument("--expected-generation",type=int,required=True);args=parser.parse_args()
    try:
        engine,_=BoundedEventStreamEngine.load_with_generation(args.filename);receipt=engine.publish(args.filename,expected_generation=args.expected_generation)
        print(json.dumps({"published":True,"generation":receipt.generation}));return 0
    except ValueError as error:
        print(json.dumps({"published":False,"error":str(error)}));return 0
if __name__=="__main__":raise SystemExit(main())
