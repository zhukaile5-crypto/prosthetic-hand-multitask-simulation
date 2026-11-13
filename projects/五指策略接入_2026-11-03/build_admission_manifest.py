"""Convenient entry point for the independent evidence-chain reviewer.

All signing checks and historical manifest preservation live in the independently
maintained review/independent_admission.py. This wrapper cannot bypass a gate.
"""
from pathlib import Path
import runpy

if __name__=='__main__':
    runpy.run_path(str(Path(__file__).resolve().parent/'review/independent_admission.py'),run_name='__main__')
