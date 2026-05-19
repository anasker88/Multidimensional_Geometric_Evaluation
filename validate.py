import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
	runpy.run_module("cli.validate", run_name="__main__")
