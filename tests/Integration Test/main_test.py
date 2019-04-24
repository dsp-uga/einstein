import argparse
from einstein.__main__ import run, get_parser


def test_get_parser():
    parser = get_parser()
    # Testing if `parser` is an ArgumentParser
    assert isinstance(parameter_dict, ArgumentParser()) == True

def test_run():
    output = run()
    # Testing that the model summary table is being printed
    assert len(str(output)) > 0