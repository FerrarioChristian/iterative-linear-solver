import argparse

def parse_arguments():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-iter", type=int, default=20000, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="Skip matrix properties check"
    )
    return parser.parse_args()

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'