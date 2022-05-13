from pathlib import Path
import sys

import pytest

if __name__ == "__main__":
    print(sys.argv)
    dirname = Path(__file__).absolute().parent
    sys.exit(pytest.main([str(dirname)].extend(sys.argv[1:])))
