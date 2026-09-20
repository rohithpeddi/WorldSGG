"""Render the WorldWise+ representation swap."""
from common import parser, worldwise_plus

if __name__ == "__main__":
    worldwise_plus().save("worldwise_plus", parser(__doc__).parse_args())
