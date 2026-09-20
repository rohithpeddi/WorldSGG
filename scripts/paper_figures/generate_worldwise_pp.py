"""Render the WorldWise++ entity decoder and joint detection."""
from common import parser, worldwise_pp

if __name__ == "__main__":
    worldwise_pp().save("worldwise_pp", parser(__doc__).parse_args())
