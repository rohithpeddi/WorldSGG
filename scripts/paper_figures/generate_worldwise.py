"""Render the WorldWise v2e architecture."""
from common import parser, worldwise

if __name__ == "__main__":
    worldwise().save("worldwise", parser(__doc__).parse_args())
