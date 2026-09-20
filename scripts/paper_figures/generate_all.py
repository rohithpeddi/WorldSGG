"""Render all three WorldWise architecture figures."""
from common import parser, worldwise, worldwise_plus, worldwise_pp

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    for name, build in [("worldwise", worldwise), ("worldwise_plus", worldwise_plus),
                        ("worldwise_pp", worldwise_pp)]:
        build().save(name, args)
