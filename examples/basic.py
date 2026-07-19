"""Resolve and print secret names (not values) for the token in $SEEKRIT_TOKEN.

    export SEEKRIT_TOKEN=skt_...
    python examples/basic.py
"""

import seekrit


def main() -> None:
    client = seekrit.Client()
    secrets = client.resolve()
    print(f"resolved {len(secrets)} secret(s):")
    for name in sorted(secrets):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
