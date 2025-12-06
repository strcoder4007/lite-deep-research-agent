try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
