default:
    just --list

dev:
    uv run uvicorn main:app --host 127.0.0.1 --port 7932 --reload --reload-include main.py
