from config import APP_TITLE, SEPARATOR
from agent import agent, new_session


def main() -> None:
    print(SEPARATOR)
    print(APP_TITLE)
    print("Type 'exit' or 'quit' to leave. Type 'new' to reset the session.")
    print(SEPARATOR)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "new":
            new_session()
            print("Started a new session.")
            continue

        try:
            result = agent.run(user_input)
            print(f"\nAgent: {result}")
        except Exception as exc:
            print(f"\nAgent error: {exc}")


if __name__ == "__main__":
    main()
