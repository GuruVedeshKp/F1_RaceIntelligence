from __future__ import annotations

import httpx

BASE_URL = "https://api.openf1.org/v1"


def main() -> None:
    """Find the 2024 Bahrain Grand Prix race session."""

    params = {
        "year": 2024,
        "country_name": "Bahrain",
        "session_name": "Race",
    }

    try:
        with httpx.Client(
            base_url=BASE_URL,
            timeout=30.0,
            headers={"User-Agent": "F1-Race-Intelligence/0.1"},
        ) as client:
            response = client.get("/sessions", params=params)
            response.raise_for_status()
            sessions = response.json()

        if not sessions:
            print("No matching race session found.")
            return

        print("OpenF1 API connection successful!")
        print(f"Sessions found: {len(sessions)}")

        for session in sessions:
            print("-" * 45)
            print(f"Grand Prix: {session.get('country_name')}")
            print(f"Circuit: {session.get('circuit_short_name')}")
            print(f"Session: {session.get('session_name')}")
            print(f"Date: {session.get('date_start')}")
            print(f"Meeting key: {session.get('meeting_key')}")
            print(f"Session key: {session.get('session_key')}")

    except httpx.HTTPStatusError as exc:
        print(f"API returned HTTP {exc.response.status_code}")
        print(exc.response.text)

    except httpx.RequestError as exc:
        print(f"Could not connect to OpenF1: {exc}")


if __name__ == "__main__":
    main()
