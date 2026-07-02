import json

import requests


def main() -> None:
    url = "https://shl-smart-advisor-a-stateless-agentic.onrender.com/chat"
    payload = {
        "messages": [
            {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"},
            {"role": "assistant", "content": "Sure. What is the seniority level?"},
            {"role": "user", "content": "Mid-level, around 4 years"},
        ]
    }
    headers = {"Content-Type": "application/json"}

    print("Sending request to /chat...")
    response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)

    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")


if __name__ == "__main__":
    main()
