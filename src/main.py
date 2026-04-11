import json

from final_rank import get_default_model


def main():
    model = get_default_model()
    result = model.analyze_ticker("AAPL", top_k=5)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()