import json
import sys

from retrieval import get_default_retriever


def main():
    query = " ".join(sys.argv[1:]).strip() or "NVDA AI chips"
    retriever = get_default_retriever()
    result = retriever.search(query, top_k=3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
