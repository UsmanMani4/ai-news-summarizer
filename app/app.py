from extractive.textrank import summarize_text
from abstractive.t5_model import generate_summary


def run_pipeline(article_text):
    print("🔹 Running Extractive Summarization...")
    extractive_summary = summarize_text(article_text)

    print("🔹 Running Abstractive Summarization...")
    final_summary = generate_summary(extractive_summary)

    return final_summary


if __name__ == "__main__":
    article = input("📰 Paste your news article:\n")

    result = run_pipeline(article)

    print("\n✅ FINAL SUMMARY:\n")
    print(result)