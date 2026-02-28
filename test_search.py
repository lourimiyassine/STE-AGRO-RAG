# test_search.py — Validation Test Suite
"""
Validation test for the RAG Bakery Semantic Search Module.

Runs 5 domain-specific questions against the database and displays
formatted results. This is used to prove the search pipeline works
end-to-end after ingestion.

Usage:
    python test_search.py
"""

from src.search import semantic_search, format_results
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    """
    Run 5 test queries to validate the semantic search pipeline.

    Test questions cover key bakery ingredient topics:
        1. Alpha-amylase dosage
        2. Xylanase role in bread-making
        3. Ascorbic acid as oxidizing agent
        4. Dough extensibility
        5. Enzymes for bread texture
    """
    test_questions = [
        "Quelles sont les quantités recommandées d'alpha-amylase ?",
        "Quel est le rôle de la xylanase dans la panification ?",
        "Dosage de l'acide ascorbique comme agent oxydant ?",
        "Comment améliorer l'extensibilité de la pâte ?",
        "Enzymes utilisées pour améliorer la texture du pain ?",
    ]

    print("\n" + "=" * 70)
    print("  RAG BAKERY — VALIDATION TEST SUITE")
    print("  Testing 5 domain-specific queries")
    print("=" * 70)

    total_results = 0
    passed = 0

    for i, question in enumerate(test_questions, start=1):
        print(f"\n{'=' * 60}")
        print(f"  TEST {i}/{len(test_questions)}")
        print(f"  QUESTION: {question}")
        print(f"{'=' * 60}")

        results = semantic_search(question)
        total_results += len(results)

        if results:
            passed += 1
            print(format_results(results))
        else:
            print("  ⚠ No results found!")

    # Summary
    print("\n" + "=" * 70)
    print("  VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Questions tested  : {len(test_questions)}")
    print(f"  Questions passed  : {passed}/{len(test_questions)}")
    print(f"  Total results     : {total_results}")
    print(f"  Avg results/query : {total_results / len(test_questions):.1f}")
    print("=" * 70)

    if passed == len(test_questions):
        print("\n  ✅ ALL TESTS PASSED — Search pipeline is working!\n")
    else:
        print(f"\n  ⚠ {len(test_questions) - passed} test(s) returned no results.\n")
        print("  This may indicate:")
        print("    - PDFs have not been ingested yet (run: python main.py --ingest)")
        print("    - Database is not running")
        print("    - No relevant content in the database for these queries\n")


if __name__ == '__main__':
    main()
