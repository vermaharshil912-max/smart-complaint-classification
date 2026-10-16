import os
import sys
from server import (
    health_check,
    get_categories,
    classify_complaint,
    get_complaint_history,
    get_analytics,
    get_metrics,
    get_keywords,
    ClassifyRequest,
    get_db_connection
)

print("=" * 60)
print("RUNNING COMPREHENSIVE BACKEND & DATABASE TESTS")
print("=" * 60)

print("\n[1/6] Testing Health Endpoint...")
health = health_check()
assert health["status"] == "healthy"
print(f" -> Health OK: Status='{health['status']}', Service='{health['service']}'")

print("\n[2/6] Testing Categories...")
cats_res = get_categories()
assert len(cats_res["categories"]) == 8
print(f" -> Categories OK: {len(cats_res['categories'])} distinct departments found:")
for c in cats_res["categories"]:
    print(f"    • {c}")

print("\n[3/6] Testing Real Prediction & SQLite Persistence...")
test_complaints = [
    "My payment was deducted twice for the same order and I see two charges on my statement.",
    "My package shows delivered by the courier but I never received anything at my house.",
    "The product arrived completely broken with damaged components and missing parts.",
    "I forgot my password and cannot log into my account or reset my credentials.",
    "The application crashes every single time I try to open the checkout screen.",
    "I requested a refund several days ago but still haven't received the reimbursement.",
    "I want to cancel my active monthly subscription immediately.",
    "What plans and services do you offer for new users?"
]

classified_ids = []
for text in test_complaints:
    req = ClassifyRequest(complaint=text)
    res = classify_complaint(req)
    assert res["id"] is not None
    assert res["predicted_category"] is not None
    assert res["score_value"] > 0
    classified_ids.append(res["id"])
    print(f" -> Classified '{text[:35]}...'")
    print(f"    Category: {res['predicted_category']} | Conf: {res['score_percentage']} | DB ID: {res['id']}")

print(f"\n[4/6] Verifying SQLite Database Table Directly...")
conn = get_db_connection()
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM complaints")
total_rows = cursor.fetchone()[0]
print(f" -> SQLite Database Total Verified Rows: {total_rows}")
assert total_rows >= len(test_complaints)

cursor.execute("SELECT id, predicted_category, confidence_score, created_at FROM complaints ORDER BY created_at DESC LIMIT 3")
recent = cursor.fetchall()
print(" -> Last 3 entries in SQLite:")
for r in recent:
    print(f"    ID: {r['id']} | Cat: {r['predicted_category']} | Score: {r['confidence_score']:.4f} | Time: {r['created_at']}")
conn.close()

print("\n[5/6] Testing History & Filter Queries...")
history = get_complaint_history(search=None, category=None, sort="desc", limit=10, offset=0)
assert history["total"] >= len(test_complaints)
assert len(history["complaints"]) > 0
print(f" -> History OK: Total {history['total']} records returned by API.")

search_res = get_complaint_history(search="payment", category=None, sort="desc", limit=10, offset=0)
print(f" -> Search for 'payment' returned {search_res['total']} match(es).")
assert search_res["total"] >= 1

filter_res = get_complaint_history(search=None, category="Delivery", sort="desc", limit=10, offset=0)
print(f" -> Filter for 'Delivery' returned {filter_res['total']} match(es).")
assert filter_res["total"] >= 1

print("\n[6/6] Testing Analytics, Metrics & Keywords Endpoints...")
analytics = get_analytics()
assert analytics["total_processed"] >= len(test_complaints)
print(f" -> Analytics OK: Total Processed={analytics['total_processed']}, Top Category={analytics['most_frequent_category']}, Avg Conf={analytics['average_confidence']}%")

metrics = get_metrics()
assert len(metrics["all_metrics"]) == 2
print(f" -> Metrics OK: Best Model={metrics['best_model_name']}, Compared Models={[m['model_name'] for m in metrics['all_metrics']]}")

keywords = get_keywords()
assert len(keywords["top_keywords_by_category"]) == 8
print(f" -> Keywords OK: Top keywords available for all 8 categories.")

print("\n" + "=" * 60)
print("SUCCESS: ALL BACKEND & DATABASE INTEGRATION TESTS PASSED 100%!")
print("=" * 60)
