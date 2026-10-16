import os
import sys
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure src/ is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from predict import predict_complaint
from preprocess import clean_text

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "complaints.db"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# ==========================================
# DATABASE INITIALIZATION
# ==========================================

def get_db_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id TEXT PRIMARY KEY,
            complaint_text TEXT NOT NULL,
            cleaned_text TEXT,
            predicted_category TEXT NOT NULL,
            category_description TEXT,
            confidence_score REAL NOT NULL,
            score_percentage TEXT NOT NULL,
            score_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            matched_keywords TEXT,
            all_scores TEXT,
            both_models TEXT,
            status TEXT DEFAULT 'classified',
            created_at TEXT NOT NULL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_complaints_created ON complaints(created_at DESC)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(predicted_category)")
    conn.commit()
    conn.close()

# Initialize Database on import
init_db()

# Load model metadata
def load_metadata() -> Dict[str, Any]:
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

METADATA = load_metadata()

# ==========================================
# FASTAPI APPLICATION SETUP
# ==========================================

app = FastAPI(
    title="Smart Complaint Classification API",
    description="REST API for real-time customer complaint categorization using TF-IDF & Classical ML",
    version="1.0.0"
)

# Enable CORS for Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# SCHEMAS
# ==========================================

class ClassifyRequest(BaseModel):
    complaint: str = Field(..., min_length=2, description="The customer complaint text to classify.")

class ComplaintRecord(BaseModel):
    id: str
    complaint_text: str
    cleaned_text: Optional[str] = None
    predicted_category: str
    category_description: Optional[str] = None
    confidence_score: float
    score_percentage: str
    score_type: str
    model_name: str
    matched_keywords: List[str] = []
    all_scores: Optional[Dict[str, float]] = None
    both_models: Optional[Dict[str, Any]] = None
    status: str = "classified"
    created_at: str

# ==========================================
# ROUTES
# ==========================================

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Smart Complaint Classification Backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": "SQLite connected",
        "categories_count": len(METADATA.get("categories", [])),
        "best_model": METADATA.get("best_model_name", "Logistic Regression")
    }

@app.get("/api/categories")
def get_categories():
    categories = METADATA.get("categories", [
        "Account & Login",
        "Billing & Payment",
        "Cancellation",
        "Delivery",
        "General Inquiry",
        "Product Issue",
        "Refund",
        "Technical Issue"
    ])
    descriptions = METADATA.get("category_descriptions", {})
    return {
        "categories": categories,
        "descriptions": descriptions
    }

@app.post("/api/classify")
def classify_complaint(payload: ClassifyRequest):
    text = payload.complaint.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")

    # Run inference using existing backend model
    pred_res = predict_complaint(text)
    if pred_res.get("status") == "error":
        raise HTTPException(status_code=422, detail=pred_res.get("message", "Classification failed."))

    complaint_id = f"CMP-{uuid.uuid4().hex[:8].upper()}"
    created_at = datetime.now(timezone.utc).isoformat()

    matched_keywords_json = json.dumps(pred_res.get("matched_keywords", []))
    all_scores_json = json.dumps(pred_res.get("all_scores", {}))
    both_models_json = json.dumps(pred_res.get("both_models", {}))

    # Persist record into SQLite
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO complaints (
            id, complaint_text, cleaned_text, predicted_category,
            category_description, confidence_score, score_percentage,
            score_type, model_name, matched_keywords, all_scores,
            both_models, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        complaint_id,
        pred_res.get("original_complaint", text),
        pred_res.get("cleaned_complaint", ""),
        pred_res.get("predicted_category"),
        pred_res.get("category_description", ""),
        pred_res.get("score_value", 0.0),
        pred_res.get("score_percentage", "0.0%"),
        pred_res.get("score_type", "Confidence Score"),
        pred_res.get("model_name", "Logistic Regression"),
        matched_keywords_json,
        all_scores_json,
        both_models_json,
        "classified",
        created_at
    ))
    conn.commit()
    conn.close()

    return {
        "id": complaint_id,
        "original_complaint": pred_res.get("original_complaint", text),
        "cleaned_complaint": pred_res.get("cleaned_complaint", ""),
        "predicted_category": pred_res.get("predicted_category"),
        "category_description": pred_res.get("category_description", ""),
        "score_type": pred_res.get("score_type", "Confidence Score"),
        "score_value": pred_res.get("score_value", 0.0),
        "score_percentage": pred_res.get("score_percentage", "0.0%"),
        "all_scores": pred_res.get("all_scores", {}),
        "matched_keywords": pred_res.get("matched_keywords", []),
        "model_name": pred_res.get("model_name", "Logistic Regression"),
        "both_models": pred_res.get("both_models", {}),
        "created_at": created_at
    }

@app.get("/api/history")
def get_complaint_history(
    search: Optional[str] = Query(None, description="Search term for complaint text"),
    category: Optional[str] = Query(None, description="Filter by complaint category"),
    sort: str = Query("desc", pattern="^(asc|desc)$", description="Sort order by date"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    conn = get_db_connection()
    cursor = conn.cursor()

    conditions = []
    params = []

    if search and search.strip():
        conditions.append("complaint_text LIKE ?")
        params.append(f"%{search.strip()}%")

    if category and category.strip() and category != "All":
        conditions.append("predicted_category = ?")
        params.append(category.strip())

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    # Count total
    count_sql = f"SELECT COUNT(*) FROM complaints {where_clause}"
    cursor.execute(count_sql, params)
    total = cursor.fetchone()[0]

    # Fetch rows
    order_direction = "DESC" if sort.lower() == "desc" else "ASC"
    sql = f"""
        SELECT * FROM complaints
        {where_clause}
        ORDER BY created_at {order_direction}
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])
    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        keywords = []
        try:
            keywords = json.loads(row["matched_keywords"]) if row["matched_keywords"] else []
        except Exception:
            keywords = []

        all_scores = {}
        try:
            all_scores = json.loads(row["all_scores"]) if row["all_scores"] else {}
        except Exception:
            all_scores = {}

        both_models = {}
        try:
            both_models = json.loads(row["both_models"]) if row["both_models"] else {}
        except Exception:
            both_models = {}

        results.append({
            "id": row["id"],
            "complaint_text": row["complaint_text"],
            "cleaned_text": row["cleaned_text"],
            "predicted_category": row["predicted_category"],
            "category_description": row["category_description"],
            "confidence_score": row["confidence_score"],
            "score_percentage": row["score_percentage"],
            "score_type": row["score_type"],
            "model_name": row["model_name"],
            "matched_keywords": keywords,
            "all_scores": all_scores,
            "both_models": both_models,
            "status": row["status"],
            "created_at": row["created_at"]
        })

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "complaints": results
    }

@app.get("/api/complaints/{complaint_id}")
def get_complaint_detail(complaint_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    keywords = []
    try:
        keywords = json.loads(row["matched_keywords"]) if row["matched_keywords"] else []
    except Exception:
        pass

    all_scores = {}
    try:
        all_scores = json.loads(row["all_scores"]) if row["all_scores"] else {}
    except Exception:
        pass

    both_models = {}
    try:
        both_models = json.loads(row["both_models"]) if row["both_models"] else {}
    except Exception:
        pass

    return {
        "id": row["id"],
        "complaint_text": row["complaint_text"],
        "cleaned_text": row["cleaned_text"],
        "predicted_category": row["predicted_category"],
        "category_description": row["category_description"],
        "confidence_score": row["confidence_score"],
        "score_percentage": row["score_percentage"],
        "score_type": row["score_type"],
        "model_name": row["model_name"],
        "matched_keywords": keywords,
        "all_scores": all_scores,
        "both_models": both_models,
        "status": row["status"],
        "created_at": row["created_at"]
    }

@app.delete("/api/complaints/{complaint_id}")
def delete_complaint(complaint_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM complaints WHERE id = ?", (complaint_id,))
    conn.commit()
    changes = conn.total_changes
    conn.close()

    if changes == 0:
        raise HTTPException(status_code=404, detail="Complaint not found.")

    return {"status": "success", "message": f"Complaint {complaint_id} deleted successfully."}

@app.get("/api/analytics")
def get_analytics():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM complaints")
    total_processed = cursor.fetchone()[0]

    cursor.execute("""
        SELECT predicted_category, COUNT(*) as cnt, AVG(confidence_score) as avg_conf
        FROM complaints
        GROUP BY predicted_category
        ORDER BY cnt DESC
    """)
    cat_rows = cursor.fetchall()

    category_dist = []
    most_frequent_cat = None
    if cat_rows:
        most_frequent_cat = cat_rows[0]["predicted_category"]
        for r in cat_rows:
            category_dist.append({
                "category": r["predicted_category"],
                "count": r["cnt"],
                "percentage": round((r["cnt"] / total_processed) * 100, 1) if total_processed > 0 else 0,
                "avg_confidence": round(r["avg_conf"] * 100, 1) if r["avg_conf"] else 0.0
            })

    # Timeline activity (grouped by date)
    cursor.execute("""
        SELECT substr(created_at, 1, 10) as day, COUNT(*) as cnt
        FROM complaints
        GROUP BY day
        ORDER BY day ASC
        LIMIT 30
    """)
    timeline_rows = cursor.fetchall()
    timeline = [{"date": r["day"], "count": r["cnt"]} for r in timeline_rows]

    # Overall avg confidence
    cursor.execute("SELECT AVG(confidence_score) FROM complaints")
    avg_conf_row = cursor.fetchone()[0]
    avg_confidence = round((avg_conf_row or 0.0) * 100, 1)

    # Fetch last 5 recent complaints
    cursor.execute("SELECT * FROM complaints ORDER BY created_at DESC LIMIT 5")
    recent_rows = cursor.fetchall()
    recent = []
    for r in recent_rows:
        recent.append({
            "id": r["id"],
            "complaint_text": r["complaint_text"],
            "predicted_category": r["predicted_category"],
            "score_percentage": r["score_percentage"],
            "created_at": r["created_at"]
        })

    conn.close()

    # Training dataset stats from model_metadata
    training_stats = METADATA.get("dataset_stats", {
        "total_records": 1600,
        "num_categories": 8,
        "is_balanced": True,
        "category_counts": {c: 200 for c in METADATA.get("categories", [])}
    })

    return {
        "total_processed": total_processed,
        "most_frequent_category": most_frequent_cat,
        "average_confidence": avg_confidence,
        "category_distribution": category_dist,
        "timeline": timeline,
        "recent_complaints": recent,
        "training_stats": training_stats
    }

@app.get("/api/keywords")
def get_keywords():
    return {
        "top_keywords_by_category": METADATA.get("top_keywords_by_category", {})
    }

@app.get("/api/metrics")
def get_metrics():
    return {
        "best_model_name": METADATA.get("best_model_name", "Logistic Regression"),
        "best_metrics": METADATA.get("best_metrics", {}),
        "all_metrics": METADATA.get("all_metrics", []),
        "per_class_reports": METADATA.get("per_class_reports", {}),
        "tfidf_config": METADATA.get("tfidf_config", {})
    }

# ==========================================
# PRODUCTION STATIC ASSETS & SPA ROUTING
# ==========================================
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa_frontend(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        target_file = FRONTEND_DIST / full_path
        if target_file.is_file():
            return FileResponse(target_file)
        return FileResponse(FRONTEND_DIST / "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)

