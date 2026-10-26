
import os
import sys
import re
import io
import requests
from dotenv import load_dotenv

# Force UTF-8 encoding for Windows standard output/error
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import PyPDF2

# --- 1. MODELS ---
class ChatHistoryMessage(BaseModel):
    role: str
    content: str

class Query(BaseModel):
    question: str
    conversation_history: list[ChatHistoryMessage] | None = None

# --- 2. CREATE FASTAPI APP ---
app = FastAPI(
    title="MED AI - Multi-Domain Medical & Pathology Intelligence API",
    description="Universal medical assistant with dynamic intent classification, PDF ingestion, and clinical knowledge."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. VECTOR DATABASE & KNOWLEDGE BASE TRACKER ---
db = None
retriever = None

active_knowledge_bases = [
    {
        "id": "robbins-pathology",
        "title": "Robbins & Cotran: Pathologic Basis of Disease",
        "category": "Pathology & Histology",
        "type": "core",
        "description": "Gold-standard reference for disease mechanisms, etiology, and morphology."
    },
    {
        "id": "clinical-pharmacology",
        "title": "Clinical Pharmacology & Therapeutics Manual",
        "category": "Pharmacology & Drugs",
        "type": "core",
        "description": "Drug classes, mechanisms of action, adverse effects, and indications."
    }
]

BOILERPLATE_TERMS = [
    "acknowledgments", "acknowledgements", "we thank our spouses", "copyright",
    "all rights reserved", "isbn", "elsevier saunders", "table of contents",
    "in memory of", "contributors", "preface we launch", "veterans of previous editions",
    "author index", "subject index"
]

def initialize_database():
    global db, retriever
    if db is None:
        print("[INFO] Initializing FAISS database and HuggingFace embeddings...")
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_core.documents import Document
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "my_faiss_index")

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"FAISS index folder not found at {db_path}")

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

        # Ingest clinical pharmacology guide if present
        pharm_path = os.path.join(base_dir, "data", "clinical_pharmacology_guide.txt")
        if os.path.exists(pharm_path):
            try:
                with open(pharm_path, "r", encoding="utf-8") as f:
                    pharm_text = f.read()
                splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
                pharm_chunks = splitter.split_text(pharm_text)
                pharm_docs = [
                    Document(page_content=chunk, metadata={"source": "Clinical Pharmacology & Therapeutics Manual"})
                    for chunk in pharm_chunks
                ]
                db.add_documents(pharm_docs)
                print(f"[OK] Ingested {len(pharm_docs)} pharmacology reference chunks.")
            except Exception as e:
                print(f"[WARN] Failed to ingest pharmacology guide: {e}")

        retriever = db.as_retriever(search_kwargs={"k": 8})
        print("[OK] Vector database fully initialized.")

# --- 4. HEALTH, ROOT & KNOWLEDGE BASES ENDPOINTS ---
@app.get("/")
async def root():
    return {"status": "ok", "message": "MED AI Universal Medical Backend is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/documents")
async def get_documents():
    """Returns active knowledge bases (built-in libraries and user-uploaded files)."""
    return {"documents": active_knowledge_bases}

# --- 5. DYNAMIC DOCUMENT UPLOAD ENDPOINT ---
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Ingests and indexes any user-uploaded medical PDF or text file into the active vector database.
    """
    try:
        initialize_database()
        from langchain_core.documents import Document
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter

        filename = file.filename
        content_bytes = await file.read()
        extracted_text = ""

        # Handle PDF files
        if filename.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(io.BytesIO(content_bytes))
            pages_text = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages_text.append(text)
            extracted_text = "\n".join(pages_text)
        # Handle TXT / MD files
        elif filename.lower().endswith((".txt", ".md")):
            extracted_text = content_bytes.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload a .pdf, .txt, or .md medical document."
            )

        if not extracted_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from the uploaded document."
            )

        # Split into semantic chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=120)
        chunks = splitter.split_text(extracted_text)

        if not chunks:
            raise HTTPException(status_code=400, detail="Document content was too short to index.")

        docs = [
            Document(page_content=clean_text_passage(c), metadata={"source": filename})
            for c in chunks
        ]

        # Add to active vector database
        db.add_documents(docs)

        # Register in active knowledge base list
        active_knowledge_bases.append({
            "id": f"upload-{len(active_knowledge_bases)}",
            "title": filename,
            "category": "User Uploaded Document",
            "type": "uploaded",
            "chunks": len(docs),
            "description": f"Custom medical document indexed with {len(docs)} text chunks."
        })

        print(f"[OK] Successfully indexed {filename} ({len(docs)} chunks).")
        return {
            "success": True,
            "filename": filename,
            "chunks": len(docs),
            "message": f"Successfully indexed '{filename}' ({len(docs)} chunks) into the medical knowledge base!"
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

# --- 6. PHASE 3: MEDICAL INTENT CLASSIFIER & ROUTING ---
MEDICAL_INTENTS = {
    "PATIENT_CARE_RELIEF": {
        "category": "Patient Care & Symptom Relief",
        "icon": "🌡️",
        "focus": "Immediate Relief Steps, Temperature Reduction, Safe Medications & Home Comfort",
        "keywords": [
            "decrease", "reduce", "lower", "bring down", "how to treat", "how to cure",
            "how to decrease", "how to reduce", "remedy", "home remedy", "relief", "feel better",
            "what to do", "what should i do", "medicine for fever", "temp", "temperature",
            "have fever", "had fever", "getting fever", "got fever", "high fever", "101", "102",
            "103", "100", "headache remedy", "cough remedy", "cold remedy", "fever relief",
            "how to stop", "home care", "paracetamol", "acetaminophen", "ibuprofen", "dolo"
        ],
        "prompt_guidelines": (
            "Specialized Intent: PATIENT CARE & SYMPTOM RELIEF.\n"
            "CRITICAL REQUIREMENT: The user is experiencing symptoms right now and needs simple, friendly, highly practical, and understandable guidance.\n"
            "DO NOT overwhelm the user with rare genetic disorders (like FMF, HLH), ICU lab panels, or heavy medical jargon!\n"
            "Structure your response in clear, everyday language:\n"
            "1. 🌡️ **What Your Temperature Means**: Explain clearly (e.g., 101°F is a mild-to-moderate fever, a natural immune response fighting an infection like a viral cold or flu).\n"
            "2. ⚡ **Immediate Steps to Lower Temperature**: Simple actions (rest in a cool room, drink lots of water/electrolytes, wear light clothing).\n"
            "3. 💊 **Medicines You Can Take Safely**: Plain-language guidance on Acetaminophen (Paracetamol) or Ibuprofen, including standard adult doses, timing, and safety cautions.\n"
            "4. 🧊 **Home Comfort Measures**: Lukewarm sponge bath, cold damp cloth on forehead; what to avoid (no ice-cold baths, don't pile on thick heavy blankets).\n"
            "5. 🚨 **When to See a Doctor**: Clear, reassuring red flags (fever >103°F / 39.4°C, stiff neck, trouble breathing, lasting >3 days)."
        )
    },
    "PHARMACOLOGY_DRUG": {
        "category": "Pharmacology & Drug Therapy",
        "icon": "💊",
        "focus": "Mechanism of Action (MOA), Clinical Dosing, Adverse Reactions & Interactions",
        "keywords": [
            "drug", "drugs", "medication", "medications", "dose", "dosing", "dosage",
            "mechanism of action", "moa", "side effect", "side effects", "contraindication",
            "contraindications", "pharmacokinetics", "pharmacodynamics", "inhibitor", "agonist",
            "antagonist", "antibiotic", "antibiotics", "beta-blocker", "statin", "statins",
            "nsaid", "nsaids", "steroid", "steroids", "metformin", "lisinopril", "aspirin",
            "penicillin", "vancomycin", "insulin", "toxicity", "pharmacology"
        ],
        "prompt_guidelines": (
            "Specialized Intent: PHARMACOLOGY & DRUG THERAPY.\n"
            "Structure your clinical answer clearly:\n"
            "- 💊 **Drug Class & How It Works**\n"
            "- 📋 **Approved Uses & Standard Dosing**\n"
            "- ⚖️ **Common Side Effects & Who Should Avoid It**\n"
            "- 🔄 **Important Drug Interactions**"
        )
    },
    "DIAGNOSTIC_SYMPTOM": {
        "category": "Differential Diagnosis & Symptoms",
        "icon": "🩺",
        "focus": "Clinical Presentation, Diagnostic Algorithms & Differential Likelihoods",
        "keywords": [
            "symptom", "symptoms", "sign", "signs", "differential", "differential diagnosis",
            "presentation", "workup", "pain", "fever", "cough", "rash", "headache",
            "dyspnea", "chest pain", "shortness of breath", "abdominal pain", "labs",
            "imaging", "ct scan", "mri", "ultrasound", "biomarker", "troponin", "diagnose"
        ],
        "prompt_guidelines": (
            "Specialized Intent: CLINICAL PRESENTATION & COMMON CAUSES.\n"
            "Structure your answer in simple, understandable terms:\n"
            "- 🩺 **Understanding the Symptoms** (In everyday language)\n"
            "- 📊 **Most Common Everyday Causes** (Focus on 3-4 common causes like cold, flu, viral infection first)\n"
            "- 🩹 **Immediate Relief & Self-Care Steps**\n"
            "- ⚠️ **Red Flag Warning Signs** (When to contact a physician)"
        )
    },
    "TREATMENT_SURGERY": {
        "category": "Clinical Management & Surgery",
        "icon": "🩹",
        "focus": "First-line Medical Regimens, Surgical Interventions & Care Protocols",
        "keywords": [
            "treatment", "treat", "treatments", "surgery", "surgical", "operation",
            "operative", "therapy", "therapies", "management", "protocol", "guidelines",
            "procedure", "resection", "emergency care", "resuscitation", "intervention",
            "lifestyle", "rehabilitation"
        ],
        "prompt_guidelines": (
            "Specialized Intent: CLINICAL MANAGEMENT & INTERVENTIONS.\n"
            "Structure your clinical answer with:\n"
            "- 🩹 **First-Line Medical & Conservative Therapy**\n"
            "- 🏥 **Interventional & Surgical Procedures (Indications & Timing)**\n"
            "- 📈 **Patient Monitoring, Prognostic Factors & Follow-Up**\n"
            "- ⚠️ **Critical Complications to Prevent & Monitor**"
        )
    },
    "PATHOLOGY_DISEASE": {
        "category": "Pathology & Disease Mechanisms",
        "icon": "🔬",
        "focus": "Etiology, Cellular Pathogenesis & Histomorphologic Features",
        "keywords": [
            "pathology", "pathogenesis", "etiology", "morphology", "histology",
            "cellular", "cell injury", "necrosis", "apoptosis", "infarct", "infarction",
            "hypertrophy", "atrophy", "metaplasia", "inflammation", "tumor", "tumors",
            "cancer", "carcinoma", "neoplasm", "mutation", "biopsy", "microscopic"
        ],
        "prompt_guidelines": (
            "Specialized Intent: PATHOLOGY & DISEASE MECHANISMS.\n"
            "Structure your clinical answer with:\n"
            "- 🔬 **Etiology (Underlying Genetic & Environmental Triggers)**\n"
            "- 🧬 **Pathogenesis (Molecular & Cellular Sequence of Events)**\n"
            "- 🔍 **Morphologic Hallmarks (Gross Appearance & Histopathology)**\n"
            "- 💡 **Clinical Correlation & Disease Progression**"
        )
    }
}

def classify_medical_intent(query: str) -> dict:
    q_lower = query.lower()
    scores = {}
    for intent_key, data in MEDICAL_INTENTS.items():
        score = sum(1 for kw in data["keywords"] if re.search(r'\b' + re.escape(kw) + r'\b', q_lower))
        scores[intent_key] = score

    best_match = max(scores, key=scores.get)
    if scores[best_match] > 0:
        return MEDICAL_INTENTS[best_match]

    return {
        "category": "General Medical Education",
        "icon": "💡",
        "focus": "Core Clinical Principles & Patient Guidance",
        "prompt_guidelines": (
            "Specialized Intent: GENERAL MEDICAL EDUCATION.\n"
            "Provide a clear, high-yield, patient-friendly yet medically rigorous explanation "
            "covering definitions, clinical relevance, and essential guidance."
        )
    }

# --- 7. PHASE 5: CLINICAL RED-FLAG TRIAGE SYSTEM ---
EMERGENCY_RED_FLAGS = [
    {
        "id": "acute_coronary",
        "patterns": [r"crushing.*chest", r"chest.*pain.*radiat", r"chest.*pressure.*sweat", r"substernal.*pain", r"heart.*attack.*symptom"],
        "title": "Possible Acute Coronary Syndrome / Myocardial Infarction",
        "guidance": "CRITICAL EMERGENCY: Symptoms described are suspicious for acute myocardial ischemia or infarction. Call emergency medical services (911 / 112 / 999) or proceed to the nearest Emergency Department immediately. Do NOT drive yourself."
    },
    {
        "id": "acute_stroke",
        "patterns": [r"face.*droop", r"arm.*weakness", r"slur.*speech", r"sudden.*numbness.*side", r"sudden.*paralysis", r"fast.*stroke"],
        "title": "Possible Acute Stroke (CVA / TIA)",
        "guidance": "TIME-CRITICAL EMERGENCY (FAST): Symptoms indicate potential acute cerebral ischemia. Brain tissue is time-sensitive. Contact emergency services immediately for hyperacute stroke evaluation and reperfusion."
    },
    {
        "id": "severe_anaphylaxis",
        "patterns": [r"anaphylaxis", r"throat.*clos", r"swollen.*tongue.*breath", r"lip.*swell.*breath", r"stridor.*hives"],
        "title": "Severe Anaphylactic Reaction",
        "guidance": "IMMEDIATE LIFE THREAT: Severe airway compromise and circulatory risk. Administer intramuscular epinephrine (EpiPen) immediately into the lateral thigh and call emergency services right away."
    },
    {
        "id": "thunderclap_headache",
        "patterns": [r"worst.*headache.*life", r"thunderclap", r"explosive.*headache", r"sudden.*severe.*occipital"],
        "title": "Thunderclap Headache (Possible Subarachnoid Hemorrhage)",
        "guidance": "NEUROLOGICAL EMERGENCY: A sudden-onset, peak-intensity headache warrants immediate non-contrast head CT and emergency clinical assessment to rule out vascular aneurysmal rupture."
    },
    {
        "id": "poisoning_overdose",
        "patterns": [r"overdose", r"swallowed.*poison", r"ingested.*bleach", r"took.*entire.*bottle", r"lethal.*dose"],
        "title": "Acute Toxicology / Suspected Poisoning",
        "guidance": "POISON CONTROL ALERT: Immediately call your local Poison Control Center (US: 1-800-222-1222) or emergency services. Have the container, dosage, and time of ingestion ready."
    },
    {
        "id": "mental_health_crisis",
        "patterns": [r"kill.*myself", r"end.*my.*life", r"suicid.*plan", r"commit.*suicide", r"want.*to.*die"],
        "title": "Crisis Intervention Support",
        "guidance": "CRISIS SUPPORT: You are not alone. Please reach out right now: call or text 988 (Suicide & Crisis Lifeline in US/Canada), call 111 (UK), or contact your local emergency department for immediate compassionate care."
    }
]

def check_emergency_red_flags(query_text: str) -> dict | None:
    text_lower = query_text.lower()
    for item in EMERGENCY_RED_FLAGS:
        for pat in item["patterns"]:
            if re.search(pat, text_lower):
                return {
                    "is_emergency": True,
                    "title": item["title"],
                    "guidance": item["guidance"]
                }
    return None

# --- 8. UNIVERSAL MEDICAL LLM CALLER (WITH MULTI-TURN MEMORY) ---
def try_llm_generation(question: str, context: str, intent: dict, conversation_history: list = None) -> str | None:
    system_prompt = (
        "You are MED AI, an advanced medical AI assistant with comprehensive clinical expertise "
        "across all medical domains (including pathology, clinical pharmacology, internal medicine, "
        "cardiology, diagnostics, surgery, and patient education).\n\n"
        f"Detected Medical Domain: {intent['category']}\n"
        f"{intent['prompt_guidelines']}\n\n"
        "Guidelines:\n"
        "1. You can answer ANY medical question thoroughly, accurately, and in a friendly, easy-to-understand manner.\n"
        "2. Ground your answer in evidence-based medicine.\n"
        "3. PATIENT-FIRST CLARITY: Always write in plain, accessible, and compassionate language. NEVER overwhelm a user with dense academic pathology jargon or rare genetic diseases (e.g. FMF, HLH, Rickettsia) when they ask about common everyday symptoms like fever, cold, headache, or nausea.\n"
        "4. When a user asks how to decrease/treat a symptom (like a 101°F fever), prioritize immediate actionable relief: what to do right now, safe OTC medications (Paracetamol/Ibuprofen) with clear timing, home comfort tips, and simple warning signs.\n"
        "5. If relevant excerpts from medical literature are provided in the context below, seamlessly incorporate them in everyday language.\n"
        "6. Maintain continuity with prior clinical turns when conversation history is present.\n"
        "7. Do NOT include author greetings, publishing notes, or textbook boilerplate."
    )
    user_prompt = f"Question: {question}\n\nRelevant Literature Context (if available):\n{context}\n\nPlease provide a comprehensive, clear, and medically accurate response:"

    # Assemble multi-turn conversation messages
    chat_messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        for turn in conversation_history[-6:]:
            role = turn.role if hasattr(turn, 'role') else turn.get('role', 'user')
            content = turn.content if hasattr(turn, 'content') else turn.get('content', '')
            norm_role = "assistant" if role in ["assistant", "ai"] else "user"
            if content.strip():
                chat_messages.append({"role": norm_role, "content": content.strip()[:1000]})
    chat_messages.append({"role": "user", "content": user_prompt})

    # 1. Check Groq (Fastest & 100% Free: openai/gpt-oss-120b, qwen/qwen3.8-27b, llama-3.3-70b-versatile)
    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    if groq_key:
        for model in ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            try:
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}"},
                    json={
                        "model": model,
                        "messages": chat_messages,
                        "temperature": 0.2,
                        "max_tokens": 1400
                    },
                    timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    print(f"[OK] Response generated via Groq ({model}) for [{intent['category']}]")
                    return data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[WARN] Groq request failed for {model}: {e}")

    # 2. Check Google Gemini (Free at aistudio.google.com)
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        for model in ["gemini-1.5-flash", "gemini-2.0-flash"]:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
                contents = []
                for m in chat_messages:
                    r = "model" if m["role"] == "assistant" else "user"
                    contents.append({"role": r, "parts": [{"text": m["content"]}]})
                resp = requests.post(
                    url,
                    json={
                        "contents": contents,
                        "generationConfig": {
                            "temperature": 0.2,
                            "maxOutputTokens": 1400
                        }
                    },
                    timeout=15
                )
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        if parts and "text" in parts[0]:
                            print(f"[OK] Response generated via Google Gemini ({model})")
                            return parts[0]["text"].strip()
            except Exception as e:
                print(f"[WARN] Gemini request failed for {model}: {e}")

    # 3. Check Together AI
    together_key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if together_key and together_key != "your_together_ai_api_key_here":
        try:
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {together_key}"},
                json={
                    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
                    "messages": chat_messages,
                    "temperature": 0.2,
                    "max_tokens": 1400
                },
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                print("[OK] Response generated via Together AI")
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[WARN] Together AI request failed: {e}")

    # 4. Check OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": chat_messages,
                    "temperature": 0.2,
                    "max_tokens": 1400
                },
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                print("[OK] Response generated via OpenAI")
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[WARN] OpenAI request failed: {e}")

    return None

# --- 8. LOCAL SMART MEDICAL EXTRACTIVE SYNTHESIZER ---
def clean_text_passage(text: str) -> str:
    """Removes OCR glitches, excessive whitespace, and dangling symbols."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b[A-Z]\s+[A-Z]\s+[A-Z]\s+[A-Z]\b', '', text)
    return text.strip()

def build_structured_summary(question: str, clean_docs: list, intent: dict) -> str:
    """Builds an intent-guided structured medical synthesis directly from local document chunks."""
    combined_text = " ".join([clean_text_passage(doc.page_content) for doc in clean_docs])
    sentences = re.split(r'(?<=[.!?])\s+', combined_text)

    informative_sentences = [
        s.strip() for s in sentences
        if len(s.strip()) > 35 and not any(term in s.lower() for term in BOILERPLATE_TERMS)
    ]

    key_points = []
    seen = set()
    for s in informative_sentences:
        lower_s = s.lower()
        if lower_s not in seen:
            seen.add(lower_s)
            key_points.append(s)
        if len(key_points) >= 6:
            break

    summary_parts = [
        f"### {intent['icon']} **Clinical Assessment: {intent['category']}**",
        ""
    ]

    if key_points:
        summary_parts.append(f"> **{key_points[0]}**")
        summary_parts.append("")

    if len(key_points) > 1:
        summary_parts.append(f"### 🔬 **Clinical Evidence & {intent['focus']}**")
        for point in key_points[1:4]:
            summary_parts.append(f"- {point}")
        summary_parts.append("")

    if len(key_points) > 4:
        summary_parts.append("### 💡 **Key Considerations & Guidelines**")
        for point in key_points[4:6]:
            summary_parts.append(f"- {point}")
        summary_parts.append("")

    summary_parts.append(
        "\n---\n*📚 **Reference:** Robbins & Cotran: Pathologic Basis of Disease & Clinical Pharmacology*"
    )

    return "\n".join(summary_parts)

# --- 9. QUESTION ANSWERING ENDPOINT ---
@app.post("/ask", summary="Ask a question")
async def ask_question(query: Query):
    try:
        initialize_database()
        question = query.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        # 1. Phase 5: Emergency Red-Flag Triage
        red_flag = check_emergency_red_flags(question)
        if red_flag:
            print(f"[EMERGENCY ALERT] Red flag triggered: {red_flag['title']}")

        # 2. Phase 3: Medical Intent Classification
        intent = classify_medical_intent(question)
        print(f"[INTENT] Question: '{question[:40]}...' -> Detected Intent: {intent['icon']} {intent['category']}")

        # 3. Retrieve candidate chunks from active vector store
        candidate_docs = retriever.invoke(question)

        # 4. Filter out boilerplate
        clean_docs = []
        for doc in candidate_docs:
            content_lower = doc.page_content.lower()
            if not any(bp in content_lower for bp in BOILERPLATE_TERMS):
                cleaned_doc_content = clean_text_passage(doc.page_content)
                if len(cleaned_doc_content) > 50:
                    doc.page_content = cleaned_doc_content
                    clean_docs.append(doc)

        if not clean_docs:
            clean_docs = candidate_docs[:3]

        # Use top 3 clean docs for citation with precise source attribution
        final_sources = [
            {
                "content": doc.page_content[:450] + "...",
                "metadata": {
                    "source": doc.metadata.get("source", "Robbins & Cotran Pathologic Basis of Disease")
                }
            }
            for doc in clean_docs[:3]
        ]

        # Prepare context text
        context_text = "\n\n".join([
            f"[Source {idx+1}: {doc.metadata.get('source', 'Medical Reference')}]:\n{doc.page_content}"
            for idx, doc in enumerate(clean_docs[:4])
        ])

        # 5. Attempt Universal LLM Generation with Intent-Tailored Prompting & Multi-Turn History
        llm_answer = try_llm_generation(
            question=question,
            context=context_text,
            intent=intent,
            conversation_history=query.conversation_history
        )

        intent_payload = {
            "category": intent["category"],
            "icon": intent["icon"],
            "focus": intent["focus"]
        }

        if llm_answer:
            return {
                "answer": llm_answer,
                "sources": final_sources,
                "intent": intent_payload,
                "red_flag": red_flag
            }

        # 6. Fallback to Local Smart Medical Extractive Synthesis
        structured_answer = build_structured_summary(question, clean_docs[:4], intent)
        return {
            "answer": structured_answer,
            "sources": final_sources,
            "intent": intent_payload,
            "red_flag": red_flag
        }

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    app_dir = os.path.dirname(os.path.abspath(__file__))
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=app_dir)
