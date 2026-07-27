import os
import json
import pickle
import urllib.parse
import numpy as np
import nltk
import random
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from google import genai
from google.genai import types
from dotenv import load_dotenv
from database import engine, Base
import models
import auth
from PIL import Image
import io

# --- Setup ---
nltk.data.path.append(r"D:\AI_Chat\nltk_data")
lemmatizer = WordNetLemmatizer()

# --- Initialize Gemini ---
load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Create a .env file in this folder "
        "(D:\\AI_Chat) containing:\nGEMINI_API_KEY=your_new_key_here"
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

GROUNDED_CONFIG = types.GenerateContentConfig(
    tools=[types.Tool(google_search=types.GoogleSearch())]
)

def extract_grounding_sources(response):
    try:
        chunks = response.candidates[0].grounding_metadata.grounding_chunks
        if not chunks:
            return ""
        seen = set()
        lines = []
        for chunk in chunks:
            web = getattr(chunk, "web", None)
            if not web or not web.uri or web.uri in seen:
                continue
            seen.add(web.uri)
            label = web.domain or web.title or "source"
            lines.append(f"- {label}: {web.uri}")
            if len(lines) >= 6:
                break
        return "\n\nVerified sources:\n" + "\n".join(lines) if lines else ""
    except (AttributeError, IndexError, TypeError):
        return ""

# --- FastAPI App Initialization ---
app = FastAPI(title="AI Chatbot API", description="FastAPI server for University Lab Project")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Auth setup ---
Base.metadata.create_all(bind=engine)
app.include_router(auth.router)
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Load the AI Brain & Data (single source: intents.json) ---
try:
    model = load_model("chatbot_model.h5")
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    with open("intents.json", "r") as file:
        intents = json.load(file)
except Exception as e:
    print(f"Error loading core intent/model files: {e}")
    model = None
    words = []
    classes = []
    intents = {"intents": [], "muscle_groups": {}, "weekly_split": {}}

# intents.json also carries the muscle-group exercise/nutrition data and
# the weekly split — pulled out here once so the rest of the file doesn't
# need to know it all lives in the same JSON as the training patterns.
MUSCLE_GROUPS = intents.get("muscle_groups", {})
WEEKLY_SPLIT = intents.get("weekly_split", {})

# --- Helper Functions ---
STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "being", "what", "who", "whom", 
    "which", "this", "that", "these", "those", "do", "does", "did", "doing", "of", "at", "by", "for", 
    "with", "about", "to", "from", "in", "on", "up", "down", "and", "or", "but", "if", "i", "me", "my", 
    "you", "your", "he", "she", "it", "we", "they", "can", "could", "will", "would", "should", "may", 
    "might", "must",
}

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [
        lemmatizer.lemmatize(word.lower()) 
        for word in sentence_words if word.lower() not in STOPWORDS
    ]

def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    if model is None:
        return []
    p = bag_of_words(sentence, words)
    if not p.any():
        return []
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.50  # Your 50% confidence guardrail
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def find_intent(intents_json, tag):
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return intent
    return None

def ask_gemini_via_api(user_message):
    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=user_message,
            config=GROUNDED_CONFIG,
        )
        return (response.text or "") + extract_grounding_sources(response)
    except Exception as e:
        return f"Gemini Error: {str(e)}"

# --- Deterministic Fitness-Data Matching ---
# None of this depends on the trained classifier, so it works regardless
# of phrasing. Two ways in: name a specific exercise ("bicep curl form"),
# or name a muscle group / day of the week ("biceps exercises", "what do
# i train monday") to get every exercise in that group at once.

MUSCLE_GROUP_ALIASES = {
    "chest": ["chest", "pecs", "pec"],
    "back": ["back", "lats"],
    "biceps": ["biceps", "bicep"],
    "triceps": ["triceps", "tricep"],
    "shoulders": ["shoulders", "shoulder", "delts", "delt"],
    "legs": ["legs", "leg day", "quads", "quad", "hamstrings"],
    "rest": ["rest day", "rest"],
}

NUTRITION_CONTEXT_WORDS = [
    "eat", "food", "meal", "diet", "nutrition", "nutrients", "breakfast", "lunch", "dinner", "snack",
]

DAYS_OF_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

def detect_sub_category(user_message_clean):
    if any(w in user_message_clean for w in ["posture", "form", "position", "align", "spine"]):
        return "posture"
    if any(w in user_message_clean for w in ["mistake", "wrong", "avoid", "error"]):
        return "mistakes"
    if any(w in user_message_clean for w in ["reps", "rep range", "sets", "volume"]):
        return "reps"
    return "steps"

def find_specific_exercise(user_message_clean):
    """Search every muscle group's exercise list for a name match.
    Returns (group_key, exercise_dict) or (None, None)."""
    for group_key, group_data in MUSCLE_GROUPS.items():
        for ex in group_data.get("exercises", []):
            if ex["key"].replace("_", " ") in user_message_clean:
                return group_key, ex
    return None, None

def find_muscle_group(user_message_clean):
    for group_key, aliases in MUSCLE_GROUP_ALIASES.items():
        if any(alias in user_message_clean for alias in aliases):
            return group_key
    return None

def resolve_day_to_group(user_message_clean):
    for day in DAYS_OF_WEEK:
        if day in user_message_clean:
            return WEEKLY_SPLIT.get(day)
    return None

def format_single_exercise(ex, sub_category):
    if sub_category == "posture":
        body = "\n".join(f"- {p}" for p in ex["posture"])
        return f"**{ex['name']} — Form & Posture:**\n{body}"
    if sub_category == "mistakes":
        body = "\n".join(f"- {m}" for m in ex["common_mistakes"])
        return f"**Mistakes to avoid on {ex['name']}:**\n{body}"
    if sub_category == "reps":
        return f"**{ex['name']} — Sets & Reps:**\n{ex['rep_guidance']}"
    body = "\n".join(f"{i+1}. {s}" for i, s in enumerate(ex["steps"]))
    return f"**{ex['name']} — Step by Step:**\n{body}"

def format_muscle_group_response(group_key, sub_category):
    group = MUSCLE_GROUPS.get(group_key)
    if not group:
        return None
    exercises = group.get("exercises", [])
    display = group.get("display_name", group_key.capitalize())

    if not exercises:
        return {
            "response": f"**{display}:** No exercises scheduled — this is a recovery day.",
            "source": "local_database",
        }

    labels = {"posture": "Posture & Form", "mistakes": "Common Mistakes", "reps": "Sets & Reps", "steps": "Exercises"}
    parts = [f"### {display} — {labels[sub_category]}"]
    for ex in exercises:
        parts.append(format_single_exercise(ex, sub_category))
    return {"response": "\n\n".join(parts), "source": "local_database"}

def format_muscle_group_nutrition(group_key):
    group = MUSCLE_GROUPS.get(group_key)
    if not group:
        return None
    display = group.get("display_name", group_key.capitalize())
    focus = group.get("nutrition_focus", "")
    return {"response": f"### {display} Day Nutrition\n{focus}", "source": "local_database"}

# --- API Endpoints ---
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "online", "message": "Chatbot API is running!"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    user_message_clean = user_message.lower()

    # 1. A specific exercise named? Most precise match, check first —
    # "hammer curl form" should answer for hammer curl, not all of biceps.
    group_key, ex = find_specific_exercise(user_message_clean)
    if ex:
        sub_category = detect_sub_category(user_message_clean)
        return {"response": format_single_exercise(ex, sub_category), "source": "local_database"}

    # 2. A muscle group named directly, or implied by a day of the week
    # ("what do i train monday" -> resolved via the weekly split)?
    group_key = find_muscle_group(user_message_clean)
    if not group_key:
        group_key = resolve_day_to_group(user_message_clean)

    if group_key:
        if any(w in user_message_clean for w in NUTRITION_CONTEXT_WORDS):
            nut_response = format_muscle_group_nutrition(group_key)
            if nut_response:
                return nut_response
        sub_category = detect_sub_category(user_message_clean)
        group_response = format_muscle_group_response(group_key, sub_category)
        if group_response:
            return group_response

    # 3. Everything else (greeting, who-are-you, goodbye, thanks, and
    # general fallback) goes through the trained intent classifier.
    ints = predict_class(user_message, model)

    if not ints:
        gemini_response = ask_gemini_via_api(user_message)
        return {"response": gemini_response, "source": "gemini_api"}

    tag = ints[0]['intent']
    intent_obj = find_intent(intents, tag)

    if not intent_obj:
        gemini_response = ask_gemini_via_api(user_message)
        return {"response": gemini_response, "source": "gemini_api"}

    return {"response": random.choice(intent_obj['responses']), "source": "local_model"}


@app.post("/chat-vision")
async def chat_with_vision(
    file: UploadFile = File(...), 
    prompt: str = Form(None)  # Optional text prompt sent alongside the image
):
    # Read the incoming file bytes
    image_bytes = await file.read()
    
    # Send it to our helper function
    analysis_result = ask_gemini_vision(image_bytes, prompt)
    
    return {
        "response_source": "Gemini Multimodal Vision API",
        "bot_response": analysis_result
    }

class VisionAnalysis(BaseModel):
    object_name: str
    description: str
    youtube_search_query: str  # empty string only if no reasonable video topic applies
    reference_website_url: str  # empty string unless genuinely confident it's real

VISION_INSTRUCTION = (
    "Identify the main object in this image for a chatbot app. Give a clear "
    "object_name, and in description explain what it is and how to use it "
    "in 2-4 sentences (if it's gym/fitness equipment, include proper form "
    "and common mistakes to avoid). In youtube_search_query, suggest a "
    "short, specific search phrase someone could use to find a good "
    "instructional or how-to video about it (e.g. 'lat pulldown machine "
    "proper form tutorial', or 'how to balance a wobbly ceiling fan'). "
    "Leave youtube_search_query as an empty string only if no reasonable "
    "video topic applies. In reference_website_url, include a real URL to "
    "a well-known reference page if you're confident one exists and is "
    "correct — for example a Wikipedia article or an official manufacturer "
    "page. Only include a URL you are genuinely confident is real and "
    "correctly spelled; leave it as an empty string rather than guess."
)

def ask_gemini_vision(image_bytes: bytes, user_prompt: str | None = None):
    try:
        # Load the image using Pillow from raw bytes
        image = Image.open(io.BytesIO(image_bytes))

        prompt_parts = [VISION_INSTRUCTION]
        if user_prompt:
            prompt_parts.append(f"Additional context from the user: {user_prompt}")

        # Structured output instead of grounding: Gemini only has to identify
        # the object and suggest a search phrase (things it's actually good
        # at), not produce a real URL itself (which is what kept going
        # wrong). The backend builds the actual link deterministically below,
        # so it can never point to a hallucinated or dead video.
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[*prompt_parts, image],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=VisionAnalysis,
            ),
        )

        analysis = response.parsed
        if not isinstance(analysis, VisionAnalysis):
            return response.text or "Sorry, I couldn't analyze that image."

        result = f"This is a {analysis.object_name}.\n\n{analysis.description}"
        if analysis.youtube_search_query:
            search_url = (
                "https://www.youtube.com/results?search_query="
                + urllib.parse.quote_plus(analysis.youtube_search_query)
            )
            result += f"\n\nWatch tutorials: {search_url}"
        if analysis.reference_website_url.startswith("http"):
            result += f"\n\nMore info: {analysis.reference_website_url}"
        return result

    except Exception as e:
        return f"Error analyzing image with Gemini: {str(e)}"