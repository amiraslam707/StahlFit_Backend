# StahlFit — Hybrid AI Fitness Chatbot

*A gym-focused conversational AI that answers instantly from a locally-trained model when it's confident, and falls back to Google Gemini when it isn't.*

## Objective

To build a hybrid chatbot backend for a fitness app that combines a custom-trained neural network intent classifier with a cloud LLM (Gemini) fallback — so common fitness questions get fast, pre-validated answers, while anything outside the training data still gets a useful response instead of a dead end.

## How It Works

1. An incoming message is tokenized and lemmatized with NLTK, then passed through a locally-trained Keras/TensorFlow neural network to predict the closest matching intent.
2. If the model's confidence for that prediction is **≥ 0.50** *(verify this against `main.py` — this value has changed during development)*, the app returns the matching pre-written response from `intents.json` directly, with no external API call.
3. If confidence falls below that threshold, the message is forwarded to the Gemini API instead, so out-of-scope or complex questions still get answered.
4. Every response includes a `response_source` field (`"Local Trained Model"` or `"Cloud LLM"`) so the routing decision is always visible — useful for demoing or grading the fallback logic.

## Features

- Local intent classification covering a Push/Pull/Legs split — chest, back, biceps, triceps, shoulders, legs — plus nutrition and general Q&A.
- Each exercise intent stores posture cues, step-by-step instructions, common mistakes, and a linked demonstration video.
- Automatic Gemini fallback for anything the local model isn't confident about.
- Image-based gym equipment recognition via a vision endpoint powered by Gemini Vision.
- JWT-based authentication API (signup, login, `/me` profile with avatar upload), implemented and testable through `/docs`.

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Local NLP model | TensorFlow / Keras (Sequential — Dense + Dropout layers, SGD optimizer) |
| Text preprocessing | NLTK (tokenization, lemmatization) |
| Cloud LLM | Google Gemini API (`google-generativeai`) |
| Auth & storage | SQLite + SQLAlchemy, bcrypt, JWT |

## Project Structure

```
AI_Chat/
├── main.py                # FastAPI app, routing logic, endpoints
├── train.py                # Trains the intent classifier from intents.json
├── intents.json             # Local knowledge base (dataset)
├── chatbot_model.h5         # Trained model weights
├── words.pkl                 # Vocabulary used by the model
├── classes.pkl                # Intent labels used by the model
├── database.py, models.py, schemas.py, security.py, auth.py   # Auth backend
├── requirements.txt
└── nltk_data/                 # Local punkt / punkt_tab tokenizer data
```

## Installation

1. Install Python 3.11.x.
2. Create and activate a virtual environment in the project folder:
   ```powershell
   python -m venv venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make sure NLTK data is available:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```
   (Skip this if `nltk_data/` is already included in this submission.)
5. Create a `.env` file in the project root with your Gemini key:
   ```
   GEMINI_API_KEY=your_key_here
   ```
6. Only if `intents.json` was edited, regenerate the model:
   ```bash
   python train.py
   ```

## Running the Project

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI, where the chat endpoint can be tested directly without the mobile frontend.

## Expected Output

**Example 1 — handled locally**
Request: `{"message": "How do I do a bench press?"}`
The message matches the "chest" intent above the confidence threshold, so the pre-written response is returned directly with no Gemini call, tagged `response_source: "Local Trained Model"`.

**Example 2 — handled by Gemini**
Request: `{"message": "What's a good bulking diet during Ramadan?"}`
No local intent clears the threshold, so the message is forwarded to Gemini and its answer returned instead, tagged `response_source: "Cloud LLM"`.

*(Exact JSON field names/shape should be checked against the live `main.py` output before this goes in the report.)*

## Dataset

`intents.json` — a custom-built dataset of gym/fitness intents. Each entry has a `tag`, a list of `patterns` (example user phrasings), and either plain `responses` or an `exercises` array with structured `posture` / `instructions` / `common_mistakes` / `video_url` fields.

## Trained Model

`chatbot_model.h5` (weights), plus `words.pkl` and `classes.pkl` (vocabulary and intent labels) — generated from `intents.json` by `train.py`.

## Known Limitations

- The backend is reachable over the local network via a manually-set IP, updated as needed.
- Authentication is fully built on the backend but not yet wired into the mobile frontend.
- Workout logging is planned but not yet implemented.





REQUIREMNETS.TXT FOR THE PROJECT


# Best-effort starting point based on the documented stack.
# For the authoritative version, activate your working venv and run:
#   pip freeze > requirements.txt
# then strip out anything Windows-only (e.g. pywin32) before submitting.

fastapi
uvicorn[standard]
tensorflow==2.15.0
keras==2.15.0
nltk
google-generativeai
SQLAlchemy
bcrypt
PyJWT
python-multipart
python-dotenv