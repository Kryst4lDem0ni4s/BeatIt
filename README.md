# BeatIt
AI Music Generator for Musicians that need assistance

## DEMO VIDEO:
<video controls src="WhatsApp Video 2025-04-17 at 4.29.37 PM.mp4" title="Title"></video>
```markdown
# BEAT IT - AI Music Generation System

![Project Banner](https://via.placeholder.com/1200x400.png?text=Beat+It+AI+Music+Generator)

An end-to-end AI-powered music composition system that combines textual descriptions, audio references, and musical parameters to generate custom tracks.

## 🔥 Key Features
- **Multimodel Input Support** (Text + Audio + Parameters)
- **Vocal Synthesis** with pitch control
- **Genre Avoidance System**
- **Lyrics Integration** (Custom/Auto-generated)
- **Professional Music Parameters** (Tempo, Key, Instrumentation)
- **Reference Track Analysis** (MP3/WAV decomposition)

## 🛠 Technical Stack
### Backend
- **Flask** (REST API)
- **PYTHON**
- **Facebook AudioCraft** (Music Generation)
- **Redis** (Task Queue)
- **MongoDB** (User Data Storage)
- **AWS S3** (Audio Storage)

### Frontend
- Flutter & Dart (Cross-platform App Interface)
- Waveform Visualizer (Custom Widget using Audio Data)
- D3-like Visualizations (Handled via Flutter packages for charts/animations)
- Flutter Test (For widget and unit testing

## 🌐 System Architecture
```
┌─────────────┐       ┌───────────────┐       ┌───────────────┐
│   Frontend  │       │  API Gateway  │       │  Audio Storage│
└──────┬──────┘       └──────┬────────┘       └──────┬────────┘
       │                      │                       │
       │ POST /generate       │                       │
       ├─────────────────────►│                       │
       │                      │                       │
       │           ┌──────────┴─────────┐             │
       │           │  Feature Extraction│             │
       │           │  (NER Model)       │             │
       │           └──────────┬─────────┘             │
       │                      │                       │
       │           ┌──────────▼─────────┐             │
       │           │ Audio Analysis     │             │
       │           │ (AudioCraft)       │             │
       │           └──────────┬─────────┘             │
       │                      │                       │
       │           ┌──────────▼─────────┐             │
       │           │ Music Synthesis    ├─────────────┤
       │           │ (AudioCraft)       │   Upload    │
       │           └──────────┬─────────┘             │
       │                      │                       │
       │ 200 OK (Audio URL)   │                       │
       ◄──────────────────────┤                       │
                              

```

## 🚀 Installation
```
# Clone repo
git clone https://github.com/yourusername/beat-it.git

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

## ⚙ Configuration
`.env` Template:
```
# AudioCraft
AUDIOCRAFT_MODEL_PATH=models/audiocraft
NER_MODEL_PATH=models/ner

# AWS
S3_BUCKET=beat-it-audio
AWS_ACCESS_KEY=your_key
AWS_SECRET_KEY=your_secret

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
```

## 📡 API Documentation
**Endpoint**  
`POST /api/v1/generate`

**Sample Request**
```
{
  "generation_type": "instrumental_vocal",
  "lyrics_source": "custom",
  "input_type": "both",
  "prompt": "Arabian Gothic Pop Soul with gamified elements",
  "reference_audio": "",
  "custom_lyrics": "Fly through the night...",
  "duration": 180.5,
  "instruments": ["oud", "synth", "darbuka"],
  "tempo": 112,
  "key": "C# minor",
  "avoid": {"genres": ["reggaeton"]},
  "artists": "Beyoncé meets Tool",
  "vocal_pitch": "Androgynous Alto",
  "style": "Indo-Western Fusion"
}
```

**Response**
```
{
  "status": "processing",
  "track_id": "BEAT-8932",
  "queue_position": 5,
  "webhook_url": "https://api.beat.it/status/BEAT-8932"
}
```

## 👥 Team Roles
- **Manvendra**  
  *AI/ML Architect*  
  Model fine-tuning, feature extraction pipelines

- **Khwaish**  
  *Backend Engineer*  
  API development, audio processing workflows

- **Bhavika & Karthikey**  
  *Frontend Team*  
  User interface, visualization systems

## 📍 Roadmap
- [ ] Phase 1: Core Generation Engine (Q3 2025)
- [ ] Phase 2: Collaborative Editing (Q4 2025) 
- [ ] Phase 3: DAW Integration Plugin (Q1 2026)
- [ ] Phase 4: Mobile App (Q2 2026)

## 📜 License
MIT License - See [LICENSE.md](LICENSE.md) for details

---

**🚨 Important Note**  
This project uses Facebook's AudioCraft under non-commercial research license. Commercial use requires direct authorization from Meta.
```

---
Answer from Perplexity: pplx.ai/share
