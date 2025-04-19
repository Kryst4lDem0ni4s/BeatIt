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
- **FastAPI** (API)
- **PYTHON**
- **Facebook AudioCraft** (Music Generation)
- **FireStore** (User Data Storage)

### Frontend
- Flutter & Dart (Cross-platform App Interface)
- D3-like Visualizations (Handled via Flutter packages for charts/animations)
- Flutter Test (For widget and unit testing)

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

# Backend setup
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## 👥 Team Roles

  **Khwaish**  
  *Backend Engineer and AI Architect*  
  AI model development, audio processing pipelines, API development, Planning and Design

- **Manvendra**  
  *Data Architect*  
  Data Scientist, feature extraction pipelines

- **Bhavika & Karthikey**  
  *Frontend Team*  
  User interface, visualization systems, user experience design, integration

**🚨 Important Note**  
This project uses Facebook's AudioCraft under non-commercial research license. Commercial use requires direct authorization from Meta.
```
