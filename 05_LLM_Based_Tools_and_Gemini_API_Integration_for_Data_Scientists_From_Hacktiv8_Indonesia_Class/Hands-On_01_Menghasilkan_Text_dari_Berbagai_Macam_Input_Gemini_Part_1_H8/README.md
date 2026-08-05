# 🤖 05 — LLM-Based Tools & Gemini API Integration

> Hands-on exploration of the Google Gemini API — generating and understanding text, images, documents, and audio, then applying core prompt engineering techniques side-by-side on the same real input. Learned during Hacktiv8 Indonesia's **"Maju Bareng AI"** program.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-API-4285F4?logo=google&logoColor=white)
![Status](https://img.shields.io/badge/status-learning%20project-yellow)

---

## ⚡ TL;DR — 30-Second Summary

- **Problem:** Learn to work with a multimodal LLM API in practice — not just send a text prompt, but handle images, documents, audio, multi-turn conversations, and understand *why* different prompting techniques produce different quality outputs.
- **Approach:** Hands-on notebook covering both the legacy `google-generativeai` SDK and the newer `google-genai` Client SDK — text generation, streaming, image/PDF/audio understanding, stateful chat sessions, and 7 distinct prompt engineering techniques.
- **Result:** Working implementations across every input modality, plus a direct side-by-side comparison of 7 prompting techniques applied to the *exact same image*, making the practical difference between techniques concrete instead of theoretical.
- **Key Highlight:** Rather than treating prompting techniques as abstract theory, this project runs **Role-Based, Priming, Comparative, Instructional, Few-Shot, One-Shot, and Zero-Shot prompting** on one identical humpback whale photo — so the *only* variable that changes between runs is the technique itself.

---

## 💼 Skills Demonstrated

| Competency | Where it shows up |
|---|---|
| **Prompt Engineering** | 7 distinct techniques implemented and compared on the same image input |
| **Multimodal AI Processing** | Image (PIL), PDF documents (base64 + `httpx`), and audio (MP3) fed directly into Gemini |
| **SDK Integration** | Working with both the legacy `google.generativeai` SDK and the newer `google.genai` Client SDK in the same project |
| **Stateful Chat Sessions** | Multi-turn conversations with tracked history, including verifying the model actually retains and reasons over prior context |
| **Applied Problem-Solving** | Converting an unstructured price-list image into a clean, structured `pandas` DataFrame purely through iterative prompting |

---

## 📑 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Inputs & Data Types Used](#-inputs--data-types-used)
- [Project Workflow](#-project-workflow)
- [Technologies & Tools Used](#-technologies--tools-used)
- [Results & Key Insights](#-results--key-insights)
- [Prompt Engineering Techniques Applied](#-prompt-engineering-techniques-applied)
- [Tech Stack](#-tech-stack)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [What I Learned](#-what-i-learned)
- [Learning Resources](#-learning-resources)
- [Acknowledgements](#-acknowledgements)
- [Author](#-author)

---

## 📖 Project Overview

This is **Part 1** of a hands-on series exploring the Google Gemini API — the foundation stage before building more complex LLM-based tools. Rather than just calling `generate_content()` on plain text, this project deliberately works through every major *input modality* Gemini supports (text, images, PDF documents, audio) and closes with a focused exercise applying seven core prompt engineering techniques to a single real-world image, so their effects can be compared directly rather than described abstractly.

This project was built as part of Hacktiv8 Indonesia's **"Maju Bareng AI"** program.

---

## 🎯 Objectives

- Understand the fundamentals of interacting with the Gemini API using both the legacy `google-generativeai` SDK and the newer `google-genai` Client SDK.
- Implement standard and streaming text generation, and understand why streaming matters for responsiveness.
- Process multimodal inputs: images, PDF documents, and audio files.
- Build and inspect stateful, multi-turn chat sessions with tracked history.
- Apply and compare 7 distinct prompt engineering techniques on identical input to observe their real effect on output quality and structure.

---

## 📥 Inputs & Data Types Used

Instead of a single tabular dataset, this project works across several unstructured input types — closer to what a real LLM-powered tool needs to handle:

| Input Type | Example Used | Purpose |
|---|---|---|
| 📝 Text | Direct prompts (e.g., *"Explain how AI works"*) | Baseline text generation & streaming |
| 🖼️ Image | A salon price-list graphic (JPG) | Extracting structured data (a price table) from an unstructured image |
| 🖼️ Image | A humpback whale photograph | Base image for the 7-technique prompt engineering exercise |
| 📄 Document | An academic PDF (fetched via URL) | Document summarization |
| 🔊 Audio | An MP3 recording (a 1961 State of the Union address) | Audio content description |

---

## 🔄 Project Workflow

```text
Install & Configure Gemini SDK
      │
      ▼
Text Generation
      ├── Basic generate_content()
      ├── Streaming responses (stream=True)
      └── List available Gemini models
      │
      ▼
Multimodal Understanding
      ├── Image → Text (PIL + generate_content)
      │     └── Iteratively prompted: raw description → table → table + pandas code
      ├── Document (PDF) → Text (base64 + httpx)
      └── Audio (MP3) → Text (genai.upload_file)
      │
      ▼
Stateful Chat Sessions
      ├── start_chat(history=[])
      ├── Multi-turn send_message() calls
      └── Inspecting chat.history for context retention
      │
      ▼
Exercise: Prompt Engineering Techniques
      └── Applied 7 techniques to the same whale image using the new
          google.genai Client SDK: Role-Based, Priming, Comparative,
          Instructional, Few-Shot, One-Shot, Zero-Shot
```

---

## 🛠️ Technologies & Tools Used

| Tool / Library | Used for |
|---|---|
| `google.generativeai` | Legacy Gemini SDK — text generation, streaming, image/PDF/audio understanding, chat sessions |
| `google.genai` (`Client`) | Newer Gemini SDK — used specifically in the prompt engineering exercise |
| `PIL` (Pillow) | Loading and passing image files to the model |
| `httpx` + `base64` | Fetching a PDF from a URL and encoding it for the API |
| `pandas` | Structuring the extracted price-list data into a DataFrame |
| `IPython.display.Markdown` | Rendering model responses as formatted Markdown in the notebook |
| Google Colab `userdata` | Securely storing and retrieving the `GOOGLE_API_KEY` secret |

---

## 📊 Results & Key Insights

- **Streaming vs. blocking responses:** Demonstrated the visible difference between waiting for a full response versus receiving it chunk-by-chunk (`stream=True`), including a slowed-down version (`time.sleep(0.5)` between chunks) to make the streaming behavior visible in the notebook output.
- **Turning an image into structured data:** Fed the model a salon price-list image with progressively refined prompts — from no prompt at all, to *"make this into a table,"* to *"make a table, and also give me the Python code to turn it into a DataFrame."* The final prompt correctly produced working `pandas` code that reconstructs the actual price list (Hair Treatment, Hair Colouring, Hair Spa/Mask categories with real Rupiah prices).
- **Document understanding without local storage:** Summarized an external academic PDF by streaming it directly from its URL (`httpx` → `base64`) straight into the API call, with no need to save the file to disk first.
- **Genuine context retention in chat:** Verified the chat session actually reasons over history — after telling the model *"I have 2 dogs in my house,"* asking *"how many paws are in my house?"* correctly required the model to recall the earlier fact and do the math, not just answer in isolation.
- **Same input, same model, 7 different outputs:** Running all 7 prompting techniques on the identical whale photograph made the practical differences tangible — e.g., Zero-Shot gave a short direct answer, while Instructional Prompting (with a strict 4-step, Markdown-formatted requirement) produced a structured, heading-organized breakdown of species, size, and fin function.

---

## 🧠 Prompt Engineering Techniques Applied

All 7 techniques below were applied to the **same humpback whale photograph**, using the newer `google.genai` Client SDK:

| # | Technique | How it was applied |
|---|---|---|
| 1 | **Role-Based Prompting** | Asked the model to respond *as a cetologist* (marine mammal expert), shaping both the vocabulary and depth of the explanation. |
| 2 | **Priming (Contextual Prompting)** | Set up a marine biology research-vessel scenario in the Atlantic Ocean before asking the model to identify the animal and its behavior. |
| 3 | **Comparative Prompting** | Asked the model to compare the humpback whale in the photo against a Bottlenose Dolphin across body size, pectoral fin structure, and communication style. |
| 4 | **Instructional Prompting** | Gave a strict, numbered 4-step instruction set (species ID → size summary → pectoral fin function → Markdown-formatted output), forcing a specific output structure. |
| 5 | **Few-Shot Prompting** | Provided two labeled example analyses (a dolphin, a turtle) before asking the model to analyze the whale in the exact same format. |
| 6 | **One-Shot Prompting** | Provided exactly one labeled example (a dolphin) before asking for the same classification format applied to the whale. |
| 7 | **Zero-Shot Prompting** | Asked directly, *"what type of marine mammal is in this image?"* — no prior examples, no extra context. |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| LLM / Generative AI | Google Gemini API (`google-generativeai`, `google-genai`) |
| Multimodal Handling | `Pillow (PIL)`, `httpx`, `base64` |
| Data Handling | `pandas` |
| Environment | Google Colab |

---

## ▶️ How to Run

### Option 1 — Run on Google Colab (Recommended)

1. Open the notebook directly in Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoihansLab/Machine-Learning-Projects/blob/main/05_LLM_Based_Tools_and_Gemini_API_Integration_for_Data_Scientists_From_Hacktiv8_Indonesia_Class/Hands-On_01_Menghasilkan_Text_dari_Berbagai_Macam_Input_Gemini_Part_1_H8/Hands-On_01_Menghasilkan_Text_dari_Berbagai_Macam_Input_Gemini_Part_1_H8.ipynb)
2. Add your own Gemini API key as a Colab secret named `GOOGLE_API_KEY`.
3. Run all cells from top to bottom.

### Option 2 — Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/RoihansLab/Machine-Learning-Projects.git
cd Machine-Learning-Projects/05_LLM_Based_Tools_and_Gemini_API_Integration_for_Data_Scientists_From_Hacktiv8_Indonesia_Class
```

**2. Install the required Python libraries**
```bash
pip install google-generativeai google-genai pillow httpx pandas jupyter
```

**3. Set your API key**
The notebook uses Colab's `userdata.get('GOOGLE_API_KEY')` to fetch the key. Running locally, replace that line with your own key, e.g. via an environment variable:
```python
import os
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
```

**4. Launch Jupyter Notebook**
```bash
jupyter notebook Hands-On_01_Menghasilkan_Text_dari_Berbagai_Macam_Input_Gemini_Part_1_H8.ipynb
```

---

## 🗂️ Project Structure

```text
05_LLM_Based_Tools_and_Gemini_API_Integration_for_Data_Scientists_From_Hacktiv8_Indonesia_Class/
│
├── README.md
│
└── Hands-On_01_Menghasilkan_Text_dari_Berbagai_Macam_Input_Gemini_Part_1_H8/
    └── Hands-On_01_Menghasilkan_Text_dari_Berbagai_Macam_Input_Gemini_Part_1_H8.ipynb
        → Text generation & streaming, image/PDF/audio understanding,
          stateful chat sessions, and the 7-technique prompt engineering exercise
```

*(Structured as "Part 1" of a hands-on series — later parts are expected to be added as folders alongside this one as the LLM tooling track progresses.)*

---

## 💡 What I Learned

- The practical difference between blocking and streaming responses, and why streaming matters for a responsive user experience.
- How to work with multimodal input beyond plain text — and that each modality needs its own encoding approach (a raw `PIL` object for images, base64-encoded bytes for documents, an uploaded file reference for audio).
- The difference between the legacy `google.generativeai` SDK and the newer `google.genai` Client-based SDK, and how to use both in the same project.
- How chat session history is actually tracked and passed back to the model to maintain context across turns.
- Concretely — not just in theory — how the *same* input and *same* underlying model can produce meaningfully different output structure and depth depending purely on which prompting technique is used.

---

## 📚 Learning Resources

- [Explore vision capabilities with the Gemini API](https://ai.google.dev/gemini-api/docs/vision?lang=python)
- [Explore audio capabilities with the Gemini API](https://ai.google.dev/gemini-api/docs/audio?lang=python)
- Hacktiv8 Indonesia — **"Maju Bareng AI"** program

---

## 🙏 Acknowledgements

Terima kasih banyak untuk **Hacktiv8 Indonesia**, khususnya program **"Maju Bareng AI"**, atas ilmu dan ide project yang menjadi dasar dari hands-on ini.

Materi pembelajaran pada notebook ini disusun oleh **[Sardi Irfansyah](https://www.linkedin.com/in/sirfansyah/)** — terima kasih atas materi yang jelas dan mudah diikuti untuk mempelajari Gemini API dari dasar.

---

## 👤 Author

**Roihan Saputra**
*Aspiring AI/ML Engineer*
GitHub: [https://github.com/RoihansLab](https://github.com/RoihansLab)

Open to feedback, collaboration, or a conversation about this project — feel free to reach out via GitHub.
