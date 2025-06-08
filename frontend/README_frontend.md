
# 🧠 AI-Powered Misinformation & Deepfake Detector – Frontend

This is the **Next.js frontend** for the real-time AI-based misinformation and deepfake detection system. It provides a modern, responsive, and interactive UI for users to upload media/text and instantly receive results powered by advanced NLP and computer vision models running in the backend.

---

## ⚙️ Tech Stack

- **Framework**: Next.js (React, TypeScript)
- **Styling**: Tailwind CSS
- **Backend API**: FastAPI (runs separately)
- **Design**: Fully responsive, mobile-friendly

---

## 🚀 Features

- Upload **text**, **images**, and **videos** for real-time analysis
- Get instant **verdicts** with explainability
- Beautiful UI with dark mode support
- Clean architecture with reusable components

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/misinfo-detector-frontend.git
cd misinfo-detector-frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Set environment variables

Create a `.env.local` file in the root and define:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

> 🔁 Update the URL if your backend is hosted elsewhere.

---

## 🧪 Run the App

```bash
npm run dev
```

Your app will be live at [http://localhost:3000](http://localhost:3000)

---

## 📁 Do Not Upload to GitHub

- `.next/` – Build cache
- `node_modules/` – Regenerated via `npm install`

These are already excluded in `.gitignore`.

---

## 🧠 Backend Repo

The backend (FastAPI, ML models) is in a separate repository:
[👉 View Backend Repo](https://github.com/yourusername/misinfo-detector-backend)

---

## 📬 Contact

Have questions or suggestions? Open an issue or email me at [you@example.com](mailto:you@example.com)

---

## 📄 License

MIT License – Feel free to use, modify, and build on top of this.
