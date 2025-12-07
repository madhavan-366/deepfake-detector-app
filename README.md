# Explainable Deepfake Detector

A Full-Stack AI application that detects deepfake faces using Deep Learning (XceptionNet).

🔗 **Live Demo:** [Click Here to View App](https://madhavan-366.github.io/deepfake-detector-app/)

## 🚀 Key Features

* **🔍 Deep Learning Analysis:** Uses a fine-tuned XceptionNet model to classify images as "Real" or "Fake".
* **📄 AI-Powered Reporting:** Generates a downloadable PDF forensic report using **Google Gemini AI** to explain *why* an image is likely fake.
* **🔒 Secure Authentication:** Complete Login/Signup system using JWT (JSON Web Tokens) and MongoDB.
* **⚡ Real-Time Processing:** Fast API communication between Node.js and the Python AI Engine.
* **📱 Responsive Design:** Fully optimized for desktop and mobile devices using Tailwind CSS.

---

## 🛠️ Tech Stack & Architecture

This project follows a **Microservices-style Architecture**:

### 1. Frontend (Client)
* **Tech:** React.js, Vite, Tailwind CSS, Axios.
* **Hosting:** GitHub Pages.
* **Role:** User interface for uploading images and viewing results.

### 2. Backend (API Gateway)
* **Tech:** Node.js, Express.js, JWT, Multer.
* **Database:** MongoDB Atlas (Cloud).
* **Hosting:** Render.
* **Role:** Handles user auth, manages file uploads, and coordinates between the Frontend and the AI Engine.

### 3. AI Service (The Brain)
* **Tech:** Python, PyTorch, PIL (Pillow), Flask/Gradio.
* **Model:** Custom XceptionNet trained on the FaceForensics++ dataset.
* **Hosting:** Hugging Face Spaces.
* **Role:** Performs the actual deep learning inference on the image.

---

## ⚙️ How to Run Locally

If you want to run this project on your local machine, follow these steps:

## Prerequisites
* Node.js & npm installed
* Python 3.8+ installed
* MongoDB URI

## 1. Clone the Repository
```bash
git clone [https://github.com/madhavan-366/deepfake-detector-app.git](https://github.com/madhavan-366/deepfake-detector-app.git)
cd deepfake-detector-app
