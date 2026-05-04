# 🌱 Cotton Disease Detection System

<div align="center">

![Cotton Disease Detection](static/assets/img/logo.png)

**AI-Powered Cotton Plant Disease Detection using Deep Learning**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Diseases](#supported-diseases)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

## 🌟 Overview

The Cotton Disease Detection System is an AI-powered web application that helps farmers and agricultural experts identify diseases in cotton plants through image analysis. Using advanced deep learning techniques with a ResNet-50 model, the system can accurately detect and classify various cotton plant diseases from uploaded leaf images.

### 🎯 Key Objectives

- **Early Disease Detection**: Identify cotton diseases before they spread
- **Accurate Classification**: Distinguish between different disease types
- **User-Friendly Interface**: Simple upload and prediction workflow
- **Detailed Reporting**: Generate PDF reports for record-keeping
- **Subscription Management**: Tiered access with Stripe integration

## ✨ Features

### 🔍 Core Functionality
- **AI-Powered Detection**: ResNet-50 based deep learning model
- **Real-time Prediction**: Instant disease classification with confidence scores
- **Multi-Disease Support**: Detects 5 major cotton diseases
- **Image Preprocessing**: Automatic image validation and enhancement

### 👥 User Management
- **User Authentication**: Secure login/registration system
- **Role-Based Access**: Admin and regular user roles
- **Upload Limits**: Tiered subscription plans (Free, Premium, Diamond)
- **Session Management**: Secure user sessions with Flask-Login

### 📊 Reporting & Analytics
- **PDF Report Generation**: Detailed prediction reports with ReportLab
- **Prediction History**: Track all user predictions
- **Admin Dashboard**: User analytics and system monitoring
- **Download Reports**: Export predictions as PDF files

### 💳 Subscription System
- **Stripe Integration**: Secure payment processing
- **Multiple Plans**: Free (3 uploads), Premium (20 uploads), Diamond (100 uploads)
- **Payment History**: Track subscription and payment records
- **Automatic Upgrades**: Seamless plan transitions

## 🦠 Supported Diseases

| Disease | Description | Symptoms |
|---------|-------------|----------|
| **Aphids** | Small insects that feed on plant sap | Yellowing leaves, stunted growth |
| **Army Worm** | Caterpillar pest that damages leaves | Irregular holes in leaves |
| **Bacterial Blight** | Bacterial infection | Dark spots with yellow halos |
| **Powdery Mildew** | Fungal disease | White powdery coating on leaves |
| **Target Spot** | Fungal infection | Circular spots with concentric rings |

## 🛠 Technology Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask**: Web framework
- **TensorFlow/Keras**: Deep learning framework
- **SQLAlchemy**: Database ORM
- **SQLite**: Database (development)

### Frontend
- **HTML5/CSS3**: Structure and styling
- **Bootstrap 5**: Responsive design framework
- **JavaScript (ES6+)**: Interactive functionality
- **jQuery**: DOM manipulation

### AI/ML
- **ResNet-50**: Pre-trained CNN model
- **MobileNetV2**: Image filtering model
- **OpenCV**: Image processing
- **NumPy**: Numerical computations

### Payment & Security
- **Stripe**: Payment processing
- **Flask-Login**: User session management
- **Werkzeug**: Password hashing
- **python-dotenv**: Environment variable management

### Additional Tools
- **ReportLab**: PDF generation
- **Pillow**: Image manipulation
- **Flask-WTF**: CSRF protection

## 🚀 Installation

### Prerequisites

- Python 3.12.5
- pip (Python package manager)
- Git
- Git LFS (for model files)

### Step 1: Clone the Repository

```bash
git clone https://github.com/anomusly/Cotton-Disease-Detection.git
cd Cotton-Disease-Detection
```

### Step 2: Install Git LFS (if not already installed)

```bash
# On Windows (using Git for Windows)
git lfs install

# On macOS (using Homebrew)
brew install git-lfs
git lfs install

# On Ubuntu/Debian
sudo apt install git-lfs
git lfs install
```

### Step 3: Pull LFS Files

```bash
git lfs pull
```

### Step 4: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables Setup

1. **Copy the example environment file:**
```bash
cp .env.example .env
```

2. **Edit `.env` file with your actual values:**
```env
# Stripe API Keys (Get from Stripe Dashboard)
STRIPE_PUBLISHABLE_KEY=pk_test_your_publishable_key_here
STRIPE_SECRET_KEY=sk_test_your_secret_key_here

# Flask Secret Key (Generate a secure random string)
FLASK_SECRET_KEY=your_secure_random_secret_key_here

# Stripe Price IDs (Create in Stripe Dashboard)
STRIPE_PREMIUM_PRICE_ID=price_your_premium_price_id_here
STRIPE_DIAMOND_PRICE_ID=price_your_diamond_price_id_here
```

### Database Setup

```bash
# Initialize the database
python create_db.py
```

## 🎮 Usage

### Starting the Application

```bash
python app.py
```

The application will be available at `http://localhost:5000`

### Using the System

1. **Homepage**: Navigate to the main page to learn about the system
2. **Get Started**: Click "Get Started" to access the prediction interface
3. **Upload Image**: Choose a clear cotton leaf image (PNG, JPG, JPEG)
4. **Predict**: Click "Predict" to analyze the image
5. **View Results**: See the disease prediction with confidence score
6. **Download Report**: Generate and download a PDF report (for registered users)

### User Registration

1. Click "Login" → "Signup"
2. Fill in username, email, and password
3. Login with your credentials
4. Enjoy 3 free predictions
5. Subscribe for more uploads

## 📡 API Endpoints

### Authentication
- `GET /login` - Login page
- `POST /login` - Process login
- `GET /register` - Registration page
- `POST /register` - Process registration
- `GET /logout` - Logout user

### Core Functionality
- `GET /` - Homepage
- `GET /try` - Prediction interface
- `POST /predict` - Image prediction endpoint
- `GET /my-reports` - User's prediction history
- `GET /download_report/<id>` - Download PDF report

### Subscription
- `GET /subscription` - Pricing page
- `POST /create_checkout_session` - Create Stripe session
- `GET /payment_success` - Payment success handler
- `GET /payment_cancel` - Payment cancellation handler

### Admin
- `GET /admin_dashboard` - Admin dashboard (admin only)

## 📁 Project Structure

```
Cotton-Disease-Detection/
├── app.py                 # Main Flask application
├── create_db.py          # Database initialization
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── .gitignore           # Git ignore rules
├── README.md            # Project documentation
├── SETUP.md             # Setup instructions
├── resnet50.h5          # Trained model file (Git LFS)
├── static/              # Static assets
│   └── assets/
│       ├── css/         # Stylesheets
│       ├── js/          # JavaScript files
│       ├── img/         # Images
│       └── vendor/      # Third-party libraries
├── templates/           # HTML templates
│   ├── base.html        # Base template
│   ├── index.html       # Homepage
│   ├── try.html         # Prediction interface
│   ├── login.html       # Login page
│   ├── register.html    # Registration page
│   ├── profile.html     # User profile
│   ├── subscription.html # Pricing page
│   ├── my-reports.html  # Reports page
│   ├── admin_dashboard.html # Admin panel
│   └── [disease].html   # Disease info pages
├── uploads/             # User uploaded images
├── instance/            # Database files
└── __pycache__/         # Python cache files
```

## 📸 Screenshots

### Homepage
![Homepage](static/assets/img/home_page.jpg)

### Prediction Interface
The clean and intuitive interface allows users to upload cotton leaf images and get instant predictions.

### Disease Information
Comprehensive information about each supported disease with symptoms and treatment recommendations.

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting

## 👥 Team

**Project ID: BSE-2132**

| Name | Role | Contact |
|------|------|---------|
| **Jannat Yousaf** | Lead Developer | - |
| **Sarah Riaz** | ML Engineer | - |
| **Usama Dar** | Frontend Developer | - |

**Contact Information:**
- 📧 Email: finitelyi8@gmail.com
- 📱 Phone: +92-8123920743

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- Flask community for the web framework
- Bootstrap team for the responsive design framework
- Stripe for payment processing capabilities
- All contributors and testers

---

<div align="center">

**Made with ❤️ by Team BSE-2132**

[⬆ Back to Top](#-cotton-disease-detection-system)

</div>
