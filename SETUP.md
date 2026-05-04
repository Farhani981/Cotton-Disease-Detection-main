# Cotton Disease Detection - Setup Guide

## Prerequisites

Before setting up the application, ensure you have:

- Python 3.8 or higher
- Git with Git LFS support
- pip (Python package manager)

## Git LFS Setup

This project uses Git LFS (Large File Storage) to handle the machine learning model file (`resnet50.h5`).

### Install Git LFS

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

### Clone and Pull LFS Files

```bash
git clone https://github.com/anomusly/Cotton-Disease-Detection.git
cd Cotton-Disease-Detection
git lfs pull
```

## Environment Variables Setup

This application uses environment variables to store sensitive information like API keys. Follow these steps to set up your environment:

### 1. Create Environment File

Copy the example environment file and fill in your actual values:

```bash
cp .env.example .env
```

### 2. Configure Your API Keys

Edit the `.env` file and replace the placeholder values with your actual keys:

```env
# Stripe API Keys (Get these from your Stripe Dashboard)
STRIPE_PUBLISHABLE_KEY=pk_test_your_actual_publishable_key_here
STRIPE_SECRET_KEY=sk_test_your_actual_secret_key_here

# Flask Secret Key (Generate a secure random string)
FLASK_SECRET_KEY=your_secure_random_secret_key_here

# Stripe Price IDs (Create these in your Stripe Dashboard)
STRIPE_PREMIUM_PRICE_ID=price_your_actual_premium_price_id_here
STRIPE_DIAMOND_PRICE_ID=price_your_actual_diamond_price_id_here
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

## Important Security Notes

- **Never commit the `.env` file to version control**
- The `.env` file is already included in `.gitignore`
- Always use environment variables for sensitive data
- Regenerate your API keys if they were ever exposed in version control

## Getting Stripe API Keys

1. Go to [Stripe Dashboard](https://dashboard.stripe.com/)
2. Navigate to Developers > API keys
3. Copy your Publishable key and Secret key
4. For Price IDs, go to Products and create your subscription products

## Troubleshooting

If you encounter issues:

1. Make sure all environment variables are set correctly
2. Verify your Stripe keys are valid and from the correct environment (test/live)
3. Check that your Price IDs exist in your Stripe account
