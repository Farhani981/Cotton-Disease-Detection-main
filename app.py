import os,csv,requests, base64,json
import io
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from flask import session, request,Response,make_response
from reportlab.lib import colors
from reportlab.lib.utils import simpleSplit,ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime,timedelta
from flask_login import UserMixin
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False

try:
    from tensorflow.keras.preprocessing import image
except ImportError:
    image = None

from PIL import Image
import numpy as np
from flask_wtf.csrf import CSRFProtect
from sqlalchemy.orm import joinedload
from collections import defaultdict
from sqlalchemy import func,and_

MAX_UPLOADS_GUEST = 1

# ------------------- App Config -------------------

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'fallback-secret-key')

# Database Configuration
db_type = os.getenv('DB_TYPE', 'sqlite')
if db_type == 'mysql':
    db_user = os.getenv('DB_USER', 'root')
    db_password = os.getenv('DB_PASSWORD', '')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_name = os.getenv('DB_NAME', 'cotton_disease_db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'

basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "fallback-secret-key")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
# In app.py or settings
app.config['WTF_CSRF_HEADERS'] = ['X-CSRFToken']
csrf = CSRFProtect(app)

# Mail Configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)
s = URLSafeTimedSerializer(app.config['SECRET_KEY'])


db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

def send_notification_email(to, subject, body):
    try:
        msg = Message(subject, recipients=[to])
        msg.body = body
        mail.send(msg)
        return True
    except Exception as e:
        log_debug(f"Failed to send email to {to}: {str(e)}")
        print(f"Error sending email: {e}")
        return False

# Load the trained model (TFLite or H5)
MODEL_PATH_TFLITE = 'resnet50.tflite'
MODEL_PATH_H5 = 'resnet50.h5'

interpreter = None
input_details = None
output_details = None
model_h5 = None

if TFLITE_AVAILABLE and os.path.exists(MODEL_PATH_TFLITE):
    try:
        interpreter = tflite.Interpreter(model_path=MODEL_PATH_TFLITE)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        log_debug("TFLite model loaded successfully.")
        print("TFLite model loaded successfully.")
    except Exception as e:
        log_debug(f"Error loading TFLite model: {e}")
        interpreter = None

if interpreter is None and os.path.exists(MODEL_PATH_H5):
    try:
        from tensorflow.keras.models import load_model
        model_h5 = load_model(MODEL_PATH_H5)
        log_debug("H5 model loaded successfully.")
        print("H5 model loaded successfully.")
    except Exception as e:
        log_debug(f"Error loading H5 model: {e}")
        model_h5 = None
def is_plant_image(img_path):
    """
    Simple image file validation - accept any image file.
    """
    try:
        # Check if file exists
        if not os.path.exists(img_path):
            print(f"File not found: {img_path}")
            return False
        
        # Check file size
        file_size = os.path.getsize(img_path)
        if file_size < 100:  # At least 100 bytes
            print(f"File too small: {file_size} bytes")
            return False
        
        # Accept any file with valid image extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.jfif'}
        file_ext = os.path.splitext(img_path)[1].lower()
        
        if file_ext not in valid_extensions:
            log_debug(f"Unusual extension: {file_ext} - Accepting anyway")
        
        log_debug(f"Image accepted: {img_path} ({file_size} bytes)")
        return True
        
    except Exception as e:
        log_debug(f"Error validating image: {e}")
        return False

def model_predict(img_path, confidence_threshold=0.55):
    log_debug(f"Starting prediction for: {img_path}")
    
    # 1. Try TFLite Prediction (Fast & Light)
    if interpreter is not None:
        try:
            log_debug("Using TFLite interpreter...")
            img = Image.open(img_path).convert('RGB').resize((224, 224))
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            
            confidence = np.max(prediction)
            results_index = np.argmax(prediction, axis=1)[0]
            
            disease_labels = {
                0: "The leaf shows signs of Aphids",
                1: "The leaf shows signs of Army Worm",
                2: "The leaf shows signs of Bacterial Blight",
                3: "The leaf is Healthy",
                4: "The leaf shows signs of Powdery Mildew",
                5: "The leaf shows signs of Target Spot"
            }
            
            result_text = f"{disease_labels.get(results_index, 'Unknown disease')} (Confidence: {confidence*100:.2f}%)"
            log_debug(f"TFLite Prediction result: {result_text}")
            return result_text
        except Exception as e:
            log_debug(f"TFLite prediction error: {e}")

    # 2. Fallback to H5 Prediction (If TFLite fails and H5 is available)
    if model_h5 is not None:
        try:
            log_debug("Falling back to H5 model...")
            loaded_image = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(loaded_image) / 255.0
            x = np.expand_dims(x, axis=0)
            prediction = model_h5.predict(x)
            
            confidence = np.max(prediction)
            results_index = np.argmax(prediction, axis=1)[0]
            
            disease_labels = {
                0: "The leaf shows signs of Aphids",
                1: "The leaf shows signs of Army Worm",
                2: "The leaf shows signs of Bacterial Blight",
                3: "The leaf is Healthy",
                4: "The leaf shows signs of Powdery Mildew",
                5: "The leaf shows signs of Target Spot"
            }
            
            return f"{disease_labels.get(results_index, 'Unknown disease')} (Confidence: {confidence*100:.2f}%)"
        except Exception as e:
            log_debug(f"H5 prediction error: {e}")

    return "AI Model Not Available - Please ensure model files are present. (Confidence: 0%)"

def analyze_image_with_gemini(img_path):
    """
    Uses Gemini 1.5 Flash to analyze the cotton leaf image for diseases as a fallback.
    """
    try:
        with open(img_path, "rb") as image_file:
            img_data = base64.b64encode(image_file.read()).decode('utf-8')

        prompt = """
        You are an expert plant pathologist specializing in cotton diseases. 
        Analyze the provided image of a cotton leaf.
        Identify the disease if present. The possible diseases are: 
        Aphids, Army Worm, Bacterial Blight, Healthy, Powdery Mildew, Target Spot.
        
        Provide the result in this exact format:
        The leaf shows signs of [Disease Name] (Confidence: [Estimated Confidence]%)
        
        If it's healthy, say: The leaf is Healthy (Confidence: [Estimated Confidence]%)
        
        Be very specific and accurate.
        """

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_data
                        }
                    }
                ]
            }]
        }

        response = requests.post(GEMINI_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract text from Gemini response
        prediction_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
        log_debug(f"Gemini Fallback Prediction: {prediction_text}")
        return prediction_text
    except Exception as e:
        log_debug(f"Error in Gemini analysis: {str(e)}")
        return f"Gemini Analysis Error (Confidence: 0%)"

# ------------------- Models -------------------

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    upload_attempts = db.Column(db.Integer, default=0)
    is_admin = db.Column(db.Integer, default=0)  # Changed from Boolean to Integer for MySQL compatibility
    role = db.Column(db.String(50), default='user')
    reports = db.relationship('Report', backref='user', lazy=True)


    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password,password)


class SignupHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class LoginHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    username = db.Column(db.String(150))
    success = db.Column(db.Integer)  # Changed from Boolean to Integer for MySQL compatibility
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class SessionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    username = db.Column(db.String(150))
    action = db.Column(db.String(50))  # login or logout
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class ForumMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    username = db.Column(db.String(150), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())


@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    if current_user.is_admin:
        report = Report.query.filter_by(id=report_id).first()
    else:
        report = Report.query.filter_by(id=report_id, user_id=current_user.id).first()

    if not report:
        flash("Report not found or unauthorized access.", "danger")
        return redirect(url_for('home'))

    user = User.query.filter_by(id=report.user_id).first()
    username = user.username if user else "N/A"
    image_path = image_path = os.path.join("uploads", report.filename)
    # Format confidence safely
    try:
        confidence_value = float(report.confidence)
        confidence_text = f"{confidence_value:.2f}%"
    except (TypeError, ValueError):
        confidence_text = "N/A"

    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Colors
    header_color = colors.HexColor('#1a4d2e')
    box_bg_color = colors.HexColor('#f1f5f9')
    text_color = colors.HexColor('#0f172a')

    # Header
    p.setFillColor(header_color)
    p.rect(0, height - 80, width, 80, fill=True, stroke=False)
    p.setFillColor(colors.white)
    p.setFont("Helvetica-Bold", 20)
    p.drawString(69, height - 50, "Cotton Disease Detection Report")
    
    
  # Logo (draw AFTER header)
    logo_path = os.path.join("static", "assets", "img", "logo.png")
    logo_width = 50
    logo_height = 50
    logo_x = 20  # Align left
    logo_y = height - 70  # Lower than top edge

    if os.path.exists(logo_path):
        try:
           p.drawImage(ImageReader(logo_path), logo_x, logo_y, width=logo_width, height=logo_height, mask='auto')
        except Exception as e:
          print(f"Logo not embedded: {e}")

    # Image
    image_width = 280
    image_height = 200
    image_x = (width - image_width) / 2
    image_y = height - 100 - image_height  # 100 below top

    if os.path.exists(image_path):
        try:
            p.drawImage(ImageReader(image_path), image_x, image_y, width=image_width, height=image_height)

        except Exception as e:
            print(f"Error embedding image: {e}")
            p.setFont("Helvetica", 10)
            p.drawString(image_x, image_y + 70, "Could not embed image.")
    else:
        p.setFont("Helvetica", 10)
        p.drawString(image_x, image_y + 70, "Image not found.")

    # Report Info Box - dynamically below the image
    box_top_y = image_y - 40  # Space below image
    box_height = 310
    x, y = 50, box_top_y - box_height
    box_width = width - 100

    p.setFillColor(box_bg_color)
    p.roundRect(x, y, box_width, box_height, 12, fill=True, stroke=True)

    # Text content
    p.setFont("Helvetica-Bold", 12)
    p.setFillColor(text_color)
    padding = 20
    label_x = x + padding
    value_x = x + 160
    line_height = 30
    start_y = y + box_height - 40

   # Username
    p.drawString(label_x, start_y, "Username:")
    p.setFont("Helvetica", 12)
    p.drawString(value_x, start_y, username)

    # Filename (shifted downward)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(label_x, start_y - line_height, "Filename:")
    p.setFont("Helvetica", 12)
    p.drawString(value_x, start_y - line_height, report.filename)

     # Prediction (multi-line)
    p.setFont("Helvetica-Bold", 12)
    p.drawString(label_x, start_y - line_height*2, "Prediction:")
    prediction_text = report.prediction
    max_text_width = box_width - (2 * padding + 110)
    prediction_lines = simpleSplit(prediction_text, "Helvetica", 12, max_text_width)

    p.setFont("Helvetica", 12)
    for i, line in enumerate(prediction_lines):
        p.drawString(value_x, start_y - line_height*2 - (i * 15), line)

    # Adjust position for next items
    extra_offset = line_height*2 + (len(prediction_lines) * 15)
    current_y = start_y - extra_offset

    # Confidence
    p.setFont("Helvetica-Bold", 12)
    p.drawString(label_x, current_y, "Confidence:")
    p.setFont("Helvetica", 12)
    p.drawString(value_x, current_y, confidence_text)

    # Treatment Action
    treatments = {
        "Aphids": "Use insecticidal soap or neem oil. Encourage natural predators like ladybugs.",
        "Army Worm": "Apply Bacillus thuringiensis (Bt) or use organic spinosad sprays.",
        "Bacterial Blight": "Use copper-based bactericides and ensure good air circulation. Remove infected leaves.",
        "Healthy": "No treatment needed. Maintain optimal watering and fertilization.",
        "Powdery Mildew": "Apply sulfur or potassium bicarbonate sprays. Ensure plants are not overcrowded.",
        "Target Spot": "Apply appropriate fungicides and avoid overhead watering to keep leaves dry."
    }
    treatment_text = "Consult an agricultural expert."
    for d_name, remedy in treatments.items():
        if d_name.lower() in report.prediction.lower():
            treatment_text = remedy
            break
            
    p.setFont("Helvetica-Bold", 12)
    p.drawString(label_x, current_y - line_height, "Action:")
    
    treatment_lines = simpleSplit(treatment_text, "Helvetica", 12, max_text_width)
    p.setFont("Helvetica", 12)
    for i, line in enumerate(treatment_lines):
        p.drawString(value_x, current_y - line_height - (i * 15), line)

    extra_offset_treatment = len(treatment_lines) * 15
    new_y = current_y - line_height - extra_offset_treatment - 10

    # Date
    p.setFont("Helvetica-Bold", 12)
    p.drawString(label_x, new_y, "Date:")
    p.setFont("Helvetica", 12)
    p.drawString(value_x, new_y, report.timestamp.strftime('%Y-%m-%d %H:%M:%S'))

    # Footer
    p.setStrokeColor(colors.HexColor("#cccccc"))
    p.line(40, 60, width - 40, 60)
    p.setFont("Helvetica-Oblique", 10)
    p.setFillColor(colors.HexColor("#777777"))
    p.drawString(40, 45, "Generated by Cotton Disease Detection System")

    p.showPage()
    p.save()

    buffer.seek(0)
    return send_file(buffer, as_attachment=True,
                     download_name=f"{report.filename}_report.pdf",
                     mimetype='application/pdf')

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ------------------- Routes -------------------
@csrf.exempt
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Removed upload limits for personal use
        pass

    try:
        f = request.files['file']
        if not f or f.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400

        # Save file and predict
        file_path = os.path.join(os.path.dirname(__file__), 'uploads', f.filename)
        f.save(file_path)
        # GEMINI VALIDATION
        if not is_plant_image(file_path):
            return jsonify({
                'status': 'error',
                'message': 'Invalid image. Please upload plant leaf image.'
            }), 400
        preds = model_predict(file_path)

        # Extract prediction and confidence properly
        prediction = preds.split("(")[0].strip() if "(" in preds else preds
        confidence = None
        if "Confidence:" in preds:
            try:
                confidence = float(preds.split("Confidence:")[1].replace("%", "").replace(")", "").strip())
            except:
                pass

        # ✅ Gemini Fallback: If confidence is low, use Gemini to re-evaluate
        if confidence is not None and confidence < 60.0:
            log_debug(f"Confidence low ({confidence}%). Using Gemini Fallback...")
            gemini_preds = analyze_image_with_gemini(file_path)
            
            # Only update if Gemini was successful
            if "Gemini Analysis Error" not in gemini_preds:
                preds = gemini_preds
                prediction = preds.split("(")[0].replace("The leaf shows signs of", "").replace("The leaf is", "").strip() if "(" in preds else preds
                if "Confidence:" in preds:
                    try:
                        confidence = float(preds.split("Confidence:")[1].replace("%", "").replace(")", "").strip())
                    except:
                        pass
                log_debug(f"Gemini successfully re-evaluated: {prediction} ({confidence}%)")
            else:
                log_debug("Gemini fallback failed. Reverting to local model prediction.")

        # Prepare treatment
        treatments = {
            "Aphids": "Use insecticidal soap or neem oil. Encourage natural predators like ladybugs.",
            "Army Worm": "Apply Bacillus thuringiensis (Bt) or use organic spinosad sprays.",
            "Bacterial Blight": "Use copper-based bactericides and ensure good air circulation. Remove infected leaves.",
            "Healthy": "No treatment needed. Maintain optimal watering and fertilization.",
            "Powdery Mildew": "Apply sulfur or potassium bicarbonate sprays. Ensure plants are not overcrowded.",
            "Target Spot": "Apply appropriate fungicides and avoid overhead watering to keep leaves dry."
        }
        treatment_text = "Consult an agricultural expert for detailed diagnosis."
        for d_name, remedy in treatments.items():
            if d_name.lower() in prediction.lower():
                treatment_text = remedy
                break

        # Prepare response
        response_data = {
            'status': 'success',
            'prediction': prediction,
            'full_result': preds,
            'confidence': confidence,
            'treatment': treatment_text
        }

        if current_user.is_authenticated:
            # Save to database
            report = Report(
                user_id=current_user.id,
                filename=f.filename,
                prediction=prediction,
                confidence=confidence
            )
            db.session.add(report)
            current_user.upload_attempts += 1
            db.session.commit()
            response_data['report_id'] = report.id
            log_debug(f"Sending authenticated response: {response_data}")
            return jsonify(response_data)

        else:
            log_debug(f"Sending guest response: {response_data}")
            response = jsonify(response_data)
            response.set_cookie('guest_upload_done', 'true', max_age=60*60*24*30)  # 30 days
            return response

    except Exception as e:
        log_debug(f"Error in /predict route: {str(e)}")
        import traceback
        log_debug(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@csrf.exempt
@app.route('/get_latest_report_id')
@login_required
def get_latest_report_id():
    latest_report = Report.query.filter_by(user_id=current_user.id).order_by(Report.id.desc()).first()
    if latest_report:
        return jsonify({'report_id': latest_report.id})
    return jsonify({'report_id': None})


@app.route('/')
def home():
    total_users = User.query.count()
    total_reports = Report.query.count()
    return render_template('index.html', total_users=total_users, total_reports=total_reports)

@csrf.exempt
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password or not confirm_password:
            flash('All fields are required.', 'warning')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match.', 'warning')
            return redirect(url_for('register'))

        if len(password) < 6:
            flash('Password must be at least 6 characters long.', 'warning')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose another.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please log in or use another.', 'danger')
            return redirect(url_for('register'))

        # Create user and hash password
        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

       # Save to SignupHistory
        signup_log = SignupHistory(username=username)
        db.session.add(signup_log)
        db.session.commit()

        # Send Welcome Email
        subject = "Welcome to Cotton Disease Detection!"
        body = f"Hello {username},\n\nThank you for registering on our platform. We are glad to have you with us!\n\nRegards,\nCotton Disease Detection Team"
        send_notification_email(email, subject, body)

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@csrf.exempt
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash('You are already logged in.', 'info')
        return redirect(url_for('home'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        login_success = user and check_password_hash(user.password, password)

        if login_success:
            # Send Login Alert
            subject = "Login Alert - Cotton Disease Detection"
            body = f"Hello {user.username},\n\nYour account was just logged into. If this wasn't you, please reset your password immediately.\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\nRegards,\nCotton Disease Detection Team"
            send_notification_email(user.email, subject, body)


 # ✅ Clear guest upload flag after login
        session.pop('guest_uploaded', None)
        
        login_log = LoginHistory(
            user_id=user.id if user else None,
            username=username,
            success=login_success
        )
        db.session.add(login_log)

        if not login_success:
            db.session.commit()
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))

        login_user(user)
        session['is_admin'] = user.is_admin
        db.session.add(SessionHistory(user_id=user.id, username=user.username, action='login'))
        db.session.commit()

        flash(f'Welcome, {user.username}!', 'success')
        return redirect(url_for('home'))

    return render_template('login.html')

@csrf.exempt
@app.route('/verify-user', methods=['POST'])
def verify_user():
    data = request.get_json()
    email = data.get('email')
    username = data.get('username')

    # Query user from DB
    user = User.query.filter_by(username=username).first()
    if user and user.email.lower() == email.lower():
        return jsonify({'exists': True})
    return jsonify({'exists': False})

@csrf.exempt
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    email = request.form.get('forgot_email', '').strip()
    username = request.form.get('forgot_username', '').strip()

    user = User.query.filter_by(username=username).first()
    if not user or user.email.lower() != email.lower():
        flash("User does not exist or incorrect credentials.", "danger")
        return redirect(url_for('login'))

    # Generate token
    token = s.dumps(user.email, salt='password-reset-salt')
    reset_url = url_for('reset_token', token=token, _external=True)

    # Send reset email
    subject = "Password Reset Request - Cotton Disease Detection"
    body = f"Hello {user.username},\n\nTo reset your password, visit the following link:\n{reset_url}\n\nIf you did not make this request, simply ignore this email.\n\nRegards,\nCotton Disease Detection Team"
    
    if send_notification_email(user.email, subject, body):
        flash("A password reset link has been sent to your email.", "info")
    else:
        flash("Failed to send reset email. Please try again later.", "danger")

    return redirect(url_for('login'))

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    try:
        email = s.loads(token, salt='password-reset-salt', max_age=3600)
    except:
        flash("The reset link is invalid or has expired.", "warning")
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_new_password')

        if new_password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('reset_token', token=token))

        if len(new_password) < 6:
            flash("Password must be at least 6 characters long.", "danger")
            return redirect(url_for('reset_token', token=token))

        user = User.query.filter_by(email=email).first()
        if user:
            user.set_password(new_password)
            db.session.commit()
            flash("Your password has been updated! You can now log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("User not found.", "danger")
            return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)


@app.route('/logout')
@login_required
def logout():
    try:
        session_record = SessionHistory(
            user_id=current_user.id,
            username=current_user.username,
            action='logout',
            timestamp=datetime.utcnow()
        )
        db.session.add(session_record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Failed to log logout: {e}")
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

@app.route('/profile')
@login_required
def profile():
    user = User.query.get(current_user.id)
    return render_template("profile.html",  user=user)

@app.route('/try')
@login_required
def try_page():
    return render_template('try.html')


@app.route('/my-reports')
@app.route('/my-reports/page/<int:page>')
@login_required
def my_reports(page=1):
    per_page = 10
    username_filter = request.args.get('username', '').strip()
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if current_user.is_admin:
        # Admin: fetch all reports, optionally filter by username
        query = Report.query.options(joinedload(Report.user)).order_by(Report.timestamp.desc())

        if username_filter:
            query = query.join(User).filter(User.username.ilike(f"%{username_filter}%"))
        # Apply date filter if valid
        if start_date_str and end_date_str:
            try:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                query = query.filter(Report.timestamp.between(start_date, end_date))
            except ValueError:
                flash("Invalid date format", "warning")
    else:
        # Regular user: only their reports
        query = Report.query.filter_by(user_id=current_user.id).order_by(Report.timestamp.desc())
    total_reports = query.count()
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    reports = pagination.items
     # ✅ Disease frequency analysis (admin only)
    disease_counts = {}
    if current_user.is_admin:
        all_filtered_reports = query.all()
        for report in all_filtered_reports:
            disease = report.prediction.split("(")[0].strip()
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    return render_template('my-reports.html', reports=reports, pagination=pagination, username_filter=username_filter,start_date=start_date_str,
        end_date=end_date_str,
        disease_counts=disease_counts if current_user.is_admin else None,total_reports=total_reports)


@app.route('/diseases-info')
def diseases_info():
    return render_template('5 diseases info.html')

@app.route('/aphids')
def aphids():
    return render_template('aphids.html')

@app.route('/army-worm')
def army_worm():
    return render_template('army-worm.html')

@app.route('/bacterial-blight')
def bacterial_blight():
    return render_template('bacterial-blight.html')

@app.route('/powdery-mildew')
def powdery_mildew():
    return render_template('powdery-mildew.html')

@app.route('/target-spot')
def target_spot():
    return render_template('target-spot.html')

@csrf.exempt
@app.route('/forum', methods=['GET', 'POST'])
@login_required
def forum():
    if request.method == 'POST':
        content = request.form.get('content', '').strip()
        if content:
            msg = ForumMessage(user_id=current_user.id, username=current_user.username, content=content)
            db.session.add(msg)
            db.session.commit()
            flash("Message posted to community!", "success")
            return redirect(url_for('forum'))
        else:
            flash("Message cannot be empty.", "warning")
            
    messages = ForumMessage.query.order_by(ForumMessage.timestamp.desc()).all()
    return render_template('forum.html', messages=messages)



@csrf.exempt
@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('home'))
    # filering singup
    signup_username = request.args.get('signup_username', '').strip()
    signup_start = request.args.get('signup_start')
    signup_end = request.args.get('signup_end')

    signup_query = SignupHistory.query
    # Filter by username
    if signup_username:
         signup_query = signup_query.filter(SignupHistory.username.ilike(f"%{signup_username}%"))
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')
    # Filter by date range
    if signup_start and signup_end:
        try:
             start = datetime.strptime(signup_start, '%Y-%m-%d')
             end = datetime.strptime(signup_end, '%Y-%m-%d')
             signup_query = signup_query.filter(
             SignupHistory.timestamp >= start,
             SignupHistory.timestamp < end + timedelta(days=1))
        except ValueError:
         flash("Invalid signup date format", "warning")
    # --- Login Filter ---
    login_username = request.args.get('login_username', '').strip()
    login_start = request.args.get('login_start')
    login_end = request.args.get('login_end')  
    login_query = LoginHistory.query

    if login_username:
         login_query = login_query.filter(LoginHistory.username.ilike(f"%{login_username}%")) 
    if login_start and login_end:
        try:
            start = datetime.strptime(login_start, '%Y-%m-%d')
            end = datetime.strptime(login_end, '%Y-%m-%d') + timedelta(days=1)
            login_query = login_query.filter(LoginHistory.timestamp >= start, LoginHistory.timestamp < end)
        except ValueError:
             flash("Invalid login date format", "warning")
    # Revenue calculation removed for personal use
    payments = []
    total_revenue = 0
    revenue_by_plan = {'Premium': 0, 'Diamond': 0}
    # --- Session Logs Filter ---
    session_username = request.args.get('session_username', '').strip()
    session_start = request.args.get('session_start')
    session_end = request.args.get('session_end')
    session_query = SessionHistory.query

    if session_username:
        session_query = session_query.filter(SessionHistory.username.ilike(f"%{session_username}%"))
    
    if session_start and session_end:
        try:
         start = datetime.strptime(session_start, '%Y-%m-%d')
         end = datetime.strptime(session_end, '%Y-%m-%d') + timedelta(days=1)
         session_query = session_query.filter(SessionHistory.timestamp >= start, SessionHistory.timestamp < end)
        except ValueError:
         flash("Invalid session date format", "warning")
    
    user_search = request.args.get('user_search', '').strip()
    user_query = User.query
    if user_search:
        user_query = user_query.filter(User.username.ilike(f"%{user_search}%"))
    signup_logs = signup_query.order_by(SignupHistory.timestamp.desc()).all()
    login_logs = login_query.order_by(LoginHistory.timestamp.desc()).all()
    session_logs = session_query.order_by(SessionHistory.timestamp.desc()).all()
    users = user_query.all()  # Replace old 'users = User.query.all()'

    # Count successful logins by username for Chart.js
    login_usernames = [log.username for log in login_logs if log.success]
    login_chart_data = dict(Counter(login_usernames))
    plan_counts = {'Total Users': len(users)}

    return render_template(
        'admin_dashboard.html',
        signup_logs=signup_logs,
        login_logs=login_logs,
        session_logs=session_logs,
        login_chart_data=login_chart_data,
        payments=payments,
        plan_counts=plan_counts,
        total_revenue=total_revenue,
        revenue_by_plan=revenue_by_plan,
        start_date=start_date_str,
        end_date=end_date_str,
        users=users
    )
@app.before_request
def restrict_to_admin():
    if request.path.startswith('/admin') and not session.get('is_admin'):
        return redirect(url_for('login'))


@csrf.exempt
@app.route('/admin/view-users')
@login_required
def view_users():
    if not current_user.is_admin:
        return redirect(url_for('home'))
    users = User.query.all()
    return render_template('view_users.html', users=users)

# Route to Add a New User
@csrf.exempt
@app.route('/admin/add_user', methods=['POST'])
def add_user():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    role = request.form['role']
    
    if User.query.filter_by(username=username).first():
        flash('Username already exists', 'error')
        return redirect(url_for('admin_dashboard'))
    # Optional: check if user already exists
    if User.query.filter_by(email=email).first():
        flash('Email already exists', 'error')
        return redirect(url_for('admin_dashboard'))

    if len(password) < 6:
        flash('Password must be at least 6 characters long.', 'error')
        return redirect(url_for('admin_dashboard'))

    hashed_password = generate_password_hash(password)
     # Determine is_admin based on role
    is_admin = 1 if role == 'admin' else 0  # Use 1/0 for MySQL compatibility

    new_user = User(username=username, email=email, password=hashed_password, role=role,  is_admin=is_admin)
    db.session.add(new_user)
    db.session.commit()

    flash('User added successfully', 'success')
    return redirect(url_for('admin_dashboard'))

@csrf.exempt
@app.route('/manage_users')
def manage_users():
    search_query = request.args.get('search', '').strip().lower()
    if search_query:
        users = User.query.filter(
            (User.username.ilike(f'%{search_query}%')) |
            (User.email.ilike(f'%{search_query}%'))
        ).all()
    else:
        users = User.query.all()
    
    # Include required data for dashboard context
    payments = []
    total_revenue = 0
    revenue_by_plan = {'Premium': 0, 'Diamond': 0}
    plan_counts = {'Total Users': len(users)}

    return render_template('admin_dashboard.html', users=users,
        payments=payments,
        plan_counts=plan_counts,
        total_revenue=total_revenue,
        revenue_by_plan=revenue_by_plan,
        login_logs=LoginHistory.query.all(),
        session_logs=SessionHistory.query.all(),
        signup_logs=SignupHistory.query.all(),
        start_date=None,
        end_date=None)


@csrf.exempt
@app.route('/admin/edit_user/<int:user_id>', methods=['POST']) 
def edit_user(user_id):
    user = User.query.get_or_404(user_id)

    new_username = request.form.get('username')
    new_email = request.form.get('email')
    new_password = request.form.get('password')
    new_role = request.form.get('role')

    print(f"Edit Request - ID: {user_id}, New Username: {new_username}, New Email: {new_email}")

    # Check for conflicts
    existing_username = User.query.filter(User.username == new_username, User.id != user_id).first()
    existing_email = User.query.filter(User.email == new_email, User.id != user_id).first()

    if existing_username:
        print("❌ Username already exists.")
        flash("Username already exists.", "danger")
        return redirect(url_for('admin_dashboard'))

    if existing_email:
        print("❌ Email already exists.")
        flash("Email already exists.", "danger")
        return redirect(url_for('admin_dashboard'))

    # Track whether any changes were made
    changes = False

    if new_username != user.username:
        user.username = new_username
        changes = True

    if new_email != user.email:
        user.email = new_email
        changes = True

    if new_password:
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('admin_dashboard'))
        user.password = generate_password_hash(new_password)
        changes = True

    if new_role != user.role:
        user.role = new_role
        user.is_admin = 1 if new_role == 'admin' else 0  # Use 1/0 for MySQL compatibility
        changes = True

    if changes:
        try:
            db.session.commit()
            print("✅ User updated successfully.")
            flash("User updated successfully.", "success")
        except Exception as e:
            db.session.rollback()
            print("❌ Commit failed:", e)
            flash("Something went wrong during update.", "danger")
    else:
        print("ℹ️ No changes were made.")
        flash("No changes were made.", "info")

    return redirect(url_for('admin_dashboard'))



# Route to Delete a User
@csrf.exempt
@app.route('/admin/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()

    flash('User deleted successfully', 'success')
    return redirect(url_for('admin_dashboard'))

@app.route('/export_summary_csv')
@login_required
def export_summary_csv():
    if not current_user.is_admin:
        return "Access denied", 403

    users = User.query.all()

    # Prepare CSV
    output = []
    output.append(['User Summary'])
    output.append(['Total Users', len(users)])

    si = '\n'.join([','.join(map(str, row)) for row in output])
    return Response(
        si,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=summary.csv"}
    )

@app.route('/export_disease_frequency_csv')
@login_required
def export_disease_frequency_csv():
    if not current_user.is_admin:
        flash("Access denied", "danger")
        return redirect(url_for('home'))

    # Apply the same filters from /my-reports
    username_filter = request.args.get('username', '').strip()
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    query = Report.query.join(User)

    if username_filter:
        query = query.filter(User.username.ilike(f"%{username_filter}%"))

    if start_date_str and end_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            query = query.filter(Report.timestamp.between(start_date, end_date))
        except ValueError:
            flash("Invalid date format", "warning")

    # Count diseases
    disease_counts = {}
    for report in query.all():
        disease = report.prediction.split("(")[0].strip()
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    # Generate CSV
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Disease', 'Count'])
    for disease, count in disease_counts.items():
        writer.writerow([disease, count])

    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={
            "Content-Disposition": "attachment; filename=disease_frequency.csv"
        }
    )

@app.route('/export_signup_csv')
@login_required
def export_signup_csv():
    if not current_user.is_admin:
        flash("Access denied", "danger")
        return redirect(url_for('home'))

    username = request.args.get('signup_username', '').strip()
    start = request.args.get('signup_start', '')
    end = request.args.get('signup_end', '')

    query = SignupHistory.query

    if username:
        query = query.filter(SignupHistory.username.ilike(f"%{username}%"))

    if start and end:
     try:
        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)  # add 1 day
        query = query.filter(
            SignupHistory.timestamp >= start_date,
            SignupHistory.timestamp < end_date
        )
     except ValueError:
        flash("Invalid date format", "warning")

    logs = query.order_by(SignupHistory.timestamp.desc()).all()

    # ✅ Write CSV to memory using StringIO
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Username", "Timestamp"])
    for log in logs:
        writer.writerow([log.id, log.username, log.timestamp.strftime('%Y-%m-%d %H:%M:%S')])

    # ✅ Prepare response
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=signup_logs.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/export_login_csv')
@login_required
def export_login_csv():
    if not current_user.is_admin:
        flash("Access denied", "danger")
        return redirect(url_for('home'))

    username = request.args.get('login_username', '').strip()
    start = request.args.get('login_start', '')
    end = request.args.get('login_end', '')

    query = LoginHistory.query

    if username:
        query = query.filter(LoginHistory.username.ilike(f"%{username}%"))

    if start and end:
        try:
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(LoginHistory.timestamp >= start_date, LoginHistory.timestamp < end_date)
        except ValueError:
            flash("Invalid date format", "warning")

    logs = query.order_by(LoginHistory.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "User ID", "Username", "Success", "Timestamp"])
    for log in logs:
        writer.writerow([log.id, log.user_id, log.username, "Yes" if log.success else "No", log.timestamp.strftime('%Y-%m-%d %H:%M:%S')])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=login_logs.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/export_session_csv')
@login_required
def export_session_csv():
    if not current_user.is_admin:
        flash("Access denied", "danger")
        return redirect(url_for('home'))

    username = request.args.get('session_username', '').strip()
    start = request.args.get('session_start', '')
    end = request.args.get('session_end', '')

    query = SessionHistory.query

    if username:
        query = query.filter(SessionHistory.username.ilike(f"%{username}%"))

    if start and end:
        try:
            start_date = datetime.strptime(start, "%Y-%m-%d")
            end_date = datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)
            query = query.filter(SessionHistory.timestamp >= start_date, SessionHistory.timestamp < end_date)
        except ValueError:
            flash("Invalid date format", "warning")

    logs = query.order_by(SessionHistory.timestamp.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "User ID", "Username", "Action", "Timestamp"])
    for log in logs:
        writer.writerow([log.id, log.user_id, log.username, log.action, log.timestamp.strftime('%Y-%m-%d %H:%M:%S')])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=session_logs.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/export_users_csv')
@login_required
def export_users_csv():
    if not current_user.is_admin:
        flash("Access denied", "danger")
        return redirect(url_for('home'))

    user_search = request.args.get('user_search', '').strip()
    query = User.query
    if user_search:
        query = query.filter(User.username.ilike(f"%{user_search}%"))

    users = query.order_by(User.id).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Username", "Email", "Role", "Upload Attempts"])

    for user in users:
        writer.writerow([
            user.id,
            user.username,
            user.email,
            user.role,
            user.upload_attempts
        ])

    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=filtered_users.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@csrf.exempt
@app.route('/api/login_chart_data')
@login_required
def get_login_chart_data():
    if current_user.username != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403

    login_logs = LoginHistory.query.filter_by(success=True).all()
    login_usernames = [log.username for log in login_logs]
    login_chart_data = dict(Counter(login_usernames))

    return jsonify(login_chart_data)

# ------------------- AI Assistant Routes -------------------

@app.route('/ai-assistant')
def ai_assistant():
    return render_template('ai_assistant.html')

@app.route('/chat', methods=['POST'])
@csrf.exempt
def chat():
    if not request.is_json:
        return jsonify({'status': 'error', 'message': 'Request must be JSON'}), 400
        
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'status': 'error', 'message': 'No message provided'}), 400

    try:
        # Professional agricultural context
        prompt = f"""
        You are 'CottonBot', a professional AI Agricultural Assistant for the Cotton Disease Detection project.
        Your goal is to help farmers with cotton crop management, disease prevention, and treatments.
        
        Guidelines:
        - Respond ONLY in English. Do not use Urdu unless the user specifically asks you to.
        - Even if the first message is in Urdu, respond in English and ask if they would like to switch to Urdu.
        - Be professional, helpful, and provide accurate agricultural advice.
        - Knowledge cutoff: 2024.
        
        User: {user_message}
        """
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        
        log_debug(f"Sending chat request to Gemini: {user_message[:50]}...")
        response = requests.post(GEMINI_URL, json=payload)
        
        if response.status_code != 200:
            log_debug(f"Gemini Chat API Error: {response.status_code} - {response.text}")
            return jsonify({'status': 'error', 'message': f'Gemini API error ({response.status_code})'}), response.status_code
            
        result = response.json()
        bot_message = result['candidates'][0]['content']['parts'][0]['text'].strip()
        
        return jsonify({'status': 'success', 'message': bot_message})
    except Exception as e:
        log_debug(f"Chat API Error: {str(e)}")
        return jsonify({'status': 'error', 'message': 'I am having trouble connecting to my brain right now. Please try again later.'}), 500

# ------------------- Run App -------------------

if __name__ == '__main__':
    with app.app_context():
     db.create_all()
    app.run(port=5000,debug=True)



