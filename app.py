"""
app.py — CropGuard AI Backend
Run: python app.py
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import json
import os
import sqlite3
import base64
from datetime import datetime
from PIL import Image
import io

# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# ---- Load model and classes ----
BASE_DIR = os.path.dirname(__file__)
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'crop_disease_model.h5'))

with open(os.path.join(BASE_DIR, 'classes.json')) as f:
    CLASSES = json.load(f)

IMG_SIZE = 224

# ---- Disease treatment database ----
TREATMENTS = {
    'Apple___Apple_scab': {
        'display': 'Apple Scab',
        'crop': 'Apple', 'severity': 'Medium',
        'description': 'Fungal disease causing dark, scaly lesions on leaves and fruit.',
        'treatment': ['Apply fungicide (Captan or Mancozeb) at bud break', 'Remove and destroy infected leaves', 'Ensure good air circulation by pruning', 'Avoid overhead irrigation'],
        'prevention': 'Plant resistant varieties. Apply protective sprays before rain.',
        'organic': 'Neem oil spray every 7-14 days during wet weather.'
    },
    'Apple___Black_rot': {
        'display': 'Apple Black Rot',
        'crop': 'Apple', 'severity': 'High',
        'description': 'Fungal disease causing black, rotting spots on fruit and cankers on branches.',
        'treatment': ['Prune out dead and diseased wood', 'Apply copper-based fungicide', 'Remove mummified fruit from trees', 'Spray Captan during growing season'],
        'prevention': 'Maintain tree vigor with proper fertilization. Remove all debris.',
        'organic': 'Copper hydroxide spray. Remove all infected material immediately.'
    },
    'Apple___Cedar_apple_rust': {
        'display': 'Cedar Apple Rust',
        'crop': 'Apple', 'severity': 'Medium',
        'description': 'Fungal disease requiring both apple and cedar/juniper trees to complete lifecycle.',
        'treatment': ['Apply myclobutanil or triadimefon fungicide', 'Remove nearby juniper/cedar trees if possible', 'Spray during bloom period', 'Repeat every 7-10 days in wet weather'],
        'prevention': 'Plant resistant varieties. Remove galls from cedars in late winter.',
        'organic': 'Sulfur-based fungicide from pink stage through petal fall.'
    },
    'Apple___healthy': {
        'display': 'Healthy Apple',
        'crop': 'Apple', 'severity': 'None',
        'description': 'Your apple plant appears healthy with no visible signs of disease.',
        'treatment': ['Continue regular watering schedule', 'Maintain balanced fertilization', 'Monitor regularly for early signs of disease'],
        'prevention': 'Regular inspection, proper spacing, and good cultural practices.',
        'organic': 'Maintain soil health with compost. Encourage beneficial insects.'
    },
    'Blueberry___healthy': {
        'display': 'Healthy Blueberry',
        'crop': 'Blueberry', 'severity': 'None',
        'description': 'Your blueberry plant appears healthy.',
        'treatment': ['Maintain soil pH between 4.5-5.5', 'Apply acidic fertilizer', 'Ensure consistent moisture'],
        'prevention': 'Mulch with pine needles. Avoid alkaline water sources.',
        'organic': 'Compost tea application every 2 weeks.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'display': 'Cherry Powdery Mildew',
        'crop': 'Cherry', 'severity': 'Medium',
        'description': 'Fungal disease causing white powdery coating on leaves and shoots.',
        'treatment': ['Apply sulfur or potassium bicarbonate spray', 'Prune to improve air circulation', 'Avoid excessive nitrogen fertilization', 'Apply neem oil weekly'],
        'prevention': 'Plant in sunny locations with good air flow. Avoid wetting foliage.',
        'organic': 'Baking soda solution (1 tbsp per gallon water) spray weekly.'
    },
    'Cherry_(including_sour)___healthy': {
        'display': 'Healthy Cherry',
        'crop': 'Cherry', 'severity': 'None',
        'description': 'Your cherry plant appears healthy.',
        'treatment': ['Continue regular care routine', 'Prune after harvest', 'Monitor for pests'],
        'prevention': 'Regular inspection and balanced nutrition.',
        'organic': 'Apply compost in spring for sustained nutrition.'
    },
    'Corn_(maize)___Cercospora_leaf_spot': {
        'display': 'Corn Gray Leaf Spot',
        'crop': 'Corn (Maize)', 'severity': 'High',
        'description': 'Fungal disease causing rectangular gray-brown lesions on corn leaves.',
        'treatment': ['Apply strobilurin or triazole fungicide at VT stage', 'Rotate crops — avoid corn-on-corn', 'Till infected crop residue after harvest', 'Use resistant hybrids next season'],
        'prevention': 'Crop rotation is the most effective prevention. Plant resistant varieties.',
        'organic': 'Copper-based fungicide. Ensure proper plant spacing for air flow.'
    },
    'Corn_(maize)___Common_rust': {
        'display': 'Corn Common Rust',
        'crop': 'Corn (Maize)', 'severity': 'Medium',
        'description': 'Fungal disease causing small, circular to elongated brown pustules on leaves.',
        'treatment': ['Apply fungicide (Mancozeb) if infection is severe', 'Plant rust-resistant hybrids', 'Scout fields regularly in humid conditions'],
        'prevention': 'Use resistant hybrids. Early planting can help avoid peak rust periods.',
        'organic': 'Neem oil or copper spray at first sign of infection.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'display': 'Northern Leaf Blight',
        'crop': 'Corn (Maize)', 'severity': 'High',
        'description': 'Fungal disease causing long, cigar-shaped gray-green lesions on leaves.',
        'treatment': ['Apply propiconazole or azoxystrobin fungicide', 'Rotate with non-host crops', 'Bury or remove infected crop debris', 'Plant resistant varieties'],
        'prevention': 'Crop rotation and resistant hybrids are most effective.',
        'organic': 'Copper hydroxide spray. Remove infected debris after harvest.'
    },
    'Corn_(maize)___healthy': {
        'display': 'Healthy Corn',
        'crop': 'Corn (Maize)', 'severity': 'None',
        'description': 'Your corn plant appears healthy.',
        'treatment': ['Maintain regular irrigation', 'Side-dress with nitrogen at V6 stage', 'Monitor for pest pressure'],
        'prevention': 'Scout weekly. Maintain balanced soil nutrition.',
        'organic': 'Apply compost or fish emulsion for sustained nutrition.'
    },
    'Grape___Black_rot': {
        'display': 'Grape Black Rot',
        'crop': 'Grape', 'severity': 'High',
        'description': 'Fungal disease causing brown leaf lesions and black, shriveled fruit.',
        'treatment': ['Apply myclobutanil or mancozeb fungicide', 'Remove and destroy mummified berries', 'Prune to improve air circulation', 'Begin sprays at bud swell'],
        'prevention': 'Remove all mummies and debris in winter. Prune for open canopy.',
        'organic': 'Copper sulfate spray from bud break. Sulfur sprays during season.'
    },
    'Grape___Esca_(Black_Measles)': {
        'display': 'Grape Esca (Black Measles)',
        'crop': 'Grape', 'severity': 'High',
        'description': 'Fungal complex causing tiger-stripe leaf patterns and internal wood decay.',
        'treatment': ['No chemical cure available — management only', 'Remove severely infected vines', 'Protect pruning wounds with fungicide paste', 'Avoid large pruning cuts'],
        'prevention': 'Protect all pruning wounds immediately. Use clean pruning tools.',
        'organic': 'Trichoderma-based biological control on pruning wounds.'
    },
    'Grape___Leaf_blight': {
        'display': 'Grape Leaf Blight',
        'crop': 'Grape', 'severity': 'Medium',
        'description': 'Fungal disease causing angular brown lesions on grape leaves.',
        'treatment': ['Apply copper-based fungicide', 'Remove infected leaves promptly', 'Improve canopy air circulation', 'Avoid overhead irrigation'],
        'prevention': 'Good canopy management. Avoid wetting leaves.',
        'organic': 'Copper hydroxide spray every 10-14 days.'
    },
    'Grape___healthy': {
        'display': 'Healthy Grape',
        'crop': 'Grape', 'severity': 'None',
        'description': 'Your grape plant appears healthy.',
        'treatment': ['Maintain proper trellising', 'Prune annually for open canopy', 'Monitor for fungal diseases in wet weather'],
        'prevention': 'Good canopy management and regular scouting.',
        'organic': 'Compost mulch around base. Avoid over-watering.'
    },
    'Orange___Haunglongbing': {
        'display': 'Citrus Greening (HLB)',
        'crop': 'Orange (Citrus)', 'severity': 'Critical',
        'description': 'Bacterial disease spread by Asian citrus psyllid. No cure exists — most destructive citrus disease worldwide.',
        'treatment': ['Remove and destroy infected trees immediately', 'Control Asian citrus psyllid with insecticide', 'Do NOT move plant material from infected areas', 'Report to local agriculture authority'],
        'prevention': 'Strict quarantine. Use certified disease-free planting material only.',
        'organic': 'No organic cure. Early removal is the only management option.'
    },
    'Peach___Bacterial_spot': {
        'display': 'Peach Bacterial Spot',
        'crop': 'Peach', 'severity': 'High',
        'description': 'Bacterial disease causing water-soaked spots on leaves, fruit, and twigs.',
        'treatment': ['Apply copper bactericide during dormancy', 'Spray oxytetracycline during bloom', 'Prune infected twigs', 'Avoid injury to trees'],
        'prevention': 'Plant resistant varieties. Avoid working in orchards when wet.',
        'organic': 'Copper spray at petal fall and after harvest.'
    },
    'Peach___healthy': {
        'display': 'Healthy Peach',
        'crop': 'Peach', 'severity': 'None',
        'description': 'Your peach plant appears healthy.',
        'treatment': ['Thin fruit for better size', 'Apply balanced fertilizer in spring', 'Monitor for brown rot near harvest'],
        'prevention': 'Regular inspection. Good sanitation of fallen fruit.',
        'organic': 'Neem oil spray preventatively during humid periods.'
    },
    'Pepper___Bacterial_spot': {
        'display': 'Pepper Bacterial Spot',
        'crop': 'Bell Pepper', 'severity': 'High',
        'description': 'Bacterial disease causing water-soaked lesions on leaves and fruit.',
        'treatment': ['Apply copper bactericide spray', 'Remove infected plant debris', 'Avoid overhead irrigation', 'Use disease-free seed'],
        'prevention': 'Use certified seed. Crop rotation every 2-3 years.',
        'organic': 'Copper hydroxide spray. Remove infected leaves immediately.'
    },
    'Pepper___healthy': {
        'display': 'Healthy Bell Pepper',
        'crop': 'Bell Pepper', 'severity': 'None',
        'description': 'Your pepper plant appears healthy.',
        'treatment': ['Maintain consistent watering', 'Support plants with stakes', 'Fertilize with potassium at fruit set'],
        'prevention': 'Regular inspection. Mulch to retain moisture.',
        'organic': 'Compost tea spray for foliar nutrition.'
    },
    'Potato___Early_blight': {
        'display': 'Potato Early Blight',
        'crop': 'Potato', 'severity': 'Medium',
        'description': 'Fungal disease causing dark brown concentric ring lesions on older leaves.',
        'treatment': ['Apply chlorothalonil or mancozeb fungicide', 'Remove infected lower leaves', 'Ensure adequate potassium nutrition', 'Maintain proper plant spacing'],
        'prevention': 'Rotate crops every 3 years. Use certified seed potatoes.',
        'organic': 'Copper spray every 7-10 days. Remove infected leaves.'
    },
    'Potato___Late_blight': {
        'display': 'Potato Late Blight',
        'crop': 'Potato', 'severity': 'Critical',
        'description': 'Caused by Phytophthora infestans — the pathogen responsible for the Irish Famine. Can destroy a crop rapidly.',
        'treatment': ['Apply metalaxyl or cymoxanil fungicide immediately', 'Destroy all infected plant material — do NOT compost', 'Hill up soil around plants', 'Harvest tubers early if infection is severe'],
        'prevention': 'Plant resistant varieties. Avoid overhead irrigation. Scout daily in humid weather.',
        'organic': 'Copper-based fungicide preventatively. Destroy all infected material.'
    },
    'Potato___healthy': {
        'display': 'Healthy Potato',
        'crop': 'Potato', 'severity': 'None',
        'description': 'Your potato plant appears healthy.',
        'treatment': ['Hill up plants as they grow', 'Maintain consistent moisture', 'Monitor for Colorado potato beetle'],
        'prevention': 'Use certified seed potatoes. Rotate crops annually.',
        'organic': 'Compost mulch. Neem oil for pest prevention.'
    },
    'Raspberry___healthy': {
        'display': 'Healthy Raspberry',
        'crop': 'Raspberry', 'severity': 'None',
        'description': 'Your raspberry plant appears healthy.',
        'treatment': ['Prune old canes after harvest', 'Maintain trellis support', 'Apply balanced fertilizer in spring'],
        'prevention': 'Good air circulation. Remove old canes annually.',
        'organic': 'Compost mulch around base.'
    },
    'Soybean___healthy': {
        'display': 'Healthy Soybean',
        'crop': 'Soybean', 'severity': 'None',
        'description': 'Your soybean plant appears healthy.',
        'treatment': ['Maintain proper plant density', 'Monitor for Asian soybean rust', 'Apply foliar micronutrients if needed'],
        'prevention': 'Scout weekly for disease and pests.',
        'organic': 'Biological inoculants at planting for nitrogen fixation.'
    },
    'Squash___Powdery_mildew': {
        'display': 'Squash Powdery Mildew',
        'crop': 'Squash', 'severity': 'Medium',
        'description': 'Fungal disease causing white powdery coating on leaves, reducing photosynthesis.',
        'treatment': ['Apply potassium bicarbonate or sulfur fungicide', 'Remove severely infected leaves', 'Improve air circulation', 'Avoid high nitrogen fertilization'],
        'prevention': 'Plant resistant varieties. Space plants adequately.',
        'organic': 'Baking soda + neem oil spray weekly. Milk dilution spray (40% milk:water).'
    },
    'Strawberry___Leaf_scorch': {
        'display': 'Strawberry Leaf Scorch',
        'crop': 'Strawberry', 'severity': 'Medium',
        'description': 'Fungal disease causing purple-bordered spots with light centers on strawberry leaves.',
        'treatment': ['Apply captan or thiram fungicide', 'Remove infected leaves', 'Renovate planting after harvest', 'Avoid overhead irrigation'],
        'prevention': 'Plant certified disease-free transplants. Renovate beds annually.',
        'organic': 'Copper spray at renovation. Remove all old infected foliage.'
    },
    'Strawberry___healthy': {
        'display': 'Healthy Strawberry',
        'crop': 'Strawberry', 'severity': 'None',
        'description': 'Your strawberry plant appears healthy.',
        'treatment': ['Renovate bed after fruiting', 'Apply balanced fertilizer', 'Ensure good drainage'],
        'prevention': 'Annual renovation. Good air circulation.',
        'organic': 'Compost mulch. Neem oil preventatively.'
    },
    'Tomato___Bacterial_spot': {
        'display': 'Tomato Bacterial Spot',
        'crop': 'Tomato', 'severity': 'High',
        'description': 'Bacterial disease causing water-soaked spots on leaves and fruit, turning brown with yellow halos.',
        'treatment': ['Apply copper bactericide + mancozeb mixture', 'Remove infected plant parts', 'Avoid working with wet plants', 'Use disease-free transplants'],
        'prevention': 'Use certified transplants. Crop rotation every 3 years.',
        'organic': 'Copper hydroxide spray every 7-10 days.'
    },
    'Tomato___Early_blight': {
        'display': 'Tomato Early Blight',
        'crop': 'Tomato', 'severity': 'Medium',
        'description': 'Fungal disease causing dark brown concentric ring lesions (bull\'s-eye pattern) on lower leaves.',
        'treatment': ['Apply chlorothalonil or mancozeb fungicide', 'Remove infected lower leaves', 'Mulch to prevent soil splash', 'Maintain adequate plant nutrition'],
        'prevention': 'Crop rotation. Stake plants to improve air circulation.',
        'organic': 'Copper spray every 7 days. Remove infected leaves. Mulch well.'
    },
    'Tomato___Late_blight': {
        'display': 'Tomato Late Blight',
        'crop': 'Tomato', 'severity': 'Critical',
        'description': 'Extremely destructive fungal-like disease causing large water-soaked lesions. Can destroy entire crop in days.',
        'treatment': ['Apply metalaxyl or cymoxanil + mancozeb immediately', 'Remove and bag all infected material', 'Do NOT compost infected plants', 'Consider early harvest of green fruit'],
        'prevention': 'Avoid overhead irrigation. Scout daily in cool, wet weather.',
        'organic': 'Copper-based fungicide preventatively. Destroy infected plants.'
    },
    'Tomato___Leaf_Mold': {
        'display': 'Tomato Leaf Mold',
        'crop': 'Tomato', 'severity': 'Medium',
        'description': 'Fungal disease causing olive-green to gray mold on undersides of leaves, yellowing on top.',
        'treatment': ['Improve greenhouse ventilation', 'Apply fungicide (chlorothalonil)', 'Remove infected leaves', 'Reduce humidity below 85%'],
        'prevention': 'Good ventilation in greenhouse. Avoid excessive humidity.',
        'organic': 'Copper spray. Remove infected leaves. Increase air circulation.'
    },
    'Tomato___Septoria_leaf_spot': {
        'display': 'Septoria Leaf Spot',
        'crop': 'Tomato', 'severity': 'Medium',
        'description': 'Fungal disease causing small circular spots with dark borders and light centers on leaves.',
        'treatment': ['Apply mancozeb or chlorothalonil fungicide', 'Remove infected lower leaves', 'Mulch to prevent soil splash', 'Avoid overhead watering'],
        'prevention': 'Crop rotation every 3 years. Stake plants. Use drip irrigation.',
        'organic': 'Copper spray every 7-10 days. Mulch heavily around base.'
    },
    'Tomato___Spider_mites': {
        'display': 'Spider Mites (Two-Spotted)',
        'crop': 'Tomato', 'severity': 'Medium',
        'description': 'Tiny arachnid pest causing stippled, bronze, or silvery leaves with fine webbing underneath.',
        'treatment': ['Apply miticide (abamectin or bifenazate)', 'Use strong water spray to dislodge mites', 'Apply insecticidal soap', 'Release predatory mites (Phytoseiulus persimilis)'],
        'prevention': 'Avoid water stress. Monitor in hot, dry weather. Avoid broad-spectrum insecticides.',
        'organic': 'Neem oil spray. Insecticidal soap. Release predatory mites.'
    },
    'Tomato___Target_Spot': {
        'display': 'Tomato Target Spot',
        'crop': 'Tomato', 'severity': 'Medium',
        'description': 'Fungal disease causing concentric ring lesions (target pattern) on leaves and fruit.',
        'treatment': ['Apply azoxystrobin or chlorothalonil fungicide', 'Remove infected plant debris', 'Improve plant spacing', 'Avoid excessive nitrogen'],
        'prevention': 'Crop rotation. Good air circulation. Balanced fertilization.',
        'organic': 'Copper spray preventatively. Remove and destroy infected leaves.'
    },
    'Tomato___Yellow_Leaf_Curl_Virus': {
        'display': 'Tomato Yellow Leaf Curl Virus',
        'crop': 'Tomato', 'severity': 'Critical',
        'description': 'Viral disease spread by whiteflies causing severe leaf curling, yellowing, and stunted growth. No cure.',
        'treatment': ['Remove and destroy infected plants immediately', 'Control whitefly populations with imidacloprid', 'Use reflective mulch to deter whiteflies', 'Install insect-proof nets'],
        'prevention': 'Use virus-resistant varieties. Control whiteflies aggressively from transplant.',
        'organic': 'Yellow sticky traps for whiteflies. Neem oil spray. Reflective mulch.'
    },
    'Tomato___mosaic_virus': {
        'display': 'Tomato Mosaic Virus',
        'crop': 'Tomato', 'severity': 'High',
        'description': 'Viral disease causing mosaic patterns, leaf distortion, and reduced fruit quality. Spreads by contact.',
        'treatment': ['Remove and destroy infected plants', 'Disinfect all tools with 10% bleach solution', 'Wash hands thoroughly after handling', 'Control aphid vectors'],
        'prevention': 'Use virus-free seed. Disinfect tools. Avoid tobacco use near plants.',
        'organic': 'No cure. Prevention through hygiene and resistant varieties.'
    },
    'Tomato___healthy': {
        'display': 'Healthy Tomato',
        'crop': 'Tomato', 'severity': 'None',
        'description': 'Your tomato plant appears healthy and free of visible disease.',
        'treatment': ['Continue regular watering (deep, infrequent)', 'Apply balanced fertilizer at fruit set', 'Stake or cage plants for support', 'Monitor weekly for early disease signs'],
        'prevention': 'Rotate crops annually. Mulch to prevent soil splash.',
        'organic': 'Compost tea foliar spray. Neem oil preventatively every 2 weeks.'
    },
}

SEVERITY_COLORS = {
    'None': '#1a7a4a',
    'Medium': '#c4622a',
    'High': '#c42a2a',
    'Critical': '#8b0000'
}

# ---- Database ----
DB_PATH = os.path.join(BASE_DIR, 'detections.db')

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            filename    TEXT,
            disease     TEXT,
            crop        TEXT,
            severity    TEXT,
            confidence  REAL,
            top3        TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_detection(filename, disease, crop, severity, confidence, top3):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    import json as _json
    c.execute('''
        INSERT INTO detections (timestamp, filename, disease, crop, severity, confidence, top3)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), filename, disease, crop, severity, confidence, _json.dumps(top3)))
    conn.commit()
    conn.close()

def get_history(limit=30):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY id DESC LIMIT ?', (limit,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM detections')
    total = c.fetchone()[0]
    c.execute('SELECT severity, COUNT(*) FROM detections GROUP BY severity')
    severity_dist = dict(c.fetchall())
    c.execute('SELECT crop, COUNT(*) FROM detections GROUP BY crop ORDER BY COUNT(*) DESC LIMIT 5')
    top_crops = [{'crop': r[0], 'count': r[1]} for r in c.fetchall()]
    c.execute('SELECT disease, COUNT(*) FROM detections WHERE severity != "None" GROUP BY disease ORDER BY COUNT(*) DESC LIMIT 5')
    top_diseases = [{'disease': r[0], 'count': r[1]} for r in c.fetchall()]
    conn.close()
    return {'total': total, 'severity_dist': severity_dist, 'top_crops': top_crops, 'top_diseases': top_diseases}

init_db()

# ---- Image preprocessing ----
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

# ---- Routes ----
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        allowed = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext not in allowed:
            return jsonify({'error': 'Invalid file type. Use JPG, PNG, or WEBP.'}), 400

        image_bytes = file.read()
        arr = preprocess_image(image_bytes)

        preds = model.predict(arr, verbose=0)[0]
        top_idx = np.argsort(preds)[::-1][:3]

        top3 = []
        for idx in top_idx:
            cls = CLASSES[idx]
            info = TREATMENTS.get(cls, {})
            top3.append({
                'class': cls,
                'display': info.get('display', cls.replace('___', ' — ').replace('_', ' ')),
                'confidence': round(float(preds[idx]) * 100, 1)
            })

        best_class = CLASSES[top_idx[0]]
        best_conf = round(float(preds[top_idx[0]]) * 100, 1)
        info = TREATMENTS.get(best_class, {
            'display': best_class,
            'crop': 'Unknown', 'severity': 'Unknown',
            'description': 'No information available.',
            'treatment': ['Consult a local agronomist'],
            'prevention': 'N/A', 'organic': 'N/A'
        })

        save_detection(file.filename, info['display'], info['crop'], info['severity'], best_conf, top3)

        return jsonify({
            'disease': info['display'],
            'crop': info['crop'],
            'severity': info['severity'],
            'severity_color': SEVERITY_COLORS.get(info['severity'], '#888'),
            'confidence': best_conf,
            'description': info['description'],
            'treatment': info['treatment'],
            'prevention': info['prevention'],
            'organic': info['organic'],
            'top3': top3
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history')
def api_history():
    return jsonify(get_history())

@app.route('/api/stats')
def api_stats():
    return jsonify(get_stats())

@app.route('/api/clear', methods=['DELETE'])
def clear_all():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM detections')
    conn.commit()
    conn.close()
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    print("CropGuard AI starting on http://localhost:5000")
    app.run(debug=True, port=5000)
