# 🏗️ OSHA Construction Site Safety Assistant

An AI-powered chatbot that automatically analyzes construction site images to detect OSHA safety compliance violations using YOLOv8 object detection.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Features

- **Real-time Safety Detection**: Identifies missing PPE (hard hats, safety vests, gloves, masks, safety shoes)
- **Confidence Scoring**: Reports detection confidence levels (≥75% threshold)
- **Actionable Feedback**: Provides OSHA-compliant corrective actions
- **Risk Assessment**: Categorizes violations into LOW/MEDIUM/HIGH risk levels
- **Visual Annotations**: Highlights detected violations with bounding boxes
- **User-Friendly Interface**: Gradio-based chatbot for easy image upload and analysis
- **Batch Processing**: Analyze multiple images at once
- **Detailed Reports**: Generate compliance reports with timestamps

## 📋 Table of Contents

- [Demo](#demo)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Dataset](#dataset)
- [Training](#training)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## 🎬 Demo

![Demo Screenshot](assets/demo_screenshot.png)

### Sample Output:
```
============================================================
🏗️  OSHA SAFETY COMPLIANCE REPORT
============================================================
📅 Timestamp: 2026-01-22 14:30:45

⚠️  SAFETY VIOLATIONS DETECTED:
------------------------------------------------------------

❌ HARDHAT
   • Count: 2 instance(s)
   • Detected as: NO-Hardhat
   • Avg Confidence: 89.3%

   📋 Action Required:
   Ensure all workers wear ANSI-approved hard hats (Type I or II)
   before entering the construction zone. Post signage at entry points.

------------------------------------------------------------
🎯 OVERALL RISK LEVEL: HIGH
   ⛔ Critical safety issues detected. IMMEDIATE action required.
============================================================
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB RAM minimum (16GB recommended)

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/osha-safety-assistant.git
cd osha-safety-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Google Colab Installation

```python
# Install required packages
!pip install ultralytics gradio roboflow opencv-python-headless pillow

# Clone repository
!git clone https://github.com/yourusername/osha-safety-assistant.git
%cd osha-safety-assistant

# Run the application
!python main.py
```

## ⚡ Quick Start

### Option 1: Use Pre-trained Model (Demo)

```python
python main.py
# Select mode 2 when prompted
```

### Option 2: Train on Custom Dataset

```python
python main.py
# Select mode 1 when prompted
# Follow the prompts to configure training
```

### Option 3: Use in Jupyter/Colab Notebook

```python
from main import create_chatbot_interface

# Launch interface
interface = create_chatbot_interface('path/to/model.pt')
interface.launch(share=True)
```

## 📖 Usage

### Basic Usage

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Upload an image** through the Gradio interface

3. **Review the analysis**:
   - Safety violations detected
   - Confidence scores
   - Annotated image with bounding boxes
   - Recommended corrective actions

### Advanced Usage

#### Batch Processing

```python
from main import batch_process_images

# Process multiple images
reports = batch_process_images('path/to/images/', model_path='best.pt')

# Access individual reports
for report in reports:
    print(f"File: {report['filename']}")
    print(report['report'])
```

#### Custom Confidence Threshold

```python
from main import OSHASafetyAnalyzer

# Initialize with custom threshold
analyzer = OSHASafetyAnalyzer(
    model_path='best.pt',
    confidence_threshold=0.80  # 80% confidence
)

# Analyze image
results = analyzer.analyze_image('site_photo.jpg')
report = analyzer.generate_report(results)
```

#### Save Reports

```python
from main import save_report

# Generate and save report
save_report(report_text, filename='inspection_report_2026_01_22.txt')
```

## 📊 Dataset

This project uses the [Construction Site Safety Dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) from Roboflow Universe.

### Dataset Classes:

- Hardhat / NO-Hardhat
- Safety Vest / NO-Safety Vest
- Gloves / NO-Gloves
- Mask / NO-Mask
- Safety Shoes / NO-Safety Shoes
- Person
- Machinery
- Safety Cone

### Using Your Own Dataset

1. **Prepare your dataset** in YOLO format:
   ```
   dataset/
   ├── train/
   │   ├── images/
   │   └── labels/
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── data.yaml
   ```

2. **Update data.yaml**:
   ```yaml
   path: ./dataset
   train: train/images
   val: valid/images
   
   names:
     0: Hardhat
     1: NO-Hardhat
     # Add your classes
   ```

3. **Run training**:
   ```bash
   python main.py
   # Select mode 1
   ```

## 🎓 Training

### Training Parameters

```python
# In main.py, configure:
epochs = 50          # Number of training iterations
batch_size = 16      # Images per batch
imgsz = 640         # Image size
model = 'yolov8n'   # Model size (n/s/m/l/x)
```

### Model Options

- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (recommended)
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (best accuracy)

### Training on GPU

```python
# Automatic GPU detection
device = 0 if torch.cuda.is_available() else 'cpu'
```

### Monitor Training

Training metrics are saved to:
```
industry_safety/yolov8n_osha/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.png
├── confusion_matrix.png
└── training_plots/
```

## 📁 Project Structure

```
osha-safety-assistant/
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── LICENSE                # MIT License
├── .gitignore            # Git ignore rules
├── data.yaml             # Dataset configuration
├── assets/               # Images and media
│   └── demo_screenshot.png
├── models/               # Trained models
│   └── best.pt
├── datasets/             # Training data
│   ├── train/
│   ├── valid/
│   └── test/
├── reports/              # Generated reports
└── notebooks/            # Jupyter notebooks
    └── training_analysis.ipynb
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```env
ROBOFLOW_API_KEY=your_api_key_here
CONFIDENCE_THRESHOLD=0.75
MODEL_PATH=models/best.pt
```

### Custom Settings

Edit `config.py`:

```python
# Detection settings
CONFIDENCE_THRESHOLD = 0.75
IOU_THRESHOLD = 0.45

# Model settings
MODEL_SIZE = 'yolov8n'
IMAGE_SIZE = 640

# Risk thresholds
HIGH_RISK_VIOLATIONS = ['hardhat', 'safety_shoes']
CRITICAL_VIOLATION_COUNT = 3
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the object detection framework
- [Roboflow](https://roboflow.com/) for the construction safety dataset
- [Gradio](https://gradio.app/) for the user interface
- OSHA for safety compliance guidelines


Project Link: [https://github.com/yourusername/osha-safety-assistant](https://github.com/hamzajami/osha-safety-assistant)

## 🔮 Future Enhancements

- [ ] Real-time video stream processing
- [ ] Mobile app integration
- [ ] Multi-language support
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Integration with incident reporting systems
- [ ] Email/SMS alerts for critical violations
- [ ] Dashboard for site managers
- [ ] Historical trend analysis
- [ ] PDF report generation
- [ ] Integration with IoT cameras

## 📚 Documentation

For detailed documentation, visit our [Wiki](https://github.com/hamzajami/osha-safety-assistant/wiki).


---

**Made with ❤️ for construction safety**
