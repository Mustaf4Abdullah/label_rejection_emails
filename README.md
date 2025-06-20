# Job Application Rejection Email Detector 📧🤖

An intelligent system that automatically detects and labels job application rejection emails in your Gmail inbox using a hybrid approach combining machine learning and regex pattern matching.

## 🎯 What it Does

This tool automatically:
- Scans your Gmail inbox for job application emails
- Uses a trained machine learning model to identify rejection emails
- Applies regex pattern matching as a backup detection method
- Automatically labels detected rejection emails with a custom Gmail label
- Supports both English and Turkish rejection patterns

## 🧠 How it Works

The system uses a **hybrid detection approach** combining:

1. **Machine Learning Model**: A trained classifier using TF-IDF vectorization
2. **Regex Pattern Matching**: Pre-defined patterns for common rejection phrases
3. **Confidence Scoring**: Combines both methods for improved accuracy

### Why the Hybrid Approach?

The dataset used for training is relatively small (limited rejection email samples), which makes the ML model somewhat weak on its own. To compensate for this limitation, we implemented additional regex pattern matching that catches common rejection phrases the model might miss. This hybrid approach significantly improves detection accuracy.

## 📊 Dataset

The machine learning model was trained on the [Application Rejection Emails Dataset](https://www.kaggle.com/datasets/sethpoly/application-rejection-emails) from Kaggle.

**Dataset Limitations:**
- Small dataset size (~100 samples)
- Limited variety in rejection email formats
- Potential overfitting due to data scarcity

This is why we enhanced the system with regex patterns to catch rejection emails that the ML model might miss due to its limited training data.

## 🚀 Setup Instructions

### Prerequisites

- Python 3.7+
- Gmail account
- Google Cloud Console access

### 1. Enable Gmail API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Gmail API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Gmail API"
   - Click on it and press "Enable"

### 2. Get Client Secret File

1. In Google Cloud Console, go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth 2.0 Client IDs"
3. Configure OAuth consent screen if prompted
4. Choose "Desktop application" as application type
5. Download the JSON file
6. Rename it to match the filename in the code or update the `CLIENT_SECRET_FILE` variable

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `google-api-python-client`
- `google-auth-httplib2`
- `google-auth-oauthlib`
- `pandas`
- `scikit-learn`
- `joblib`
- `beautifulsoup4`
- `langdetect` (optional, for language detection)

### 4. Run the Application

```bash
python label_rejection_emails_fixed.py
```

On first run:
1. A browser window will open for Gmail authentication
2. Grant the required permissions
3. The app will create a `token.pickle` file for future authentications

## 📁 Project Structure

```
├── label_rejection_emails_fixed.py    # Main application script
├── emails.csv                         # Training dataset
├── rejection_model.pkl                # Trained ML model
├── vectorizer.pkl                      # TF-IDF vectorizer
├── client_secret_*.json               # OAuth2 credentials (you need to add this)
├── token.pickle                       # Generated authentication token
└── README.md                          # This file
```

## 🔧 Configuration

### Key Variables

- `LABEL_NAME`: Gmail label name for rejected emails (default: "Red Mailleri")
- `SCOPES`: Gmail API permissions required
- `CLIENT_SECRET_FILE`: Path to your OAuth2 credentials file

### Model Parameters

- **ML Threshold**: 0.35 (confidence threshold for ML prediction)
- **Hybrid Threshold**: 1.0 (combined score from regex + ML)
- **Processing Limit**: 1000 emails per run

## 🎯 Detection Patterns

The system recognizes various rejection patterns including:

**English:**
- "we regret to inform you"
- "unfortunately, we will not be moving forward"
- "your application was not selected"
- "we have chosen another candidate"
- "position has been filled"

**Turkish:**
- "başvurunuz için teşekkür ederiz"
- "üzülerek belirtmek isteriz ki"
- "değerlendirme süreci sonunda olumsuz sonuçlanmıştır"

## 📈 Performance Notes

- **Accuracy**: Hybrid approach provides better accuracy than ML alone
- **False Positives**: Minimal due to conservative thresholds
- **Processing Speed**: ~1-2 seconds per email
- **API Limits**: Respects Gmail API rate limits

## 🛠️ Model Limitations

Due to the small training dataset:
- The ML model may miss some rejection email formats
- Performance varies with email complexity and language
- Regex patterns help compensate for model weaknesses
- Consider retraining with more data for better performance

## 🔒 Privacy & Security

- All processing happens locally on your machine
- OAuth2 ensures secure Gmail access
- No email content is stored or transmitted externally
- Token files contain only authentication credentials

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Expand the training dataset
- Add more language support
- Improve regex patterns
- Optimize processing speed

## 📜 License

This project is open source. Please ensure you comply with Gmail API terms of service.

## ⚠️ Disclaimer

This tool is for personal use only. Always review automatically labeled emails to ensure accuracy. The authors are not responsible for any missed or incorrectly labeled emails.

---

*Built with ❤️ to help job seekers better organize their application responses*
