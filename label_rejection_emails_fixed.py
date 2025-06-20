import base64
import os
import joblib
import pandas as pd
import re
from bs4 import BeautifulSoup
try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("⚠️ langdetect not installed. Install with: pip install langdetect")
    LANGDETECT_AVAILABLE = False

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

# === CONFIG ===
SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
CLIENT_SECRET_FILE = 'client_secret_643607407325-bvuuvthnu85hg1bt6hlrha7o5phih8ld.apps.googleusercontent.com.json'
MODEL_PATH = 'rejection_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
TOKEN_PATH = 'token.pickle'
LABEL_NAME = 'Red Mailleri'

# === Load ML model ===
def load_models():
    try:
        clf = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Models loaded successfully")
        return clf, vectorizer
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None, None

# === Authenticate to Gmail API ===
def authenticate_gmail():
    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'wb') as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)

# === Get or create label ID ===
def get_or_create_label_id(service, label_name):
    try:
        labels = service.users().labels().list(userId='me').execute().get('labels', [])
        for label in labels:
            if label['name'].lower() == label_name.lower():
                return label['id']
        label_object = {
            'name': label_name,
            'labelListVisibility': 'labelShow',
            'messageListVisibility': 'show'
        }
        created_label = service.users().labels().create(userId='me', body=label_object).execute()
        return created_label['id']
    except Exception as e:
        print(f"❌ Error with label operations: {e}")
        return None

# === Robust content extraction with recursive search for plain text or html parts ===
def extract_part_content(part):
    body = part.get('body', {})
    data = body.get('data')
    if data:
        try:
            decoded_bytes = base64.urlsafe_b64decode(data.encode('ASCII'))
            charset = part.get('headers', [])
            try:
                return decoded_bytes.decode('utf-8', errors='ignore')
            except:
                return decoded_bytes.decode('latin1', errors='ignore')
        except Exception as e:
            return ""
    if 'parts' in part:
        for subpart in part['parts']:
            content = extract_part_content(subpart)
            if content:
                return content
    return ""

def get_message_content(message):
    try:
        payload = message['payload']
        mime_type = payload.get('mimeType', '')
        if mime_type == 'text/plain':
            content = extract_part_content(payload)
            return content
        elif mime_type == 'text/html':
            html_content = extract_part_content(payload)
            return BeautifulSoup(html_content, 'html.parser').get_text(separator='\n')
        parts = payload.get('parts', [])
        text_parts = []
        for part in parts:
            if part.get('mimeType') == 'text/plain':
                content = extract_part_content(part)
                if content:
                    text_parts.append(content)
            elif part.get('mimeType') == 'text/html':
                html_content = extract_part_content(part)
                if html_content:
                    text_parts.append(BeautifulSoup(html_content, 'html.parser').get_text(separator='\n'))
            else:
                content = extract_part_content(part)
                if content:
                    text_parts.append(content)
        return "\n".join(text_parts).strip()
    except Exception as e:
        print(f"⚠️ Error extracting message content: {e}")
        return ""

REJECTION_PATTERNS = [
    r"we (?:regret|are sorry|are saddened) to inform you",
    r"your application (?:was|has been) (?:rejected|not successful|unsuccessful|declined|denied|unsuitable)",
    r"thank you for your (?:application|interest)(?:,)? but",
    r"unfortunately,? we (?:will not|can't) (?:be moving forward|proceed)",
    r"we (?:have|have already|will) (?:chosen|selected) another candidate",
    r"your profile does not (?:meet|fit|match) (?:our|the) requirements",
    r"we appreciate your (?:interest|time|application|effort)(?:,)? but",
    r"this position has been (?:filled|closed)",
    r"not selected for further consideration",
    r"we will keep your (?:resume|cv|application) on file",
    r"no longer considered for this position",
    r"we do not have a match for your skills",
    r"position has been filled by another candidate",
    r"we won't be proceeding with your application",
    r"this time we decided to move forward with other applicants",
    r"after careful consideration, we decided not to proceed",
    r"we decided not to continue with your candidacy",
    r"we are moving forward with other candidates",
    r"your candidacy was not selected",
    r"not a match for the role",
    r"you were not shortlisted",
    r"better suited candidates",
    r"başvurunuz için teşekkür ederiz",
    r"üzülerek belirtmek isteriz ki",
    r"bu pozisyon için başka bir aday ile ilerlemeye karar verdik",
    r"değerlendirme süreci sonunda olumsuz sonuçlanmıştır",
    r"ne yazık ki sizi sonraki aşamaya davet edemiyoruz",
    r"başvurunuz olumsuz sonuçlanmıştır",
    r"başvurunuz değerlendirilmiş ancak uygun bulunmamıştır",
    r"şartlarımızı karşılamadığınız için",
    r"şirket ihtiyaçları doğrultusunda sizinle devam etmeme kararı aldık",
    r"bu pozisyon için aradığımız kriterlere uygun bulunmadınız",
    r"pozisyon başka bir adayla doldurulmuştur",
    r"sizi değerlendirmeye alamayacağız",
    r"görüşme süreci sizinle devam etmeyecektir",
    r"bu aşamada ilerlememe kararı aldık",
    r"başka adaylarla devam etme kararı alınmıştır",
    r"sizi işe al(a)mayacağımızı üzülerek bildiririz",
    r"başvurunuz değerlendirmeye alınmamıştır",
    r"önümüzdeki süreçlerde tekrar görüşmek dileğiyle",
]

REJECTION_REGEX = re.compile("|".join(REJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)

# === Modified Hybrid Rejection Detection ===
def is_rejection_email(text, clf, vectorizer, threshold=0.35):
    try:
        if len(text.strip()) < 10:
            return False

        

        regex_score = int(bool(REJECTION_REGEX.search(text)))
        if regex_score:
            print("   Regex heuristic: rejection phrase matched")

        vector = vectorizer.transform([text])
        confidence = clf.predict_proba(vector)[0][1]
        print(f"   ML Confidence: {confidence:.3f}")

        score = regex_score + confidence
        return score >= 1.0  # Hybrid threshold (1 from regex + 0.35 from ML)
    except Exception as e:
        print(f"⚠️ Prediction failed: {e}")
        return False

def get_email_subject(message):
    headers = message['payload'].get('headers', [])
    for header in headers:
        if header['name'].lower() == 'subject':
            return header['value']
    return "No Subject"

def fetch_paginated_messages(service, skip, limit, label_ids=None):
    all_messages = []
    next_page_token = None
    fetched = 0

    while fetched < skip + limit:
        response = service.users().messages().list(
            userId='me',
            labelIds=label_ids or ['INBOX'],
            maxResults=100,
            pageToken=next_page_token
        ).execute()

        messages = response.get('messages', [])
        next_page_token = response.get('nextPageToken')
        if not messages:
            break

        for msg in messages:
            fetched += 1
            if fetched > skip:
                all_messages.append(msg)
            if len(all_messages) >= limit:
                return all_messages

        if not next_page_token:
            break

    return all_messages

def main():
    print("🚀 Starting Gmail Rejection Detector")

    clf, vectorizer = load_models()
    if clf is None or vectorizer is None:
        return

    service = authenticate_gmail()
    if service is None:
        return

    label_id = get_or_create_label_id(service, LABEL_NAME)
    if not label_id:
        return

    try:
        print("📬 Fetching messages 501 to 1000 from inbox...")
        messages = fetch_paginated_messages(service, skip=0, limit=1000, label_ids=['INBOX'])
        print(f"📧 Found {len(messages)} emails to process")
        rejection_count = 0

        for i, msg in enumerate(messages, 1):
            msg_id = msg['id']
            print(f"\n📧 Processing email {i}/{len(messages)} (ID: {msg_id})")

            try:
                full_msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
                content = get_message_content(full_msg)
                subject = get_email_subject(full_msg)

                print(f"   Subject: {subject[:60]}...")
                if not content.strip():
                    print("   ⚠️ No content found, skipping")
                    continue

                preview_text = content[:100].replace('\n', ' ').replace('\r', '')
                print(f"   Content preview: {preview_text}...")

                if is_rejection_email(content, clf, vectorizer):
                    print(f"   🔴 REJECTION DETECTED! Applying label...")
                    service.users().messages().modify(
                        userId='me', id=msg_id, body={'addLabelIds': [label_id]}
                    ).execute()
                    rejection_count += 1
                    print(f"   ✅ Label applied successfully")
                else:
                    print(f"   ✅ Not a rejection email")
            except Exception as e:
                print(f"   ❌ Error processing email {msg_id}: {e}")
                continue

        print("\n🎯 Processing complete!")
        print(f"📊 Found and labeled {rejection_count} rejection emails out of {len(messages)} total emails")

    except Exception as e:
        print(f"❌ Main error: {e}")

if __name__ == "__main__":
    main()
