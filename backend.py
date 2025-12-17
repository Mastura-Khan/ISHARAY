import cv2
import numpy as np
import mediapipe as mp
import requests

import os
import re
from moviepy import VideoFileClip, concatenate_videoclips


# --- CONFIGURATION ---
SERVER_URL = "http://127.0.0.1:5000"
FONT_NAME = "Kalpurush-Bold.ttf"  

# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SCRIPT_DIR, "videos")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "output.mp4")

# Ensure videos directory exists
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)
    print(f"Created videos directory at: {VIDEO_DIR}")

#word
MEANS_WORD = np.array([
    np.float64(0.47006019444934105), np.float64(0.5165959440652239), np.float64(1.2170209920939631e-07),
    np.float64(0.4685120926467393), np.float64(0.5027695480275198), np.float64(-0.011181679841586007),
    np.float64(0.46663440094906794), np.float64(0.48614866413439956), np.float64(-0.019510935749891285),
    np.float64(0.4642780892646956), np.float64(0.47437062126517304), np.float64(-0.027371546274358144),
    np.float64(0.4613413314578833), np.float64(0.46638515959961885), np.float64(-0.034287961660568626),
    np.float64(0.46231324631637283), np.float64(0.4667927362355511), np.float64(-0.019283451704720407),
    np.float64(0.46000446129209144), np.float64(0.450102198732514), np.float64(-0.034370238296913234),
    np.float64(0.46031378795443795), np.float64(0.44209661246496945), np.float64(-0.04266048527441943),
    np.float64(0.46055047387153825), np.float64(0.4362583536823011), np.float64(-0.04740474852587782),
    np.float64(0.4624830076766128), np.float64(0.4706372978251296), np.float64(-0.022575974828048347),
    np.float64(0.4641453574400657), np.float64(0.4572031561309288), np.float64(-0.04018863092496762),
    np.float64(0.46986486602973987), np.float64(0.4543092231776067), np.float64(-0.04615473059623061),
    np.float64(0.4729511847301591), np.float64(0.4502023673668159), np.float64(-0.0476999521938832),
    np.float64(0.4630643116271028), np.float64(0.47768706359975627), np.float64(-0.026742781616659734),
    np.float64(0.46707137294751955), np.float64(0.4666144207329068), np.float64(-0.04380136175938732),
    np.float64(0.4762874849087598), np.float64(0.4645094277666981), np.float64(-0.04420095862382094),
    np.float64(0.4817746991976509), np.float64(0.46095422550387416), np.float64(-0.041200576697538696),
    np.float64(0.4636635183915155), np.float64(0.4866850977732736), np.float64(-0.031614833982852614),
    np.float64(0.46664799439779486), np.float64(0.47838837884833285), np.float64(-0.04314677833622172),
    np.float64(0.4730501483192498), np.float64(0.47570061364155947), np.float64(-0.041246743644634355),
    np.float64(0.47745938831277), np.float64(0.47219806148186544), np.float64(-0.03728623488891074)
])
SCALES_WORD = np.array([
    np.float64(0.173035524335599), np.float64(0.17123174103785815), np.float64(4.0388616822152545e-07),
    np.float64(0.14957760130075642), np.float64(0.16124759603142877), np.float64(0.03344569604741358),
    np.float64(0.12789176627579302), np.float64(0.1588335392941), np.float64(0.047124430505913494),
    np.float64(0.12289472800783728), np.float64(0.1612710257563215), np.float64(0.055778577512672435),
    np.float64(0.12701399660397286), np.float64(0.1673284088393599), np.float64(0.06399824391088725),
    np.float64(0.13598744434998167), np.float64(0.16608930509250402), np.float64(0.050515284838263494),
    np.float64(0.14483026897822132), np.float64(0.1689272676508447), np.float64(0.06414627268849109),
    np.float64(0.15068809099085625), np.float64(0.1720975354271785), np.float64(0.0704749768459573),
    np.float64(0.16065832515732612), np.float64(0.17593861140628872), np.float64(0.07432716722124136),
    np.float64(0.14451945723240525), np.float64(0.17032799557942335), np.float64(0.04291856578223273),
    np.float64(0.14845237251295854), np.float64(0.17209904930554237), np.float64(0.06054909842935352),
    np.float64(0.14955548315306946), np.float64(0.17863530278052828), np.float64(0.066111399155956),
    np.float64(0.1583932237773107), np.float64(0.18633167150025257), np.float64(0.06897860250535046),
    np.float64(0.15156646222653086), np.float64(0.17591468469944474), np.float64(0.03986449320546457),
    np.float64(0.15293333431557682), np.float64(0.17611875038923913), np.float64(0.054374425322619865),
    np.float64(0.1515374464986599), np.float64(0.18087569674446716), np.float64(0.0562588916954322),
    np.float64(0.15710909400654707), np.float64(0.18697979948571863), np.float64(0.05727103016187577),
    np.float64(0.1587680511597956), np.float64(0.18237902714502616), np.float64(0.04322869354585223),
    np.float64(0.15938055366699827), np.float64(0.1824438856574705), np.float64(0.052161365507534085),
    np.float64(0.15781373011793073), np.float64(0.1844349976151168), np.float64(0.05373306213286361),
    np.float64(0.1608892821287359), np.float64(0.18782088509312364), np.float64(0.055211427792518274)
])

#alpha
MEANS_ALPHA = np.array([
    4.76583696e-01, 6.74053574e-01, -1.12268572e-04, 4.64894092e-01, 6.17634188e-01, -3.19521322e-02, 4.65599428e-01,
    5.35721938e-01, -5.60489619e-02, 4.74558482e-01, 4.72480834e-01, -7.65044485e-02, 4.83939101e-01, 4.27397032e-01,
    -9.10640385e-02, 4.90854491e-01, 4.61275049e-01, -5.04848107e-02, 4.98728745e-01, 3.73450723e-01, -8.46456483e-02,
    4.98048113e-01, 3.48935519e-01, -1.00308012e-01, 4.97088422e-01, 3.37394369e-01, -1.07311741e-01, 5.02175543e-01,
    4.74764228e-01, -5.15106988e-02, 5.08252946e-01, 3.91414181e-01, -9.08812283e-02, 5.01600252e-01, 3.92020785e-01,
    -9.89224716e-02, 4.99591766e-01, 3.95430696e-01, -9.66397137e-02, 5.10702350e-01, 5.02899418e-01, -5.61966800e-02,
    5.15703231e-01, 4.33458826e-01, -9.13313766e-02, 5.03518071e-01, 4.45249556e-01, -9.29343805e-02, 4.97366986e-01,
    4.55865112e-01, -8.54631234e-02, 5.18078778e-01, 5.38395741e-01, -6.26625256e-02, 5.21605166e-01, 4.83680472e-01,
    -8.76558640e-02, 5.13592346e-01, 4.84447352e-01, -8.97334006e-02, 5.08931161e-01, 4.87445341e-01, -8.53661359e-02

])
SCALES_ALPHA = np.array([
    9.95178120e-02, 1.28730733e-01, 7.39711861e-05, 1.04401705e-01, 1.14669301e-01, 7.69002681e-02, 1.12707886e-01,
    1.05078141e-01, 1.08205160e-01, 1.17800216e-01, 1.10737762e-01, 1.27192512e-01, 1.30578465e-01, 1.30254909e-01,
    1.42889895e-01, 8.97618217e-02, 7.33727911e-02, 1.07314546e-01, 1.02819679e-01, 9.63095577e-02, 1.22696742e-01,
    1.09558210e-01, 1.16289598e-01, 1.32878700e-01, 1.21075834e-01, 1.43188607e-01, 1.42103986e-01, 8.58183146e-02,
    7.19864809e-02, 8.24899100e-02, 9.55260037e-02, 1.04981844e-01, 1.11938443e-01, 9.46868977e-02, 1.36741418e-01,
    1.23500739e-01, 1.04097555e-01, 1.71729625e-01, 1.28562027e-01, 9.60729409e-02, 8.33093250e-02, 6.99687360e-02,
    1.04779218e-01, 1.09724886e-01, 1.06781027e-01, 9.35714060e-02, 1.31587313e-01, 1.19658614e-01, 9.41548755e-02,
    1.58303999e-01, 1.20582326e-01, 1.13128869e-01, 1.01554924e-01, 8.05559341e-02, 1.21593613e-01, 1.14953431e-01,
    1.06444416e-01, 1.14297060e-01, 1.28359969e-01, 1.18996089e-01, 1.13863267e-01, 1.48177880e-01, 1.23905116e-01
])




LABELS_WORD = {
    0: "খারাপ", 1: "সুন্দর(Beautiful)", 2: "ভালো",
    3: "আমাকে", 4: "আমার", 5: "তুমি",
    6: "রঙ", 7: "প্রতিজ্ঞা(Promise)", 8: "সালাম", 9: "তারা", 10: "চিন্তা"
}

LABELS_ALPHA = {
    0: "(অ/য়)", 1: "আ", 2: "ই/ঈ", 3: "উ/ঊ", 4: "র/ঋ/ড়/ঢ়", 5: "এ", 6: "ঐ", 7: "ও", 8: "ঔ", 9: "ক",
    10: "খ/ক্ষ", 11: "গ", 12: "ঘ", 13: "ঙ", 14: "চ", 15: "ছ", 16: "জ/ঝ", 17: "ঝ", 18: "ঞ", 19: "ট",
    20: "ঠ", 21: "ড", 22: "ঢ", 23: "ণ/ন", 24: "ত", 25: "থ", 26: "দ", 27: "ধ", 28: "প", 29: "ফ",
    30: "ব/ভ", 31: "ম", 32: "ল", 33: "শ/ষ/স", 34: "হ", 35: "ং", 36: "ঁ", 37: "০", 38: "১",
    39: "২", 40: "৩", 41: "৪", 42: "৫", 43: "৬", 44: "৭", 45: "৮", 46: "৯"
}

# Sign Language Video Mapping
SIGN_MAP = {
    "অলস": "অলস.mp4",
    "আমার": "আমার.mp4",
    "চালাক": "চালাক.mp4",
    "ধন্যবাদ": "ধন্যবাদ.mp4",
    "ভালো": "ভালো.mp4",
    "শান্ত": "শান্ত.mp4",
    "স্বাগত": "স্বাগত.mp4",

    "আপনাদের": "আপনাদের.mp4",
    "আমি": "আমি.mp4",
    "ছোট": "ছোট.mp4",
    "নতুন": "নতুন.mp4",
    "মাধ্যমিক": "মাধ্যমিক.mp4",
    "সত্য": "সত্য.mp4",
    "সালাম": "সালাম.mp4",

    "আপনি": "আপনি.mp4",
    "কৃপণ": "কৃপণ.mp4",
    "তিনি": "তিনি.mp4",
    "নাম": "নাম.mp4",
    "মিথ্যা": "মিথ্যা.mp4",

    "আমাকে": "আমাকে.mp4",
    "খারাপ": "খারাপ.mp4",
    "তোমার": "তোমার.mp4",
    "নিঃসন্দেহ": "নিঃসন্দেহ.mp4",
    "মুখ": "মুখ.mp4",
    "সুন্দর": "সুন্দর.mp4",

    "আমাদের": "আমাদের.mp4",
    "গরিব": "গরিব.mp4",
    "ধনি": "ধনি.mp4",
    "বড়": "বড়.mp4",
    "লোভ": "লোভ.mp4",
    "সে": "সে.mp4"
}


class BackendProcessor:
    def __init__(self):
        # Mediapipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

        # Detection state
        self.mode = "word"
        self.probs_ema = None
        self.cap = None

    def initialize_camera(self):
        """Initialize the camera"""
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        return self.cap.isOpened()

    def release_camera(self):
        """Release camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def set_mode(self, mode):
        """Set detection mode (word/alpha)"""
        self.mode = mode
        self.probs_ema = None

    def process_frame(self, frame):
        """Process a single frame for hand detection"""
        if frame is None:
            return None, "No frame", 0.0, []

        # Flip horizontally for mirror view
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        overlay_text = "No hand detected"
        current_detection = "No detection"
        current_confidence = 0.0
        top_predictions = []

        if results.multi_hand_landmarks:
            lm = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, lm, self.mp_hands.HAND_CONNECTIONS)

            # Extract features
            features = np.array([[p.x, p.y, p.z] for p in lm.landmark]).flatten()

            # Normalize features
            if self.mode == "word":
                norm_features = (features - MEANS_WORD) / SCALES_WORD
            else:
                norm_features = (features - MEANS_ALPHA) / SCALES_ALPHA

            # Send to server
            try:
                endpoint = f"{SERVER_URL}/predict/{self.mode}"
                resp = requests.post(endpoint, json={'features': norm_features.tolist()}, timeout=0.2)

                if resp.status_code == 200:
                    probs = np.array(resp.json()['probabilities'])

                    # Smoothing
                    if self.probs_ema is None:
                        self.probs_ema = probs
                    else:
                        self.probs_ema = 0.3 * probs + 0.7 * self.probs_ema

                    idx = np.argmax(self.probs_ema)
                    conf = self.probs_ema[idx] * 100

                    # Get label
                    current_map = LABELS_WORD if self.mode == "word" else LABELS_ALPHA
                    label_text = current_map.get(idx, "?")

                    if conf > 50:
                        overlay_text = f"{label_text} ({conf:.1f}%)"
                        current_detection = label_text
                        current_confidence = conf

                        # Get top 3 predictions
                        top_indices = np.argsort(self.probs_ema)[-3:][::-1]
                        for top_idx in top_indices:
                            pred_text = current_map.get(top_idx, "?")
                            pred_conf = self.probs_ema[top_idx] * 100
                            top_predictions.append((pred_text, pred_conf))
            except Exception:
                overlay_text = "Connection error"
                current_detection = "Connection error"
                current_confidence = 0.0

        # Convert BGR to RGB for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame_rgb, overlay_text, current_confidence, top_predictions, current_detection

    def get_camera_frame(self):
        """Get a frame from the camera"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return ret, frame
        return False, None

    def clean_word(self, word):
        """Clean Bangla text - keep only Bangla characters"""
        return re.sub(r'[^\u0980-\u09FF]', '', word)

    def generate_sign_video(self, text, progress_callback=None):
        """Generate sign language video from text"""
        words = text.strip().split()
        clips = []
        status_messages = []

        for word in words:
            cleaned = self.clean_word(word)

            if cleaned in SIGN_MAP:
                video_path = os.path.join(VIDEO_DIR, SIGN_MAP[cleaned])

                if os.path.exists(video_path):
                    message = f"[✔] Adding: {cleaned} → {SIGN_MAP[cleaned]}"
                    status_messages.append(message)
                    if progress_callback:
                        progress_callback(message)
                    clips.append(VideoFileClip(video_path))
                else:
                    message = f"[✘] FILE MISSING for: {cleaned} → {video_path}"
                    status_messages.append(message)
                    if progress_callback:
                        progress_callback(message)
            else:
                message = f"[✘] No sign available for: {cleaned}"
                status_messages.append(message)
                if progress_callback:
                    progress_callback(message)

        if not clips:
            final_message = "❌ No videos were added. Cannot generate output."
            status_messages.append(final_message)
            if progress_callback:
                progress_callback(final_message)
            return False, status_messages, None

        if progress_callback:
            progress_callback("⏳ Creating final video...")

        try:
            # Concatenate videos
            final = concatenate_videoclips(clips, method="compose")
            final.write_videofile(OUTPUT_FILE, codec="libx264", audio_codec="aac")

            final_message = f"🎉 DONE! Final BdSL video saved as: {OUTPUT_FILE}"
            status_messages.append(final_message)
            if progress_callback:
                progress_callback(final_message)

            return True, status_messages, OUTPUT_FILE
        except Exception as e:
            error_msg = f"Error generating video: {str(e)}"
            status_messages.append(f"❌ ERROR: {error_msg}")
            if progress_callback:
                progress_callback(f"❌ ERROR: {error_msg}")
            return False, status_messages, None

    def get_available_words(self):
        """Get list of available words for sign language"""
        return list(SIGN_MAP.keys())

    def check_video_file(self, filepath):
        """Check if a video file exists and can be opened"""
        return os.path.exists(filepath) if filepath else False

    def get_video_capture(self, filepath):
        """Get video capture object for a file"""
        if os.path.exists(filepath):
            return cv2.VideoCapture(filepath)
        return None