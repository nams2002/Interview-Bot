import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
import speech_recognition as sr
import mediapipe as mp
import matplotlib.pyplot as plt
import openai
from ultralytics import YOLO
import os
import json
import random
from gtts import gTTS
import pygame
from pygame import mixer
import threading
from PIL import Image
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import re
import nltk
from nltk.tokenize import sent_tokenize
from scipy.stats import entropy
import functools
import logging
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Interview System",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data with error handling
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data with caching"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_data()

@st.cache_resource
class AITextDetector:
    """Optimized AI text detection with caching and performance improvements"""

    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limit features for performance
        self.api_key = None

        # Compiled regex patterns for better performance
        self.patterns = {
            'repetitive_phrases': re.compile(r'\b(\w+\s+\w+\s+\w+)\b.+\b\1\b', re.IGNORECASE),
            'common_ai_phrases': re.compile(r'\b(as an ai language model|as an assistant|as a language model|in conclusion)\b', re.IGNORECASE),
            'complex_transitions': re.compile(r'\b(furthermore|nevertheless|consequently|subsequently|additionally)\b', re.IGNORECASE),
            'perfect_grammar': re.compile(r'\b(whom|thereby|wherein|whereby|thus)\b', re.IGNORECASE),
            'standardized_responses': re.compile(r'\b(i would be happy to|certainly|absolutely|it\'s important to note)\b', re.IGNORECASE),
            'extended_elaboration': re.compile(r'\b(specifically|particularly|notably|crucially|significantly)\b', re.IGNORECASE)
        }

        # Use LRU cache for better memory management
        self.cached_scores = {}
        self.max_cache_size = 100

        self.confidence_thresholds = {
            'ai_score': 0.7,
            'plagiarism': 0.5
        }

        # Optimized tracking
        self.api_calls = 0
        self.api_success = 0
        self.detection_history = []
    
    def set_api_key(self, api_key: str) -> None:
        """Set API key for Eden AI detection service"""
        self.api_key = api_key
        # Clear cached scores when changing API
        self.cached_scores = {}

    def _manage_cache(self) -> None:
        """Manage cache size to prevent memory issues"""
        if len(self.cached_scores) > self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self.cached_scores.keys())[:len(self.cached_scores) - self.max_cache_size + 10]
            for key in keys_to_remove:
                del self.cached_scores[key]

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def check_eden_ai(self, text: str) -> Optional[Tuple[float, Dict[str, Any]]]:
        """Check text using Eden AI's content moderation API for AI detection

        Eden AI aggregates multiple AI detection services in one API
        """
        if not self.api_key or len(text) < 100:
            return None

        # Cache lookup with hash for better memory management
        cache_key = hash(text[:1000])  # Use hash instead of string for memory efficiency
        if cache_key in self.cached_scores:
            return self.cached_scores[cache_key]

        try:
            # Track API usage
            self.api_calls += 1

            # Eden AI endpoint for content detection
            url = "https://api.edenai.run/v2/text/ai_detection"

            # Get providers from session state with fallback
            providers = getattr(st.session_state, 'eden_providers', "sapling,writer,originalityai")

            # Ensure we have at least one provider
            if not providers:
                providers = "sapling"
            
            payload = {
                "providers": providers,
                "text": text,
                "fallback_providers": ""
            }
            
            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                "authorization": f"Bearer {self.api_key}"
            }
            
            # Add timeout and error handling
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                self.api_success += 1
                result = response.json()
                
                # Process Eden AI results
                ai_scores = []
                confidence_scores = []
                details = {}
                providers_used = []
                
                # Extract scores from each provider
                for provider_data in result.items():
                    provider = provider_data[0]
                    data = provider_data[1]
                    
                    if isinstance(data, dict) and 'ai_score' in data:
                        score = data['ai_score']
                        ai_scores.append(score)
                        # Calculate confidence based on provider-specific metrics
                        confidence = data.get('confidence', 0.8)  # Default confidence if not provided
                        confidence_scores.append(confidence)
                        details[provider] = data
                        providers_used.append(provider)
                
                # Calculate weighted average score if we have results
                if ai_scores:
                    # Weight scores by confidence
                    if sum(confidence_scores) > 0:
                        weighted_scores = [s * c for s, c in zip(ai_scores, confidence_scores)]
                        avg_score = sum(weighted_scores) / sum(confidence_scores)
                    else:
                        avg_score = sum(ai_scores) / len(ai_scores)
                    
                    # Calculate overall confidence metric
                    provider_agreement = self._calculate_agreement(ai_scores)
                    overall_confidence = min(1.0, sum(confidence_scores) / len(confidence_scores) * provider_agreement)
                    
                    # Add timestamp for tracking
                    timestamp = datetime.now().isoformat()
                    
                    # Create comprehensive analysis
                    analysis = {
                        'ai_probability': avg_score,
                        'is_likely_ai': avg_score > self.confidence_thresholds['ai_score'],
                        'source': 'eden_ai',
                        'details': details,
                        'providers_used': providers_used,
                        'confidence': overall_confidence,
                        'timestamp': timestamp,
                        'text_length': len(text),
                        'agreement_score': provider_agreement
                    }
                    
                    # Track detection history
                    self.detection_history.append({
                        'timestamp': timestamp,
                        'ai_score': avg_score,
                        'providers': len(providers_used),
                        'text_length': len(text),
                        'confidence': overall_confidence
                    })
                    
                    # Cache the result
                    self.cached_scores[cache_key] = (avg_score, analysis)
                    return avg_score, analysis
                    
            return None
        except Exception as e:
            print(f"Error with Eden AI detection API: {e}")
            return None
    
    def _calculate_agreement(self, scores):
        """Calculate how much the providers agree with each other (0-1 scale)"""
        if len(scores) <= 1:
            return 1.0
        
        # Calculate standard deviation and normalize to 0-1 scale
        # Lower std dev means more agreement
        std_dev = np.std(scores)
        # Convert std_dev to agreement score (1 - normalized std_dev)
        agreement = 1.0 - min(1.0, std_dev * 2)
        return agreement

    def check_linguistic_patterns(self, text: str) -> Dict[str, float]:
        """Check text for linguistic patterns common in AI-generated content"""
        scores = {}
        text_lower = text.lower()  # Convert once for efficiency
        word_count = len(text.split())

        # Check for pattern matches using compiled regex
        for pattern_name, pattern in self.patterns.items():
            matches = len(pattern.findall(text_lower))
            scores[pattern_name] = min(matches / max(word_count / 20, 1), 1.0)
        
        # Check sentence complexity and variance
        sentences = sent_tokenize(text)
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            scores['sentence_length_variance'] = np.var(sentence_lengths) / (np.mean(sentence_lengths) + 1)
            
            # Perplexity (higher is more human-like)
            if len(sentences) > 1:
                word_counts = {}
                for sentence in sentences:
                    words = sentence.lower().split()
                    for word in words:
                        word_counts[word] = word_counts.get(word, 0) + 1
                
                probs = np.array(list(word_counts.values())) / sum(word_counts.values())
                scores['language_entropy'] = entropy(probs)
            else:
                scores['language_entropy'] = 0
                
            # Check for repeated sentence beginnings
            beginnings = [s.split()[:2] for s in sentences if len(s.split()) >= 2]
            if beginnings:
                beginning_counts = {}
                for b in beginnings:
                    key = ' '.join(b).lower()
                    beginning_counts[key] = beginning_counts.get(key, 0) + 1
                
                # Calculate the ratio of repeated beginnings
                repeated = sum(1 for count in beginning_counts.values() if count > 1)
                scores['repeated_beginnings'] = min(1.0, repeated / len(sentences))
        
        # Check for unnatural token repetition
        words = text.lower().split()
        if words:
            # Count distribution of unique words
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
            
            # Calculate the lexical diversity (unique words / total words)
            lexical_diversity = len(word_counts) / len(words)
            scores['lexical_diversity'] = lexical_diversity
        
        return scores

    def detect_ai_content(self, text, threshold=0.6):
        """Detect if content is likely AI-generated, using Eden AI if available
        
        Returns:
        - float: AI probability score (0-1 scale)
        - dict: Detailed analysis results
        """
        if not text or len(text) < 50:
            return 0.0, {"error": "Text too short for reliable detection"}
        
        # Normalize text to ensure consistent detection
        normalized_text = text.strip()
        
        # Check if we've already analyzed this text
        cache_key = normalized_text[:1000]  # Use truncated text as cache key
        if cache_key in self.cached_scores:
            return self.cached_scores[cache_key]
        
        # Try Eden AI first if API key is available
        eden_result = self.check_eden_ai(normalized_text)
        if eden_result:
            return eden_result
        
        # Fallback to enhanced built-in detection
        pattern_scores = self.check_linguistic_patterns(normalized_text)
        
        # Calculate features
        repetitive_score = pattern_scores.get('repetitive_phrases', 0)
        ai_phrases_score = pattern_scores.get('common_ai_phrases', 0)
        transition_score = pattern_scores.get('complex_transitions', 0) 
        grammar_score = pattern_scores.get('perfect_grammar', 0)
        standard_resp_score = pattern_scores.get('standardized_responses', 0)
        elab_score = pattern_scores.get('extended_elaboration', 0)
        
        # Add new features
        lexical_diversity = pattern_scores.get('lexical_diversity', 0.5)  # Default to 0.5 if not available
        repeated_beginnings = pattern_scores.get('repeated_beginnings', 0)
        
        # Calculate sentence statistics
        sentences = sent_tokenize(normalized_text)
        sentence_length_variance = pattern_scores.get('sentence_length_variance', 0)
        entropy_score = pattern_scores.get('language_entropy', 0)
        
        # Calculate final score (weighted average)
        ai_probability = (
            repetitive_score * 0.15 +
            ai_phrases_score * 0.15 +
            transition_score * 0.1 +
            grammar_score * 0.1 +
            standard_resp_score * 0.1 +  # New weight for standardized responses
            elab_score * 0.05 +          # New weight for elaborate phrases
            (1 - min(sentence_length_variance, 1.0)) * 0.15 +
            (1 - min(entropy_score / 3.5, 1.0)) * 0.1 +
            (1 - lexical_diversity) * 0.05 +  # Lower lexical diversity suggests AI
            repeated_beginnings * 0.05        # More repeated beginnings suggests AI
        )
        
        # Clamp between 0 and 1
        ai_probability = max(0.0, min(1.0, ai_probability))
        
        # Create detailed analysis
        analysis = {
            'ai_probability': ai_probability,
            'is_likely_ai': ai_probability > threshold,
            'source': 'built-in',
            'confidence': 0.7,  # Built-in detector has lower confidence than Eden AI
            'timestamp': datetime.now().isoformat(),
            'text_length': len(normalized_text),
            'details': {
                'repetitive_patterns': repetitive_score,
                'ai_phrases': ai_phrases_score,
                'complex_transitions': transition_score,
                'perfect_grammar': grammar_score,
                'standardized_responses': standard_resp_score,
                'elaborate_phrasing': elab_score,
                'sentence_variance': sentence_length_variance,
                'language_entropy': entropy_score,
                'lexical_diversity': lexical_diversity,
                'repeated_beginnings': repeated_beginnings
            }
        }
        
        # Track detection history
        self.detection_history.append({
            'timestamp': analysis['timestamp'],
            'ai_score': ai_probability,
            'providers': 0,  # Built-in detector
            'text_length': len(normalized_text),
            'confidence': 0.7
        })
        
        # Cache the result
        self.cached_scores[cache_key] = (ai_probability, analysis)
        
        return ai_probability, analysis

    def check_plagiarism(self, text):
        """Enhanced implementation to detect common plagiarism patterns"""
        if not text or len(text) < 50:
            return 0.0, {"error": "Text too short for reliable detection"}
        
        # Normalize text
        normalized_text = text.strip()
        
        # Look for patterns that might indicate copy-pasted content
        analysis = {}
        
        # Check for unusual formatting that might indicate copying
        unusual_format = len(re.findall(r'[\t\n]{2,}|\||\*{2,}|_{2,}|#{2,}', normalized_text))
        analysis['unusual_formatting'] = unusual_format > 0
        
        # Check for citation patterns
        citations = len(re.findall(r'\(\d{4}\)|\[[\d,\s]+\]|et al\.', normalized_text))
        analysis['has_citations'] = citations > 0
        
        # Check for web-specific content that might be copied
        web_elements = len(re.findall(r'https?://|www\.|\.com|\.org|\.net', normalized_text))
        analysis['has_web_elements'] = web_elements > 0
        
        # Check for academic-style writing
        academic_style = len(re.findall(r'according to|as stated in|as mentioned by|cited in', normalized_text.lower()))
        analysis['academic_style'] = academic_style > 0
        
        # Check for inconsistent formatting or styling
        # Look for mixed formatting styles (e.g., different quote styles, dash types)
        mixed_quotes = len(re.findall(r'["""].*?["""]', normalized_text)) > 0 and len(re.findall(r"'.*?'", normalized_text)) > 0
        mixed_dashes = len(re.findall(r'--', normalized_text)) > 0 and len(re.findall(r'—', normalized_text)) > 0
        analysis['inconsistent_formatting'] = mixed_quotes or mixed_dashes
        
        # Calculate overall plagiarism probability (enhanced)
        plagiarism_probability = min(1.0, (
            (0.3 if analysis['unusual_formatting'] else 0) +
            (0.2 if analysis['has_citations'] else 0) +
            (0.2 if analysis['has_web_elements'] else 0) +
            (0.15 if analysis['academic_style'] else 0) +
            (0.15 if analysis['inconsistent_formatting'] else 0) +
            (0.1 if len(normalized_text) > 1000 else 0)  # Very long answers are sometimes copy-pasted
        ))
        
        # Create timestamp for tracking
        timestamp = datetime.now().isoformat()
        
        return plagiarism_probability, {
            'plagiarism_probability': plagiarism_probability,
            'is_likely_plagiarized': plagiarism_probability > self.confidence_thresholds['plagiarism'],
            'timestamp': timestamp,
            'text_length': len(normalized_text),
            'details': analysis
        }
    
    def get_detection_stats(self):
        """Return statistics about the AI detection performance"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'api_success_rate': 0,
                'avg_confidence': 0,
                'avg_ai_score': 0
            }
        
        total = len(self.detection_history)
        api_success_rate = self.api_success / max(1, self.api_calls)
        
        # Calculate averages
        avg_confidence = sum(d['confidence'] for d in self.detection_history) / total
        avg_ai_score = sum(d['ai_score'] for d in self.detection_history) / total
        
        # Calculate distribution of scores
        ai_scores = [d['ai_score'] for d in self.detection_history]
        
        return {
            'total_detections': total,
            'api_calls': self.api_calls,
            'api_success_rate': api_success_rate,
            'avg_confidence': avg_confidence,
            'avg_ai_score': avg_ai_score,
            'ai_score_distribution': ai_scores,
            'high_confidence_ai': sum(1 for d in self.detection_history if d['ai_score'] > 0.7 and d['confidence'] > 0.8)
        }

class SecurityMonitor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        mixer.init()
        self.last_warning_time = 0
        self.warning_cooldown = 5
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.tts_lock = threading.Lock()

    def speak_text(self, text):
        """Convert text to speech and play it"""
        try:
            with self.tts_lock:
                tts = gTTS(text=text, lang='en')
                audio_file = f"speech_{int(time.time())}.mp3"
                tts.save(audio_file)
                
                while mixer.music.get_busy():
                    time.sleep(0.1)
                
                mixer.music.load(audio_file)
                mixer.music.play()
                
                while mixer.music.get_busy():
                    time.sleep(0.1)
                
                try:
                    os.remove(audio_file)
                except:
                    pass
        except Exception as e:
            print(f"Error in speech: {e}")

    def check_face_alignment(self, frame):
        """Check if face is properly aligned using MediaPipe Face Mesh"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return False, "Please face the camera"
            
        face_landmarks = results.multi_face_landmarks[0]
        nose_tip = face_landmarks.landmark[4]
        image_height, image_width = frame.shape[:2]
        nose_x = int(nose_tip.x * image_width)
        
        center_x = image_width // 2
        offset = abs(center_x - nose_x)
        max_offset = image_width * 0.1
        
        if offset > max_offset:
            return False, "Please center your face"
            
        return True, "Face properly aligned"

    def detect_prohibited_objects(self, frame, yolo_model):
        """Detect prohibited objects in frame"""
        prohibited_items = {
            'cell phone': 'Mobile phone detected',
            'laptop': 'Laptop detected',
            'book': 'Book detected',
            'tablet': 'Tablet detected',
            'remote': 'Remote control detected',
            'keyboard': 'External keyboard detected',
            'tv': 'Television detected'
        }
        
        results = yolo_model(frame)
        warnings = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                name = yolo_model.names[cls]
                if name in prohibited_items:
                    warnings.append(prohibited_items[name])
        
        return warnings

    def issue_warning(self, warning_text):
        """Issue warning with cooldown"""
        current_time = time.time()
        if current_time - self.last_warning_time >= self.warning_cooldown:
            self.last_warning_time = current_time

class AIInterviewer:
    def __init__(self, openai_key):
        self.security_monitor = SecurityMonitor()
        self.recognizer = sr.Recognizer()
        self.yolo_model = YOLO('yolov8n.pt')
        openai.api_key = openai_key
        
        # Add the enhanced AI detector
        self.ai_detector = AITextDetector()
        
        # Set Eden AI key if available in session state
        if hasattr(st, 'session_state') and 'eden_api_key' in st.session_state and st.session_state.eden_api_key:
            self.ai_detector.set_api_key(st.session_state.eden_api_key)
        
        self.skip_phrases = [
            "i don't know",
            "i dont know",
            "don't know",
            "dont know",
            "ask something else",
            "different question",
            "skip this",
            "another question",
            "can we skip",
            "next question"
        ]
        
        self.violation_history = []
        self.response_metrics = {
            'total_questions': 0,
            'questions_skipped': 0,
            'avg_response_time': 0,
            'total_violations': 0,
            'start_times': {},
            'response_times': []
        }
        
        self.stages = {
            'introduction': {
                'name': 'Introduction',
                'description': 'Getting to know you and your background'
            },
            'technical': {
                'name': 'Technical Assessment',
                'description': 'Evaluating technical skills and knowledge'
            },
            'behavioral': {
                'name': 'Behavioral Questions',
                'description': 'Understanding your work style and experiences'
            },
            'role_specific': {
                'name': 'Role-Specific Discussion',
                'description': 'Questions specific to the position'
            },
            'closing': {
                'name': 'Closing Discussion',
                'description': 'Final thoughts and next steps'
            }
        }
        
        self.transitions = [
            "Let's try a different approach...",
            "Here's another perspective to consider...",
            "Let me rephrase that...",
            "Let's explore this from another angle...",
            "Perhaps we can discuss this differently..."
        ]
        
        # Initialize authenticity statistics tracking
        self.authenticity_metrics = {
            'ai_scores': [],
            'plagiarism_scores': [],
            'flagged_responses': 0,
            'high_confidence_detections': 0,
            'detected_patterns': {}
        }

    def analyze_response_authenticity(self, response_text):
        """Analyze if the response is likely AI-generated or plagiarized with enhanced accuracy"""
        text_length = len(response_text) if response_text else 0
        if not response_text or text_length < 50:
            return {
                'ai_score': 0.0,
                'plagiarism_score': 0.0,
                'is_authentic': True,
                'warning': None,
                'text_length': text_length,
                'confidence': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Check for AI generation
        ai_score, ai_analysis = self.ai_detector.detect_ai_content(response_text)
        
        # Check for plagiarism patterns
        plagiarism_score, plagiarism_analysis = self.ai_detector.check_plagiarism(response_text)
        
        # Get confidence from the analysis
        confidence = ai_analysis.get('confidence', 0.7)
        
        # Determine if response might be inauthentic - adjusted for confidence
        is_authentic = (ai_score < 0.7 or confidence < 0.65) and plagiarism_score < 0.5
        
        # Create warning if applicable
        warning = None
        
        if ai_score >= 0.7 and confidence >= 0.65:
            source = ai_analysis.get('source', 'built-in')
            providers = ""
            if source == "eden_ai" and 'providers_used' in ai_analysis:
                providers = f" (detected by {', '.join(ai_analysis['providers_used'])})"
            warning = f"This response shows characteristics commonly found in AI-generated text{providers}."
            
            # Track patterns detected
            if 'details' in ai_analysis:
                for pattern, score in ai_analysis['details'].items():
                    if score > 0.5:  # Only track significant patterns
                        self.authenticity_metrics['detected_patterns'][pattern] = self.authenticity_metrics['detected_patterns'].get(pattern, 0) + 1
            
            # Track high confidence detections
            if confidence >= 0.8:
                self.authenticity_metrics['high_confidence_detections'] += 1
                
        elif plagiarism_score >= 0.5:
            warning = "This response contains patterns that might indicate copied content."
        
        # Update authenticity metrics
        self.authenticity_metrics['ai_scores'].append(ai_score)
        self.authenticity_metrics['plagiarism_scores'].append(plagiarism_score)
        if warning:
            self.authenticity_metrics['flagged_responses'] += 1
        
        # Create comprehensive result
        result = {
            'ai_score': ai_score,
            'plagiarism_score': plagiarism_score,
            'is_authentic': is_authentic,
            'warning': warning,
            'ai_details': ai_analysis,
            'plagiarism_details': plagiarism_analysis,
            'text_length': text_length,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        return result

    def should_skip_question(self, response_text):
        """Check if the response indicates we should skip to a different question"""
        return any(phrase in response_text.lower() for phrase in self.skip_phrases)

    def generate_interview_context(self, position):
        """Generate position-specific interview context"""
        try:
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": """As an expert interviewer, create a brief context for interviewing a candidate.
                    Include:
                    1. Key skills and competencies to evaluate
                    2. Technical knowledge requirements
                    3. Important behavioral traits
                    4. Industry-specific considerations
                    Keep it concise but comprehensive."""
                }, {
                    "role": "user",
                    "content": f"Create interview context for position: {position}"
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error generating context: {str(e)}"

    def speak_question(self, question):
        """Speak the interview question"""
        self.security_monitor.speak_text(question)

    def generate_alternative_question(self, position, stage, previous_question, previous_responses=None):
        """Generate an alternative question when the candidate asks to skip"""
        try:
            context = "No previous responses" if not previous_responses else \
                     f"Previous responses: {json.dumps(previous_responses[-3:])}"
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": f"""You are interviewing for a {position} position.
                    The candidate couldn't answer or wanted to skip this question: "{previous_question}"
                    Current stage: {stage}
                    Generate a different question that:
                    1. Is easier or approaches the topic differently
                    2. Is specific to the {position} role
                    3. Matches the current interview stage
                    4. Maintains the professional tone
                    5. Avoids similar concepts to the skipped question
                    
                    Return only the alternative question text."""
                }, {
                    "role": "user",
                    "content": f"Stage: {stage}\n{context}"
                }]
            )
            
            transition = random.choice(self.transitions)
            self.security_monitor.speak_text(transition)
            
            new_question = completion.choices[0].message.content
            threading.Thread(target=self.speak_question, args=(new_question,)).start()
            
            return new_question
        except Exception as e:
            fallback_question = self.generate_next_question(position, stage, previous_responses)
            return fallback_question

    def generate_next_question(self, position, stage, previous_responses=None):
        """Generate and speak the next interview question"""
        try:
            context = "No previous responses" if not previous_responses else \
                     f"Previous responses: {json.dumps(previous_responses[-3:])}"
            
            # Add authenticity analysis context if available
            authenticity_context = ""
            if previous_responses and len(previous_responses) > 0:
                ai_detected = sum(1 for r in previous_responses if 'authenticity' in r and r['authenticity'].get('ai_score', 0) > 0.7)
                if ai_detected > 0:
                    authenticity_context = f"\nNote: {ai_detected} of the candidate's previous responses have shown AI-like patterns."
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": f"""You are interviewing for a {position} position.
                    Current stage: {stage}
                    Generate a natural, conversational question that:
                    1. Is specific to the {position} role
                    2. Matches the current interview stage
                    3. Builds on previous responses when available
                    4. Feels natural and engaging
                    
                    Return only the question text.{authenticity_context}"""
                }, {
                    "role": "user",
                    "content": f"Stage: {stage}\n{context}"
                }]
            )
            
            question = completion.choices[0].message.content
            threading.Thread(target=self.speak_question, args=(question,)).start()
            return question
        except Exception as e:
            return "Could you tell me about your relevant experience?"

    def analyze_response(self, response_text, question, position):
        """Analyze interview response with enhanced AI detection"""
        if self.should_skip_question(response_text):
            return "Let's try a different question that might be more suitable."
        
        # Enhanced authenticity check with confidence rating
        authenticity = self.analyze_response_authenticity(response_text)
        
        try:
            # Add detailed information about AI detection to the analysis request
            system_prompt = f"""As an expert {position} interviewer, analyze this response.
            Provide feedback that:
            1. Highlights specific strengths
            2. Notes areas for improvement
            3. Suggests specific enhancements
            4. Relates to the {position} role
            
            AI Detection Results:
            - AI Generation Probability: {authenticity['ai_score']:.2f} (confidence: {authenticity['confidence']:.2f})
            - Plagiarism Probability: {authenticity['plagiarism_score']:.2f}
            
            If the AI detection score is high (>0.7) with good confidence (>0.65), note this concern in your feedback.
            If the plagiarism score is high (>0.5), mention this in your feedback.
            
            Keep feedback constructive and actionable."""
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": system_prompt
                }, {
                    "role": "user",
                    "content": f"Question: {question}\nResponse: {response_text}"
                }]
            )
            
            feedback = completion.choices[0].message.content
            
            # Add explicit warning about AI generation if needed - using confidence threshold
            if authenticity['warning'] and not authenticity['is_authentic']:
                confidence_level = ""
                if authenticity['confidence'] > 0.85:
                    confidence_level = " (HIGH CONFIDENCE)"
                elif authenticity['confidence'] > 0.7:
                    confidence_level = " (MEDIUM CONFIDENCE)"
                else:
                    confidence_level = " (LOW CONFIDENCE)"
                
                feedback = f"⚠️ **AUTHENTICITY CONCERN{confidence_level}:** {authenticity['warning']}\n\n{feedback}"
            
            return feedback
        except Exception as e:
            # If the API call fails, still show the authenticity warning
            if authenticity['warning'] and not authenticity['is_authentic']:
                return f"⚠️ **AUTHENTICITY CONCERN:** {authenticity['warning']}\n\nError analyzing response: {str(e)}"
            return f"Error analyzing response: {str(e)}"

    def monitor_interview_environment(self, frame):
        """Monitor interview environment for security violations"""
        aligned, face_message = self.security_monitor.check_face_alignment(frame)
        if not aligned:
            self.security_monitor.issue_warning(face_message)
            self.violation_history.append({
                'type': 'face_alignment',
                'message': face_message,
                'timestamp': datetime.now().isoformat()
            })
            return face_message, "face_alignment"
        
        object_warnings = self.security_monitor.detect_prohibited_objects(frame, self.yolo_model)
        if object_warnings:
            warning_message = "Warning! " + ", ".join(object_warnings)
            self.security_monitor.issue_warning(warning_message)
            self.violation_history.append({
                'type': 'prohibited_objects',
                'message': warning_message,
                'timestamp': datetime.now().isoformat()
            })
            return warning_message, "prohibited_objects"
        
        return None, None

    def calculate_interview_metrics(self, responses=None, violation_history=None):
        """Calculate detailed interview metrics with enhanced AI detection analysis"""
        if responses is None:
            responses = []
        if violation_history is None:
            violation_history = self.violation_history

        metrics = {
            'total_questions': len(responses),
            'questions_per_stage': {},
            'avg_response_length': 0,
            'response_quality_trend': [],
            'violations_by_type': {},
            'stage_completion': {},
            'technical_score': 0,
            'communication_score': 0,
            'violations_timeline': [],
            # Add new authenticity metrics
            'authenticity': {
                'avg_ai_score': 0,
                'avg_plagiarism_score': 0,
                'responses_flagged': 0,
                'high_confidence_detections': 0,
                'detection_confidence': 0,
                'top_ai_patterns': {},
                'ai_scores_by_stage': {}
            }
        }
        
        if responses:
            total_length = sum(len(r['response']) for r in responses)
            metrics['avg_response_length'] = total_length / len(responses)
            
            # Stage metrics
            for response in responses:
                stage = response['stage']
                metrics['questions_per_stage'][stage] = metrics['questions_per_stage'].get(stage, 0) + 1
            
            if self.response_metrics['response_times']:
                metrics['avg_response_time'] = sum(self.response_metrics['response_times']) / len(self.response_metrics['response_times'])
            
            technical_responses = [r for r in responses if r['stage'] == 'technical']
            if technical_responses:
                metrics['technical_score'] = len(technical_responses) / 3 * 100
            
            metrics['communication_score'] = (metrics['avg_response_length'] / 200) * 100
            
            # Enhanced authenticity metrics
            ai_scores = []
            plagiarism_scores = []
            confidence_scores = []
            ai_patterns = {}
            ai_scores_by_stage = {}
            
            for r in responses:
                if 'authenticity' in r:
                    stage = r['stage']
                    auth = r['authenticity']
                    ai_score = auth.get('ai_score', 0)
                    plagiarism_score = auth.get('plagiarism_score', 0)
                    confidence = auth.get('confidence', 0)
                    
                    # Collect scores
                    ai_scores.append(ai_score)
                    plagiarism_scores.append(plagiarism_score)
                    confidence_scores.append(confidence)
                    
                    # Track AI scores by stage
                    if stage not in ai_scores_by_stage:
                        ai_scores_by_stage[stage] = []
                    ai_scores_by_stage[stage].append(ai_score)
                    
                    # Collect AI detection patterns if available
                    if 'ai_details' in auth and isinstance(auth['ai_details'], dict) and 'details' in auth['ai_details']:
                        for pattern, score in auth['ai_details']['details'].items():
                            if score > 0.5:
                                ai_patterns[pattern] = ai_patterns.get(pattern, 0) + 1
            
            if ai_scores:
                # Calculate average scores
                metrics['authenticity']['avg_ai_score'] = sum(ai_scores) / len(ai_scores)
                metrics['authenticity']['avg_plagiarism_score'] = sum(plagiarism_scores) / len(plagiarism_scores)
                metrics['authenticity']['responses_flagged'] = sum(1 for score in ai_scores if score > 0.7)
                metrics['authenticity']['high_confidence_detections'] = sum(1 for s, c in zip(ai_scores, confidence_scores) if s > 0.7 and c > 0.75)
                
                # Calculate detection confidence
                if confidence_scores:
                    metrics['authenticity']['detection_confidence'] = sum(confidence_scores) / len(confidence_scores)
                
                # Process stage-specific AI scores
                metrics['authenticity']['ai_scores_by_stage'] = {
                    stage: sum(scores) / len(scores) for stage, scores in ai_scores_by_stage.items()
                }
                
                # Get top AI patterns
                if ai_patterns:
                    sorted_patterns = sorted(ai_patterns.items(), key=lambda x: x[1], reverse=True)
                    metrics['authenticity']['top_ai_patterns'] = dict(sorted_patterns[:5])  # Top 5 patterns
        
        # Process violation history
        if violation_history:
            for violation in violation_history:
                if isinstance(violation, dict):
                    violation_type = violation.get('type', 'unknown')
                    timestamp = violation.get('timestamp', datetime.now().isoformat())
                else:
                    violation_type = str(violation)
                    timestamp = datetime.now().isoformat()
                
                metrics['violations_by_type'][violation_type] = metrics['violations_by_type'].get(violation_type, 0) + 1
                
                metrics['violations_timeline'].append({
                    'type': violation_type,
                    'timestamp': timestamp
                })
        
        # Calculate stage completion percentage
        total_stages = len(self.stages)
        completed_stages = len(set(metrics['questions_per_stage'].keys()))
        metrics['stage_completion'] = (completed_stages / total_stages) * 100
        
        return metrics
        
    def generate_comprehensive_report(self, position, responses, warnings, metrics):
        """Generate a comprehensive report with enhanced AI detection analysis"""
        try:
            # Calculate advanced AI detection metrics
            detection_stats = self.ai_detector.get_detection_stats()
            
            # Calculate confidence-weighted AI scores
            weighted_ai_scores = []
            confidence_scores = []
            providers_used = set()
            authenticity_warnings = 0
            
            for response in responses:
                if 'authenticity' in response:
                    auth = response['authenticity']
                    ai_score = auth.get('ai_score', 0)
                    confidence = auth.get('confidence', 0.5)
                    weighted_ai_scores.append(ai_score * confidence)
                    confidence_scores.append(confidence)
                    
                    if not auth.get('is_authentic', True):
                        authenticity_warnings += 1
                    
                    # Collect providers used
                    if 'ai_details' in auth and isinstance(auth['ai_details'], dict) and 'providers_used' in auth['ai_details']:
                        providers_used.update(auth['ai_details']['providers_used'])
            
            # Calculate weighted average AI score
            avg_ai_score = sum(weighted_ai_scores) / sum(confidence_scores) if confidence_scores else 0
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            # Get Eden AI usage if applicable
            eden_used = len(providers_used) > 0
            eden_info = f"Using {len(providers_used)} AI detection engines: {', '.join(providers_used)}" if eden_used else "Using built-in AI detection"
            
            # Add response authenticity metrics to the report prompt
            authenticity_metrics = f"""
            Content Authenticity Metrics:
            - Average AI Detection Score: {avg_ai_score:.2f} (lower is better)
            - Detection Confidence: {avg_confidence:.2f}
            - AI Detection Method: {eden_info}
            - Responses with Authenticity Concerns: {authenticity_warnings} out of {len(responses)}
            - High Confidence AI Detections: {metrics['authenticity'].get('high_confidence_detections', 0)}
            """
            
            # Generate AI pattern information if available
            pattern_info = ""
            if 'top_ai_patterns' in metrics['authenticity'] and metrics['authenticity']['top_ai_patterns']:
                pattern_info = "Most Common AI Patterns Detected:\n"
                for pattern, count in metrics['authenticity']['top_ai_patterns'].items():
                    pattern_info += f"- {pattern.replace('_', ' ').title()}: {count} occurrences\n"
            
            report_prompt = f"""As an expert {position} interviewer, provide a detailed analysis of this interview session.
            
            Interview Statistics:
            - Total Questions: {metrics['total_questions']}
            - Average Response Length: {int(metrics.get('avg_response_length', 0))} characters
            - Technical Score: {int(metrics.get('technical_score', 0))}%
            - Communication Score: {int(metrics.get('communication_score', 0))}%
            - Total Security Violations: {len(warnings)}
            
            {authenticity_metrics}
            {pattern_info}
            
            Analyze the following aspects:
            1. Overall Performance
            2. Interview Integrity and Authenticity 
            3. Strengths and Weaknesses
            4. Position Fit Analysis
            5. Detailed Recommendations
            
            In the Authenticity section, carefully examine whether the candidate may be using AI tools to generate responses, based on the detection metrics.
            
            Provide a comprehensive, constructive analysis."""
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": report_prompt
                }, {
                    "role": "user",
                    "content": f"Interview Responses: {json.dumps(responses)}"
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error generating comprehensive report: {str(e)}"

    def generate_recommendations(self, position, responses):
        """Generate recommendations with authenticity considerations"""
        try:
            # Calculate authenticity metrics for recommendations
            authenticity_concerns = sum(1 for r in responses if 'authenticity' in r and not r['authenticity'].get('is_authentic', True))
            ai_concerns_percent = (authenticity_concerns / len(responses)) * 100 if responses else 0
            
            authenticity_context = ""
            if ai_concerns_percent > 20:
                authenticity_context = f"\nNote: {authenticity_concerns} responses ({ai_concerns_percent:.1f}%) showed potential AI-generation patterns. Address this in your recommendations."
            
            completion = openai.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": f"""As an expert {position} interviewer, provide specific recommendations.
                    Include:
                    1. Areas to focus on
                    2. Resources for improvement
                    3. Next steps
                    4. Interview preparation tips
                    5. Authenticity considerations and presentation skills
                    Make recommendations practical and actionable.{authenticity_context}"""
                }, {
                    "role": "user",
                    "content": f"Responses: {json.dumps(responses)}"
                }]
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error generating recommendations: {str(e)}"

    def generate_authenticity_report(self, responses):
        """Generate a dedicated report on response authenticity"""
        if not responses:
            return "No responses to analyze."
            
        try:
            # Collect authenticity data
            authenticity_data = []
            for i, response in enumerate(responses):
                if 'authenticity' in response:
                    auth = response['authenticity']
                    authenticity_data.append({
                        'question_num': i + 1,
                        'stage': response['stage'],
                        'ai_score': auth.get('ai_score', 0),
                        'plagiarism_score': auth.get('plagiarism_score', 0),
                        'confidence': auth.get('confidence', 0),
                        'is_authentic': auth.get('is_authentic', True),
                        'text_length': auth.get('text_length', len(response['response'])),
                        'warning': auth.get('warning', None)
                    })
            
            if not authenticity_data:
                return "No authenticity data available."
                
            # Calculate statistics
            avg_ai_score = sum(d['ai_score'] for d in authenticity_data) / len(authenticity_data)
            avg_confidence = sum(d['confidence'] for d in authenticity_data) / len(authenticity_data)
            flagged_responses = sum(1 for d in authenticity_data if not d['is_authentic'])
            high_confidence_flags = sum(1 for d in authenticity_data if d['ai_score'] > 0.7 and d['confidence'] > 0.8)
            
            # Compile report
            report = f"""## Response Authenticity Analysis

### Summary Statistics
- **Average AI Detection Score**: {avg_ai_score:.2f} (lower is better)
- **Detection Confidence**: {avg_confidence:.2f}
- **Flagged Responses**: {flagged_responses} out of {len(authenticity_data)} ({(flagged_responses/len(authenticity_data)*100):.1f}%)
- **High Confidence Detections**: {high_confidence_flags}

### Analysis by Interview Stage
"""
            
            # Group by stage
            stages = {}
            for d in authenticity_data:
                stage = d['stage']
                if stage not in stages:
                    stages[stage] = []
                stages[stage].append(d)
                
            # Report by stage
            for stage, data in stages.items():
                avg_stage_score = sum(d['ai_score'] for d in data) / len(data)
                flagged_in_stage = sum(1 for d in data if not d['is_authentic'])
                report += f"- **{stage.title()}**: Avg Score {avg_stage_score:.2f}, {flagged_in_stage} flagged responses\n"
            
            # Detailed flagged responses
            if flagged_responses > 0:
                report += "\n### Detailed Analysis of Flagged Responses\n"
                for d in authenticity_data:
                    if not d['is_authentic']:
                        report += f"- Question {d['question_num']} ({d['stage']}): AI Score {d['ai_score']:.2f}, Confidence {d['confidence']:.2f}\n"
                        if d['warning']:
                            report += f"  - Warning: {d['warning']}\n"
            
            # Overall assessment
            report += "\n### Overall Assessment\n"
            if avg_ai_score > 0.7 and flagged_responses > len(authenticity_data) * 0.3:
                report += """The candidate's responses show strong indicators of AI-generated content. The high AI detection scores and multiple flagged responses suggest potential use of AI assistance during the interview."""
            elif avg_ai_score > 0.5 and flagged_responses > 0:
                report += """Some of the candidate's responses show moderate indicators of AI-generated content. While not conclusive, there are patterns that warrant attention."""
            else:
                report += """The candidate's responses generally appear to be authentic. The AI detection scores are low, suggesting minimal or no use of AI assistance during the interview."""
                
            return report
        except Exception as e:
            return f"Error generating authenticity report: {str(e)}"

def generate_csv_report(responses):
    """Generate CSV report from interview responses with enhanced authenticity metrics"""
    csv_data = []
    for response in responses:
        data_row = {
            'Stage': response['stage'],
            'Question': response['question'],
            'Response': response['response'],
            'Feedback': response['feedback'],
            'Timestamp': response['timestamp']
        }
        
        # Add enhanced authenticity data
        if 'authenticity' in response:
            data_row['AI_Score'] = response['authenticity'].get('ai_score', 0)
            data_row['Plagiarism_Score'] = response['authenticity'].get('plagiarism_score', 0)
            data_row['Detection_Confidence'] = response['authenticity'].get('confidence', 0)
            data_row['Authenticity_Warning'] = response['authenticity'].get('warning', '')
            
            # Add Eden AI provider information if available
            if ('ai_details' in response['authenticity'] and 
                isinstance(response['authenticity']['ai_details'], dict) and
                'providers_used' in response['authenticity']['ai_details']):
                data_row['Detection_Providers'] = ','.join(response['authenticity']['ai_details']['providers_used'])
        
        csv_data.append(data_row)
    
    return pd.DataFrame(csv_data)

def save_csv_report(df, position):
    """Save interview report as CSV"""
    os.makedirs('interview_reports', exist_ok=True)
    filename = f"interview_report_{position}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = os.path.join('interview_reports', filename)
    df.to_csv(filepath, index=False)
    return filepath, filename

def generate_report(interviewer, position, responses, warnings, start_time):
    """Generate complete interview report with enhanced authenticity analysis"""
    metrics = interviewer.calculate_interview_metrics(responses, warnings)
    
    # Generate authenticity-specific report
    authenticity_report = interviewer.generate_authenticity_report(responses)
    
    # Generate comprehensive report with authenticity insights
    comprehensive_report = interviewer.generate_comprehensive_report(
        position,
        responses,
        warnings,
        metrics
    )
    
    # Get AI detection statistics
    detection_stats = interviewer.ai_detector.get_detection_stats()
    
    report_data = {
        'position': position,
        'timestamp': datetime.now().isoformat(),
        'duration': time.time() - start_time,
        'responses': responses,
        'warnings': warnings,
        'metrics': metrics,
        'summary': comprehensive_report,
        'recommendations': interviewer.generate_recommendations(
            position,
            responses
        ),
        'authenticity_report': authenticity_report,
        'detection_stats': detection_stats
    }
    
    return report_data

def save_report(report_data):
    """Save report to file and return filepath"""
    os.makedirs('interview_reports', exist_ok=True)
    filename = f"interview_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join('interview_reports', filename)
    
    # Convert report_data to be JSON serializable
    report_data_serializable = convert_to_serializable(report_data)
    
    with open(filepath, 'w') as f:
        json.dump(report_data_serializable, f, indent=2)
    
    return filepath, filename

def convert_to_serializable(obj):
    """Convert object to be JSON serializable, handling NumPy types"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'isoformat'):  
        return obj.isoformat()
    else:
        return obj

def cleanup_resources():
    """Clean up system resources"""
    try:
        mixer.music.stop()
        mixer.quit()
    except:
        pass
    
    if 'cap' in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
    
    if 'running' in st.session_state:
        st.session_state.running = False

@st.cache_resource
def get_ai_detector():
    """Get cached AI detector instance"""
    return AITextDetector()

def initialize_session_state():
    """Initialize or reset session state variables with optimized defaults"""
    defaults = {
        'running': True,
        'responses': [],
        'current_question': None,
        'interview_stage': 'introduction',
        'warnings': [],
        'position': "",
        'interview_context': "",
        'start_time': time.time(),
        'violation_history': [],
        'report_data': None,
        'cap': None,
        'ai_detection_enabled': True,
        'eden_api_key': "",
        'eden_providers': "sapling,writer,originalityai",
        'performance_metrics': {
            'api_calls': 0,
            'cache_hits': 0,
            'errors': 0
        }
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def main():
    """Main application function with optimized performance and error handling"""
    # Initialize session state first
    initialize_session_state()

    # Sidebar configuration with improved UX
    with st.sidebar:
        st.title("🎤 Interview Setup")

        # Get OpenAI API key from secrets only
        try:
            openai_key = st.secrets["openai"]["api_key"]
        except KeyError:
            st.error("⚠️ OpenAI API key not configured")
            st.info("💡 Please configure your OpenAI API key in Streamlit secrets")
            st.code("""
# Add this to your Streamlit secrets:
[openai]
api_key = "your-openai-api-key-here"
            """)
            st.stop()
        except Exception as e:
            st.error(f"Error accessing secrets: {str(e)}")
            st.stop()

        st.divider()

        position = st.text_input(
            "Position you're interviewing for",
            help="Enter the job title you're applying for",
            placeholder="e.g., Software Engineer"
        )
        
        if position and position != st.session_state.position:
            st.session_state.position = position
            st.session_state.interviewer = AIInterviewer(openai_key)
            st.session_state.interview_context = st.session_state.interviewer.generate_interview_context(position)
            st.session_state.responses = []
            st.session_state.interview_stage = 'introduction'
            st.session_state.current_question = None
            st.session_state.start_time = time.time()
            st.session_state.violation_history = []
            st.session_state.report_data = None
        
        st.divider()

        st.subheader("Interview Settings")
        camera_enabled = st.toggle("Enable Camera", value=True)
        audio_enabled = st.toggle("Enable Microphone", value=True)
        
        # Enhanced AI detection settings
        ai_detection_enabled = st.toggle("Enable AI Response Detection", value=True,
                                       help="Detect if responses might be AI-generated or plagiarized")

        # Eden AI API integration with enhanced UI
        with st.expander("AI Detection Settings"):
            st.markdown("### Eden AI Integration")
            st.info("Eden AI combines multiple professional AI detection engines for more accurate results.")

            # Get Eden AI key from secrets if available
            try:
                eden_api_key = st.secrets.get("eden_ai", {}).get("api_key", "")
                if eden_api_key:
                    st.success("✅ Eden AI configured via secrets")
                else:
                    st.warning("⚠️ Eden AI not configured - using built-in detection")
            except:
                eden_api_key = ""
                st.warning("⚠️ Eden AI not configured - using built-in detection")
            
            # Which providers to use with better layout
            if eden_api_key:
                st.write("**Select AI detection providers:**")
                provider_cols = st.columns(3)
                with provider_cols[0]:
                    use_sapling = st.checkbox("Sapling", value=True, 
                                            help="Academic text detector with high precision")
                with provider_cols[1]:
                    use_writer = st.checkbox("Writer", value=True,
                                           help="Good for detecting ChatGPT/GPT outputs")
                with provider_cols[2]:
                    use_originalityai = st.checkbox("Originality.ai", value=True,
                                                  help="Best for long-form content analysis")
                
                # Detection threshold settings
                st.write("**Detection Settings:**")
                threshold_cols = st.columns(2)
                with threshold_cols[0]:
                    ai_threshold = st.slider("AI Score Threshold", 0.5, 0.9, 0.7, 0.05,
                                           help="Score above which content is flagged as AI-generated")
                with threshold_cols[1]:
                    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.9, 0.65, 0.05,
                                                   help="Minimum confidence level for reliable detection")
                
                # Display detection examples
                st.markdown("#### What This Detects:")
                st.markdown("""
                - **AI-Generated Text**: Responses created by ChatGPT, Claude, or other LLMs
                - **Plagiarism Patterns**: Indicators of copied content
                - **Mixed Authorship**: Content that combines human and AI writing
                """)
                
                # Build provider string
                providers = []
                if use_sapling:
                    providers.append("sapling")
                if use_writer:
                    providers.append("writer")
                if use_originalityai:
                    providers.append("originalityai")
                
                provider_string = ",".join(providers)
                
                # Store settings in session state
                st.session_state.eden_providers = provider_string
                
                # Store thresholds if we have an interviewer
                if hasattr(st.session_state, 'interviewer') and hasattr(st.session_state.interviewer, 'ai_detector'):
                    st.session_state.interviewer.ai_detector.confidence_thresholds['ai_score'] = ai_threshold
            else:
                st.warning("Without Eden AI integration, the system will use basic built-in detection which is less accurate.")
            
            # Store the setting in session state
            # Store the setting in session state
            st.session_state.eden_api_key = eden_api_key
            if eden_api_key and hasattr(st.session_state, 'interviewer'):
                st.session_state.interviewer.ai_detector.set_api_key(eden_api_key)
        
        # Store the AI detection settings in session state
        st.session_state.ai_detection_enabled = ai_detection_enabled

    st.title(f"AI Interview System{f' - {position}' if position else ''}")
    
    if not st.session_state.position:
        st.info("👋 Welcome! Please enter the position you're interviewing for to begin.")
        return

    # Main content layout
    col1, col2 = st.columns([2, 3])
    
    # Interview control column
    with col1:
        st.subheader("Interview Progress")

        # End Interview button
        if st.button("🛑 End Interview Early", type="primary", help="Click to end the interview and generate report"):
            if st.session_state.responses:
                cleanup_resources()
                
                # Generate CSV report
                df = generate_csv_report(st.session_state.responses)
                csv_filepath, csv_filename = save_csv_report(df, st.session_state.position)
                
                # Generate enhanced report with authenticity analysis
                report_data = generate_report(
                    st.session_state.interviewer,
                    st.session_state.position,
                    st.session_state.responses,
                    st.session_state.warnings,
                    st.session_state.start_time
                )
                
                # Save reports
                json_filepath, json_filename = save_report(report_data)
                
                # Store report information
                st.session_state.report_data = report_data
                st.session_state.csv_filepath = csv_filepath
                st.session_state.csv_filename = csv_filename
                st.session_state.json_filepath = json_filepath
                st.session_state.json_filename = json_filename
                
                st.session_state.interview_stage = 'closing'
                st.rerun()
            else:
                st.error("Please provide at least one response before ending the interview.")
        
        # Display current stage information
        current_stage = st.session_state.interviewer.stages[st.session_state.interview_stage]
        st.write(f"**Current Stage:** {current_stage['name']}")
        st.write(current_stage['description'])
        
        # Progress bar
        progress = list(st.session_state.interviewer.stages.keys()).index(st.session_state.interview_stage) / \
                  len(st.session_state.interviewer.stages)
        st.progress(progress)
        
        # Timer
        elapsed_time = time.time() - st.session_state.start_time
        st.write(f"⏱️ Interview Duration: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
        
        # Generate question if needed
        if st.session_state.current_question is None:
            st.session_state.current_question = st.session_state.interviewer.generate_next_question(
                st.session_state.position,
                st.session_state.interview_stage,
                st.session_state.responses
            )
        
        # Display current question
        st.write("### Current Question:")
        st.write(st.session_state.current_question)
        
        # Response interface
        response_tab, feedback_tab = st.tabs(["Your Response", "Previous Feedback"])
        
        with response_tab:
            # Audio recording interface
            if audio_enabled:
                if st.button("🎤 Record Response", key="record"):
                    with st.spinner("🎙️ Recording... (Speaking now)"):
                        with sr.Microphone() as source:
                            try:
                                audio = st.session_state.interviewer.recognizer.listen(source, timeout=15)
                                response_text = st.session_state.interviewer.recognizer.recognize_google(audio)
                                st.success("✅ Response recorded!")
                                
                                # Check if we should skip this question
                                if st.session_state.interviewer.should_skip_question(response_text):
                                    st.info("Generating an alternative question...")
                                    st.session_state.current_question = st.session_state.interviewer.generate_alternative_question(
                                        st.session_state.position,
                                        st.session_state.interview_stage,
                                        st.session_state.current_question,
                                        st.session_state.responses
                                    )
                                    st.rerun()
                                else:
                                    with st.spinner("💭 Analyzing your response..."):
                                        feedback = st.session_state.interviewer.analyze_response(
                                            response_text,
                                            st.session_state.current_question,
                                            st.session_state.position
                                        )
                                    
                                    # Get enhanced authenticity analysis if enabled
                                    authenticity = None
                                    if st.session_state.ai_detection_enabled:
                                        authenticity = st.session_state.interviewer.analyze_response_authenticity(response_text)
                                    
                                    # Save response with enhanced authenticity data if available
                                    response_data = {
                                        'stage': st.session_state.interview_stage,
                                        'question': st.session_state.current_question,
                                        'response': response_text,
                                        'feedback': feedback,
                                        'timestamp': datetime.now().isoformat()
                                    }
                                    
                                    if authenticity:
                                        response_data['authenticity'] = authenticity
                                    
                                    st.session_state.responses.append(response_data)
                                    
                                    st.write("### Feedback:")
                                    st.write(feedback)
                                    
                                    # Display authenticity warning with confidence level if applicable
                                    if authenticity and authenticity['warning']:
                                        confidence = authenticity['confidence']
                                        confidence_msg = ""
                                        if confidence > 0.85:
                                            confidence_msg = " (High confidence)"
                                        elif confidence > 0.7:
                                            confidence_msg = " (Medium confidence)"
                                        else:
                                            confidence_msg = " (Low confidence)"
                                        
                                        st.warning(f"{authenticity['warning']}{confidence_msg}")
                                
                            except sr.WaitTimeoutError:
                                st.error("⚠️ No speech detected. Please try again.")
                            except Exception as e:
                                st.error(f"⚠️ Error: {str(e)}")
            
            # Text response interface
            response_text = st.text_area("Or type your response:", key="text_response")
            if st.button("Submit Response", key="submit"):
                if response_text:
                    # Check if we should skip this question
                    if st.session_state.interviewer.should_skip_question(response_text):
                        st.info("Generating an alternative question...")
                        st.session_state.current_question = st.session_state.interviewer.generate_alternative_question(
                            st.session_state.position,
                            st.session_state.interview_stage,
                            st.session_state.current_question,
                            st.session_state.responses
                        )
                        st.rerun()
                    else:
                        with st.spinner("💭 Analyzing your response..."):
                            feedback = st.session_state.interviewer.analyze_response(
                                response_text,
                                st.session_state.current_question,
                                st.session_state.position
                            )
                        
                        # Get enhanced authenticity analysis if enabled
                        authenticity = None
                        if st.session_state.ai_detection_enabled:
                            authenticity = st.session_state.interviewer.analyze_response_authenticity(response_text)
                        
                        # Save response with enhanced authenticity data if available
                        response_data = {
                            'stage': st.session_state.interview_stage,
                            'question': st.session_state.current_question,
                            'response': response_text,
                            'feedback': feedback,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        if authenticity:
                            response_data['authenticity'] = authenticity
                        
                        st.session_state.responses.append(response_data)
                        
                        st.write("### Feedback:")
                        st.write(feedback)
                        
                        # Display enhanced AI detection results
                        if st.session_state.ai_detection_enabled and authenticity:
                            # Display in a nicer expander
                            with st.expander("Response Authenticity Analysis"):
                                # Create columns for the metrics
                                cols = st.columns(2)
                                
                                # Left column - AI Score
                                with cols[0]:
                                    ai_score = authenticity['ai_score']
                                    ai_color = "red" if ai_score > 0.7 else "orange" if ai_score > 0.5 else "green"
                                    confidence = authenticity.get('confidence', 0.5)
                                    conf_text = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                                    
                                    st.markdown(f"**AI Generated Score:** <span style='color:{ai_color}'>{ai_score:.2f}</span> (Confidence: {conf_text})", unsafe_allow_html=True)
                                    
                                    # Improved gauge chart for AI score
                                    fig = plt.figure(figsize=(3, 2))
                                    ax = fig.add_subplot(111)
                                    ax.set_xlim(0, 1)
                                    ax.set_ylim(0, 0.5)
                                    ax.barh([0.25], [ai_score], color=ai_color, height=0.2)
                                    ax.barh([0.25], [1], color='lightgrey', height=0.2, alpha=0.3)
                                    ax.set_yticks([])
                                    ax.set_xticks([0, 0.5, 1])
                                    ax.set_xticklabels(['Human', '|', 'AI'])
                                    # Add confidence indicator
                                    ax.text(ai_score, 0.35, f"{conf_text}", fontsize=8, ha='center')
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Right column - Plagiarism Score
                                with cols[1]:
                                    plag_score = authenticity['plagiarism_score']
                                    plag_color = "red" if plag_score > 0.5 else "orange" if plag_score > 0.3 else "green"
                                    st.markdown(f"**Plagiarism Score:** <span style='color:{plag_color}'>{plag_score:.2f}</span>", unsafe_allow_html=True)
                                    
                                    # Improved gauge chart for plagiarism score
                                    fig = plt.figure(figsize=(3, 2))
                                    ax = fig.add_subplot(111)
                                    ax.set_xlim(0, 1)
                                    ax.set_ylim(0, 0.5)
                                    ax.barh([0.25], [plag_score], color=plag_color, height=0.2)
                                    ax.barh([0.25], [1], color='lightgrey', height=0.2, alpha=0.3)
                                    ax.set_yticks([])
                                    ax.set_xticks([0, 0.5, 1])
                                    ax.set_xticklabels(['Original', '|', 'Copied'])
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Eden AI provider information with improved display
                                if ('ai_details' in authenticity and 
                                    isinstance(authenticity['ai_details'], dict) and 
                                    'source' in authenticity['ai_details'] and
                                    authenticity['ai_details']['source'] == 'eden_ai' and
                                    'providers_used' in authenticity['ai_details']):
                                    st.write("#### Detection Method")
                                    providers = authenticity['ai_details']['providers_used']
                                    st.info(f"Using {len(providers)} professional AI detection engines: {', '.join(providers)}")
                                    
                                    # Display individual provider scores with better visualization
                                    if 'details' in authenticity['ai_details']:
                                        st.write("**Provider Scores:**")
                                        provider_data = []
                                        for provider, data in authenticity['ai_details']['details'].items():
                                            if isinstance(data, dict) and 'ai_score' in data:
                                                provider_data.append({
                                                    "provider": provider.title(),
                                                    "score": data['ai_score']
                                                })
                                        
                                        # Create a bar chart for provider scores
                                        if provider_data:
                                            provider_df = pd.DataFrame(provider_data)
                                            fig, ax = plt.subplots(figsize=(6, 3))
                                            bars = ax.barh(provider_df['provider'], provider_df['score'], color=['red' if s > 0.7 else 'orange' if s > 0.5 else 'green' for s in provider_df['score']])
                                            ax.set_xlim(0, 1)
                                            ax.set_xlabel('AI Score')
                                            ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7)
                                            
                                            # Add text labels
                                            for i, bar in enumerate(bars):
                                                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{provider_df['score'].iloc[i]:.2f}", va='center')
                                            
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                else:
                                    st.write("#### Detection Method")
                                    st.info("Using built-in AI pattern detection")
                                    
                                    # Show regular analysis details with better visualization
                                    if 'ai_details' in authenticity and isinstance(authenticity['ai_details'], dict) and 'details' in authenticity['ai_details']:
                                        st.write("**AI Patterns Detected:**")
                                        ai_details = authenticity['ai_details']['details']
                                        
                                        # Create a sorted list of patterns by score
                                        patterns = [(k.replace('_', ' ').title(), v) for k, v in ai_details.items()]
                                        patterns.sort(key=lambda x: x[1], reverse=True)
                                        
                                        # Display as a horizontal bar chart
                                        if patterns:
                                            pattern_df = pd.DataFrame(patterns, columns=['Pattern', 'Score'])
                                            fig, ax = plt.subplots(figsize=(6, max(3, len(patterns) * 0.4)))
                                            bars = ax.barh(pattern_df['Pattern'], pattern_df['Score'], 
                                                    color=['red' if s > 0.7 else 'orange' if s > 0.5 else 'blue' for s in pattern_df['Score']])
                                            ax.set_xlim(0, 1)
                                            ax.set_xlabel('Pattern Strength')
                                            plt.tight_layout()
                                            st.pyplot(fig)
                
                                # Add an explanation section
                                st.write("#### What This Means")
                                ai_score = authenticity['ai_score']
                                confidence = authenticity.get('confidence', 0.5)
                                
                                if ai_score > 0.7 and confidence > 0.7:
                                    st.warning("""
                                    This response shows **strong indicators** of AI-generated content. The high score with good confidence suggests this may be generated by tools like ChatGPT, Claude, or other AI assistants.
                                    """)
                                elif ai_score > 0.5:
                                    st.info("""
                                    This response shows **some characteristics** of AI-generated content, though not conclusive. It may contain portions of AI-assisted writing or exhibit patterns common in AI outputs.
                                    """)
                                else:
                                    st.success("""
                                    This response appears to be **human-written** with low probability of being AI-generated.
                                    """)
                                
                                if authenticity['plagiarism_score'] > 0.5:
                                    st.warning("The response shows patterns that might indicate copied content.")
                else:
                    st.warning("Please provide a response.")
        
        # Previous feedback tab
        with feedback_tab:
            if st.session_state.responses:
                for i, response in enumerate(reversed(st.session_state.responses[-3:]), 1):
                    with st.expander(f"Previous Response {i}"):
                        st.write("**Question:**", response['question'])
                        st.write("**Your Response:**", response['response'])
                        st.write("**Feedback:**", response['feedback'])
                        
                        # Show enhanced authenticity data if available
                        if 'authenticity' in response and st.session_state.ai_detection_enabled:
                            auth = response['authenticity']
                            st.write("**Authenticity:**")
                            ai_color = "red" if auth['ai_score'] > 0.7 else "orange" if auth['ai_score'] > 0.5 else "green"
                            plag_color = "red" if auth['plagiarism_score'] > 0.5 else "orange" if auth['plagiarism_score'] > 0.3 else "green"
                            
                            st.markdown(f"- AI Score: <span style='color:{ai_color}'>{auth['ai_score']:.2f}</span> (Confidence: {auth.get('confidence', 0.5):.2f})", unsafe_allow_html=True)
                            st.markdown(f"- Plagiarism Score: <span style='color:{plag_color}'>{auth['plagiarism_score']:.2f}</span>", unsafe_allow_html=True)
                            
                            if auth['warning']:
                                st.warning(auth['warning'])
        
        # Continue button
        if st.button("Continue ➡️", key="next"):
            stages = list(st.session_state.interviewer.stages.keys())
            current_stage_index = stages.index(st.session_state.interview_stage)
            
            if len(st.session_state.responses) % 3 == 0 and current_stage_index < len(stages) - 1:
                st.session_state.interview_stage = stages[current_stage_index + 1]
                transition = random.choice(st.session_state.interviewer.transitions)
                st.write(transition)
                # Speak the transition
                threading.Thread(
                    target=st.session_state.interviewer.security_monitor.speak_text,
                    args=(transition,)
                ).start()
            
            st.session_state.current_question = None
            st.rerun()

    # Camera feed and monitoring column
    with col2:
        if camera_enabled:
            st.subheader("Interview Environment")
            video_feed = st.empty()
            warning_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            if st.session_state.cap is None:
                st.session_state.cap = cv2.VideoCapture(0)
            
            try:
                while st.session_state.running and st.session_state.cap is not None:
                    ret, frame = st.session_state.cap.read()
                    if ret:
                        warning, warning_type = st.session_state.interviewer.monitor_interview_environment(frame)
                        
                        # Draw alignment guides
                        frame_height, frame_width = frame.shape[:2]
                        center_x = frame_width // 2
                        
                        cv2.line(frame, (center_x, 0), (center_x, frame_height), (0, 255, 0), 1)
                        
                        box_width = int(frame_width * 0.4)
                        box_height = int(frame_height * 0.6)
                        box_x = center_x - box_width // 2
                        box_y = frame_height // 4
                        
                        cv2.rectangle(frame, 
                                    (box_x, box_y),
                                    (box_x + box_width, box_y + box_height),
                                    (0, 255, 0), 1)
                        
                        if warning:
                            warning_placeholder.warning(warning)
                            if warning_type == "prohibited_objects":
                                frame = cv2.addWeighted(
                                    frame,
                                    1,
                                    np.full(frame.shape, [0, 0, 50], dtype=np.uint8),
                                    0.3,
                                    0
                                )
                            if warning not in st.session_state.warnings:
                                st.session_state.warnings.append(warning)
                        else:
                            warning_placeholder.empty()
                        
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_feed.image(frame)
                        
                        # Enhanced metrics display with authenticity metrics
                        if len(st.session_state.responses) > 0:
                            with metrics_placeholder:
                                cols = st.columns(4)  # Added one more column for AI detection
                                with cols[0]:
                                    st.metric("Questions Answered", len(st.session_state.responses))
                                with cols[1]:
                                    avg_response_length = np.mean([len(r['response']) for r in st.session_state.responses])
                                    st.metric("Avg Response Length", f"{int(avg_response_length)} chars")
                                with cols[2]:
                                    warnings_count = len(st.session_state.warnings)
                                    st.metric("Security Violations", warnings_count)
                                
                                # Enhanced AI detection metrics if enabled
                                if st.session_state.ai_detection_enabled:
                                    with cols[3]:
                                        # Calculate average AI score with confidence weighting
                                        ai_scores = []
                                        confidence_scores = []
                                        for r in st.session_state.responses:
                                            if 'authenticity' in r:
                                                ai_scores.append(r['authenticity'].get('ai_score', 0))
                                                confidence_scores.append(r['authenticity'].get('confidence', 0.5))
                                        
                                        if ai_scores:
                                            # Weight by confidence
                                            weighted_sum = sum(s * c for s, c in zip(ai_scores, confidence_scores))
                                            avg_ai_score = weighted_sum / sum(confidence_scores) if sum(confidence_scores) > 0 else 0
                                            
                                            # Display with color coding
                                            color = "normal"
                                            if avg_ai_score > 0.7:
                                                color = "off"
                                            elif avg_ai_score > 0.5:
                                                color = "inverse"
                                                
                                            st.metric("Avg AI Score", f"{avg_ai_score:.2f}", delta_color=color)
                        
                        time.sleep(0.1)
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
                cleanup_resources()
        
        # Closing stage handling
        if st.session_state.interview_stage == 'closing':
            st.success("🎉 Interview Complete!")
            
            # Download buttons for reports
            if hasattr(st.session_state, 'csv_filepath') and os.path.exists(st.session_state.csv_filepath):
                with open(st.session_state.csv_filepath, 'r') as f:
                    st.download_button(
                        label="📥 Download Interview Report (CSV)",
                        data=f.read(),
                        file_name=st.session_state.csv_filename,
                        mime="text/csv",
                        key="csv_download"
                    )
            
            if hasattr(st.session_state, 'json_filepath') and os.path.exists(st.session_state.json_filepath):
                with open(st.session_state.json_filepath, 'r') as f:
                    st.download_button(
                        label="📥 Download Detailed Report (JSON)",
                        data=f.read(),
                        file_name=st.session_state.json_filename,
                        mime="application/json",
                        key="json_download"
                    )
            
            # Enhanced report tabs - added dedicated Authenticity tab
            if st.session_state.report_data:
                tabs = st.tabs(["Overview", "Performance Analysis", "Security & Compliance", 
                              "Technical Evaluation", "Recommendations", "Response Authenticity"])
                
                with tabs[0]:
                    st.write("#### Interview Overview")
                    metrics = st.session_state.report_data['metrics']
                    
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Total Questions", metrics['total_questions'])
                    with metrics_cols[1]:
                        st.metric("Security Violations", len(st.session_state.warnings))
                    with metrics_cols[2]:
                        st.metric("Avg Response Length", f"{int(metrics['avg_response_length'])} chars")
                    with metrics_cols[3]:
                        # Add authenticity metric to overview
                        if 'authenticity' in metrics and 'avg_ai_score' in metrics['authenticity']:
                            ai_score = metrics['authenticity']['avg_ai_score']
                            st.metric("Authenticity Score", f"{(1-ai_score)*100:.0f}%", 
                                    delta=f"{-ai_score*100:.0f}% AI indicators", 
                                    delta_color="normal" if ai_score < 0.3 else "off")
                        else:
                            st.metric("Technical Score", f"{int(metrics.get('technical_score', 0))}%")
                    
                    st.write(st.session_state.report_data['summary'])
                
                with tabs[1]:
                    st.write("#### Performance Analysis")
                    
                    # Create enhanced performance visualization
                    fig = plt.figure(figsize=(12, 8))
                    gs = fig.add_gridspec(2, 2)
                    
                    # Response length trend
                    response_lengths = [len(r['response']) for r in st.session_state.responses]
                    ax1 = fig.add_subplot(gs[0, :])
                    ax1.plot(response_lengths, marker='o')
                    ax1.set_title("Response Length Trend")
                    ax1.set_xlabel("Question Number")
                    ax1.set_ylabel("Response Length (characters)")
                    
                    # Responses by stage
                    stage_counts = pd.Series([r['stage'] for r in st.session_state.responses]).value_counts()
                    ax2 = fig.add_subplot(gs[1, 0])
                    stage_counts.plot(kind='pie', ax=ax2, title="Responses by Stage")
                    ax2.set_ylabel('')
                    
                    # Security violations
                    if st.session_state.warnings:
                        ax3 = fig.add_subplot(gs[1, 1])
                        violation_counts = pd.Series(st.session_state.warnings).value_counts()
                        violation_counts.plot(kind='bar', ax=ax3, title="Security Violations")
                        ax3.tick_params(axis='x', rotation=45)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Performance summary
                    st.write("#### Numerical Summary")
                    summary_cols = st.columns(3)
                    with summary_cols[0]:
                        st.metric("Average Response Length", 
                                f"{int(np.mean(response_lengths))} chars",
                                f"{int(np.std(response_lengths))} σ")
                    with summary_cols[1]:
                        st.metric("Total Responses by Stage", 
                                len(stage_counts),
                                f"{stage_counts.index[0]} (most frequent)")
                    with summary_cols[2]:
                        if st.session_state.warnings:
                            violation_counts = pd.Series(st.session_state.warnings).value_counts()
                            st.metric("Total Violations",
                                    len(st.session_state.warnings),
                                    f"{violation_counts.index[0]} (most common)")
                        else:
                            st.metric("Total Violations", "0", "No violations detected")
                
                with tabs[2]:
                    st.write("#### Security and Compliance Report")
                    if st.session_state.warnings:
                        st.error(f"Total Security Violations: {len(st.session_state.warnings)}")
                        violation_df = pd.DataFrame(st.session_state.warnings, columns=['Violation'])
                        violation_counts = violation_df['Violation'].value_counts()
                        st.bar_chart(violation_counts)
                        
                        for i, warning in enumerate(st.session_state.warnings, 1):
                            st.warning(f"Violation {i}: {warning}")
                    else:
                        st.success("No security violations detected during the interview")
                
                with tabs[3]:
                    st.write("#### Technical Competency Analysis")
                    technical_responses = [r for r in st.session_state.responses if r['stage'] == 'technical']
                    if technical_responses:
                        for resp in technical_responses:
                            with st.expander(f"Technical Question: {resp['question'][:100]}..."):
                                st.write("**Response:**", resp['response'])
                                st.write("**Evaluation:**", resp['feedback'])
                                
                        # Technical score visualization
                        technical_score = metrics.get('technical_score', 0)
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.barh(['Technical Score'], [technical_score], color='blue')
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Score (%)')
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info("No technical questions were asked during this interview session.")
                
                with tabs[4]:
                    st.write("#### Recommendations and Next Steps")
                    st.write(st.session_state.report_data['recommendations'])
                    
                    # Areas for improvement
                    st.write("##### Areas for Improvement:")
                    improvement_points = []
                    
                    if metrics['violations_by_type']:
                        improvement_points.append("- Work on interview environment setup and compliance")
                    if metrics['avg_response_length'] < 100:
                        improvement_points.append("- Provide more detailed and comprehensive responses")
                    if metrics.get('technical_score', 0) < 70:
                        improvement_points.append("- Focus on strengthening technical knowledge")
                    if metrics.get('communication_score', 0) < 70:
                        improvement_points.append("- Enhance communication clarity and depth")
                    
                    # Add authenticity-specific improvement points
                    if 'authenticity' in metrics and metrics['authenticity'].get('avg_ai_score', 0) > 0.5:
                        improvement_points.append("- Work on providing more original and authentic responses")
                        if metrics['authenticity'].get('avg_ai_score', 0) > 0.7:
                            improvement_points.append("- Avoid using AI assistance during interviews - demonstrate your own knowledge")
                    
                    for point in improvement_points:
                        st.write(point)
                    
                    # Enhanced skills radar chart with authenticity dimension
                    skills_data = {
                        'Technical': metrics.get('technical_score', 0) / 100,
                        'Communication': metrics.get('communication_score', 0) / 100,
                        'Environment': 1 - (len(st.session_state.warnings) / max(len(st.session_state.responses), 1)),
                        'Response Quality': min(metrics['avg_response_length'] / 200, 1),
                        'Stage Progress': progress
                    }
                    
                    # Add authenticity to skills radar if available
                    if 'authenticity' in metrics and 'avg_ai_score' in metrics['authenticity']:
                        skills_data['Authenticity'] = 1 - metrics['authenticity']['avg_ai_score']
                    
                    fig = plt.figure(figsize=(8, 8))
                    ax = fig.add_subplot(111, projection='polar')
                    
                    angles = np.linspace(0, 2*np.pi, len(skills_data), endpoint=False)
                    values = list(skills_data.values())
                    values.append(values[0])
                    angles = np.concatenate((angles, [angles[0]]))
                    
                    ax.plot(angles, values)
                    ax.fill(angles, values, alpha=0.25)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(skills_data.keys())
                    ax.set_ylim(0, 1)
                    plt.title("Skills Assessment")
                    st.pyplot(fig)
                
                # Enhanced Response Authenticity tab
                with tabs[5]:
                    st.write("#### Response Authenticity Analysis")
                    
                    # Display the dedicated authenticity report
                    if 'authenticity_report' in st.session_state.report_data:
                        st.markdown(st.session_state.report_data['authenticity_report'])
                    
                    # Calculate authenticity metrics
                    ai_scores = []
                    confidence_scores = []
                    plagiarism_scores = []
                    responses_with_warning = []
                    
                    for i, response in enumerate(st.session_state.responses):
                        if 'authenticity' in response:
                            auth = response['authenticity']
                            ai_scores.append(auth.get('ai_score', 0))
                            confidence_scores.append(auth.get('confidence', 0.5))
                            plagiarism_scores.append(auth.get('plagiarism_score', 0))
                            
                            if auth.get('warning'):
                                responses_with_warning.append(i)
                    
                    if ai_scores:
                        # Display enhanced summary metrics
                        metrics_cols = st.columns(3)
                        with metrics_cols[0]:
                            # Calculate confidence-weighted AI score
                            weighted_ai = sum(s * c for s, c in zip(ai_scores, confidence_scores)) / sum(confidence_scores) if confidence_scores else 0
                            
                            # Color-code based on score
                            color = "green" if weighted_ai < 0.3 else "orange" if weighted_ai < 0.7 else "red"
                            
                            st.markdown(f"#### AI Generated Content Score: <span style='color:{color}'>{weighted_ai:.2f}</span>", unsafe_allow_html=True)
                            st.markdown(f"*Average confidence: {sum(confidence_scores)/len(confidence_scores):.2f}*")
                        
                        with metrics_cols[1]:
                            avg_plag = sum(plagiarism_scores) / len(plagiarism_scores) if plagiarism_scores else 0
                            plag_color = "green" if avg_plag < 0.3 else "orange" if avg_plag < 0.5 else "red"
                            
                            st.markdown(f"#### Plagiarism Score: <span style='color:{plag_color}'>{avg_plag:.2f}</span>", unsafe_allow_html=True)
                            st.markdown(f"*Lower is better*")
                        
                        with metrics_cols[2]:
                            percent_flagged = len(responses_with_warning)/len(st.session_state.responses)*100
                            flag_color = "green" if percent_flagged < 20 else "orange" if percent_flagged < 50 else "red"
                            
                            st.markdown(f"#### Responses Flagged: <span style='color:{flag_color}'>{len(responses_with_warning)} ({percent_flagged:.1f}%)</span>", unsafe_allow_html=True)
                            st.markdown(f"*Out of {len(st.session_state.responses)} total*")
                        
                        # Display Eden AI usage information
                        eden_used = False
                        providers_used = set()
                        
                        for r in st.session_state.responses:
                            if ('authenticity' in r and 'ai_details' in r['authenticity'] and 
                                isinstance(r['authenticity']['ai_details'], dict) and 
                                r['authenticity']['ai_details'].get('source') == 'eden_ai' and
                                'providers_used' in r['authenticity']['ai_details']):
                                eden_used = True
                                providers_used.update(r['authenticity']['ai_details']['providers_used'])
                        
                        if eden_used:
                            st.info(f"🔍 Responses were analyzed using Eden AI's professional detection engines: {', '.join(providers_used)}")
                        else:
                            st.info("🔍 Responses were analyzed using built-in AI pattern detection")
                        
                        # Enhanced trend visualization
                        st.write("### AI Detection Score Trend")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        
                        # Plot both scores and confidence
                        x = range(len(ai_scores))
                        ax.plot(x, ai_scores, marker='o', label='AI Detection Score', color='blue', linewidth=2)
                        ax.plot(x, confidence_scores, marker='s', label='Detection Confidence', color='green', linewidth=1, alpha=0.7)
                        ax.plot(x, plagiarism_scores, marker='^', label='Plagiarism Score', color='orange', linewidth=1, alpha=0.7)
                        
                        # Add thresholds
                        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='AI Alert Threshold (0.7)')
                        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate Concern (0.5)')
                        
                        # Highlight flagged responses
                        for i in responses_with_warning:
                            ax.axvspan(i-0.3, i+0.3, alpha=0.2, color='red')
                        
                        ax.set_ylim(0, 1.05)
                        ax.set_xlabel('Response Number')
                        ax.set_ylabel('Score')
                        ax.set_xticks(x)
                        ax.legend(loc='upper right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Add analysis by interview stage
                        st.write("### Analysis by Interview Stage")
                        
                        # Group data by stage
                        stage_data = {}
                        for i, response in enumerate(st.session_state.responses):
                            if 'authenticity' in response:
                                stage = response['stage']
                                if stage not in stage_data:
                                    stage_data[stage] = []
                                
                                stage_data[stage].append({
                                    'ai_score': response['authenticity'].get('ai_score', 0),
                                    'confidence': response['authenticity'].get('confidence', 0.5),
                                    'plagiarism': response['authenticity'].get('plagiarism_score', 0),
                                    'is_authentic': response['authenticity'].get('is_authentic', True)
                                })
                        
                        # Create stage comparison chart
                        if stage_data:
                            stage_ai_scores = {stage: sum(d['ai_score'] for d in data)/len(data) for stage, data in stage_data.items()}
                            stage_flags = {stage: sum(0 if d['is_authentic'] else 1 for d in data) for stage, data in stage_data.items()}
                            
                            # Create a combined DataFrame for visualization
                            stage_df = pd.DataFrame({
                                'Stage': list(stage_ai_scores.keys()),
                                'AI Score': list(stage_ai_scores.values()),
                                'Flagged': list(stage_flags.values())
                            })
                            
                            # Plot two side-by-side charts
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                            
                            # AI scores by stage
                            bars1 = ax1.bar(stage_df['Stage'], stage_df['AI Score'], color=['red' if s > 0.7 else 'orange' if s > 0.5 else 'green' for s in stage_df['AI Score']])
                            ax1.set_title('Average AI Score by Stage')
                            ax1.set_ylim(0, 1)
                            ax1.set_ylabel('AI Score')
                            ax1.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
                            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
                            
                            # Add data labels
                            for bar in bars1:
                                height = bar.get_height()
                                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                        f'{height:.2f}', ha='center', va='bottom')
                            
                            # Flagged responses by stage
                            bars2 = ax2.bar(stage_df['Stage'], stage_df['Flagged'], color='orange')
                            ax2.set_title('Flagged Responses by Stage')
                            ax2.set_ylabel('Count')
                            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
                            
                            # Add data labels
                            for bar in bars2:
                                height = bar.get_height()
                                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                        f'{int(height)}', ha='center', va='bottom')
                                
                            plt.tight_layout()
                            st.pyplot(fig)
                        
                        # Display flagged responses section
                        if responses_with_warning:
                            st.write("### Responses with Authenticity Concerns")
                            
                            for i in responses_with_warning:
                                response = st.session_state.responses[i]
                                auth = response['authenticity']
                                
                                # Calculate warning level for UI
                                warning_level = "High"
                                warning_color = "red"
                                
                                if auth['ai_score'] > 0.8 and auth.get('confidence', 0) > 0.8:
                                    warning_level = "Very High"
                                    warning_color = "darkred"
                                elif auth['ai_score'] > 0.7:
                                    warning_level = "High"
                                    warning_color = "red"
                                elif auth['ai_score'] > 0.5:
                                    warning_level = "Moderate"
                                    warning_color = "orange"
                                
                                # Create expander with color indicator
                                with st.expander(f"Question {i+1}: {response['question'][:70]}... [{warning_level} Concern]"):
                                    st.markdown(f"<h4 style='color:{warning_color}'>Warning: {auth['warning']}</h4>", unsafe_allow_html=True)
                                    
                                    # Display response text
                                    st.write("**Response:**", response['response'])
                                    
                                    # Create metrics columns
                                    cols = st.columns(3)
                                    with cols[0]:
                                        st.metric("AI Score", f"{auth['ai_score']:.2f}")
                                    with cols[1]:
                                        st.metric("Confidence", f"{auth.get('confidence', 0):.2f}")
                                    with cols[2]:
                                        st.metric("Plagiarism", f"{auth.get('plagiarism_score', 0):.2f}")
                                    
                                    # Show detection method used
                                    if ('ai_details' in auth and isinstance(auth['ai_details'], dict) and 
                                        auth['ai_details'].get('source') == 'eden_ai' and
                                        'providers_used' in auth['ai_details']):
                                        providers = auth['ai_details']['providers_used']
                                        st.info(f"Detected using: {', '.join(providers)}")
                                        
                                        # Show individual provider scores if available
                                        if 'details' in auth['ai_details']:
                                            st.write("**Provider Results:**")
                                            for provider, details in auth['ai_details']['details'].items():
                                                if isinstance(details, dict) and 'ai_score' in details:
                                                    score = details['ai_score']
                                                    conf = details.get('confidence', 0.5)
                                                    score_color = "red" if score > 0.7 else "orange" if score > 0.5 else "green"
                                                    st.markdown(f"- {provider.title()}: <span style='color:{score_color}'>{score:.2f}</span> (confidence: {conf:.2f})", unsafe_allow_html=True)
                                    
                                    # For built-in detection, show pattern analysis
                                    elif 'ai_details' in auth and 'details' in auth['ai_details']:
                                        st.write("**AI Patterns Detected:**")
                                        details = auth['ai_details']['details']
                                        
                                        # Sort patterns by score
                                        patterns = [(k.replace('_', ' ').title(), v) for k, v in details.items()]
                                        patterns.sort(key=lambda x: x[1], reverse=True)
                                        
                                        # Display top patterns
                                        for pattern, score in patterns[:5]:  # Show top 5
                                            if score > 0.3:  # Only show significant patterns
                                                score_color = "red" if score > 0.7 else "orange" if score > 0.5 else "blue"
                                                st.markdown(f"- {pattern}: <span style='color:{score_color}'>{score:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.success("No responses were flagged for authenticity concerns!")
                    else:
                        st.info("AI detection was not enabled for this interview session or no responses were recorded.")
            
            # Additional actions
            st.divider()
            st.subheader("Additional Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Start New Interview"):
                    cleanup_resources()
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            with col2:
                if hasattr(st.session_state, 'report_filepath') and os.path.exists(st.session_state.report_filepath):
                    with open(st.session_state.report_filepath, 'r') as f:
                        st.download_button(
                            label="📥 Download Complete Report",
                            data=f.read(),
                            file_name=f"complete_{st.session_state.report_filename}",
                            mime="application/json",
                            key="complete_download"
                        )

if __name__ == "__main__":
    main()