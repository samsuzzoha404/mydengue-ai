"""
Firebase Configuration and Initialization
Using provided credentials from dengu-230be project
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Firebase configuration from provided credentials
FIREBASE_CONFIG = {
    "apiKey": "AIzaSyAG7ZwT6DROKONyij1UkybXO1yGvADYdAo",
    "authDomain": "dengu-230be.firebaseapp.com",
    "databaseURL": "https://dengu-230be-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "dengu-230be",
    "storageBucket": "dengu-230be.firebasestorage.app",
    "messagingSenderId": "1053122580440",
    "appId": "1:1053122580440:web:9837596031b1e1afc6acad",
    "measurementId": "G-CKVVNN6E33"
}

class FirebaseManager:
    """
    Firebase integration manager for real-time data and cloud storage
    """
    
    def __init__(self):
        self.initialized = False
        self.app = None
        self.db = None
        self.storage = None
        
    def initialize_firebase(self) -> bool:
        """Initialize Firebase with service account credentials"""
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore, storage
            
            # Path to service account credentials
            cred_path = os.path.join(os.path.dirname(__file__), '..', '..', 'firebase-credentials.json')
            
            if not os.path.exists(cred_path):
                logger.warning(f"Firebase credentials not found at {cred_path}")
                return False
            
            # Initialize with service account
            cred = credentials.Certificate(cred_path)
            self.app = firebase_admin.initialize_app(cred, {
                'storageBucket': FIREBASE_CONFIG['storageBucket'],
                'databaseURL': FIREBASE_CONFIG['databaseURL']
            })
            
            # Initialize Firestore
            self.db = firestore.client()
            
            # Initialize Storage
            self.storage = storage.bucket()
            
            self.initialized = True
            logger.info("✅ Firebase initialized successfully with real credentials")
            return True
            
        except ImportError:
            logger.warning("⚠️ Firebase Admin SDK not installed")
            return False
        except Exception as e:
            logger.error(f"Firebase initialization failed: {e}")
            return False
    
    async def store_citizen_report(self, report_data: dict) -> bool:
        """Store citizen report in Firestore"""
        if not self.initialized:
            logger.warning("Firebase not initialized")
            return False
        
        try:
            # Store in 'citizen_reports' collection
            doc_ref = self.db.collection('citizen_reports').document()
            doc_ref.set(report_data)
            
            logger.info(f"✅ Citizen report stored in Firebase: {doc_ref.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store citizen report: {e}")
            return False
    
    async def store_ai_analysis(self, analysis_data: dict) -> bool:
        """Store AI analysis results in Firestore"""
        if not self.initialized:
            logger.warning("Firebase not initialized")
            return False
        
        try:
            # Store in 'ai_analyses' collection
            doc_ref = self.db.collection('ai_analyses').document()
            doc_ref.set(analysis_data)
            
            logger.info(f"✅ AI analysis stored in Firebase: {doc_ref.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store AI analysis: {e}")
            return False
    
    async def send_push_notification(self, message: str, tokens: list = None) -> bool:
        """Send push notification via Firebase Cloud Messaging"""
        if not self.initialized:
            logger.warning("Firebase not initialized")
            return False
        
        try:
            from firebase_admin import messaging
            
            # Create notification
            notification = messaging.Notification(
                title="Dengue Alert",
                body=message
            )
            
            if tokens:
                # Send to specific devices
                multicast_message = messaging.MulticastMessage(
                    notification=notification,
                    tokens=tokens
                )
                response = messaging.send_multicast(multicast_message)
                logger.info(f"✅ Push notification sent to {response.success_count} devices")
            else:
                # Send to topic (all subscribers)
                message = messaging.Message(
                    notification=notification,
                    topic='dengue_alerts'
                )
                response = messaging.send(message)
                logger.info(f"✅ Push notification sent to topic: {response}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send push notification: {e}")
            return False
    
    def get_status(self) -> dict:
        """Get Firebase integration status"""
        return {
            "initialized": self.initialized,
            "project_id": FIREBASE_CONFIG["projectId"],
            "has_firestore": self.db is not None,
            "has_storage": self.storage is not None,
            "features": [
                "Real-time Database",
                "Cloud Firestore", 
                "Cloud Storage",
                "Cloud Messaging",
                "Authentication"
            ] if self.initialized else []
        }

# Global Firebase manager instance
firebase_manager = FirebaseManager()

def initialize_firebase() -> bool:
    """Initialize Firebase on application startup"""
    return firebase_manager.initialize_firebase()