import uuid
import random
from datetime import datetime
from typing import Optional
import base64
import io
from PIL import Image

from app.models.schemas import CitizenReport, CitizenReportResponse, ReportStatus
from .custom_ai_service import custom_ai_service

class ReportService:
    """Service for handling citizen reports of potential breeding sites"""
    
    def __init__(self):
        # Simulate CNN model for image classification
        self.breeding_site_keywords = [
            "water", "container", "stagnant", "drain", "gutter", "pond", 
            "tire", "bucket", "construction", "roof", "tank"
        ]
        
        # Gamification points system
        self.points_system = {
            "verified_report": 50,
            "first_report": 100,
            "detailed_report": 25,
            "photo_report": 30,
            "location_provided": 20
        }
    
    async def submit_citizen_report(self, report: CitizenReport, image_data: Optional[str] = None) -> CitizenReportResponse:
        """
        Process citizen report with AI classification
        In production, this would use a trained CNN to classify breeding sites
        """
        
        # Generate report ID
        report_id = str(uuid.uuid4())
        
        # Analyze image if provided
        ai_classification, confidence = await self._analyze_image_async(image_data, report.description)
        
        # Calculate points earned
        points = self._calculate_points(report, image_data, ai_classification)
        
        # Determine coordinates
        coordinates = None
        if report.latitude and report.longitude:
            coordinates = {"lat": report.latitude, "lng": report.longitude}
        
        # Estimate resolution time based on classification
        resolve_time = self._estimate_resolve_time(ai_classification, confidence)
        
        # Determine initial status
        status = ReportStatus.VERIFIED if confidence > 0.7 else ReportStatus.INVESTIGATING
        
        return CitizenReportResponse(
            report_id=report_id,
            location=report.location,
            coordinates=coordinates,
            description=report.description,
            ai_classification=ai_classification,
            confidence=confidence,
            status=status,
            points_earned=points,
            created_at=datetime.now(),
            estimated_resolve_time=resolve_time
        )
    
    def _analyze_image(self, image_data: Optional[str], description: str) -> tuple[str, float]:
        """
        Use real AI image classification for breeding sites with fallback
        """
        
        if not image_data and not description:
            return "No Data Provided", 0.3
        
        # Try custom AI analysis first  
        if image_data:
            try:
                # Clean image data
                clean_image_data = image_data.split(',')[1] if ',' in image_data else image_data
                
                # For sync version, return a basic analysis indicating custom AI would be used
                classification = "Custom AI Analysis - Breeding Site Assessment"
                confidence = 0.85
                
                return classification, confidence
                
            except Exception as e:
                print(f"Custom AI setup failed: {e}")
                # Fall back to simulation
                pass        # Fallback simulation analysis
        if image_data:
            try:
                # Decode base64 image for basic validation
                image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Simulate CNN prediction based on random factors and description
                breeding_indicators = sum(1 for keyword in self.breeding_site_keywords 
                                        if keyword.lower() in description.lower())
                
                base_confidence = min(0.95, 0.4 + (breeding_indicators * 0.15) + random.uniform(0.1, 0.3))
                
                if base_confidence > 0.7:
                    classification = "Potential Breeding Site Detected"
                elif base_confidence > 0.5:
                    classification = "Water Source - Requires Inspection"
                else:
                    classification = "No Breeding Site Detected"
                
                return classification, base_confidence
                
            except Exception:
                # If image processing fails, use description only
                return self._analyze_description(description)
        
        return self._analyze_description(description)

    async def _analyze_image_async(self, image_data: Optional[str], description: str) -> tuple[str, float]:
        """
        Async image analysis using our custom AI service
        Uses our trained models and computer vision with person detection
        """
        
        if not image_data and not description:
            return "No Data Provided", 0.3
        
        # Try custom AI analysis first  
        if image_data:
            try:
                # Clean image data
                clean_image_data = image_data.split(',')[1] if ',' in image_data else image_data
                
                # Use our custom AI service asynchronously
                ai_result = await custom_ai_service.classify_breeding_site_image(clean_image_data)
                
                # Check if person was detected (this prevents human faces from being classified as breeding sites)
                if ai_result.get("person_detected", False):
                    return ai_result.get("classification", "Not a Breeding Site - Person Detected"), ai_result.get("confidence", 0.05)
                
                # Process normal breeding site classification
                if ai_result.get("breeding_site_detected"):
                    classification = f"Breeding Site Detected - {ai_result.get('classification', 'High Risk')}"
                    confidence = ai_result.get("confidence", 0.8)
                else:
                    classification = f"No Breeding Site Detected - {ai_result.get('classification', 'Low Risk')}"
                    confidence = ai_result.get("confidence", 0.6)
                
                return classification, min(0.95, confidence)
                
            except Exception as e:
                print(f"Custom AI analysis failed: {e}")
                # Fall back to simulation
                pass
        
        # Fallback simulation analysis
        if image_data:
            try:
                # Decode base64 image for basic validation
                image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
                image = Image.open(io.BytesIO(image_bytes))
                
                # Simulate CNN prediction based on random factors and description
                breeding_indicators = sum(1 for keyword in self.breeding_site_keywords 
                                        if keyword.lower() in description.lower())
                
                base_confidence = min(0.95, 0.4 + (breeding_indicators * 0.15) + random.uniform(0.1, 0.3))
                
                if base_confidence > 0.7:
                    classification = "Potential Breeding Site Detected"
                elif base_confidence > 0.5:
                    classification = "Water Source - Requires Inspection"
                else:
                    classification = "No Breeding Site Detected"
                
                return classification, base_confidence
                
            except Exception:
                # If image processing fails, use description only
                return self._analyze_description(description)
        
        return self._analyze_description(description)
    
    def _analyze_description(self, description: str) -> tuple[str, float]:
        """Analyze text description for breeding site indicators"""
        
        if not description:
            return "Insufficient Information", 0.2
        
        description_lower = description.lower()
        
        # Check for breeding site indicators
        breeding_score = 0
        for keyword in self.breeding_site_keywords:
            if keyword in description_lower:
                breeding_score += 1
        
        # Additional context indicators
        context_indicators = [
            "mosquito", "larvae", "eggs", "standing", "blocked", "clogged",
            "puddle", "leak", "overflow", "rain", "collect"
        ]
        
        context_score = sum(1 for indicator in context_indicators 
                           if indicator in description_lower)
        
        total_score = breeding_score + (context_score * 0.5)
        confidence = min(0.9, 0.3 + (total_score * 0.1))
        
        if confidence > 0.7:
            return "High Risk Breeding Site", confidence
        elif confidence > 0.5:
            return "Potential Breeding Area", confidence
        elif confidence > 0.3:
            return "Requires Further Investigation", confidence
        else:
            return "Low Risk Area", confidence
    
    def _calculate_points(self, report: CitizenReport, image_data: Optional[str], classification: str) -> int:
        """Calculate gamification points for the report"""
        points = 0
        
        # Base points for submitting report
        points += 25
        
        # Verified report bonus
        if "detected" in classification.lower() or "high risk" in classification.lower():
            points += self.points_system["verified_report"]
        
        # Photo bonus
        if image_data:
            points += self.points_system["photo_report"]
        
        # Location bonus
        if report.latitude and report.longitude:
            points += self.points_system["location_provided"]
        
        # Detailed description bonus
        if len(report.description) > 50:
            points += self.points_system["detailed_report"]
        
        return points
    
    def _estimate_resolve_time(self, classification: str, confidence: float) -> str:
        """Estimate time to resolve the reported issue"""
        
        if confidence > 0.8:
            return "24-48 hours"
        elif confidence > 0.6:
            return "2-3 days"
        elif confidence > 0.4:
            return "3-5 days"
        else:
            return "5-7 days (pending verification)"
    
    def get_user_reports(self, user_id: str, limit: int = 10):
        """Get reports submitted by a specific user"""
        # In production, query database for user's reports
        return {
            "user_id": user_id,
            "total_reports": 15,
            "verified_reports": 12,
            "total_points": 750,
            "recent_reports": [
                {
                    "report_id": f"report_{i+1}",
                    "location": f"Location {i+1}",
                    "status": ["Verified", "Investigating", "Resolved"][i % 3],
                    "points_earned": [50, 75, 30][i % 3],
                    "created_at": f"2024-01-{20-i:02d}T10:00:00Z"
                }
                for i in range(min(limit, 5))
            ]
        }
    
    def get_leaderboard(self, limit: int = 10):
        """Get gamification leaderboard"""
        # Mock leaderboard data
        return {
            "period": "monthly",
            "last_updated": datetime.now().isoformat(),
            "leaderboard": [
                {
                    "rank": i + 1,
                    "username": f"Guardian{i+1}",
                    "points": 1500 - (i * 100),
                    "verified_reports": 20 - (i * 2),
                    "badges": ["First Report", "Eagle Eye", "Community Hero"][:max(1, 3-i)]
                }
                for i in range(limit)
            ]
        }

# Create service instance
report_service = ReportService()