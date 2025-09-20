from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from datetime import datetime
import uuid

from app.models.schemas import AlertRequest, AlertResponse, RiskLevel

router = APIRouter()

class AlertService:
    """Service for managing dengue alerts and notifications"""
    
    def __init__(self):
        self.translation_map = {
            "en": {
                "high_risk_alert": "HIGH RISK: Dengue outbreak risk detected in your area",
                "medium_risk_alert": "MODERATE RISK: Increased dengue activity in your area", 
                "prevention_reminder": "REMINDER: Check and remove stagnant water around your home",
                "outbreak_confirmed": "OUTBREAK CONFIRMED: Take immediate preventive measures"
            },
            "ms": {
                "high_risk_alert": "RISIKO TINGGI: Risiko wabak denggi dikesan di kawasan anda",
                "medium_risk_alert": "RISIKO SEDERHANA: Peningkatan aktiviti denggi di kawasan anda",
                "prevention_reminder": "PERINGATAN: Periksa dan buang air bertakung di sekitar rumah",
                "outbreak_confirmed": "WABAK DISAHKAN: Ambil langkah pencegahan segera"
            }
        }
    
    def send_alert(self, request: AlertRequest) -> AlertResponse:
        """Send multilingual alert to target audience"""
        
        alert_id = str(uuid.uuid4())
        
        # Translate message to requested languages
        translated_messages = {}
        for lang in request.languages:
            if lang in self.translation_map:
                translated_messages[lang] = self._translate_message(request.message, lang)
            else:
                translated_messages[lang] = request.message  # Fallback to original
        
        # Calculate estimated recipients
        recipients = self._estimate_recipients(request.location, request.target_audience)
        
        # Determine delivery channels
        channels = self._get_delivery_channels(request.severity)
        
        return AlertResponse(
            alert_id=alert_id,
            location=request.location,
            message=translated_messages,
            severity=request.severity,
            sent_at=datetime.now(),
            recipients_count=recipients,
            delivery_channels=channels
        )
    
    def _translate_message(self, message: str, language: str) -> str:
        """Translate message to target language"""
        # In production, would use Google Translate API
        # For now, use predefined translations or return original
        
        message_lower = message.lower()
        
        for key, translations in self.translation_map[language].items():
            if any(keyword in message_lower for keyword in key.split("_")):
                return translations
        
        return message  # Fallback to original if no translation found
    
    def _estimate_recipients(self, location: str, target_audience: List[str]) -> int:
        """Estimate number of alert recipients based on location and audience"""
        
        base_population = {
            "kuala lumpur": 50000,
            "selangor": 80000,
            "johor": 60000,
            "penang": 40000,
            "perak": 35000
        }
        
        # Get base population for the area
        location_lower = location.lower()
        population = 10000  # Default
        
        for area, pop in base_population.items():
            if area in location_lower:
                population = pop
                break
        
        # Adjust based on target audience
        multiplier = 1.0
        if "citizens" in target_audience:
            multiplier += 0.8
        if "authorities" in target_audience:
            multiplier += 0.1
        if "healthcare" in target_audience:
            multiplier += 0.05
        
        return int(population * multiplier * 0.7)  # 70% app penetration assumption
    
    def _get_delivery_channels(self, severity: RiskLevel) -> List[str]:
        """Determine delivery channels based on alert severity"""
        
        channels = ["app_notification"]  # Always include app notifications
        
        if severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            channels.extend(["sms", "email", "public_announcement"])
        
        if severity == RiskLevel.CRITICAL:
            channels.extend(["emergency_broadcast", "social_media"])
        
        return channels

alert_service = AlertService()

@router.post("/alerts", response_model=AlertResponse)
async def send_dengue_alert(request: AlertRequest):
    """
    Send dengue outbreak alert to target audiences
    
    Sends multilingual alerts via multiple channels based on severity level
    """
    try:
        response = alert_service.send_alert(request)
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert sending error: {str(e)}"
        )

@router.get("/alerts/recent")
async def get_recent_alerts(limit: int = 10, severity: Optional[str] = None):
    """
    Get recent dengue alerts with optional severity filtering
    """
    try:
        alerts = []
        severities = ["High", "Critical", "Medium", "Low"]
        
        if severity:
            severities = [s for s in severities if s.lower() == severity.lower()]
        
        for i in range(min(limit, 15)):
            alerts.append({
                "alert_id": f"alert_{i+1}",
                "location": f"District {i+1}, Selangor" if i % 2 == 0 else f"Area {i+1}, KL",
                "message": {
                    "en": f"Alert message {i+1} in English",
                    "ms": f"Mesej amaran {i+1} dalam Bahasa Malaysia"
                },
                "severity": severities[i % len(severities)],
                "sent_at": f"2024-01-{20-i:02d}T{8+i%16:02d}:00:00Z",
                "recipients_count": 5000 + (i * 500),
                "delivery_channels": ["app_notification", "sms", "email"][:2+i%2]
            })
        
        return {
            "total_alerts": len(alerts),
            "severity_filter": severity,
            "alerts": alerts
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert retrieval error: {str(e)}"
        )

@router.get("/alerts/active")
async def get_active_alerts():
    """
    Get currently active dengue alerts
    """
    try:
        return {
            "active_alerts": 3,
            "total_affected": 125000,
            "alerts": [
                {
                    "alert_id": "alert_critical_1",
                    "location": "Mont Kiara, Kuala Lumpur",
                    "severity": "Critical",
                    "message": "Dengue outbreak confirmed. Take immediate action.",
                    "issued_at": "2024-01-19T06:00:00Z",
                    "expires_at": "2024-01-22T23:59:59Z",
                    "affected_population": 50000
                },
                {
                    "alert_id": "alert_high_1",
                    "location": "Petaling Jaya, Selangor", 
                    "severity": "High",
                    "message": "High risk conditions detected. Increase prevention measures.",
                    "issued_at": "2024-01-18T14:00:00Z",
                    "expires_at": "2024-01-21T23:59:59Z",
                    "affected_population": 45000
                },
                {
                    "alert_id": "alert_medium_1",
                    "location": "Shah Alam, Selangor",
                    "severity": "Medium", 
                    "message": "Moderate dengue activity increase observed.",
                    "issued_at": "2024-01-17T10:00:00Z",
                    "expires_at": "2024-01-20T23:59:59Z",
                    "affected_population": 30000
                }
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Active alerts retrieval error: {str(e)}"
        )

@router.get("/alerts/stats")
async def get_alert_statistics():
    """
    Get alert system statistics
    """
    try:
        return {
            "total_sent": 234,
            "this_week": 12,
            "active_alerts": 3,
            "delivery_success_rate": 0.94,
            "avg_response_time": "3.2 minutes",
            "channels_performance": {
                "app_notification": {"sent": 180000, "delivered": 175000, "rate": 0.97},
                "sms": {"sent": 45000, "delivered": 42000, "rate": 0.93}, 
                "email": {"sent": 25000, "delivered": 23500, "rate": 0.94}
            },
            "language_distribution": {
                "en": 0.60,
                "ms": 0.30,
                "ta": 0.07,
                "zh": 0.03
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics retrieval error: {str(e)}"
        )