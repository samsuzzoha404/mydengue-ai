from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class ReportStatus(str, Enum):
    PENDING = "Pending"
    INVESTIGATING = "Investigating"
    VERIFIED = "Verified"
    RESOLVED = "Resolved"
    REJECTED = "Rejected"

class PredictionRequest(BaseModel):
    location: str = Field(..., description="Location name (e.g., 'Mont Kiara, KL')")
    state: str = Field(..., description="Malaysian state")
    date: Optional[str] = Field(None, description="Target prediction date (ISO format)")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    rainfall: Optional[float] = Field(None, description="Rainfall in mm")
    wind_speed: Optional[float] = Field(None, description="Wind speed in km/h")
    description: Optional[str] = Field(None, description="Additional context")

class PredictionResponse(BaseModel):
    prediction_id: str
    location: str
    state: str
    risk_level: RiskLevel
    predicted_cases: int
    confidence: float = Field(..., ge=0, le=1)
    prediction_date: datetime
    weather_factors: Dict[str, Any]
    risk_factors: List[str]
    recommendations: List[str]
    created_at: datetime

class CitizenReport(BaseModel):
    location: str = Field(..., description="Report location")
    latitude: Optional[float] = Field(None, description="GPS latitude")
    longitude: Optional[float] = Field(None, description="GPS longitude")
    description: str = Field(..., description="Description of the breeding site")
    image_url: Optional[str] = Field(None, description="Uploaded image URL")
    reporter_contact: Optional[str] = Field(None, description="Reporter contact info")
    
class CitizenReportResponse(BaseModel):
    report_id: str
    location: str
    coordinates: Optional[Dict[str, float]]
    description: str
    ai_classification: str
    confidence: float
    status: ReportStatus
    points_earned: int
    created_at: datetime
    estimated_resolve_time: str

class AlertRequest(BaseModel):
    location: str
    message: str
    severity: RiskLevel
    target_audience: List[str] = Field(default=["citizens", "authorities"])
    languages: List[str] = Field(default=["en", "ms"])

class AlertResponse(BaseModel):
    alert_id: str
    location: str
    message: Dict[str, str]  # language -> translated message
    severity: RiskLevel
    sent_at: datetime
    recipients_count: int
    delivery_channels: List[str]

class HeatmapRequest(BaseModel):
    state: Optional[str] = Field(None, description="Filter by specific state")
    risk_level: Optional[RiskLevel] = Field(None, description="Filter by risk level")
    date_range: Optional[int] = Field(7, description="Days to look back")

class HeatmapData(BaseModel):
    state: str
    district: str
    coordinates: Dict[str, float]  # lat, lng
    risk_level: RiskLevel
    active_cases: int
    citizen_reports: int
    predicted_cases: int
    last_updated: datetime

class HeatmapResponse(BaseModel):
    data: List[HeatmapData]
    total_states: int
    high_risk_areas: int
    last_updated: datetime

class DashboardStats(BaseModel):
    total_cases_week: int
    high_risk_zones: int
    citizen_reports: int
    ai_accuracy: float
    weekly_trend: float
    
class DashboardResponse(BaseModel):
    stats: DashboardStats
    recent_predictions: List[Dict[str, Any]]
    hotspots: List[Dict[str, Any]]
    recent_reports: List[Dict[str, Any]]
    weather_summary: Dict[str, Any]
    last_updated: datetime

# Gamification Models
class UserProfile(BaseModel):
    user_id: str
    username: str
    points: int = 0
    level: int = 1
    badges: List[str] = []
    reports_submitted: int = 0
    reports_verified: int = 0
    created_at: datetime

class Reward(BaseModel):
    reward_id: str
    user_id: str
    points: int
    reason: str
    badge: Optional[str] = None
    earned_at: datetime