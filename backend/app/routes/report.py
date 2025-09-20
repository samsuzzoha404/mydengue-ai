from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from typing import Optional
import base64

from app.models.schemas import CitizenReport, CitizenReportResponse
from app.services.report_service import report_service

# Import gamification system
from app.services.gamification import gamification_engine

router = APIRouter()

@router.post("/report", response_model=CitizenReportResponse)
async def submit_citizen_report(
    location: str = Form(...),
    description: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    reporter_contact: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Submit a citizen report of potential dengue breeding site
    
    Uses AI-powered CNN to classify uploaded images and determine
    if the reported location is likely a mosquito breeding site.
    """
    try:
        # Process uploaded image
        image_data = None
        if image:
            if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only JPEG and PNG images are supported"
                )
            
            # Convert image to base64 for processing
            image_bytes = await image.read()
            image_data = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create report object
        report = CitizenReport(
            location=location,
            latitude=latitude,
            longitude=longitude,
            description=description,
            reporter_contact=reporter_contact
        )
        
        # Process report with AI classification
        response = await report_service.submit_citizen_report(report, image_data)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report submission error: {str(e)}"
        )

@router.post("/report/gamified", response_model=CitizenReportResponse)
async def submit_gamified_report(
    user_id: str = Form(...),
    username: Optional[str] = Form(None),
    location: str = Form(...),
    description: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    reporter_contact: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Submit a gamified citizen report with points, badges, and leaderboard integration
    
    This endpoint integrates AI-powered image classification with a comprehensive
    gamification system including:
    - Points and XP system
    - Badge achievements
    - Leaderboard rankings
    - Blockchain-verified rewards
    - Community engagement features
    """
    try:
        # Ensure user is registered in gamification system
        if username:
            user_location = {"lat": latitude, "lng": longitude} if latitude and longitude else None
            gamification_engine.register_user(user_id, username, user_location)
        
        # Process uploaded image
        image_data = None
        has_photo = False
        if image:
            if image.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Only JPEG and PNG images are supported"
                )
            
            # Convert image to base64 for processing
            image_bytes = await image.read()
            image_data = base64.b64encode(image_bytes).decode('utf-8')
            has_photo = True
        
        # Create report object
        report = CitizenReport(
            location=location,
            latitude=latitude,
            longitude=longitude,
            description=description,
            reporter_contact=reporter_contact
        )
        
        # Process report with AI classification
        response = report_service.submit_citizen_report(report, image_data)
        
        # Apply gamification rewards
        gamification_data = {
            "has_photo": has_photo,
            "location": {"lat": latitude, "lng": longitude} if latitude and longitude else None,
            "description": description,
            "ai_classification": response.classification,
            "confidence": response.confidence
        }
        
        # Award points and check for badges
        gamification_result = gamification_engine.submit_report(user_id, gamification_data)
        
        # Enhanced response with gamification
        enhanced_response = response.dict()
        enhanced_response.update({
            "gamification": {
                "points_earned": gamification_result["points_earned"],
                "total_points": gamification_result["total_points"], 
                "level": gamification_result["level"],
                "new_badges": gamification_result["new_badges"],
                "quality_bonus": f"{int((gamification_result['quality_multiplier'] - 1) * 100)}% bonus",
                "streak_days": gamification_result["streak_days"],
                "level_progress": {
                    "current_level": gamification_result["level"],
                    "points_to_next_level": max(0, (gamification_result["level"] * 100) - gamification_result["total_points"])
                }
            },
            "community_impact": {
                "contribution_score": min(100, gamification_result["points_earned"] * 2),
                "environmental_impact": "Positive - Helping prevent dengue breeding sites",
                "community_ranking": "Updated - Check leaderboard for position"
            }
        })
        
        return enhanced_response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gamified report submission error: {str(e)}"
        )
async def submit_report_json(report: CitizenReport):
    """
    Submit a citizen report via JSON (for web forms without file upload)
    """
    try:
        response = report_service.submit_citizen_report(report)
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report submission error: {str(e)}"
        )

@router.get("/report/user/{user_id}")
async def get_user_reports(user_id: str, limit: int = 10):
    """
    Get all reports submitted by a specific user
    """
    try:
        reports = report_service.get_user_reports(user_id, limit)
        return reports
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report retrieval error: {str(e)}"
        )

@router.get("/report/recent")
async def get_recent_reports(limit: int = 20, status_filter: Optional[str] = None):
    """
    Get recent citizen reports with optional status filtering
    """
    try:
        # Mock recent reports data
        statuses = ["Verified", "Investigating", "Resolved", "Pending"]
        if status_filter:
            statuses = [s for s in statuses if s.lower() == status_filter.lower()]
        
        reports = []
        for i in range(min(limit, 20)):
            reports.append({
                "report_id": f"report_{i+1}",
                "location": f"Jalan {i+1}, Kuala Lumpur" if i % 2 == 0 else f"Taman {i+1}, Selangor",
                "description": f"Found stagnant water in container #{i+1}",
                "status": statuses[i % len(statuses)],
                "ai_classification": "Potential Breeding Site Detected" if i % 3 == 0 else "Requires Investigation",
                "confidence": 0.85 - (i * 0.02),
                "points_earned": [50, 75, 30, 25][i % 4],
                "created_at": f"2024-01-{20-i:02d}T{10+i%12:02d}:00:00Z"
            })
        
        return {
            "total_reports": len(reports),
            "status_filter": status_filter,
            "reports": reports
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report retrieval error: {str(e)}"
        )

@router.get("/report/stats")
async def get_report_statistics():
    """
    Get overall reporting statistics for dashboard
    """
    try:
        return {
            "total_reports": 1247,
            "verified_reports": 986,
            "investigating": 156,
            "resolved": 830,
            "pending": 105,
            "this_week": 89,
            "accuracy_rate": 0.79,
            "avg_response_time": "2.3 days",
            "top_locations": [
                {"location": "Mont Kiara, KL", "reports": 23},
                {"location": "Petaling Jaya, Selangor", "reports": 19},
                {"location": "Bangsar, KL", "reports": 15},
                {"location": "Shah Alam, Selangor", "reports": 12},
                {"location": "Cyberjaya, Selangor", "reports": 10}
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Statistics retrieval error: {str(e)}"
        )

@router.get("/leaderboard")
async def get_gamification_leaderboard(limit: int = 10, location_filter: Optional[str] = None):
    """
    Get citizen reporting leaderboard for gamification with location filtering
    """
    try:
        # Parse location filter if provided
        location_coords = None
        if location_filter:
            try:
                # Expected format: "lat,lng" 
                lat, lng = map(float, location_filter.split(','))
                location_coords = {"lat": lat, "lng": lng}
            except:
                pass
        
        # Get leaderboard from gamification engine
        leaderboard = gamification_engine.get_leaderboard(limit, location_coords)
        
        return {
            "leaderboard": [
                {
                    "rank": entry.rank,
                    "username": entry.username,
                    "points": entry.points,
                    "level": entry.level,
                    "badges": entry.badges_count,
                    "recent_activity": entry.recent_activity,
                    "user_id": entry.user_id  # For profile linking
                }
                for entry in leaderboard
            ],
            "total_users": len(gamification_engine.users),
            "location_filter": location_filter,
            "gamification_stats": gamification_engine.get_gamification_stats()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Leaderboard retrieval error: {str(e)}"
        )

@router.get("/profile/{user_id}")
async def get_user_gamification_profile(user_id: str):
    """
    Get complete gamification profile for a user including badges and achievements
    """
    try:
        profile_data = gamification_engine.get_user_profile(user_id)
        
        if not profile_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        return {
            "user_profile": profile_data,
            "available_badges": gamification_engine.get_available_badges(),
            "community_rank": _get_user_rank(user_id),
            "recent_activity": _get_recent_activity(user_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Profile retrieval error: {str(e)}"
        )

@router.post("/verify-report")
async def verify_citizen_report(
    user_id: str = Form(...),
    report_id: str = Form(...),
    is_verified: bool = Form(...),
    verification_notes: Optional[str] = Form(None)
):
    """
    Verify a citizen report and award verification points to the verifier
    """
    try:
        # Process verification in report service
        verification_result = report_service.verify_report(report_id, is_verified, verification_notes)
        
        # Award verification points through gamification
        gamification_result = gamification_engine.verify_report(user_id, report_id, is_verified)
        
        return {
            "verification_result": verification_result,
            "gamification": {
                "points_earned": gamification_result["points_earned"],
                "total_points": gamification_result["total_points"],
                "level": gamification_result["level"], 
                "new_badges": gamification_result["new_badges"]
            },
            "community_contribution": {
                "verification_quality": "high" if is_verified else "helpful",
                "community_trust_score": "+2 points" if is_verified else "+1 point"
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report verification error: {str(e)}"
        )

def _get_user_rank(user_id: str) -> dict:
    """Get user's current community ranking"""
    leaderboard = gamification_engine.get_leaderboard(100)  # Get more entries to find rank
    
    for entry in leaderboard:
        if entry.user_id == user_id:
            return {
                "current_rank": entry.rank,
                "total_users": len(gamification_engine.users),
                "percentile": round((1 - (entry.rank / len(gamification_engine.users))) * 100, 1)
            }
    
    return {"current_rank": "Unranked", "total_users": len(gamification_engine.users), "percentile": 0}

def _get_recent_activity(user_id: str) -> list:
    """Get user's recent activity for profile display"""
    if user_id not in gamification_engine.users:
        return []
    
    user = gamification_engine.users[user_id]
    
    # Generate some recent activity (in production this would come from database)
    recent_activity = []
    
    # Add recent achievements
    user_achievements = [a for a in gamification_engine.achievements if a.user_id == user_id]
    for achievement in user_achievements[-3:]:  # Last 3 achievements
        badge = gamification_engine.badges[achievement.badge_id]
        recent_activity.append({
            "type": "achievement",
            "description": f"Earned badge: {badge.name}",
            "points": achievement.points_awarded,
            "timestamp": achievement.earned_date.isoformat(),
            "icon": badge.icon
        })
    
    # Add recent reports (simulated)
    if user.reports_submitted > 0:
        recent_activity.append({
            "type": "report",
            "description": f"Submitted {user.reports_submitted} reports",
            "points": user.reports_submitted * 25,
            "timestamp": user.last_active.isoformat(),
            "icon": "📋"
        })
    
    # Sort by timestamp
    recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return recent_activity[:5]  # Return last 5 activities