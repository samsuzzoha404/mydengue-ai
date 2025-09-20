from fastapi import APIRouter, HTTPException, status
from typing import Optional, List
from datetime import datetime, timedelta
import random

from app.models.schemas import HeatmapRequest, HeatmapResponse, HeatmapData, RiskLevel

router = APIRouter()

class HeatmapService:
    """Service for generating dengue risk heatmaps"""
    
    def __init__(self):
        # Malaysian states with major districts and coordinates
        self.locations_data = {
            "Selangor": [
                {"district": "Petaling Jaya", "lat": 3.1073, "lng": 101.6067, "population": 800000},
                {"district": "Shah Alam", "lat": 3.0733, "lng": 101.5185, "population": 700000},
                {"district": "Subang Jaya", "lat": 3.1478, "lng": 101.5820, "population": 650000},
                {"district": "Klang", "lat": 3.0319, "lng": 101.4469, "population": 900000},
                {"district": "Ampang", "lat": 3.1538, "lng": 101.7581, "population": 500000},
            ],
            "Kuala Lumpur": [
                {"district": "Mont Kiara", "lat": 3.1727, "lng": 101.6507, "population": 200000},
                {"district": "Bangsar", "lat": 3.1285, "lng": 101.6732, "population": 180000},
                {"district": "Cheras", "lat": 3.0926, "lng": 101.7261, "population": 450000},
                {"district": "KLCC", "lat": 3.1578, "lng": 101.7123, "population": 300000},
                {"district": "Wangsa Maju", "lat": 3.2051, "lng": 101.7394, "population": 250000},
            ],
            "Johor": [
                {"district": "Johor Bahru", "lat": 1.4655, "lng": 103.7578, "population": 1200000},
                {"district": "Skudai", "lat": 1.5329, "lng": 103.6569, "population": 300000},
                {"district": "Pasir Gudang", "lat": 1.4730, "lng": 103.9070, "population": 200000},
                {"district": "Kulai", "lat": 1.6573, "lng": 103.6004, "population": 250000},
            ],
            "Penang": [
                {"district": "Georgetown", "lat": 5.4141, "lng": 100.3288, "population": 400000},
                {"district": "Bayan Lepas", "lat": 5.2946, "lng": 100.2659, "population": 200000},
                {"district": "Butterworth", "lat": 5.3991, "lng": 100.3641, "population": 300000},
            ],
            "Perak": [
                {"district": "Ipoh", "lat": 4.5841, "lng": 101.0829, "population": 500000},
                {"district": "Taiping", "lat": 4.8500, "lng": 100.7333, "population": 200000},
            ]
        }
    
    def generate_heatmap(self, request: HeatmapRequest) -> HeatmapResponse:
        """Generate risk heatmap data for Malaysian districts"""
        
        heatmap_data = []
        high_risk_count = 0
        
        # Filter states if specified
        states_to_process = [request.state] if request.state else list(self.locations_data.keys())
        
        for state in states_to_process:
            if state not in self.locations_data:
                continue
                
            for location in self.locations_data[state]:
                # Generate risk data for each district
                risk_level, cases, reports, predicted = self._generate_district_data(
                    state, location, request.risk_level
                )
                
                if risk_level == RiskLevel.HIGH or risk_level == RiskLevel.CRITICAL:
                    high_risk_count += 1
                
                # Apply risk level filter if specified
                if request.risk_level and risk_level != request.risk_level:
                    continue
                
                heatmap_data.append(HeatmapData(
                    state=state,
                    district=location["district"],
                    coordinates={"lat": location["lat"], "lng": location["lng"]},
                    risk_level=risk_level,
                    active_cases=cases,
                    citizen_reports=reports,
                    predicted_cases=predicted,
                    last_updated=datetime.now()
                ))
        
        return HeatmapResponse(
            data=heatmap_data,
            total_states=len(states_to_process),
            high_risk_areas=high_risk_count,
            last_updated=datetime.now()
        )
    
    def _generate_district_data(self, state: str, location: dict, risk_filter: Optional[RiskLevel]) -> tuple:
        """Generate realistic risk data for a district"""
        
        # Base risk calculation based on population density and random factors
        population = location.get("population", 100000)
        
        # Higher population generally means higher risk
        population_factor = min(1.0, population / 500000)
        
        # Add seasonal and random factors
        seasonal_factor = random.uniform(0.6, 1.4)  # Monsoon season variation
        random_factor = random.uniform(0.7, 1.3)
        
        # Calculate risk score
        risk_score = population_factor * seasonal_factor * random_factor
        
        # Determine risk level
        if risk_score > 1.2:
            risk_level = RiskLevel.CRITICAL
        elif risk_score > 0.9:
            risk_level = RiskLevel.HIGH
        elif risk_score > 0.6:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Generate case numbers based on risk and population
        base_cases = int((population / 10000) * risk_score)
        active_cases = max(1, base_cases + random.randint(-5, 15))
        
        # Generate citizen reports (correlated with cases)
        reports = max(0, int(active_cases * 0.3) + random.randint(-2, 5))
        
        # Predicted cases (usually higher than current)
        predicted_cases = int(active_cases * random.uniform(1.1, 1.5))
        
        return risk_level, active_cases, reports, predicted_cases

heatmap_service = HeatmapService()

@router.post("/heatmap", response_model=HeatmapResponse)
async def get_dengue_heatmap(request: HeatmapRequest):
    """
    Generate dengue risk heatmap data for Malaysian districts
    
    Returns real-time risk levels, active cases, and predictions
    for visualization on maps
    """
    try:
        heatmap = heatmap_service.generate_heatmap(request)
        return heatmap
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heatmap generation error: {str(e)}"
        )

@router.get("/heatmap/states")
async def get_state_summary():
    """
    Get summary statistics for all Malaysian states
    """
    try:
        state_data = []
        
        for state, locations in heatmap_service.locations_data.items():
            total_cases = 0
            total_reports = 0
            high_risk_districts = 0
            
            for location in locations:
                risk_level, cases, reports, _ = heatmap_service._generate_district_data(state, location, None)
                total_cases += cases
                total_reports += reports
                
                if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                    high_risk_districts += 1
            
            # Determine overall state risk
            if high_risk_districts > len(locations) * 0.5:
                state_risk = "High"
            elif high_risk_districts > 0:
                state_risk = "Medium"
            else:
                state_risk = "Low"
            
            state_data.append({
                "state": state,
                "districts": len(locations),
                "total_cases": total_cases,
                "total_reports": total_reports,
                "high_risk_districts": high_risk_districts,
                "risk_level": state_risk,
                "population": sum(loc.get("population", 100000) for loc in locations)
            })
        
        return {
            "states": state_data,
            "last_updated": datetime.now().isoformat(),
            "total_states": len(state_data)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"State summary error: {str(e)}"
        )

@router.get("/heatmap/hotspots")
async def get_current_hotspots(limit: int = 10):
    """
    Get current dengue hotspots (highest risk areas)
    """
    try:
        hotspots = []
        
        # Generate hotspot data from all locations
        for state, locations in heatmap_service.locations_data.items():
            for location in locations:
                risk_level, cases, reports, predicted = heatmap_service._generate_district_data(state, location, None)
                
                hotspots.append({
                    "location": f"{location['district']}, {state}",
                    "state": state,
                    "district": location["district"],
                    "coordinates": {"lat": location["lat"], "lng": location["lng"]},
                    "risk_level": risk_level.value,
                    "active_cases": cases,
                    "citizen_reports": reports,
                    "predicted_cases": predicted,
                    "risk_score": cases + (reports * 2) + (1 if risk_level == RiskLevel.CRITICAL else 0.5 if risk_level == RiskLevel.HIGH else 0)
                })
        
        # Sort by risk score and limit results
        hotspots.sort(key=lambda x: x["risk_score"], reverse=True)
        top_hotspots = hotspots[:limit]
        
        return {
            "hotspots": top_hotspots,
            "total_analyzed": len(hotspots),
            "critical_areas": len([h for h in hotspots if h["risk_level"] == "Critical"]),
            "high_risk_areas": len([h for h in hotspots if h["risk_level"] == "High"]),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hotspots retrieval error: {str(e)}"
        )

@router.get("/heatmap/trends/{state}")
async def get_state_trends(state: str, days: int = 7):
    """
    Get historical trend data for a specific state
    """
    try:
        if state not in heatmap_service.locations_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"State '{state}' not found"
            )
        
        trends = []
        base_date = datetime.now() - timedelta(days=days)
        
        for i in range(days):
            date = base_date + timedelta(days=i)
            
            # Generate trend data (simulated)
            total_cases = random.randint(50, 200)
            new_cases = random.randint(5, 25)
            
            trends.append({
                "date": date.strftime("%Y-%m-%d"),
                "total_cases": total_cases,
                "new_cases": new_cases,
                "active_alerts": random.randint(0, 3),
                "citizen_reports": random.randint(2, 15)
            })
        
        return {
            "state": state,
            "period_days": days,
            "trends": trends,
            "summary": {
                "avg_daily_cases": sum(t["new_cases"] for t in trends) / len(trends),
                "total_cases": sum(t["new_cases"] for t in trends),
                "trend_direction": "increasing" if trends[-1]["new_cases"] > trends[0]["new_cases"] else "decreasing"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trends retrieval error: {str(e)}"
        )