"""
Advanced API routes for quantum optimization and data ecosystem features
"""

from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any
import logging

from app.services.quantum_optimization import quantum_optimizer, optimize_dengue_response
from app.services.data_ecosystem import data_ecosystem

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/api/v1/quantum/status")
async def get_quantum_status():
    """Get quantum computing integration status"""
    return quantum_optimizer.get_quantum_status()

@router.post("/api/v1/quantum/optimize-routes")
async def optimize_fogging_routes(request: Dict[str, Any]):
    """
    Quantum optimization for fogging team routes
    
    Expected request format:
    {
        "districts": [
            {"name": "District1", "latitude": 3.1390, "longitude": 101.6869, "risk_level": 0.8},
            {"name": "District2", "latitude": 3.1073, "longitude": 101.6067, "risk_level": 0.6}
        ],
        "fogging_teams": 3
    }
    """
    try:
        districts = request.get("districts", [])
        fogging_teams = request.get("fogging_teams", 3)
        
        if not districts:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Districts list cannot be empty"
            )
        
        result = quantum_optimizer.optimize_fogging_routes(districts, fogging_teams)
        return {
            "status": "success",
            "optimization_result": result
        }
        
    except Exception as e:
        logger.error(f"Route optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )

@router.post("/api/v1/quantum/allocate-resources")
async def optimize_resource_allocation(request: Dict[str, Any]):
    """
    Quantum optimization for resource allocation
    
    Expected request format:
    {
        "areas": [
            {"name": "Area1", "risk_level": 0.8, "population": 50000},
            {"name": "Area2", "risk_level": 0.6, "population": 30000}
        ],
        "resources": {
            "fogging_machines": 10,
            "inspection_teams": 15,
            "awareness_kits": 100
        }
    }
    """
    try:
        areas = request.get("areas", [])
        resources = request.get("resources", {})
        
        if not areas or not resources:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Areas and resources cannot be empty"
            )
        
        result = quantum_optimizer.optimize_resource_allocation(areas, resources)
        return {
            "status": "success",
            "allocation_result": result
        }
        
    except Exception as e:
        logger.error(f"Resource allocation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resource allocation failed: {str(e)}"
        )

@router.post("/api/v1/quantum/simulate-outbreak")
async def quantum_outbreak_simulation(request: Dict[str, Any]):
    """
    Quantum simulation of dengue outbreak scenarios
    
    Expected request format:
    {
        "scenarios": [
            {"name": "Heavy_Rain", "weather_risk": 0.9, "population_density": 0.7},
            {"name": "Normal_Weather", "weather_risk": 0.4, "population_density": 0.5}
        ]
    }
    """
    try:
        scenarios = request.get("scenarios", [])
        
        if not scenarios:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Scenarios list cannot be empty"
            )
        
        result = quantum_optimizer.quantum_risk_simulation(scenarios)
        return {
            "status": "success",
            "simulation_result": result
        }
        
    except Exception as e:
        logger.error(f"Quantum simulation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quantum simulation failed: {str(e)}"
        )

@router.post("/api/v1/quantum/full-optimization")
async def full_quantum_optimization(request: Dict[str, Any]):
    """
    Complete quantum optimization including routes, resources, and risk simulation
    
    Expected request format:
    {
        "districts": [...],
        "fogging_teams": 3,
        "resources": {...},
        "scenarios": [...]
    }
    """
    try:
        districts = request.get("districts", [])
        fogging_teams = request.get("fogging_teams", 3)
        
        if not districts:
            # Use default Malaysian districts
            districts = [
                {"name": "Kuala Lumpur", "latitude": 3.1390, "longitude": 101.6869, "risk_level": 0.8},
                {"name": "Petaling Jaya", "latitude": 3.1073, "longitude": 101.6067, "risk_level": 0.7},
                {"name": "Shah Alam", "latitude": 3.0733, "longitude": 101.5185, "risk_level": 0.6},
                {"name": "Klang", "latitude": 3.0449, "longitude": 101.4457, "risk_level": 0.5},
                {"name": "Johor Bahru", "latitude": 1.4927, "longitude": 103.7414, "risk_level": 0.7}
            ]
        
        result = await optimize_dengue_response(districts, fogging_teams)
        return {
            "status": "success",
            "full_optimization": result
        }
        
    except Exception as e:
        logger.error(f"Full quantum optimization failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Full optimization failed: {str(e)}"
        )

@router.get("/api/v1/data-ecosystem/status")
async def get_data_ecosystem_status():
    """Get data ecosystem status and summary"""
    return data_ecosystem.get_data_summary()

@router.get("/api/v1/data-ecosystem/weather")
async def get_weather_data():
    """Get current weather data for all monitored cities"""
    try:
        weather_data = await data_ecosystem.collect_weather_data()
        return {
            "status": "success",
            "weather_data": [
                {
                    "location": w.location,
                    "temperature": w.temperature,
                    "humidity": w.humidity,
                    "rainfall": w.rainfall,
                    "wind_speed": w.wind_speed,
                    "timestamp": w.timestamp.isoformat()
                }
                for w in weather_data
            ]
        }
    except Exception as e:
        logger.error(f"Weather data collection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Weather data collection failed: {str(e)}"
        )

@router.get("/api/v1/data-ecosystem/hospital-data")
async def get_hospital_data():
    """Get hospital dengue case data summary"""
    try:
        hospital_df = data_ecosystem.load_hospital_datasets()
        
        # Generate summary statistics
        summary = {
            "total_cases": len(hospital_df),
            "cases_by_location": hospital_df['location'].value_counts().to_dict(),
            "severity_distribution": hospital_df['severity'].value_counts().to_dict(),
            "status_distribution": hospital_df['status'].value_counts().to_dict(),
            "age_stats": {
                "mean": float(hospital_df['age'].mean()),
                "min": int(hospital_df['age'].min()),
                "max": int(hospital_df['age'].max())
            },
            "recent_cases": len(hospital_df[hospital_df['date_reported'] >= hospital_df['date_reported'].max() - pd.Timedelta(days=30)])
        }
        
        return {
            "status": "success",
            "hospital_data_summary": summary
        }
        
    except Exception as e:
        logger.error(f"Hospital data retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hospital data retrieval failed: {str(e)}"
        )

@router.post("/api/v1/data-ecosystem/citizen-report")
async def process_citizen_report_advanced(request: Dict[str, Any]):
    """Process citizen report with advanced data ecosystem storage"""
    try:
        result = await data_ecosystem.process_citizen_reports(request)
        return {
            "status": "success",
            "processing_result": result
        }
    except Exception as e:
        logger.error(f"Advanced citizen report processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report processing failed: {str(e)}"
        )

@router.post("/api/v1/alerts/send-notification")
async def send_alert_notification(request: Dict[str, Any]):
    """Send alert notifications using Firebase Cloud Messaging"""
    try:
        message = request.get("message", "Dengue alert notification")
        locations = request.get("locations", [])
        
        await data_ecosystem.send_alert_notification(message, locations)
        
        return {
            "status": "success",
            "message": "Alert notification sent successfully"
        }
        
    except Exception as e:
        logger.error(f"Alert notification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Alert notification failed: {str(e)}"
        )

@router.get("/api/v1/system/capabilities")
async def get_system_capabilities():
    """Get comprehensive system capabilities overview"""
    
    # Check quantum computing
    quantum_status = quantum_optimizer.get_quantum_status()
    
    # Check data ecosystem
    ecosystem_status = data_ecosystem.get_data_summary()
    
    # Check AI models
    from app.services.custom_ai_service import custom_ai_service
    
    capabilities = {
        "ai_models": {
            "custom_lstm_gru": custom_ai_service.model_available,
            "advanced_cnn": custom_ai_service.advanced_cnn_available,
            "computer_vision": True,
            "outbreak_prediction": True
        },
        "quantum_computing": {
            "available": quantum_status["quantum_available"],
            "features": quantum_status["supported_features"]
        },
        "data_ecosystem": {
            "firebase_integration": ecosystem_status["firebase_initialized"],
            "weather_apis": ecosystem_status["apis_configured"],
            "data_sources": ecosystem_status["data_sources"]
        },
        "advanced_features": {
            "99_percent_accuracy_target": True,
            "real_time_processing": True,
            "cloud_integration": True,
            "quantum_optimization": quantum_status["quantum_available"],
            "professional_dashboard": True
        },
        "competition_ready": True,
        "hackathon": "D3CODE 2025"
    }
    
    return capabilities