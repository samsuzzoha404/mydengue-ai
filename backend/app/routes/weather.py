"""
Weather API Routes - Real-time weather data endpoints
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from app.services.weather_service import weather_service, WeatherData, ForecastData
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/current/{city}")
async def get_current_weather(
    city: str,
    country: str = Query("MY", description="Country code (default: MY for Malaysia)")
) -> WeatherData:
    """
    Get current weather data for a specific city
    
    - **city**: City name (e.g., "Kuala Lumpur", "George Town")
    - **country**: Country code (default: "MY" for Malaysia)
    
    Returns real-time weather conditions including temperature, humidity, 
    rainfall, and calculated dengue breeding risk factors.
    """
    try:
        weather_data = await weather_service.get_current_weather(city, country)
        if not weather_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Weather data not found for {city}, {country}"
            )
        return weather_data
    except Exception as e:
        logger.error(f"Error fetching weather for {city}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")

@router.get("/coordinates")
async def get_weather_by_coordinates(
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude")
) -> WeatherData:
    """
    Get current weather data by geographic coordinates
    
    More accurate than city name queries, useful for precise location-based predictions.
    
    - **lat**: Latitude (-90 to 90)
    - **lon**: Longitude (-180 to 180)
    """
    try:
        # Validate coordinates
        if not (-90 <= lat <= 90):
            raise HTTPException(status_code=400, detail="Latitude must be between -90 and 90")
        if not (-180 <= lon <= 180):
            raise HTTPException(status_code=400, detail="Longitude must be between -180 and 180")
        
        weather_data = await weather_service.get_weather_by_coordinates(lat, lon)
        if not weather_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Weather data not found for coordinates ({lat}, {lon})"
            )
        return weather_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching weather for coordinates ({lat}, {lon}): {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")

@router.get("/forecast/{city}")
async def get_weather_forecast(
    city: str,
    country: str = Query("MY", description="Country code (default: MY for Malaysia)"),
    days: int = Query(16, description="Number of forecast days (1-16)", ge=1, le=16)
) -> List[ForecastData]:
    """
    Get weather forecast for a specific city
    
    - **city**: City name (e.g., "Kuala Lumpur", "George Town")
    - **country**: Country code (default: "MY" for Malaysia)
    - **days**: Number of forecast days (1-16)
    
    Returns daily weather forecast with calculated dengue breeding risk scores.
    """
    try:
        forecast_data = await weather_service.get_16_day_forecast(city, country)
        if not forecast_data:
            raise HTTPException(
                status_code=404, 
                detail=f"Forecast data not found for {city}, {country}"
            )
        
        # Limit to requested number of days
        return forecast_data[:days]
    except Exception as e:
        logger.error(f"Error fetching forecast for {city}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch forecast data")

@router.get("/malaysia-cities")
async def get_malaysia_weather():
    """
    Get current weather for major Malaysian cities
    
    Returns weather data for 10 major Malaysian cities commonly used 
    for dengue outbreak monitoring and prediction.
    """
    try:
        cities = [
            "Kuala Lumpur", "George Town", "Ipoh", "Shah Alam", "Petaling Jaya",
            "Johor Bahru", "Melaka", "Alor Setar", "Miri", "Kota Kinabalu"
        ]
        
        weather_data = []
        for city in cities:
            try:
                data = await weather_service.get_current_weather(city, "MY")
                if data:
                    weather_data.append(data)
            except Exception as e:
                logger.warning(f"Failed to fetch weather for {city}: {str(e)}")
                continue
        
        if not weather_data:
            raise HTTPException(
                status_code=500, 
                detail="Failed to fetch weather data for Malaysian cities"
            )
        
        return weather_data
    except Exception as e:
        logger.error(f"Error fetching Malaysia cities weather: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")

@router.get("/breeding-risk/{city}")
async def get_breeding_risk_forecast(
    city: str,
    country: str = Query("MY", description="Country code (default: MY for Malaysia)")
) -> dict:
    """
    Get dengue breeding risk forecast based on weather conditions
    
    - **city**: City name (e.g., "Kuala Lumpur", "George Town") 
    - **country**: Country code (default: "MY" for Malaysia)
    
    Returns detailed breeding risk analysis with weather-based risk scores,
    optimal breeding conditions identification, and risk trend predictions.
    """
    try:
        # Get current weather and forecast
        current_weather = await weather_service.get_current_weather(city, country)
        forecast_data = await weather_service.get_16_day_forecast(city, country)
        
        if not current_weather or not forecast_data:
            raise HTTPException(
                status_code=404,
                detail=f"Weather data not found for {city}, {country}"
            )
        
        # Calculate current breeding risk
        current_risk = weather_service._calculate_breeding_risk_score(
            current_weather.temperature,
            current_weather.humidity,
            current_weather.rainfall
        )
        
        # Analyze forecast trends
        risk_trend = []
        high_risk_days = 0
        avg_risk = 0
        
        for forecast in forecast_data:
            risk_trend.append({
                "date": forecast.date,
                "risk_score": forecast.breeding_risk_score,
                "temperature": forecast.temp_day,
                "humidity": forecast.humidity,
                "rainfall": forecast.rainfall,
                "risk_level": (
                    "High" if forecast.breeding_risk_score > 0.7 else
                    "Medium" if forecast.breeding_risk_score > 0.4 else
                    "Low"
                )
            })
            
            if forecast.breeding_risk_score > 0.7:
                high_risk_days += 1
            avg_risk += forecast.breeding_risk_score
        
        avg_risk = avg_risk / len(forecast_data) if forecast_data else 0
        
        # Generate recommendations
        recommendations = []
        if current_risk > 0.7:
            recommendations.extend([
                "High breeding risk detected - inspect and remove stagnant water",
                "Increase community surveillance for breeding sites",
                "Consider targeted vector control measures"
            ])
        elif current_risk > 0.4:
            recommendations.extend([
                "Moderate breeding risk - maintain regular site inspections",
                "Monitor rainfall and water accumulation areas"
            ])
        else:
            recommendations.append("Low breeding risk - continue routine monitoring")
        
        if high_risk_days > 5:
            recommendations.append(f"Extended high-risk period forecasted ({high_risk_days} days)")
        
        return {
            "city": city,
            "country": country,
            "current_risk": {
                "score": round(current_risk, 2),
                "level": (
                    "High" if current_risk > 0.7 else
                    "Medium" if current_risk > 0.4 else
                    "Low"
                ),
                "temperature": current_weather.temperature,
                "humidity": current_weather.humidity,
                "rainfall": current_weather.rainfall
            },
            "forecast_summary": {
                "average_risk": round(avg_risk, 2),
                "high_risk_days": high_risk_days,
                "forecast_period": f"{len(forecast_data)} days"
            },
            "risk_trend": risk_trend[:7],  # Show next 7 days
            "recommendations": recommendations,
            "last_updated": current_weather.timestamp.isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating breeding risk for {city}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to calculate breeding risk")

@router.on_event("shutdown")
async def shutdown_weather_service():
    """Clean up weather service on application shutdown"""
    await weather_service.close()