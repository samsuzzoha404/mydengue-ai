"""
Weather Data Service - OpenWeatherMap API Integration
Provides real-time weather data for dengue outbreak prediction
"""

import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from app.utils.config import settings

logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Current weather data structure"""
    location: str
    country: str
    latitude: float
    longitude: float
    temperature: float  # Celsius
    feels_like: float
    humidity: int  # Percentage
    pressure: int  # hPa
    wind_speed: float  # m/s
    wind_direction: int  # degrees
    cloudiness: int  # Percentage
    rainfall: float  # mm (last 3h)
    weather_main: str  # Rain, Clouds, Clear, etc.
    weather_description: str
    timestamp: datetime
    
@dataclass
class ForecastData:
    """Daily forecast data structure"""
    date: str
    temp_day: float
    temp_min: float
    temp_max: float
    humidity: int
    pressure: int
    wind_speed: float
    rainfall: float
    weather_main: str
    weather_description: str
    breeding_risk_score: float  # Calculated dengue breeding risk

class OpenWeatherMapService:
    """OpenWeatherMap API integration service"""
    
    def __init__(self):
        self.api_key = settings.openweathermap_api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.session = None
        
        # Malaysia major cities coordinates for demo
        self.malaysia_cities = {
            "Kuala Lumpur": (3.139, 101.6869),
            "George Town": (5.4164, 100.3327),
            "Ipoh": (4.5975, 101.0901),
            "Shah Alam": (3.0733, 101.5185),
            "Petaling Jaya": (3.1073, 101.6067),
            "Johor Bahru": (1.4927, 103.7414),
            "Melaka": (2.1896, 102.2501),
            "Alor Setar": (6.1248, 100.3678),
            "Miri": (4.4148, 113.9919),
            "Kota Kinabalu": (5.9804, 116.0735)
        }
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close the aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def get_current_weather(self, city: str, country: str = "MY") -> Optional[WeatherData]:
        """
        Get current weather data for a specific city
        Uses OpenWeatherMap Current Weather API
        """
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured, using demo data")
            return self._get_demo_weather_data(city)
        
        try:
            session = await self.get_session()
            url = f"{self.base_url}/weather"
            params = {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "units": "metric"  # Celsius, m/s
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_current_weather(data)
                elif response.status == 401:
                    logger.error("Invalid OpenWeatherMap API key")
                    return self._get_demo_weather_data(city)
                else:
                    logger.error(f"OpenWeatherMap API error: {response.status}")
                    return self._get_demo_weather_data(city)
                    
        except asyncio.TimeoutError:
            logger.error("OpenWeatherMap API timeout")
            return self._get_demo_weather_data(city)
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            return self._get_demo_weather_data(city)
    
    async def get_weather_by_coordinates(self, lat: float, lon: float) -> Optional[WeatherData]:
        """
        Get current weather data by geographic coordinates
        More accurate than city name queries
        """
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured, using demo data")
            return self._get_demo_weather_data_by_coords(lat, lon)
        
        try:
            session = await self.get_session()
            url = f"{self.base_url}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric"
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_current_weather(data)
                else:
                    logger.error(f"OpenWeatherMap API error: {response.status}")
                    return self._get_demo_weather_data_by_coords(lat, lon)
                    
        except Exception as e:
            logger.error(f"Error fetching weather data by coordinates: {str(e)}")
            return self._get_demo_weather_data_by_coords(lat, lon)
    
    async def get_16_day_forecast(self, city: str, country: str = "MY") -> List[ForecastData]:
        """
        Get 16-day daily forecast data
        Uses OpenWeatherMap 16 Day Daily Forecast API
        """
        if not self.api_key:
            logger.warning("OpenWeatherMap API key not configured, using demo data")
            return self._get_demo_forecast_data(city)
        
        try:
            # Get coordinates first
            coords = await self._get_city_coordinates(city, country)
            if not coords:
                return self._get_demo_forecast_data(city)
            
            lat, lon = coords
            session = await self.get_session()
            url = f"{self.base_url}/forecast/daily"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": self.api_key,
                "units": "metric",
                "cnt": 16  # 16 days
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_forecast_data(data)
                else:
                    logger.error(f"OpenWeatherMap Forecast API error: {response.status}")
                    return self._get_demo_forecast_data(city)
                    
        except Exception as e:
            logger.error(f"Error fetching forecast data: {str(e)}")
            return self._get_demo_forecast_data(city)
    
    async def _get_city_coordinates(self, city: str, country: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a city using Geocoding API"""
        if city in self.malaysia_cities:
            return self.malaysia_cities[city]
        
        try:
            session = await self.get_session()
            url = "http://api.openweathermap.org/geo/1.0/direct"
            params = {
                "q": f"{city},{country}",
                "appid": self.api_key,
                "limit": 1
            }
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data:
                        return (data[0]["lat"], data[0]["lon"])
                        
        except Exception as e:
            logger.error(f"Error getting city coordinates: {str(e)}")
        
        return None
    
    def _parse_current_weather(self, data: Dict) -> WeatherData:
        """Parse OpenWeatherMap current weather API response"""
        return WeatherData(
            location=data["name"],
            country=data["sys"]["country"],
            latitude=data["coord"]["lat"],
            longitude=data["coord"]["lon"],
            temperature=data["main"]["temp"],
            feels_like=data["main"]["feels_like"],
            humidity=data["main"]["humidity"],
            pressure=data["main"]["pressure"],
            wind_speed=data.get("wind", {}).get("speed", 0),
            wind_direction=data.get("wind", {}).get("deg", 0),
            cloudiness=data.get("clouds", {}).get("all", 0),
            rainfall=data.get("rain", {}).get("3h", 0),  # 3-hour rainfall
            weather_main=data["weather"][0]["main"],
            weather_description=data["weather"][0]["description"],
            timestamp=datetime.now()
        )
    
    def _parse_forecast_data(self, data: Dict) -> List[ForecastData]:
        """Parse OpenWeatherMap 16-day forecast API response"""
        forecasts = []
        
        for item in data["list"]:
            date_str = datetime.fromtimestamp(item["dt"]).strftime("%Y-%m-%d")
            
            # Calculate breeding risk score based on weather conditions
            breeding_risk = self._calculate_breeding_risk_score(
                temperature=item["temp"]["day"],
                humidity=item["humidity"],
                rainfall=item.get("rain", 0)
            )
            
            forecast = ForecastData(
                date=date_str,
                temp_day=item["temp"]["day"],
                temp_min=item["temp"]["min"],
                temp_max=item["temp"]["max"],
                humidity=item["humidity"],
                pressure=item["pressure"],
                wind_speed=item["speed"],
                rainfall=item.get("rain", 0),
                weather_main=item["weather"][0]["main"],
                weather_description=item["weather"][0]["description"],
                breeding_risk_score=breeding_risk
            )
            forecasts.append(forecast)
        
        return forecasts
    
    def _calculate_breeding_risk_score(self, temperature: float, humidity: int, rainfall: float) -> float:
        """
        Calculate dengue breeding risk score based on weather conditions
        Returns score from 0.0 (low risk) to 1.0 (high risk)
        """
        # Optimal conditions for Aedes mosquito breeding:
        # Temperature: 25-30°C (optimal ~28°C)
        # Humidity: 70-90% (optimal ~80%)
        # Rainfall: Moderate (creates stagnant water but not too much flow)
        
        # Temperature factor
        if 25 <= temperature <= 30:
            temp_factor = 0.9 if 27 <= temperature <= 29 else 0.7
        elif 20 <= temperature <= 35:
            temp_factor = 0.5
        else:
            temp_factor = 0.2
        
        # Humidity factor
        if 70 <= humidity <= 90:
            humidity_factor = 0.9 if 75 <= humidity <= 85 else 0.7
        elif 60 <= humidity <= 95:
            humidity_factor = 0.5
        else:
            humidity_factor = 0.3
        
        # Rainfall factor (moderate rain creates breeding sites)
        if 5 <= rainfall <= 20:  # mm per day
            rain_factor = 0.8
        elif 1 <= rainfall <= 30:
            rain_factor = 0.6
        elif rainfall > 30:
            rain_factor = 0.4  # Too much rain washes away breeding sites
        else:
            rain_factor = 0.3  # No rain, fewer new breeding sites
        
        # Combined risk score
        risk_score = (temp_factor * 0.4 + humidity_factor * 0.4 + rain_factor * 0.2)
        return min(1.0, max(0.0, risk_score))
    
    def _get_demo_weather_data(self, city: str) -> WeatherData:
        """Generate realistic demo weather data for Malaysian cities"""
        import random
        
        # Get coordinates if available
        coords = self.malaysia_cities.get(city, (3.139, 101.6869))  # Default to KL
        
        return WeatherData(
            location=city,
            country="MY",
            latitude=coords[0],
            longitude=coords[1],
            temperature=random.uniform(26, 34),  # Typical Malaysian temperature
            feels_like=random.uniform(28, 38),
            humidity=random.randint(70, 95),    # High humidity typical for Malaysia
            pressure=random.randint(1008, 1015),
            wind_speed=random.uniform(2, 8),
            wind_direction=random.randint(0, 360),
            cloudiness=random.randint(20, 90),
            rainfall=random.uniform(0, 15),
            weather_main=random.choice(["Rain", "Clouds", "Clear"]),
            weather_description=random.choice([
                "scattered clouds", "broken clouds", "light rain", 
                "moderate rain", "clear sky", "overcast clouds"
            ]),
            timestamp=datetime.now()
        )
    
    def _get_demo_weather_data_by_coords(self, lat: float, lon: float) -> WeatherData:
        """Generate demo weather data for specific coordinates"""
        import random
        
        # Find closest city name
        closest_city = "Unknown Location"
        min_distance = float('inf')
        for city, (city_lat, city_lon) in self.malaysia_cities.items():
            distance = ((lat - city_lat) ** 2 + (lon - city_lon) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_city = city
        
        return WeatherData(
            location=closest_city,
            country="MY",
            latitude=lat,
            longitude=lon,
            temperature=random.uniform(26, 34),
            feels_like=random.uniform(28, 38),
            humidity=random.randint(70, 95),
            pressure=random.randint(1008, 1015),
            wind_speed=random.uniform(2, 8),
            wind_direction=random.randint(0, 360),
            cloudiness=random.randint(20, 90),
            rainfall=random.uniform(0, 15),
            weather_main=random.choice(["Rain", "Clouds", "Clear"]),
            weather_description=random.choice([
                "scattered clouds", "broken clouds", "light rain", 
                "moderate rain", "clear sky", "overcast clouds"
            ]),
            timestamp=datetime.now()
        )
    
    def _get_demo_forecast_data(self, city: str) -> List[ForecastData]:
        """Generate realistic demo 16-day forecast data"""
        import random
        
        forecasts = []
        base_temp = random.uniform(28, 32)
        
        for i in range(16):
            date = datetime.now() + timedelta(days=i)
            temp_day = base_temp + random.uniform(-3, 3)
            humidity = random.randint(70, 95)
            rainfall = random.uniform(0, 20) if random.random() > 0.3 else 0
            
            forecast = ForecastData(
                date=date.strftime("%Y-%m-%d"),
                temp_day=temp_day,
                temp_min=temp_day - random.uniform(2, 5),
                temp_max=temp_day + random.uniform(2, 5),
                humidity=humidity,
                pressure=random.randint(1008, 1015),
                wind_speed=random.uniform(2, 8),
                rainfall=rainfall,
                weather_main=random.choice(["Rain", "Clouds", "Clear"]),
                weather_description=random.choice([
                    "scattered clouds", "broken clouds", "light rain", 
                    "moderate rain", "clear sky", "overcast clouds"
                ]),
                breeding_risk_score=self._calculate_breeding_risk_score(temp_day, humidity, rainfall)
            )
            forecasts.append(forecast)
        
        return forecasts

# Global weather service instance
weather_service = OpenWeatherMapService()