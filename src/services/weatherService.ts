/**
 * Weather Service - Integrates with multiple weather APIs
 *
 * APIs Used:
 * 1. WeatherStack API - For current weather conditions
 * 2. Malaysia Government APIs - For local forecasts and warnings
 */

// Weather API Configuration
const WEATHERSTACK_API_KEY = import.meta.env.VITE_WEATHERSTACK_API_KEY || "";
const WEATHERSTACK_BASE_URL = "http://api.weatherstack.com";
const MALAYSIA_API_BASE_URL = "https://api.data.gov.my";

// Types
export interface WeatherData {
  temperature: string;
  humidity: string;
  rainfall: string;
  riskFactor: "Low" | "Medium" | "High" | "Critical";
  location: string;
  lastUpdated: string;
}

export interface WeatherStackResponse {
  current: {
    temperature: number;
    humidity: number;
    precip: number;
    weather_descriptions: string[];
    feelslike: number;
    uv_index: number;
    visibility: number;
  };
  location: {
    name: string;
    country: string;
    region: string;
  };
}

export interface MalaysiaWeatherForecast {
  data: Array<{
    date: string;
    location: string;
    temperature_max: number;
    temperature_min: number;
    humidity: number;
    rainfall: number;
    weather_desc: string;
  }>;
}

class WeatherService {
  private cache: { [key: string]: { data: unknown; timestamp: number } } = {};
  private cacheTimeout = 10 * 60 * 1000; // 10 minutes cache

  /**
   * Get weather data from WeatherStack API
   */
  async getWeatherStackData(
    location: string = "Kuala Lumpur"
  ): Promise<WeatherStackResponse | null> {
    // Check if API key is configured and valid
    if (!WEATHERSTACK_API_KEY || WEATHERSTACK_API_KEY.length < 10) {
      console.log("⚠️ WeatherStack API key not configured - using DEMO mode");
      return this.getDemoWeatherData(location);
    }

    const cacheKey = `weatherstack_${location}`;

    // Check cache
    if (
      this.cache[cacheKey] &&
      Date.now() - this.cache[cacheKey].timestamp < this.cacheTimeout
    ) {
      return this.cache[cacheKey].data as WeatherStackResponse;
    }

    try {
      const response = await fetch(
        `${WEATHERSTACK_BASE_URL}/current?access_key=${WEATHERSTACK_API_KEY}&query=${encodeURIComponent(
          location
        )}`
      );

      if (!response.ok) {
        console.warn(`WeatherStack API error: ${response.status} - switching to DEMO mode`);
        return this.getDemoWeatherData(location);
      }

      const data = await response.json();

      // Check if the API returned an error (e.g., invalid API key)
      if (data.error) {
        console.warn('WeatherStack API error - switching to DEMO mode:', data.error.info || data.error.type);
        return this.getDemoWeatherData(location);
      }

      // Cache the result
      this.cache[cacheKey] = { data, timestamp: Date.now() };

      return data as WeatherStackResponse;
    } catch (error) {
      console.warn("WeatherStack API error - using DEMO mode:", error);
      return this.getDemoWeatherData(location);
    }
  }

  /**
   * Generate realistic demo weather data for Malaysian locations
   */
  private getDemoWeatherData(location: string = "Kuala Lumpur"): WeatherStackResponse {
    // Different conditions for different Malaysian cities
    const demoData: { [key: string]: any } = {
      "Kuala Lumpur": { temp: 31, humidity: 78, precip: 5, desc: "Partly cloudy" },
      "Selangor": { temp: 32, humidity: 80, precip: 8, desc: "Scattered showers" },
      "Johor": { temp: 30, humidity: 75, precip: 3, desc: "Sunny" },
      "Penang": { temp: 29, humidity: 82, precip: 12, desc: "Thunderstorms" },
      "Perak": { temp: 28, humidity: 73, precip: 2, desc: "Clear" },
      "Sabah": { temp: 27, humidity: 85, precip: 15, desc: "Heavy rain" },
      "Sarawak": { temp: 28, humidity: 84, precip: 10, desc: "Rain" },
    };

    const cityData = demoData[location] || demoData["Kuala Lumpur"];
    
    // Add some randomness to make it more realistic
    const tempVariation = (Math.random() - 0.5) * 4;
    const humidityVariation = (Math.random() - 0.5) * 10;
    
    return {
      current: {
        temperature: Math.round(cityData.temp + tempVariation),
        humidity: Math.round(Math.max(60, Math.min(95, cityData.humidity + humidityVariation))),
        precip: cityData.precip + Math.round((Math.random() - 0.5) * 5),
        weather_descriptions: [cityData.desc],
        feelslike: Math.round(cityData.temp + tempVariation + 2),
        uv_index: 7,
        visibility: 10,
      },
      location: {
        name: location,
        country: "Malaysia",
        region: location,
      },
    };
  }

  /**
   * Get Malaysia government weather forecast
   */
  async getMalaysiaWeatherForecast(): Promise<MalaysiaWeatherForecast | null> {
    const cacheKey = "malaysia_forecast";

    // Check cache
    if (
      this.cache[cacheKey] &&
      Date.now() - this.cache[cacheKey].timestamp < this.cacheTimeout
    ) {
      return this.cache[cacheKey].data as MalaysiaWeatherForecast;
    }

    try {
      const response = await fetch(`${MALAYSIA_API_BASE_URL}/weather/forecast`);

      if (!response.ok) {
        throw new Error(`Malaysia API error: ${response.status}`);
      }

      const data: MalaysiaWeatherForecast = await response.json();

      // Cache the result
      this.cache[cacheKey] = { data, timestamp: Date.now() };

      return data;
    } catch (error) {
      console.error("Malaysia Weather API error:", error);
      return null;
    }
  }

  /**
   * Calculate breeding risk based on weather conditions
   */
  calculateBreedingRisk(
    temperature: number,
    humidity: number,
    rainfall: number
  ): "Low" | "Medium" | "High" | "Critical" {
    // Dengue breeding conditions:
    // - Temperature: 25-30°C is optimal for mosquito breeding
    // - Humidity: Above 60% increases breeding risk
    // - Rainfall: Recent rain creates breeding sites, but too much can wash away larvae

    let riskScore = 0;

    // Temperature scoring (optimal range 25-30°C)
    if (temperature >= 25 && temperature <= 30) {
      riskScore += 3; // High risk
    } else if (temperature >= 20 && temperature < 25) {
      riskScore += 2; // Medium risk
    } else if (temperature > 30 && temperature <= 35) {
      riskScore += 2; // Still risky but less optimal
    } else {
      riskScore += 1; // Lower risk
    }

    // Humidity scoring
    if (humidity >= 80) {
      riskScore += 3; // Very high humidity
    } else if (humidity >= 60) {
      riskScore += 2; // High humidity
    } else if (humidity >= 40) {
      riskScore += 1; // Moderate humidity
    }
    // Below 40% humidity gets no additional risk

    // Rainfall scoring (past 24h)
    if (rainfall > 10 && rainfall < 50) {
      riskScore += 3; // Optimal breeding conditions
    } else if (rainfall > 5 && rainfall <= 10) {
      riskScore += 2; // Good breeding conditions
    } else if (rainfall >= 50) {
      riskScore += 1; // Too much rain, some breeding sites washed away
    }
    // No rain gets no additional risk

    // Convert score to risk level
    if (riskScore >= 7) {
      return "Critical";
    } else if (riskScore >= 5) {
      return "High";
    } else if (riskScore >= 3) {
      return "Medium";
    } else {
      return "Low";
    }
  }

  /**
   * Get comprehensive weather data with breeding risk assessment
   */
  async getWeatherWithRisk(
    location: string = "Kuala Lumpur"
  ): Promise<WeatherData> {
    try {
      // Try to get WeatherStack data first (more detailed)
      const weatherStackData = await this.getWeatherStackData(location);

      if (weatherStackData) {
        const temperature = weatherStackData.current.temperature;
        const humidity = weatherStackData.current.humidity;
        const rainfall = weatherStackData.current.precip || 0;

        const riskFactor = this.calculateBreedingRisk(
          temperature,
          humidity,
          rainfall
        );

        return {
          temperature: `${temperature}°C`,
          humidity: `${humidity}%`,
          rainfall: `${rainfall}mm`,
          riskFactor,
          location: `${weatherStackData.location.name}, ${weatherStackData.location.region}`,
          lastUpdated: new Date().toLocaleTimeString(),
        };
      }

      // Fallback to Malaysia API
      const malaysiaData = await this.getMalaysiaWeatherForecast();
      if (malaysiaData && malaysiaData.data && malaysiaData.data.length > 0) {
        const latestData = malaysiaData.data[0];
        const avgTemp =
          (latestData.temperature_max + latestData.temperature_min) / 2;
        const humidity = latestData.humidity;
        const rainfall = latestData.rainfall;

        const riskFactor = this.calculateBreedingRisk(
          avgTemp,
          humidity,
          rainfall
        );

        return {
          temperature: `${Math.round(avgTemp)}°C`,
          humidity: `${humidity}%`,
          rainfall: `${rainfall}mm`,
          riskFactor,
          location: latestData.location,
          lastUpdated: new Date().toLocaleTimeString(),
        };
      }

      // Final fallback with reasonable estimates for Malaysia
      return this.getFallbackWeatherData();
    } catch (error) {
      console.error("Weather service error:", error);
      return this.getFallbackWeatherData();
    }
  }

  /**
   * Fallback weather data based on typical Malaysian conditions
   */
  private getFallbackWeatherData(): WeatherData {
    // Malaysian typical weather patterns
    const temperature = 28 + Math.random() * 6; // 28-34°C
    const humidity = 65 + Math.random() * 20; // 65-85%
    const rainfall = Math.random() * 20; // 0-20mm

    const riskFactor = this.calculateBreedingRisk(
      temperature,
      humidity,
      rainfall
    );

    return {
      temperature: `${Math.round(temperature)}°C`,
      humidity: `${Math.round(humidity)}%`,
      rainfall: `${Math.round(rainfall)}mm`,
      riskFactor,
      location: "Malaysia (Demo)",
      lastUpdated: new Date().toLocaleTimeString() + " (Demo Data)",
    };
  }

  /**
   * Get weather warnings from Malaysia API
   */
  async getWeatherWarnings() {
    try {
      const response = await fetch(`${MALAYSIA_API_BASE_URL}/weather/warning`);
      if (response.ok) {
        return await response.json();
      }
    } catch (error) {
      console.error("Weather warnings API error:", error);
    }
    return null;
  }
}

// Export singleton instance
export const weatherService = new WeatherService();
export default weatherService;
