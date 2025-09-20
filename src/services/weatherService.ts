/**
 * Weather Service - Integrates with multiple weather APIs
 *
 * APIs Used:
 * 1. WeatherStack API - For current weather conditions
 * 2. Malaysia Government APIs - For local forecasts and warnings
 */

// Weather API Configuration
const WEATHERSTACK_API_KEY = "bc84b5b2cc1bd78082aa639929e9c533";
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
        throw new Error(`WeatherStack API error: ${response.status}`);
      }

      const data: WeatherStackResponse = await response.json();

      // Cache the result
      this.cache[cacheKey] = { data, timestamp: Date.now() };

      return data;
    } catch (error) {
      console.error("WeatherStack API error:", error);
      return null;
    }
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
      if (malaysiaData && malaysiaData.data.length > 0) {
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
      location: "Malaysia",
      lastUpdated: new Date().toLocaleTimeString() + " (Estimated)",
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
