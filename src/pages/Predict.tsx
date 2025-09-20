import { useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  MapPin,
  Calendar,
  Thermometer,
  Droplets,
  Wind,
  Shield,
  ArrowRight,
  AlertTriangle,
  Brain,
  Zap,
  Award,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import {
  advancedAiClient,
  type AdvancedOutbreakPrediction,
} from "@/lib/advanced-dengue-ai-client";

const Predict = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState({
    location: "",
    state: "",
    date: "",
    temperature: "",
    humidity: "",
    rainfall: "",
    windSpeed: "",
    description: "",
  });

  const malaysianStates = [
    "Kuala Lumpur",
    "Selangor",
    "Johor",
    "Penang",
    "Perak",
    "Kedah",
    "Kelantan",
    "Terengganu",
    "Pahang",
    "Negeri Sembilan",
    "Melaka",
    "Sabah",
    "Sarawak",
    "Perlis",
    "Putrajaya",
    "Labuan",
  ];

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.location || !formData.state) {
      toast({
        title: "Missing Information",
        description: "Please fill in location and state fields.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);

    try {
      // Prepare data for Advanced AI model
      const locationData = {
        location: `${formData.location}, ${formData.state}`,
        temperature: parseFloat(formData.temperature) || 30,
        humidity: parseFloat(formData.humidity) || 70,
        rainfall: parseFloat(formData.rainfall) || 10,
        populationDensity: 1000, // Population per sq km
        previousCases: 0, // Could be enhanced with historical data
      };

      // Call Advanced AI model for prediction
      const prediction = await advancedAiClient.predictOutbreak(locationData);

      // Transform advanced prediction to match existing results format
      const results = {
        riskLevel: prediction.prediction.risk_level,
        riskScore: Math.round(prediction.prediction.outbreak_probability * 100),
        location: formData.location,
        state: formData.state,
        factors: {
          temperature: locationData.temperature,
          humidity: locationData.humidity,
          rainfall: locationData.rainfall,
          populationDensity: locationData.populationDensity,
        },
        environmentalImpacts: prediction.environmental_factors,
        recommendations: prediction.recommendations,
        predictedCases: prediction.prediction.predicted_cases,
        realAiAnalysis: prediction.real_ai_analysis,
        quantumInsights: prediction.quantum_insights,
        timestamp: new Date().toISOString(),
      };

      // Store results in sessionStorage
      sessionStorage.setItem("predictionResults", JSON.stringify(results));

      toast({
        title: "🧠 Advanced AI Prediction Complete",
        description: `Risk Level: ${
          prediction.prediction.risk_level
        } - Outbreak probability: ${(
          prediction.prediction.outbreak_probability * 100
        ).toFixed(1)}% (Model: ${prediction.real_ai_analysis.model_used})`,
        duration: 6000,
      });

      // Navigate to results page
      navigate("/results");
    } catch (error) {
      console.error("Prediction error:", error);
      toast({
        title: "Prediction Failed",
        description: "Could not generate prediction. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getRiskRecommendations = (risk: string) => {
    switch (risk) {
      case "High":
        return [
          "Immediately remove all standing water around your property",
          "Use mosquito repellent when outdoors, especially dawn and dusk",
          "Ensure air conditioning/fans are used to reduce mosquito activity",
          "Seek immediate medical attention for fever symptoms",
          "Report any suspected breeding sites to authorities",
        ];
      case "Medium":
        return [
          "Check and empty water containers weekly",
          "Use mosquito nets while sleeping",
          "Apply repellent when in outdoor areas",
          "Monitor for fever symptoms and consult doctor if needed",
          "Keep surroundings clean and dry",
        ];
      case "Low":
        return [
          "Maintain good drainage around property",
          "Store water containers with tight-fitting lids",
          "Keep grass short and vegetation trimmed",
          "Be aware of dengue symptoms: fever, headache, body aches",
          "Support community prevention efforts",
        ];
      default:
        return [];
    }
  };

  return (
    <div className="min-h-screen bg-background py-8">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-primary/10 rounded-full">
              <Shield className="w-8 h-8 text-primary" />
            </div>
            <h1 className="text-3xl sm:text-4xl font-poppins font-bold text-foreground">
              AI Dengue Risk Prediction
            </h1>
          </div>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Get personalized dengue risk assessment for your location using
            advanced AI analysis
          </p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Prediction Form */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5 text-primary" />
                  Location & Environmental Data
                </CardTitle>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-6">
                  {/* Location Information */}
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="location">Specific Location</Label>
                      <Input
                        id="location"
                        placeholder="e.g., Petaling Jaya, Jalan Ampang"
                        value={formData.location}
                        onChange={(e) =>
                          handleInputChange("location", e.target.value)
                        }
                        className="w-full"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="state">State</Label>
                      <Select
                        value={formData.state}
                        onValueChange={(value) =>
                          handleInputChange("state", value)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select state" />
                        </SelectTrigger>
                        <SelectContent>
                          {malaysianStates.map((state) => (
                            <SelectItem key={state} value={state}>
                              {state}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  {/* Date */}
                  <div className="space-y-2">
                    <Label htmlFor="date" className="flex items-center gap-2">
                      <Calendar className="w-4 h-4" />
                      Prediction Date
                    </Label>
                    <Input
                      id="date"
                      type="date"
                      value={formData.date}
                      onChange={(e) =>
                        handleInputChange("date", e.target.value)
                      }
                      min={new Date().toISOString().split("T")[0]}
                    />
                  </div>

                  {/* Environmental Conditions */}
                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold text-foreground flex items-center gap-2">
                      <Thermometer className="w-5 h-5" />
                      Environmental Conditions
                    </h3>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="temperature">Temperature (°C)</Label>
                        <Input
                          id="temperature"
                          type="number"
                          placeholder="30"
                          value={formData.temperature}
                          onChange={(e) =>
                            handleInputChange("temperature", e.target.value)
                          }
                          min="15"
                          max="45"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label
                          htmlFor="humidity"
                          className="flex items-center gap-2"
                        >
                          <Droplets className="w-4 h-4" />
                          Humidity (%)
                        </Label>
                        <Input
                          id="humidity"
                          type="number"
                          placeholder="75"
                          value={formData.humidity}
                          onChange={(e) =>
                            handleInputChange("humidity", e.target.value)
                          }
                          min="0"
                          max="100"
                        />
                      </div>
                    </div>

                    <div className="grid md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="rainfall">Recent Rainfall (mm)</Label>
                        <Input
                          id="rainfall"
                          type="number"
                          placeholder="15"
                          value={formData.rainfall}
                          onChange={(e) =>
                            handleInputChange("rainfall", e.target.value)
                          }
                          min="0"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label
                          htmlFor="windSpeed"
                          className="flex items-center gap-2"
                        >
                          <Wind className="w-4 h-4" />
                          Wind Speed (km/h)
                        </Label>
                        <Input
                          id="windSpeed"
                          type="number"
                          placeholder="5"
                          value={formData.windSpeed}
                          onChange={(e) =>
                            handleInputChange("windSpeed", e.target.value)
                          }
                          min="0"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Additional Information */}
                  <div className="space-y-2">
                    <Label htmlFor="description">
                      Additional Context (Optional)
                    </Label>
                    <Textarea
                      id="description"
                      placeholder="Describe any specific environmental conditions, recent construction, or other relevant factors..."
                      value={formData.description}
                      onChange={(e) =>
                        handleInputChange("description", e.target.value)
                      }
                      rows={3}
                    />
                  </div>

                  {/* Submit Button */}
                  <Button
                    type="submit"
                    className="w-full bg-gradient-to-r from-primary to-primary/90 text-white font-semibold py-3 px-6 hover:opacity-90 transition-all"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        AI Analyzing...
                      </div>
                    ) : (
                      <div className="flex items-center gap-2">
                        <Brain className="w-5 h-5" />
                        Predict with AI
                        <ArrowRight className="w-4 h-4" />
                      </div>
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>

          {/* Information Panel */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">How It Works</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-bold">
                    1
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">Input Data</h4>
                    <p className="text-sm text-muted-foreground">
                      Provide location and environmental conditions
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-bold">
                    2
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">AI Analysis</h4>
                    <p className="text-sm text-muted-foreground">
                      Advanced algorithms analyze multiple risk factors
                    </p>
                  </div>
                </div>
                <div className="flex items-start gap-3">
                  <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center text-sm font-bold">
                    3
                  </div>
                  <div>
                    <h4 className="font-medium text-foreground">Get Results</h4>
                    <p className="text-sm text-muted-foreground">
                      Receive personalized risk assessment and recommendations
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-warning" />
                  Risk Factors
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="space-y-2">
                  <Badge variant="secondary" className="w-full justify-start">
                    Temperature: 26-30°C optimal for breeding
                  </Badge>
                  <Badge variant="secondary" className="w-full justify-start">
                    Humidity: Over 60% increases mosquito survival
                  </Badge>
                  <Badge variant="secondary" className="w-full justify-start">
                    Rainfall: Creates standing water breeding sites
                  </Badge>
                  <Badge variant="secondary" className="w-full justify-start">
                    Wind: Low speeds allow mosquito movement
                  </Badge>
                </div>
              </CardContent>
            </Card>

            <Card className="bg-success/10 border-success/20">
              <CardContent className="pt-6">
                <div className="text-center">
                  <Shield className="w-8 h-8 text-success mx-auto mb-2" />
                  <h4 className="font-semibold text-foreground mb-2">
                    85% Accuracy
                  </h4>
                  <p className="text-sm text-muted-foreground">
                    Our AI model has been trained on historical dengue outbreak
                    data and weather patterns across Malaysia
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Predict;
