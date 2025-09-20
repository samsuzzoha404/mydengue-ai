import { useState, useEffect } from "react";
import {
  Camera,
  MapPin,
  Upload,
  CheckCircle,
  AlertCircle,
  Loader,
  Brain,
  Award,
  TrendingUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import {
  advancedAiClient,
  type AdvancedAnalysisResult,
} from "@/lib/advanced-dengue-ai-client";

const Report = () => {
  const { toast } = useToast();
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiResult, setAiResult] = useState<AdvancedAnalysisResult | null>(null);
  const [location, setLocation] = useState<string>("");
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [aiBackendStatus, setAiBackendStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");
  const [formData, setFormData] = useState({
    address: "",
    description: "",
    contact: "",
  });

  // Check AI backend status on component mount
  useEffect(() => {
    const checkAiStatus = async () => {
      try {
        const health = await advancedAiClient.healthCheck();
        setAiBackendStatus(
          health.status === "healthy" ? "connected" : "disconnected"
        );
      } catch (error) {
        setAiBackendStatus("disconnected");
      }
    };

    checkAiStatus();
  }, []);

  const handleImageUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    console.log("Upload function triggered", event);
    const file = event.target.files?.[0];
    console.log("Selected file:", file);
    if (!file) return;

    // Validate file (basic validation)
    if (!file.type.startsWith("image/")) {
      toast({
        title: "Invalid File",
        description: "Please select a valid image file",
        variant: "destructive",
      });
      return;
    }

    // Store the file for analysis
    setUploadedFile(file);

    // Create preview URL
    const reader = new FileReader();
    reader.onload = (e) => {
      setUploadedImage(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // Just show success message for upload, don't analyze yet
    toast({
      title: "📷 Image Uploaded Successfully",
      description: "Click 'Analyze with Advanced AI' to detect breeding sites",
    });
  };

  const handleAnalyzeImage = async () => {
    if (!uploadedFile) {
      toast({
        title: "No Image",
        description: "Please upload an image first",
        variant: "destructive",
      });
      return;
    }

    // Check AI backend connection first
    setIsAnalyzing(true);
    setAiResult(null);

    try {
      // Test connection to Advanced AI backend
      await advancedAiClient.healthCheck();

      toast({
        title: "🤖 Advanced AI Connected",
        description:
          "Analyzing your image with state-of-the-art EfficientNet AI...",
      });

      // Analyze with Advanced AI
      const result = await advancedAiClient.analyzeImage(
        uploadedFile,
        location
      );
      setAiResult(result);

      // Show gamification results
      if (result.gamification) {
        toast({
          title: "🎉 Rewards Earned!",
          description: `+${result.gamification.points_awarded} points, +${result.gamification.xp_gained} XP`,
        });
      }

      const confidenceLevel =
        result.classification.confidence > 0.7
          ? "High"
          : result.classification.confidence > 0.4
          ? "Medium"
          : "Low";

      toast({
        title: "🧠 Advanced AI Analysis Complete",
        description: `${confidenceLevel} confidence: ${result.classification.category}`,
        duration: 5000,
      });
    } catch (error) {
      console.error("Advanced AI Analysis error:", error);
      toast({
        title: "⚠️ Advanced AI Backend Unavailable",
        description:
          "Using mock analysis. Please start the Advanced AI backend server.",
        variant: "destructive",
        duration: 7000,
      });

      // Fallback: Show mock analysis for development
      const isHotspot = Math.random() > 0.5;
      const confidence = Math.random() * 0.6 + 0.3; // 30-90%

      const mockResult: AdvancedAnalysisResult = {
        status: "success",
        classification: {
          category: isHotspot ? "Hotspot" : "Not Hotspot",
          confidence: confidence,
          description:
            "Mock analysis - Advanced AI backend unavailable. This is demonstration data.",
          risk_level: isHotspot
            ? confidence > 0.7
              ? "Critical"
              : "High"
            : confidence > 0.6
            ? "Medium"
            : "Low",
        },
        ai_analysis: {
          efficientnet_prediction: "Mock EfficientNet prediction",
          confidence_score: Math.random() * 0.6 + 0.3,
          feature_analysis: ["Mock feature 1", "Mock feature 2"],
          recommendations: [
            "Start the Advanced AI backend server for real analysis",
            "Run: cd backend && python -m uvicorn app.main:app --reload",
            "Check advanced backend connection",
          ],
        },
        gamification: {
          points_awarded: 50,
          xp_gained: 25,
        },
        timestamp: new Date().toISOString(),
        image_id: "mock-image-id",
      };
      setAiResult(mockResult);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleGetLocation = () => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition((position) => {
        setLocation(
          `${position.coords.latitude.toFixed(
            6
          )}, ${position.coords.longitude.toFixed(6)}`
        );
      });
    }
  };

  const handleSubmit = async () => {
    if (!uploadedFile || !location) {
      toast({
        title: "Missing Information",
        description: "Please upload an image and provide location information.",
        variant: "destructive",
      });
      return;
    }

    // Here you would typically send the report to your backend
    // For now, we'll just show success
    try {
      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1000));

      toast({
        title: "Report Submitted Successfully!",
        description: "Thank you for contributing to dengue prevention efforts.",
      });

      setIsSubmitted(true);
    } catch (error) {
      toast({
        title: "Submission Failed",
        description: "Could not submit report. Please try again.",
        variant: "destructive",
      });
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  if (isSubmitted) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center p-4">
        <Card className="w-full max-w-md">
          <CardContent className="p-8 text-center">
            <CheckCircle className="w-16 h-16 text-eco-green mx-auto mb-6" />
            <h2 className="text-2xl font-poppins font-bold text-foreground mb-4">
              Report Submitted Successfully!
            </h2>
            <p className="text-muted-foreground mb-6">
              Thank you for contributing to Malaysia's dengue prevention
              efforts. Your report has been logged and will be reviewed by local
              authorities.
            </p>
            <Button onClick={() => setIsSubmitted(false)} className="w-full">
              Submit Another Report
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background p-4 sm:p-6 lg:p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-3xl sm:text-4xl font-poppins font-bold text-foreground">
              Report Breeding Site
            </h1>

            {/* AI Status Indicator */}
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  aiBackendStatus === "connected"
                    ? "bg-green-500 animate-pulse"
                    : aiBackendStatus === "disconnected"
                    ? "bg-red-500"
                    : "bg-yellow-500"
                }`}
              ></div>
              <span className="text-sm text-muted-foreground">
                AI{" "}
                {aiBackendStatus === "connected"
                  ? "Connected"
                  : aiBackendStatus === "disconnected"
                  ? "Offline"
                  : "Checking..."}
              </span>
            </div>
          </div>

          <p className="text-lg text-muted-foreground mb-4">
            Help protect your community by reporting potential dengue breeding
            sites. Every report helps prevent outbreaks.
          </p>

          {/* AI Backend Notice */}
          {aiBackendStatus === "disconnected" && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle className="w-5 h-5 text-yellow-600" />
                <span className="font-medium text-yellow-800">
                  AI Backend Offline
                </span>
              </div>
              <p className="text-yellow-700 text-sm">
                The AI analysis server is not running. Images will use
                demonstration analysis. To enable real AI:{" "}
                <code className="bg-yellow-100 px-2 py-1 rounded text-xs">
                  cd backend && python app.py
                </code>
              </p>
            </div>
          )}

          {aiBackendStatus === "connected" && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-4">
              <div className="flex items-center gap-2">
                <Brain className="w-5 h-5 text-green-600" />
                <span className="font-medium text-green-800">
                  🤖 AI Analysis Ready
                </span>
                <span className="text-green-700 text-sm">
                  Upload images for real-time breeding site detection
                </span>
              </div>
            </div>
          )}
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Image Upload */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Camera className="w-5 h-5 text-primary" />
                Photo Evidence
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-smooth">
                {uploadedImage ? (
                  <div className="space-y-4">
                    <img
                      src={uploadedImage}
                      alt="Uploaded site"
                      className="w-full h-48 object-cover rounded-lg"
                    />
                    {isAnalyzing ? (
                      <div className="space-y-3">
                        <div className="flex items-center justify-center gap-2">
                          <Loader className="w-5 h-5 animate-spin text-primary" />
                          <span className="text-sm font-medium text-primary">
                            AI Analyzing Image...
                          </span>
                        </div>
                        <div className="text-xs text-muted-foreground space-y-1">
                          <div className="flex items-center justify-center gap-1">
                            <div className="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                            <span>Processing image data</span>
                          </div>
                          <div className="flex items-center justify-center gap-1">
                            <div className="w-2 h-2 bg-primary rounded-full animate-pulse animation-delay-150"></div>
                            <span>Detecting breeding sites</span>
                          </div>
                          <div className="flex items-center justify-center gap-1">
                            <div className="w-2 h-2 bg-primary rounded-full animate-pulse animation-delay-300"></div>
                            <span>Calculating confidence score</span>
                          </div>
                        </div>
                      </div>
                    ) : aiResult ? (
                      <div className="space-y-4">
                        {/* Main Result Badge */}
                        <div className="flex justify-center">
                          <Badge
                            className={`text-sm px-4 py-2 ${
                              aiResult.classification.category === "Hotspot" ||
                              aiResult.classification.category === "Potential"
                                ? "bg-red-50 text-red-700 border-red-200 hover:bg-red-100"
                                : "bg-green-50 text-green-700 border-green-200 hover:bg-green-100"
                            }`}
                          >
                            {aiResult.classification.category === "Hotspot" ||
                            aiResult.classification.category === "Potential" ? (
                              <AlertCircle className="w-4 h-4 mr-2" />
                            ) : (
                              <CheckCircle className="w-4 h-4 mr-2" />
                            )}
                            {aiResult.classification.category}
                          </Badge>
                        </div>

                        {/* Advanced AI Features */}
                        {aiResult.gamification && (
                          <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 space-y-2">
                            <div className="flex items-center gap-2">
                              <Award className="w-5 h-5 text-yellow-600" />
                              <span className="text-sm font-medium text-gray-700">
                                Gamification Rewards:
                              </span>
                            </div>
                            <div className="flex justify-between text-sm">
                              <span>
                                Points Earned: +
                                {aiResult.gamification.points_awarded}
                              </span>
                              <span>
                                XP Gained: +{aiResult.gamification.xp_gained}
                              </span>
                            </div>
                            {aiResult.gamification.achievement_unlocked && (
                              <div className="text-sm text-purple-700 font-medium">
                                🏆 Achievement:{" "}
                                {aiResult.gamification.achievement_unlocked}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Confidence and Risk Analysis */}
                        <div className="bg-gray-50 rounded-lg p-4 space-y-3">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-700">
                              Advanced AI Confidence:
                            </span>
                            <div className="flex items-center gap-2">
                              <div className="w-24 bg-gray-200 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full transition-all duration-500 ${
                                    aiResult.classification.confidence > 0.7
                                      ? "bg-green-500"
                                      : aiResult.classification.confidence > 0.4
                                      ? "bg-yellow-500"
                                      : "bg-red-500"
                                  } ${
                                    aiResult.classification.confidence > 0.8
                                      ? "w-full"
                                      : aiResult.classification.confidence > 0.6
                                      ? "w-4/5"
                                      : aiResult.classification.confidence > 0.4
                                      ? "w-3/5"
                                      : aiResult.classification.confidence > 0.2
                                      ? "w-2/5"
                                      : "w-1/5"
                                  }`}
                                ></div>
                              </div>
                              <span className="text-sm font-bold text-gray-900">
                                {(
                                  aiResult.classification.confidence * 100
                                ).toFixed(1)}
                                %
                              </span>
                            </div>
                          </div>

                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-700">
                              Risk Level:
                            </span>
                            <Badge
                              variant="outline"
                              className={`
                              ${
                                aiResult.classification.risk_level ===
                                "Critical"
                                  ? "border-red-600 text-red-800 bg-red-50"
                                  : ""
                              }
                              ${
                                aiResult.classification.risk_level === "High"
                                  ? "border-red-500 text-red-700 bg-red-50"
                                  : ""
                              }
                              ${
                                aiResult.classification.risk_level === "Medium"
                                  ? "border-yellow-500 text-yellow-700 bg-yellow-50"
                                  : ""
                              }
                              ${
                                aiResult.classification.risk_level === "Low"
                                  ? "border-green-500 text-green-700 bg-green-50"
                                  : ""
                              }
                            `}
                            >
                              {aiResult.classification.risk_level}
                            </Badge>
                          </div>
                        </div>

                        {/* Advanced Analysis Description */}
                        <div className="text-left space-y-3">
                          <div className="p-3 bg-blue-50 rounded-lg">
                            <p className="text-sm text-gray-700 mb-2 leading-relaxed">
                              <strong>🧠 EfficientNet AI Analysis:</strong>{" "}
                              {aiResult.classification.description}
                            </p>
                            <p className="text-xs text-blue-700">
                              <strong>Model:</strong>{" "}
                              {aiResult.ai_analysis.efficientnet_prediction}
                            </p>
                          </div>

                          {/* Feature Analysis */}
                          {aiResult.ai_analysis.feature_analysis &&
                            aiResult.ai_analysis.feature_analysis.length >
                              0 && (
                              <div className="p-3 bg-purple-50 rounded-lg">
                                <p className="text-sm font-medium text-gray-700 mb-2">
                                  🔍 Feature Analysis:
                                </p>
                                <div className="flex flex-wrap gap-1">
                                  {aiResult.ai_analysis.feature_analysis
                                    .slice(0, 3)
                                    .map((feature, index) => (
                                      <Badge
                                        key={index}
                                        variant="secondary"
                                        className="text-xs"
                                      >
                                        {feature}
                                      </Badge>
                                    ))}
                                </div>
                              </div>
                            )}

                          {/* Advanced AI Recommendations */}
                          {aiResult.ai_analysis.recommendations &&
                            aiResult.ai_analysis.recommendations.length > 0 && (
                              <div>
                                <p className="text-sm font-medium text-gray-700 mb-2">
                                  💡 Advanced AI Recommendations:
                                </p>
                                <ul className="space-y-1">
                                  {aiResult.ai_analysis.recommendations
                                    .slice(0, 3)
                                    .map((rec, index) => (
                                      <li
                                        key={index}
                                        className="text-xs text-gray-600 flex items-start gap-2"
                                      >
                                        <span className="w-1 h-1 bg-primary rounded-full mt-2 flex-shrink-0"></span>
                                        <span className="leading-relaxed">
                                          {rec}
                                        </span>
                                      </li>
                                    ))}
                                </ul>
                              </div>
                            )}

                          {/* Quantum Optimization Insights */}
                          {aiResult.quantum_optimization && (
                            <div className="p-3 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg">
                              <div className="flex items-center gap-2 mb-2">
                                <TrendingUp className="w-4 h-4 text-purple-600" />
                                <span className="text-sm font-medium text-purple-700">
                                  Quantum Optimization Score:{" "}
                                  {
                                    aiResult.quantum_optimization
                                      .optimization_score
                                  }
                                </span>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div>
                    <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-muted-foreground mb-4">
                      Upload a photo of the suspected breeding site
                    </p>
                  </div>
                )}

                <div className="space-y-2">
                  <input
                    type="file"
                    accept="image/*"
                    id="file-upload"
                    className="hidden"
                    onChange={handleImageUpload}
                    aria-label="Upload image file"
                    title="Upload image file"
                  />
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() => {
                      console.log("Button clicked, triggering file input");
                      document.getElementById("file-upload")?.click();
                    }}
                    type="button"
                  >
                    <Camera className="w-4 h-4 mr-2" />
                    {uploadedImage ? "Change Photo" : "Take Photo / Upload"}
                  </Button>
                </div>
              </div>

              <p className="text-xs text-muted-foreground">
                Upload an image first, then use the "Analyze with Advanced AI"
                button to detect breeding sites
              </p>
            </CardContent>
          </Card>

          {/* Report Form */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MapPin className="w-5 h-5 text-primary" />
                Location & Details
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="location">Location</Label>
                <div className="flex gap-2">
                  <Input
                    id="location"
                    placeholder="GPS coordinates will appear here"
                    value={location}
                    onChange={(e) => setLocation(e.target.value)}
                    className="flex-1"
                  />
                  <Button
                    variant="outline"
                    onClick={handleGetLocation}
                    className="shrink-0"
                  >
                    <MapPin className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="address">Address (Optional)</Label>
                <Input
                  id="address"
                  placeholder="e.g., Jalan Sultan Ismail, Kuala Lumpur"
                  value={formData.address}
                  onChange={(e) => handleInputChange("address", e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Textarea
                  id="description"
                  placeholder="Describe what you observed (e.g., stagnant water in flower pots, blocked drains, etc.)"
                  rows={4}
                  value={formData.description}
                  onChange={(e) =>
                    handleInputChange("description", e.target.value)
                  }
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="contact">Contact Number (Optional)</Label>
                <Input
                  id="contact"
                  placeholder="+60 XX-XXX XXXX"
                  type="tel"
                  value={formData.contact}
                  onChange={(e) => handleInputChange("contact", e.target.value)}
                />
              </div>

              {uploadedImage && (
                <Button
                  onClick={handleAnalyzeImage}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                  disabled={isAnalyzing || !location}
                >
                  {isAnalyzing ? (
                    <>
                      <Loader className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing with Advanced AI...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2" />
                      🤖 Analyze with Advanced AI
                    </>
                  )}
                </Button>
              )}

              <Button
                onClick={handleSubmit}
                className="w-full bg-eco-green hover:bg-eco-green-dark text-white"
                disabled={!uploadedImage || !location || isAnalyzing}
              >
                {isAnalyzing ? (
                  <>
                    <Loader className="w-4 h-4 mr-2 animate-spin" />
                    Analyzing Image...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 mr-2" />
                    Submit Report
                  </>
                )}
              </Button>

              <p className="text-xs text-muted-foreground text-center">
                Reports are reviewed by local health authorities within 24-48
                hours
              </p>
            </CardContent>
          </Card>
        </div>

        {/* Info Cards */}
        <div className="grid md:grid-cols-2 gap-6 mt-8">
          <Card className="border-eco-green/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <CheckCircle className="w-6 h-6 text-eco-green" />
                <h3 className="font-semibold text-foreground">
                  What to Report
                </h3>
              </div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Stagnant water in containers</li>
                <li>• Blocked drains or gutters</li>
                <li>• Water storage without covers</li>
                <li>• Neglected swimming pools</li>
                <li>• Construction site water pools</li>
              </ul>
            </CardContent>
          </Card>

          <Card className="border-healthcare-blue/20">
            <CardContent className="p-6">
              <div className="flex items-center gap-3 mb-3">
                <AlertCircle className="w-6 h-6 text-healthcare-blue" />
                <h3 className="font-semibold text-foreground">
                  Why It Matters
                </h3>
              </div>
              <ul className="space-y-1 text-sm text-muted-foreground">
                <li>• Aedes mosquitoes breed in stagnant water</li>
                <li>• Early detection prevents outbreaks</li>
                <li>• Community reports are 85% accurate</li>
                <li>• Your report could save lives</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Report;
