import { useState, useCallback } from "react";
import {
  GoogleMap,
  useJsApiLoader,
  Marker,
  Circle,
  InfoWindow,
} from "@react-google-maps/api";
import { MapPin, Filter, Info, TrendingUp, AlertTriangle, ExternalLink, Copy } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { isApiKeyConfigured, GOOGLE_MAPS_SETUP_GUIDE } from "@/lib/google-maps-setup";

const libraries: ("places" | "geometry")[] = ["places", "geometry"];

const Map = () => {
  const [selectedFilter, setSelectedFilter] = useState("risk");
  const [selectedState, setSelectedState] = useState<string | null>(null);
  const [map, setMap] = useState<google.maps.Map | null>(null);
  const [selectedMarker, setSelectedMarker] = useState<string | null>(null);

  // Check if Google Maps API key is configured
  const apiKey = import.meta.env.VITE_GOOGLE_MAPS_API_KEY;
  const hasApiKey = isApiKeyConfigured(apiKey);

  const { isLoaded, loadError } = useJsApiLoader({
    id: "google-map-script",
    googleMapsApiKey: hasApiKey ? apiKey : "",
    libraries: libraries,
  });

  const filters = [
    { id: "risk", label: "Risk Level", icon: AlertTriangle },
    { id: "cases", label: "Active Cases", icon: TrendingUp },
    { id: "reports", label: "Citizen Reports", icon: MapPin },
  ];

  const riskData = {
    Selangor: {
      risk: "High",
      cases: 156,
      reports: 23,
      color: "alert-red",
      lat: 3.0738,
      lng: 101.5183,
    },
    "Kuala Lumpur": {
      risk: "High",
      cases: 89,
      reports: 15,
      color: "alert-red",
      lat: 3.139,
      lng: 101.6869,
    },
    Johor: {
      risk: "Medium",
      cases: 67,
      reports: 12,
      color: "warning",
      lat: 1.4927,
      lng: 103.7414,
    },
    Penang: {
      risk: "Medium",
      cases: 45,
      reports: 8,
      color: "warning",
      lat: 5.4164,
      lng: 100.3327,
    },
    Perak: {
      risk: "Low",
      cases: 23,
      reports: 5,
      color: "eco-green",
      lat: 4.5975,
      lng: 101.0901,
    },
    Sabah: {
      risk: "Low",
      cases: 18,
      reports: 3,
      color: "eco-green",
      lat: 5.9804,
      lng: 116.0735,
    },
    Sarawak: {
      risk: "Low",
      cases: 15,
      reports: 2,
      color: "eco-green",
      lat: 1.5533,
      lng: 110.3592,
    },
  };

  // Generate visualization data for modern Google Maps (replaces deprecated HeatmapLayer)
  const generateVisualizationData = () => {
    return Object.entries(riskData).map(([state, data]) => ({
      id: state,
      state: state,
      position: { lat: data.lat, lng: data.lng },
      risk: data.risk,
      cases: data.cases,
      reports: data.reports,
      color: data.color,
      // Calculate circle radius based on cases
      radius: Math.max(15000, data.cases * 500), // Minimum 15km radius
      // Calculate marker icon color
      markerColor: data.risk === "High" ? "#ef4444" : 
                   data.risk === "Medium" ? "#f59e0b" : "#10b981"
    }));
  };

  const visualizationData = generateVisualizationData();

  const mapContainerStyle = {
    width: "100%",
    height: "100%",
  };

  const center = {
    lat: 4.2105, // Malaysia center
    lng: 101.9758,
  };

  const mapOptions = {
    zoom: 6,
    center: center,
    mapTypeId: "roadmap" as google.maps.MapTypeId,
    styles: [
      {
        featureType: "all",
        stylers: [{ saturation: -20 }],
      },
    ],
  };

  const onLoad = useCallback((map: google.maps.Map) => {
    setMap(map);
  }, []);

  const onUnmount = useCallback(() => {
    setMap(null);
  }, []);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "High":
        return "text-alert-red bg-alert-red/10 border-alert-red/20";
      case "Medium":
        return "text-warning bg-warning/10 border-warning/20";
      case "Low":
        return "text-eco-green bg-eco-green/10 border-eco-green/20";
      default:
        return "text-muted-foreground bg-muted/10";
    }
  };

  return (
    <div className="min-h-screen bg-background p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl sm:text-4xl font-poppins font-bold text-foreground mb-4">
            Malaysia Dengue Risk Map
          </h1>
          <p className="text-lg text-muted-foreground mb-6">
            Real-time dengue risk assessment across Malaysian states and
            territories
          </p>

          {/* Filters */}
          <div className="flex flex-wrap gap-3">
            {filters.map((filter) => (
              <Button
                key={filter.id}
                variant={selectedFilter === filter.id ? "default" : "outline"}
                onClick={() => setSelectedFilter(filter.id)}
                className="flex items-center gap-2"
              >
                <filter.icon className="w-4 h-4" />
                {filter.label}
              </Button>
            ))}
          </div>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Map Area */}
          <div className="lg:col-span-2 space-y-6">
            <Card className="h-[600px] shadow-md border-2">
              <CardHeader className="border-b bg-background/95 backdrop-blur-sm sticky top-0 z-10">
                <CardTitle className="flex items-center gap-2">
                  <MapPin className="w-5 h-5 text-healthcare-blue" />
                  Interactive Malaysia Map
                </CardTitle>
              </CardHeader>
              <CardContent className="h-[calc(100%-4rem)] p-0">
                {/* Google Maps with Heatmap or Demo Static Map */}
                <div className="w-full h-full">
                  {!hasApiKey ? (
                    /* Demo Static Map View - No API Key Required */
                    <div className="w-full h-full bg-gradient-to-br from-slate-100 via-blue-50 to-emerald-50 dark:from-slate-900 dark:via-blue-950 dark:to-emerald-950 rounded-lg overflow-hidden relative">
                      {/* Demo Mode Badge */}
                      <div className="absolute top-4 left-4 z-10">
                        <Badge className="bg-blue-600 text-white shadow-lg">
                          <Info className="w-3 h-3 mr-1" />
                          Demo Mode - Static View
                        </Badge>
                      </div>

                      {/* Malaysia Map Representation with State Markers */}
                      <div className="relative w-full h-full p-8">
                        {/* Simple Malaysia Map Outline */}
                        <svg viewBox="0 0 800 600" className="w-full h-full opacity-20 absolute inset-0">
                          <path
                            d="M 400 100 L 500 150 L 550 200 L 580 280 L 560 350 L 520 400 L 450 420 L 380 410 L 320 380 L 280 320 L 260 250 L 280 180 L 350 130 Z"
                            fill="currentColor"
                            className="text-slate-400"
                          />
                        </svg>

                        {/* State Risk Indicators */}
                        <div className="relative z-10 grid grid-cols-2 md:grid-cols-3 gap-4 h-full content-center">
                          {Object.entries(riskData).map(([state, data]) => (
                            <button
                              key={state}
                              onClick={() => setSelectedState(state === selectedState ? null : state)}
                              className={`
                                p-4 rounded-lg border-2 transition-all duration-200
                                ${selectedState === state ? 'ring-4 ring-primary/50 scale-105' : 'hover:scale-102'}
                                ${data.risk === 'High' ? 'bg-red-50 border-red-300 dark:bg-red-950 dark:border-red-700' :
                                  data.risk === 'Medium' ? 'bg-yellow-50 border-yellow-300 dark:bg-yellow-950 dark:border-yellow-700' :
                                  'bg-green-50 border-green-300 dark:bg-green-950 dark:border-green-700'}
                              `}
                            >
                              <div className="flex items-start justify-between mb-2">
                                <h4 className="font-semibold text-sm">{state}</h4>
                                <MapPin className={`w-4 h-4 ${
                                  data.risk === 'High' ? 'text-red-600' :
                                  data.risk === 'Medium' ? 'text-yellow-600' : 'text-green-600'
                                }`} />
                              </div>
                              <div className="space-y-1 text-xs">
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">Risk:</span>
                                  <span className={`font-semibold ${
                                    data.risk === 'High' ? 'text-red-700 dark:text-red-400' :
                                    data.risk === 'Medium' ? 'text-yellow-700 dark:text-yellow-400' :
                                    'text-green-700 dark:text-green-400'
                                  }`}>{data.risk}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">Cases:</span>
                                  <span className="font-medium">{data.cases}</span>
                                </div>
                                <div className="flex justify-between">
                                  <span className="text-muted-foreground">Reports:</span>
                                  <span className="font-medium">{data.reports}</span>
                                </div>
                              </div>
                            </button>
                          ))}
                        </div>

                        {/* Info Footer */}
                        <div className="absolute bottom-4 right-4 bg-white/90 dark:bg-slate-800/90 p-3 rounded-lg shadow-md max-w-xs">
                          <p className="text-xs text-muted-foreground">
                            <Info className="w-3 h-3 inline mr-1" />
                            Add Google Maps API key for interactive map view
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : loadError ? (
                    <div className="w-full h-full bg-gradient-to-br from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20 rounded-lg flex items-center justify-center">
                      <div className="text-center space-y-4 p-6">
                        <AlertTriangle className="w-16 h-16 text-red-600 mx-auto" />
                        <div>
                          <h3 className="text-xl font-semibold text-foreground mb-2">
                            Failed to Load Google Maps
                          </h3>
                          <p className="text-muted-foreground">
                            There was an error loading the Google Maps API. Please check your API key and try again.
                          </p>
                        </div>
                      </div>
                    </div>
                  ) : isLoaded ? (
                    <GoogleMap
                      mapContainerStyle={mapContainerStyle}
                      center={center}
                      zoom={6}
                      onLoad={onLoad}
                      onUnmount={onUnmount}
                      options={mapOptions}
                    >
                      {/* Risk Circles for each state */}
                      {visualizationData.map((stateData) => (
                        <Circle
                          key={`circle-${stateData.id}`}
                          center={stateData.position}
                          radius={stateData.radius}
                          options={{
                            fillColor: stateData.markerColor,
                            fillOpacity: 0.25,
                            strokeColor: stateData.markerColor,
                            strokeOpacity: 0.8,
                            strokeWeight: 2,
                          }}
                        />
                      ))}
                      
                      {/* Markers for each state */}
                      {visualizationData.map((stateData) => (
                        <Marker
                          key={`marker-${stateData.id}`}
                          position={stateData.position}
                          onClick={() => setSelectedMarker(stateData.id)}
                          icon={{
                            path: google.maps.SymbolPath.CIRCLE,
                            scale: 12,
                            fillColor: stateData.markerColor,
                            fillOpacity: 1,
                            strokeColor: "#ffffff",
                            strokeWeight: 2,
                          }}
                        />
                      ))}
                      
                      {/* Info Windows */}
                      {selectedMarker && (
                        <InfoWindow
                          position={visualizationData.find(d => d.id === selectedMarker)?.position}
                          onCloseClick={() => setSelectedMarker(null)}
                        >
                          <div className="p-2 max-w-xs">
                            {(() => {
                              const data = visualizationData.find(d => d.id === selectedMarker);
                              if (!data) return null;
                              return (
                                <div>
                                  <h3 className="font-semibold text-lg mb-2">{data.state}</h3>
                                  <div className="space-y-1 text-sm">
                                    <div className="flex justify-between">
                                      <span>Risk Level:</span>
                                      <span className={`font-medium ${
                                        data.risk === "High" ? "text-red-600" :
                                        data.risk === "Medium" ? "text-yellow-600" : "text-green-600"
                                      }`}>{data.risk}</span>
                                    </div>
                                    <div className="flex justify-between">
                                      <span>Active Cases:</span>
                                      <span className="font-medium">{data.cases}</span>
                                    </div>
                                    <div className="flex justify-between">
                                      <span>Reports:</span>
                                      <span className="font-medium">{data.reports}</span>
                                    </div>
                                  </div>
                                </div>
                              );
                            })()}
                          </div>
                        </InfoWindow>
                      )}
                    </GoogleMap>
                  ) : (
                    <div className="w-full h-full bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 rounded-lg flex items-center justify-center">
                      <div className="text-center space-y-4 p-6">
                        <MapPin className="w-16 h-16 text-healthcare-blue/60 mx-auto animate-pulse" />
                        <div className="max-w-md">
                          <h3 className="text-xl font-semibold text-foreground mb-2">
                            Loading Google Maps...
                          </h3>
                          <p className="text-muted-foreground">
                            Setting up interactive heatmap visualization
                          </p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Legend */}
            <Card className="shadow-sm border-2">
              <CardHeader className="pb-3 border-b">
                <CardTitle className="flex items-center gap-2 text-base">
                  <Info className="w-4 h-4 text-healthcare-blue" />
                  Risk Level Legend
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className="flex items-center gap-3 bg-alert-red/5 p-3 rounded-lg">
                    <div className="w-4 h-4 bg-alert-red rounded-sm shadow-sm"></div>
                    <span className="text-sm font-medium whitespace-nowrap">
                      High Risk (&gt;100 cases)
                    </span>
                  </div>
                  <div className="flex items-center gap-3 bg-warning/5 p-3 rounded-lg">
                    <div className="w-4 h-4 bg-warning rounded-sm shadow-sm"></div>
                    <span className="text-sm font-medium whitespace-nowrap">
                      Medium Risk (25-100)
                    </span>
                  </div>
                  <div className="flex items-center gap-3 bg-eco-green/5 p-3 rounded-lg">
                    <div className="w-4 h-4 bg-eco-green rounded-sm shadow-sm"></div>
                    <span className="text-sm font-medium whitespace-nowrap">
                      Low Risk (&lt;25)
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Fallback Data Visualization when Maps API is not available */}
            {!hasApiKey && (
              <Card className="shadow-sm border-2">
                <CardHeader className="pb-3 border-b">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <MapPin className="w-4 h-4 text-healthcare-blue" />
                    Risk Data Overview
                  </CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Interactive map unavailable - showing data table view
                  </p>
                </CardHeader>
                <CardContent className="pt-4">
                  <div className="space-y-3">
                    {Object.entries(riskData).map(([state, data]) => (
                      <div 
                        key={state}
                        className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/50 transition-colors"
                      >
                        <div className="flex items-center gap-3">
                          <div className={`w-3 h-3 rounded-full ${
                            data.risk === "High" ? "bg-red-500" :
                            data.risk === "Medium" ? "bg-yellow-500" : "bg-green-500"
                          }`}></div>
                          <span className="font-medium">{state}</span>
                        </div>
                        <div className="flex items-center gap-4 text-sm">
                          <Badge variant={data.risk === "High" ? "destructive" : 
                                        data.risk === "Medium" ? "default" : "secondary"}>
                            {data.risk} Risk
                          </Badge>
                          <span className="text-muted-foreground">{data.cases} cases</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* State Details */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Info className="w-5 h-5 text-primary" />
                  State Risk Overview
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {Object.entries(riskData).map(([state, data]) => (
                  <div
                    key={state}
                    className={`p-4 rounded-lg border cursor-pointer transition-smooth ${
                      selectedState === state
                        ? "ring-2 ring-primary"
                        : "hover:border-primary/20"
                    }`}
                    onClick={() =>
                      setSelectedState(selectedState === state ? null : state)
                    }
                  >
                    <div className="flex justify-between items-center mb-2">
                      <h4 className="font-semibold text-foreground">{state}</h4>
                      <Badge className={getRiskColor(data.risk)}>
                        {data.risk}
                      </Badge>
                    </div>

                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Cases:</span>
                        <span className="ml-1 font-medium">{data.cases}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Reports:</span>
                        <span className="ml-1 font-medium">{data.reports}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Weekly Trend</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="text-center">
                    <div className="text-3xl font-bold text-alert-red mb-1">
                      ↑ 12%
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Cases increased from last week
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span>High Risk States</span>
                      <span className="font-medium">2</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Medium Risk States</span>
                      <span className="font-medium">2</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span>Low Risk States</span>
                      <span className="font-medium">9</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Map;
