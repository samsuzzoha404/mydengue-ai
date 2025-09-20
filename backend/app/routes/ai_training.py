"""
Advanced AI Training Routes
Expose comprehensive breeding site detection and training capabilities
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import traceback
import os

# Create router
router = APIRouter(prefix="/ai-training", tags=["AI Training"])

class TrainingExample(BaseModel):
    image_data: str
    true_category: str
    feedback: Optional[str] = None

class ImageAnalysisRequest(BaseModel):
    image_data: str
    context: Optional[Dict[str, Any]] = None

# Import services
try:
    from ..services.advanced_breeding_detector import advanced_breeding_detector
    from ..services.breeding_site_trainer import breeding_site_trainer
    from ..services.training_pipeline import training_pipeline
    TRAINING_SERVICES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Training services not available: {e}")
    TRAINING_SERVICES_AVAILABLE = False

@router.post('/analyze-breeding-site')
async def analyze_breeding_site_advanced(request: ImageAnalysisRequest):
    """
    Advanced breeding site analysis with expert guidelines
    Returns detailed classification with confidence scores and recommendations
    """
    if not TRAINING_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training services not available")
    
    try:
        print("🔍 Starting advanced breeding site analysis...")
        
        # Perform comprehensive analysis
        result = advanced_breeding_detector.analyze_breeding_site(
            request.image_data, request.context
        )
        
        return {
            "success": True,
            "analysis": result,
            "categories": {
                "hotspot": "Confirmed breeding site - immediate action needed",
                "potential": "Potential breeding site - needs verification",
                "not_hotspot": "Not a breeding site - low risk",
                "uncertain": "Ambiguous - human review recommended",
                "invalid": "Image quality insufficient for analysis"
            },
            "message": "Advanced analysis completed successfully"
        }
        
    except Exception as e:
        print(f"❌ Advanced analysis error: {str(e)}")
        print(f"📍 Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to perform advanced analysis"
        })

@router.post('/add-training-example')
async def add_training_example(example: TrainingExample):
    """Add a training example with feedback to improve the AI system"""
    if not TRAINING_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training services not available")
    
    try:
        # First, get current prediction
        current_result = advanced_breeding_detector.analyze_breeding_site(example.image_data)
        predicted_category = current_result['category']
        confidence = current_result['confidence']
        
        # Add to training pipeline
        training_result = training_pipeline.add_training_example(
            example.image_data,
            example.true_category,
            predicted_category,
            confidence,
            example.feedback
        )
        
        # Also train the advanced detector directly
        detector_training = advanced_breeding_detector.train_on_example(
            example.image_data,
            example.true_category,
            example.feedback
        )
        
        return {
            "success": True,
            "training_result": training_result,
            "detector_training": detector_training,
            "current_prediction": {
                "category": predicted_category,
                "confidence": round(confidence * 100, 2),
                "correct": predicted_category == example.true_category
            },
            "message": "Training example added successfully"
        }
        
    except Exception as e:
        print(f"❌ Training example error: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to add training example"
        })

@router.get('/training-statistics')
async def get_training_statistics():
    """Get comprehensive training statistics and performance metrics"""
    if not TRAINING_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training services not available")
    
    try:
        # Get pipeline statistics
        pipeline_stats = training_pipeline.get_training_statistics()
        
        # Get detector training stats
        detector_stats = advanced_breeding_detector.get_training_stats()
        
        # Get training data summary
        trainer_summary = breeding_site_trainer.get_training_summary()
        
        return {
            "success": True,
            "pipeline_statistics": pipeline_stats,
            "detector_statistics": detector_stats,
            "synthetic_data_summary": trainer_summary,
            "system_status": {
                "advanced_detector_available": True,
                "training_pipeline_active": True,
                "synthetic_trainer_loaded": True
            },
            "message": "Training statistics retrieved successfully"
        }
        
    except Exception as e:
        print(f"❌ Training statistics error: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve training statistics"
        })

@router.post('/trigger-retraining')
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """Trigger model retraining with accumulated examples"""
    if not TRAINING_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training services not available")
    
    try:
        # Trigger retraining in background
        def perform_retraining():
            return training_pipeline.trigger_retraining()
        
        # For now, run synchronously to get immediate feedback
        retraining_result = perform_retraining()
        
        return {
            "success": True,
            "retraining_result": retraining_result,
            "message": "Model retraining initiated"
        }
        
    except Exception as e:
        print(f"❌ Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to trigger retraining"
        })

@router.get('/generate-synthetic-data')
async def generate_synthetic_training_data():
    """Generate comprehensive synthetic training data based on expert guidelines"""
    if not TRAINING_SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Training services not available")
    
    try:
        # Generate comprehensive training set
        training_data = breeding_site_trainer.generate_comprehensive_training_set()
        
        # Save to file
        output_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'synthetic_training_data.json'
        )
        
        save_result = breeding_site_trainer.save_training_data(output_path)
        
        return {
            "success": True,
            "training_data_summary": {
                "total_scenarios": training_data['metadata']['total_scenarios'],
                "categories": training_data['metadata']['categories'],
                "location_types": training_data['metadata']['location_types']
            },
            "save_result": save_result,
            "data_structure": {
                category: len(examples) 
                for category, examples in training_data['training_examples'].items()
            },
            "message": "Synthetic training data generated successfully"
        }
        
    except Exception as e:
        print(f"❌ Synthetic data generation error: {str(e)}")
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to generate synthetic training data"
        })

@router.get('/classification-guidelines')
async def get_classification_guidelines():
    """Get expert classification guidelines for breeding site detection"""
    
    guidelines = {
        "categories": {
            "hotspot": {
                "name": "Hotspot (Breeding Site)",
                "description": "Confirmed breeding site with visible stagnant water",
                "criteria": [
                    "Visible stagnant water present",
                    "Container-like structure that can hold water",
                    "Water can remain for 3-5+ days",
                    "Suitable for Aedes mosquito breeding"
                ],
                "examples": [
                    "Flower vases with stagnant water",
                    "Buckets/pails with rainwater",
                    "Tires filled with water",
                    "Clogged drains with standing water",
                    "Plant saucers with water",
                    "Tree holes with rainwater"
                ],
                "action_required": "Immediate - Remove water or treat area"
            },
            "potential": {
                "name": "Potential Breeding Site",
                "description": "Container present but needs verification after rain",
                "criteria": [
                    "Container present but no visible water",
                    "Outdoor location with rain exposure",
                    "Shape suitable for water collection",
                    "Requires verification after rain"
                ],
                "examples": [
                    "Empty buckets outdoors",
                    "Dry tires lying flat",
                    "Blocked gutters without visible water",
                    "Construction sites with uneven surfaces"
                ],
                "action_required": "Monitor regularly, especially after rain"
            },
            "not_hotspot": {
                "name": "Not a Hotspot",
                "description": "Low risk for mosquito breeding",
                "criteria": [
                    "No water present and no collection potential",
                    "Moving/flowing water",
                    "Maintained water bodies",
                    "Indoor dry surfaces"
                ],
                "examples": [
                    "Dry pavements",
                    "Flowing rivers",
                    "Swimming pools with circulation",
                    "Clean dry containers",
                    "Vehicles, people, furniture"
                ],
                "action_required": "Continue regular monitoring"
            },
            "uncertain": {
                "name": "Uncertain - Needs Review",
                "description": "Ambiguous features requiring human verification",
                "criteria": [
                    "Poor image quality preventing analysis",
                    "Ambiguous water presence",
                    "Partially obscured containers",
                    "Conflicting visual indicators"
                ],
                "examples": [
                    "Partially visible containers",
                    "Reflective surfaces of unclear type",
                    "Shadowed areas with possible water"
                ],
                "action_required": "Manual inspection recommended"
            },
            "invalid": {
                "name": "Invalid/Unusable",
                "description": "Image quality insufficient for analysis",
                "criteria": [
                    "Image too blurry/dark for analysis",
                    "Corrupted or distorted image data",
                    "Insufficient resolution",
                    "Motion blur or focus issues"
                ],
                "examples": [
                    "Blurred motion photos",
                    "Extremely dark images",
                    "Pixelated or corrupted images"
                ],
                "action_required": "Retake photo with better quality"
            }
        },
        "expert_tips": [
            "Focus on stagnant water in small containers",
            "Even small volumes (bottle caps) can be breeding sites",
            "Check shaded areas where water persists longer",
            "Look for organic matter in water (higher breeding risk)",
            "Consider seasonal patterns and rain collection"
        ],
        "high_risk_locations": [
            "Household containers (buckets, vases, plant trays)",
            "Tire storage areas",
            "Construction sites with water collection",
            "Blocked drainage systems",
            "Tree holes and bamboo stumps",
            "Air conditioner drip areas"
        ]
    }
    
    return {
        "success": True,
        "guidelines": guidelines,
        "message": "Classification guidelines retrieved successfully"
    }

@router.get('/system-status')
async def get_ai_system_status():
    """Get comprehensive status of AI training system"""
    
    system_status = {
        "services_available": TRAINING_SERVICES_AVAILABLE,
        "components": {
            "advanced_breeding_detector": TRAINING_SERVICES_AVAILABLE,
            "synthetic_data_generator": TRAINING_SERVICES_AVAILABLE,
            "training_pipeline": TRAINING_SERVICES_AVAILABLE
        },
        "capabilities": [
            "5-category breeding site classification",
            "Expert guideline-based analysis",
            "Automated training with feedback",
            "Synthetic data generation",
            "Performance tracking and improvement",
            "Confidence scoring and uncertainty handling"
        ],
        "version": "2.0.0",
        "last_updated": "2025-09-19"
    }
    
    if TRAINING_SERVICES_AVAILABLE:
        try:
            # Get current performance metrics
            stats = training_pipeline.get_training_statistics()
            system_status["current_performance"] = stats["overall_statistics"]
        except:
            system_status["current_performance"] = "Not available"
    
    return {
        "success": True,
        "system_status": system_status,
        "message": "AI training system status retrieved"
    }