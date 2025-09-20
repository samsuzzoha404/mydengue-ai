"""
PDF Processing Route
Extract dengue area information from PDF documents
"""

from fastapi import APIRouter, HTTPException
import traceback
import os
import json
from ..services.dengue_pdf_processor import dengue_pdf_processor

# Create router
router = APIRouter(prefix="/pdf", tags=["PDF Processing"])

@router.get('/process-dengue-areas')
async def process_dengue_areas():
    """Process PDF files to extract dengue potential area information"""
    try:
        print("🔍 Starting PDF processing for dengue areas...")
        
        # Process all PDF files
        results = dengue_pdf_processor.process_all_pdfs()
        
        # Get training summary
        training_summary = dengue_pdf_processor.get_training_summary()
        
        response = {
            "success": True,
            "message": "PDF processing completed successfully",
            "processing_results": results,
            "training_summary": training_summary,
            "timestamp": "2025-09-19T10:00:00Z"
        }
        
        return response
        
    except Exception as e:
        print(f"❌ PDF processing error: {str(e)}")
        print(f"📍 Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to process PDF files",
            "timestamp": "2025-09-19T10:00:00Z"
        })

@router.get('/training-data')
async def get_training_data():
    """Get extracted training data for AI enhancement"""
    try:
        # Get training summary
        training_summary = dengue_pdf_processor.get_training_summary()
        
        # Check if training data file exists
        training_data_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'dengue_areas_training.json'
        )
        
        training_data = None
        if os.path.exists(training_data_path):
            with open(training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
        
        response = {
            "success": True,
            "training_summary": training_summary,
            "has_training_data": training_data is not None,
            "training_data": training_data,
            "message": "Training data retrieved successfully"
        }
        
        return response
        
    except Exception as e:
        print(f"❌ Training data retrieval error: {str(e)}")
        
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve training data"
        })

@router.post('/enhance-ai')
async def enhance_ai_with_pdf_data():
    """Enhance AI model with extracted PDF data"""
    try:
        # Get current AI service
        from ..services.custom_ai_service import custom_ai_service
        from ..services.real_dengue_ai import real_dengue_ai
        
        # Load PDF training data
        training_data_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'dengue_areas_training.json'
        )
        
        if not os.path.exists(training_data_path):
            raise HTTPException(status_code=400, detail={
                "success": False,
                "error": "No PDF training data found",
                "message": "Please process PDF files first"
            })
        
        with open(training_data_path, 'r', encoding='utf-8') as f:
            pdf_data = json.load(f)
        
        # Extract area information for AI enhancement
        locations = []
        statistical_data = []
        image_analyses = []
        
        for area in pdf_data.get("areas", []):
            if area["type"] == "location":
                locations.append(area["name"])
            elif area["type"] == "statistical_data":
                statistical_data.append({
                    "value": area["value"],
                    "context": area["context"]
                })
            elif area["type"] == "image_analysis":
                image_analyses.append({
                    "water_potential": area["analysis"]["water_potential"],
                    "blue_dominance": area["analysis"]["blue_dominance"]
                })
        
        # Create enhanced prediction with PDF insights
        enhancement_data = {
            "pdf_locations": locations,
            "statistical_insights": statistical_data,
            "image_analysis": image_analyses,
            "total_areas_processed": len(pdf_data.get("areas", []))
        }
        
        # Test enhanced prediction
        sample_prediction = real_dengue_ai.predict_dengue_cases(
            weather_data={
                'temperature': 28.5,
                'humidity': 75.0,
                'rainfall': 12.0,
                'wind_speed': 5.0
            },
            location="Malaysia"
        )
        
        response = {
            "success": True,
            "message": "AI enhanced with PDF data successfully",
            "enhancement_summary": {
                "locations_added": len(locations),
                "statistical_points": len(statistical_data),
                "image_analyses": len(image_analyses),
                "total_training_areas": enhancement_data["total_areas_processed"]
            },
            "sample_enhanced_prediction": sample_prediction,
            "pdf_insights": enhancement_data
        }
        
        return response
        
    except Exception as e:
        print(f"❌ AI enhancement error: {str(e)}")
        print(f"📍 Full traceback: {traceback.format_exc()}")
        
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to enhance AI with PDF data"
        })

@router.get('/status')
async def pdf_processing_status():
    """Get current PDF processing status"""
    try:
        # Check for PDF files
        root_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        pdf_files = ['statistik.pdf', 'tapak_pembinaan.pdf']
        
        files_status = {}
        for pdf_file in pdf_files:
            pdf_path = os.path.join(root_path, pdf_file)
            files_status[pdf_file] = {
                "exists": os.path.exists(pdf_path),
                "size": os.path.getsize(pdf_path) if os.path.exists(pdf_path) else 0
            }
        
        # Check training data
        training_summary = dengue_pdf_processor.get_training_summary()
        
        response = {
            "success": True,
            "pdf_files": files_status,
            "training_data_available": "error" not in training_summary,
            "training_summary": training_summary,
            "processor_ready": True
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "success": False,
            "error": str(e),
            "message": "Failed to get PDF processing status"
        })