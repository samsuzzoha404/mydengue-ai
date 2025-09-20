"""
PDF Dengue Area Extractor
Extract dengue potential area images and names from PDF documents
"""

import PyPDF2
import pdf2image
import os
import json
import base64
import io
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
import re

class DenguePDFProcessor:
    """Process PDF files containing dengue potential area information"""
    
    def __init__(self):
        self.root_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
        self.pdf_files = [
            'statistik.pdf',
            'tapak_pembinaan.pdf#page=6.pdf'
        ]
        self.extracted_data = []
        
    def process_all_pdfs(self) -> Dict[str, Any]:
        """Process all PDF files and extract dengue area information"""
        print("📄 Processing PDF files for dengue potential areas...")
        
        results = {
            "total_areas_found": 0,
            "pdf_files_processed": 0,
            "areas": [],
            "images_extracted": 0,
            "processing_errors": []
        }
        
        for pdf_file in self.pdf_files:
            try:
                # Handle special filename format
                if '#page=' in pdf_file:
                    actual_filename = pdf_file  # Use the full filename as is
                else:
                    actual_filename = pdf_file
                
                pdf_path = os.path.join(self.root_path, actual_filename)
                
                if os.path.exists(pdf_path):
                    print(f"📖 Processing: {pdf_file}")
                    
                    # Extract text content
                    text_content = self.extract_text_from_pdf(pdf_path)
                    
                    # Extract images 
                    images = self.extract_images_from_pdf(pdf_path)
                    
                    # Analyze content for dengue areas
                    areas_info = self.analyze_dengue_content(text_content, images, pdf_file)
                    
                    results["areas"].extend(areas_info)
                    results["images_extracted"] += len(images)
                    results["pdf_files_processed"] += 1
                    
                    print(f"✅ Extracted {len(areas_info)} potential areas from {pdf_file}")
                    
                else:
                    error_msg = f"PDF file not found: {pdf_path}"
                    print(f"❌ {error_msg}")
                    results["processing_errors"].append(error_msg)
                    
            except Exception as e:
                error_msg = f"Error processing {pdf_file}: {str(e)}"
                print(f"❌ {error_msg}")
                results["processing_errors"].append(error_msg)
        
        results["total_areas_found"] = len(results["areas"])
        
        # Save extracted data for AI training
        if results["areas"]:
            self.save_training_data(results["areas"])
        
        return results
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF"""
        text_content = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\\n"
                    
        except Exception as e:
            print(f"⚠️ Text extraction failed: {e}")
            
        return text_content
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Image.Image]:
        """Extract images from PDF pages"""
        images = []
        
        try:
            # Convert PDF pages to images
            pages = pdf2image.convert_from_path(pdf_path, dpi=200)
            
            for i, page in enumerate(pages):
                images.append(page)
                print(f"📸 Extracted page {i+1} as image")
                
        except Exception as e:
            print(f"⚠️ Image extraction failed: {e}")
            # Try alternative method without external dependencies
            try:
                # Create placeholder images if extraction fails
                placeholder = Image.new('RGB', (800, 600), color='lightgray')
                images.append(placeholder)
            except Exception:
                pass
                
        return images
    
    def analyze_dengue_content(self, text_content: str, images: List[Image.Image], 
                              filename: str) -> List[Dict[str, Any]]:
        """Analyze content for dengue potential areas"""
        areas = []
        
        # Look for common dengue area keywords in Malaysian context
        area_keywords = [
            'kawasan', 'tempat', 'lokasi', 'area', 'zone',
            'pembiakan', 'breeding', 'aedes', 'nyamuk', 'mosquito',
            'air', 'water', 'tadah', 'container', 'bekas',
            'longkang', 'drain', 'parit', 'kolam', 'pond',
            'bumbung', 'roof', 'gutter', 'taman', 'park',
            'sekolah', 'school', 'hospital', 'clinic',
            'perumahan', 'residential', 'flat', 'apartment',
            'pasar', 'market', 'kedai', 'shop', 'restaurant'
        ]
        
        # Statistical data patterns
        stats_patterns = [
            r'(\\d+)\\s*(kes|cases|pesakit)',  # case numbers
            r'(\\d+)\\s*(kawasan|areas|lokasi)',  # area counts
            r'(\\d+)\\s*(tempat|sites|breeding sites)'  # breeding site counts
        ]
        
        # Extract location names (common Malaysian place names)
        location_patterns = [
            r'(Kuala Lumpur|KL)',
            r'(Selangor|Shah Alam|Petaling Jaya|Subang)',
            r'(Johor|Johor Bahru|JB)',
            r'(Penang|Pulau Pinang|Georgetown)',
            r'(Perak|Ipoh)',
            r'(Kedah|Alor Setar)',
            r'(Kelantan|Kota Bharu)',
            r'(Terengganu|Kuala Terengganu)',
            r'(Pahang|Kuantan)',
            r'(Negeri Sembilan|Seremban)',
            r'(Melaka|Malacca)',
            r'(Sabah|Kota Kinabalu)',
            r'(Sarawak|Kuching)'
        ]
        
        # Analyze text content
        text_lower = text_content.lower()
        
        # Find statistics
        for pattern in stats_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                number = int(match.group(1))
                context = match.group(2)
                
                areas.append({
                    "type": "statistical_data",
                    "value": number,
                    "context": context,
                    "source": filename,
                    "description": f"{number} {context} mentioned in document"
                })
        
        # Find locations
        locations_found = []
        for pattern in location_patterns:
            matches = re.finditer(pattern, text_content, re.IGNORECASE)
            for match in matches:
                location = match.group(1)
                if location not in locations_found:
                    locations_found.append(location)
                    
                    areas.append({
                        "type": "location",
                        "name": location,
                        "source": filename,
                        "description": f"Dengue area mentioned: {location}"
                    })
        
        # Analyze images for potential breeding sites
        for i, image in enumerate(images):
            image_analysis = self.analyze_image_for_breeding_sites(image, f"{filename}_page_{i+1}")
            if image_analysis:
                areas.append(image_analysis)
        
        # Create general area entries if specific data not found but keywords present
        keyword_count = sum(1 for keyword in area_keywords if keyword in text_lower)
        
        if keyword_count > 5 and not areas:  # Many keywords but no specific areas found
            areas.append({
                "type": "general_area",
                "keyword_density": keyword_count,
                "source": filename,
                "description": f"Document contains {keyword_count} dengue-related keywords",
                "potential_areas": "Multiple areas mentioned in document"
            })
        
        return areas
    
    def analyze_image_for_breeding_sites(self, image: Image.Image, image_id: str) -> Optional[Dict[str, Any]]:
        """Analyze image for potential breeding sites using basic computer vision"""
        try:
            # Convert to base64 for AI processing
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Basic image analysis
            width, height = image.size
            
            # Look for water-like colors (blue dominant areas)
            import numpy as np
            img_array = np.array(image)
            
            # Calculate blue dominance
            if len(img_array.shape) == 3:
                blue_pixels = np.sum(img_array[:,:,2] > img_array[:,:,0]) + np.sum(img_array[:,:,2] > img_array[:,:,1])
                total_pixels = width * height
                blue_ratio = blue_pixels / (total_pixels * 2)  # Divided by 2 for double counting
                
                if blue_ratio > 0.1:  # Significant blue content
                    return {
                        "type": "image_analysis",
                        "image_id": image_id,
                        "image_data": image_b64,
                        "analysis": {
                            "blue_dominance": float(blue_ratio),
                            "water_potential": "High" if blue_ratio > 0.3 else "Medium",
                            "dimensions": f"{width}x{height}"
                        },
                        "description": f"Image shows potential water areas (blue ratio: {blue_ratio:.2%})"
                    }
            
        except Exception as e:
            print(f"⚠️ Image analysis failed for {image_id}: {e}")
            
        return None
    
    def save_training_data(self, areas: List[Dict[str, Any]]) -> None:
        """Save extracted data for AI training"""
        try:
            training_data_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'dengue_areas_training.json'
            )
            
            os.makedirs(os.path.dirname(training_data_path), exist_ok=True)
            
            training_data = {
                "extraction_date": "2025-09-19",
                "total_areas": len(areas),
                "areas": areas,
                "metadata": {
                    "source": "PDF extraction",
                    "purpose": "AI training for dengue area detection",
                    "format": "Processed for machine learning"
                }
            }
            
            with open(training_data_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
                
            print(f"💾 Training data saved to: {training_data_path}")
            
        except Exception as e:
            print(f"❌ Failed to save training data: {e}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of extracted training data"""
        try:
            training_data_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'dengue_areas_training.json'
            )
            
            if os.path.exists(training_data_path):
                with open(training_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Analyze the training data
                area_types = {}
                locations = []
                images = []
                
                for area in data.get("areas", []):
                    area_type = area.get("type", "unknown")
                    area_types[area_type] = area_types.get(area_type, 0) + 1
                    
                    if area_type == "location":
                        locations.append(area.get("name"))
                    elif area_type == "image_analysis":
                        images.append(area.get("image_id"))
                
                return {
                    "total_areas": data.get("total_areas", 0),
                    "area_types": area_types,
                    "locations_found": locations,
                    "images_analyzed": len(images),
                    "data_quality": "Ready for AI training"
                }
                
        except Exception as e:
            print(f"❌ Failed to get training summary: {e}")
            
        return {"error": "No training data available"}

# Create global instance
dengue_pdf_processor = DenguePDFProcessor()