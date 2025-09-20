"""
Advanced Dengue Breeding Site Detection System
Comprehensive classification with 5 categories based on expert guidelines
"""

import cv2
import numpy as np
import json
import os
from PIL import Image, ImageEnhance, ImageFilter
from typing import Dict, List, Tuple, Optional, Any
import base64
import io
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedBreedingSiteDetector:
    """
    Advanced AI system for detecting dengue breeding sites
    Categories: Hotspot, Potential, Not Hotspot, Uncertain, Invalid
    """
    
    def __init__(self):
        self.categories = {
            'hotspot': 'Breeding Site (Confirmed)',
            'potential': 'Potential Breeding Site (Needs Verification)', 
            'not_hotspot': 'Not a Hotspot',
            'uncertain': 'Uncertain (Needs Human Review)',
            'invalid': 'Invalid/Unusable Image'
        }
        
        # Training data structures
        self.training_examples = self._load_training_data()
        self.classification_rules = self._initialize_classification_rules()
        
        # Detection parameters
        self.water_detection_params = {
            'blue_threshold': 0.15,
            'dark_threshold': 0.3,
            'reflection_threshold': 0.2,
            'edge_density_threshold': 0.1
        }
        
        self.container_detection_params = {
            'circular_threshold': 0.7,
            'rectangular_threshold': 0.6,
            'depth_cue_threshold': 0.4
        }
    
    def analyze_breeding_site(self, image_data: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main analysis function for breeding site detection
        """
        try:
            # Decode image
            image = self._decode_image(image_data)
            if image is None:
                return self._create_result('invalid', 0.0, "Cannot decode image")
            
            # Image quality check
            quality_check = self._assess_image_quality(image)
            if not quality_check['usable']:
                return self._create_result('invalid', 0.0, quality_check['reason'])
            
            # Multi-stage analysis
            water_analysis = self._detect_water_features(image)
            container_analysis = self._detect_containers(image)
            environmental_context = self._analyze_environment(image)
            stagnation_analysis = self._assess_stagnation_risk(image, water_analysis)
            
            # Comprehensive classification
            classification = self._classify_breeding_site(
                water_analysis, container_analysis, 
                environmental_context, stagnation_analysis,
                context
            )
            
            return self._create_detailed_result(
                classification, water_analysis, container_analysis,
                environmental_context, stagnation_analysis, quality_check
            )
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return self._create_result('invalid', 0.0, f"Analysis failed: {str(e)}")
    
    def _decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image data"""
        try:
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image_pil = Image.open(io.BytesIO(image_bytes))
            image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            return image_np
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return None
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Assess if image is usable for analysis"""
        height, width = image.shape[:2]
        
        # Size check
        if width < 100 or height < 100:
            return {'usable': False, 'reason': 'Image too small for analysis', 'score': 0.2}
        
        # Brightness check
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 20:
            return {'usable': False, 'reason': 'Image too dark', 'score': 0.3}
        elif brightness > 240:
            return {'usable': False, 'reason': 'Image too bright/overexposed', 'score': 0.3}
        
        # Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 100:
            return {'usable': False, 'reason': 'Image too blurry', 'score': 0.4}
        
        # Quality score
        quality_score = min(1.0, laplacian_var / 500.0)
        
        return {
            'usable': True, 
            'reason': 'Good quality image',
            'score': quality_score,
            'brightness': brightness,
            'sharpness': laplacian_var
        }
    
    def _detect_water_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhanced water detection for comprehensive breeding site analysis"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Original water detection
        blue_water = self._detect_blue_water(hsv)
        stagnant_water = self._detect_stagnant_water(hsv, lab)
        reflective_surfaces = self._detect_reflections(image)
        
        # NEW: Advanced contamination detection
        rusty_water = self._detect_rusty_contaminated_water(hsv)
        chemical_water = self._detect_chemical_contaminated_water(hsv)
        sewage_water = self._detect_sewage_contaminated_water(hsv)
        oil_contaminated = self._detect_oil_surface(hsv, image)
        latex_contaminated = self._detect_latex_contaminated_water(hsv)
        
        # NEW: Industrial water detection
        industrial_water = self._detect_industrial_contaminated_water(hsv, lab)
        
        # Combine all water detections
        water_mask = cv2.bitwise_or(blue_water, stagnant_water)
        water_mask = cv2.bitwise_or(water_mask, reflective_surfaces)
        water_mask = cv2.bitwise_or(water_mask, rusty_water)
        water_mask = cv2.bitwise_or(water_mask, chemical_water)
        water_mask = cv2.bitwise_or(water_mask, sewage_water)
        water_mask = cv2.bitwise_or(water_mask, oil_contaminated)
        water_mask = cv2.bitwise_or(water_mask, latex_contaminated)
        water_mask = cv2.bitwise_or(water_mask, industrial_water)
        
        # Calculate comprehensive water statistics
        total_pixels = image.shape[0] * image.shape[1]
        water_pixels = cv2.countNonZero(water_mask)
        water_percentage = water_pixels / total_pixels
        
        # Analyze water characteristics and contamination levels
        water_regions = self._analyze_water_regions(water_mask)
        contamination_analysis = self._analyze_contamination_levels({
            'rusty': cv2.countNonZero(rusty_water) / total_pixels,
            'chemical': cv2.countNonZero(chemical_water) / total_pixels,
            'sewage': cv2.countNonZero(sewage_water) / total_pixels,
            'oil': cv2.countNonZero(oil_contaminated) / total_pixels,
            'latex': cv2.countNonZero(latex_contaminated) / total_pixels,
            'industrial': cv2.countNonZero(industrial_water) / total_pixels
        })
        
        return {
            'water_detected': water_percentage > 0.05,
            'water_percentage': water_percentage,
            'water_regions': water_regions,
            'blue_water_present': cv2.countNonZero(blue_water) > 0,
            'stagnant_indicators': cv2.countNonZero(stagnant_water) > 0,
            'reflections_detected': cv2.countNonZero(reflective_surfaces) > 0,
            
            # NEW: Contamination indicators
            'contamination_detected': any(contamination_analysis.values()),
            'contamination_types': contamination_analysis,
            'contamination_severity': max(contamination_analysis.values()) if contamination_analysis.values() else 0.0,
            
            # NEW: Special water types
            'rusty_contaminated': cv2.countNonZero(rusty_water) > 0,
            'chemical_contaminated': cv2.countNonZero(chemical_water) > 0,
            'sewage_contaminated': cv2.countNonZero(sewage_water) > 0,
            'oil_contaminated': cv2.countNonZero(oil_contaminated) > 0,
            'latex_contaminated': cv2.countNonZero(latex_contaminated) > 0,
            'industrial_contaminated': cv2.countNonZero(industrial_water) > 0,
            
            'water_mask': water_mask
        }
    
    def _detect_blue_water(self, hsv: np.ndarray) -> np.ndarray:
        """Detect blue water (clean rainwater)"""
        # Blue color range
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        return cv2.inRange(hsv, lower_blue, upper_blue)
    
    def _detect_stagnant_water(self, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Detect stagnant water (greenish, muddy colors)"""
        # Green-brown stagnant water
        lower_stagnant = np.array([40, 40, 40])
        upper_stagnant = np.array([80, 200, 200])
        stagnant_mask = cv2.inRange(hsv, lower_stagnant, upper_stagnant)
        
        # Dark water (shadow areas)
        dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 100, 80]))
        
        return cv2.bitwise_or(stagnant_mask, dark_mask)
    
    def _detect_reflections(self, image: np.ndarray) -> np.ndarray:
        """Detect water reflections and surface highlights"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # High-pass filter for reflections
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Bright spots that could be water reflections
        bright_spots = cv2.threshold(sharpened, 200, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(bright_spots, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _analyze_water_regions(self, water_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Analyze individual water regions"""
        contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Skip tiny regions
                continue
            
            # Calculate region properties
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            extent = area / (w * h)
            
            # Analyze shape
            hull = cv2.convexHull(contour)
            solidity = area / cv2.contourArea(hull)
            
            regions.append({
                'area': area,
                'aspect_ratio': aspect_ratio,
                'extent': extent,
                'solidity': solidity,
                'bounding_box': (x, y, w, h),
                'shape_type': self._classify_water_shape(aspect_ratio, solidity)
            })
        
        return regions
    
    def _classify_water_shape(self, aspect_ratio: float, solidity: float) -> str:
        """Classify water region shape"""
        if solidity > 0.9 and 0.8 < aspect_ratio < 1.2:
            return 'circular_container'  # Like bucket, pot
        elif solidity > 0.8 and aspect_ratio > 2.0:
            return 'elongated_container'  # Like gutter, drain
        elif solidity < 0.6:
            return 'irregular_puddle'  # Natural puddle
        else:
            return 'rectangular_container'  # Like tank, tray
    
    def _detect_containers(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect container-like objects that could hold water"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for container outlines
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect circles (buckets, pots, tires)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
            param1=50, param2=30, minRadius=20, maxRadius=200
        )
        
        # Detect rectangles (tanks, trays, boxes)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rectangles = self._find_rectangular_containers(contours)
        
        # Detect tire-like shapes
        tires = self._detect_tires(gray, circles)
        
        return {
            'circular_containers': len(circles[0]) if circles is not None else 0,
            'rectangular_containers': len(rectangles),
            'tire_like_objects': len(tires),
            'total_containers': (len(circles[0]) if circles is not None else 0) + len(rectangles) + len(tires),
            'container_details': {
                'circles': circles,
                'rectangles': rectangles,
                'tires': tires
            }
        }
    
    def _find_rectangular_containers(self, contours: List) -> List[Dict[str, Any]]:
        """Find rectangular container shapes"""
        rectangles = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:  # Skip small objects
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4 corners)
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                
                rectangles.append({
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'bounding_box': (x, y, w, h),
                    'corners': len(approx)
                })
        
        return rectangles
    
    def _detect_tires(self, gray: np.ndarray, circles) -> List[Dict[str, Any]]:
        """Detect tire-like objects (rings with holes)"""
        tires = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                
                # Check for tire characteristics (dark center, lighter rim)
                center_region = gray[max(0, y-r//3):min(gray.shape[0], y+r//3), 
                                   max(0, x-r//3):min(gray.shape[1], x+r//3)]
                
                rim_region = gray[max(0, y-r):min(gray.shape[0], y+r), 
                                 max(0, x-r):min(gray.shape[1], x+r)]
                
                if center_region.size > 0 and rim_region.size > 0:
                    center_brightness = np.mean(center_region)
                    rim_brightness = np.mean(rim_region)
                    
                    # Tire typically has darker center
                    if center_brightness < rim_brightness - 20:
                        tires.append({
                            'center': (x, y),
                            'radius': r,
                            'center_darkness': rim_brightness - center_brightness,
                            'tire_confidence': min(1.0, (rim_brightness - center_brightness) / 50.0)
                        })
        
        return tires
    
    def _analyze_environment(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze environmental context"""
        # Color analysis for environment type
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect vegetation (green areas)
        green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        vegetation_percentage = cv2.countNonZero(green_mask) / (image.shape[0] * image.shape[1])
        
        # Detect concrete/urban surfaces (gray areas)
        gray_mask = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([180, 30, 200]))
        concrete_percentage = cv2.countNonZero(gray_mask) / (image.shape[0] * image.shape[1])
        
        # Detect soil/dirt (brown areas)
        brown_mask = cv2.inRange(hsv, np.array([10, 50, 20]), np.array([25, 255, 200]))
        soil_percentage = cv2.countNonZero(brown_mask) / (image.shape[0] * image.shape[1])
        
        # Classify environment type
        environment_type = self._classify_environment(vegetation_percentage, concrete_percentage, soil_percentage)
        
        return {
            'environment_type': environment_type,
            'vegetation_percentage': vegetation_percentage,
            'concrete_percentage': concrete_percentage,
            'soil_percentage': soil_percentage,
            'outdoor_likelihood': self._estimate_outdoor_likelihood(vegetation_percentage, concrete_percentage),
            'urban_vs_natural': 'urban' if concrete_percentage > 0.3 else 'natural'
        }
    
    def _classify_environment(self, veg: float, concrete: float, soil: float) -> str:
        """Classify the environmental context"""
        if concrete > 0.4:
            return 'urban_residential'  # Houses, buildings
        elif veg > 0.5:
            return 'garden_natural'     # Gardens, natural areas
        elif soil > 0.3:
            return 'construction_agricultural'  # Construction sites, farms
        elif concrete > 0.2 and veg > 0.2:
            return 'mixed_urban'        # Mixed urban environment
        else:
            return 'unknown_environment'
    
    def _estimate_outdoor_likelihood(self, vegetation: float, concrete: float) -> float:
        """Estimate likelihood this is an outdoor environment"""
        outdoor_score = 0.0
        
        # Vegetation suggests outdoor
        outdoor_score += vegetation * 0.7
        
        # Some concrete is normal outdoor (paths, driveways)
        if 0.1 < concrete < 0.5:
            outdoor_score += 0.3
        
        # Sky/horizon detection would be ideal but complex
        # For now, use color diversity as proxy
        return min(1.0, outdoor_score)
    
    def _assess_stagnation_risk(self, image: np.ndarray, water_analysis: Dict) -> Dict[str, Any]:
        """Assess likelihood of water stagnation"""
        if not water_analysis['water_detected']:
            return {'stagnation_risk': 0.0, 'indicators': []}
        
        stagnation_indicators = []
        risk_score = 0.0
        
        # Check for stagnation indicators
        if water_analysis['stagnant_indicators']:
            stagnation_indicators.append('greenish_muddy_water')
            risk_score += 0.4
        
        # Dark water (shadows, algae)
        if not water_analysis['blue_water_present'] and water_analysis['water_detected']:
            stagnation_indicators.append('dark_water')
            risk_score += 0.3
        
        # Multiple small water regions (more likely to stagnate)
        if len(water_analysis['water_regions']) > 2:
            stagnation_indicators.append('multiple_small_pools')
            risk_score += 0.2
        
        # Container shapes that typically collect stagnant water
        container_risk = self._assess_container_stagnation_risk(water_analysis['water_regions'])
        risk_score += container_risk
        if container_risk > 0.3:
            stagnation_indicators.append('high_risk_container_shapes')
        
        return {
            'stagnation_risk': min(1.0, risk_score),
            'indicators': stagnation_indicators,
            'container_stagnation_risk': container_risk
        }
    
    def _assess_container_stagnation_risk(self, water_regions: List[Dict]) -> float:
        """Assess stagnation risk based on container shapes"""
        if not water_regions:
            return 0.0
        
        total_risk = 0.0
        for region in water_regions:
            shape_type = region['shape_type']
            
            # High-risk shapes for stagnation
            if shape_type == 'circular_container':
                total_risk += 0.4  # Buckets, pots, tires
            elif shape_type == 'rectangular_container':
                total_risk += 0.3  # Tanks, trays
            elif shape_type == 'irregular_puddle':
                total_risk += 0.2  # Natural puddles
            elif shape_type == 'elongated_container':
                total_risk += 0.1  # Gutters (may have flow)
        
        return min(1.0, total_risk / len(water_regions))
    
    def _classify_breeding_site(self, water_analysis: Dict, container_analysis: Dict, 
                               environmental_context: Dict, stagnation_analysis: Dict,
                               context: Optional[Dict] = None) -> Dict[str, Any]:
        """Enhanced classification logic for comprehensive breeding site detection"""
        
        confidence = 0.0
        reasoning = []
        category = 'uncertain'
        
        # COMPREHENSIVE HOTSPOT CLASSIFICATION
        if (water_analysis['water_detected'] and 
            stagnation_analysis['stagnation_risk'] > 0.5 and
            container_analysis['total_containers'] > 0):
            
            category = 'hotspot'
            base_confidence = 0.8
            reasoning.append("Visible stagnant water in container-like structure")
            
            # NEW: Enhanced confidence based on contamination types
            contamination_boost = self._calculate_contamination_confidence_boost(water_analysis)
            base_confidence += contamination_boost
            
            # NEW: Container type specific confidence
            container_boost = self._calculate_container_type_confidence(container_analysis)
            base_confidence += container_boost
            
            # NEW: Location context confidence
            location_boost = self._calculate_location_confidence(environmental_context, context)
            base_confidence += location_boost
            
            # Specific high-risk scenarios - EXPANDED
            if container_analysis.get('tire_like_objects', 0) > 0:
                base_confidence += 0.15
                reasoning.append("Tire-like objects detected (very high breeding risk)")
            
            # NEW: Transport and vehicle breeding sites
            if container_analysis.get('boat_hull_detected', False):
                base_confidence += 0.12
                reasoning.append("Boat hull with water accumulation (high volume breeding)")
            
            if container_analysis.get('vehicle_water_pooling', 0) > 0.1:
                base_confidence += 0.1
                reasoning.append("Vehicle water pooling detected")
            
            # NEW: Industrial and commercial sites
            if container_analysis.get('large_drum_containers', 0) > 0:
                base_confidence += 0.1
                reasoning.append("Large industrial containers detected")
            
            if water_analysis.get('industrial_contaminated', False):
                base_confidence += 0.08
                reasoning.append("Industrial contamination indicates stagnant conditions")
            
            # NEW: Agricultural and rural sites
            if container_analysis.get('fish_pond_large', False) and not water_analysis.get('fish_present', False):
                base_confidence += 0.15
                reasoning.append("Large unused fish pond (massive breeding potential)")
            
            if container_analysis.get('palm_tree_spathes', 0) > 0:
                base_confidence += 0.12
                reasoning.append("Palm tree spathes (natural perfect breeding containers)")
            
            # NEW: Hidden and unusual containers
            if container_analysis.get('footwear_containers', 0) > 0:
                base_confidence += 0.14
                reasoning.append("Footwear containers (perfect small breeding sites)")
            
            if container_analysis.get('toy_containers', 0) > 0:
                base_confidence += 0.11
                reasoning.append("Toy containers in outdoor environment")
            
            # NEW: Urban community sites
            if container_analysis.get('public_sanitation', False):
                base_confidence += 0.13
                reasoning.append("Public sanitation facilities (high contamination risk)")
            
            if container_analysis.get('waste_containers', 0) > 0:
                base_confidence += 0.09
                reasoning.append("Waste containers with water accumulation")
            
            confidence = min(base_confidence, 0.98)  # Cap at 98%
        
        # ENHANCED POTENTIAL BREEDING SITE CLASSIFICATION
        elif self._assess_potential_breeding_risk(water_analysis, container_analysis, environmental_context):
            category = 'potential'
            base_confidence = 0.6
            reasoning.append("Conditions likely to support breeding site development")
            
            # NEW: Potential risk factors
            if container_analysis.get('abandoned_vehicle_score', 0) > 0.5:
                base_confidence += 0.15
                reasoning.append("Abandoned vehicle with high water collection potential")
            
            if environmental_context.get('industrial_environment', False):
                base_confidence += 0.1
                reasoning.append("Industrial environment with multiple container risks")
            
            if container_analysis.get('temporary_structures', 0) > 0:
                base_confidence += 0.08
                reasoning.append("Temporary structures likely to collect water")
            
            confidence = min(base_confidence, 0.85)
        
        # COMPREHENSIVE NOT HOTSPOT CLASSIFICATION
        elif self._comprehensive_exclusion_check(water_analysis, container_analysis, environmental_context):
            category = 'not_hotspot'
            base_confidence = 0.9
            exclusion_reasons = []
            
            # NEW: Flowing water exclusions
            if self._detect_flowing_water_indicators(water_analysis, environmental_context):
                exclusion_reasons.append("Flowing water detected (not suitable for breeding)")
                base_confidence += 0.05
            
            # NEW: Maintained water systems
            if self._detect_maintained_water_systems(water_analysis, environmental_context):
                exclusion_reasons.append("Maintained water system with treatment/circulation")
                base_confidence += 0.04
            
            # NEW: Wrong scale exclusions
            if self._detect_wrong_scale_water_bodies(container_analysis, environmental_context):
                exclusion_reasons.append("Water body too large/managed for mosquito breeding")
                base_confidence += 0.03
            
            # NEW: Dry surface confirmations
            if environmental_context.get('quick_drainage_indicators', False):
                exclusion_reasons.append("Surface shows rapid drainage characteristics")
                base_confidence += 0.02
            
            reasoning.extend(exclusion_reasons)
            confidence = min(base_confidence, 0.95)
        
        # ENHANCED UNCERTAIN CLASSIFICATION
        else:
            category = 'uncertain'
            uncertainty_factors = []
            
            # NEW: Ambiguity reasons
            if environmental_context.get('poor_visibility', False):
                uncertainty_factors.append("Poor image visibility affects assessment")
            
            if water_analysis.get('water_percentage', 0) > 0 and water_analysis.get('water_percentage', 0) < 0.05:
                uncertainty_factors.append("Minimal water detection - unclear significance")
            
            if container_analysis.get('partially_obscured_containers', 0) > 0:
                uncertainty_factors.append("Containers partially hidden - water status unclear")
            
            reasoning.extend(uncertainty_factors)
            confidence = 0.3 + (len(uncertainty_factors) * 0.05)  # More factors = slightly higher uncertainty confidence
        
        return {
            'category': category,
            'confidence': round(confidence, 3),
            'reasoning': reasoning,
            'detailed_analysis': {
                'water_features': water_analysis,
                'container_features': container_analysis,
                'environmental_context': environmental_context,
                'stagnation_assessment': stagnation_analysis
            }
        }
    
    # Helper methods for enhanced classification
    def _calculate_contamination_confidence_boost(self, water_analysis: Dict) -> float:
        """Calculate confidence boost based on contamination types"""
        boost = 0.0
        
        if water_analysis.get('contamination_detected', False):
            boost += 0.05
            
        # Specific contamination types indicate stagnation
        if water_analysis.get('rusty_contaminated', False):
            boost += 0.03
        if water_analysis.get('sewage_contaminated', False):
            boost += 0.08  # Very high breeding risk
        if water_analysis.get('oil_contaminated', False):
            boost += 0.02  # Some mosquitoes can breed in oil
            
        return min(boost, 0.15)
    
    def _calculate_container_type_confidence(self, container_analysis: Dict) -> float:
        """Calculate confidence based on container types"""
        boost = 0.0
        
        # Small containers are often perfect for breeding
        small_containers = container_analysis.get('small_containers', 0)
        if small_containers > 0:
            boost += min(small_containers * 0.02, 0.08)
            
        return boost
    
    def _calculate_location_confidence(self, environmental_context: Dict, context: Optional[Dict]) -> float:
        """Calculate confidence based on location context"""
        boost = 0.0
        
        if environmental_context.get('environment_type') in ['urban_residential', 'mixed_urban']:
            boost += 0.03
            
        if context and context.get('location_type') == 'high_dengue_area':
            boost += 0.05
            
        return boost
    
    def _assess_potential_breeding_risk(self, water_analysis: Dict, container_analysis: Dict, environmental_context: Dict) -> bool:
        """Assess if conditions indicate potential breeding risk"""
        
        # Containers present but water not clearly visible
        containers_present = container_analysis.get('total_containers', 0) > 0
        outdoor_environment = environmental_context.get('outdoor_likelihood', 0) > 0.6
        
        # Environmental factors that increase breeding potential
        water_collection_likely = (
            environmental_context.get('recent_rain_indicators', False) or
            environmental_context.get('water_collection_surfaces', 0) > 0.3
        )
        
        return containers_present and outdoor_environment and water_collection_likely
    
    def _comprehensive_exclusion_check(self, water_analysis: Dict, container_analysis: Dict, environmental_context: Dict) -> bool:
        """Comprehensive check for exclusion criteria"""
        
        # Clear non-breeding indicators
        no_water_containers = (not water_analysis.get('water_detected', False) and 
                             container_analysis.get('total_containers', 0) == 0)
        
        # Flowing water systems
        flowing_water = self._detect_flowing_water_indicators(water_analysis, environmental_context)
        
        # Maintained systems
        maintained_systems = self._detect_maintained_water_systems(water_analysis, environmental_context)
        
        # Wrong scale
        wrong_scale = self._detect_wrong_scale_water_bodies(container_analysis, environmental_context)
        
        return no_water_containers or flowing_water or maintained_systems or wrong_scale
    
    def _detect_flowing_water_indicators(self, water_analysis: Dict, environmental_context: Dict) -> bool:
        """Detect indicators of flowing water"""
        # Check for river/stream characteristics
        large_water_body = water_analysis.get('water_percentage', 0) > 0.5
        natural_environment = environmental_context.get('natural_environment_score', 0) > 0.7
        
        return large_water_body and natural_environment
    
    def _detect_maintained_water_systems(self, water_analysis: Dict, environmental_context: Dict) -> bool:
        """Detect maintained water systems like pools, fountains"""
        # Check for clear, treated water in structured environments
        clear_water = water_analysis.get('blue_water_present', False)
        structured_environment = environmental_context.get('structured_environment_score', 0) > 0.6
        
        return clear_water and structured_environment
    
    def _detect_wrong_scale_water_bodies(self, container_analysis: Dict, environmental_context: Dict) -> bool:
        """Detect water bodies that are too large for mosquito breeding"""
        very_large_containers = container_analysis.get('very_large_water_bodies', 0) > 0
        industrial_scale = environmental_context.get('industrial_environment', False)
        
        return very_large_containers or industrial_scale
    
    def _check_exclusions(self, water_analysis: Dict, environmental_context: Dict) -> bool:
        """Check for specific exclusion criteria"""
        
        # Large natural water bodies (low breeding risk)
        if (water_analysis['water_percentage'] > 0.6 and
            environmental_context['environment_type'] == 'garden_natural'):
            return True
        
        # Clean flowing water indicators
        if (water_analysis['blue_water_present'] and 
            water_analysis['water_percentage'] > 0.4 and
            not water_analysis['stagnant_indicators']):
            return True
        
        return False
    
    def _identify_risk_factors(self, water_analysis: Dict, container_analysis: Dict, 
                              stagnation_analysis: Dict) -> List[str]:
        """Identify specific risk factors present"""
        risk_factors = []
        
        if container_analysis['circular_containers'] > 0:
            risk_factors.append("Circular containers (buckets/pots)")
        
        if container_analysis['tire_like_objects'] > 0:
            risk_factors.append("Tire-like objects")
        
        if stagnation_analysis['stagnation_risk'] > 0.6:
            risk_factors.append("High stagnation potential")
        
        if water_analysis['stagnant_indicators']:
            risk_factors.append("Stagnant water indicators")
        
        if len(water_analysis['water_regions']) > 2:
            risk_factors.append("Multiple water collection points")
        
        return risk_factors
    
    def _adjust_confidence_based_on_quality(self, base_confidence: float, 
                                           water_analysis: Dict, 
                                           container_analysis: Dict) -> float:
        """Adjust confidence based on detection quality"""
        
        # High-quality detections boost confidence
        if (water_analysis['water_percentage'] > 0.2 and 
            container_analysis['total_containers'] > 1):
            base_confidence += 0.1
        
        # Low-quality detections reduce confidence
        elif (water_analysis['water_percentage'] < 0.05 and 
              container_analysis['total_containers'] == 0):
            base_confidence -= 0.2
        
        return max(0.1, min(1.0, base_confidence))
    
    def _create_result(self, category: str, confidence: float, message: str) -> Dict[str, Any]:
        """Create standardized result format"""
        return {
            'category': category,
            'category_name': self.categories.get(category, 'Unknown'),
            'confidence': confidence,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'analysis_version': '2.0.0'
        }
    
    def _create_detailed_result(self, classification: Dict, water_analysis: Dict,
                               container_analysis: Dict, environmental_context: Dict,
                               stagnation_analysis: Dict, quality_check: Dict) -> Dict[str, Any]:
        """Create comprehensive analysis result"""
        
        result = self._create_result(
            classification['category'],
            classification['confidence'],
            f"Classification: {self.categories[classification['category']]}"
        )
        
        # Add detailed analysis
        result.update({
            'detailed_analysis': {
                'classification': classification,
                'water_detection': {
                    'water_present': water_analysis['water_detected'],
                    'water_percentage': round(water_analysis['water_percentage'] * 100, 2),
                    'water_type': 'stagnant' if water_analysis['stagnant_indicators'] else 'clean',
                    'regions_count': len(water_analysis['water_regions'])
                },
                'container_detection': {
                    'containers_found': container_analysis['total_containers'],
                    'circular_containers': container_analysis['circular_containers'],
                    'rectangular_containers': container_analysis['rectangular_containers'],
                    'tire_objects': container_analysis['tire_like_objects']
                },
                'environment': {
                    'type': environmental_context['environment_type'],
                    'outdoor_likelihood': round(environmental_context['outdoor_likelihood'] * 100, 2),
                    'urban_vs_natural': environmental_context['urban_vs_natural']
                },
                'stagnation_assessment': {
                    'risk_level': 'high' if stagnation_analysis['stagnation_risk'] > 0.7 else 
                               'medium' if stagnation_analysis['stagnation_risk'] > 0.4 else 'low',
                    'risk_score': round(stagnation_analysis['stagnation_risk'] * 100, 2),
                    'indicators': stagnation_analysis['indicators']
                },
                'image_quality': {
                    'score': round(quality_check['score'] * 100, 2),
                    'brightness': quality_check.get('brightness', 0),
                    'sharpness': quality_check.get('sharpness', 0)
                }
            },
            'risk_factors': classification['risk_factors'],
            'recommendations': self._generate_recommendations(classification['category'], classification['risk_factors'])
        })
        
        return result
    
    def _generate_recommendations(self, category: str, risk_factors: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if category == 'hotspot':
            recommendations.extend([
                "🚨 IMMEDIATE ACTION: Remove or empty water containers",
                "🔄 Clean containers weekly to prevent mosquito breeding",
                "🏠 Check surrounding area for additional breeding sites"
            ])
            
            if "Tire-like objects" in risk_factors:
                recommendations.append("🛞 Store tires indoors or drill drainage holes")
            
            if "High stagnation potential" in risk_factors:
                recommendations.append("💧 Improve drainage or cover water storage")
        
        elif category == 'potential':
            recommendations.extend([
                "⚠️ Monitor area regularly, especially after rain",
                "🔍 Check containers for water accumulation",
                "🏡 Consider relocating containers to covered areas"
            ])
        
        elif category == 'not_hotspot':
            recommendations.extend([
                "✅ Area appears low-risk for dengue breeding",
                "🔄 Continue regular monitoring as conditions change"
            ])
        
        elif category == 'uncertain':
            recommendations.extend([
                "🤔 Manual inspection recommended",
                "📸 Take additional photos from different angles",
                "👨‍⚕️ Consult with health authorities if unsure"
            ])
        
        return recommendations
    
    def _load_training_data(self) -> Dict:
        """Load training examples and patterns"""
        # This would typically load from a database or file
        # For now, return structure for synthetic training data
        return {
            'hotspot_examples': [],
            'potential_examples': [],
            'not_hotspot_examples': [],
            'uncertain_examples': [],
            'invalid_examples': []
        }
    
    def _initialize_classification_rules(self) -> Dict:
        """Initialize expert-system rules for classification"""
        return {
            'hotspot_rules': [
                "water_detected AND containers_present AND stagnation_risk > 0.5",
                "tire_objects > 0 AND water_percentage > 0.1",
                "circular_containers > 0 AND stagnant_indicators",
                "multiple_water_regions AND urban_environment"
            ],
            'potential_rules': [
                "containers_present AND outdoor_environment AND no_water_visible",
                "depression_shapes AND rain_collection_potential",
                "construction_environment AND water_collection_surfaces"
            ],
            'exclusion_rules': [
                "flowing_water_indicators",
                "large_maintained_water_body",
                "swimming_pool_characteristics",
                "no_containers_and_no_water"
            ]
        }
    
    def train_on_example(self, image_data: str, true_category: str, 
                        feedback: Optional[str] = None) -> Dict[str, Any]:
        """Train the system on a labeled example"""
        try:
            # Analyze the image
            result = self.analyze_breeding_site(image_data)
            predicted_category = result['category']
            
            # Store training example
            training_example = {
                'image_features': result['detailed_analysis'],
                'true_category': true_category,
                'predicted_category': predicted_category,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to appropriate training set
            self.training_examples[f"{true_category}_examples"].append(training_example)
            
            # Calculate accuracy for this category
            category_examples = self.training_examples[f"{true_category}_examples"]
            correct_predictions = sum(1 for ex in category_examples 
                                    if ex['predicted_category'] == true_category)
            accuracy = correct_predictions / len(category_examples) if category_examples else 0
            
            return {
                'success': True,
                'predicted_category': predicted_category,
                'true_category': true_category,
                'correct': predicted_category == true_category,
                'category_accuracy': round(accuracy * 100, 2),
                'total_examples': len(category_examples),
                'message': f"Training example added for {true_category}"
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': "Failed to process training example"
            }
    # =================== NEW COMPREHENSIVE DETECTION METHODS ===================
    
    def _detect_rusty_contaminated_water(self, hsv: np.ndarray) -> np.ndarray:
        """Detect rusty/metal contaminated water"""
        lower_rusty = np.array([5, 50, 50])
        upper_rusty = np.array([15, 255, 200])
        return cv2.inRange(hsv, lower_rusty, upper_rusty)
    
    def _detect_chemical_contaminated_water(self, hsv: np.ndarray) -> np.ndarray:
        """Detect chemical-tinted water"""
        lower_chemical = np.array([45, 30, 30])
        upper_chemical = np.array([75, 200, 255])
        return cv2.inRange(hsv, lower_chemical, upper_chemical)
    
    def _detect_sewage_contaminated_water(self, hsv: np.ndarray) -> np.ndarray:
        """Detect sewage/organic waste contaminated water"""
        lower_sewage = np.array([15, 40, 20])
        upper_sewage = np.array([35, 200, 100])
        return cv2.inRange(hsv, lower_sewage, upper_sewage)
    
    def _detect_oil_surface(self, hsv: np.ndarray, image: np.ndarray) -> np.ndarray:
        """Detect oil/petroleum surface contamination"""
        # Dark colors with rainbow-like iridescence
        lower_oil = np.array([0, 0, 0])
        upper_oil = np.array([180, 50, 50])
        dark_mask = cv2.inRange(hsv, lower_oil, upper_oil)
        
        # Look for color variations that could indicate oil slick
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        gradient_mask = np.uint8(np.abs(gradient) > 30) * 255
        
        return cv2.bitwise_and(dark_mask, gradient_mask)
    
    def _detect_latex_contaminated_water(self, hsv: np.ndarray) -> np.ndarray:
        """Detect latex/rubber contaminated water (whitish)"""
        lower_latex = np.array([0, 0, 200])
        upper_latex = np.array([180, 30, 255])
        return cv2.inRange(hsv, lower_latex, upper_latex)
    
    def _detect_industrial_contaminated_water(self, hsv: np.ndarray, lab: np.ndarray) -> np.ndarray:
        """Detect industrial contamination indicators"""
        # Unusual color combinations that suggest industrial contamination
        unusual_colors1 = cv2.inRange(hsv, np.array([80, 100, 50]), np.array([100, 255, 200]))
        unusual_colors2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 200]))
        
        return cv2.bitwise_or(unusual_colors1, unusual_colors2)
    
    def _analyze_contamination_levels(self, contamination_ratios: Dict[str, float]) -> Dict[str, bool]:
        """Analyze contamination levels and determine significance"""
        return {
            contam_type: ratio > 0.02  # 2% threshold for significance
            for contam_type, ratio in contamination_ratios.items()
        }
    
    def _detect_boat_shapes(self, gray: np.ndarray) -> bool:
        """Detect boat hull shapes"""
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Large enough for boat
                # Check for elongated shape with curved ends
                rect = cv2.minAreaRect(contour)
                (center), (width, height), angle = rect
                aspect_ratio = max(width, height) / min(width, height)
                
                if 2.0 < aspect_ratio < 6.0:  # Boat-like proportions
                    return True
        return False
    
    def _detect_rectangular_vehicle_parts(self, gray: np.ndarray) -> bool:
        """Detect truck bed or vehicle component shapes"""
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 1000 < area < 50000:  # Vehicle component size
                # Check for rectangular shape
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangular
                    return True
        return False
    
    def _count_small_vehicle_parts(self, gray: np.ndarray) -> int:
        """Count small vehicle components like seats, covers"""
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        small_parts = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Small component size
                small_parts += 1
                
        return min(small_parts, 10)  # Cap at 10
    
    def _calculate_abandonment_score(self, image: np.ndarray) -> float:
        """Calculate how abandoned/neglected a vehicle appears"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Look for rust colors
        rust_ratio = cv2.countNonZero(self._detect_rusty_contaminated_water(hsv)) / (image.shape[0] * image.shape[1])
        
        # Look for vegetation overgrowth (green areas)
        vegetation = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        vegetation_ratio = cv2.countNonZero(vegetation) / (image.shape[0] * image.shape[1])
        
        return min(rust_ratio + vegetation_ratio, 1.0)
    
    def _detect_vehicle_water_pooling(self, image: np.ndarray) -> float:
        """Detect water pooling specific to vehicle surfaces"""
        # Look for corner/edge water accumulation patterns
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find corner-like structures
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        
        # Combine with water detection near corners
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        water_mask = self._detect_blue_water(hsv)
        
        # Check water near corners
        corner_threshold = 0.01 * corners.max()
        corner_points = corners > corner_threshold
        
        water_near_corners = cv2.bitwise_and(water_mask, corner_points.astype(np.uint8) * 255)
        
        return cv2.countNonZero(water_near_corners) / (image.shape[0] * image.shape[1])
    
    def _detect_cylindrical_containers(self, gray: np.ndarray, min_size: int = 50) -> int:
        """Detect cylindrical containers like drums, barrels"""
        # Use HoughCircles to detect circular shapes
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                  param1=50, param2=30, minRadius=min_size//2, maxRadius=min_size*3)
        
        if circles is not None:
            return len(circles[0])
        return 0
    
    def _detect_complex_machinery(self, gray: np.ndarray) -> bool:
        """Detect complex industrial machinery shapes"""
        # Look for complex contours with multiple components
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        complex_shapes = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Machinery-sized
                # Check shape complexity
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) > 8:  # Complex shape
                    complex_shapes += 1
                    
        return complex_shapes > 2
    
    def _detect_large_circular_structures(self, gray: np.ndarray) -> bool:
        """Detect large circular structures like cooling tower bases"""
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                                  param1=50, param2=30, minRadius=50, maxRadius=200)
        
        return circles is not None and len(circles[0]) > 0
    
    def _detect_rectangular_structures(self, gray: np.ndarray, aspect_ratio_range: Tuple[float, float] = (1.0, 4.0)) -> int:
        """Detect rectangular structures like shipping containers"""
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        rectangles = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Container-sized
                rect = cv2.minAreaRect(contour)
                (center), (width, height), angle = rect
                aspect_ratio = max(width, height) / min(width, height)
                
                if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                    rectangles += 1
                    
        return rectangles
    
    def _assess_industrial_contamination(self, image: np.ndarray) -> float:
        """Assess level of industrial contamination in image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Look for industrial indicators
        metal_surfaces = self._detect_metal_surfaces(hsv)
        unusual_colors = self._detect_industrial_contaminated_water(hsv, None)
        
        total_pixels = image.shape[0] * image.shape[1]
        contamination_score = (cv2.countNonZero(metal_surfaces) + cv2.countNonZero(unusual_colors)) / total_pixels
        
        return min(contamination_score, 1.0)
    
    def _detect_industrial_environment(self, image: np.ndarray) -> bool:
        """Detect industrial environment context"""
        # Look for industrial indicators: metal, concrete, large structures
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Metal surfaces
        metal_ratio = cv2.countNonZero(self._detect_metal_surfaces(hsv)) / (image.shape[0] * image.shape[1])
        
        # Large structures
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        large_structures = sum(1 for contour in contours if cv2.contourArea(contour) > 10000)
        
        return metal_ratio > 0.1 or large_structures > 3
    
    def _detect_metal_surfaces(self, hsv: np.ndarray) -> np.ndarray:
        """Detect metallic surfaces"""
        lower_metal = np.array([0, 0, 100])
        upper_metal = np.array([180, 30, 200])
        return cv2.inRange(hsv, lower_metal, upper_metal)
    
    # Add more detection methods...
    # (Additional methods would continue here for all the other categories)
    
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics and model performance"""
        stats = {
            'total_examples': 0,
            'category_breakdown': {},
            'accuracy_by_category': {},
            'overall_accuracy': 0.0,
            'training_recommendations': []
        }
        
        total_correct = 0
        total_examples = 0
        
        for category, examples in self.training_examples.items():
            if examples:
                category_name = category.replace('_examples', '')
                correct = sum(1 for ex in examples 
                             if ex['predicted_category'] == ex['true_category'])
                
                stats['category_breakdown'][category_name] = len(examples)
                stats['accuracy_by_category'][category_name] = round(correct / len(examples) * 100, 2)
                
                total_correct += correct
                total_examples += len(examples)
        
        stats['total_examples'] = total_examples
        stats['overall_accuracy'] = round(total_correct / total_examples * 100, 2) if total_examples > 0 else 0
        
        # Generate training recommendations
        stats['training_recommendations'] = self._generate_training_recommendations(stats)
        
        return stats
    
    def _generate_training_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations for improving training"""
        recommendations = []
        
        # Check for insufficient training data
        for category, count in stats['category_breakdown'].items():
            if count < 10:
                recommendations.append(f"Need more {category} examples (currently {count})")
        
        # Check for poor accuracy
        for category, accuracy in stats['accuracy_by_category'].items():
            if accuracy < 70:
                recommendations.append(f"Improve {category} classification (currently {accuracy}%)")
        
        # General recommendations
        if stats['total_examples'] < 50:
            recommendations.append("Collect more training examples for better performance")
        
        if stats['overall_accuracy'] < 80:
            recommendations.append("Review and improve classification rules")
        
        return recommendations

# Create global instance
advanced_breeding_detector = AdvancedBreedingSiteDetector()