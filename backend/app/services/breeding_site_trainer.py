"""
Synthetic Training Data Generator for Dengue Breeding Site Detection
Creates comprehensive training datasets based on expert specifications
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

class BreedingSiteTrainingDataGenerator:
    """Generate synthetic training data for breeding site classification"""
    
    def __init__(self):
        self.training_categories = [
            'hotspot',
            'potential', 
            'not_hotspot',
            'uncertain',
            'invalid'
        ]
        
        self.location_types = [
            'household_domestic',
            'outdoor_waste_garbage',
            'construction_sites',
            'urban_drainage_infrastructure',
            'natural_semi_natural',
            'agricultural_areas',
            'public_spaces'
        ]
        
        # Initialize comprehensive training scenarios
        self.training_scenarios = self._initialize_training_scenarios()
    
    def _initialize_training_scenarios(self) -> Dict[str, Any]:
        """Initialize comprehensive training scenarios based on expert guidelines"""
        
        scenarios = {
            'hotspot_scenarios': [
                # HOUSEHOLD & DOMESTIC AREAS
                {
                    'description': 'Flower vase with stagnant water and floating leaves',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'cloudy'},
                    'container_type': 'circular_deep',
                    'risk_factors': ['organic_matter', 'indoor_warm', 'undisturbed'],
                    'confidence': 0.95
                },
                {
                    'description': 'Plastic bucket half-filled with rainwater outdoors',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'murky'},
                    'container_type': 'circular_deep',
                    'risk_factors': ['rainwater_collection', 'outdoor_exposure', 'large_volume'],
                    'confidence': 0.9
                },
                {
                    'description': 'Pet water bowl outdoors with old stagnant water',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'dirty'},
                    'container_type': 'circular_shallow',
                    'risk_factors': ['organic_contamination', 'outdoor_location', 'neglected'],
                    'confidence': 0.85
                },
                {
                    'description': 'Plant saucer under pot with standing water',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'clear_to_murky'},
                    'container_type': 'circular_shallow',
                    'risk_factors': ['regular_watering', 'warm_environment', 'small_volume'],
                    'confidence': 0.88
                },
                {
                    'description': 'Air conditioner drip tray with accumulated water',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'clear'},
                    'container_type': 'rectangular_shallow',
                    'risk_factors': ['continuous_dripping', 'warm_location', 'hidden_area'],
                    'confidence': 0.92
                },
                
                # OUTDOOR WASTE & GARBAGE
                {
                    'description': 'Discarded tire lying flat filled with rainwater',
                    'location': 'outdoor_waste_garbage',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'dark_murky'},
                    'container_type': 'tire_circular',
                    'risk_factors': ['rubber_material', 'perfect_breeding_shape', 'heat_retention'],
                    'confidence': 0.98
                },
                {
                    'description': 'Empty plastic bottles and cans with rainwater',
                    'location': 'outdoor_waste_garbage',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'varies'},
                    'container_type': 'cylindrical_narrow',
                    'risk_factors': ['multiple_containers', 'irregular_cleaning', 'weather_exposed'],
                    'confidence': 0.87
                },
                {
                    'description': 'Coconut shells with water collected inside',
                    'location': 'outdoor_waste_garbage',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'brown_organic'},
                    'container_type': 'natural_bowl',
                    'risk_factors': ['organic_container', 'natural_nutrients', 'tropical_climate'],
                    'confidence': 0.9
                },
                
                # CONSTRUCTION SITES
                {
                    'description': 'Open cement tank with collected rainwater',
                    'location': 'construction_sites',
                    'water_features': {'stagnant': True, 'volume': 'very_large', 'clarity': 'cement_muddy'},
                    'container_type': 'rectangular_deep',
                    'risk_factors': ['construction_materials', 'temporary_structure', 'large_surface'],
                    'confidence': 0.85
                },
                {
                    'description': 'Plastic sheets with water pooled on construction materials',
                    'location': 'construction_sites',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'dirty'},
                    'container_type': 'irregular_depression',
                    'risk_factors': ['temporary_pooling', 'construction_debris', 'uneven_surfaces'],
                    'confidence': 0.8
                },
                
                # URBAN DRAINAGE
                {
                    'description': 'Clogged roadside drain with stagnant dirty water',
                    'location': 'urban_drainage_infrastructure',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'very_dirty'},
                    'container_type': 'rectangular_drain',
                    'risk_factors': ['urban_pollution', 'blocked_flow', 'organic_debris'],
                    'confidence': 0.93
                },
                {
                    'description': 'Blocked rain gutters with green stagnant water',
                    'location': 'urban_drainage_infrastructure',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'green_algae'},
                    'container_type': 'elongated_channel',
                    'risk_factors': ['leaf_blockage', 'elevated_position', 'algae_growth'],
                    'confidence': 0.91
                },
                {
                    'description': 'Potholes in road filled with stagnant rainwater',
                    'location': 'urban_drainage_infrastructure',
                    'water_features': {'stagnant': True, 'volume': 'small_to_medium', 'clarity': 'muddy'},
                    'container_type': 'irregular_depression',
                    'risk_factors': ['road_debris', 'poor_drainage', 'vehicle_contamination'],
                    'confidence': 0.82
                },
                
                # NATURAL & SEMI-NATURAL
                {
                    'description': 'Tree holes filled with rainwater and organic matter',
                    'location': 'natural_semi_natural',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'organic_brown'},
                    'container_type': 'natural_cavity',
                    'risk_factors': ['natural_breeding_site', 'organic_nutrients', 'protected_location'],
                    'confidence': 0.94
                },
                {
                    'description': 'Bamboo stumps cut and filled with water',
                    'location': 'natural_semi_natural',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'clear_to_brown'},
                    'container_type': 'cylindrical_natural',
                    'risk_factors': ['perfect_breeding_cylinder', 'natural_material', 'shaded_area'],
                    'confidence': 0.96
                },
                
                # AGRICULTURAL AREAS
                {
                    'description': 'Irrigation channel with stagnant water between crops',
                    'location': 'agricultural_areas',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'muddy_organic'},
                    'container_type': 'elongated_channel',
                    'risk_factors': ['agricultural_runoff', 'seasonal_stagnation', 'fertilizer_nutrients'],
                    'confidence': 0.8
                },
                {
                    'description': 'Rubber tapping cups after rain collection',
                    'location': 'agricultural_areas',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'latex_contaminated'},
                    'container_type': 'small_cup',
                    'risk_factors': ['multiple_small_containers', 'tree_shade', 'regular_collection_cycle'],
                    'confidence': 0.88
                },
                
                # TRANSPORT & VEHICLES - NEW COMPREHENSIVE CATEGORY
                {
                    'description': 'Boat or canoe with rainwater collected inside hull',
                    'location': 'transport_vehicles',
                    'water_features': {'stagnant': True, 'volume': 'very_large', 'clarity': 'rainwater_clear'},
                    'container_type': 'boat_hull',
                    'risk_factors': ['large_surface_area', 'infrequent_use', 'outdoor_exposure'],
                    'confidence': 0.95
                },
                {
                    'description': 'Truck bed with water pooling in corners',
                    'location': 'transport_vehicles',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'dirty_rainwater'},
                    'container_type': 'vehicle_bed',
                    'risk_factors': ['metal_surface', 'corner_pooling', 'commercial_neglect'],
                    'confidence': 0.88
                },
                {
                    'description': 'Motorbike seat covers forming puddles after rain',
                    'location': 'transport_vehicles',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'clear_rainwater'},
                    'container_type': 'vehicle_component',
                    'risk_factors': ['plastic_surface', 'frequent_exposure', 'urban_pollution'],
                    'confidence': 0.82
                },
                {
                    'description': 'Abandoned rickshaw cart holding accumulated water',
                    'location': 'transport_vehicles',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'stagnant_dirty'},
                    'container_type': 'abandoned_vehicle',
                    'risk_factors': ['no_maintenance', 'permanent_stagnation', 'urban_waste'],
                    'confidence': 0.93
                },
                
                # COMMERCIAL & INDUSTRIAL - NEW COMPREHENSIVE CATEGORY
                {
                    'description': 'Cooling tower with stagnant water at base (poorly maintained)',
                    'location': 'commercial_industrial',
                    'water_features': {'stagnant': True, 'volume': 'very_large', 'clarity': 'industrial_contaminated'},
                    'container_type': 'industrial_equipment',
                    'risk_factors': ['warm_temperature', 'chemical_contamination', 'large_breeding_area'],
                    'confidence': 0.91
                },
                {
                    'description': 'Factory yard unused machinery trapping rainwater',
                    'location': 'commercial_industrial',
                    'water_features': {'stagnant': True, 'volume': 'variable', 'clarity': 'rusty_contaminated'},
                    'container_type': 'machinery',
                    'risk_factors': ['metal_surfaces', 'irregular_drainage', 'industrial_chemicals'],
                    'confidence': 0.86
                },
                {
                    'description': 'Open oil drums or chemical containers in storage yards',
                    'location': 'commercial_industrial',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'chemical_contaminated'},
                    'container_type': 'industrial_drum',
                    'risk_factors': ['toxic_environment', 'long_term_storage', 'resistant_mosquitoes'],
                    'confidence': 0.94
                },
                {
                    'description': 'Shipping container roof depressions with rainwater pools',
                    'location': 'commercial_industrial',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'metal_contaminated'},
                    'container_type': 'shipping_container',
                    'risk_factors': ['metal_surface', 'roof_pooling', 'port_environment'],
                    'confidence': 0.89
                },
                
                # RURAL & AGRICULTURAL EXPANSION - NEW COMPREHENSIVE CATEGORY
                {
                    'description': 'Unused fish pond with stagnant dirty water (no fish)',
                    'location': 'rural_agricultural_expansion',
                    'water_features': {'stagnant': True, 'volume': 'massive', 'clarity': 'green_algae'},
                    'container_type': 'pond',
                    'risk_factors': ['no_fish_predators', 'organic_matter', 'perfect_temperature'],
                    'confidence': 0.97
                },
                {
                    'description': 'Irrigation tank not drained or covered properly',
                    'location': 'rural_agricultural_expansion',
                    'water_features': {'stagnant': True, 'volume': 'very_large', 'clarity': 'agricultural_runoff'},
                    'container_type': 'irrigation_tank',
                    'risk_factors': ['seasonal_stagnation', 'fertilizer_nutrients', 'open_top'],
                    'confidence': 0.92
                },
                {
                    'description': 'Palm tree spathes (leaf casing) collecting water naturally',
                    'location': 'rural_agricultural_expansion',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'natural_clear'},
                    'container_type': 'natural_plant',
                    'risk_factors': ['perfect_breeding_size', 'natural_container', 'tropical_climate'],
                    'confidence': 0.96
                },
                {
                    'description': 'Unused livestock water trough filled with rainwater',
                    'location': 'rural_agricultural_expansion',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'organic_contaminated'},
                    'container_type': 'livestock_equipment',
                    'risk_factors': ['animal_waste', 'irregular_cleaning', 'farm_neglect'],
                    'confidence': 0.90
                },
                {
                    'description': 'Fertilizer/grain sacks forming water pockets on plastic surfaces',
                    'location': 'rural_agricultural_expansion',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'chemical_tinted'},
                    'container_type': 'agricultural_storage',
                    'risk_factors': ['plastic_material', 'chemical_contamination', 'outdoor_storage'],
                    'confidence': 0.85
                },
                
                # COMMUNITY & URBAN SETTINGS - NEW COMPREHENSIVE CATEGORY
                {
                    'description': 'Street vendor umbrella or tarpaulin sagging with rainwater',
                    'location': 'community_urban',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'urban_polluted'},
                    'container_type': 'temporary_structure',
                    'risk_factors': ['temporary_pooling', 'frequent_formation', 'urban_pollution'],
                    'confidence': 0.87
                },
                {
                    'description': 'Public toilet broken tanks or buckets left unemptied',
                    'location': 'community_urban',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'sewage_contaminated'},
                    'container_type': 'public_sanitation',
                    'risk_factors': ['human_waste', 'poor_maintenance', 'public_health_risk'],
                    'confidence': 0.95
                },
                {
                    'description': 'Market waste bins with water accumulated in bottoms',
                    'location': 'community_urban',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'waste_contaminated'},
                    'container_type': 'waste_container',
                    'risk_factors': ['food_waste', 'organic_nutrients', 'regular_contamination'],
                    'confidence': 0.91
                },
                {
                    'description': 'Park trash cans with open tops collecting rainwater',
                    'location': 'community_urban',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'mixed_waste'},
                    'container_type': 'public_waste',
                    'risk_factors': ['mixed_waste', 'public_access', 'weather_exposure'],
                    'confidence': 0.88
                },
                {
                    'description': 'Concrete flowerbeds or park planters with clogged drainage',
                    'location': 'community_urban',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'plant_debris'},
                    'container_type': 'landscape_feature',
                    'risk_factors': ['drainage_failure', 'plant_matter', 'landscaping_neglect'],
                    'confidence': 0.89
                },
                
                # HIDDEN/UNUSUAL PLACES - NEW COMPREHENSIVE CATEGORY
                {
                    'description': 'Old shoes, boots, or gloves left outside filled with water',
                    'location': 'hidden_unusual',
                    'water_features': {'stagnant': True, 'volume': 'very_small', 'clarity': 'footwear_contaminated'},
                    'container_type': 'footwear',
                    'risk_factors': ['perfect_breeding_size', 'hidden_location', 'frequently_overlooked'],
                    'confidence': 0.96
                },
                {
                    'description': 'Hollow bamboo fencing or decorative poles with water inside',
                    'location': 'hidden_unusual',
                    'water_features': {'stagnant': True, 'volume': 'small', 'clarity': 'natural_clear'},
                    'container_type': 'decorative_bamboo',
                    'risk_factors': ['natural_container', 'difficult_cleaning', 'multiple_segments'],
                    'confidence': 0.93
                },
                {
                    'description': 'Plastic toys (cars, dolls, buckets) abandoned outdoors with water',
                    'location': 'hidden_unusual',
                    'water_features': {'stagnant': True, 'volume': 'very_small', 'clarity': 'toy_contaminated'},
                    'container_type': 'toy',
                    'risk_factors': ['child_environment', 'frequently_overlooked', 'plastic_material'],
                    'confidence': 0.90
                },
                {
                    'description': 'Old mattresses or furniture absorbing rain forming water pools',
                    'location': 'hidden_unusual',
                    'water_features': {'stagnant': True, 'volume': 'medium', 'clarity': 'fabric_contaminated'},
                    'container_type': 'furniture',
                    'risk_factors': ['absorbent_material', 'long_term_pooling', 'waste_environment'],
                    'confidence': 0.85
                },
                {
                    'description': 'Retired aircraft in graveyards with open compartments holding water',
                    'location': 'hidden_unusual',
                    'water_features': {'stagnant': True, 'volume': 'very_large', 'clarity': 'aviation_contaminated'},
                    'container_type': 'aircraft',
                    'risk_factors': ['multiple_compartments', 'remote_location', 'aviation_fluids'],
                    'confidence': 0.94
                }
            ],
            
            'potential_scenarios': [
                {
                    'description': 'Empty plastic bucket placed outdoors',
                    'location': 'household_domestic',
                    'container_type': 'circular_deep',
                    'risk_factors': ['rain_collection_potential', 'outdoor_placement', 'large_capacity'],
                    'confidence': 0.7
                },
                {
                    'description': 'Dry tire lying flat in yard',
                    'location': 'outdoor_waste_garbage',
                    'container_type': 'tire_circular',
                    'risk_factors': ['perfect_breeding_shape_when_wet', 'outdoor_exposure', 'heat_retention'],
                    'confidence': 0.75
                },
                {
                    'description': 'Construction site with uneven surfaces and materials',
                    'location': 'construction_sites',
                    'container_type': 'irregular_multiple',
                    'risk_factors': ['water_collection_surfaces', 'temporary_structures', 'debris_accumulation'],
                    'confidence': 0.65
                },
                {
                    'description': 'Roof gutters that appear blocked but no visible water',
                    'location': 'urban_drainage_infrastructure',
                    'container_type': 'elongated_channel',
                    'risk_factors': ['drainage_blockage_potential', 'elevated_position', 'leaf_accumulation'],
                    'confidence': 0.68
                },
                {
                    'description': 'Garden area with multiple empty containers',
                    'location': 'household_domestic',
                    'container_type': 'multiple_varied',
                    'risk_factors': ['multiple_collection_points', 'garden_watering', 'outdoor_storage'],
                    'confidence': 0.72
                }
            ],
            
            'not_hotspot_scenarios': [
                # DRY/EMPTY SURFACES
                {
                    'description': 'Empty clean flower vase indoors',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'circular_deep',
                    'exclusion_reasons': ['no_water', 'indoor_location', 'regularly_cleaned'],
                    'confidence': 0.95
                },
                {
                    'description': 'Dry concrete pavement with no puddles',
                    'location': 'urban_drainage_infrastructure', 
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'flat_surface',
                    'exclusion_reasons': ['no_water_retention', 'good_drainage', 'solid_surface'],
                    'confidence': 0.98
                },
                
                # MOVING OR MAINTAINED WATER
                {
                    'description': 'Flowing river with clear moving water',
                    'location': 'natural_semi_natural',
                    'water_features': {'stagnant': False, 'volume': 'very_large', 'clarity': 'clear'},
                    'container_type': 'natural_flowing',
                    'exclusion_reasons': ['flowing_water', 'too_large', 'natural_ecosystem'],
                    'confidence': 0.99
                },
                {
                    'description': 'Swimming pool with clear chlorinated water',
                    'location': 'public_spaces',
                    'water_features': {'stagnant': False, 'volume': 'very_large', 'clarity': 'clear'},
                    'container_type': 'rectangular_maintained',
                    'exclusion_reasons': ['chlorinated_water', 'maintained', 'circulation_system'],
                    'confidence': 0.99
                },
                {
                    'description': 'Fish pond with aquatic life and circulation',
                    'location': 'natural_semi_natural',
                    'water_features': {'stagnant': False, 'volume': 'large', 'clarity': 'clear_to_green'},
                    'container_type': 'natural_maintained',
                    'exclusion_reasons': ['fish_present', 'ecosystem_balance', 'natural_predators'],
                    'confidence': 0.94
                },
                
                # IRRELEVANT CONTENT
                {
                    'description': 'Parked car in driveway',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'vehicle',
                    'exclusion_reasons': ['irrelevant_object', 'no_water_retention', 'solid_surface'],
                    'confidence': 0.99
                },
                {
                    'description': 'Person standing in garden',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'human',
                    'exclusion_reasons': ['human_subject', 'irrelevant_to_breeding', 'living_being'],
                    'confidence': 0.99
                },
                {
                    'description': 'Clean dry dishes on kitchen counter',
                    'location': 'household_domestic',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'kitchen_items',
                    'exclusion_reasons': ['indoor_location', 'clean_dry_surfaces', 'food_preparation_area'],
                    'confidence': 0.97
                },
                
                # NEW COMPREHENSIVE EXCLUSIONS - FLOWING OR LARGE WATER BODIES
                {
                    'description': 'Waterfall with fast-moving water',
                    'location': 'natural_large_water',
                    'water_features': {'stagnant': False, 'volume': 'massive', 'clarity': 'clear'},
                    'container_type': 'natural_flowing',
                    'exclusion_reasons': ['high_water_velocity', 'natural_ecosystem', 'too_large_scale'],
                    'confidence': 0.99
                },
                {
                    'description': 'Irrigation channel with constant water flow',
                    'location': 'agricultural_flowing',
                    'water_features': {'stagnant': False, 'volume': 'large', 'clarity': 'muddy_flowing'},
                    'container_type': 'channel_flowing',
                    'exclusion_reasons': ['water_current', 'agricultural_irrigation', 'maintained_flow'],
                    'confidence': 0.96
                },
                {
                    'description': 'Ocean coastline with waves',
                    'location': 'natural_large_water',
                    'water_features': {'stagnant': False, 'volume': 'massive', 'clarity': 'saltwater'},
                    'container_type': 'natural_massive',
                    'exclusion_reasons': ['saltwater', 'too_large', 'constant_movement'],
                    'confidence': 0.99
                },
                {
                    'description': 'Large lake or reservoir with circulation',
                    'location': 'natural_large_water',
                    'water_features': {'stagnant': False, 'volume': 'massive', 'clarity': 'clear_deep'},
                    'container_type': 'natural_managed',
                    'exclusion_reasons': ['too_large_volume', 'natural_circulation', 'ecosystem_present'],
                    'confidence': 0.98
                },
                
                # NEW COMPREHENSIVE EXCLUSIONS - MAINTAINED MAN-MADE WATER
                {
                    'description': 'Swimming pool with chlorine treatment and filtration',
                    'location': 'maintained_facilities',
                    'water_features': {'stagnant': False, 'volume': 'large', 'clarity': 'crystal_clear'},
                    'container_type': 'pool_maintained',
                    'exclusion_reasons': ['chemical_treatment', 'filtration_system', 'regular_maintenance'],
                    'confidence': 0.99
                },
                {
                    'description': 'Water tank with sealed lid and regular use',
                    'location': 'household_maintained',
                    'water_features': {'stagnant': False, 'volume': 'large', 'clarity': 'treated_water'},
                    'container_type': 'sealed_tank',
                    'exclusion_reasons': ['sealed_container', 'regular_use', 'treated_water'],
                    'confidence': 0.97
                },
                {
                    'description': 'Fish pond stocked with larvivorous fish (guppies, tilapia)',
                    'location': 'aquaculture_maintained',
                    'water_features': {'stagnant': True, 'volume': 'large', 'clarity': 'fish_ecosystem'},
                    'container_type': 'fish_pond_active',
                    'exclusion_reasons': ['fish_predators', 'managed_ecosystem', 'biological_control'],
                    'confidence': 0.95
                },
                {
                    'description': 'Decorative fountain with circulating pump',
                    'location': 'public_maintained',
                    'water_features': {'stagnant': False, 'volume': 'medium', 'clarity': 'circulating_clear'},
                    'container_type': 'fountain_active',
                    'exclusion_reasons': ['water_circulation', 'pump_system', 'maintained_feature'],
                    'confidence': 0.96
                },
                
                # NEW COMPREHENSIVE EXCLUSIONS - DRY/NON-WATER SURFACES
                {
                    'description': 'Dry concrete floor after rain (absorbed quickly)',
                    'location': 'urban_surfaces',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'concrete_dry',
                    'exclusion_reasons': ['quick_drainage', 'no_water_retention', 'dry_surface'],
                    'confidence': 0.98
                },
                {
                    'description': 'Open field with soil that absorbs water rapidly',
                    'location': 'natural_drainage',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'soil_draining',
                    'exclusion_reasons': ['natural_drainage', 'soil_absorption', 'no_container_shape'],
                    'confidence': 0.97
                },
                {
                    'description': 'Dry trash pile without visible containers',
                    'location': 'waste_dry',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'waste_dry',
                    'exclusion_reasons': ['no_water_present', 'dry_materials', 'no_containers'],
                    'confidence': 0.94
                },
                {
                    'description': 'Sunlit roof that dries quickly with no depressions',
                    'location': 'building_rooftop',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'roof_flat',
                    'exclusion_reasons': ['direct_sunlight', 'quick_evaporation', 'no_water_traps'],
                    'confidence': 0.96
                },
                
                # NEW COMPREHENSIVE EXCLUSIONS - WRONG SCALE/SIZE
                {
                    'description': 'Large dam or hydroelectric plant',
                    'location': 'industrial_massive',
                    'water_features': {'stagnant': False, 'volume': 'massive', 'clarity': 'deep_water'},
                    'container_type': 'dam_structure',
                    'exclusion_reasons': ['massive_scale', 'industrial_facility', 'engineered_flow'],
                    'confidence': 0.99
                },
                {
                    'description': 'Industrial cooling pond with constant monitoring',
                    'location': 'industrial_managed',
                    'water_features': {'stagnant': False, 'volume': 'very_large', 'clarity': 'industrial_clear'},
                    'container_type': 'cooling_pond',
                    'exclusion_reasons': ['industrial_monitoring', 'constant_flow', 'too_large_scale'],
                    'confidence': 0.97
                },
                
                # NEW COMPREHENSIVE EXCLUSIONS - IRRELEVANT OBJECTS
                {
                    'description': 'Pets (dogs, cats) in outdoor setting',
                    'location': 'household_outdoor',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'living_animal',
                    'exclusion_reasons': ['living_creature', 'irrelevant_subject', 'no_water_container'],
                    'confidence': 0.99
                },
                {
                    'description': 'People walking in park or street',
                    'location': 'public_spaces',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'human_subjects',
                    'exclusion_reasons': ['human_subjects', 'irrelevant_to_breeding', 'no_containers'],
                    'confidence': 0.99
                },
                {
                    'description': 'Garden with healthy drainage and no standing water',
                    'location': 'landscaped_maintained',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'garden_maintained',
                    'exclusion_reasons': ['good_drainage', 'maintained_landscaping', 'no_water_accumulation'],
                    'confidence': 0.95
                },
                {
                    'description': 'Various animals in natural habitat (no containers)',
                    'location': 'natural_wildlife',
                    'water_features': {'stagnant': False, 'volume': 'none', 'clarity': 'none'},
                    'container_type': 'wildlife_subjects',
                    'exclusion_reasons': ['wildlife_focus', 'natural_subjects', 'no_breeding_containers'],
                    'confidence': 0.98
                }
            ],
            
            'uncertain_scenarios': [
                {
                    'description': 'Partially visible container behind vegetation',
                    'location': 'natural_semi_natural',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'unknown'},
                    'container_type': 'partially_obscured',
                    'uncertainty_factors': ['limited_visibility', 'unclear_water_presence', 'vegetation_blocking'],
                    'confidence': 0.4
                },
                {
                    'description': 'Reflective surface that could be water or wet pavement',
                    'location': 'urban_drainage_infrastructure',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'reflective'},
                    'container_type': 'ambiguous_surface',
                    'uncertainty_factors': ['reflection_ambiguity', 'wet_vs_water_unclear', 'surface_type_unclear'],
                    'confidence': 0.35
                },
                {
                    'description': 'Dark shadowed area with possible water accumulation',
                    'location': 'construction_sites',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'too_dark'},
                    'container_type': 'shadow_obscured',
                    'uncertainty_factors': ['poor_lighting', 'shadow_obstruction', 'unclear_surface_type'],
                    'confidence': 0.3
                },
                
                # NEW COMPREHENSIVE AMBIGUOUS SCENARIOS
                {
                    'description': 'Collapsed tent or canopy with possible water inside (angle unclear)',
                    'location': 'outdoor_ambiguous',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'hidden'},
                    'container_type': 'collapsed_structure',
                    'uncertainty_factors': ['structural_collapse', 'hidden_water', 'viewing_angle_poor'],
                    'confidence': 0.4
                },
                {
                    'description': 'Partially blocked roadside drain (water visibility obstructed)',
                    'location': 'urban_infrastructure_ambiguous',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'obscured'},
                    'container_type': 'drain_blocked',
                    'uncertainty_factors': ['partial_blockage', 'water_level_unclear', 'debris_obstruction'],
                    'confidence': 0.45
                },
                {
                    'description': 'Satellite image where surface reflectivity could be puddles or shadows',
                    'location': 'aerial_ambiguous',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'aerial_unclear'},
                    'container_type': 'aerial_surface',
                    'uncertainty_factors': ['satellite_resolution', 'reflection_vs_shadow', 'scale_ambiguity'],
                    'confidence': 0.35
                },
                {
                    'description': 'Thick vegetation where leaves may or may not hold water',
                    'location': 'natural_vegetation_dense',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'vegetation_hidden'},
                    'container_type': 'plant_matter',
                    'uncertainty_factors': ['vegetation_density', 'natural_containers_unclear', 'leaf_water_unknown'],
                    'confidence': 0.42
                },
                {
                    'description': 'Rusted metal drums outdoors (image resolution too low to see water)',
                    'location': 'industrial_ambiguous',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'resolution_limited'},
                    'container_type': 'metal_containers',
                    'uncertainty_factors': ['low_image_resolution', 'rust_obstruction', 'water_level_unclear'],
                    'confidence': 0.38
                },
                {
                    'description': 'Underground or covered areas with unclear water status',
                    'location': 'underground_ambiguous',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'covered'},
                    'container_type': 'covered_containers',
                    'uncertainty_factors': ['covered_surface', 'underground_location', 'access_limited'],
                    'confidence': 0.33
                },
                {
                    'description': 'Weather-damaged structures with potential water collection points',
                    'location': 'storm_damaged',
                    'water_features': {'stagnant': None, 'volume': 'unknown', 'clarity': 'debris_mixed'},
                    'container_type': 'damaged_structure',
                    'uncertainty_factors': ['structural_damage', 'debris_obstruction', 'water_collection_unclear'],
                    'confidence': 0.41
                }
            ],
            
            'invalid_scenarios': [
                {
                    'description': 'Blurred motion photo unusable for analysis',
                    'quality_issues': ['motion_blur', 'out_of_focus', 'unusable'],
                    'confidence': 0.1
                },
                {
                    'description': 'Extremely dark image with no visible details',
                    'quality_issues': ['too_dark', 'no_details_visible', 'poor_lighting'],
                    'confidence': 0.1
                },
                {
                    'description': 'Heavily pixelated or corrupted image data',
                    'quality_issues': ['pixelation', 'data_corruption', 'low_resolution'],
                    'confidence': 0.1
                }
            ]
        }
        
        return scenarios
    
    def generate_comprehensive_training_set(self) -> Dict[str, Any]:
        """Generate comprehensive training dataset"""
        
        training_data = {
            'metadata': {
                'generated_date': datetime.now().isoformat(),
                'version': '1.0.0',
                'total_scenarios': 0,
                'categories': list(self.training_categories),
                'location_types': list(self.location_types),
                'purpose': 'AI training for dengue breeding site detection'
            },
            'training_examples': {},
            'classification_guidelines': self._get_classification_guidelines(),
            'feature_importance': self._get_feature_importance_guidelines()
        }
        
        # Generate examples for each category
        total_scenarios = 0
        for category in self.training_categories:
            scenarios_key = f"{category}_scenarios"
            if scenarios_key in self.training_scenarios:
                category_examples = []
                
                for i, scenario in enumerate(self.training_scenarios[scenarios_key]):
                    example = {
                        'id': f"{category}_{i+1:03d}",
                        'category': category,
                        'scenario': scenario,
                        'training_features': self._extract_training_features(scenario, category),
                        'expected_classification': {
                            'category': category,
                            'confidence_range': [
                                max(0.0, scenario.get('confidence', 0.5) - 0.1),
                                min(1.0, scenario.get('confidence', 0.5) + 0.1)
                            ]
                        }
                    }
                    category_examples.append(example)
                    total_scenarios += 1
                
                training_data['training_examples'][category] = category_examples
        
        training_data['metadata']['total_scenarios'] = total_scenarios
        
        return training_data
    
    def _extract_training_features(self, scenario: Dict, category: str) -> Dict[str, Any]:
        """Extract training features from scenario"""
        features = {
            'visual_features': {},
            'contextual_features': {},
            'risk_indicators': {},
            'exclusion_indicators': {}
        }
        
        # Visual features
        if 'water_features' in scenario:
            features['visual_features']['water'] = scenario['water_features']
        
        if 'container_type' in scenario:
            features['visual_features']['container'] = {
                'type': scenario['container_type'],
                'breeding_suitability': self._assess_container_breeding_suitability(scenario['container_type'])
            }
        
        # Contextual features
        if 'location' in scenario:
            features['contextual_features']['location'] = {
                'type': scenario['location'],
                'outdoor_likelihood': self._assess_outdoor_likelihood(scenario['location']),
                'urban_vs_natural': self._classify_urban_natural(scenario['location'])
            }
        
        # Risk indicators
        if 'risk_factors' in scenario:
            features['risk_indicators'] = {
                'factors': scenario['risk_factors'],
                'risk_score': self._calculate_risk_score(scenario['risk_factors'])
            }
        
        # Exclusion indicators
        if 'exclusion_reasons' in scenario:
            features['exclusion_indicators'] = {
                'reasons': scenario['exclusion_reasons'],
                'exclusion_strength': len(scenario['exclusion_reasons'])
            }
        
        return features
    
    def _assess_container_breeding_suitability(self, container_type: str) -> str:
        """Assess breeding suitability of container type"""
        high_risk = ['tire_circular', 'circular_deep', 'natural_cavity', 'cylindrical_natural']
        medium_risk = ['circular_shallow', 'rectangular_shallow', 'small_cup']
        low_risk = ['elongated_channel', 'flat_surface', 'irregular_depression']
        
        if container_type in high_risk:
            return 'high'
        elif container_type in medium_risk:
            return 'medium'
        elif container_type in low_risk:
            return 'low'
        else:
            return 'unknown'
    
    def _assess_outdoor_likelihood(self, location: str) -> float:
        """Assess likelihood of outdoor environment"""
        outdoor_scores = {
            'household_domestic': 0.5,  # Mixed indoor/outdoor
            'outdoor_waste_garbage': 1.0,
            'construction_sites': 1.0,
            'urban_drainage_infrastructure': 1.0,
            'natural_semi_natural': 1.0,
            'agricultural_areas': 1.0,
            'public_spaces': 0.8
        }
        return outdoor_scores.get(location, 0.5)
    
    def _classify_urban_natural(self, location: str) -> str:
        """Classify location as urban or natural"""
        urban_locations = ['household_domestic', 'urban_drainage_infrastructure', 'construction_sites', 'public_spaces']
        natural_locations = ['natural_semi_natural', 'agricultural_areas']
        
        if location in urban_locations:
            return 'urban'
        elif location in natural_locations:
            return 'natural'
        else:
            return 'mixed'
    
    def _calculate_risk_score(self, risk_factors: List[str]) -> float:
        """Calculate overall risk score from risk factors"""
        # High-risk factors
        high_risk_factors = [
            'perfect_breeding_shape', 'rubber_material', 'heat_retention',
            'organic_matter', 'natural_nutrients', 'multiple_containers'
        ]
        
        # Medium-risk factors
        medium_risk_factors = [
            'outdoor_exposure', 'rainwater_collection', 'large_volume',
            'warm_environment', 'shaded_area', 'irregular_cleaning'
        ]
        
        score = 0.0
        for factor in risk_factors:
            if factor in high_risk_factors:
                score += 0.3
            elif factor in medium_risk_factors:
                score += 0.2
            else:
                score += 0.1
        
        return min(1.0, score)
    
    def _get_classification_guidelines(self) -> Dict[str, Any]:
        """Get comprehensive classification guidelines"""
        return {
            'hotspot_criteria': [
                "Visible stagnant water present",
                "Container-like structure that can hold water",
                "Water can remain for 3-5+ days",
                "Suitable for Aedes mosquito breeding"
            ],
            'potential_criteria': [
                "Container present but no visible water",
                "Outdoor location with rain exposure",
                "Shape suitable for water collection",
                "Requires verification after rain"
            ],
            'not_hotspot_criteria': [
                "No water present and no collection potential",
                "Moving/flowing water",
                "Maintained water bodies (pools, aquariums)",
                "Indoor dry surfaces",
                "Irrelevant objects (vehicles, people, furniture)"
            ],
            'uncertain_criteria': [
                "Poor image quality preventing analysis",
                "Ambiguous water presence",
                "Partially obscured containers",
                "Conflicting visual indicators"
            ],
            'invalid_criteria': [
                "Image too blurry/dark for analysis",
                "Corrupted or distorted image data",
                "Insufficient resolution",
                "Motion blur or focus issues"
            ]
        }
    
    def _get_feature_importance_guidelines(self) -> Dict[str, Any]:
        """Get feature importance for AI training"""
        return {
            'critical_features': [
                'water_presence',
                'water_stagnation',
                'container_shape',
                'outdoor_location'
            ],
            'important_features': [
                'water_volume',
                'water_clarity',
                'container_depth',
                'environmental_context'
            ],
            'supporting_features': [
                'organic_matter_presence',
                'shading_conditions',
                'maintenance_indicators',
                'multiple_containers'
            ],
            'feature_weights': {
                'water_presence': 0.25,
                'stagnation_indicators': 0.20,
                'container_suitability': 0.20,
                'environmental_context': 0.15,
                'risk_factors': 0.10,
                'image_quality': 0.10
            }
        }
    
    def save_training_data(self, output_path: str) -> Dict[str, Any]:
        """Save generated training data to file"""
        try:
            training_data = self.generate_comprehensive_training_set()
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
            
            return {
                'success': True,
                'output_path': output_path,
                'total_scenarios': training_data['metadata']['total_scenarios'],
                'categories': list(training_data['training_examples'].keys()),
                'message': 'Training data generated and saved successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to save training data'
            }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of available training scenarios"""
        summary = {
            'total_categories': len(self.training_categories),
            'total_location_types': len(self.location_types),
            'scenarios_per_category': {}
        }
        
        for category in self.training_categories:
            scenarios_key = f"{category}_scenarios"
            if scenarios_key in self.training_scenarios:
                summary['scenarios_per_category'][category] = len(self.training_scenarios[scenarios_key])
            else:
                summary['scenarios_per_category'][category] = 0
        
        summary['total_scenarios'] = sum(summary['scenarios_per_category'].values())
        
        return summary

# Create global instance
breeding_site_trainer = BreedingSiteTrainingDataGenerator()