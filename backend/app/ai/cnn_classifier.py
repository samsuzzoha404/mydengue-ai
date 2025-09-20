"""
CNN Breeding Site Classifier for Enhanced Image Analysis
Integrates with existing advanced breeding detector
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import base64
import io
from PIL import Image
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple

class CNNBreedingSiteClassifier:
    """Advanced CNN model for breeding site image classification"""
    
    def __init__(self):
        self.model = None
        self.input_shape = (224, 224, 3)
        self.categories = [
            'hotspot',           # Clear breeding site
            'potential',         # Possible breeding site
            'not_hotspot',       # Not a breeding site
            'uncertain',         # Unclear/ambiguous
            'contaminated',      # Contaminated water
            'invalid'            # Invalid/unclear image
        ]
        self.model_path = 'models/cnn_breeding_classifier.h5'
        
    def build_model(self):
        """Build advanced CNN model for breeding site classification"""
        self.model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Second Convolutional Block  
            Conv2D(64, (3,3), activation='relu'),
            BatchNormalization(),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3,3), activation='relu'),
            BatchNormalization(),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3,3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2,2),
            Dropout(0.25),
            
            # Dense Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(len(self.categories), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ CNN Model built successfully")
        print(f"   - Input shape: {self.input_shape}")
        print(f"   - Categories: {len(self.categories)}")
        print(f"   - Total parameters: {self.model.count_params():,}")
        
    def load_model(self):
        """Load trained CNN model"""
        try:
            self.model = load_model(self.model_path)
            print("✅ CNN Model loaded successfully")
            return True
        except Exception as e:
            print(f"⚠️ Could not load CNN model: {e}")
            return False
    
    def prepare_image(self, image_base64: str) -> np.ndarray:
        """Prepare image for CNN classification"""
        try:
            # Handle base64 format
            if 'data:image' in image_base64:
                image_data = base64.b64decode(image_base64.split(',')[1])
            else:
                image_data = base64.b64decode(image_base64)
                
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            print(f"⚠️ Image preparation error: {e}")
            return None
    
    def classify_breeding_site(self, image_base64: str) -> Dict[str, Any]:
        """Classify breeding site using CNN"""
        if self.model is None:
            if not self.load_model():
                # Fallback to existing advanced detector
                return self._fallback_classification(image_base64)
        
        # Prepare image
        image_array = self.prepare_image(image_base64)
        if image_array is None:
            return {
                'classification': 'invalid',
                'confidence': 0.0,
                'reasoning': 'Image preprocessing failed',
                'model_type': 'CNN_error'
            }
        
        try:
            # Get CNN prediction
            prediction = self.model.predict(image_array, verbose=0)
            
            # Extract results
            predicted_class_idx = np.argmax(prediction[0])
            predicted_class = self.categories[predicted_class_idx]
            confidence = float(np.max(prediction[0]))
            
            # Get all probabilities
            all_probabilities = {
                self.categories[i]: float(prediction[0][i])
                for i in range(len(self.categories))
            }
            
            # Generate reasoning based on probabilities
            reasoning = self._generate_cnn_reasoning(all_probabilities, predicted_class)
            
            # Enhanced analysis: combine with computer vision features
            enhanced_analysis = self._enhance_with_computer_vision(
                image_base64, predicted_class, confidence
            )
            
            return {
                'classification': predicted_class,
                'confidence': confidence,
                'reasoning': reasoning,
                'model_type': 'CNN_enhanced',
                'all_probabilities': all_probabilities,
                'computer_vision_features': enhanced_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ CNN classification error: {e}")
            return self._fallback_classification(image_base64)
    
    def _generate_cnn_reasoning(self, probabilities: Dict[str, float], predicted_class: str) -> str:
        """Generate human-readable reasoning for CNN prediction"""
        top_2 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:2]
        
        reasoning_templates = {
            'hotspot': f"CNN detected clear breeding site indicators with {probabilities['hotspot']:.1%} confidence",
            'potential': f"CNN identified possible breeding site features with {probabilities['potential']:.1%} confidence", 
            'not_hotspot': f"CNN found no breeding site characteristics with {probabilities['not_hotspot']:.1%} confidence",
            'uncertain': f"CNN analysis inconclusive, image features ambiguous ({probabilities['uncertain']:.1%})",
            'contaminated': f"CNN detected contaminated water patterns with {probabilities['contaminated']:.1%} confidence",
            'invalid': f"CNN unable to analyze image properly ({probabilities['invalid']:.1%})"
        }
        
        base_reasoning = reasoning_templates.get(predicted_class, f"CNN classified as {predicted_class}")
        
        # Add secondary consideration if close
        if len(top_2) > 1 and top_2[1][1] > 0.2:
            base_reasoning += f". Secondary consideration: {top_2[1][0]} ({top_2[1][1]:.1%})"
        
        return base_reasoning
    
    def _enhance_with_computer_vision(self, image_base64: str, cnn_class: str, cnn_confidence: float) -> Dict[str, Any]:
        """Enhance CNN results with computer vision analysis"""
        try:
            # Import our existing advanced detector
            from ..services.advanced_breeding_detector import AdvancedBreedingSiteDetector
            
            detector = AdvancedBreedingSiteDetector()
            cv_result = detector.analyze_image(image_base64)
            
            # Combine CNN and computer vision insights
            combined_confidence = (cnn_confidence * 0.7) + (cv_result.get('confidence', 0.5) * 0.3)
            
            # Check for agreement between methods
            agreement = (
                cnn_class == cv_result.get('classification') or 
                (cnn_class == 'hotspot' and cv_result.get('classification') in ['hotspot', 'potential'])
            )
            
            return {
                'computer_vision_classification': cv_result.get('classification'),
                'computer_vision_confidence': cv_result.get('confidence'),
                'water_features': cv_result.get('water_analysis', {}),
                'container_features': cv_result.get('container_analysis', {}),
                'combined_confidence': combined_confidence,
                'methods_agree': agreement,
                'enhancement_applied': True
            }
            
        except Exception as e:
            print(f"⚠️ Computer vision enhancement error: {e}")
            return {
                'enhancement_applied': False,
                'error': str(e)
            }
    
    def _fallback_classification(self, image_base64: str) -> Dict[str, Any]:
        """Fallback to existing advanced breeding detector"""
        try:
            from ..services.advanced_breeding_detector import AdvancedBreedingSiteDetector
            
            detector = AdvancedBreedingSiteDetector()
            result = detector.analyze_image(image_base64)
            
            # Format to match CNN output
            result['model_type'] = 'Fallback_AdvancedDetector'
            result['reasoning'] = f"Fallback analysis: {result.get('reasoning', 'Advanced computer vision analysis')}"
            
            return result
            
        except Exception as e:
            print(f"⚠️ Fallback classification error: {e}")
            return {
                'classification': 'invalid',
                'confidence': 0.0,
                'reasoning': f'Both CNN and fallback analysis failed: {str(e)}',
                'model_type': 'Error'
            }
    
    def generate_synthetic_training_data(self, num_samples=1000) -> Dict[str, Any]:
        """Generate synthetic training data descriptions for model training"""
        print("📊 Generating synthetic CNN training data descriptions...")
        
        synthetic_data = {
            'hotspot': [
                'Stagnant water in tire with visible mosquito larvae',
                'Water-filled container showing algae growth', 
                'Abandoned bucket with dark stagnant water',
                'Vehicle hull collecting rainwater',
                'Palm tree spathe with trapped water',
                'Industrial drum with standing water',
                'Broken drainage pipe with water accumulation',
                'Unused livestock trough with stagnant water'
            ],
            'potential': [
                'Container with small amount of water',
                'Partially covered water storage',
                'Recent rainwater collection in outdoor container',
                'Drainage area with slow water movement',
                'Plant pot saucer with water',
                'Construction site with water pooling',
                'Vendor setup with potential water collection',
                'Urban infrastructure with water retention'
            ],
            'not_hotspot': [
                'Fast-flowing river or stream',
                'Chlorinated swimming pool',
                'Well-maintained fountain with circulation',
                'Fish pond with active fish population',
                'Ocean or large water body',
                'Solid surfaces without water retention',
                'Covered water storage systems',
                'Active irrigation with water movement'
            ],
            'uncertain': [
                'Blurry image with unclear water presence',
                'Shadows obscuring water visibility',
                'Dense vegetation hiding water containers',
                'Poor lighting making analysis difficult',
                'Partial container view with unclear water status',
                'Debris-covered potential water sources',
                'Structural collapse hiding water containers',
                'Weather obstruction (rain, fog) affecting visibility'
            ],
            'contaminated': [
                'Oil-contaminated water surface',
                'Chemical runoff in water container',
                'Sewage-contaminated stagnant water',
                'Industrial waste in water collection',
                'Rusty water from metal containers',
                'Latex processing waste water',
                'Paint or chemical spill in water',
                'Heavily polluted urban runoff'
            ],
            'invalid': [
                'No water or containers visible',
                'Indoor scenes without breeding potential',
                'Pure landscape without water features',
                'Extremely blurry or damaged images',
                'Non-relevant objects (food, electronics, etc.)',
                'Abstract or artistic images',
                'Screenshots or text documents',
                'Images with no environmental context'
            ]
        }
        
        # Generate training manifest
        training_manifest = {
            'total_categories': len(self.categories),
            'samples_per_category': {
                category: len(samples) for category, samples in synthetic_data.items()
            },
            'total_base_samples': sum(len(samples) for samples in synthetic_data.values()),
            'augmentation_factor': num_samples // sum(len(samples) for samples in synthetic_data.values()),
            'target_total_samples': num_samples,
            'data_augmentation_needed': True,
            'training_recommendations': {
                'epochs': 50,
                'batch_size': 32,
                'validation_split': 0.2,
                'augmentation_strategies': [
                    'rotation (±15 degrees)',
                    'width/height shift (0.1)',
                    'zoom (0.1)',
                    'horizontal flip',
                    'brightness adjustment (0.2)'
                ]
            }
        }
        
        print(f"✅ Generated training data manifest:")
        print(f"   - Categories: {len(self.categories)}")
        print(f"   - Base samples: {training_manifest['total_base_samples']}")
        print(f"   - Target samples: {num_samples}")
        
        return {
            'synthetic_descriptions': synthetic_data,
            'training_manifest': training_manifest,
            'categories': self.categories
        }
    
    def create_training_data_generator(self, data_directory: str):
        """Create data generator for CNN training"""
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        validation_generator = datagen.flow_from_directory(
            data_directory,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        return train_generator, validation_generator
    
    def train_model(self, data_directory: str, epochs=50):
        """Train the CNN model"""
        if self.model is None:
            self.build_model()
        
        print("🏋️ Training CNN Breeding Site Classifier...")
        
        # Create data generators
        train_gen, val_gen = self.create_training_data_generator(data_directory)
        
        # Train model
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            verbose=1
        )
        
        # Save model
        self.model.save(self.model_path)
        print(f"✅ CNN Model trained and saved to {self.model_path}")
        
        return history

# Test the CNN classifier
if __name__ == "__main__":
    classifier = CNNBreedingSiteClassifier()
    
    # Build model
    classifier.build_model()
    
    # Generate training data descriptions
    training_data = classifier.generate_synthetic_training_data()
    
    print("\n🧠 CNN Breeding Site Classifier Ready")
    print("   - Model architecture: Advanced CNN with BatchNorm and Dropout")
    print("   - Input size: 224x224x3 RGB images")
    print(f"   - Output classes: {len(classifier.categories)}")
    print("   - Training data: Synthetic descriptions generated")
    print("   - Integration: Enhanced with computer vision analysis")
    
    # Mock test (would need actual image data)
    print("\n📝 Note: To fully test, provide training images in directory structure:")
    for category in classifier.categories:
        print(f"   dataset/{category}/[images...]")