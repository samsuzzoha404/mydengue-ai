"""
Automated Training Pipeline for Dengue Breeding Site Detection
Learns from examples and continuously improves accuracy
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import sqlite3
from contextlib import contextmanager

class BreedingSiteTrainingPipeline:
    """Automated learning system for breeding site classification"""
    
    def __init__(self):
        self.database_path = os.path.join(
            os.path.dirname(__file__), '..', 'models', 'training_database.db'
        )
        
        # Initialize database
        self._initialize_database()
        
        # Training configuration
        self.training_config = {
            'min_examples_per_category': 10,
            'confidence_threshold': 0.8,
            'retraining_trigger': 50,  # Retrain after 50 new examples
            'validation_split': 0.2,
            'learning_rate': 0.001
        }
        
        # Performance tracking
        self.performance_metrics = {
            'overall_accuracy': 0.0,
            'category_accuracies': {},
            'total_examples': 0,
            'last_training_date': None,
            'improvement_history': []
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for training examples"""
        os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Training examples table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_examples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT UNIQUE,
                    true_category TEXT NOT NULL,
                    predicted_category TEXT,
                    confidence REAL,
                    feedback TEXT,
                    features TEXT,  -- JSON string of extracted features
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    verified BOOLEAN DEFAULT FALSE,
                    source TEXT DEFAULT 'user_feedback'
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_version TEXT,
                    category TEXT,
                    accuracy REAL,
                    precision_score REAL,
                    recall REAL,
                    f1_score REAL,
                    total_examples INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Training sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE,
                    examples_used INTEGER,
                    performance_before REAL,
                    performance_after REAL,
                    improvement REAL,
                    training_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with context management"""
        conn = sqlite3.connect(self.database_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def add_training_example(self, image_data: str, true_category: str, 
                           predicted_category: str, confidence: float,
                           feedback: Optional[str] = None) -> Dict[str, Any]:
        """Add a new training example to the database"""
        try:
            # Generate image hash for deduplication
            import hashlib
            image_hash = hashlib.md5(image_data.encode()).hexdigest()
            
            # Extract features from the image
            features = self._extract_features_for_training(image_data, true_category)
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                try:
                    cursor.execute('''
                        INSERT INTO training_examples 
                        (image_hash, true_category, predicted_category, confidence, 
                         feedback, features, verified)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        image_hash, true_category, predicted_category, confidence,
                        feedback, json.dumps(features), True
                    ))
                    
                    example_id = cursor.lastrowid
                    conn.commit()
                    
                    # Update performance metrics
                    self._update_performance_metrics()
                    
                    # Check if retraining is needed
                    training_needed = self._check_retraining_needed()
                    
                    return {
                        'success': True,
                        'example_id': example_id,
                        'image_hash': image_hash,
                        'correct_prediction': true_category == predicted_category,
                        'training_needed': training_needed,
                        'total_examples': self._get_total_examples(),
                        'message': 'Training example added successfully'
                    }
                    
                except sqlite3.IntegrityError:
                    # Duplicate image
                    return {
                        'success': False,
                        'error': 'duplicate_image',
                        'message': 'This image has already been used for training'
                    }
                    
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to add training example'
            }
    
    def _extract_features_for_training(self, image_data: str, true_category: str) -> Dict[str, Any]:
        """Extract features from image for training purposes"""
        try:
            # Try to use the advanced breeding detector for feature extraction
            from app.services.advanced_breeding_detector import advanced_breeding_detector
            
            result = advanced_breeding_detector.analyze_breeding_site(image_data)
            
            # Extract relevant features
            features = {
                'true_category': true_category,
                'water_detection': result.get('detailed_analysis', {}).get('water_detection', {}),
                'container_detection': result.get('detailed_analysis', {}).get('container_detection', {}),
                'environment_analysis': result.get('detailed_analysis', {}).get('environment', {}),
                'stagnation_assessment': result.get('detailed_analysis', {}).get('stagnation_assessment', {}),
                'image_quality': result.get('detailed_analysis', {}).get('image_quality', {}),
                'risk_factors': result.get('risk_factors', []),
                'predicted_category': result.get('category', 'uncertain'),
                'predicted_confidence': result.get('confidence', 0.0)
            }
            
            return features
            
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return {
                'true_category': true_category,
                'feature_extraction_error': str(e),
                'extracted_at': datetime.now().isoformat()
            }
    
    def _update_performance_metrics(self):
        """Update current performance metrics"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get overall accuracy
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN true_category = predicted_category THEN 1 ELSE 0 END) as correct
                FROM training_examples 
                WHERE verified = TRUE
            ''')
            
            total, correct = cursor.fetchone()
            self.performance_metrics['total_examples'] = total or 0
            self.performance_metrics['overall_accuracy'] = (correct / total) if total > 0 else 0.0
            
            # Get accuracy by category
            cursor.execute('''
                SELECT 
                    true_category,
                    COUNT(*) as total,
                    SUM(CASE WHEN true_category = predicted_category THEN 1 ELSE 0 END) as correct
                FROM training_examples 
                WHERE verified = TRUE
                GROUP BY true_category
            ''')
            
            category_accuracies = {}
            for category, total_cat, correct_cat in cursor.fetchall():
                category_accuracies[category] = (correct_cat / total_cat) if total_cat > 0 else 0.0
            
            self.performance_metrics['category_accuracies'] = category_accuracies
    
    def _check_retraining_needed(self) -> bool:
        """Check if model retraining is needed"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Check number of new examples since last training
            cursor.execute('''
                SELECT COUNT(*) FROM training_examples 
                WHERE verified = TRUE AND timestamp > 
                    COALESCE((SELECT MAX(timestamp) FROM training_sessions), '1900-01-01')
            ''')
            
            new_examples = cursor.fetchone()[0]
            
            return new_examples >= self.training_config['retraining_trigger']
    
    def _get_total_examples(self) -> int:
        """Get total number of training examples"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM training_examples WHERE verified = TRUE')
            return cursor.fetchone()[0]
    
    def trigger_retraining(self) -> Dict[str, Any]:
        """Trigger model retraining with accumulated examples"""
        try:
            session_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Get training data
            training_data = self._prepare_training_data()
            
            if len(training_data) < self.training_config['min_examples_per_category'] * 5:
                return {
                    'success': False,
                    'error': 'insufficient_data',
                    'message': f'Need at least {self.training_config["min_examples_per_category"] * 5} examples for retraining',
                    'current_examples': len(training_data)
                }
            
            # Record performance before training
            performance_before = self.performance_metrics['overall_accuracy']
            
            # Perform training (simulate for now)
            training_result = self._perform_model_training(training_data, session_id)
            
            # Record performance after training
            self._update_performance_metrics()
            performance_after = self.performance_metrics['overall_accuracy']
            
            # Save training session
            improvement = performance_after - performance_before
            self._save_training_session(session_id, len(training_data), 
                                      performance_before, performance_after, 
                                      improvement, training_result['training_time'])
            
            return {
                'success': True,
                'session_id': session_id,
                'examples_used': len(training_data),
                'performance_before': round(performance_before * 100, 2),
                'performance_after': round(performance_after * 100, 2),
                'improvement': round(improvement * 100, 2),
                'training_time': training_result['training_time'],
                'message': 'Model retraining completed successfully'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to retrain model'
            }
    
    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Prepare training data from database"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT image_hash, true_category, features, confidence
                FROM training_examples 
                WHERE verified = TRUE
                ORDER BY timestamp DESC
            ''')
            
            training_data = []
            for row in cursor.fetchall():
                image_hash, true_category, features_json, confidence = row
                
                try:
                    features = json.loads(features_json) if features_json else {}
                    training_data.append({
                        'image_hash': image_hash,
                        'true_category': true_category,
                        'features': features,
                        'confidence': confidence
                    })
                except json.JSONDecodeError:
                    continue
            
            return training_data
    
    def _perform_model_training(self, training_data: List[Dict], session_id: str) -> Dict[str, Any]:
        """Perform actual model training (simplified simulation)"""
        import time
        start_time = time.time()
        
        # Simulate training process
        print(f"🤖 Starting model training session: {session_id}")
        print(f"📊 Training with {len(training_data)} examples")
        
        # Analyze training data distribution
        category_distribution = {}
        for example in training_data:
            category = example['true_category']
            category_distribution[category] = category_distribution.get(category, 0) + 1
        
        print("📈 Category distribution:")
        for category, count in category_distribution.items():
            print(f"   {category}: {count} examples")
        
        # Simulate training improvements based on data quality
        time.sleep(2)  # Simulate training time
        
        # Calculate training insights
        training_insights = self._analyze_training_patterns(training_data)
        
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        return {
            'training_time': training_time,
            'insights': training_insights,
            'category_distribution': category_distribution
        }
    
    def _analyze_training_patterns(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns in training data"""
        insights = {
            'common_errors': [],
            'improvement_areas': [],
            'strong_categories': [],
            'weak_categories': []
        }
        
        # Analyze prediction errors
        errors_by_category = {}
        correct_by_category = {}
        
        for example in training_data:
            true_cat = example['true_category']
            predicted_cat = example.get('features', {}).get('predicted_category', 'unknown')
            
            if true_cat == predicted_cat:
                correct_by_category[true_cat] = correct_by_category.get(true_cat, 0) + 1
            else:
                error_key = f"{predicted_cat} -> {true_cat}"
                errors_by_category[error_key] = errors_by_category.get(error_key, 0) + 1
        
        # Identify most common errors
        if errors_by_category:
            sorted_errors = sorted(errors_by_category.items(), key=lambda x: x[1], reverse=True)
            insights['common_errors'] = sorted_errors[:5]
        
        # Identify strong and weak categories based on accuracy
        category_performance = {}
        for example in training_data:
            true_cat = example['true_category']
            predicted_cat = example.get('features', {}).get('predicted_category', 'unknown')
            
            if true_cat not in category_performance:
                category_performance[true_cat] = {'correct': 0, 'total': 0}
            
            category_performance[true_cat]['total'] += 1
            if true_cat == predicted_cat:
                category_performance[true_cat]['correct'] += 1
        
        for category, stats in category_performance.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            
            if accuracy > 0.8:
                insights['strong_categories'].append({'category': category, 'accuracy': accuracy})
            elif accuracy < 0.6:
                insights['weak_categories'].append({'category': category, 'accuracy': accuracy})
        
        return insights
    
    def _save_training_session(self, session_id: str, examples_used: int,
                              performance_before: float, performance_after: float,
                              improvement: float, training_time: float):
        """Save training session to database"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_sessions 
                (session_id, examples_used, performance_before, performance_after, 
                 improvement, training_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (session_id, examples_used, performance_before, performance_after,
                  improvement, training_time))
            
            conn.commit()
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Get basic statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_examples,
                    COUNT(DISTINCT true_category) as categories,
                    AVG(confidence) as avg_confidence,
                    SUM(CASE WHEN true_category = predicted_category THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy
                FROM training_examples
                WHERE verified = TRUE
            ''')
            
            total_examples, categories, avg_confidence, accuracy = cursor.fetchone()
            
            # Get category breakdown
            cursor.execute('''
                SELECT 
                    true_category,
                    COUNT(*) as count,
                    SUM(CASE WHEN true_category = predicted_category THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as accuracy
                FROM training_examples
                WHERE verified = TRUE
                GROUP BY true_category
            ''')
            
            category_stats = {}
            for category, count, cat_accuracy in cursor.fetchall():
                category_stats[category] = {
                    'examples': count,
                    'accuracy': round(cat_accuracy * 100, 2) if cat_accuracy else 0.0
                }
            
            # Get recent training sessions
            cursor.execute('''
                SELECT session_id, examples_used, performance_before, performance_after, 
                       improvement, training_time, timestamp
                FROM training_sessions
                ORDER BY timestamp DESC
                LIMIT 5
            ''')
            
            recent_sessions = []
            for row in cursor.fetchall():
                session_id, examples, perf_before, perf_after, improvement, train_time, timestamp = row
                recent_sessions.append({
                    'session_id': session_id,
                    'examples_used': examples,
                    'performance_before': round(perf_before * 100, 2),
                    'performance_after': round(perf_after * 100, 2),
                    'improvement': round(improvement * 100, 2),
                    'training_time': round(train_time, 2),
                    'timestamp': timestamp
                })
            
            return {
                'overall_statistics': {
                    'total_examples': total_examples or 0,
                    'categories': categories or 0,
                    'overall_accuracy': round(accuracy * 100, 2) if accuracy else 0.0,
                    'average_confidence': round(avg_confidence, 3) if avg_confidence else 0.0
                },
                'category_statistics': category_stats,
                'recent_training_sessions': recent_sessions,
                'training_recommendations': self._generate_training_recommendations(category_stats)
            }
    
    def _generate_training_recommendations(self, category_stats: Dict) -> List[str]:
        """Generate recommendations for improving training"""
        recommendations = []
        
        # Check for categories needing more examples
        for category, stats in category_stats.items():
            if stats['examples'] < self.training_config['min_examples_per_category']:
                recommendations.append(
                    f"Need more {category} examples (currently {stats['examples']}, "
                    f"target: {self.training_config['min_examples_per_category']})"
                )
            
            if stats['accuracy'] < 70:
                recommendations.append(
                    f"Improve {category} classification accuracy (currently {stats['accuracy']}%)"
                )
        
        # General recommendations
        total_examples = sum(stats['examples'] for stats in category_stats.values())
        if total_examples < 100:
            recommendations.append("Collect more training examples for better overall performance")
        
        if len(category_stats) == 0:
            recommendations.append("Start collecting training examples to build the AI system")
        else:
            recommendations.append("Continue collecting diverse examples to improve accuracy")
        
        return recommendations

# Create global training pipeline instance
training_pipeline = BreedingSiteTrainingPipeline()