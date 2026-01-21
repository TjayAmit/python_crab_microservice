"""
Model Comparison API Module
Compares MobileNetV2, EfficientNetB0, and NASNetMobile training results
Usage: from model_comparison import get_model_comparison
"""

import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelMetrics:
    """Stores metrics for a trained model"""
    name: str
    architecture: str
    final_train_accuracy: float
    final_val_accuracy: float
    final_train_loss: float
    final_val_loss: float
    best_val_accuracy: float
    total_epochs: int
    overfitting_gap: float
    top_2_accuracy: float
    total_parameters: int
    trainable_parameters: int
    image_size: tuple
    batch_size: int
    training_time_estimate: str
    convergence_speed: str  # Fast, Medium, Slow
    model_size_mb: float
    per_class_accuracy: Dict[str, float]
    
    def to_dict(self):
        return asdict(self)


class ModelComparison:
    """Handles comparison of multiple trained models"""
    
    def __init__(self):
        self.models = {
            'mobilenet': {
                'dir': 'model',
                'name': 'MobileNetV2',
                'history': 'training_history.json',
                'class_names': 'class_names.json',
                'model_file': 'my_model.keras',
                'params': 3538000,  # Approximate
                'architecture_desc': 'Inverted residual blocks with depth-wise separable convolutions'
            },
            'efficientnet': {
                'dir': 'model_efficientnet',
                'name': 'EfficientNetB0',
                'history': 'training_history.json',
                'class_names': 'class_names.json',
                'model_file': 'efficientnet_model.keras',
                'params': 5330000,  # Approximate
                'architecture_desc': 'Compound scaling with mobile inverted bottleneck blocks'
            },
            'nasnet': {
                'dir': 'model_nasnet',
                'name': 'NASNetMobile',
                'history': 'training_history.json',
                'class_names': 'class_names.json',
                'model_file': 'nasnet_model.keras',
                'params': 5326000,  # Approximate
                'architecture_desc': 'Neural Architecture Search cells with learned architecture'
            }
        }
        
    def _load_training_history(self, model_key: str) -> Optional[Dict]:
        """Load training history from JSON file"""
        model_info = self.models[model_key]
        history_path = os.path.join(model_info['dir'], model_info['history'])
        
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except json.JSONDecodeError:
            return None
    
    def _load_class_names(self, model_key: str) -> Optional[List[str]]:
        """Load class names from JSON file"""
        model_info = self.models[model_key]
        class_path = os.path.join(model_info['dir'], model_info['class_names'])
        
        try:
            with open(class_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def _get_model_size(self, model_key: str) -> float:
        """Get model file size in MB"""
        model_info = self.models[model_key]
        model_path = os.path.join(model_info['dir'], model_info['model_file'])
        
        try:
            size_bytes = os.path.getsize(model_path)
            return round(size_bytes / (1024 * 1024), 2)
        except FileNotFoundError:
            return 0.0
    
    def _calculate_convergence_speed(self, history: Dict, target_accuracy: float = 0.7) -> tuple:
        """
        Calculate how fast model converged to target accuracy
        Returns (speed_category, epochs_to_target)
        """
        val_acc = history.get('val_accuracy', [])
        
        # Find epoch where val_accuracy first exceeds target
        epochs_to_target = None
        for i, acc in enumerate(val_acc):
            if acc >= target_accuracy:
                epochs_to_target = i + 1
                break
        
        if epochs_to_target is None:
            return "Did not converge", len(val_acc)
        elif epochs_to_target <= 10:
            return "Very Fast", epochs_to_target
        elif epochs_to_target <= 20:
            return "Fast", epochs_to_target
        elif epochs_to_target <= 40:
            return "Medium", epochs_to_target
        else:
            return "Slow", epochs_to_target
    
    def _estimate_training_time(self, total_epochs: int, model_key: str) -> str:
        """Estimate training time based on model and epochs"""
        # Approximate time per epoch (in seconds) on typical hardware
        time_per_epoch = {
            'mobilenet': 30,      # Fastest
            'efficientnet': 35,   # Medium
            'nasnet': 40          # Slowest
        }
        
        total_seconds = total_epochs * time_per_epoch.get(model_key, 35)
        
        if total_seconds < 60:
            return f"{total_seconds}s"
        elif total_seconds < 3600:
            minutes = total_seconds / 60
            return f"{minutes:.1f} min"
        else:
            hours = total_seconds / 3600
            return f"{hours:.1f} hours"
    
    def _calculate_per_class_metrics(self, model_key: str, history: Dict) -> Dict[str, float]:
        """
        Calculate per-class accuracy (placeholder - would need actual predictions)
        In production, this should load actual per-class results from evaluation
        """
        # This is a placeholder - in real implementation, 
        # you'd load per-class results from a separate file
        class_names = self._load_class_names(model_key)
        if not class_names:
            return {}
        
        # Return empty dict as placeholder
        # In production, load from per_class_results.json or similar
        return {name: 0.0 for name in class_names}
    
    def get_model_metrics(self, model_key: str) -> Optional[ModelMetrics]:
        """Extract metrics for a single model"""
        history = self._load_training_history(model_key)
        if not history:
            return None
        
        model_info = self.models[model_key]
        
        # Extract final metrics
        final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0.0
        final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0.0
        final_train_loss = history['loss'][-1] if history['loss'] else 0.0
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else 0.0
        best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0.0
        total_epochs = len(history['accuracy'])
        overfitting_gap = final_train_acc - final_val_acc
        final_top2 = history['val_top_2_accuracy'][-1] if 'val_top_2_accuracy' in history else 0.0
        
        # Calculate convergence
        speed, epochs_to_target = self._calculate_convergence_speed(history)
        convergence_info = f"{speed} ({epochs_to_target} epochs to 70% accuracy)"
        
        # Get model size
        model_size = self._get_model_size(model_key)
        
        # Estimate training time
        training_time = self._estimate_training_time(total_epochs, model_key)
        
        # Per-class metrics
        per_class = self._calculate_per_class_metrics(model_key, history)
        
        return ModelMetrics(
            name=model_info['name'],
            architecture=model_info['architecture_desc'],
            final_train_accuracy=round(final_train_acc, 4),
            final_val_accuracy=round(final_val_acc, 4),
            final_train_loss=round(final_train_loss, 4),
            final_val_loss=round(final_val_loss, 4),
            best_val_accuracy=round(best_val_acc, 4),
            total_epochs=total_epochs,
            overfitting_gap=round(overfitting_gap, 4),
            top_2_accuracy=round(final_top2, 4),
            total_parameters=model_info['params'],
            trainable_parameters=model_info['params'],  # Simplified
            image_size=(224, 224),
            batch_size=32,
            training_time_estimate=training_time,
            convergence_speed=convergence_info,
            model_size_mb=model_size,
            per_class_accuracy=per_class
        )
    
    def compare_all_models(self) -> Dict:
        """Compare all three models and return comprehensive analysis"""
        results = {}
        metrics = {}
        
        # Collect metrics for each model
        for key in self.models.keys():
            model_metrics = self.get_model_metrics(key)
            if model_metrics:
                metrics[key] = model_metrics
                results[key] = model_metrics.to_dict()
        
        if not metrics:
            return {
                'error': 'No trained models found',
                'message': 'Please train at least one model first'
            }
        
        # Perform comparison analysis
        comparison = self._analyze_models(metrics)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'models': results,
            'comparison': comparison,
            'summary': self._generate_summary(metrics, comparison)
        }
    
    def _analyze_models(self, metrics: Dict[str, ModelMetrics]) -> Dict:
        """Analyze and compare model performance"""
        analysis = {
            'accuracy': {},
            'speed': {},
            'efficiency': {},
            'generalization': {},
            'recommendations': {}
        }
        
        # Find best performers
        if metrics:
            # Best accuracy
            best_acc_model = max(metrics.items(), 
                                key=lambda x: x[1].best_val_accuracy)
            analysis['accuracy']['best_model'] = best_acc_model[0]
            analysis['accuracy']['best_score'] = best_acc_model[1].best_val_accuracy
            analysis['accuracy']['ranking'] = sorted(
                [(k, v.best_val_accuracy) for k, v in metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            # Fastest convergence
            convergence_speeds = {
                k: int(v.convergence_speed.split('(')[1].split()[0]) 
                for k, v in metrics.items()
                if '(' in v.convergence_speed
            }
            if convergence_speeds:
                fastest_model = min(convergence_speeds.items(), key=lambda x: x[1])
                analysis['speed']['fastest_convergence'] = fastest_model[0]
                analysis['speed']['epochs_to_70_percent'] = fastest_model[1]
                analysis['speed']['ranking'] = sorted(
                    convergence_speeds.items(),
                    key=lambda x: x[1]
                )
            
            # Best generalization (lowest overfitting)
            best_gen_model = min(metrics.items(),
                                key=lambda x: abs(x[1].overfitting_gap))
            analysis['generalization']['best_model'] = best_gen_model[0]
            analysis['generalization']['overfitting_gap'] = best_gen_model[1].overfitting_gap
            analysis['generalization']['ranking'] = sorted(
                [(k, abs(v.overfitting_gap)) for k, v in metrics.items()],
                key=lambda x: x[1]
            )
            
            # Model efficiency (accuracy per parameter)
            efficiency_scores = {
                k: (v.best_val_accuracy / v.total_parameters) * 1e6
                for k, v in metrics.items()
            }
            best_eff_model = max(efficiency_scores.items(), key=lambda x: x[1])
            analysis['efficiency']['best_model'] = best_eff_model[0]
            analysis['efficiency']['score'] = round(best_eff_model[1], 4)
            analysis['efficiency']['ranking'] = sorted(
                efficiency_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(metrics, analysis)
        
        return analysis
    
    def _generate_recommendations(self, metrics: Dict[str, ModelMetrics], 
                                 analysis: Dict) -> Dict:
        """Generate recommendations based on different use cases"""
        recommendations = {}
        
        # Best overall
        acc_ranks = {k: i for i, (k, _) in enumerate(analysis['accuracy']['ranking'])}
        speed_ranks = {k: i for i, (k, _) in enumerate(analysis['speed'].get('ranking', []))}
        gen_ranks = {k: i for i, (k, _) in enumerate(analysis['generalization']['ranking'])}
        
        # Calculate combined score (lower is better)
        combined_scores = {}
        for key in metrics.keys():
            score = acc_ranks.get(key, 99) + speed_ranks.get(key, 99) + gen_ranks.get(key, 99)
            combined_scores[key] = score
        
        best_overall = min(combined_scores.items(), key=lambda x: x[1])
        recommendations['best_overall'] = {
            'model': best_overall[0],
            'reason': 'Best balance of accuracy, speed, and generalization'
        }
        
        # Production deployment
        recommendations['production_deployment'] = {
            'model': analysis['accuracy']['best_model'],
            'reason': f"Highest validation accuracy ({analysis['accuracy']['best_score']:.2%})"
        }
        
        # Fast training/iteration
        if 'fastest_convergence' in analysis['speed']:
            recommendations['fast_iteration'] = {
                'model': analysis['speed']['fastest_convergence'],
                'reason': f"Fastest convergence ({analysis['speed']['epochs_to_70_percent']} epochs)"
            }
        
        # Mobile deployment
        smallest_model = min(metrics.items(), key=lambda x: x[1].model_size_mb)
        recommendations['mobile_deployment'] = {
            'model': smallest_model[0],
            'reason': f"Smallest model size ({smallest_model[1].model_size_mb} MB)"
        }
        
        # Resource constrained
        recommendations['resource_constrained'] = {
            'model': analysis['efficiency']['best_model'],
            'reason': 'Best accuracy per parameter ratio'
        }
        
        return recommendations
    
    def _generate_summary(self, metrics: Dict[str, ModelMetrics], 
                         comparison: Dict) -> Dict:
        """Generate executive summary of comparison"""
        summary = {
            'total_models_compared': len(metrics),
            'best_accuracy_model': comparison['accuracy']['best_model'],
            'best_accuracy_score': f"{comparison['accuracy']['best_score']:.2%}",
        }
        
        if 'fastest_convergence' in comparison['speed']:
            summary['fastest_training_model'] = comparison['speed']['fastest_convergence']
            summary['fastest_training_epochs'] = comparison['speed']['epochs_to_70_percent']
        
        summary['best_generalization_model'] = comparison['generalization']['best_model']
        summary['recommended_for_production'] = comparison['recommendations']['best_overall']['model']
        
        # Calculate average metrics
        avg_val_acc = sum(m.best_val_accuracy for m in metrics.values()) / len(metrics)
        summary['average_validation_accuracy'] = f"{avg_val_acc:.2%}"
        
        return summary


# ============= API ENDPOINT FUNCTIONS =============

def get_model_comparison() -> Dict:
    """
    Main function to get comprehensive model comparison
    Use this in your API endpoint
    
    Returns:
        Dict containing full comparison results
    """
    comparator = ModelComparison()
    return comparator.compare_all_models()


def get_single_model_info(model_name: str) -> Dict:
    """
    Get information for a single model
    
    Args:
        model_name: One of 'mobilenet', 'efficientnet', 'nasnet'
    
    Returns:
        Dict containing model metrics or error
    """
    comparator = ModelComparison()
    
    if model_name not in comparator.models:
        return {
            'error': f'Invalid model name: {model_name}',
            'valid_models': list(comparator.models.keys())
        }
    
    metrics = comparator.get_model_metrics(model_name)
    
    if not metrics:
        return {
            'error': f'No training data found for {model_name}',
            'message': 'Please train this model first'
        }
    
    return {
        'model': model_name,
        'metrics': metrics.to_dict(),
        'timestamp': datetime.now().isoformat()
    }


def get_quick_summary() -> Dict:
    """
    Get a quick summary of all models
    Lightweight endpoint for dashboard displays
    
    Returns:
        Dict containing quick comparison metrics
    """
    comparator = ModelComparison()
    full_comparison = comparator.compare_all_models()
    
    if 'error' in full_comparison:
        return full_comparison
    
    # Extract only essential information
    quick_summary = {
        'timestamp': full_comparison['timestamp'],
        'models_available': list(full_comparison['models'].keys()),
        'best_accuracy': {
            'model': full_comparison['comparison']['accuracy']['best_model'],
            'score': full_comparison['comparison']['accuracy']['best_score']
        },
        'recommendations': full_comparison['comparison']['recommendations']['best_overall'],
        'summary': full_comparison['summary']
    }
    
    return quick_summary


# ============= FLASK API EXAMPLE =============

"""
Example Flask API implementation:

from flask import Flask, jsonify
from model_comparison import get_model_comparison, get_single_model_info, get_quick_summary

app = Flask(__name__)

@app.route('/difference-of-training-models', methods=['GET'])
def comparison_endpoint():
    result = get_model_comparison()
    return jsonify(result)

@app.route('/model/<model_name>', methods=['GET'])
def single_model_endpoint(model_name):
    result = get_single_model_info(model_name)
    return jsonify(result)

@app.route('/models/summary', methods=['GET'])
def summary_endpoint():
    result = get_quick_summary()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
"""


# ============= FASTAPI EXAMPLE =============

"""
Example FastAPI implementation:

from fastapi import FastAPI
from model_comparison import get_model_comparison, get_single_model_info, get_quick_summary

app = FastAPI(title="Model Comparison API")

@app.get("/difference-of-training-models")
async def comparison_endpoint():
    return get_model_comparison()

@app.get("/model/{model_name}")
async def single_model_endpoint(model_name: str):
    return get_single_model_info(model_name)

@app.get("/models/summary")
async def summary_endpoint():
    return get_quick_summary()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""


# ============= COMMAND LINE USAGE =============

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("MODEL COMPARISON TOOL")
    print("="*60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'summary':
            result = get_quick_summary()
        elif sys.argv[1] in ['mobilenet', 'efficientnet', 'nasnet']:
            result = get_single_model_info(sys.argv[1])
        else:
            result = get_model_comparison()
    else:
        result = get_model_comparison()
    
    print(json.dumps(result, indent=2))