"""
GAIT ANALYSIS AI TOOL - SINGLE FILE IMPLEMENTATION
All-in-one plug-and-play application with modular architecture patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import base64
import hashlib
import datetime
import time
import warnings
import sys
import io
import re
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# CONFIGURATION SYSTEM (JSON/YAML-like structure in Python dict)
# ============================================================================

class Config:
    """Centralized configuration system"""
    
    # Clinical thresholds and parameters
    CLINICAL_PARAMS = {
        "gait_velocity_thresholds": {
            "normal": 1.2,  # m/s
            "impaired": 0.8,
            "severe": 0.6
        },
        "stride_length_thresholds": {
            "normal": 1.4,  # m
            "impaired": 1.0,
            "severe": 0.8
        },
        "cadence_thresholds": {
            "normal": 100,  # steps/min
            "impaired": 80,
            "severe": 60
        },
        "stance_phase_percentage": {
            "normal": (60, 65),
            "impaired": (65, 75),
            "severe": (75, 85)
        },
        "confidence_threshold": 0.75,
        "risk_score_weights": {
            "velocity": 0.35,
            "stride_length": 0.25,
            "cadence": 0.20,
            "symmetry": 0.20
        }
    }
    
    # Model configuration
    MODEL_CONFIG = {
        "default_model": "gait_analysis_v1",
        "available_models": [
            "gait_analysis_v1",
            "gait_analysis_v2",
            "fall_risk_v1",
            "parkinson_detection_v1"
        ],
        "model_paths": {},  # Will be populated at runtime
        "ensemble_weights": {
            "gait_analysis_v1": 0.4,
            "fall_risk_v1": 0.3,
            "parkinson_detection_v1": 0.3
        }
    }
    
    # System settings
    SYSTEM = {
        "log_level": "INFO",
        "cache_size": 1000,
        "max_file_size_mb": 100,
        "data_retention_days": 30,
        "enable_audit_log": True,
        "encryption_enabled": True
    }
    
    # UI settings
    UI = {
        "theme": "light",
        "chart_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        "refresh_interval": 60,
        "enable_realtime": True
    }

# ============================================================================
# DATA MODELS AND ENUMS
# ============================================================================

class GaitPhase(Enum):
    """Enum for gait phases"""
    INITIAL_CONTACT = "initial_contact"
    LOADING_RESPONSE = "loading_response"
    MID_STANCE = "mid_stance"
    TERMINAL_STANCE = "terminal_stance"
    PRE_SWING = "pre_swing"
    INITIAL_SWING = "initial_swing"
    MID_SWING = "mid_swing"
    TERMINAL_SWING = "terminal_swing"

class RiskLevel(Enum):
    """Enum for risk levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"

@dataclass
class GaitMetrics:
    """Data class for storing gait metrics"""
    velocity: float
    stride_length: float
    cadence: int
    stance_percentage: float
    swing_percentage: float
    double_support_time: float
    step_width: float
    symmetry_index: float
    variability: float
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

@dataclass
class PatientInfo:
    """Data class for patient information"""
    patient_id: str
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    medical_conditions: List[str]
    medications: List[str]
    previous_falls: int
    assessment_date: str
    
    def get_bmi(self) -> float:
        return self.weight_kg / ((self.height_cm / 100) ** 2)

# ============================================================================
# KNOWLEDGE BASE (Clinical Data in structured format)
# ============================================================================

class ClinicalKnowledgeBase:
    """Structured clinical knowledge repository"""
    
    # Gait patterns for different conditions
    GAIT_PATTERNS = {
        "normal": {
            "velocity_range": (1.2, 1.4),
            "stride_length_range": (1.3, 1.5),
            "cadence_range": (95, 115),
            "symmetry": "high",
            "variability": "low",
            "characteristics": ["Regular rhythm", "Arm swing present", "Upright posture"]
        },
        "parkinsonian": {
            "velocity_range": (0.6, 0.9),
            "stride_length_range": (0.8, 1.0),
            "cadence_range": (110, 140),
            "symmetry": "moderate",
            "variability": "high",
            "characteristics": ["Shuffling steps", "Reduced arm swing", "Festination", "Freezing"]
        },
        "hemiplegic": {
            "velocity_range": (0.4, 0.7),
            "stride_length_range": (0.7, 0.9),
            "cadence_range": (60, 90),
            "symmetry": "low",
            "variability": "moderate",
            "characteristics": ["Circumduction", "Stiff knee", "Foot drop", "Asymmetric timing"]
        },
        "ataxic": {
            "velocity_range": (0.5, 0.8),
            "stride_length_range": (0.6, 0.9),
            "cadence_range": (70, 100),
            "symmetry": "low",
            "variability": "very_high",
            "characteristics": ["Wide base", "Irregular steps", "Truncal sway", "Lack of coordination"]
        },
        "antalgic": {
            "velocity_range": (0.7, 1.0),
            "stride_length_range": (0.9, 1.1),
            "cadence_range": (80, 100),
            "symmetry": "very_low",
            "variability": "moderate",
            "characteristics": ["Short stance phase on affected side", "Limping", "Pain avoidance"]
        }
    }
    
    # Fall risk factors with weights
    RISK_FACTORS = {
        "previous_falls": 0.25,
        "gait_velocity": 0.20,
        "stride_variability": 0.15,
        "medication_count": 0.10,
        "balance_issues": 0.15,
        "muscle_strength": 0.10,
        "cognitive_impairment": 0.05
    }
    
    # Clinical recommendations based on findings
    RECOMMENDATIONS = {
        "low_risk": [
            "Continue regular activities",
            "Annual gait assessment",
            "Balance exercises 2-3 times per week"
        ],
        "moderate_risk": [
            "Physical therapy evaluation",
            "Home safety assessment",
            "Assistive device consideration",
            "Balance training program"
        ],
        "high_risk": [
            "Immediate physical therapy",
            "Medical evaluation for underlying causes",
            "Assistive device prescription",
            "Fall prevention program",
            "Home modifications"
        ],
        "severe_risk": [
            "Urgent medical attention",
            "Comprehensive geriatric assessment",
            "24-hour supervision consideration",
            "Environmental adaptations",
            "Medication review"
        ]
    }
    
    @classmethod
    def get_pattern_by_metrics(cls, metrics: GaitMetrics) -> List[Dict]:
        """Match gait metrics to known patterns"""
        matches = []
        
        for pattern_name, pattern in cls.GAIT_PATTERNS.items():
            score = 0
            total = 0
            
            # Check velocity
            v_min, v_max = pattern["velocity_range"]
            if v_min <= metrics.velocity <= v_max:
                score += 1
            total += 1
            
            # Check stride length
            sl_min, sl_max = pattern["stride_length_range"]
            if sl_min <= metrics.stride_length <= sl_max:
                score += 1
            total += 1
            
            # Check cadence
            c_min, c_max = pattern["cadence_range"]
            if c_min <= metrics.cadence <= c_max:
                score += 1
            total += 1
            
            confidence = score / total
            if confidence > 0.5:  # At least 2 out of 3 metrics match
                matches.append({
                    "pattern": pattern_name,
                    "confidence": confidence,
                    "characteristics": pattern["characteristics"]
                })
        
        return sorted(matches, key=lambda x: x["confidence"], reverse=True)

# ============================================================================
# MODEL REGISTRY & MANAGEMENT
# ============================================================================

class ModelRegistry:
    """Plug-and-play model registry with hot-swappable models"""
    
    def __init__(self):
        self.models = {}
        self.model_versions = defaultdict(list)
        self.active_models = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with built-in dummy models (in production, these would load from files)"""
        # Model 1: Basic gait analysis
        self.register_model(
            name="gait_analysis_v1",
            version="1.0.0",
            model_type="classification",
            predict_func=self._dummy_gait_model,
            metadata={
                "description": "Basic gait pattern classifier",
                "trained_date": "2024-01-15",
                "accuracy": 0.87,
                "features": ["velocity", "stride_length", "cadence", "symmetry"]
            }
        )
        
        # Model 2: Fall risk prediction
        self.register_model(
            name="fall_risk_v1",
            version="1.0.0",
            model_type="regression",
            predict_func=self._dummy_fall_risk_model,
            metadata={
                "description": "Fall risk assessment model",
                "trained_date": "2024-01-20",
                "accuracy": 0.92,
                "features": ["velocity", "previous_falls", "age", "balance_score"]
            }
        )
        
        # Model 3: Parkinson's detection
        self.register_model(
            name="parkinson_detection_v1",
            version="1.0.0",
            model_type="classification",
            predict_func=self._dummy_parkinson_model,
            metadata={
                "description": "Parkinsonian gait detection",
                "trained_date": "2024-02-01",
                "accuracy": 0.89,
                "features": ["stride_variability", "velocity", "arm_swing", "freezing_episodes"]
            }
        )
    
    def register_model(self, name: str, version: str, model_type: str, 
                      predict_func: callable, metadata: Dict = None):
        """Register a new model in the registry"""
        model_id = f"{name}_{version}"
        
        self.models[model_id] = {
            "name": name,
            "version": version,
            "type": model_type,
            "predict": predict_func,
            "metadata": metadata or {},
            "registered_at": datetime.datetime.now().isoformat()
        }
        
        self.model_versions[name].append(version)
        
        if name not in self.active_models:
            self.active_models[name] = model_id
        
        print(f"✅ Model registered: {model_id}")
        return model_id
    
    def get_model(self, name: str, version: str = None):
        """Retrieve a model by name and optional version"""
        if version:
            model_id = f"{name}_{version}"
        else:
            model_id = self.active_models.get(name)
        
        if model_id in self.models:
            return self.models[model_id]
        return None
    
    def set_active_version(self, name: str, version: str):
        """Set the active version for a model"""
        model_id = f"{name}_{version}"
        if model_id in self.models:
            self.active_models[name] = model_id
            return True
        return False
    
    def list_models(self):
        """List all registered models"""
        return [
            {
                "id": model_id,
                "active": self.active_models.get(model["name"]) == model_id,
                **model
            }
            for model_id, model in self.models.items()
        ]
    
    def ensemble_predict(self, features: Dict) -> Dict:
        """Make predictions using ensemble of active models"""
        predictions = {}
        
        for name, model_id in self.active_models.items():
            model = self.models[model_id]
            try:
                pred = model["predict"](features)
                predictions[name] = {
                    "prediction": pred,
                    "confidence": pred.get("confidence", 0.5),
                    "model_info": {
                        "name": model["name"],
                        "version": model["version"],
                        "type": model["type"]
                    }
                }
            except Exception as e:
                predictions[name] = {
                    "error": str(e),
                    "confidence": 0.0
                }
        
        # Combine predictions (weighted average for regression, voting for classification)
        return self._combine_predictions(predictions)
    
    def _combine_predictions(self, predictions: Dict) -> Dict:
        """Combine multiple model predictions"""
        if not predictions:
            return {"error": "No predictions available"}
        
        # Simple ensemble logic (customize based on your needs)
        combined = {
            "ensemble_predictions": predictions,
            "average_confidence": np.mean([p.get("confidence", 0) for p in predictions.values()]),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        return combined
    
    # Dummy model implementations (replace with actual ML models)
    def _dummy_gait_model(self, features: Dict) -> Dict:
        time.sleep(0.1)  # Simulate processing time
        velocity = features.get("velocity", 1.0)
        
        if velocity > 1.1:
            gait_type = "normal"
            confidence = 0.85
        elif velocity > 0.8:
            gait_type = "impaired"
            confidence = 0.75
        else:
            gait_type = "severe"
            confidence = 0.90
        
        return {
            "gait_type": gait_type,
            "confidence": confidence,
            "recommendations": ["Consider physical therapy evaluation"]
        }
    
    def _dummy_fall_risk_model(self, features: Dict) -> Dict:
        time.sleep(0.1)
        velocity = features.get("velocity", 1.0)
        age = features.get("age", 65)
        previous_falls = features.get("previous_falls", 0)
        
        risk_score = (1.0 - velocity) * 0.4 + (age / 100) * 0.3 + (min(previous_falls, 5) / 5) * 0.3
        risk_score = min(max(risk_score, 0), 1)
        
        if risk_score < 0.3:
            risk_level = "low"
        elif risk_score < 0.6:
            risk_level = "moderate"
        elif risk_score < 0.8:
            risk_level = "high"
        else:
            risk_level = "severe"
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": 0.88,
            "factors": {
                "velocity_contribution": (1.0 - velocity) * 0.4,
                "age_contribution": (age / 100) * 0.3,
                "falls_contribution": (min(previous_falls, 5) / 5) * 0.3
            }
        }
    
    def _dummy_parkinson_model(self, features: Dict) -> Dict:
        time.sleep(0.1)
        variability = features.get("variability", 0.05)
        velocity = features.get("velocity", 1.0)
        
        # Simple heuristic
        parkinson_prob = min(max((0.1 - variability) * 10 + (1.0 - velocity) * 2, 0), 1)
        
        return {
            "parkinson_probability": parkinson_prob,
            "detected": parkinson_prob > 0.5,
            "confidence": abs(parkinson_prob - 0.5) * 2,
            "key_indicators": ["Reduced velocity", "Increased variability"]
        }

# ============================================================================
# CORE GAIT ANALYZER ENGINE
# ============================================================================

class GaitAnalyzer:
    """Main analysis engine with separation of concerns"""
    
    def __init__(self, model_registry: ModelRegistry = None):
        self.model_registry = model_registry or ModelRegistry()
        self.knowledge_base = ClinicalKnowledgeBase()
        self.config = Config()
        self.audit_log = []
        self.cache = {}
        
    def analyze_gait(self, patient_data: Dict, gait_metrics: GaitMetrics) -> Dict:
        """Main analysis pipeline"""
        start_time = time.time()
        
        # 1. Validate inputs
        self._validate_inputs(patient_data, gait_metrics)
        
        # 2. Calculate derived metrics
        derived_metrics = self._calculate_derived_metrics(gait_metrics)
        
        # 3. Get model predictions
        model_input = {
            **patient_data,
            **gait_metrics.to_dict(),
            **derived_metrics
        }
        
        predictions = self.model_registry.ensemble_predict(model_input)
        
        # 4. Match with clinical knowledge
        pattern_matches = self.knowledge_base.get_pattern_by_metrics(gait_metrics)
        
        # 5. Generate risk assessment
        risk_assessment = self._assess_risk(patient_data, gait_metrics, predictions)
        
        # 6. Generate recommendations
        recommendations = self._generate_recommendations(risk_assessment, pattern_matches)
        
        # 7. Create comprehensive report
        report = {
            "patient_info": patient_data,
            "gait_metrics": gait_metrics.to_dict(),
            "derived_metrics": derived_metrics,
            "model_predictions": predictions,
            "pattern_matches": pattern_matches,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "clinical_insights": self._generate_insights(gait_metrics, pattern_matches),
            "analysis_timestamp": datetime.datetime.now().isoformat(),
            "processing_time_ms": (time.time() - start_time) * 1000
        }
        
        # 8. Audit logging
        self._log_analysis(patient_data.get("patient_id", "unknown"), report)
        
        return report
    
    def _validate_inputs(self, patient_data: Dict, gait_metrics: GaitMetrics):
        """Validate input data"""
        if not patient_data.get("patient_id"):
            raise ValueError("Patient ID is required")
        
        if gait_metrics.velocity <= 0:
            raise ValueError("Velocity must be positive")
        
        if not (0 < gait_metrics.stance_percentage < 100):
            raise ValueError("Stance percentage must be between 0 and 100")
    
    def _calculate_derived_metrics(self, metrics: GaitMetrics) -> Dict:
        """Calculate additional gait metrics"""
        # Gait cycle time (seconds)
        cycle_time = 60 / metrics.cadence if metrics.cadence > 0 else 0
        
        # Single support time
        single_support = 100 - metrics.stance_percentage
        
        # Velocity normalized by height
        normalized_velocity = metrics.velocity
        
        # Symmetry ratio (left/right if available, otherwise estimate)
        symmetry_ratio = metrics.symmetry_index
        
        return {
            "gait_cycle_time_sec": cycle_time,
            "single_support_percentage": single_support,
            "normalized_velocity": normalized_velocity,
            "symmetry_ratio": symmetry_ratio,
            "stride_time_variability": metrics.variability
        }
    
    def _assess_risk(self, patient_data: Dict, metrics: GaitMetrics, predictions: Dict) -> Dict:
        """Comprehensive risk assessment"""
        risk_factors = []
        
        # Velocity risk
        if metrics.velocity < self.config.CLINICAL_PARAMS["gait_velocity_thresholds"]["severe"]:
            risk_factors.append(("Low velocity", 0.8))
        elif metrics.velocity < self.config.CLINICAL_PARAMS["gait_velocity_thresholds"]["impaired"]:
            risk_factors.append(("Reduced velocity", 0.5))
        
        # Age risk
        age = patient_data.get("age", 65)
        if age > 80:
            risk_factors.append(("Advanced age", 0.6))
        elif age > 65:
            risk_factors.append(("Elderly", 0.3))
        
        # Previous falls risk
        previous_falls = patient_data.get("previous_falls", 0)
        if previous_falls >= 3:
            risk_factors.append(("Multiple previous falls", 0.9))
        elif previous_falls >= 1:
            risk_factors.append(("Previous fall", 0.4))
        
        # Variability risk
        if metrics.variability > 0.1:
            risk_factors.append(("High gait variability", 0.7))
        
        # Calculate overall risk score
        if risk_factors:
            risk_score = sum(weight for _, weight in risk_factors) / len(risk_factors)
        else:
            risk_score = 0.1
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = RiskLevel.SEVERE
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.3:
            risk_level = RiskLevel.MODERATE
        else:
            risk_level = RiskLevel.LOW
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level.value,
            "risk_factors": [{"factor": f, "weight": w} for f, w in risk_factors],
            "confidence": 0.85  # Could be based on model confidence
        }
    
    def _generate_recommendations(self, risk_assessment: Dict, pattern_matches: List) -> List[Dict]:
        """Generate personalized recommendations"""
        recommendations = []
        risk_level = risk_assessment["risk_level"]
        
        # Base recommendations from knowledge base
        base_recs = self.knowledge_base.RECOMMENDATIONS.get(risk_level, [])
        
        for rec in base_recs:
            recommendations.append({
                "type": "general",
                "priority": "medium",
                "recommendation": rec,
                "rationale": f"Based on {risk_level} risk level"
            })
        
        # Pattern-specific recommendations
        if pattern_matches:
            best_pattern = pattern_matches[0]
            if best_pattern["pattern"] == "parkinsonian":
                recommendations.append({
                    "type": "specific",
                    "priority": "high",
                    "recommendation": "Consider neurological evaluation for Parkinson's disease",
                    "rationale": "Gait pattern matches Parkinsonian characteristics"
                })
        
        # Velocity-specific recommendations
        if risk_assessment["risk_score"] > 0.5:
            recommendations.append({
                "type": "intervention",
                "priority": "high",
                "recommendation": "Immediate physical therapy referral",
                "rationale": "High fall risk score requires urgent intervention"
            })
        
        return recommendations
    
    def _generate_insights(self, metrics: GaitMetrics, pattern_matches: List) -> List[str]:
        """Generate clinical insights"""
        insights = []
        
        # Velocity insight
        if metrics.velocity < 0.6:
            insights.append("Critically low gait velocity indicates severe mobility impairment")
        elif metrics.velocity < 0.8:
            insights.append("Reduced gait velocity suggests moderate impairment")
        
        # Symmetry insight
        if metrics.symmetry_index < 0.8:
            insights.append("Asymmetric gait pattern detected - may indicate unilateral impairment")
        
        # Variability insight
        if metrics.variability > 0.15:
            insights.append("High gait variability suggests impaired motor control")
        
        # Pattern insights
        if pattern_matches:
            best_match = pattern_matches[0]
            insights.append(f"Gait pattern most closely matches {best_match['pattern']} (confidence: {best_match['confidence']:.1%})")
        
        return insights
    
    def _log_analysis(self, patient_id: str, report: Dict):
        """Log analysis for audit trail"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "patient_id": patient_id,
            "action": "gait_analysis",
            "risk_level": report["risk_assessment"]["risk_level"],
            "models_used": list(report["model_predictions"].get("ensemble_predictions", {}).keys()),
            "summary": f"Analysis completed for {patient_id} - Risk: {report['risk_assessment']['risk_level']}"
        }
        
        self.audit_log.append(log_entry)
        
        # Keep only last 1000 entries in memory
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

# ============================================================================
# SECURITY & DATA ENCRYPTION
# ============================================================================

class SecurityManager:
    """Handles data encryption and security compliance"""
    
    @staticmethod
    def anonymize_patient_data(data: Dict) -> Dict:
        """Anonymize patient data for logging/analytics"""
        anonymized = data.copy()
        
        if "patient_id" in anonymized:
            # Hash the patient ID
            anonymized["patient_id"] = hashlib.sha256(
                anonymized["patient_id"].encode()
            ).hexdigest()[:12]
        
        # Remove sensitive fields
        sensitive_fields = ["name", "email", "phone", "address", "ssn", "mrn"]
        for field in sensitive_fields:
            if field in anonymized:
                del anonymized[field]
        
        return anonymized
    
    @staticmethod
    def encrypt_sensitive_data(data: str) -> str:
        """Simple encryption for demonstration (use proper encryption in production)"""
        # This is a simple base64 encoding for demo purposes
        # In production, use proper encryption like AES
        encoded = base64.b64encode(data.encode()).decode()
        return f"encrypted:{encoded}"
    
    @staticmethod
    def decrypt_sensitive_data(encrypted: str) -> str:
        """Decrypt sensitive data"""
        if encrypted.startswith("encrypted:"):
            encoded = encrypted[10:]  # Remove 'encrypted:' prefix
            return base64.b64decode(encoded).decode()
        return encrypted

# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

class DataValidator:
    """Validates and cleans input data"""
    
    @staticmethod
    def validate_gait_data(data: Dict) -> Tuple[bool, List[str]]:
        """Validate gait data structure and values"""
        errors = []
        
        required_fields = ["velocity", "stride_length", "cadence"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Validate numerical ranges
        if "velocity" in data:
            if data["velocity"] <= 0 or data["velocity"] > 3:
                errors.append("Velocity must be between 0 and 3 m/s")
        
        if "cadence" in data:
            if data["cadence"] <= 0 or data["cadence"] > 200:
                errors.append("Cadence must be between 0 and 200 steps/min")
        
        if "stance_percentage" in data:
            if not (0 <= data["stance_percentage"] <= 100):
                errors.append("Stance percentage must be between 0 and 100")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_numeric_values(data: Dict) -> Dict:
        """Clean and normalize numeric values"""
        cleaned = data.copy()
        
        numeric_fields = ["velocity", "stride_length", "stance_percentage", 
                         "swing_percentage", "double_support_time", "step_width",
                         "symmetry_index", "variability"]
        
        for field in numeric_fields:
            if field in cleaned:
                try:
                    cleaned[field] = float(cleaned[field])
                except (ValueError, TypeError):
                    cleaned[field] = 0.0
        
        return cleaned

# ============================================================================
# STREAMLIT UI COMPONENTS
# ============================================================================

class UIComponents:
    """Reusable Streamlit UI components"""
    
    @staticmethod
    def create_metric_card(title: str, value: Any, delta: str = None, 
                          help_text: str = None, color: str = None):
        """Create a metric card display"""
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                help=help_text
            )
        
        if color:
            with col2:
                # Color indicator
                st.markdown(
                    f'<div style="width: 20px; height: 20px; background-color: {color}; '
                    f'border-radius: 50%; margin-top: 10px;"></div>',
                    unsafe_allow_html=True
                )
    
    @staticmethod
    def create_gait_visualization(metrics: GaitMetrics):
        """Create visualization of gait metrics"""
        fig = go.Figure()
        
        # Radar chart for key metrics
        categories = ['Velocity', 'Stride Length', 'Cadence', 'Symmetry', 'Stability']
        
        # Normalize values for radar chart
        normalized_velocity = metrics.velocity / 1.5  # Normalize to 1.5 m/s
        normalized_stride = metrics.stride_length / 1.5  # Normalize to 1.5 m
        normalized_cadence = metrics.cadence / 120  # Normalize to 120 steps/min
        symmetry = metrics.symmetry_index
        stability = 1 - min(metrics.variability * 10, 1)  # Invert variability
        
        values = [
            normalized_velocity,
            normalized_stride,
            normalized_cadence,
            symmetry,
            stability
        ]
        
        # Close the radar chart
        values = values + [values[0]]
        categories = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Patient Gait',
            line_color='#1f77b4'
        ))
        
        # Add reference line for normal gait
        normal_values = [0.8, 0.8, 0.8, 0.8, 0.8] + [0.8]  # 80% of max for all metrics
        fig.add_trace(go.Scatterpolar(
            r=normal_values,
            theta=categories,
            fill='toself',
            name='Normal Range',
            line_color='#2ca02c',
            opacity=0.3
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Gait Metrics Radar Chart",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_risk_gauge(risk_score: float, risk_level: str):
        """Create a risk gauge visualization"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fall Risk Score", 'font': {'size': 24}},
            delta={'reference': 50, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': 'green'},
                    {'range': [30, 60], 'color': 'yellow'},
                    {'range': [60, 80], 'color': 'orange'},
                    {'range': [80, 100], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': risk_score * 100
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MAIN STREAMLIT APPLICATION
# ============================================================================

class GaitAnalysisApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.analyzer = GaitAnalyzer()
        self.validator = DataValidator()
        self.security = SecurityManager()
        self.ui = UIComponents()
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        
        if 'patient_history' not in st.session_state:
            st.session_state.patient_history = {}
        
        if 'model_registry' not in st.session_state:
            st.session_state.model_registry = self.analyzer.model_registry
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Gait Analysis AI Tool",
            page_icon="🚶",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._inject_css()
        
        # Sidebar
        with st.sidebar:
            self._render_sidebar()
        
        # Main content
        self._render_header()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Analysis", 
            "👤 Patient Info", 
            "🤖 Models", 
            "📈 History",
            "⚙️ Settings"
        ])
        
        with tab1:
            self._render_analysis_tab()
        
        with tab2:
            self._render_patient_tab()
        
        with tab3:
            self._render_models_tab()
        
        with tab4:
            self._render_history_tab()
        
        with tab5:
            self._render_settings_tab()
    
    def _inject_css(self):
        """Inject custom CSS"""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            padding-bottom: 1rem;
            border-bottom: 2px solid #1f77b4;
        }
        .risk-low { color: green; font-weight: bold; }
        .risk-moderate { color: orange; font-weight: bold; }
        .risk-high { color: red; font-weight: bold; }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar content"""
        st.title("🚶 Gait Analysis")
        st.markdown("---")
        
        st.markdown("### Quick Actions")
        
        if st.button("🔄 New Analysis", use_container_width=True):
            st.rerun()
        
        if st.button("📥 Export Report", use_container_width=True):
            self._export_report()
        
        if st.button("🧹 Clear History", use_container_width=True):
            st.session_state.analysis_results = []
            st.success("History cleared!")
        
        st.markdown("---")
        
        st.markdown("### System Status")
        st.info(f"Models: {len(self.analyzer.model_registry.models)} loaded")
        st.info(f"Analyses: {len(st.session_state.analysis_results)}")
        
        if self.analyzer.audit_log:
            latest = self.analyzer.audit_log[-1]
            st.caption(f"Latest: {latest.get('summary', '')}")
    
    def _render_header(self):
        """Render main header"""
        st.markdown('<h1 class="main-header">🚶 AI Gait Analysis Tool</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        *Clinical gait analysis and fall risk assessment using AI models*
        """)
    
    def _render_analysis_tab(self):
        """Render the main analysis tab"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Gait Metrics Input")
            
            # Input form
            with st.form("gait_analysis_form"):
                st.subheader("Patient Information")
                
                patient_id = st.text_input("Patient ID", value="PT-001")
                age = st.number_input("Age", min_value=0, max_value=120, value=65)
                height = st.number_input("Height (cm)", min_value=50, max_value=250, value=170)
                weight = st.number_input("Weight (kg)", min_value=20, max_value=200, value=70)
                previous_falls = st.number_input("Previous Falls (last year)", 
                                                min_value=0, max_value=50, value=0)
                
                st.subheader("Gait Metrics")
                
                # Create columns for metrics
                m1, m2 = st.columns(2)
                with m1:
                    velocity = st.number_input("Velocity (m/s)", 
                                              min_value=0.0, max_value=3.0, 
                                              value=0.9, step=0.1)
                    stride_length = st.number_input("Stride Length (m)", 
                                                   min_value=0.0, max_value=3.0, 
                                                   value=1.1, step=0.1)
                    cadence = st.number_input("Cadence (steps/min)", 
                                             min_value=0, max_value=200, 
                                             value=85, step=1)
                
                with m2:
                    stance_percentage = st.slider("Stance Phase (%)", 
                                                 min_value=0, max_value=100, 
                                                 value=65)
                    symmetry_index = st.slider("Symmetry Index", 
                                              min_value=0.0, max_value=1.0, 
                                              value=0.75, step=0.05)
                    variability = st.slider("Gait Variability", 
                                           min_value=0.0, max_value=0.3, 
                                           value=0.08, step=0.01)
                
                # Additional metrics
                step_width = st.number_input("Step Width (m)", 
                                            min_value=0.0, max_value=1.0, 
                                            value=0.15, step=0.01)
                double_support = st.number_input("Double Support Time (s)", 
                                                min_value=0.0, max_value=2.0, 
                                                value=0.25, step=0.05)
                
                submit_button = st.form_submit_button("🚀 Analyze Gait", use_container_width=True)
        
        with col2:
            st.header("Quick Preview")
            
            if submit_button:
                # Validate data
                gait_data = {
                    "velocity": velocity,
                    "stride_length": stride_length,
                    "cadence": cadence,
                    "stance_percentage": stance_percentage,
                    "swing_percentage": 100 - stance_percentage,
                    "double_support_time": double_support,
                    "step_width": step_width,
                    "symmetry_index": symmetry_index,
                    "variability": variability
                }
                
                is_valid, errors = self.validator.validate_gait_data(gait_data)
                
                if not is_valid:
                    for error in errors:
                        st.error(f"Validation Error: {error}")
                else:
                    # Create patient info
                    patient_info = {
                        "patient_id": patient_id,
                        "age": age,
                        "height_cm": height,
                        "weight_kg": weight,
                        "previous_falls": previous_falls,
                        "assessment_date": datetime.datetime.now().strftime("%Y-%m-%d")
                    }
                    
                    # Create gait metrics object
                    metrics = GaitMetrics(**gait_data)
                    
                    # Perform analysis
                    with st.spinner("Analyzing gait patterns..."):
                        result = self.analyzer.analyze_gait(patient_info, metrics)
                    
                    # Store result
                    st.session_state.analysis_results.append(result)
                    st.session_state.patient_history[patient_id] = result
                    
                    # Display results
                    self._display_analysis_results(result)
            else:
                st.info("Enter metrics and click 'Analyze Gait' to see results")
                st.image("https://via.placeholder.com/400x300?text=Gait+Preview", 
                        caption="Sample Gait Analysis")
    
    def _display_analysis_results(self, result: Dict):
        """Display analysis results"""
        st.success("✅ Analysis Complete!")
        
        # Risk gauge
        risk_score = result["risk_assessment"]["risk_score"]
        risk_level = result["risk_assessment"]["risk_level"]
        
        self.ui.create_risk_gauge(risk_score, risk_level)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.ui.create_metric_card(
                "Gait Velocity",
                f"{result['gait_metrics']['velocity']:.2f} m/s",
                "Low" if result['gait_metrics']['velocity'] < 1.0 else "Normal",
                color="red" if result['gait_metrics']['velocity'] < 0.8 else "green"
            )
        
        with col2:
            self.ui.create_metric_card(
                "Fall Risk",
                risk_level.upper(),
                f"Score: {risk_score:.1%}",
                color={
                    "low": "green",
                    "moderate": "yellow",
                    "high": "orange",
                    "severe": "red"
                }.get(risk_level, "gray")
            )
        
        with col3:
            best_pattern = result["pattern_matches"][0]["pattern"] if result["pattern_matches"] else "Unknown"
            self.ui.create_metric_card(
                "Gait Pattern",
                best_pattern.title(),
                f"Confidence: {result['pattern_matches'][0]['confidence']:.1%}" if result["pattern_matches"] else ""
            )
        
        with col4:
            self.ui.create_metric_card(
                "Model Confidence",
                f"{result['model_predictions'].get('average_confidence', 0):.1%}",
                "Ensemble"
            )
        
        # Visualization
        st.subheader("Gait Visualization")
        metrics_obj = GaitMetrics.from_dict(result["gait_metrics"])
        self.ui.create_gait_visualization(metrics_obj)
        
        # Recommendations
        st.subheader("Clinical Recommendations")
        for i, rec in enumerate(result["recommendations"], 1):
            priority_color = {
                "high": "red",
                "medium": "orange",
                "low": "green"
            }.get(rec["priority"], "gray")
            
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 1rem; border-left: 4px solid {priority_color}; 
                        margin-bottom: 0.5rem; border-radius: 0.25rem;">
                <strong>{i}. {rec['recommendation']}</strong><br>
                <small>Priority: <span style="color: {priority_color}">{rec['priority'].upper()}</span> | 
                Type: {rec['type']}</small><br>
                <em>{rec['rationale']}</em>
            </div>
            """, unsafe_allow_html=True)
        
        # Insights
        if result["clinical_insights"]:
            st.subheader("Clinical Insights")
            for insight in result["clinical_insights"]:
                st.info(f"💡 {insight}")
        
        # Model details (expandable)
        with st.expander("📊 View Detailed Model Predictions"):
            for model_name, prediction in result["model_predictions"].get("ensemble_predictions", {}).items():
                st.write(f"**{model_name}**")
                st.json(prediction)
    
    def _render_patient_tab(self):
        """Render patient information tab"""
        st.header("Patient Management")
        
        # Patient search
        search_col, action_col = st.columns([3, 1])
        with search_col:
            patient_search = st.text_input("Search Patient ID")
        
        with action_col:
            if st.button("➕ New Patient", use_container_width=True):
                st.info("New patient form would open here")
        
        # Display patient history
        if st.session_state.patient_history:
            for patient_id, analysis in list(st.session_state.patient_history.items())[:5]:
                with st.expander(f"Patient: {patient_id}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Age:** {analysis['patient_info'].get('age', 'N/A')}")
                        st.write(f"**Last Assessment:** {analysis['patient_info'].get('assessment_date', 'N/A')}")
                        st.write(f"**Previous Falls:** {analysis['patient_info'].get('previous_falls', 0)}")
                    
                    with col2:
                        risk_level = analysis['risk_assessment']['risk_level']
                        st.write(f"**Risk Level:** {risk_level.upper()}")
                        st.write(f"**Gait Velocity:** {analysis['gait_metrics']['velocity']:.2f} m/s")
                        st.write(f"**Cadence:** {analysis['gait_metrics']['cadence']} steps/min")
                    
                    if st.button(f"View Full Report", key=f"view_{patient_id}"):
                        self._display_analysis_results(analysis)
        else:
            st.info("No patient history available. Run an analysis first.")
    
    def _render_models_tab(self):
        """Render model management tab"""
        st.header("Model Registry")
        st.markdown("Plug-and-play model management")
        
        # List all models
        models = self.analyzer.model_registry.list_models()
        
        for model in models:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.write(f"**{model['name']}** (v{model['version']})")
                    st.caption(model['metadata'].get('description', 'No description'))
                    st.write(f"Type: {model['type']} | Accuracy: {model['metadata'].get('accuracy', 'N/A')}")
                
                with col2:
                    if model['active']:
                        st.success("✅ Active")
                    else:
                        st.warning("Inactive")
                
                with col3:
                    if not model['active']:
                        if st.button("Activate", key=f"activate_{model['id']}"):
                            self.analyzer.model_registry.set_active_version(
                                model['name'], model['version']
                            )
                            st.rerun()
                
                st.markdown("---")
        
        # Model statistics
        st.subheader("Model Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Models", len(models))
        
        with col2:
            active_count = sum(1 for m in models if m['active'])
            st.metric("Active Models", active_count)
        
        with col3:
            avg_acc = np.mean([
                m['metadata'].get('accuracy', 0) 
                for m in models 
                if isinstance(m['metadata'].get('accuracy'), (int, float))
            ])
            st.metric("Avg Accuracy", f"{avg_acc:.1%}" if avg_acc > 0 else "N/A")
    
    def _render_history_tab(self):
        """Render analysis history tab"""
        st.header("Analysis History")
        
        if not st.session_state.analysis_results:
            st.info("No analysis history available. Run an analysis first.")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_filter = st.selectbox(
                "Filter by Risk Level",
                ["All", "low", "moderate", "high", "severe"]
            )
        
        with col2:
            date_filter = st.date_input(
                "Filter by Date",
                datetime.date.today()
            )
        
        with col3:
            items_per_page = st.selectbox(
                "Items per page",
                [5, 10, 25, 50],
                index=0
            )
        
        # Display history table
        history_df = self._create_history_dataframe()
        
        if risk_filter != "All":
            history_df = history_df[history_df["risk_level"] == risk_filter]
        
        # Pagination
        total_items = len(history_df)
        page_number = st.number_input(
            "Page", 
            min_value=1, 
            max_value=max(1, (total_items // items_per_page) + 1),
            value=1
        )
        
        start_idx = (page_number - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        if total_items > 0:
            st.dataframe(
                history_df.iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True
            )
            
            st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_items} records")
        else:
            st.info("No records match the selected filters.")
    
    def _render_settings_tab(self):
        """Render settings tab"""
        st.header("Application Settings")
        
        # Configuration editor
        st.subheader("Clinical Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            normal_velocity = st.number_input(
                "Normal Gait Velocity (m/s)",
                value=Config.CLINICAL_PARAMS["gait_velocity_thresholds"]["normal"],
                min_value=0.5,
                max_value=2.0,
                step=0.1
            )
            
            normal_stride = st.number_input(
                "Normal Stride Length (m)",
                value=Config.CLINICAL_PARAMS["stride_length_thresholds"]["normal"],
                min_value=0.5,
                max_value=2.0,
                step=0.1
            )
        
        with col2:
            confidence_thresh = st.slider(
                "Model Confidence Threshold",
                value=Config.CLINICAL_PARAMS["confidence_threshold"],
                min_value=0.5,
                max_value=1.0,
                step=0.05
            )
            
            if st.button("💾 Save Configuration", use_container_width=True):
                Config.CLINICAL_PARAMS["gait_velocity_thresholds"]["normal"] = normal_velocity
                Config.CLINICAL_PARAMS["stride_length_thresholds"]["normal"] = normal_stride
                Config.CLINICAL_PARAMS["confidence_threshold"] = confidence_thresh
                st.success("Configuration saved!")
        
        # System settings
        st.subheader("System Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=["DEBUG", "INFO", "WARNING", "ERROR"].index(Config.SYSTEM["log_level"])
            )
            
            enable_audit = st.checkbox(
                "Enable Audit Logging",
                value=Config.SYSTEM["enable_audit_log"]
            )
        
        with col2:
            cache_size = st.number_input(
                "Cache Size",
                value=Config.SYSTEM["cache_size"],
                min_value=100,
                max_value=10000,
                step=100
            )
            
            encryption_enabled = st.checkbox(
                "Enable Encryption",
                value=Config.SYSTEM["encryption_enabled"]
            )
        
        if st.button("🔄 Update System Settings", use_container_width=True):
            Config.SYSTEM.update({
                "log_level": log_level,
                "cache_size": cache_size,
                "enable_audit_log": enable_audit,
                "encryption_enabled": encryption_enabled
            })
            st.success("System settings updated!")
    
    def _create_history_dataframe(self) -> pd.DataFrame:
        """Convert analysis history to DataFrame"""
        data = []
        
        for result in st.session_state.analysis_results:
            data.append({
                "patient_id": result["patient_info"]["patient_id"],
                "age": result["patient_info"]["age"],
                "velocity": result["gait_metrics"]["velocity"],
                "cadence": result["gait_metrics"]["cadence"],
                "risk_score": result["risk_assessment"]["risk_score"],
                "risk_level": result["risk_assessment"]["risk_level"],
                "pattern": result["pattern_matches"][0]["pattern"] if result["pattern_matches"] else "Unknown",
                "timestamp": result["analysis_timestamp"],
                "processing_time": result["processing_time_ms"]
            })
        
        return pd.DataFrame(data)
    
    def _export_report(self):
        """Export current analysis report"""
        if not st.session_state.analysis_results:
            st.warning("No analysis to export")
            return
        
        latest_result = st.session_state.analysis_results[-1]
        
        # Convert to JSON
        report_json = json.dumps(latest_result, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Download JSON Report",
            data=report_json,
            file_name=f"gait_analysis_{latest_result['patient_info']['patient_id']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        # Also show preview
        with st.expander("📄 Report Preview"):
            st.json(latest_result)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    # Initialize the application
    app = GaitAnalysisApp()
    
    # Run the application
    app.run()

if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore')
    
    # Run the application
    main()