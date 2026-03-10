"""Health Analysis Tools - Medical report analysis and health insights.

This module contains all the medical analysis tools for blood reports,
health indicators, and medical insights similar to HIA functionality.
"""

import logging
import re
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime

import pdfplumber
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)

# =============================================================================
# Medical Analysis Constants
# =============================================================================

# Health indicators to track
HEALTH_INDICATORS = [
    "hemoglobin", "glucose", "cholesterol", "triglycerides", 
    "hdl", "ldl", "wbc", "rbc", "platelet", "creatinine",
    "alt", "ast", "alp", "bilirubin", "tsh", "t4", "sodium",
    "potassium", "bun", "protein", "albumin"
]

# Reference ranges for common blood tests
REFERENCE_RANGES = {
    "hemoglobin": {"min": 12.0, "max": 15.5, "unit": "g/dL"},
    "wbc": {"min": 4000, "max": 11000, "unit": "/µL"},
    "rbc": {"min": 4.0, "max": 5.2, "unit": "M/µL"},
    "platelets": {"min": 150000, "max": 450000, "unit": "/µL"},
    "glucose": {"min": 70, "max": 100, "unit": "mg/dL"},
    "creatinine": {"min": 0.6, "max": 1.2, "unit": "mg/dL"},
    "cholesterol": {"min": 0, "max": 200, "unit": "mg/dL"},
    "hdl": {"min": 40, "max": 999, "unit": "mg/dL"},
    "ldl": {"min": 0, "max": 100, "unit": "mg/dL"},
    "triglycerides": {"min": 0, "max": 150, "unit": "mg/dL"},
    "alt": {"min": 7, "max": 56, "unit": "U/L"},
    "ast": {"min": 10, "max": 40, "unit": "U/L"},
    "alp": {"min": 44, "max": 147, "unit": "U/L"},
    "bilirubin": {"min": 0.3, "max": 1.2, "unit": "mg/dL"},
    "tsh": {"min": 0.4, "max": 4.0, "unit": "µIU/mL"},
    "t4": {"min": 0.8, "max": 1.8, "unit": "ng/dL"},
    "sodium": {"min": 135, "max": 145, "unit": "mEq/L"},
    "potassium": {"min": 3.5, "max": 5.0, "unit": "mEq/L"},
    "bun": {"min": 7, "max": 20, "unit": "mg/dL"},
    "protein": {"min": 6.0, "max": 8.0, "unit": "g/dL"},
    "albumin": {"min": 3.5, "max": 5.5, "unit": "g/dL"},
}

def extract_text_from_pdf(pdf_content: str) -> Dict[str, Any]:
    """Extract and analyze text from medical report PDF content.
    
    Args:
        pdf_content: Raw text content from PDF
        
    Returns:
        Dictionary with extracted text and analysis results
    """
    try:
        # Validate content
        if not pdf_content or len(pdf_content.strip()) < 50:
            return {
                "success": False,
                "error": "Extracted text is too short. Please ensure the PDF contains valid medical report text."
            }
        
        # Check for medical content
        medical_terms = [
            'blood', 'test', 'report', 'laboratory', 'lab', 'patient', 'specimen',
            'reference range', 'analysis', 'results', 'medical', 'diagnostic',
            'hemoglobin', 'wbc', 'rbc', 'platelet', 'glucose', 'creatinine'
        ]
        
        text_lower = pdf_content.lower()
        term_matches = sum(1 for term in medical_terms if term in text_lower)
        
        if term_matches < 3:
            return {
                "success": False,
                "error": "The uploaded content doesn't appear to be a medical report. Please upload a valid medical report."
            }
        
        # Extract health indicators
        extracted_values = extract_health_indicators(pdf_content)
        
        return {
            "success": True,
            "text": pdf_content,
            "extracted_values": extracted_values,
            "medical_terms_found": term_matches
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF content: {e}")
        return {
            "success": False,
            "error": f"Error processing medical report: {str(e)}"
        }

def extract_health_indicators(text: str) -> Dict[str, Any]:
    """Extract health indicator values from medical report text.
    
    Args:
        text: Medical report text
        
    Returns:
        Dictionary with extracted indicator values and analysis
    """
    extracted = {}
    abnormalities = []
    
    for indicator in HEALTH_INDICATORS:
        # Look for patterns like "Hemoglobin: 13.5 g/dL" or "Glucose 95 mg/dL"
        patterns = [
            rf"{indicator}:\s*([\d.]+)\s*([a-zA-Z/µ]+)",
            rf"{indicator}\s*([\d.]+)\s*([a-zA-Z/µ]+)",
            rf"([\d.]+)\s*([a-zA-Z/µ]+)\s*{indicator}",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                extracted[indicator] = {"value": value, "unit": unit}
                
                # Check if value is abnormal
                if indicator in REFERENCE_RANGES:
                    ref_range = REFERENCE_RANGES[indicator]
                    ref_min = float(ref_range["min"])
                    ref_max = float(ref_range["max"])
                    
                    # Skip comparison if max is 0 (indicating no upper limit)
                    if ref_max > 0 and (value < ref_min or value > ref_max):
                        abnormalities.append({
                            "indicator": indicator,
                            "value": value,
                            "unit": unit,
                            "reference_range": ref_range,
                            "status": "low" if value < ref_min else "high"
                        })
                break
    
    return {
        "extracted_indicators": extracted,
        "abnormalities": abnormalities,
        "total_indicators": len(extracted)
    }

def analyze_medical_report(report_text: str, patient_age: Optional[int] = None, 
                          patient_gender: Optional[str] = None) -> Dict[str, Any]:
    """Perform comprehensive medical report analysis.
    
    Args:
        report_text: Medical report text
        patient_age: Patient age (optional)
        patient_gender: Patient gender (optional)
        
    Returns:
        Comprehensive medical analysis results
    """
    try:
        # Extract content and validate
        extraction_result = extract_text_from_pdf(report_text)
        if not extraction_result["success"]:
            return extraction_result
        
        # Get extracted values
        extracted_data = extraction_result["extracted_values"]
        indicators = extracted_data.get("extracted_indicators", {})
        abnormalities = extracted_data.get("abnormalities", [])
        
        # Generate health insights
        health_insights = generate_health_insights(
            indicators, abnormalities, patient_age, patient_gender
        )
        
        # Risk assessment
        risk_assessment = assess_health_risks(abnormalities, patient_age, patient_gender)
        
        # Recommendations
        recommendations = generate_recommendations(abnormalities, indicators)
        
        return {
            "success": True,
            "analysis_timestamp": datetime.now().isoformat(),
            "patient_profile": {
                "age": patient_age,
                "gender": patient_gender
            },
            "extracted_indicators": indicators,
            "abnormalities": abnormalities,
            "health_insights": health_insights,
            "risk_assessment": risk_assessment,
            "recommendations": recommendations,
            "summary": generate_analysis_summary(abnormalities, risk_assessment)
        }
        
    except Exception as e:
        logger.error(f"Error in medical analysis: {e}")
        return {
            "success": False,
            "error": f"Medical analysis failed: {str(e)}"
        }

def generate_health_insights(indicators: Dict[str, Any], abnormalities: List[Dict], 
                           age: Optional[int], gender: Optional[str]) -> Dict[str, Any]:
    """Generate detailed health insights from indicators.
    
    Args:
        indicators: Extracted health indicators
        abnormalities: List of abnormal values
        age: Patient age
        gender: Patient gender
        
    Returns:
        Health insights analysis
    """
    insights = {
        "overall_health_status": "normal",
        "system_analysis": {},
        "key_findings": []
    }
    
    # Analyze different body systems
    system_categories = {
        "blood_count": ["hemoglobin", "wbc", "rbc", "platelets"],
        "liver_function": ["alt", "ast", "alp", "bilirubin"],
        "kidney_function": ["creatinine", "bun", "electrolytes"],
        "metabolic": ["glucose", "cholesterol", "hdl", "ldl", "triglycerides"],
        "thyroid": ["tsh", "t4"]
    }
    
    for system, indicators_list in system_categories.items():
        system_abnormalities = [
            ab for ab in abnormalities 
            if ab["indicator"] in indicators_list
        ]
        
        if system_abnormalities:
            insights["system_analysis"][system] = {
                "status": "abnormal",
                "abnormalities": system_abnormalities,
                "interpretation": interpret_system_abnormalities(system, system_abnormalities)
            }
        else:
            insights["system_analysis"][system] = {
                "status": "normal",
                "interpretation": f"{system.replace('_', ' ').title()} appears to be functioning normally."
            }
    
    # Determine overall health status
    if abnormalities:
        if len(abnormalities) > 3:
            insights["overall_health_status"] = "requires_attention"
        else:
            insights["overall_health_status"] = "mildly_abnormal"
    
    # Generate key findings
    for abnormality in abnormalities:
        indicator = abnormality["indicator"]
        value = abnormality["value"]
        status = abnormality["status"]
        
        finding = f"{indicator.title()} is {status} at {value} {abnormality['unit']}"
        insights["key_findings"].append(finding)
    
    return insights

def interpret_system_abnormalities(system: str, abnormalities: List[Dict]) -> str:
    """Intertract abnormalities for a specific body system.
    
    Args:
        system: Body system name
        abnormalities: List of abnormalities in that system
        
    Returns:
        Interpretation text
    """
    interpretations = {
        "blood_count": "Blood count abnormalities may indicate anemia, infection, or clotting issues.",
        "liver_function": "Liver function abnormalities suggest potential liver inflammation or damage.",
        "kidney_function": "Kidney function abnormalities may indicate reduced kidney performance.",
        "metabolic": "Metabolic abnormalities suggest potential diabetes, cholesterol issues, or metabolic syndrome.",
        "thyroid": "Thyroid abnormalities indicate potential thyroid dysfunction requiring evaluation."
    }
    
    return interpretations.get(system, "Abnormalities detected that require medical evaluation.")

def assess_health_risks(abnormalities: List[Dict], age: Optional[int], 
                       gender: Optional[str]) -> Dict[str, Any]:
    """Assess health risks based on abnormalities and patient profile.
    
    Args:
        abnormalities: List of abnormal values
        age: Patient age
        gender: Patient gender
        
    Returns:
        Risk assessment results
    """
    risk_factors = []
    risk_level = "low"
    
    # Risk assessment based on abnormalities
    for abnormality in abnormalities:
        indicator = abnormality["indicator"]
        
        if indicator in ["cholesterol", "ldl", "triglycerides"]:
            risk_factors.append("Cardiovascular disease risk")
        elif indicator in ["glucose"]:
            risk_factors.append("Diabetes risk")
        elif indicator in ["alt", "ast", "alp"]:
            risk_factors.append("Liver disease risk")
        elif indicator in ["creatinine", "bun"]:
            risk_factors.append("Kidney disease risk")
        elif indicator in ["hemoglobin", "rbc"]:
            risk_factors.append("Anemia risk")
    
    # Age-related risks
    if age:
        if age > 50:
            risk_factors.append("Age-related health risks")
        if age > 65:
            risk_level = "moderate"
    
    # Determine overall risk level
    if len(abnormalities) > 4:
        risk_level = "high"
    elif len(abnormalities) > 2:
        risk_level = "moderate"
    
    return {
        "risk_level": risk_level,
        "risk_factors": list(set(risk_factors)),
        "recommended_followup": get_followup_recommendations(risk_level)
    }

def get_followup_recommendations(risk_level: str) -> str:
    """Get follow-up recommendations based on risk level.
    
    Args:
        risk_level: Assessed risk level
        
    Returns:
        Follow-up recommendation text
    """
    recommendations = {
        "low": "Routine follow-up with healthcare provider recommended.",
        "moderate": "Schedule medical consultation within 2-4 weeks for further evaluation.",
        "high": "Seek medical attention promptly for comprehensive evaluation."
    }
    
    return recommendations.get(risk_level, "Consult with healthcare provider.")

def generate_recommendations(abnormalities: List[Dict], indicators: Dict[str, Any]) -> List[str]:
    """Generate specific recommendations based on test results.
    
    Args:
        abnormalities: List of abnormal values
        indicators: All extracted indicators
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    # General recommendations
    recommendations.append("Maintain a healthy lifestyle with balanced diet and regular exercise.")
    recommendations.append("Stay hydrated and get adequate sleep.")
    
    # Specific recommendations based on abnormalities
    for abnormality in abnormalities:
        indicator = abnormality["indicator"]
        
        if indicator in ["glucose"] and abnormality["status"] == "high":
            recommendations.append("Reduce sugar intake and monitor carbohydrate consumption.")
        elif indicator in ["cholesterol", "ldl"] and abnormality["status"] == "high":
            recommendations.append("Adopt heart-healthy diet low in saturated fats.")
        elif indicator in ["hemoglobin", "rbc"] and abnormality["status"] == "low":
            recommendations.append("Ensure adequate iron intake through diet or supplements.")
        elif indicator in ["alt", "ast"] and abnormality["status"] == "high":
            recommendations.append("Limit alcohol consumption and maintain healthy weight.")
    
    return recommendations

def generate_analysis_summary(abnormalities: List[Dict], risk_assessment: Dict[str, Any]) -> str:
    """Generate a concise summary of the analysis.
    
    Args:
        abnormalities: List of abnormal values
        risk_assessment: Risk assessment results
        
    Returns:
        Analysis summary text
    """
    if not abnormalities:
        return "All test results appear to be within normal ranges. Overall health status is good."
    
    abnormal_count = len(abnormalities)
    risk_level = risk_assessment.get("risk_level", "low")
    
    summary = f"Analysis found {abnormal_count} abnormal value(s). "
    summary += f"Overall risk level is assessed as {risk_level}. "
    
    if risk_level == "high":
        summary += "Medical consultation is recommended."
    elif risk_level == "moderate":
        summary += "Follow-up with healthcare provider advised."
    else:
        summary += "Monitor these values and maintain healthy lifestyle."
    
    return summary

def get_medical_reference_ranges() -> Dict[str, Any]:
    """Get reference ranges for common medical tests.
    
    Returns:
        Dictionary of reference ranges
    """
    return REFERENCE_RANGES

def validate_medical_content(text: str) -> Dict[str, Any]:
    """Validate if text appears to be from a medical report.
    
    Args:
        text: Text to validate
        
    Returns:
        Validation results
    """
    medical_keywords = [
        'blood test', 'laboratory', 'diagnostic', 'medical report',
        'hemoglobin', 'glucose', 'cholesterol', 'platelets',
        'reference range', 'patient', 'specimen'
    ]
    
    text_lower = text.lower()
    keyword_count = sum(1 for keyword in medical_keywords if keyword in text_lower)
    
    # Minimum text length check
    if len(text.strip()) < 50:
        return {
            "is_valid": False,
            "reason": "Text too short to be a valid medical report"
        }
    
    # Keyword threshold
    if keyword_count < 3:
        return {
            "is_valid": False,
            "reason": "Text doesn't contain enough medical terminology"
        }
    
    return {
        "is_valid": True,
        "confidence": min(keyword_count / 10, 1.0),
        "keywords_found": keyword_count
    }
