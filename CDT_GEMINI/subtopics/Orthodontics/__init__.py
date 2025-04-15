"""
Module for handling dental orthodontic code extraction.
"""

from .limited_orthodontic_treatment import LimitedOrthodonticTreatment, limited_orthodontic_treatment
from .comprehensive_orthodontic_treatment import ComprehensiveOrthodonticTreatment, comprehensive_orthodontic_treatment
from .minor_treatment_harmful_habits import MinorTreatmentHarmfulHabits, minor_treatment_harmful_habits
from .other_orthodontic_services import OtherOrthodonticServices, other_orthodontic_services

# Function exports
def activate_limited_orthodontic_treatment(scenario):
    return limited_orthodontic_treatment.activate_limited_orthodontic_treatment(scenario)

def activate_comprehensive_orthodontic_treatment(scenario):
    return comprehensive_orthodontic_treatment.activate_comprehensive_orthodontic_treatment(scenario)

def activate_minor_treatment_harmful_habits(scenario):
    return minor_treatment_harmful_habits.activate_minor_treatment_harmful_habits(scenario)

def activate_other_orthodontic_services(scenario):
    return other_orthodontic_services.activate_other_orthodontic_services(scenario)

__all__ = [
    'LimitedOrthodonticTreatment',
    'ComprehensiveOrthodonticTreatment',
    'MinorTreatmentHarmfulHabits',
    'OtherOrthodonticServices',
    'activate_limited_orthodontic_treatment',
    'activate_comprehensive_orthodontic_treatment',
    'activate_minor_treatment_harmful_habits',
    'activate_other_orthodontic_services',
    'limited_orthodontic_treatment',
    'comprehensive_orthodontic_treatment',
    'minor_treatment_harmful_habits',
    'other_orthodontic_services'
] 