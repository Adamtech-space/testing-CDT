"""
Module for handling dental periodontal code extraction.
"""

from .surgical_services import SurgicalServicesPeriodontics, surgical_services
from .non_surgical_services import NonSurgicalServicesPeriodontics, non_surgical_services
from .other_periodontal_services import OtherPeriodontalServices, other_periodontal_services

# Function exports
def activate_surgical_services(scenario):
    return surgical_services.activate_surgical_services(scenario)

def activate_non_surgical_services(scenario):
    return non_surgical_services.activate_non_surgical_services(scenario)

def activate_other_periodontal_services(scenario):
    return other_periodontal_services.activate_other_periodontal_services(scenario)

__all__ = [
    'SurgicalServicesPeriodontics',
    'NonSurgicalServicesPeriodontics',
    'OtherPeriodontalServices',
    'activate_surgical_services',
    'activate_non_surgical_services',
    'activate_other_periodontal_services',
    'surgical_services',
    'non_surgical_services',
    'other_periodontal_services'
] 