"""
Module for handling dental oral and maxillofacial surgery code extraction.
"""

from .extractions import ExtractionsServices
from .other_surgical_procedures import activate_other_surgical_procedures
from .alveoloplasty import alveoloplasty_service
from .vestibuloplasty import vestibuloplasty_service
from .excision_soft_tissue import excision_soft_tissue_service
from .excision_intra_osseous import excision_intra_osseous_service
from .excision_bone_tissue import excision_bone_tissue_service
from .surgical_incision import surgical_incision_service
from .closed_fractures import closed_fractures_service
from .open_fractures import open_fractures_service
from .tmj_dysfunctions import tmj_dysfunctions_service
from .traumatic_wounds import traumatic_wounds_service
from .complicated_suturing import complicated_suturing_service
from .other_repair_procedures import other_repair_procedures_service

# Create instance of extractions service
extractions_service = ExtractionsServices()

# Create compatible function interfaces for class-based modules
def activate_extractions(scenario):
    """Delegate to the extractions service instance."""
    return extractions_service.activate_extractions(scenario)

def activate_alveoloplasty(scenario):
    """Delegate to the alveoloplasty service instance."""
    return alveoloplasty_service.activate_alveoloplasty(scenario)

def activate_vestibuloplasty(scenario):
    """Delegate to the vestibuloplasty service instance."""
    return vestibuloplasty_service.activate_vestibuloplasty(scenario)

def activate_excision_soft_tissue(scenario):
    """Delegate to the excision soft tissue service instance."""
    return excision_soft_tissue_service.activate_excision_soft_tissue(scenario)

def activate_excision_intra_osseous(scenario):
    """Delegate to the excision intra osseous service instance."""
    return excision_intra_osseous_service.activate_excision_intra_osseous(scenario)

def activate_excision_bone_tissue(scenario):
    """Delegate to the excision bone tissue service instance."""
    return excision_bone_tissue_service.activate_excision_bone_tissue(scenario)

def activate_surgical_incision(scenario):
    """Delegate to the surgical incision service instance."""
    return surgical_incision_service.activate_surgical_incision(scenario)

def activate_closed_fractures(scenario):
    """Delegate to the closed fractures service instance."""
    return closed_fractures_service.activate_closed_fractures(scenario)

def activate_open_fractures(scenario):
    """Delegate to the open fractures service instance."""
    return open_fractures_service.activate_open_fractures(scenario)

def activate_tmj_dysfunctions(scenario):
    """Delegate to the TMJ dysfunctions service instance."""
    return tmj_dysfunctions_service.activate_tmj_dysfunctions(scenario)

def activate_traumatic_wounds(scenario):
    """Delegate to the traumatic wounds service instance."""
    return traumatic_wounds_service.activate_traumatic_wounds(scenario)

def activate_complicated_suturing(scenario):
    """Delegate to the complicated suturing service instance."""
    return complicated_suturing_service.activate_complicated_suturing(scenario)

def activate_other_repair_procedures(scenario):
    """Delegate to the other repair procedures service instance."""
    return other_repair_procedures_service.activate_other_repair_procedures(scenario)

__all__ = [
    'activate_extractions',
    'activate_other_surgical_procedures',
    'activate_alveoloplasty',
    'activate_vestibuloplasty',
    'activate_excision_soft_tissue',
    'activate_excision_intra_osseous',
    'activate_excision_bone_tissue',
    'activate_surgical_incision',
    'activate_closed_fractures',
    'activate_open_fractures',
    'activate_tmj_dysfunctions',
    'activate_traumatic_wounds',
    'activate_complicated_suturing',
    'activate_other_repair_procedures'
] 