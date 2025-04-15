"""
Module for handling dental fixed prosthodontics code extraction.
"""

from .fixed_partial_denture_pontics import FixedPartialDenturePonticsServices
from .fixed_partial_denture_retainers_inlays_onlays import FixedPartialDentureRetainersInlaysOnlaysServices
from .fixed_partial_denture_retainers_crowns import FixedPartialDentureRetainersCrownsServices
from .other_fixed_partial_denture_services import OtherFixedPartialDentureServicesServices

__all__ = [
    'FixedPartialDenturePonticsServices',
    'FixedPartialDentureRetainersInlaysOnlaysServices',
    'FixedPartialDentureRetainersCrownsServices',
    'OtherFixedPartialDentureServicesServices'
] 