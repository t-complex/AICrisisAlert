CRISIS_LABELS = [
    "requests_or_urgent_needs",
    "infrastructure_and_utility_damage",
    "injured_or_dead_people",
    "rescue_volunteering_or_donation_effort",
    "other_relevant_information",
    "not_humanitarian"
]

CRISIS_TYPE_MAPPING = {
    "requests_or_urgent_needs": {"impact_multiplier": 1.5, "response_time": 1, "severity": "critical"},
    "infrastructure_and_utility_damage": {"impact_multiplier": 1.2, "response_time": 2, "severity": "high"},
    "injured_or_dead_people": {"impact_multiplier": 2.0, "response_time": 1, "severity": "critical"},
    "rescue_volunteering_or_donation_effort": {"impact_multiplier": 1.0, "response_time": 3, "severity": "medium"},
    "other_relevant_information": {"impact_multiplier": 0.8, "response_time": 4, "severity": "low"},
    "not_humanitarian": {"impact_multiplier": 0.5, "response_time": 5, "severity": "none"}
} 