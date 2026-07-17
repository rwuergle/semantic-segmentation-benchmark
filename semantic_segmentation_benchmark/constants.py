PROJECT_CLASSIFIED_MAP = {
    0:  "Other",
    2:	"Ground",
    3:	"vegetation",
    6:	"Building roofs",
    21:	"Cars",
    22:	"Building facades",
    26:	"Roof structures",
}

POINTLY_CLASSIFIED_MAP = {
    0:  "Other",
    2:	"Ground",
    3:	"vegetation",
    6:	"Building",
    21:	"Cars",
}

# Default Reclassification Maps

SITN_REMAP = {
    "default": 0, 
    2:  2, 
    18: 2, 
    31: 2, 
    3:  3, 
    4:  3, 
    5:  3, 
    6:  6, 
    21: 21, 
    22: 22, 
    26: 26
}

CLASSICAL_REMAP = {
    "default":  "keep", 
    29: 0
}

FLAI_REMAP = {
    "default": 0,
    2:  2,
    3:  3, 
    6:  6, 
    21: 21, 
    22: 22, 
    20: 26
}

POINTLY_REMAP = {
    "default": 0,
    20: 21,
    5:  3,
    2:  2,
    6:  6,
}

SITN_POINTLY_REMAP = {
    "default": 0, 
    2:  2, 
    18: 2, 
    31: 2, 
    3:  3, 
    4:  3, 
    5:  3, 
    6:  6, 
    21: 21, 
    22: 6, 
    26: 6
}

SITN_MINKUNET = {
    "default": "keep",
    29: 0
}

SITN_FIRST_ITERATION_DL_REMAP = {
    "default": 0,
    1:  2,
    12: 26,
    10: 22,
    9:  21,
    4:  6,
    3:  3
}
