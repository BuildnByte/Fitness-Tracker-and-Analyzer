# workout_templates.py

WORKOUT_TEMPLATES = {
    "weight_loss": {
        3: [
            ("monday", "Full Body HIIT: 30 min high intensity"),
            ("tuesday", "Rest Day"),
            ("wednesday", "Cardio Core: 40 min steady state + 10 min core"),
            ("thursday", "Rest Day"),
            ("friday", "Metabolic Conditioning: 35 min circuit"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ],
        4: [
            ("monday", "Lower Body Focus: Strength & HIIT"),
            ("tuesday", "Rest Day"),
            ("wednesday", "Upper Body Focus: Strength & HIIT"),
            ("thursday", "Rest Day"),
            ("friday", "Full Body Circuit: 45 min"),
            ("saturday", "LISS Cardio: 45 min walking/cycling"),
            ("sunday", "Rest Day")
        ],
        5: [
            ("monday", "HIIT & Core: 45 min"),
            ("tuesday", "Upper Body Resistance: Moderate weight"),
            ("wednesday", "LISS Cardio: 60 min easy pace"),
            ("thursday", "Lower Body Resistance: Moderate weight"),
            ("friday", "Full Body Circuit: High intensity"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ]
    },
    "muscle_gain": {
        3: [
            ("monday", "Full Body Strength: Heavy compound lifts"),
            ("tuesday", "Rest Day"),
            ("wednesday", "Full Body Hypertrophy: Moderate weight, high rep"),
            ("thursday", "Rest Day"),
            ("friday", "Full Body Strength: Accessories and isolations"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ],
        4: [
            ("monday", "Upper Body Strength: Heavy focus"),
            ("tuesday", "Lower Body Strength: Heavy focus"),
            ("wednesday", "Rest Day"),
            ("thursday", "Upper Body Hypertrophy: Volume focus"),
            ("friday", "Lower Body Hypertrophy: Volume focus"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ],
        5: [
            ("monday", "Push: Chest, Shoulders, Triceps"),
            ("tuesday", "Pull: Back, Biceps, Rear Delts"),
            ("wednesday", "Legs: Quads, Hamstrings, Calves"),
            ("thursday", "Upper Body: Strength focus"),
            ("friday", "Lower Body: Strength focus"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ]
    },
    "endurance": {
        3: [
            ("monday", "Interval Running: 45 min"),
            ("tuesday", "Rest Day"),
            ("wednesday", "Tempo Run: 40 min moderate-hard pace"),
            ("thursday", "Rest Day"),
            ("friday", "Long Run: 60 min easy pace"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ],
        4: [
            ("monday", "Interval Running: 45 min"),
            ("tuesday", "Cross Training: Cycling or Swimming"),
            ("wednesday", "Rest Day"),
            ("thursday", "Tempo Run: 40 min moderate-hard pace"),
            ("friday", "Rest Day"),
            ("saturday", "Long Run: 75 min easy pace"),
            ("sunday", "Rest Day")
        ],
        5: [
            ("monday", "Interval Running: 45 min"),
            ("tuesday", "Core and Strength: Legs focus"),
            ("wednesday", "Long Run: 60 min easy pace"),
            ("thursday", "Mobility and Recovery Run: 25 min"),
            ("friday", "Tempo Run: 40 min"),
            ("saturday", "Cross Training: Cycling or Swimming"),
            ("sunday", "Rest Day")
        ]
    },
    "general": {
        3: [
            ("monday", "Full Body Mix: Strength & Cardio"),
            ("tuesday", "Rest Day"),
            ("wednesday", "Active Stability: Core & Mobility"),
            ("thursday", "Rest Day"),
            ("friday", "Endurance Walk/Jog: 45 min"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ],
        4: [
            ("monday", "Upper Body & Core: 45 min"),
            ("tuesday", "Steady Cardio: 45 min"),
            ("wednesday", "Rest Day"),
            ("thursday", "Lower Body & Mobility: 45 min"),
            ("friday", "Full Body Interval Mix: 30 min"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ],
        5: [
            ("monday", "Strength Focus: Major compound movements"),
            ("tuesday", "Cardio Intervals: 30-40 min"),
            ("wednesday", "Mobility & Core: Active recovery"),
            ("thursday", "Strength Focus: Accessory work"),
            ("friday", "Endurance Challenge: 45-60 min constant pace"),
            ("saturday", "Rest Day"),
            ("sunday", "Rest Day")
        ]
    }
}

GOAL_NORMALIZATION_MAP = {
    "weight_loss": "weight_loss",
    "lose_weight": "weight_loss",
    "lose weight": "weight_loss",
    "fat_loss": "weight_loss",
    "muscle_gain": "muscle_gain",
    "build_muscle": "muscle_gain",
    "build muscle": "muscle_gain",
    "hypertrophy": "muscle_gain",
    "endurance": "endurance",
    "stamina": "endurance",
    "general": "general",
    "health": "general",
    "maintain": "general"
}
