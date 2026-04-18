"""
Fuzzy Logic Module for mapping raw health averages to interpretable soft scores.
"""

def trimf(x, a, b, c):
    """Triangular membership function"""
    if x <= a or x >= c:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if (b - a) != 0 else 1.0
    elif b < x < c:
        return (c - x) / (c - b) if (c - b) != 0 else 1.0
    return 0.0

def trapmf(x, a, b, c, d):
    """Trapezoidal membership function"""
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a) if (b - a) != 0 else 1.0
    elif b < x <= c:
        return 1.0
    elif c < x < d:
        return (d - x) / (d - c) if (d - c) != 0 else 1.0
    return 0.0

def fuzzy_or(*args):
    """Fuzzy OR (maximum)"""
    return max(args) if args else 0.0

def fuzzy_and(*args):
    """Fuzzy AND (minimum)"""
    return min(args) if args else 0.0

def fuzzify_sleep_hours(x):
    return {
        'insufficient': trapmf(x, 0, 0, 5.5, 7.0),
        'adequate': trimf(x, 6.0, 7.5, 8.5),
        'optimal': trapmf(x, 7.5, 8.5, 24, 24)
    }

def fuzzify_sleep_quality(x):
    return {
        'poor': trapmf(x, 1, 1, 2, 3),
        'fair': trimf(x, 2, 3, 4),
        'good': trapmf(x, 3, 4, 5, 5)
    }

def fuzzify_workout_frequency(x):
    return {
        'low': trapmf(x, 0, 0, 1.5, 3),
        'medium': trimf(x, 2, 4, 5.5),
        'high': trapmf(x, 4.5, 6, 7, 7)
    }

def fuzzify_workout_duration(x):
    return {
        'short': trapmf(x, 0, 0, 20, 40),
        'medium': trimf(x, 30, 45, 60),
        'long': trapmf(x, 50, 75, 300, 300)
    }

def fuzzify_water(x):
    return {
        'low': trapmf(x, 0, 0, 1.5, 2.5),
        'adequate': trimf(x, 1.8, 3.0, 4.0),
        'optimal': trapmf(x, 3.0, 4.5, 10, 10)
    }

def fuzzify_junk_food(x):
    return {
        'low': trapmf(x, 0, 0, 1, 2),
        'medium': trimf(x, 1, 2, 3),
        'high': trapmf(x, 2, 3, 4, 4)
    }

def evaluate_rules(stats):
    """
    Evaluates fuzzy rules to produce 4 continuous scores (0 to 1).
    Uses a Sugeno-style zero-order inference mechanism (weighted average of output levels).
    """
    slp_h = fuzzify_sleep_hours(stats.get('sleep_hours', 7.0))
    slp_q = fuzzify_sleep_quality(stats.get('sleep_quality', 3.0))
    freq = fuzzify_workout_frequency(stats.get('workout_freq', 3.0))
    dur = fuzzify_workout_duration(stats.get('workout_dur', 30.0))
    water = fuzzify_water(stats.get('water_l', 2.0))
    junk = fuzzify_junk_food(stats.get('junk_food', 2.0))
    
    # Optional Calorie specific metrics if needed (e.g. tracking adherence)
    # The calorie_balance feature itself can also guide nutrition adherence softly.
    cal_bal = stats.get('calorie_balance', 0)
    # For adherence, if magnitude of balance is huge without intent, that is poor adherence.
    # For now, we lean primarily on junk food and water for general adherence as placeholders.
    cal_adherence = trapmf(abs(cal_bal), 0, 100, 300, 800) # Closer to 0 deviation = higher adherence to plan
    
    # Output 1: RECOVERY SCORE (0 to 1)
    # Rules: 
    # If sleep is optimal AND quality is good -> 1.0
    # If sleep is adequate AND quality is fair -> 0.7
    # If sleep is insufficient OR quality is poor -> 0.3
    # If freq is high AND duration is long -> limits recovery (-0.2 modifier implicitly logic)
    r1 = fuzzy_and(slp_h['optimal'], slp_q['good'])       # output = 1.0
    r2 = fuzzy_and(slp_h['adequate'], slp_q['fair'])      # output = 0.7
    r3 = fuzzy_or(slp_h['insufficient'], slp_q['poor'])   # output = 0.3
    
    # Aggregation (Weighted Average)
    weight_sum = r1 + r2 + r3
    if weight_sum > 0:
        recovery = (r1 * 1.0 + r2 * 0.7 + r3 * 0.3) / weight_sum
    else:
        recovery = 0.5
        
    # Penalty for overtraining
    overtraining = fuzzy_and(freq['high'], dur['long'])
    recovery = max(0.0, recovery - (overtraining * 0.3))

    # Output 2: NUTRITION ADHERENCE SCORE (0 to 1)
    # Rules:
    # If junk is low AND water is optimal AND cal_adherence is 1 -> 1.0
    # If junk is medium AND water is adequate -> 0.6
    # If junk is high OR water is low -> 0.2
    n1 = fuzzy_and(junk['low'], water['optimal'], cal_adherence) # output = 1.0
    n2 = fuzzy_and(junk['medium'], water['adequate'])            # output = 0.6
    n3 = fuzzy_or(junk['high'], water['low'])                    # output = 0.2
    
    n_sum = n1 + n2 + n3
    if n_sum > 0:
        nutrition = (n1 * 1.0 + n2 * 0.6 + n3 * 0.2) / n_sum
    else:
        nutrition = 0.5

    # Output 3: TRAINING LOAD SCORE (0 to 1)
    # Rules:
    # If freq high AND dur long -> 1.0
    # If freq medium AND dur medium -> 0.5
    # If freq low OR dur short -> 0.1
    t1 = fuzzy_and(freq['high'], dur['long'])        # output = 1.0
    t2 = fuzzy_and(freq['medium'], dur['medium'])    # output = 0.5
    t3 = fuzzy_or(freq['low'], dur['short'])         # output = 0.1
    
    t_sum = t1 + t2 + t3
    if t_sum > 0:
        training_load = (t1 * 1.0 + t2 * 0.5 + t3 * 0.1) / t_sum
    else:
        training_load = 0.3

    # Output 4: HYDRATION SCORE (0 to 1)
    # Rules:
    # If water optimal -> 1.0
    # If water adequate -> 0.6
    # If water low -> 0.2
    h_sum = water['optimal'] + water['adequate'] + water['low']
    if h_sum > 0:
        hydration = (water['optimal'] * 1.0 + water['adequate'] * 0.6 + water['low'] * 0.2) / h_sum
    else:
        hydration = 0.5

    return {
        "recovery_score": round(recovery, 2),
        "nutrition_adherence": round(nutrition, 2),
        "training_load": round(training_load, 2),
        "hydration_score": round(hydration, 2)
    }

def get_fuzzy_scores_for_user(weekly_stats, diet_weekly_stats, health_records):
    """
    Wraps the parameter mapping to feed evaluate_rules()
    `weekly_stats` comes from get_weekly_stats()
    `diet_weekly_stats` from get_weekly_averages_for_diet_plan()
    """
    # Try to find average calorie balance from health records
    avg_cal_balance = 0
    if health_records:
        bal_sum = 0
        bal_count = 0
        for rec in health_records.values():
            if 'calorie_balance' in rec:
                bal_sum += rec['calorie_balance']
                bal_count += 1
        if bal_count > 0:
            avg_cal_balance = bal_sum / bal_count
            
    stats_dict = {
        'sleep_hours': weekly_stats.get('avg_sleep', 7.0),
        'sleep_quality': weekly_stats.get('avg_sleep_quality', 3.0),
        'workout_freq': weekly_stats.get('active_days', 3),
        'workout_dur': weekly_stats.get('avg_workout_duration', 30),
        'water_l': weekly_stats.get('avg_water', 2.0),
        'junk_food': weekly_stats.get('avg_junk_food', 2.0),
        'calorie_balance': avg_cal_balance
    }
    
    return evaluate_rules(stats_dict)
