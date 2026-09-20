def generate_recommendation(data):

    soil_moisture = data["soil_moisture"]
    ph = data["ph"]
    nitrogen = data["nitrogen"]
    phosphorus = data["phosphorus"]
    potassium = data["potassium"]
    temperature = data["temperature"]
    humidity = data["humidity"]

    recommendations = []

    # Irrigation
    if soil_moisture < 30:
        recommendations.append({
            "action": "Increase irrigation",
            "priority": "high"
        })

    # Soil pH
    if ph < 5.5:
        recommendations.append({
            "action": "Apply agricultural lime",
            "priority": "medium"
        })

    elif ph > 7.5:
        recommendations.append({
            "action": "Reduce soil alkalinity",
            "priority": "medium"
        })

    # Nitrogen
    if nitrogen < 40:
        recommendations.append({
            "action": "Add nitrogen fertilizer",
            "priority": "medium"
        })

    # Phosphorus
    if phosphorus < 20:
        recommendations.append({
            "action": "Add phosphorus fertilizer",
            "priority": "medium"
        })

    # Potassium
    if potassium < 40:
        recommendations.append({
            "action": "Add potassium fertilizer",
            "priority": "medium"
        })

    # Heat Stress
    if temperature > 35:
        recommendations.append({
            "action": "Protect crops from heat stress",
            "priority": "high"
        })

    if len(recommendations) == 0:
        recommendations.append({
            "action": "No action required",
            "priority": "low"
        })

    return recommendations
