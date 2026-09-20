from disease_info import disease_info

def get_recommendation(disease):

    info = disease_info[disease]

    return {
        "severity": info["severity"],
        "action": info["action"]
    }