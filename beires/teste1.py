def solution(seconds):
    if seconds == 0:
        return "now"
    
    # Calculate units
    years = seconds // 31536000
    seconds = seconds % 31536000
    
    days = seconds // 86400
    seconds = seconds % 86400
    
    hours = seconds // 3600
    seconds = seconds % 3600
    
    minutes = seconds // 60
    seconds = seconds % 60
    
    # Build result
    result = []
    
    if years > 0:
        if years == 1:
            result.append("1 year")
        else:
            result.append(str(years) + " years")
    
    if days > 0:
        if days == 1:
            result.append("1 day")
        else:
            result.append(str(days) + " days")
    
    if hours > 0:
        if hours == 1:
            result.append("1 hour")
        else:
            result.append(str(hours) + " hours")
    
    if minutes > 0:
        if minutes == 1:
            result.append("1 minute")
        else:
            result.append(str(minutes) + " minutes")
    
    if seconds > 0:
        if seconds == 1:
            result.append("1 second")
        else:
            result.append(str(seconds) + " seconds")
    
    # Format output
    if len(result) == 1:
        return result[0]
    elif len(result) == 2:
        return result[0] + " and " + result[1]
    else:
        return ", ".join(result[:-1]) + " and " + result[-1]

if __name__ == "__main__":
    result = solution(62)
    print(result)