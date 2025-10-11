def solution(pin):
    adjacent = {
        '0': ['0', '8'],
        '1': ['1', '2', '4'],
        '2': ['2', '1', '3', '5'],
        '3': ['3', '2', '6'],
        '4': ['4', '1', '5', '7'],
        '5': ['5', '2', '4', '6', '8'],
        '6': ['6', '3', '5', '9'],
        '7': ['7', '4', '8'],
        '8': ['8', '5', '7', '9', '0'],
        '9': ['9', '6', '8']
    }
    
    if not pin:
        return []
    
    result = ['']
    
    for digit in pin:
        new_result = []
        for combination in result:
            for adjacent_digit in adjacent[digit]:
                new_result.append(combination + adjacent_digit)
        result = new_result
    
    result = list(set(result))
    result = [str(x) for x in sorted([int(x) for x in result])]
    
    return result

if __name__ == "__main__":
    result = solution("8")
    print(result)