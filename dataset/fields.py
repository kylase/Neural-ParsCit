def cap_feature(string):
    string = string[0]
    if string.lower() == string:
        return 0
    if string.upper() == string:
        return 1
    if string[0].upper() == string[0]:
        return 2
    return 3
