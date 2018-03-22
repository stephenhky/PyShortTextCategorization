
import re

def hasnum(word):
    return len(re.findall('\\d', word)) > 0

