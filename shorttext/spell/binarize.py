
import re

from shorttext.generators.charbase.char2concatvec import SpellingToConcatCharVecEncoder

def hasnum(word):
    return len(re.findall('\\d', word)) > 0


