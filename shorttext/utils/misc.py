
import sys

def textfile_generator(textfile, linebreak=True, encoding=None):
    """ Return a generator that reads lines in a text file.

    :param textfile: file object of a text file
    :param linebreak: whether to return a line break at the end of each line (Default: True)
    :param encoding: encoding of the text file (Default: None)
    :return: a generator that reads lines in a text file
    :type textfile: file
    :type linebreak: bool
    :type encoding: str
    :rtype: generator
    """
    for t in textfile:
        if len(t) > 0:
            if encoding is None:
                yield t.strip() + ('\n' if linebreak else '')
            else:
                yield t.decode(encoding).strip() + ('\n' if linebreak else '')

def text_input(prompt):
    """ Get user response from keyboard in stdin.

    :param prompt: prompt
    :return: user-typed value
    :type prompt: str
    :rtype: str
    """
    if sys.version[0] == '2':
        return raw_input(prompt)
    elif sys.version[0] == '3':
        return input(prompt)
    else:
        return None