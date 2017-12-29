

def textfile_generator(textfile, linebreak=True):
    """ Return a generator that reads lines in a text file.

    :param textfile: file object of a text file
    :param linebreak: whether to return a line break at the end of each line (Default: True)
    :return: a generator that reads lines in a text file
    :type textfile: file
    :type linebreak: bool
    :rtype: generator
    """
    for t in textfile:
        if len(t) > 0:
            yield t.strip() + ('\n' if linebreak else '')

