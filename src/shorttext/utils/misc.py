
from typing import Generator
from io import TextIOWrapper



def textfile_generator(
        textfile: TextIOWrapper,
        linebreak: bool=True,
        encoding: bool=None
) -> Generator[str, None, None]:
    """Generator that yields lines from a text file.

    Args:
        textfile: File object to read lines from.
        linebreak: Whether to include line break at end of each line. Default: True.
        encoding: Encoding of the text file. Default: None.

    Yields:
        Lines from the text file, stripped of whitespace.
    """
    for t in textfile:
        if len(t) > 0:
            if encoding is None:
                yield t.strip() + ('\n' if linebreak else '')
            else:
                yield t.decode(encoding).strip() + ('\n' if linebreak else '')


class SinglePoolExecutor:
    """Wrapper for Python map function.

    Provides an interface similar to concurrent.futures.Executor.map
    but using a synchronous map implementation.
    """

    def map(self, func, *iterables):
        """Apply function to iterables element-wise.

        Args:
            func: Function to apply to each element.
            iterables: One or more iterables to process.

        Returns:
            An iterator yielding the results.
        """
        return map(func, *iterables)
        return map(func, *iterables)
