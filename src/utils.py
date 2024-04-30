from pathlib import Path


def file_sort_order(file: Path) -> str:
    """Get the sort order of a file. The sort order is the number at the end of the filename.

    Args:
        file (Path): The file to get the sort order from.

    Returns:
        str: The sort order of the file."""
    return int(file.stem.split("_")[-2])


def levenshtein_distance(string1, string2):
    """Calculate the Levenshtein distance between two strings.

    Args:
        string1 (str): The first string.
        string2 (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings."""
    if len(string1) < len(string2):
        return levenshtein_distance(string2, string1)

    if len(string2) == 0:
        return len(string1)

    # dynamic programming table
    previous_row = range(len(string2) + 1)
    for i, c1 in enumerate(string1):
        current_row = [i + 1]
        for j, c2 in enumerate(string2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
