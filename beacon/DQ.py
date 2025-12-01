# Data Quality
# DQ short names
def dq_short_names():
    """
    Return the mapping of data quality (DQ) short names to their bit positions.

    Returns:
        dict: Dictionary mapping DQ category names to integer bit positions.
            Example:
                {
                    'BURST_CAT1': 0,
                    'BURST_CAT2': 1,
                    ...
                }
    """
    return {
        'BURST_CAT1': 0,
        'BURST_CAT2': 1,
        'BURST_CAT3': 2,
        'CBC_CAT1': 3,
        'CBC_CAT2': 4,
        'CBC_CAT3': 5,
        'INSPIRAL': 6
    }
    
# Convert decimal DQs into bits
def dq_dec2bits(decimal_dq, length=7):
    """
    Convert a decimal-encoded DQ mask into a list of bits.

    Args:
        decimal_dq (int): Integer DQ value to decode.
        length (int, optional): Number of bits to extract. Defaults to 7.

    Returns:
        list of int: List of bits (0 or 1), least significant bit first.
    """
    return [(decimal_dq >> i) & 1 for i in range(length)]

# Read requested DQ level
def dqlev(dq, level='BURST_CAT2'):
    """
    Read specified data quality level(s) from a decimal DQ mask.

    Args:
        dq (int): Decimal-encoded DQ mask.
        level (str, optional): DQ level name or 'all' to decode all levels.
            Defaults to 'BURST_CAT2'.

    Returns:
        dict: Dictionary mapping DQ level(s) to their bit value(s).
            If 'all', returns all levels; otherwise, returns only the requested level.
    """
    if level == 'all':
        ret = dq_dec2bits(dq)
        names = list(dq_short_names().keys())
        return dict(zip(names, ret))
    else:
        bitpos = dq_short_names()[level]
        return {level: (dq >> bitpos) & 1}  # return as dict
