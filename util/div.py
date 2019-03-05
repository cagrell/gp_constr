# Div util functions

# Time 
def formattime(sec):
    """ Format time in seconds to h:m:s depending on sec """
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)

    if h > 0: return '{} hours {} minutes {} seconds'.format('%.0f'%h, '%.0f'%m, '%.0f'%s)
    if m > 0: return '{} minutes {} seconds'.format('%.0f'%m, '%.1f'%s)
    return '{} seconds'.format('%.3f'%s)

# Other random stuff
def len_none(x):
    return 0 if x is None else len(x)
