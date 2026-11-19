class PermanentJobError(Exception):
    """A failure that retrying cannot fix (bad PDF, scanned PDF, deleted document)."""