def data_of(result):
    """
    Safe accessor for supabase-py results.

    `.maybe_single().execute()` returns **None** (not an object with data=None)
    when no row matches, so reading `.data` directly raises AttributeError.
    """
    return None if result is None else result.data
