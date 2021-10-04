# start at -1 so first id is 0
_curr_indiv_id = -1


def get_next_indiv_id():
    global _curr_indiv_id
    _curr_indiv_id += 1
    return _curr_indiv_id
