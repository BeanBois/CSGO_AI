#from https://gist.github.com/chris-hailstorm/4989643

import unicodedata

def asciify(data):
    """
SYNOPSIS
    Asciifies strings, lists and dicts, and nested versions of same

DESCRIPTION
    The JSON spec (http://www.ietf.org/rfc/rfc4627.txt) -- "JSON text SHALL
    be encoded in Unicode".  For apps that don't use unicode, this function
    walks through all levels of a JSON data structure and converts each item
    to ASCII. See http://stackoverflow.com/questions/956867/ for original.

    Can be used for any nesting of strings / lists / dicts, e.g. a list of
    dicts, a dict in which values are lists of strings etc. See LIMITATIONS.

PARAMETERS
    data        A string, unicode, list or dict, or nested versions of the
                same types.  Typically the string output from json.dumps()
                or the dict resulting from json.load() or json.loads().

RETURNS
    A Python dictionary with all keys and values converted to UTF-8.

USAGE
    There are several equivalent ways to use this function.

    (1) asciify string version of data structure before creating dict:

        s = json.dumps(x)
        d = json.loads(asciify(s))

    (2) create dict from string version of data structure, then asciify:

        s = json.dumps(x)
        d = json.loads(s)
        d = asciify(d)

    (3) asciify as the dict is being created via object hook:

        s = json.dumps(x)
        d = json.loads(s, object_hook=asciify)

    Asciifying the string first (approach (1) above) is probably the best
    approach since the input is a flat string and there's no possibility of
    the depth traversal stopping due to an unknown type.  See LIMITATIONS.


EXAMPLES
    >>> import json

    >>> s1 = 'ASCII string'
    >>> type(s1)
    <type 'str'>
    >>> s1 = asciify(s1)
    >>> type(s1)
    <type 'str'>

    >>> s2 = u'Unicode string'
    >>> type(s2)
    <type 'unicode'>
    >>> s2 = asciify(s2)
    >>> type(s2)
    <type 'str'>

    >>> s3 = 'Nestl'+unichr(0xe9)
    >>> print asciify(s3)
    Nestle

    >>> asciify(['a','b','c'])
    ['a', 'b', 'c']

    >>> asciify([u'a',u'b',u'c'])
    ['a', 'b', 'c']

    >>> asciify({'a':'aa','b':'bb','c':'cc'})
    {'a': 'aa', 'c': 'cc', 'b': 'bb'}

    >>> asciify({u'a':'aa','b':u'bb',u'c':u'cc'})
    {'a': 'aa', 'c': 'cc', 'b': 'bb'}

    >>> d = dict(a='a1',b='b2',c=dict(d='d3',e=['e4','e5','e6'],f=dict(g='g7')),h=[8,9,10])
    >>> print d
    {'a': 'a1', 'h': [8, 9, 10], 'c': {'e': ['e4', 'e5', 'e6'], 'd': 'd3', 'f': {'g': 'g7'}}, 'b': 'b2'}
    >>> print type(d)
    <type 'dict'>

    >>> asciistr = json.dumps(d)
    >>> print asciistr
    {"a": "a1", "h": [8, 9, 10], "c": {"e": ["e4", "e5", "e6"], "d": "d3", "f": {"g": "g7"}}, "b": "b2"}
    >>> print type(asciistr)
    <type 'str'>

    >>> unidict = json.loads(asciistr)
    >>> print unidict
    {u'a': u'a1', u'h': [8, 9, 10], u'c': {u'e': [u'e4', u'e5', u'e6'], u'd': u'd3', u'f': {u'g': u'g7'}}, u'b': u'b2'}
    >>> print type(unidict)
    <type 'dict'>
    >>> unidict == d
    True

    >>> asciidict1 = asciify(unidict)
    >>> print asciidict1
    {'a': 'a1', 'h': [8, 9, 10], 'c': {'e': ['e4', 'e5', 'e6'], 'd': 'd3', 'f': {'g': 'g7'}}, 'b': 'b2'}
    >>> print type(asciidict1)
    <type 'dict'>
    >>> asciidict1 == d
    True

    >>> asciidict2 = json.loads(asciistr, object_hook=asciify)
    >>> print asciidict2
    {'a': 'a1', 'h': [8, 9, 10], 'c': {'e': ['e4', 'e5', 'e6'], 'd': 'd3', 'f': {'g': 'g7'}}, 'b': 'b2'}
    >>> print type(asciidict2)
    <type 'dict'>
    >>> asciidict2 == d
    True

LIMITATIONS
    For a multi-layered data structure (dict of lists, list of strings etc.)
    depth traversal of the data structure stops when the element encountered
    is not a string, unicode, list or dict.  For example, in this dict:

        > d = {'a': { 'b': [1, 2, set(u'x', u'y'] ), 'c': u'z' } }

    ... the u'x' and u'y' items are contained within a set, and therefore
    would not be asciified, while u'z' is contained in a dict and would be
    asciified since the breadth traversal of the structure continues.

    A future @@todo could be to throw an error if a non-traversable input
    is used, or have additional parameter that can allow the non-traversable
    input to be used even though the result is a partial discard of data.

    """
    ##
    ## embedded functions
    ##
    ## see http://stackoverflow.com/a/517974
    def _remove_accents(data):
        """
    Changes accented letters to non-accented approximation, like Nestle

        """
        return unicodedata.normalize('NFKD', data).encode('ascii', 'ignore')
    ##
    def _asciify_list(data):
        """ Ascii-fies list values """
        ret = []
        for item in data:
            if isinstance(item, str):
                item = _remove_accents(item)
                item = item.encode('utf-8')
            elif isinstance(item, list):
                item = _asciify_list(item)
            elif isinstance(item, dict):
                item = _asciify_dict(item)
            ret.append(item)
        return ret
    #
    def _asciify_dict(data):
        """ Ascii-fies dict keys and values """
        ret = {}
        for key, value in iter(data.items()):
            if isinstance(key, str):
                key = _remove_accents(key)
                # key = key.encode('utf-8')
            ## note new if
            if isinstance(value, str):
                value = _remove_accents(value)
                # value = value.encode('utf-8')
            elif isinstance(value, list):
                value = _asciify_list(value)
            elif isinstance(value, dict):
                value = _asciify_dict(value)
            ret[key] = value
        return str(ret)
    ##
    ## main function
    if isinstance(data, list):
        return _asciify_list(data).encode('utf-8')
    elif isinstance(data, dict):
        return _asciify_dict(data).encode('utf-8')
    # elif isinstance(data, str):
    #     data = _remove_accents(data)
    #     return data.encode('utf-8')
    elif isinstance(data,int):
        return str(data).encode('utf-8')
    elif isinstance(data, str):
        return _remove_accents(data)
        # return data
    else:
        raise TypeError('Input must be dict, list, str or unicode')

if __name__ == "__main__":
    data= {"provider": {"name": "Counter-Strike: Global Offensive", "appid": 730, "version": 13855, "steamid": "76561198111212182", "timestamp": 1677927517 \
        }, 
        "map": {"mode": "casual", "name": "de_dust2", "phase": "live", "round": 5, 
            "team_ct": {"score": 0, "consecutive_round_losses": 5, "timeouts_remaining": 1, "matches_won_this_series": 0
            }, 
            "team_t": {"score": 4, "consecutive_round_losses": 0, "timeouts_remaining": 1, "matches_won_this_series": 0
            }, 
            "num_matches_to_win_series": 0, "current_spectators": 0, "souvenirs_total": 0
        }, 
        "round": {"phase": "over", "win_team": "T"
        }, 
        "player": {"steamid": "76561198111212182", "name": "beebeepop", "observer_slot": 4, "team": "T", "activity": "playing", 
            "state": {"health": 100, "armor": 100, "helmet": True, "flashed": 0, "smoked": 0, "burning": 0, "money": 10000, "round_kills": 3, "round_killhs": 1, "equip_value": 4800
            }, 
            "weapons": {"weapon_0": {"name": "weapon_knife_t", "paintkit": "default", "type": "Knife", "state": "active"
                }, 
                "weapon_1": {"name": "weapon_deagle", "paintkit": "default", "type": "Pistol", "ammo_clip": 6, "ammo_clip_max": 7, "ammo_reserve": 35, "state": "holstered"
                }, 
                "weapon_2": {"name": "weapon_m4a1", "paintkit": "default", "type": "Rifle", "ammo_clip": 0, "ammo_clip_max": 30, "ammo_reserve": 90, "state": "holstered"
                }
            }, 
        "match_stats": {"kills": 12, "assists": 0, "deaths": 0, "mvps": 5, "score": 35
            }
        }, "auth": {"token": "CCWJu64ZV3JHDT8hZc"
        }
    }
    asd = asciify(data)
    print(asd)