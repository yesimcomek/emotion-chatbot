"""German emotion lexicon for the rule-based classifier.

Each word maps to a weight: 2 for words that clearly signal the emotion,
1 for weak or ambiguous ones ("gut", "schön", "krass").
Inflection is handled in ``rules.py`` by allowing common suffixes, so list base forms only.

Emotions follow the BRIGHTER benchmark label set. Affection words ("liebe", "mag")
are folded into ``joy`` because the benchmark has no separate ``love`` label.
"""

from __future__ import annotations

LEXICON: dict[str, dict[str, int]] = {
    "anger": {
        "wütend": 2, "wut": 2, "sauer": 2, "stinksauer": 2, "frustriert": 2, "genervt": 2,
        "verärgert": 2, "ärgerlich": 2, "ärger": 2, "zornig": 2, "empört": 2, "aggressiv": 2,
        "hasse": 2, "hass": 2, "nervt": 2, "böse": 1, "frechheit": 2, "unerträglich": 2,
        "dreist": 2, "verdammt": 2, "scheiße": 2, "scheiß": 2, "kotzt": 2, "idiot": 2, "unverschämt": 2,
        "nervig": 2, "skandal": 1, "dumm": 1, "lüge": 1, "hetze": 1,
    },
    "disgust": {
        "ekelhaft": 2, "ekel": 2, "eklig": 2, "widerlich": 2, "widerwärtig": 2,
        "abstoßend": 2, "angewidert": 2, "igitt": 2, "pfui": 2, "verachte": 2,
        "abartig": 2, "scheußlich": 1, "schande": 2, "pervers": 2, "abscheulich": 2,
        "erbärmlich": 2, "menschenverachtend": 2, "verachtung": 2, "peinlich": 1, "armselig": 1,
        "krank": 1, "mies": 1,
    },
    "fear": {
        "ängstlich": 2, "angst": 2, "besorgt": 2, "nervös": 2, "beunruhigt": 2,
        "erschrocken": 2, "panik": 2, "furcht": 2, "fürchte": 2, "verängstigt": 2,
        "sorge": 2, "bange": 2, "unsicher": 1, "befürchte": 2, "befürchtung": 2, "verunsichert": 2,
        "gefahr": 1, "gefährlich": 1, "bedrohung": 1, "gewalt": 1, "horror": 1, "schlimm": 1,
    },
    "joy": {
        "glücklich": 2, "froh": 2, "fröhlich": 2, "freude": 2, "freue": 2, "begeistert": 2,
        "wunderbar": 2, "großartig": 2, "fantastisch": 2, "ausgezeichnet": 2, "toll": 2,
        "super": 2, "prima": 2, "klasse": 2, "genial": 2, "herrlich": 2, "zufrieden": 2,
        "erleichtert": 2, "stolz": 2, "lache": 2, "liebe": 2, "lieben": 2, "verliebt": 2,
        "gut": 1, "schön": 1, "mag": 1, "gern": 1, "gerne": 1, "glückwunsch": 2, "herzlichen": 1,
        "cool": 2, "glück": 2, "freut": 2, "geil": 2, "danke": 1, "dank": 1, "nett": 1, "endlich": 1,
        "lustig": 1, "hammer": 1, "gelungen": 1, "gewonnen": 1, "gefällt": 2, "gefallen": 1, "mega": 1,
        "sonne": 1, "urlaub": 1, "gesund": 1,
    },
    "sadness": {
        "traurig": 2, "unglücklich": 2, "deprimiert": 2, "niedergeschlagen": 2, "elend": 2,
        "einsam": 2, "verzweifelt": 2, "bedrückt": 2, "betrübt": 2, "trauer": 2, "weine": 2,
        "enttäuscht": 2, "schlecht": 1, "schade": 2, "leid": 2, "depression": 2, "depressiv": 2,
        "schmerz": 2, "vermisse": 2, "hilflos": 2, "beileid": 2, "tränen": 1, "verlust": 1,
        "verloren": 1, "gestorben": 1, "sterben": 1, "tod": 1, "allein": 1, "schrecklich": 1,
        "mitgefühl": 1,
    },
    "surprise": {
        "wow": 2, "erstaunlich": 2, "erstaunt": 2, "überrascht": 2, "überraschung": 2,
        "überraschend": 2, "unerwartet": 2, "unglaublich": 2, "verblüfft": 2,
        "sprachlos": 2, "wahnsinn": 1, "krass": 1, "schockiert": 2, "unfassbar": 2, "plötzlich": 1,
        "schock": 1,
    },
}

# Suffixes that may follow a base form ("traurig" -> "traurige", "ärgert", "ängste" ...).
SUFFIXES: tuple[str, ...] = ("", "e", "en", "er", "es", "em", "n", "t", "st", "te")

# Words that flip the polarity of an emotion word within a short window.
NEGATORS: frozenset[str] = frozenset(
    {"nicht", "kein", "keine", "keinen", "keinem", "keiner", "keines",
     "nie", "niemals", "ohne", "weder", "kaum"}
)

# How many tokens before an emotion word a negator may appear.
NEGATION_WINDOW = 3
