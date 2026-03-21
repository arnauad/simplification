SYSTEM_PROMPT = """\
Ets un assistent expert en simplificació de textos en català.

Objectiu:
Simplificar frases mantenint EXACTAMENT el mateix significat.
"""


USER_TEMPLATE = """\
Simplifica la frase següent mantenint el significat.

Respon NOMÉS amb la frase simplificada.

Frase: {sentence}
"""


SYSTEM_PROMPT_RULES = """\
Ets un assistent expert en simplificació de textos en català.

Objectiu:
Simplificar frases mantenint EXACTAMENT el mateix significat.

Segueix aquestes pautes:
{rules_text}
"""


FEW_SHOTS = [
    {
        "input": "Frase: El 2004, Barcelona va rebre 4,4 milions de turistes.",
        "output": "El 2004, Barcelona va acollir 4,4 milions de visitants."
    },
    {
        "input": "Frase: Important: aquest visat no és aplicable per a les persones amb ciutadania europea.",
        "output": "És important tenir-ho present perquè aquest visat no s'aplica als ciutadans europeus."
    },
    {
        "input": "Frase: –  Edificis d'habitatges (de propietaris únics, de propietaris que destinen els immobles a lloguer, comunitats de veïns, etc.).",
        "output": "– Edificis d'habitatges,     com per exemple:     • cases d'un sola persona propietària,      • cases que es lloguen,      • comunitats de veïns i veïnes,      • i altres.."
    }
]

RULES = [
"Utilitza paraules fàcils d’entendre que la gent conegui bé.",
"No utilitzis paraules difícils. Si has d’utilitzar paraules difícils, assegura’t d’explicar-les sempre de manera clara.",
"Utilitza exemples per explicar les coses.",
"Utilitza la mateixa paraula per descriure la mateixa cosa al llarg del document.",
"No utilitzis metàfores.",
"No utilitzis paraules d’altres llengües tret que siguin molt conegudes.",
"Evita utilitzar inicials.",
"Evita percentatges i números grans.",
"Mantén sempre les frases curtes.",
"Parla directament a les persones.",
"Utilitza frases positives en lloc de negatives.",
"Utilitza llenguatge actiu en lloc de passiu.",
"Presenta la informació en ordre fàcil d’entendre.",
"Repeteix informació important si cal.",
"Mantén la puntuació simple.",
"Evita abreviatures.",
"Una idea per frase.",
"Dona només la informació necessària."
]