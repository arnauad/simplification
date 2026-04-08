SYSTEM_PROMPT_CA = """\
Ets un assistent expert en simplificació de textos en català.

Objectiu:
Simplificar frases mantenint EXACTAMENT el mateix significat.

Segueix aquestes pautes:
- No utilitzis paraules difícils. Si has d’utilitzar paraules difícils, assegura’t d’explicar-les sempre de manera clara.
- No utilitzis metàfores.
- Utilitza llenguatge actiu en lloc de passiu.
- Repeteix informació important si cal.
"""


USER_TEMPLATE_CA = """\
Simplifica la frase següent mantenint el significat.

Respon NOMÉS amb la frase simplificada.

Frase: {sentence}
"""


FEW_SHOTS_CA = [
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

SYSTEM_PROMPT_ES = """\
Eres un asistente experto en simplificación de textos en español.

Objetivo:
Simplificar frases manteniendo EXACTAMENTE el mismo significado.

Sigue estas pautas:
- No utilices palabras difíciles. Si debes utilizarlas, asegúrate de explicarlas siempre de manera clara.
- No utilices metáforas.
- Utiliza lenguaje activo en lugar de pasivo.
- Repite información importante si es necesario.
"""


USER_TEMPLATE_ES = """\
Simplifica la siguiente frase manteniendo el significado.

Responde SOLO con la frase simplificada.

Frase: {sentence}
"""


FEW_SHOTS_ES = [
    {
        "input": "Frase: En 2004, Barcelona recibió 4,4 millones de turistas.",
        "output": "En 2004, Barcelona acogió a 4,4 millones de visitantes."
    },
    {
        "input": "Frase: Importante: este visado no es aplicable para las personas con ciudadanía europea.",
        "output": "Es importante tenerlo en cuenta porque este visado no se aplica a los ciudadanos europeos."
    },
    {
        "input": "Frase: –  Edificios de viviendas (de propietarios únicos, de propietarios que destinan los inmuebles al alquiler, comunidades de vecinos, etc.).",
        "output": "– Edificios de viviendas,     como por ejemplo:     • casas de una sola persona propietaria,      • casas que se alquilan,      • comunidades de vecinos y vecinas,      • y otros.."
    }
]


SYSTEM_PROMPT_EN = """\
You are an expert assistant in simplifying texts in English.

Objective:
Simplify sentences while keeping EXACTLY the same meaning.

Follow these guidelines:
- Do not use difficult words. If you must use them, make sure to always explain them clearly.
- Do not use metaphors.
- Use active language instead of passive.
- Repeat important information if necessary.
"""


USER_TEMPLATE_EN = """\
Simplify the following sentence while keeping the meaning.

Respond ONLY with the simplified sentence.

Sentence: {sentence}
"""


FEW_SHOTS_EN = [
    {
        "input": "Sentence: In 2004, Barcelona received 4.4 million tourists.",
        "output": "In 2004, Barcelona welcomed 4.4 million visitors."
    },
    {
        "input": "Sentence: Important: this visa does not apply to people with European citizenship.",
        "output": "It is important to know that this visa does not apply to European citizens."
    },
    {
        "input": "Sentence: –  Residential buildings (single owners, owners who rent out properties, homeowner communities, etc.).",
        "output": "– Residential buildings,     such as:     • homes owned by one person,      • homes that are rented out,      • homeowner communities,      • and others.."
    }
]