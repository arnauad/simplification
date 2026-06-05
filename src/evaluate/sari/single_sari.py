import evaluate

def sari(source, prediction, reference):
    sari = evaluate.load("sari")
    
    result = sari.compute(
        sources=[source],
        predictions=[prediction],
        references=[[reference]]
    )
    
    return result["sari"]

if __name__ == "__main__":
    # "original": "El Banc del Moviment és un servei municipal de la Xarxa Solidària de Productes de Suport de Barcelona.",
    #         "prediction": "El Banc del Moviment és un servei de la Xarxa Solidària de Productes de Suport de Barcelona. \n \n \nEl Banc del Moviment és un servei de la Xarxa Solidària de Productes de Suport de Barcelona. \n \nEl Banc del Moviment és un servei de la Xarxa Solidària de Productes de Suport de Barcelona. \n \nEl Banc del Moviment és",
    #         "reference": "El Banc del Moviment és un servei de l'Ajuntament de Barcelona. Forma part de la Xarxa Solidària de Productes de Suport de Barcelona.",

    source_sentence = "Barcelona és un gran lloc per viure, però aquí també hi ha impostos, fins i tot si no tens ingressos!"
    human_reference = "Barcelona és un bon lloc per viure, però aquí també hi ha impostos. Tothom ha de pagar impostos, fins i tot si no teniu ingressos!"

    model_prediction = "Barcelona és un gran lloc per viure, però aquí també hi ha impostos, fins i tot si no tens ingressos!"

    score = sari(source_sentence, model_prediction, human_reference)
    

    # 51.011
    # 51.5666
    print(f"SARI Score: {score:.4f}")
    print(f"Normalized (0-1) Score: {score / 100:.4f}")