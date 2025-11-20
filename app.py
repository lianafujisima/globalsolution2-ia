from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

try:
    MODELO_REGRESSAO = joblib.load('regressao_tempo.joblib')
    MODELO_CLASSIFICACAO = joblib.load('classificacao_risco.joblib')
    print("Modelos de IA carregados com sucesso!")
except Exception as e:
    print(f"Erro ao carregar modelos: {e}")
    MODELO_REGRESSAO = None
    MODELO_CLASSIFICACAO = None

FEATURE_ORDER = ['PRIORIDADE', 'DIFICULDADE', 'HORAS_PROD_MED', 'NIVEL_FOCO_MED', 
                 'ESTRESSE_MED', 'HORAS_DORMIDAS_MED', 'SATISFACAO_MED', 'CARGA_TRABALHO_MED']

@app.route('/predict/tempo_conclusao', methods=['POST'])
def predict_tempo_conclusao():
    if not MODELO_REGRESSAO:
        return jsonify({"error": "Modelo de regressão não está carregado."}), 500

    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data], columns=FEATURE_ORDER)
        prediction = MODELO_REGRESSAO.predict(input_df)
        tempo_estimado = float(prediction[0])
        
        return jsonify({
            "status": "success",
            "tempo_estimado_dias": round(tempo_estimado, 2),
            "mensagem": f"O tempo estimado para conclusão da tarefa é de {round(tempo_estimado, 2)} dias."
        })

    except Exception as e:
        return jsonify({"error": str(e), "mensagem": "Verifique o formato dos dados de entrada."}), 400

@app.route('/predict/risco_atraso', methods=['POST'])
def predict_risco_atraso():
    if not MODELO_CLASSIFICACAO:
        return jsonify({"error": "Modelo de classificação não está carregado."}), 500

    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data], columns=FEATURE_ORDER)
        prediction = MODELO_CLASSIFICACAO.predict(input_df)
        proba_atraso = MODELO_CLASSIFICACAO.predict_proba(input_df)[0][1]
        risco_atraso = int(prediction[0])

        if risco_atraso == 1:
            msg = f"ALERTA: Alto Risco de Atraso. Probabilidade de {round(proba_atraso*100, 2)}%."
        else:
            msg = f"Baixo Risco de Atraso. Probabilidade de conclusão no prazo de {round((1-proba_atraso)*100, 2)}%."

        return jsonify({
            "status": "success",
            "risco_atraso": risco_atraso,
            "probabilidade_atraso": round(proba_atraso, 4),
            "mensagem": msg
        })

    except Exception as e:
        return jsonify({"error": str(e), "mensagem": "Verifique o formato dos dados de entrada."}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)