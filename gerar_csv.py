import pandas as pd
import numpy as np

N_SAMPLES = 3000

id_tarefa = np.arange(1, N_SAMPLES + 1)

prioridade = np.random.choice(['baixa', 'média', 'alta'], N_SAMPLES, p=[0.4, 0.4, 0.2])

dificuldade = np.random.uniform(1, 10, N_SAMPLES)

horas_prod_med = np.random.normal(loc=7.0, scale=1.5, size=N_SAMPLES).clip(3, 9)
nivel_foco_med = np.random.normal(loc=7.5, scale=1.0, size=N_SAMPLES).clip(4, 10)
estresse_med = np.random.normal(loc=5.0, scale=2.0, size=N_SAMPLES).clip(1, 10)
horas_dormidas_med = np.random.normal(loc=7.2, scale=1.0, size=N_SAMPLES).clip(5, 9)
satisfacao_med = np.random.normal(loc=7.0, scale=1.5, size=N_SAMPLES).clip(3, 10)
carga_trabalho_med = np.random.normal(loc=6.5, scale=1.5, size=N_SAMPLES).clip(4, 9)

tempo_base = 2.0
tempo_dificuldade = dificuldade * 0.4  
tempo_estresse = estresse_med * 0.2
tempo_foco = nivel_foco_med * 0.3 * (-1)
ruido = np.random.normal(0, 1.5, N_SAMPLES)

prioridade_map = {'baixa': 0.5, 'média': 1.0, 'alta': 1.5}
tempo_prioridade = np.array([prioridade_map[p] for p in prioridade])

tempo_conclusao_dias = (tempo_base + tempo_dificuldade + tempo_estresse + tempo_foco + ruido) * tempo_prioridade

tempo_conclusao_dias = np.maximum(0.5, tempo_conclusao_dias)


probabilidade_atraso = (
    (dificuldade > 7) * 0.2 + 
    (estresse_med > 6) * 0.3 + 
    (horas_dormidas_med < 6.5) * 0.15 +
    (tempo_conclusao_dias > 5) * 0.3 +
    np.random.uniform(0, 0.1, N_SAMPLES)
).clip(0, 1)

atraso = (np.random.rand(N_SAMPLES) < probabilidade_atraso).astype(int)


data = pd.DataFrame({
    'ID_TAREFA': id_tarefa,
    'PRIORIDADE': prioridade,
    'DIFICULDADE': dificuldade.round(1),
    'HORAS_PROD_MED': horas_prod_med.round(1),
    'NIVEL_FOCO_MED': nivel_foco_med.round(1),
    'ESTRESSE_MED': estresse_med.round(1),
    'HORAS_DORMIDAS_MED': horas_dormidas_med.round(1),
    'SATISFACAO_MED': satisfacao_med.round(1),
    'CARGA_TRABALHO_MED': carga_trabalho_med.round(1),
    'TEMPO_CONCLUSAO_DIAS': tempo_conclusao_dias.round(2),
    'ATRASO': atraso
})


data.to_csv('dados_projeto_ia.csv', index=False)

print(f"Arquivo 'dados_mndsh.csv' gerado com {N_SAMPLES} linhas.")
print("\nPrimeiras 5 linhas do dataset:")
print(data.head())