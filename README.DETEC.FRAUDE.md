# Detecção de Fraude em Transações de Cartão de Crédito

Projeto de machine learning para identificar transações fraudulentas em um dataset real e fortemente desbalanceado, cobrindo o pipeline completo: exploração, preparação dos dados, treino e comparação de modelos, e explicabilidade com SHAP.

## O Problema

O dataset contém **284.807 transações**, das quais:

| Classe | Quantidade | Proporção |
|---|---|---|
| Normais (Class = 0) | 284.315 | 99,827% |
| Fraudes (Class = 1) | 492 | 0,173% |

Esse desbalanceamento extremo torna a **acurácia uma métrica enganosa**: um modelo que classifica todas as transações como "não fraude" já acertaria 99,8% dos casos, sem nunca identificar uma única fraude real. Por isso, o foco da avaliação está em **recall, precisão e F1-score da classe fraude (classe 1)**, não na acurácia geral.

## Preparação dos Dados

1. **Feature engineering**: criação da variável `Log_Amount` (log do valor da transação), para reduzir o impacto de outliers de valores muito altos.
2. **Remoção da coluna `Time`**: não trouxe ganho relevante para os modelos testados.
3. **Padronização**: todas as variáveis foram escaladas com `StandardScaler`.
4. **Separação treino/teste**: split de 70/30 com `stratify=y`, garantindo que a proporção de fraudes (0,173%) fosse mantida em ambos os conjuntos.

As variáveis `V1` a `V28` já vêm transformadas por PCA no dataset original, para proteção de privacidade dos dados reais dos titulares dos cartões.

## Comparação dos Modelos

Foram treinados e comparados quatro cenários, todos avaliados no conjunto de teste (85.443 transações, 148 fraudes):

| Modelo | Precisão (fraude) | Recall (fraude) | F1-score (fraude) | ROC AUC |
|---|---|---|---|---|
| Regressão Logística (baseline, `class_weight='balanced'`) | 0,06 | **0,87** | 0,12 | 0,97 |
| Regressão Logística (limiar ajustado para 0,7) | 0,12 | 0,85 | 0,22 | — |
| Random Forest (`class_weight='balanced'`) | **0,97** | 0,70 | 0,82 | — |
| **XGBoost** (`scale_pos_weight`) | 0,90 | 0,78 | **0,84** | — |

**Conclusão da comparação**: a Regressão Logística maximiza o recall, mas com precisão extremamente baixa (muitos falsos positivos). O Random Forest maximiza a precisão, mas perde recall. O **XGBoost obteve o melhor equilíbrio geral**, com o maior F1-score entre os quatro modelos testados.

### Ajuste de limiar no XGBoost

Testamos limiares de decisão diferentes do padrão (0,5) no XGBoost:

| Limiar | Precisão | Recall | F1-score |
|---|---|---|---|
| 0,2 | 0,84 | **0,80** | 0,82 |
| 0,3 | 0,86 | 0,78 | 0,82 |
| 0,4 | 0,89 | 0,78 | 0,83 |
| 0,5 (padrão) | 0,90 | 0,78 | **0,84** |

O limiar padrão (0,5) já apresentou o melhor F1-score. O limiar 0,2 é uma alternativa válida caso a prioridade seja maximizar o recall (pegar mais fraudes), aceitando uma pequena queda de 6 pontos percentuais na precisão.

**Modelo e limiar escolhidos: XGBoost, limiar 0,5**, por apresentar o melhor equilíbrio entre recall e precisão (F1 = 0,84) entre todas as configurações testadas.

## Explicabilidade (SHAP)

Utilizamos `TreeExplainer` do SHAP sobre o modelo XGBoost para entender quais variáveis mais influenciaram as decisões de classificação.

**Principais achados:**

- **V14** foi a variável de maior impacto no modelo. Valores **baixos** de V14 empurram fortemente a previsão em direção à fraude — é o sinal mais forte identificado em todo o modelo.
- **V4** foi a segunda variável mais importante, com padrão inverso: valores **altos** de V4 empurram a previsão em direção à fraude.
- Outras variáveis com impacto relevante, em ordem de importância: V12, V10, V11, V3, V19, V16 e Amount.

Esse resultado é consistente com padrões já documentados publicamente sobre esse dataset, reforçando que, mesmo sem saber o significado original das variáveis (por serem componentes de PCA), o modelo consegue extrair sinais claros e replicáveis de fraude.

## O Que Mudei em Relação ao Caminho da Expert

Em vez de comparar apenas os três modelos sugeridos na aula (Regressão Logística, Random Forest e XGBoost) com um único limiar padrão cada, ampliei a comparação em dois pontos:

1. **Testei o efeito do ajuste de limiar em dois modelos, não em um só.** Além do XGBoost, também ajustei o limiar de decisão da Regressão Logística (de 0,5 para 0,7), o que evidenciou como um baseline com recall alto mas precisão muito baixa (0,06) se comporta ao subir o limiar — a precisão melhora (para 0,12), mas ainda fica muito abaixo da dos modelos baseados em árvore.

2. **Fiz uma varredura de limiares no XGBoost** (0,2 / 0,3 / 0,4 / 0,5), em vez de usar apenas o limiar padrão. Isso mostrou que, nesse caso, o limiar padrão (0,5) já era o mais equilibrado (melhor F1 = 0,84), mas deixou registrado o trade-off: o limiar 0,2 seria a escolha certa se a prioridade fosse maximizar recall a qualquer custo.

3. **Optei por ponderação de classes (`class_weight='balanced'` e `scale_pos_weight`) em vez de under/oversampling.** Como o desafio sugere testar as duas abordagens, escolhi a ponderação porque ela evita descartar dados reais (undersampling) e não gera transações sintéticas (oversampling) — com o volume de dados disponível (mais de 284 mil transações), a ponderação já foi suficiente para alcançar recall e precisão competitivos, sem adicionar a complexidade extra de gerar ou remover amostras.

4. **Troquei o `shap.Explainer` genérico pelo `shap.TreeExplainer`** com `feature_perturbation='tree_path_dependent'`. O explicador genérico não suporta os splits categóricos internos do XGBoost e gera erro; o `TreeExplainer` é a alternativa correta e recomendada para modelos baseados em árvore, além de ser mais rápido.

5. **Adicionei as curvas ROC e Precisão-Recall como saída visual do notebook**, complementando os números do classification report — a curva Precisão-Recall em particular é mais informativa que a ROC nesse tipo de dataset desbalanceado, porque a ROC pode parecer artificialmente boa mesmo quando a precisão da classe minoritária está baixa.

## Como Rodar

1. Abra o notebook no Google Colab.
2. Execute as células em ordem (`Ambiente de execução > Executar tudo`).
3. O dataset é carregado diretamente por link, sem necessidade de download manual:
   ```python
   df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
   ```

## Estrutura do Projeto

```
.
├── notebook.ipynb   # Pipeline completo: exploração, preparação, treino, avaliação e SHAP
└── README.md        # Este arquivo
```
