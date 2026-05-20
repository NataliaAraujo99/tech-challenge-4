from pydantic import BaseModel, Field
from typing import List

class InputData(BaseModel):
    """
    Dados de entrada para a predição LSTM.
    
    Recebe uma sequência de 60 timesteps com valores normalizados.
    
    Exemplo com 60 dias de dados OHLCV:
    ```json
    {
        "data": [
            [0.5, 0.52, 0.48, 0.51, 0.8],
            [0.51, 0.53, 0.50, 0.52, 0.85],
            [0.52, 0.54, 0.51, 0.53, 0.90],
            ...
            [0.58, 0.60, 0.57, 0.59, 0.95]
        ]
    }
    ```
    Total: 60 linhas (timesteps) x 5 colunas (Open, High, Low, Close, Volume)
    """
    data: List[List[float]] = Field(..., description="Sequência de 60 timesteps com 5 features (OHLCV) normalizados entre 0 e 1")