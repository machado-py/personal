import sys
from _Monte_Carlo_v10 import ModelagemProd
 
if __name__ == "__main__":
    try:
        modelo1 = ModelagemProd()
        modelo1.executar_monte_carlo()
  
    finally:
        sys.exit()