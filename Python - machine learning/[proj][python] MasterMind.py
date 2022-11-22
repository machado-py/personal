import random
import os


# Função que solicita as 4 cores para o usuário
# ========================================================================================
def pedir_tentativa():
    global tentativa, cores, tentativas, print_cores
    
    # lower = coloca tudo para minúsculo, facilitando a comparação
    # replace = retirar espaços da frase (ex.: azul , verde  --> azul,verde)
    # split = para transformar a string em lista
    tentativa = str(input("Digite suas 4 cores separadas por vírgula:")).lower().replace(" ", "").split(',')
    
    # loop de input até que:
    # 1) possua 4 cores digitadas e que 2) todas elas estejam dentro das possibilidades de cores do jogo
    while (len(tentativa) != 4) or (not all(item in cores for item in tentativa)):
        print("\nPor favor, digite apenas as cores disponíveis, descritas no cabeçalho.")  
        tentativa = str(input("Digite suas 4 cores separadas por vírgula:")).lower().replace(" ", "").split(',')


# Função que verifica se as cores digitadas ainda estão disponíveis para escolha
# de acordo com a quantidade já utilizadas nas rodadas anteriores
# ========================================================================================
def verificar_disponibilidade():
    global rodada, tentativa, qtde_cores, chances
    
    # vai percorrer todas das cores verificando se aquela cor está disponível
    for cor in qtde_cores:
        if (qtde_cores[cor] + tentativa.count(cor)) > chances:                 # caso não esteja, envia mensagem
            pode_usar = chances - qtde_cores[cor]                              # mostrando qual cor está indisponível
            print(f"\nVocê só pode usar mais <{ pode_usar} > da cor < {cor} >. Reescreva sua tentativa.")
            verif_disponibilidade = False
            break                                                               # e não precisa verificar o restante
        else:
            qtde_cores[cor] += tentativa.count(cor)                             # caso esteja doisponível, continua
            verif_disponibilidade = True

    return verif_disponibilidade                                                # True = disponível, False= não

# Função que verifica se jogador ainda tem possibilidade de ganhar
# caso ele tenha usado já todas as possibilidades de uma cor que está contida na senha
# ========================================================================================
def verificar_chances():
    global senhas, qtde_cores, chances

    for senha in senhas:                                        # vai percorrer todas as 4 cores da senha e
        if qtde_cores[senha] == chances:                        # caso uma das cores da senha não tenha mais
            verif_chances = False                               # disponibilidade, termina o jogo
            break
        else:                                                   # caso tenha disponibilidade, 
            verif_chances = True                                # segue o jogo

    return verif_chances                                        # True = segue, False= termina


# Função que verifica se as cores digitadas existem na senha e se estão na posição correta
# ========================================================================================
def verificar_tentativa():
    global rodada, senhas, brancos, pretos, tentativa
    
    branco = 0
    preto = 0

    for i, cor in enumerate(tentativa):                         # vai percorrer todas as cores da tentativa
        if (cor in senhas) and (cor == senhas[i]):              # caso ela esteja na senha e na posição correta
            branco += 1                                         # soma na cor branca
        elif (cor in senhas):                                   # caso esteja, mas em outra posição
            preto += 1                                          # soma na cor preta

    brancos[rodada] = branco                                    # vai salvando o resultado de cada rodada 
    pretos[rodada] = preto                                      # para apresentação na tela para o usuário


# Função que irá imprimir as informações por rodada
# ========================================================================================
def imprimir():
    global rodada, brancos, pretos, qtde_cores, chances, senhas, tentativas
    
    os.system('cls' if os.name == 'nt' else 'clear')    # para sempre limpar a tela antes de imprimir novamente
    # cabeçalho ---------------------------------------------------------------------------
    print("=========================================================================================================")
    print("Desenvolvido por: MARCELO MACHADO   |   Projeto Coding Tank   |   Let's Code")
    print("----------------------------------------------------------------------------")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<   MASTER  MIND  GAME   >>>>>>>>>>>>>>>>>>>>>>>>>>")
    print("=========================================================================================================")
    print("\n >>> VOCÊ É CAPAZ DE DESCOBRIR AS 4 CORES QUE PENSEI ??? \n     Dica: elas não se repetem. Boa sorte.\n")
    print("=========================================================================================================\n")
    print(f"Rodada: {rodada+1} \n")
    
    for cor in qtde_cores:                                              # vai percorrer todas as disponibilidades
        if cor == 'amarelo':                                            # só imprime esta frase inicial na primeira
            print(f"Quantas vezes você ainda pode usar cada cor:")      # iteração do loop
    
        # end = substitui a quebra de página pelo PIPE
        print(f"{cor}: {chances - qtde_cores[cor]}", end=" | ")         # imprime todas as disponibilidades para usuário
    
    # corpo -------------------------------------------------------------------------------
    print("\n")
    print("=========================================================================================================")
    for i in range(0,chances):              # vai imprimir sempre todas as rodadas para o usuário com info atualziada
        if i == rodada:                     # imprime a seta para facilitar a identificação da rodada atual
            seta = f">>"
        else:
            seta = "  "
            
        if i == 9:                          # imprime um separador entre as linhas das rodadas
            separador = ""                  # menos na última linha
        else:
            separador = "\n- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -"
        
        print_tentativa = "  |  ".join(tentativas[i])       # cria uma string para melhor apresentação das tentativas

        # imprime as informações de cada rodada
        print(f"{seta} Rodada {i+1} \t Branco: {brancos[i]}  |  Preto: {pretos[i]} \t\t {print_tentativa} {separador}")
 
    print("=========================================================================================================")


# Função que irá definir o fluxo do jogo
# ========================================================================================
def fluxo():
    global rodada, chances, print_senhas, brancos, tentativas, tentativa
    
    for i in range(chances):                                            # loop dentro das chances oferecidas no jogo
        imprimir()                                                      # imprime as informações
        
        while True:
            pedir_tentativa()                                           # pede a tentativa do jogador
            if verificar_disponibilidade():                             # só pára o loop de pedir tentativa
                tentativas[rodada] = tentativa                          # caso tenha a disponibilidade das peças
                break                                                   # salva a tentiva da rodada para mostrar o histórico
        
        verificar_tentativa()                                           # verifica quantos acertos tiveram
        
        if brancos[rodada] == 4:                                        # caso o jogador acerte a senha
            imprimir()                                                  # já pára o loop das tentativas
            print("Você acertou !!! =O \nTu é o bixo mesmo, hem! XD")
            break

        if (not verificar_chances()) and (rodada+1 < chances):          # termina o jogo caso não tenha mais a possibilidade
            imprimir()                                                  # do jogador ganhar devido a falta de peças
            print("Infelizmente, você não tem mais chances de acertar a senha por falta de peças.")
            break

        rodada += 1
    
    if (brancos[rodada] != 4) and (rodada == chances ):                 # frase final caso o jogador não ganhe
        imprimir()                                                      # dentro das chances oferecidas
        print("Não foi dessa vez. :( \nPense melhor na próxima. ;D")
    
    print(f"\nVerifique a senha >>>  {print_senhas}")                   # imprime a senha para comparação final


# Iniciar o jogo
# ========================================================================================
jogar_novamente = "s"

while jogar_novamente == "s":
    # Declaração das variáveis globais ---------------------------------------------------
    cores = ['amarelo','verde','azul','roxo','vermelho','laranja']
    qtde_cores = {'amarelo':0,'verde':0,'azul':0,'roxo':0,'vermelho':0,'laranja':0} # para contagem da disponibilidade
    chances = 10
    rodada = 0
    brancos = [[]]*(chances+1)                                  # lista dos acertos em branco de cada rodada
    pretos = [[]]*chances                                       # lista dos acertos em preto de cada rodada
    tentativas = [[]]*chances                                   # lista das tentativas de cada rodada
    tentativa = []
    senhas = random.sample(cores, 4)                            # gera a senha de forma randômica
    print_senhas = " | ".join(senhas)                           # cria uma string para melhor apresentação da senha

    # Executa o jogo ---------------------------------------------------------------------
    fluxo()
    jogar_novamente = input("\nDeseja jogar novamente? (s/n): ")        # loop caso usuário queira continuar jogando

print("\nAté a próxima!")