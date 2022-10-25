import numpy as np
import xlwings as xw
import random as rnd
import pandas as pd
from shutil import copy2
import glob
import sys
import ctypes


class ModelagemProd():

    def __init__(self):
        """        
        Buscar na planilha os dados para iniciar a geração dos cenários.
        Atribuir em variáveis cada região de células do arquivo para utilização no modelo.
        """

        # Buscando arquivo de input
        # ==================================================================================================
        df = pd.read_excel('InputFile.xlsx', sheet_name='input')

        self.buscas = int(df.iloc[96, 3])  # quantas vezes o Monte Carlo irá rodar
        self.atual = int(df.iloc[92, 15])  # se tabela com dados manuais está OK
        
        # Disponibilidade de sites por step + variáveis de investimento
        # ==================================================================================================
        sites_padrao = np.array([[1, 0, 0],  # montagem básica > SITE3 e SITE1 pode ser 1, caso haja investimento
                                [1, 0, 1],  # montagem interior > SITE1 pode ser 1, se houver investimento
                                [0, 1, 1],  # pintura > SITE2 pode ser 1, se houver investimento
                                [0, 0, 1]]) # voo produção > SITE1 pode ser 1, se houver investimento

        invest_seletor = df.iloc[9:13, 3:6].to_numpy()

        self.sites_disponiveis = np.logical_or(sites_padrao, invest_seletor)*1

        invest_valores = df.iloc[9:13, 8:11].to_numpy()

        self.total_investimentos = (invest_seletor * invest_valores).sum()

        # Restrições de interior em SITE3
        # ==================================================================================================
        # seletor - aplica ou não a restrição?
        self.interior_flag_min_ano = df.iloc[26, 3:6].to_numpy()

        # valores - quantidade mínima de 12 aviões em site/ano
        self.interior_valor_min_ano = df.iloc[27, 3:6].to_numpy()

        # Restrições de cabine de pintura
        # ==================================================================================================
        # todas capacidades
        self.todas_capacidades = df.iloc[16:20, 13:16].to_numpy()

        # seletor - aplica ou não a restrição?
        self.pintura_flag_max_mes = df.iloc[31, 3:6].to_numpy()

        # capacidade cabine pintura
        self.pintura_valor_max_mes = self.todas_capacidades[2,:]

        # Demanda a ser atendida
        # ==================================================================================================
        self.demanda_por_mes = df.iloc[3, 3:15].to_numpy()

        # Verifica se a demanda de cada mês está dentro das restrições de capacidade da cabine de pintura
        self.restricao_cabines = self.pintura_valor_max_mes.sum()
        self.flag_restricao_cabine = all(self.demanda_por_mes <= self.restricao_cabines)

        # Matriz de parâmetros - custo mod e leadtime
        # ==================================================================================================
        self.custo_mod_detalhe_input = df.iloc[59:63, 3:6].to_numpy()

        self.leadtime_detalhe_input = df.iloc[49:53, 10:13].to_numpy()

        # [Mtg básica, Mtg interior, Pintura, Voo Prod]
        self.cenario_wip = df.iloc[66, 3]
        if self.cenario_wip == 1:
            self.WipAcDia_interior_pintura = df.iloc[70:74, 6].to_numpy()
            self.step_pintura = 2
            self.step_interior = 1
        else:
            self.WipAcDia_interior_pintura = df.iloc[70:74, 14].to_numpy()
            self.WipAcDia_interior_pintura[1], self.WipAcDia_interior_pintura[2] = self.WipAcDia_interior_pintura[2], self.WipAcDia_interior_pintura[1]
            self.step_pintura = 1
            self.step_interior = 2

        # Buffers
        # ==================================================================================================
        temp = df.iloc[55:58, 10].to_numpy()
        buffers = np.array([temp]*3)           # SITE2 > SITE1 = 5 ; SITE1 > SITE2 = 5 ; BR > SITE3 = 8
        self.matriz_buffers = np.array(np.r_[np.array([[0,0,0]]), buffers])
        del temp, buffers

        # Parâmetros de custos
        # ==================================================================================================
        self.custo_material = df.iloc[37, 3]          # [USD/ ac] - custo variável
        self.preco_venda = df.iloc[38, 3]             # [USD/ ac] - custo variável
        self.custo_overhead = df.iloc[39, 3]          # [USD/ year] - custo fixo
        self.custo_sustaining = df.iloc[40, 3]        # [USD/ year] - custo fixo
        self.despesa_sga = df.iloc[41, 3]             # [USD/ year] - custo fixo
        self.custo_capital_invest = df.iloc[8,15]

        # Perda por não ser 'made in USA'
        # ==================================================================================================
        perda_usa_min = df.iloc[43, 3]    
        perda_usa_max = df.iloc[44, 3]    
        self.fator_a = ( np.log(perda_usa_min) - np.log(perda_usa_max))/12
        self.fator_b = perda_usa_max
        self.wacc =  df.iloc[36, 3]    

        # Tabela da produção atual para comparação, apenas se não houver erro no preenchimento da tabela
        # ==================================================================================================
        self.prod_atual = []

        if self.atual == 1:
            col = 3
            for c in range(col, col+12):
                temp2=[]
                lin = 80
                for _ in range(1,5):
                    temp1 = df.iloc[lin:lin+3, c]
                    temp2.append(temp1)
                    lin +=3

                self.prod_atual.append(temp2)

            self.prod_atual = np.array(self.prod_atual)

        del df


    def gerar_combinacao(self):
        """        
        Gerar os cenários para atender a demanda solicitada de forma aleatória
        e considerando algumas regras do negócio. 
        A tabela de decisão gerada neste método será utilizada pelos métodos de
        calcular a margem do cenário e na execução da simulação de Monte Carlo.
        """

        # Construção da matriz de decisão
        # ==================================================================================================
        sites = 3    
        sites_seletor = np.array(range(sites))  # Na lista >> 0 = SITE2 ; 1 = SITE1 ; 2 = SITE3
        steps = 4    # 0 = Mtg básica ; 1 = Mtg Interior ; 2 = Pintura ; 3 = Voo Produção (AFA)
        meses = 12
        tabela_decisao = [[[0]*sites]*steps]*meses
        self.tabela_decisao = np.array(tabela_decisao)
        
        # Inicia a função para gerar combinações
        # ==================================================================================================
        if not self.flag_restricao_cabine:
            ctypes.windll.user32.MessageBoxW(0, f"Revise o input de demanda. O máximo possível são {self.restricao_cabines} aviões/mês, devido cabine de pintura.",
                                        "Revise as demandas", 0)
            sys.exit("Revise a quantidade de input.")
        
        else:
            for imes, mes in enumerate(self.tabela_decisao):
        
                for istep, step in enumerate(mes):
                    
                    for demanda in range(self.demanda_por_mes[imes]):                
                        # Início de todas as verificações para manter ou excluir site para escolha
                        # ---------------------------------------------------------------------------------
                        # verifica qual site está disponível para aquele step
                        drop_sites_seletor = self.sites_disponiveis[istep]
                        
                        # verifica qual site está disponível dependendo do que estiver em  SITE3
                        # se for interior ou pintura, tem que limitar pela restrição de pintura primeiro
                        # se for voo prod, tem que ser >= ao step anterior
                        if istep > 0:
                            if self.tabela_decisao[imes,istep,2] < self.tabela_decisao[imes,istep-1,2]:
                                drop_sites_fica_site3 = [0, 0, 1]
                            else:
                                drop_sites_fica_site3 = [1, 1, 1]
                        else:
                            drop_sites_fica_site3 = [1, 1, 1]
                        drop_sites_fica_site3 = np.array(drop_sites_fica_site3)
                        
                        
                        # verifica restrição cabine pintura
                        drop_sites_restricao_pintura = []
                        if istep <= self.step_pintura:
                            total_prod_pintura_site = self.tabela_decisao[imes,istep,:]
                            for i, flag in enumerate(self.pintura_flag_max_mes):
                                if flag == 0:
                                    drop_sites_restricao_pintura.append(1)
                                else:
                                    x = (total_prod_pintura_site[i] < self.pintura_valor_max_mes[i])*1
                                    drop_sites_restricao_pintura.append(x)
                        else:
                            drop_sites_restricao_pintura = [1, 1, 1]
                        drop_sites_restricao_pintura = np.array(drop_sites_restricao_pintura)
                        
                        
                        # verifica restrição de mínimo anual. Conta precisa ser feita considerando a partir do interior
                        drop_sites_min_site3_ano = []
                        if istep == self.step_interior:
                            total_madein_site_ano = self.tabela_decisao[:,self.step_interior,:].sum(axis=0)
                            for i, flag in enumerate(self.interior_flag_min_ano):
                                if flag == 0:
                                    drop_sites_min_site3_ano.append(1)
                                # verifica se tem capacidade na pintura, quando interior vier antes
                                elif drop_sites_restricao_pintura[i]==0 and self.step_interior < self.step_pintura:    
                                    drop_sites_min_site3_ano.append(0)
                                else:
                                    # primeiro calcula quantos meses são necessários para produzir todos aviões USA
                                    # (quantidade necessária - produzida) / restrição cabine
                                    meses_necessarios = (self.interior_valor_min_ano[i] - total_madein_site_ano[i]) / self.pintura_valor_max_mes[i]
                                    # se meses restantes do ano <= meses necessários, drop SITE2 e SITE1
                                    drop_sites_min_site3_ano = [0, 0, 1] if (12-imes) <= np.ceil(meses_necessarios) else [1, 1, 1]
                        else:
                            drop_sites_min_site3_ano = [1, 1, 1]
                        drop_sites_min_site3_ano = np.array(drop_sites_min_site3_ano)
                        

                        # Depois de todas verificações feitas, manter só as plantas que passaram por todos os filtros
                        # ---------------------------------------------------------------------------------
                        # verifica todos os drops. Se 0 = drop, se 1 = mantém
                        drop_sites = (drop_sites_fica_site3 * drop_sites_min_site3_ano * drop_sites_restricao_pintura * drop_sites_seletor).astype(bool)
                        
                        # deixa apenas sites elegíveis para escolha
                        sites_elegiveis = sites_seletor[drop_sites]
                        # escolhe 1 dentro os sites disponíveis
                        choose_site = rnd.choice(sites_elegiveis)
                        # soma um para aquele site, naquele step
                        self.tabela_decisao[imes, istep, choose_site] += 1


    def calcular_margem(self):
        """        
        A partir da tabela de decisão gerada no método gerar_combinacao,
        fazer os cálculos de receita e custo para cálculo da margem operacional.
        """

        # Inversão da Tabela de Decisão, caso seja cenário pintura > interior
        # ==================================================================================================
        tabela_decisao2 = self.tabela_decisao.copy()
        if self.cenario_wip ==2:
            for mes in tabela_decisao2:
                mes[[1]], mes[[2]] = mes[[2]], mes[[1]]

        # CUSTO MOD
        # ==================================================================================================
        total_por_step = tabela_decisao2.sum(axis=2)
        total_por_mes = total_por_step[:,1]

        # Custo de MOD para cada site para cada step.
        # ==================================================================================================
        custo_mod_detalhe_output = np.multiply(tabela_decisao2, self.custo_mod_detalhe_input)

        # Custo MOD agrupado
        # ==================================================================================================
        # por step em cada mes
        custo_mod_step_mes = custo_mod_detalhe_output.sum(axis=2)

        # Leadtimes para cada site para cada step.  
        # ==================================================================================================
        # Primeiro passo, fazendo a multiplicação da qtde de avião pelo leadtime diretamente, sem buffer.
        leadtime_output_temp = np.multiply(tabela_decisao2, self.leadtime_detalhe_input)
        # Agora, verificando se teve mudança de site para outro.
        temp1_buffer = ((self.tabela_decisao[:,1:,:] - self.tabela_decisao[:,:-1,:])>=0) * (self.tabela_decisao[:,1:,:] - self.tabela_decisao[:,:-1,:])
        # Adicionando um vetor de zeros para facilitar nultiplicação
        temp2 = np.array([0,0,0])
        temp2_buffer = np.array([np.r_[[temp2], mes] for mes in temp1_buffer])
        # Multiplicando quantidade de aviões que tiveram mudança de site pelos tempos de buffer Brasil e SITE3.
        temp3_buffer = temp2_buffer * self.matriz_buffers
        # se for cenário Pintura > WIP, inverto posições
        # pois o buffer fiz com a tabela original e não com a tabela_decisão2
        if self.cenario_wip ==2:
            for mes in temp3_buffer:
                mes[[1]], mes[[2]] = mes[[2]], mes[[1]]

        # Finalmente, somando o buffer com leadtime geral.
        leadtime_detalhe_output = leadtime_output_temp + temp3_buffer

        # Criando tabela de buffer
        # ==================================================================================================
        # soma antes
        leadtime_porStep_temp = leadtime_detalhe_output.sum(axis=2)
        # agora faz a média
        leadtime_porStep_output = leadtime_porStep_temp / total_por_step

        # WACC/dia de cada step
        # ==================================================================================================
        wacc_dia_temp1 = pow(1+self.wacc, 1/360)
        wacc_dia_step_mes = pow(wacc_dia_temp1, leadtime_porStep_output*1.5)-1

        # Custo carregamento do estoque de cada step em cada mês
        # ==================================================================================================
        custo_carregamento_step_mes = total_por_step * self.WipAcDia_interior_pintura * (leadtime_porStep_output/20) * leadtime_porStep_output * wacc_dia_step_mes

        # Verificando o total produzido no ano
        self.total_ano = total_por_mes.sum()
        # Soma o total de produção do ano na montagem interior de cada site
        self.total_madein_site_ano = self.tabela_decisao[:,self.step_interior,:].sum(axis=0)

        # Cálculo da % de perda
        self.perda_volume_entrega = self.fator_b * np.exp(self.total_madein_site_ano[2] * self.fator_a)
        # Aplica a perda
        totalAc_pos_perda = round( (1-self.perda_volume_entrega) * self.total_ano, 0)

        # montar DRE
        DRE_Receita_USD = totalAc_pos_perda * self.preco_venda
        # ---------------------------------------------------------------
        DRE_CustoMaterial_USD = -self.custo_material * totalAc_pos_perda
        # ---------------------------------------------------------------
        DRE_CustoBásica_USD = - (custo_mod_step_mes.sum(axis=0)[0] + (1-self.perda_volume_entrega)* custo_carregamento_step_mes.sum(axis=0)[0])
        DRE_CustoInterior_USD = - (custo_mod_step_mes.sum(axis=0)[1] + (1-self.perda_volume_entrega)* custo_carregamento_step_mes.sum(axis=0)[1])
        DRE_CustoPintura_USD = - (custo_mod_step_mes.sum(axis=0)[2] + (1-self.perda_volume_entrega)* custo_carregamento_step_mes.sum(axis=0)[2])
        DRE_CustoAFA_USD = - (custo_mod_step_mes.sum(axis=0)[3] + (1-self.perda_volume_entrega)* custo_carregamento_step_mes.sum(axis=0)[3])
        # ---------------------------------------------------------------
        DRE_Overhead_USD = -self.custo_overhead
        DRE_Sustaining_USD = -self.custo_sustaining
        DRE_SGA_USD = -self.despesa_sga
        DRE_CustoCapInvest_USD = -self.custo_capital_invest
        # ---------------------------------------------------------------
        DRE_Margem_USD = DRE_Receita_USD + DRE_CustoMaterial_USD + DRE_CustoBásica_USD + DRE_CustoInterior_USD + DRE_CustoPintura_USD + DRE_CustoAFA_USD + DRE_Overhead_USD + DRE_Sustaining_USD + DRE_SGA_USD + DRE_CustoCapInvest_USD
        DRE_Margem_pct = DRE_Margem_USD / DRE_Receita_USD

        # criar dicionário da DRE
        self.dre = {
            "receita":DRE_Receita_USD,
            "custo_material":DRE_CustoMaterial_USD,
            "custo_mtg_basica":DRE_CustoBásica_USD,
            "custo_mtg_interior":DRE_CustoInterior_USD,
            "custo_pintura":DRE_CustoPintura_USD,
            "custo_voo_prod":DRE_CustoAFA_USD,
            "custo_overhead":DRE_Overhead_USD,
            "custo_sustaining":DRE_Sustaining_USD,
            "desp_sga":DRE_SGA_USD,
            "custo_cap_invest":DRE_CustoCapInvest_USD,
            "margem_usd":DRE_Margem_USD,
            "margem_pct":DRE_Margem_pct
        }


    def simular_atual(self):        
        self.atual_dre = {}
        self.perda_volume_atual = 0
        self.tabela_decisao = self.prod_atual.copy()
        self.calcular_margem()
        self.atual_dre = self.dre
        self.perda_volume_atual = self.perda_volume_entrega
        

    def gerar_output_file(self):
        """        
        Criar um arquivo xlsx de output com o melhor cenário gerado pela simulação de Monte Carlo.
        O arquivo terá 2 abas: 
            uma com o cenário de input, para manter o histórico do que gerou aquele resultado
            outra com o output, com as informações de DRE e Cenário de Produção do melhor cenário.
        """

        # conta quantos arquivos de output já tem na pasta
        num = len(glob.glob('output_files/*outputfile*'))+1

        total_madein_site_ano = self.matriz_melhor_combinacao[:,self.step_interior,2].sum(axis=0)

        # copiando as abas do modelo --------------------------------------------------------------------
        excel_app = xw.App(visible=False)
        path1 = 'InputFile.xlsx'
        path2 = f'output_files/_Model_OutputFile_{num}.xlsx' 

        copy2(path1, path2)
        wb1 = xw.Book(path2)

        wb1.sheets['output'].api.Visible = True

        # Colando informações de DRE + Margens dos cenários --------------------------------------------
        sheet2 = wb1.sheets['output2']
        sheet2.range('A2').options(transpose=True).value = np.array(self.lista_margens) / 10**6
        sheet2.range('O2').options(transpose=True).value = list(self.melhor_dre.values())
        sheet2.range('O18').options(transpose=True).value = list(self.pior_dre.values())
        if self.atual == 1:
            sheet2.range('O34').options(transpose=True).value = list(self.atual_dre.values())
        
        sheet2.range('R17').value = self.perda_volume_melhor
        sheet2.range('R33').value = self.perda_volume_pior
        sheet2.range('R49').value = self.perda_volume_atual

        # construção da tabela de produção no excel - MELHOR CENÁRIO -----------------------------------
        col = 19
        for imes, mes in enumerate(self.matriz_melhor_combinacao):
            lin = 2
            for istep,  step in enumerate(mes):
                sheet2.cells(lin, col).value = step[0]
                sheet2.cells(lin+1, col).value = step[1]
                sheet2.cells(lin+2, col).value = step[2]
                lin +=3
            
            col +=1
        
        # construção da tabela de produção no excel - PIOR CENÁRIO -------------------------------------
        col = 19
        for imes, mes in enumerate(self.matriz_pior_combinacao):
            lin = 18
            for istep,  step in enumerate(mes):
                sheet2.cells(lin, col).value = step[0]
                sheet2.cells(lin+1, col).value = step[1]
                sheet2.cells(lin+2, col).value = step[2]
                lin +=3
            
            col +=1

        # fechando arquivo ------------------------------------------------------------------------------
        wb1.sheets['output'].activate()
        wb1.save()
        wb1.close()
        excel_app.quit    


    def executar_monte_carlo(self):
        """        
        Gerar a quantidade de cenário definida pelo usuário no parâmetro do método.
        A cada novo cenário gerado, comparar se o resultado de margem melhorou.
        Se sim, salvar este novo cenário para depois fazer parte do arquivo de output.
        """

        self.matriz_melhor_combinacao= np.array([])
        self.matriz_pior_combinacao= np.array([])
        self.lista_margens= []
        self.melhor_dre= {}
        self.pior_dre= {}
        self.perda_volume_pior = 0
        self.perda_volume_melhor = 0
        self.margem_Max= -100000
        self.margem_Min= 1000000000000
        numPontosBusca = self.buscas

        # desabilite caso queira salvar cada cenário simulado
        # ver também anotação na função gerar_output_file()
        # self.lista_cenarios = []   
        
        for i in range(0,numPontosBusca):
            self.gerar_combinacao()
            self.calcular_margem()

            margem = self.dre['margem_usd']
            self.lista_margens.append(margem)
            # desabilite caso queira salvar cada cenário simulado
            # ver também anotação na função gerar_output_file()
            # self.lista_cenarios.append(self.tabela_decisao)

            if margem > self.margem_Max:
                self.matriz_melhor_combinacao = self.tabela_decisao
                self.melhor_dre = self.dre
                self.perda_volume_melhor = self.perda_volume_entrega
                self.margem_Max = margem
            
            if margem < self.margem_Min:
                self.matriz_pior_combinacao = self.tabela_decisao
                self.pior_dre = self.dre
                self.perda_volume_pior = self.perda_volume_entrega
                self.margem_Min = margem

        if self.atual == 1:
            self.simular_atual()
        
        self.gerar_output_file()


