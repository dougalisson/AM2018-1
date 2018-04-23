# -*- coding: utf-8 -*-
from sklearn import metrics


def salvar_arquivo(it, P, g, s, c, funcao_objetivo, classes, clusters):
    # MUDAR NOME DO ARQUIVO
    with open('resultados_shape_com.txt', 'a') as arq:
        # MUDAR CABEÇALHO
        cabecalho = "########## VIEW SHAPE ##########" + "\n" + "ITERAÇÃO " + str(it) + "\n"
        message = ""
        for i in range(c):
            message = message + "-------------------- CLUSTER " + str(i+1) + " --------------------\n"
            message = message + "Representante: " + str(g[i]) + "\n"
            message = message + "Número de objetos: " + str(len(P[i])) + "\n"
            message = message + "Lista de objetos: " + str(P[i]) + "\n"

        message = message + "----------------------------------------" + "\n"
        message = message + "VETOR DE HIPER-PARÂMETROS: " + str(s) + "\n"
        message = message + "FUNÇÃO OBJETIVO: " + str(funcao_objetivo) + "\n"
        message = message + "ARI: " + str(metrics.adjusted_rand_score(classes, clusters)) + "\n"
        message = message + "----------------------------------------"
        arq.write(cabecalho + message + "\n\n\n")