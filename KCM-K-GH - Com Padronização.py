# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy
from SalvarArquivo import salvar_arquivo
import ReadCSV
from sklearn.preprocessing import StandardScaler


# FUNÇÃO K(s)
def funcao_k(x1, x2):
    soma_k = 0
    for j in range(p):
        dist = math.pow(x1[j] - x2[j], 2)
        soma_k += ((1 / s[j]) * dist)
    exponencial = math.exp((-1 / 2) * soma_k)

    return exponencial


# INICIALIZAÇÃO
def initialization():
    global g
    # INICIALIZAÇÃO DOS REPRESENTANTES
    g = numpy.random.rand(c, p)
    indices = numpy.random.choice(range(n), c, replace=False)
    for indice in range(len(indices)):
        g[indice] = e[indices[indice]]

    # INICIALIZAÇÃO DOS HYPER-PARÂMETROS
    for j in range(p):
        global s
        hyper_param = (math.pow(gamma, 1 / p))
        s.append(1 / hyper_param)

    # AFETAÇÃO INICIAL DOS ITENS
    for k in range(n):
        funcao_obj_inicial = []
        for i1 in range(c):
            funcao_inicial = 2 * (1 - funcao_k(e[k], g[i1]))
            funcao_obj_inicial.append(funcao_inicial)

        for i2 in range(c):
            global P
            if funcao_obj_inicial[i2] == min(funcao_obj_inicial):
                P[i2].append(k)
                break


def representation():
    # CÁLCULO DOS REPRESENTANTES (EQUAÇÃO 14)
    def funcao_prototipo(gi, i):
        soma1 = numpy.zeros(p)
        soma2 = 0
        for k in P[i]:
            valor_k = funcao_k(e[k], gi)
            soma1 = [xk * valor_k for xk in e[k]] + soma1
            soma2 += valor_k
        divisao = soma1 / soma2

        return divisao

    global g
    for i in range(c):
        g[i] = funcao_prototipo(g[i], i)


def computation_width():
    # CÁLCULO DOS HYPER-PARÂMETROS (EQUAÇÃO 16)
    def funcao_hyper(j):
        soma4 = 0
        produto = 1

        # CÁLCULO DO PRODUTÓRIO
        for h in range(p):
            soma2 = 0
            for i1 in range(c):
                soma1 = 0
                for k1 in P[i1]:
                    soma_interna = funcao_k(e[k1], g[i1]) * math.pow((e[k1][h] - g[i1][h]), 2)
                    soma1 += soma_interna
                soma2 += soma1
            produto *= soma2

        # CÁLCULO DO SOMATÓRIO (DENOMINADOR)
        for i2 in range(c):
            soma3 = 0
            for k2 in P[i2]:
                soma_interna = funcao_k(e[k2], g[i2]) * math.pow((e[k2][j] - g[i2][j]), 2)
                soma3 += soma_interna
            soma4 += soma3
        result = (math.pow(gamma, 1 / p) * math.pow(produto, 1 / p)) / soma4

        return result

    global s
    novos_s = []
    for j in range(p):
        novos_s.append(1 / funcao_hyper(j))

    for j in range(p):
        s[j] = novos_s[j]

    prod = 1
    for j in range(p):
        prod *= 1 / s[j]


# CÁLCULO DA FUNÇÃO OBJETIVO FINAL
def funcao_objetivo_final():
    soma2 = 0
    for i in range(c):
        soma1 = 0
        for k1 in P[i]:
            funcao_obj_f = 2 * (1 - funcao_k(e[k1], g[i]))
            soma1 += funcao_obj_f
        soma2 += soma1

    return soma2


path = "image_segmentation.csv"  # CAMINHO PARA O ARQUIVO COM A BASE
# BASE DE DADOS
# MUDAR VIEW: 'completa', 'shape', 'rgb'
e1 = ReadCSV.read_base(path, 'completa')

# PADRONIZAÇÃO DOS DADOS
scaler = StandardScaler()
e = scaler.fit_transform(e1)

unique = ReadCSV.calc_unique(path)
# SUBSTITUIÇÃO DAS CLASSES PELO INTEIRO CORRESPONDENTE (PARA CALCULAR ARI)
classes = ReadCSV.subst_classes(unique, path)

c = len(unique)  # NÚMERO DE CLASSES
n = len(e)  # NÚMERO DE OBJETOS
p = len(e[0])  # NÚMERO DE ATRIBUTOS

# CÁLCULO DA DISTÂNCIA EUCLIDIANA PARA OBTER O SIGMA
dist_euclidiana = []
l = 0
for i in range(n):
    for j in range(n):
        if j > i:
            soma = 0
            for k in range(p):
                distancia = math.pow((e[i][k] - e[j][k]), 2)
                soma += distancia
            dist_euclidiana.append(math.sqrt(float(soma)))
            l += 1

quantil_1 = numpy.percentile(dist_euclidiana, 10)  # QUANTIL 0.1
quantil_9 = numpy.percentile(dist_euclidiana, 90)  # QUANTIL 0.9

sigma = (quantil_1 + quantil_9) / 2
gamma = math.pow((1 / sigma), p)  # GAMMA

# INÍCIO DAS ITERAÇÔES
for it in range(100):
    s = []  # VETOR DE HYPER-PARÂMETROS
    P = [[] for cl in range(c)]  # CLUSTERES

    # INICIALIZAÇÃO
    initialization()

    test = 1
    # INÍCIO DO LOOP
    while test != 0:
        # CÁLCULO DOS REPRESENTANTES
        representation()
        # CÁLCULO DO VETOR DE HYPER-PARÂMETROS
        computation_width()
        test = 0

        for k in range(n):
            funcao_obj = []
            for i1 in range(c):
                # CÁLCULO DA FUNÇÃO OBJETIVO PARA CADA OBJETO EM CADA CLUSTER
                funcao = 2 * (1 - funcao_k(e[k], g[i1]))
                funcao_obj.append(funcao)

            # CLUSTER ATUAL DO OBJETO (ANTES DA NOVA AFETAÇÃO)
            for m in range(c):
                if k in P[m]:
                    cluster_atual = m

            # AFETAÇÃO DO OBJETO AO CLUSTER COM MENOR VALOR DA FUNÇÃO OBJETIVO
            for i2 in range(c):
                if funcao_obj[i2] == min(funcao_obj):
                    # VERIFICAÇÃO SE O OBJETO JÁ ESTAVA EM OUTRO CLUSTER
                    if i2 != cluster_atual:
                        P[cluster_atual].remove(k)
                        P[i2].append(k)
                        test = 1
                    break

    # PARTIÇÃO FINAL DA ITERAÇÃO
    print "PARTIÇÃO:", str(P)

    # VETOR COM OS RESULTADOS (PARA CALCULAR ARI)
    clusters = numpy.zeros(shape=n, dtype=int)
    for i2 in range(c):
        for k in P[i2]:
            clusters[k] = i2

    # SALVAR RESULTADOS DA ITERAÇÃO EM UM .TXT
    salvar_arquivo(it, P, g, s, c, funcao_objetivo_final(), classes, clusters)
