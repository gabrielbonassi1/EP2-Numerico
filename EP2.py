import numpy as np
import matplotlib.pyplot as plt

# ******************************************
# *    Autores: Gabriel Lujan Bonassi      *
# *             Gabriel Praca              *
# *        EP2 de MAP3121 - 2021           *
# ******************************************


float_formatter = "{:.5f}".format # Configuracao de exibicao do python
np.set_printoptions(formatter={'float_kind':float_formatter}, threshold=np.inf, linewidth=500) # Configuracao de exibicao do python
#Matriz exemplo:
#A = np.array(([2, -1, 1, 3], [-1, 1, 4, 2], [1, 4, 2, -1], [3, 2, -1, 1]))
passou = False

while passou == False:
    try:
        inp = str(input("Digite qual arquivo de input quer usar, entre input-a, input-b ou input-c: "))
        if inp == 'input-c':
            barras = np.genfromtxt(inp, dtype=float, skip_header=2)
            f = np.genfromtxt(inp, dtype=float, skip_footer=len(barras))
        else:
            A = np.genfromtxt(inp, skip_header=1, dtype=float)
        passou = True
    except:
        print("Ops! voce digitou um nome errado. \n Digite 'input-a', 'input-b' ou 'input-c' sem as aspas")


if inp == 'input-c':
    nnos = int(f[0, 0])
    nnosnfixos = int(f[0, 1])
    nbarras = int(f[0, 2])
    ro = f[1, 0]
    area = f[1, 1]
    modEl = f[1, 2] * 1e+9

    MatrizK = np.zeros((nbarras, nbarras))
    for o in range(nbarras):
        if barras[o, 0] != 11 and barras[o, 1] != 13:
            if barras[o, 0] != 12 and barras[o, 1] != 14:
                Cij = np.cos(barras[o, 2])
                Sij = np.sin(barras[o, 2])
                i = int(barras[o, 0])
                j = int(barras[o, 1])
                AEL = (area * modEl)/barras[o, 3]
                MatrizK[2 * (i - 1), 2 * (i - 1)] = MatrizK[2 * (i - 1), 2 * (i - 1)] + (AEL * (Cij ** 2))

                MatrizK[2 * i, 2 * (i - 1)] = MatrizK[2 * i, 2 * (i - 1)] + (AEL * Cij * Sij)
                MatrizK[2 * (j - 1), 2 * (i - 1)] = MatrizK[2 * (j - 1), 2 * (i - 1)] - (AEL * (Cij ** 2))
                MatrizK[2 * j, 2 * (i - 1)] = MatrizK[2 * j, 2 * i - 1] - (AEL * (Cij * Sij))

                MatrizK[2 * (i - 1), 2 * i] = MatrizK[2 * (i - 1), 2 * i] + (AEL * (Cij * Sij))
                MatrizK[2 * (i - 1), 2 * (j - 1)] = MatrizK[2 * (i - 1), 2 * (j - 1)] - (AEL * (Cij ** 2))
                MatrizK[2 * (i - 1), 2 * j] = MatrizK[2 * (i - 1), 2 * j] - (AEL * (Cij * Sij))

                MatrizK[2 * i, 2 * i] = MatrizK[2 * i, 2 * i] + (AEL * (Sij ** 2))

                MatrizK[2 * (j - 1), 2 * i] = MatrizK[2 * (j - 1), 2 * i] - (AEL * (Cij * Sij))
                MatrizK[2 * j, 2 * i] = MatrizK[2 * j, 2 * i] - (AEL * (Sij ** 2))

                MatrizK[2 * i, 2 * (j - 1)] = MatrizK[2 * i, 2 * (j - 1)] - (AEL * (Cij * Sij))
                MatrizK[2 * i, 2 * j] = MatrizK[2 * i, 2 * j] - (AEL * (Sij ** 2))

                MatrizK[2 * (j - 1), 2 * (j - 1)] = MatrizK[2 * (j - 1), 2 * (j - 1)] + (AEL * (Cij ** 2))

                MatrizK[2 * (j - 1), 2 * j] = MatrizK[2 * (j - 1), 2 * j] + (AEL * (Cij * Sij))
                MatrizK[2 * j, 2 * (j - 1)] = MatrizK[2 * j, 2 * (j - 1)] + (AEL * (Cij * Sij))

                MatrizK[2 * j, 2 * j] = MatrizK[2 * j, 2 * j] + (AEL * (Sij ** 2))
    for i in range(4):
        MatrizK = np.delete(MatrizK, len(MatrizK) - 1, 0)
        MatrizK = np.delete(MatrizK, len(MatrizK) - 1, 1)
    #Tivemos que calcular os valores das massas pra cada no na mao :(
    Mlista = [13315.4, 13315.4, 13315.4, 13315.4, 13315.4, 13315.4, 26630.9, 26630.9, 26630.9, 26630.9, 13315.4, 13315.4, 7800, 7800, 32146.3, 32146.3, 32146.3, 32146.3, 7800, 7800, 24706.6, 24706.6, 24706.6, 24706.6]
    MatrizM = np.zeros((nnosnfixos * 2, nnosnfixos * 2))
    for i in range(len(MatrizM)):
        MatrizM[i, i] = Mlista[i]
    """
    MatrizM = np.zeros((nnosnfixos * 2, nnosnfixos * 2))
    Laux = 0
    iter1 = 0
    i = 0
    aux10 = 0
    while iter1 < (nnos * 2):
        Laux += barras[iter1, 3]
        if barras[iter1, 1] == 10:
            aux10 += barras[iter1, 3]
        iter1 += 1
        aux = barras[iter1, 0]
        while aux == barras[iter1 - 1, 0] and iter1 < len(barras):
            Laux += barras[iter1, 3]
            if barras[iter1, 1] == 10:
                aux10 += barras[iter1, 3]
            iter1 += 1
            if iter1 < len(barras):
                aux = barras[iter1, 0]
        MatrizM[i, i] = ro * area * 0.5 * Laux
        MatrizM[i + 1, i + 1] = ro * area * 0.5 * Laux
        Laux = 0
        i += 2
        if aux == 11:
            MatrizM[i, i] = ro * area * 0.5 * aux10
            MatrizM[i + 1, i + 1] = ro * area * 0.5 * aux10
            i += 2
    """
    MatrizMInv = np.zeros((nnosnfixos * 2, nnosnfixos * 2))
    for i in range(len(MatrizMInv)):
        MatrizMInv[i, i] = 1 / MatrizM[i, i]

    Ktil = np.matmul(np.sqrt(MatrizMInv), MatrizK)
    Ktil = np.matmul(Ktil, np.sqrt(MatrizMInv))
    A = np.array(Ktil)
    lenA = int((len(A) / 2) + 1)
    for i in range(lenA):
        if A[i, i] == 0:
            A = np.delete(A, i, 0)
            A = np.delete(A, i, 1)
    print("Matriz K~")
    print(A)

#Algoritmo Householder
n = len(A)
naux = len(A)
w = np.zeros((n))  # vetor w
x = np.zeros((n - 1, n))  # declarando o vetor x
wxi = np.zeros((n - 1))
v = np.array(A[0])
vmenosum = np.zeros((n - 1))
aux = np.zeros((n, n))
rowDelete = np.zeros((naux-2, naux))
Afinal = np.empty((naux, naux), dtype=float)
I = np.identity(n)
Iaux = np.identity(n)
for i in range(naux-2):
    # Hwi*A
    for i1 in range(n):
        if i1 != 0:
            vmenosum[i1 - 1] = np.array(v[i1])
    alfa = np.sqrt(np.sum(vmenosum ** 2))  # Raiz da soma dos quadrados
    for i1 in range(n):
        if i1 == 0:
            w[i1] = 0
        elif i1 == 1:
            w[i1] = v[i1] - alfa
        else:
            w[i1] = v[i1]
    ww = np.vdot(w, w)
    vlinha = np.subtract(v, w)
    for i1 in range(n - 1):
        x[i1] = np.array(A[i1 + 1])
    for i1 in range(n - 1):
        wxi[i1] = np.vdot(w, x[i1])
    AHwi = np.zeros((n, n))
    for i1 in range(n):
        if i1 == 0:
            AHwi[i1] = vlinha
        else:
            aux = (2 * wxi[i1 - 1] / ww) * w
            AHwi[i1] = np.subtract(x[i1 - 1], aux)
    # Hwi*A*Hwi
    wxi = np.zeros((n - 1))
    for i1 in range(n - 1):
        x[i1] = np.array(np.transpose(AHwi)[i1 + 1])
    for i1 in range(n - 1):
        wxi[i1] = np.vdot(w, x[i1])
    HwiAHwi = np.zeros((n, n))
    for i1 in range(n):
        if i1 == 0:
            HwiAHwi[i1] = vlinha
        else:
            aux = (2 * wxi[i1 - 1] / ww) * w
            HwiAHwi[i1] = np.subtract(x[i1 - 1], aux)
    #I*Hwi
    xaux = np.zeros((naux - 1, naux))
    wxi = np.zeros((naux - 1))
    for i1 in range(naux - 1):
        xaux[i1] = np.array(np.transpose(I)[i1 + 1])
    if len(w) < naux:
        for i1 in range(naux-len(w)):
            w = np.insert(w, 0, 1, axis=0)
    for i1 in range(naux - 1):
        wxi[i1] = np.vdot(w, xaux[i1])
    IHwi = np.zeros((naux, naux))
    for i1 in range(naux):
        if i1 == 0:
            IHwi[i1] = I[i1]
        else:
            aux = (2 * wxi[i1 - 1] / ww) * w
            IHwi[i1] = np.subtract(xaux[i1 - 1], aux)
    for i2 in range(naux):
        IHwi[i2, 0] = Iaux[i2, 0]

    if i < naux - 2:
        for i3 in range(n):
            rowDelete[i, i3+i] = HwiAHwi[0, i3]
        HwiAHwi = np.delete(HwiAHwi, 0, 0)
        HwiAHwi = np.delete(HwiAHwi, 0, 1)
        A = np.array(HwiAHwi)
        n = n - 1
        w = np.zeros((n))  # vetor w
        x = np.zeros((n - 1, n))  # declarando o vetor x
        wxi = np.zeros((n - 1))
        v = np.array(A[0])
        vmenosum = np.zeros((n - 1))
        aux = np.zeros((n, n))
        I = np.array(IHwi)
    if i == naux - 3:
        for i3 in range(naux-2):
            for i4 in range(naux-i3):
                Afinal[i3, i3] = rowDelete[i3, i3]
                Afinal[i3, i3+i4] = rowDelete[i3, i3+i4]
                Afinal[i3+i4, i3] = rowDelete[i3, i3+i4]
        for i3 in range(len(HwiAHwi) - 1):
            for i4 in range(len(HwiAHwi) - 1):
                HwiAHwi[i3, i4 + 1] = HwiAHwi[i3 + 1, i4]
        for i3 in range(len(HwiAHwi)):
            for i4 in range(len(HwiAHwi)):
                Afinal[naux-n+i3, naux-n+i4] = HwiAHwi[i3, i4]
        Ifinal = np.array(I)

print("Matriz tridiagonal simétrica equivalente:")
 # Configuracao de exibicao do python
print(Afinal)
print("Matriz H^t:")
scientific_notation = "{:.3e}".format
with np.printoptions(formatter={'all':scientific_notation}):
    print(Ifinal)

# Algoritmo QR do EP1

iter = 0
iter2 = 0
eps = 0.000001
check = 0
listIt = np.zeros((4))
A = np.array(Afinal)
n = len(A)
Aatt = np.zeros((n, n))


print(f"Matriz A{n}x{n}:")
print(A)
# V1: lista com os auto-valores
V1 = np.zeros((n))
# Aaux: Matriz de teste contendo os valores originais de A
Aaux = np.zeros((n, n))
# Copiando os valores de uma matriz pra outra
for u9 in range(n):
    for c9 in range(n):
        Aaux[u9, c9] = float(A[u9, c9])

for m in range(n):
    while abs(A[n - m - 1, n - m - 2]) > eps:
        Q = np.zeros((n - 1, n, n))
        for b in range(n - 1):
            Q[b] = np.identity(n)
        if iter > 0:
            dk = (A[n - m - 2, n - m - 2] - A[n - m - 1, n - m - 1]) / 2
            mik = A[n - m - 1, n - m - 1] + dk - np.sign(dk) * np.sqrt((dk ** 2) + (A[n - m - 2, n - m - 1] ** 2))
        elif iter == 0:
            mik = 0
        id = np.identity(n)
        mikid = mik * id
        # Asub: matriz usada para a subtracao
        Asub = np.zeros((n, n))
        # Copiando os valores de uma matriz pra outra
        for u4 in range(n):
            for c4 in range(n):
                Asub[u4, c4] = float(A[u4, c4])
        np.subtract(Asub, mikid, out=A)
        for k in range(n - 1):
            # Calculo do C e do S
            Ck = A[k, k] / np.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
            Sk = -A[k + 1, k] / np.sqrt((A[k, k] ** 2) + (A[k + 1, k] ** 2))
            w = 0
            if k != 0:
                Q[k, w + k, w + k] = Ck
                Q[k, w + k, w + k + 1] = Sk
                Q[k, w + k + 1, w + k] = -Sk
                Q[k, w + k + 1, w + k + 1] = Ck
            else:
                Q[k, w, w] = Ck
                Q[k, w, w + 1] = Sk
                Q[k, w + 1, w] = -Sk
                Q[k, w + 1, w + 1] = Ck
            # Fazendo as operacoes pras linhas 1 e 2 (que mudam com cada iteracao)
            for i in range(n):
                Aatt[k, i] = float((A[k, i] * Ck) - (A[k + 1, i] * Sk))
                Aatt[k + 1, i] = float((A[k, i] * Sk) + (A[k + 1, i] * Ck))
            # Copiando as linhas que nao foram modificadas, senao vai ser tudo 0
            for l in range(n - 2):
                # Esse if serve pra nao estourar o vetor (ele tentar alterar valores que excedem o tamanho do vetor)
                if k + l + 2 < n:
                    Aatt[k + l + 2] = A[k + l + 2]
            for u4 in range(n):
                for c4 in range(n):
                    A[u4, c4] = float(Aatt[u4, c4])

        X = np.zeros((n, n, n), dtype=float)
        for u1 in range(n):
            for c1 in range(n):
                X[0, u1, c1] = float(A[u1, c1])
        for i1 in range(n - 1):
            np.matmul(X[i1], Q[i1], out=X[i1 + 1])
        V = np.zeros((n, n, n), dtype=float)
        V[0] = np.array(Ifinal)
        for i2 in range(n - 1):
            np.matmul(V[i2], Q[i2], out=V[i2 + 1])
        for u3 in range(n):
            for c3 in range(n):
                A[u3, c3] = float(X[n - 1, u3, c3])

        # Copiando o vetor com os valores mexidos para um vetor Aadd, para somar com o mi de wilkinson
        Aadd = np.zeros((n, n))
        for u in range(n):
            for c in range(n):
                Aadd[u, c] = float(A[u, c])
        np.add(Aadd, mikid, out=A)
        iter += 1
    V1[n - m - 1] = A[n - m - 1, n - m - 1]
Mauto = np.identity(n)
for s in range(n):
    Mauto[s, s] = V1[s]
Bool = True
if np.all(np.matmul(V[n - 1], Mauto)) == np.all(np.matmul(Aaux, V[n - 1])):
    Bool = True
    print("Autovetores estão corretos")
else:
    Bool = False
listIt[iter2] = iter
for t in range(n):
    print(f"Auto-valor [{t + 1}]:")
    print(np.round(V1[t], 5))
if inp == 'input-c':
    V2 = []
    V1 = np.sqrt(V1)
    for i in range(5):
        V2.append(np.amin(V1))
        V1 = np.delete(V1, np.argmin(V1), 0)
    for t in range(5):
        print(f"[{t + 1}] Menor frequência de vibração")
        print(np.round(V2[t], 5))
print("Auto-vetores:")
with np.printoptions(formatter={'all':scientific_notation}):
    print(V[n - 1])
print(f"Número de iteracoes com deslocamento espectral: {listIt[iter2]}")
iter2 += 1
iter = 0