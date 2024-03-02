from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier

# 1) Identificar as caracteristicas (1 - sim / 0 - não)
# o ser tem corpo físico?
# o ser dá calafrios?
# o ser é invocado através de um tabuleiro?

espirito1 = [0, 1, 1]
espirito2 = [0, 0, 1]
espirito3 = [0, 0, 1]

humano1 = [1, 1, 0]
humano2 = [1, 0, 0]
humano3 = [1, 1, 1]

treino_x = [espirito1, espirito2, espirito3, humano1, humano2, humano3]

# 0 - humano 
# 1 - espirito

treino_y = [1, 1, 1, 0, 0, 0]

#inicializar o modelo
modelo = LinearSVC()

#treinar o modelo
modelo.fit(treino_x, treino_y)

ser_misterioso = [1, 1, 1]

result = modelo.predict([ser_misterioso])

if result == 0:
  print ("Ser humano!")
else:
  print ("Ser espiritual!")

ser_misterioso1 = [1, 1, 1]
ser_misterioso2 = [1, 1, 0]
ser_misterioso3 = [0, 1, 1]

teste_x = [ser_misterioso1, ser_misterioso2, ser_misterioso3]

previsoes = modelo.predict(teste_x)

for i in range(len(previsoes)):
  if previsoes[i] == 0:
    print ("Ser humano!")
  else:
    print ("Ser espiritual!")

teste_y = [0, 1, 1]

acuracia = accuracy_score(teste_y, previsoes)

print ("A acurácia do modelo de aprendizado de máquina é de: {}%".format(acuracia*100))

# desenvolvendo o código de um classificador burro - não guidado

salva_acuracia_dummy = []
testes = 300

for i in range(testes):

  dummy = DummyClassifier(strategy="uniform")

  dummy.fit(teste_x, teste_y)

  predicao_dummy = dummy.predict(teste_x)

  acuracia_dummy = accuracy_score(teste_y, predicao_dummy)

  salva_acuracia_dummy.append(acuracia_dummy)

media = 0
for i in range(testes):
  media = media + salva_acuracia_dummy[i]

print ("A média da acurácia do algoritmo dummy é: {}%".format((media/testes)*100))