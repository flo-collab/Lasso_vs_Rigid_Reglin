import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt 
import numpy as np

raw_data = pd.read_csv('prostate_dataset.txt',delimiter='\t')

'''
#apercu du jeu de données
print(raw_data.head())
print(raw_data.describe())
print(raw_data.info())
print(raw_data.shape)
print(list(raw_data.columns.values))
'''

# la dernière colonne du dataset est un booléen associé à la présence du cancer (=val. discrete)
# on ne la recupere pas tout de suite 

# trainset  de 0 à 60, testset 60 à 97
X_train = raw_data.iloc[:60, 1:-3]
y_train = raw_data.iloc[:60, -2]
X_test = raw_data.iloc[60:, 1:-3]
y_test = raw_data.iloc[60:, -2]

#print(X_train)

# on crée le modele
linear = linear_model.LinearRegression()

# on entraine le modele
linear.fit(X_train, y_train)

# on prend l'erreur de norme 2 sur le dataset test comme baseline
baseline_error = np.mean((linear.predict(X_test) - y_test)**2)

print("L'erreur quadratique pour la Regression Classique est : ", baseline_error)
print('\n -------------------------------------\n' )
# On passe a la regression rigide : 
print('On passe a la regression rigide :' )
# On doit trouver un coef de regularisation alpha adapté,
# l'objectif est de biaiser un peu la prediction pour diminuer l'erreur standard  

# Test de differentes valeurs pour alpha
n_alphas = 200
alphas = np.logspace(-5, 5, n_alphas)

# On test la reg.rigide avec les != alphas
# On recupere les poids des coef et l'erreur quadratique pour chaque alpha

ridge = linear_model.Ridge()

coefs = []
errors = []

for a in alphas:
	ridge.set_params(alpha=a)
	ridge.fit(X_train,y_train)
	coefs.append(ridge.coef_)
	errors.append([baseline_error, np.mean((ridge.predict(X_test) - y_test) ** 2 )])

# Preparation du graph 1 :
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()



# preparation graph 2:
ax = plt.gca()

ax.plot(alphas, errors)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Ridge EQM as a function of the regularization ')
plt.axis('tight')
plt.show()

print ("la valeur minimum d'erreur trouvee pour la regression rigide est :", min(errors))
print('-------------------------')

print ('On passe a Lasso :')

# On teste plusieurs parametres:
n_alphas_l = 300
alphas_l = np.logspace(-5, 1, n_alphas)
lasso = linear_model.Lasso(fit_intercept=False)

coefs_l = []
errors_l = []
for a in alphas_l:
    lasso.set_params(alpha=a)
    lasso.fit(X_train, y_train)
    coefs_l.append(lasso.coef_)
    errors_l.append([baseline_error, np.mean((lasso.predict(X_test) - y_test) ** 2)])

#Graph 1 lasso
ax = plt.gca()

ax.plot(alphas_l, coefs_l)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.title('Lasso coefficients as a function of the regularization')
plt.ylabel('weights')
plt.axis('tight')
plt.show()

#Graph 2 Lasso
ax = plt.gca()

ax.plot(alphas_l, errors_l)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('error')
plt.title('Lasso EQM as a function of the regularization ')
plt.axis('tight')
plt.show()

print ("la valeur minimum d'erreur trouvee pour Lasso est :", min(errors_l[1]))
