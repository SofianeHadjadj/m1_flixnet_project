# coding: utf-8

# On déclare en tout premier utf-8 afin de pouvoir utiliser les caractères spéciaux

from numpy import *
from scipy import optimize

# Les bibliothèques NumPy et SciPy permettent d’effectuer des calculs numériques avec Python. 
# Elles introduisent une gestion facilitée des tableaux de nombres.


# On définie le nombre de film de notre catalogue dans la variable : num_movies

num_movies = 10

# On définie le nombre d'utilisateurs utilisant Flixnet dans la variable : num_users

num_users = 5


# On crée une matrice de 10 X 5 appelée "ratings" qui  contient les notes attribuées par les utilisateurs de Flixnet aux films de notre catalogue 
# La matrice d'évaluatiob fait 10 X 5 (nb_users * nb_films)
# On initialise la matrice avec des valeurs au hasard

ratings = random.randint(11, size = (num_movies, num_users))

#debug print ratings

# La variable 'did_rate' sert à determiner si l'utilisateur à noter un film ou non
# Si 'did_rate' = 0 le spectateur n'a pas noté le film, sinon il a noté le film

did_rate = (ratings != 0) * 1

#debug print did_rate


# On récupère la dimension des matrices avec la propriété 'shape'

ratings.shape
did_rate.shape

# Cas d'un utilisateur spécifique nommé : flixUser1

#_______________________________________________________________

# L'utilisateur flixUser1 fait plusieur évaluations
# Nous créons tableau de 10 colonne pour stocker ses notes

flixUser1_ratings = zeros((num_movies, 1))

#debug print flixUser1_ratings

#debug print flixUser1_ratings[9] 


# flixUser1 note 3 film

flixUser1_ratings[0] = 8 # Il attribut la note de 8 au film à la position 0
flixUser1_ratings[4] = 7 # Il attribut la note de 7 au film à la position 4
flixUser1_ratings[7] = 3 # Il attribut la note de 3 au film à la position 7

print "\nNotes attribuées par flixUser1 : \n"

print flixUser1_ratings


# Update ratings and did_rate

ratings = append(flixUser1_ratings, ratings, axis = 1)
did_rate = append(((flixUser1_ratings != 0) * 1), did_rate, axis = 1)


#debug print ratings

ratings.shape

#did_rate

#debug print did_rate

did_rate.shape


def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_movies, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_movies): 
		# Récupération de tous les index a 1
        idx = where(did_rate[i] == 1)[0]
        # Calcule de la cote moyenne d'un film à partir de note attribuéé par l'utilisateur
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean


# Normalize ratings

ratings, ratings_mean = normalize_ratings(ratings, did_rate)


# Update some key variables now

num_users = ratings.shape[1]
num_features = 3

X = array([[1, 2], [1, 5], [1, 9]])
Theta = array([[0.23], [0.34]])


#debug print X
#debug print Theta

Y = X.dot(Theta)
#debug print Y

# Initialize Parameters theta (user_prefs), X (movie_features)

movie_features = random.randn( num_movies, num_features )

'''
 __________________________________________
|           | 		  |			|		   |
|   User    | Comedie | Fiction | Aventure |
|___________|_________|_________|__________|
| 			|		  |			|		   |
| flixUser1 |	4.5	  |	  4.9	|	3.6	   |
|			|		  |			|		   |
'''

user_prefs = random.randn( num_users, num_features )

'''
 __________________________________________
|           | 		  |			|		   |
|   Film    | Comedie | Fiction | Aventure |
|___________|_________|_________|__________|
| 			|		  |			|		   |
| Boxers	|	0.8	  |	  0.5	|	0.4	   |
|			|		  |			|		   |
'''

initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]



#debug print movie_features

#debug print user_prefs

#debug print initial_X_and_theta

initial_X_and_theta.shape

movie_features.T.flatten().shape

user_prefs.T.flatten().shape

initial_X_and_theta



def unroll_params(X_and_theta, num_users, num_movies, num_features):
	# Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
	# --------------------------------------------------------------------------------------------------------------
	# Get the first 30 (10 * 3) rows in the 48 X 1 column vector
	first_30 = X_and_theta[:num_movies * num_features]
	# Reshape this column vector into a 10 X 3 matrix
	X = first_30.reshape((num_features, num_movies)).transpose()
	# Get the rest of the 18 the numbers, after the first 30
	last_18 = X_and_theta[num_movies * num_features:]
	# Reshape this column vector into a 6 X 3 matrix
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply by did_rate because we only want to consider observations for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	# wrap the gradients back into a column vector 
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	# '**' means an element-wise power
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization


# import these for advanced optimizations (like gradient descent)

# regularization paramater

reg_param = 30


# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta, 								args=(ratings, did_rate, num_users, num_movies, num_features, reg_param), 								maxiter=100, disp=True, full_output=True ) 

cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]

# unroll once again

movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)


#debug print movie_features


#debug print user_prefs

# Make some predictions (movie recommendations). Dot product

all_predictions = movie_features.dot( user_prefs.T )


#debug print all_predictions

# add back the ratings_mean column vector to my (our) predictions

predictions_for_flixUser1 = all_predictions[:, 0:1] + ratings_mean

print "\nRecommendation pour flixUser1 : \n"

print predictions_for_flixUser1

#debug print flixUser1_ratings