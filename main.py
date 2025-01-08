import random
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, redirect, url_for, session
import secrets
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)


def combineDataframes(df1: pd.DataFrame, df2: pd.DataFrame):
    return pd.concat([df1, df2], ignore_index=True)


def appendRowToDataframe(df1: pd.DataFrame, df2_dict: dict):
    return pd.concat([df1, pd.DataFrame(df2_dict, index=[0])], ignore_index=True)


def getDfs():
    # Load data from csv file
    moviesCharacteristics_df = pd.read_csv('data/tagsngenres.csv')

    # we want to drop any rows with NaN values in title OR movieId column
    moviesCharacteristics_df.dropna(subset=['title', 'movieId'], inplace=True)

    # we want to drop column tags
    moviesCharacteristics_df.drop(columns=['tags'], inplace=True)

    # we want to drop duplicate rows
    moviesCharacteristics_df.drop_duplicates(inplace=True)

    # replace "(no genres listed)" with null in genres column
    moviesCharacteristics_df['genres'].replace("(no genres listed)", None, inplace=True)

    # we want to group them by movieId
    moviesCharacteristics_df = moviesCharacteristics_df.groupby('movieId').agg(
        {'title': 'first', 'genres': 'first'}).reset_index()

    # we want to make new columns with one hot encoding for genres
    moviesCharacteristics_df = moviesCharacteristics_df.join(
        moviesCharacteristics_df['genres'].str.get_dummies('|'))

    # we want to drop column genres
    moviesCharacteristics_df.drop(columns=['genres'], inplace=True)

    genresColumnsNames = moviesCharacteristics_df.columns.drop(['movieId', 'title'])

    usersRatings_df = pd.read_csv('data/ratings_user-item.csv')

    # we want to drop any rows with NaN values in any column
    usersRatings_df.dropna(inplace=True)

    # we want to drop duplicate rows
    usersRatings_df.drop_duplicates(inplace=True)

    # we want to drop any rows with common userId and movieId AND different rating (remove both because the data are unreliable)
    usersRatings_df.drop_duplicates(subset=['userId', 'movieId'], keep=False, inplace=True)

    # we want to drop any movieId row whose id isn't in moviesCharacteristics_df
    usersRatings_df = usersRatings_df[usersRatings_df['movieId'].isin(moviesCharacteristics_df['movieId'])]

    # we want to drop any row whose rating is not between 0.5 and 5
    usersRatings_df = usersRatings_df[(usersRatings_df['rating'] >= 0.5) & (usersRatings_df['rating'] <= 5)]

    # we want to multiply the ratings by 2 (1-10)
    usersRatings_df['rating'] = usersRatings_df['rating'] * 2

    return moviesCharacteristics_df, usersRatings_df, genresColumnsNames


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
moviesCharacteristics_df, usersRatings_df, genresColumnsNames = getDfs()
tempFlag = False


def addRatingToMovie(userId: int, movieId: int, rating: float):
    global usersRatings_df

    if not usersRatings_df[(usersRatings_df['userId'] == userId) & (usersRatings_df['movieId'] == movieId)].empty:
        print("User " + str(userId) + " has changed their rating in movie " + str(movieId) + " from" + str(
            usersRatings_df.loc[
                (usersRatings_df['userId'] == userId) & (usersRatings_df['movieId'] == movieId), 'rating'].values[
                0]) + " to " + str(rating))
        usersRatings_df.loc[
            (usersRatings_df['userId'] == userId) & (usersRatings_df['movieId'] == movieId), 'rating'] = rating
    else:
        print("User " + str(userId) + " rated movie " + str(movieId) + " with rating " + str(rating))
        usersRatings_df.loc[-1] = [userId, movieId, rating]
        usersRatings_df.index = usersRatings_df.index + 1
        usersRatings_df = usersRatings_df.sort_index()
    return


def recommendMovies(wantedUserId: int, allUserRatings: pd.DataFrame, allMovies: pd.DataFrame, a: float = 4,
                    b: float = 15,
                    numberOfMovies: int = 50, minSimilarity: float = 0.6) -> pd.DataFrame:
    def calculate_similarity() -> pd.DataFrame:
        nonlocal allUserRatings, wantedUserId

        print("\t\tCalculating similarity for user with id = " + str(wantedUserId) + " ...")

        moviesIdsThatWantedUserHasRated = allUserRatings[allUserRatings['userId'] == wantedUserId][
            'movieId'].unique()
        wantedUserRatings = allUserRatings[allUserRatings['userId'] == wantedUserId]['rating'].tolist()

        similarityDf = pd.DataFrame(columns=['userId', 'similarity'])
        for currentUserId in allUserRatings[allUserRatings['userId'] != wantedUserId]['userId'].unique():

            allCurrentMovieRatings = list()
            for currentMovieId in moviesIdsThatWantedUserHasRated:
                currentUserRatingDf = allUserRatings[
                    (allUserRatings['userId'] == currentUserId) & (allUserRatings['movieId'] == currentMovieId)]

                if currentUserRatingDf.empty:
                    allCurrentMovieRatings.append(0)
                else:
                    allCurrentMovieRatings.append(currentUserRatingDf['rating'].values[0])

            similarity = cosine_similarity(np.array(wantedUserRatings).reshape(1, -1),
                                           np.array(allCurrentMovieRatings).reshape(1, -1))[0][0]

            similarityDf = appendRowToDataframe(similarityDf, {'userId': currentUserId, 'similarity': similarity})

        similarityDf['userId'] = similarityDf['userId'].astype(int)

        print("\t\tSimilarity calculated. Max similarity is " + str(similarityDf['similarity'].max()))
        return similarityDf

    print("\tReceive request for recommend " + str(numberOfMovies) + " movies")
    similarity_matrix = calculate_similarity()
    print("\tCount of users with similarity > " + str(minSimilarity) + " are " + str(
        similarity_matrix[similarity_matrix['similarity'] > minSimilarity].shape[0]))

    def getContextBasedRecommendation() -> pd.DataFrame:
        global genresColumnsNames
        nonlocal allUserRatings, allMovies, wantedUserId

        print("\t\tCalculating context based recommendations")

        wantedUserRatings = allUserRatings[allUserRatings['userId'] == wantedUserId].copy()
        wantedUserRatings.drop(columns=['userId'], inplace=True)

        wantedUserRatings = pd.merge(wantedUserRatings, allMovies, on='movieId')

        # genresColumnsNames = wantedUserRatings.columns.drop(['movieId', 'title', 'rating'])
        wantedUserRatings[genresColumnsNames] = wantedUserRatings[genresColumnsNames].apply(
            lambda value: value * wantedUserRatings['rating'])
        genresRating = wantedUserRatings[genresColumnsNames].sum(axis='rows').to_frame(name='rating')

        # Normalize the column and make the sum equal to 1
        scaler = MinMaxScaler(feature_range=(0, 1))  # Specify the range explicitly
        normalized_values = scaler.fit_transform(genresRating.values.reshape(-1, 1))
        genresRating['normalized'] = normalized_values / normalized_values.sum()

        moviesWantedUserHasNotRated = allMovies[~allMovies['movieId'].isin(wantedUserRatings['movieId'])].copy()

        for genreName in genresColumnsNames:
            moviesWantedUserHasNotRated[genreName] = moviesWantedUserHasNotRated[genreName].apply(
                lambda value: value * genresRating.loc[genreName]['normalized'])

        # create a new dataframe with the movieId and the sum of the genres columns
        moviesWantedUserHasNotRated['score'] = moviesWantedUserHasNotRated[genresColumnsNames].sum(axis='columns')
        moviesWantedUserHasNotRated['score'] = moviesWantedUserHasNotRated['score'] * 10

        recommendationsWithScore = moviesWantedUserHasNotRated[['movieId', 'score']]
        recommendationsWithScore.sort_values(by=['score'], ascending=False, inplace=True)

        print("\t\tContext based recommendations calculated. Max score is " + str(
            recommendationsWithScore['score'].max()))
        return recommendationsWithScore

    def getCollaborativeRecommendation() -> pd.DataFrame:
        nonlocal allUserRatings, allMovies, wantedUserId, similarity_matrix

        print("\t\tCalculating collaborative recommendations")

        moviesIdsThatWantedUserHasNotRated = \
            allMovies[
                ~allMovies['movieId'].isin(allUserRatings[allUserRatings['userId'] == wantedUserId]['movieId'])][
                'movieId'].unique()

        userRatingsInForeignMoviesWithoutWantedUser = allUserRatings[(allUserRatings['userId'] != wantedUserId) & (
            allUserRatings['movieId'].isin(moviesIdsThatWantedUserHasNotRated))].copy()
        userRatingsInForeignMoviesWithoutWantedUser = pd.merge(userRatingsInForeignMoviesWithoutWantedUser,
                                                               similarity_matrix, on='userId')

        recommendationsWithScore = pd.DataFrame(columns=['movieId', 'score'])
        for movieId in moviesIdsThatWantedUserHasNotRated:
            subset = userRatingsInForeignMoviesWithoutWantedUser[
                userRatingsInForeignMoviesWithoutWantedUser['movieId'] == movieId]
            subset['rating'] = subset['rating'] * subset['similarity']

            ratingsSum = subset['rating'].sum()
            weightsSum = subset['similarity'].sum()

            score = 0 if weightsSum == 0 else ratingsSum / weightsSum

            recommendationsWithScore = appendRowToDataframe(recommendationsWithScore,
                                                            {'movieId': movieId, 'score': score})

        recommendationsWithScore['movieId'] = recommendationsWithScore['movieId'].astype(int)
        recommendationsWithScore.reset_index(drop=True, inplace=True)
        recommendationsWithScore.sort_values(by=['score'], ascending=False, inplace=True)

        print("\t\tCollaborative recommendations calculated. Max score is " + str(
            recommendationsWithScore['score'].max()))
        return recommendationsWithScore

    similarUsersList = similarity_matrix[similarity_matrix['similarity'] >= minSimilarity]['userId'].tolist()

    collaborativeMovies = getCollaborativeRecommendation()

    collaborativeMoviesToAddPercentage = max(0.0, min(a * len(similarUsersList) + b, 100))
    collaborativeMoviesToAdd = round(collaborativeMoviesToAddPercentage * numberOfMovies / 100)

    print("\tAdding " + str(collaborativeMoviesToAdd) + " movies from collaborative filtering (" + str(
        collaborativeMoviesToAddPercentage) + "%)")
    collaborativeMovies = collaborativeMovies.head(collaborativeMoviesToAdd)
    collaborativeMovies['type'] = 'collaborative-filtering'

    contextBasedMovies = getContextBasedRecommendation()
    contextMoviesToAdd = numberOfMovies - collaborativeMoviesToAdd
    print("\tAdding " + str(contextMoviesToAdd) + " movies from context based filtering (" + str(
        100 - collaborativeMoviesToAddPercentage) + "%)")

    contextBasedMovies = contextBasedMovies[~contextBasedMovies['movieId'].isin(collaborativeMovies['movieId'])]
    contextBasedMovies = contextBasedMovies.head(contextMoviesToAdd)
    contextBasedMovies['type'] = 'context-based'

    recommendedMovies = combineDataframes(collaborativeMovies, contextBasedMovies)
    recommendedMovies['score'] = recommendedMovies['score'] / 2  # make it scale from 0.5 - 5
    recommendedMovies = pd.merge(recommendedMovies, allMovies, on='movieId')
    recommendedMovies.sort_values(by=['score'], ascending=False, inplace=True)
    return recommendedMovies


def evaluateRecommendationSystem(usersRatings: pd.DataFrame, movieCharacteristics: pd.DataFrame, a: float, b: float,
                                 numberOfUsers: int = 25):
    all = pd.DataFrame(columns=['movieId', 'actualRating', 'recommendRating'])

    for currentUserId in random.sample(usersRatings['userId'].unique().tolist(), numberOfUsers):
        currentUserId = int(currentUserId)

        # append in testUserRatings the 80% of the ratings of the current user randomly
        trainUserRatings = usersRatings[usersRatings['userId'] == currentUserId].sample(frac=0.8)
        testUserRatings = usersRatings[(usersRatings['userId'] == currentUserId) & (
                usersRatings['movieId'].isin(trainUserRatings['movieId']) == False)]

        allUserRatingsWithoutTestUserRatings = combineDataframes(
            usersRatings[usersRatings['userId'] != currentUserId].copy(), trainUserRatings)

        recommendedMovies = recommendMovies(wantedUserId=currentUserId,
                                            allUserRatings=allUserRatingsWithoutTestUserRatings,
                                            allMovies=movieCharacteristics,
                                            numberOfMovies=len(movieCharacteristics),
                                            a=a, b=b)
        recommendedMovies = recommendedMovies.rename(columns={'score': 'recommendRating'})
        recommendedMovies = recommendedMovies[['movieId', 'recommendRating']]
        recommendedMovies['movieId'] = recommendedMovies['movieId'].astype(int)

        results = pd.DataFrame(columns=['movieId', 'error', 'actualRating'])
        results['movieId'] = testUserRatings['movieId']
        results['movieId'] = results['movieId'].astype(int)
        results['actualRating'] = testUserRatings['rating'] / 2

        results = results.merge(recommendedMovies, on='movieId')
        results['recommendRating'].fillna(0, inplace=True)

        results['error'] = (results['actualRating'] - results['recommendRating']) ** 2

        all = combineDataframes(all, results)

    MSE = all['error'].mean()
    return MSE


def initializeUser():
    global tempFlag

    if 'currentUserId' not in session:
        session['currentUserId'] = max(usersRatings_df['userId'].tolist()) + 1
    return


@app.route('/error', methods=['GET', 'POST'])
def errorFlask(errorMessage="An error has occurred!"):
    return render_template('error.html', errorMessage=errorMessage)


@app.route('/', methods=['GET'])
def homeScreenFlask():
    global moviesCharacteristics_df, usersRatings_df

    initializeUser()

    moviesIds = moviesCharacteristics_df['movieId'].tolist()
    moviesTitles = moviesCharacteristics_df['title'].tolist()
    moviesGenres = moviesCharacteristics_df.iloc[:, 2:].apply(lambda row: ', '.join(row.index[row.astype(bool)]),
                                                              axis=1).tolist()

    return render_template('allMovies.html', moviesIds=moviesIds, moviesTitles=moviesTitles,
                           moviesGenres=moviesGenres)


@app.route('/addRatingInUser', methods=['POST'])
def addRatingInUserFlask():
    global usersRatings_df

    initializeUser()

    movieId = request.form['movieId']
    rating = request.form['rating']

    if not movieId or not rating:
        return redirect(url_for('homeScreenFlask'))

    addRatingToMovie(session['currentUserId'], int(movieId), float(rating))
    return redirect(url_for('homeScreenFlask'))


@app.route('/movie/<movieId>', methods=['GET'])
def showMovieFlask(movieId):
    global moviesCharacteristics_df, usersRatings_df

    initializeUser()

    movieId = int(movieId)
    movie = moviesCharacteristics_df[moviesCharacteristics_df['movieId'] == movieId]
    movieTitle = movie['title'].values[0]
    movieGenres = movie.iloc[:, 2:].apply(lambda row: ', '.join(row.index[row.astype(bool)]), axis=1).values[0]
    movieAverageRating = round(usersRatings_df[usersRatings_df['movieId'] == movieId]['rating'].mean(), 2)
    movieTotalRatings = len(usersRatings_df[usersRatings_df['movieId'] == movieId])

    # set a variable movieUserRating to 0 if the user has not rated the movie yet or to the user rating if he has rated it
    userMovieRating = usersRatings_df[
        (usersRatings_df['userId'] == session['currentUserId']) & (usersRatings_df['movieId'] == movieId)]
    if len(userMovieRating) > 0:
        movieUserRating = userMovieRating['rating'].values[0]
    else:
        movieUserRating = 0

    return render_template('movie.html', movieId=movieId, movieTitle=movieTitle, movieGenres=movieGenres,
                           movieAverageRating=movieAverageRating, movieUserRating=movieUserRating,
                           movieTotalRatings=movieTotalRatings)


@app.route('/recommendMovies', methods=['GET'])
def recommendMoviesFlask(minMovies=3):
    global moviesCharacteristics_df, usersRatings_df, genresColumnsNames

    initializeUser()

    # if currentUserId has rated less than minMovies movies return error
    if len(usersRatings_df[usersRatings_df['userId'] == session['currentUserId']]) < minMovies:
        return errorFlask(errorMessage="You have to rate at least " + str(
            minMovies) + " movies before access recommendation section!")

    recommendations = recommendMovies(session['currentUserId'], usersRatings_df, moviesCharacteristics_df)
    recommendations['movieId'] = recommendations['movieId'].astype(int)

    moviesIds = recommendations['movieId'].tolist()
    moviesTitles = recommendations['title'].tolist()
    moviesGenres = recommendations[genresColumnsNames].apply(lambda row: ', '.join(row.index[row.astype(bool)]),
                                                             axis=1).tolist()
    moviesType = recommendations['type'].tolist()
    moviesScore = recommendations['score'].apply(lambda x: round(x, 2)).tolist()

    return render_template('recommendedMovies.html', moviesIds=moviesIds, moviesType=moviesType,
                           moviesTitles=moviesTitles, moviesScore=moviesScore, moviesGenres=moviesGenres)


if __name__ == '__main__':
    print("Average number of the number of movies that are rated by users: " + str(
        round(len(usersRatings_df) / len(usersRatings_df['userId'].unique()), 2)))

    app.run()
