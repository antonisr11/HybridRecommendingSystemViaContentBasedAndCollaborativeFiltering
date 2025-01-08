## Problem description

In this project, we were asked to develop a recommendation system based on a hybrid approach utilizing content-based techniques and collaborative filtering techniques.

To develop this system, we relied on a dataset obtained from Kaggle that contains cinema films.

After collecting the data, we trained our model to be able to provide each user with recommended movies based on their preferences. The process and techniques followed are detailed below.

In addition, an application was developed that represents the data of the dataset with information that we chose to present, as well as the results of the suggestions provided to the user.

For the implementation, we provided a basis for both the back-end and the front-end so that someone with no programming background can handle it.

> Note 1: This project isn't time- and space-efficient. It is implemented in a read-code-friendly way so someone can understand how content-based and collaborative filtering are working.

> Node 2: This project does not connect to any database; the whole dataset is read from csvs and preprocessed each time the application is started.

## Dataset 

We used [MovieRatings from Ranja Sarkar](https://www.kaggle.com/datasets/ranja7/movieratingsbyusers)

- This dataset contains a multitude of films and information related to them, such as their categorization based on genre, their ratings by users who have watched them, information about the date of production, keywords, and more.

## How are recommendations made?

Our implementation is a hybrid model based on two recommending approaches: content-based and collaborative filtering.

When the recommendMovies function is called, first the calculate_similarity function is called, which returns the similarity of each user with the user we want to recommend movies to.

A problem that could arise is that collaborative filtering requires the user to have rated at least one movie. We solved this problem by not allowing the user to request the movie prediction unless they have rated at least three movies first.

Furthermore, it must be determined how many of the films will be collaborative and how many will be context-based out of the total.

- For instance, if 50 movies are to be recommended, the algorithm we have implemented should decide how many of the 50 movies that will be recommended will be collaborative and how many will be context-based.

To find the percentage of collaborative films, the following formula is used: Y=ax+b, with Y being the percent rate of the collaborative movies, x the total number of users that have at least 0.6 similarity with the current user, a = 4 and b = 15.

- In the example with the 50 movies, if the users with similarity > 0.6 are 10, the percentage of collaborative films will be Y=4*10+15 = 55%, so, out of the 50 films, 28 will be collaborative and the remaining 22 context-based.

## How our model was built

The a and b in the formula Y=ax+b are calculated through trial and error.

![1](https://github.com/user-attachments/assets/df00252d-6d97-4567-9a50-60026faee150)

## Screenshots of the site
#### Main screen

When running the application, the screen randomly shows all the movies that are in the csv file. On the screen, we can see the title of the movie, the year of its release, the genres for this movie, and a rate button where the user can rate this movie (0.5–5 stars).

![2](https://github.com/user-attachments/assets/a8b989d7-35b0-4632-b76c-a8d11cf4192c)

#### Rate a Movie

By clicking on the rate button, you will be redirected to this screen.

![3](https://github.com/user-attachments/assets/7b808dcb-a10b-4f76-9289-a9f85ae48ba0)

On this page, the information about each movie is displayed again, along with the average rating based on the ratings of all users. Moreover, we can see the total number of users who have rated it.

Users can rate the movie by pressing the number of stars they desire, from 0.5 to 5 stars. By pressing the button 'Save that rate!' the score is saved. Users can change their rate if they change their mind.

#### Show recommendations

Returning to the home page and selecting the 'Show recommendations' button, our application redirects us to a new page, where the app will recommend the best movies for the user based on our hybrid model.

Access to this page is not allowed if the user has not rated at least three movies.

![4](https://github.com/user-attachments/assets/6cba6a8d-3c78-4c2a-8146-b76c4d11eb10)

Movies are sorted by their score (the score is the rate our model predicts the user will give). The type column shows the approach based on which the movie is recommended to the user.

![5](https://github.com/user-attachments/assets/814572e6-a0b7-45ee-9527-6595f0ccb6ed)
