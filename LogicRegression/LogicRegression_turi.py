import turicreate as tc

movies = tc.SFrame('../_Mock/IMDB_Dataset.csv')

movies['words'] = tc.text_analytics.count_words(movies['review'])

model = tc.logistic_classifier.create(movies, features=['words'], target='sentiment')

movies['prediction'] = model.predict(movies, output_type='probability')

print(movies.sort('prediction')[-1])
print(movies.sort('prediction')[0])