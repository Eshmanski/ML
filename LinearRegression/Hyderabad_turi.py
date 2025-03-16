import turicreate as tc

data = tc.SFrame('../_Mock/Hyderabad.csv')
model = tc.linear_regression.create(data, target='Price')

house = tc.SFrame({ 'Area': [1000], 'No. of Bedrooms': [3] })
model.predict(house)

simple_model = tc.linear_regression.create(data, features=['Area'], target='Price')
print(simple_model.coefficients)